"""Step 6: the RAG pipeline, retrieval → numbered context → prompt → ``ChatOllama`` → text.

One LCEL chain serves both entry points. ``RagPipeline.stream`` yields events for the apps
(``sources`` first, then ``token`` pieces, then ``done`` with the full ``AskResponse``), and
``RagPipeline.answer`` runs the same stream to completion, so the two cannot drift apart.

Guarantees that do not depend on the model behaving:

- Sources come from the retrieved chunks' metadata, never from the generated text.
- Citation markers that point at no passage (``[4]`` with three passages) are removed, also
  while streaming, so a wrong marker never reaches the screen.
- When the top passage scores below the calibrated threshold, the prompt says so and asks for
  a refusal unless a passage answers directly (the model still decides, as in doc 03).
- An answer that opens with "I don't know", or cites nothing and says the context lacks the
  answer, is replaced by the exact refusal sentence; the closest passages are still returned.
"""

import re
import time
from dataclasses import replace
from functools import lru_cache
from typing import Iterator

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough

from rpab.config import CONFIG, REFUSAL_TEXT
from rpab.prompts import RAG_PROMPT, WEAK_CONTEXT_NOTICE, format_context
from rpab.retrievers import BM25Index, MultiQuery, Reranker, Retriever
from rpab.schemas import AskRequest, AskResponse, RetrievedChunk, Source

SNIPPET_CHARS = 220

# ---- citations ---------------------------------------------------------------------------

# One marker holds one or more numbers: [2] or [1, 3]
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# A marker plus the spaces before it, so dropping " [4]" does not leave "word ." behind
CITATION_WITH_SPACE = re.compile(r"\s*\[(\d+(?:\s*,\s*\d+)*)\]")
MAX_MARKER_CHARS = 16  # an open "[" held back longer than this is ordinary text


class CitationFilter:
    """Removes citation numbers outside 1..n from text that may arrive in pieces.

    Text is held back from an unclosed "[" (and the spaces before it) until the bracket closes,
    so a marker split across stream tokens is still judged whole. ``removed`` lists the markers
    that were dropped or trimmed, as they were written.
    """

    def __init__(self, n_passages: int):
        self.n = n_passages
        self.buffer = ""
        self.removed: list[str] = []

    def _fix(self, match: re.Match) -> str:
        numbers = [int(x) for x in re.split(r"\s*,\s*", match.group(1))]
        valid = [x for x in dict.fromkeys(numbers) if 1 <= x <= self.n]
        if len(valid) == len(numbers):
            return match.group(0)
        self.removed.append(match.group(0).strip())
        if not valid:
            return ""
        spaces = match.group(0)[: match.start(1) - 1 - match.start(0)]
        return f"{spaces}[{', '.join(map(str, valid))}]"

    def _clean(self, text: str) -> str:
        return CITATION_WITH_SPACE.sub(self._fix, text)

    def feed(self, piece: str) -> str:
        """Add a piece of streamed text; return the part that is safe to show now."""
        self.buffer += piece
        hold = len(self.buffer.rstrip())  # trailing spaces may belong to a marker that follows
        opening = self.buffer.rfind("[")
        if opening != -1 and "]" not in self.buffer[opening:] and len(self.buffer) - opening <= MAX_MARKER_CHARS:
            hold = len(self.buffer[:opening].rstrip())
        ready, self.buffer = self.buffer[:hold], self.buffer[hold:]
        return self._clean(ready)

    def flush(self) -> str:
        ready, self.buffer = self.buffer, ""
        return self._clean(ready)


def sanitize_citations(text: str, n_passages: int) -> tuple[str, list[str]]:
    """The whole-text form of ``CitationFilter``: (clean text, removed markers)."""
    citation_filter = CitationFilter(n_passages)
    clean = citation_filter.feed(text) + citation_filter.flush()
    return clean, citation_filter.removed


def cited_numbers(text: str) -> set[int]:
    """Passage numbers an answer cites."""
    return {int(x) for match in CITATION.finditer(text) for x in re.split(r"\s*,\s*", match.group(1))}


# ---- refusals and sources ----------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase letters, apostrophes and single spaces only."""
    text = CITATION.sub(" ", text.lower().replace("’", "'").replace("‘", "'"))
    return " ".join(re.sub(r"[^a-z' ]+", " ", text).split())


REFUSAL_OPENINGS = ("i don't know", "i do not know", "i dont know")
# Refusals in other words, e.g. "There is no mention of Mamba in the provided context passages."
NOT_IN_CONTEXT = re.compile(r"\b(no mention|not mentioned|no information|not (?:contain|provide|mention|include|"
                            r"discuss|cover)|does not say|do not say|cannot be answered|can't be answered)")


def is_refusal(text: str) -> bool:
    """Loose match on the refusal sentence: case, curly quotes, punctuation and citation markers
    are ignored, and any answer that opens with "I don't know" counts. An answer that cites no
    passage and whose first sentence says the context lacks the answer counts too."""
    normalised = _normalise(text)
    if normalised.startswith(REFUSAL_OPENINGS):
        return True
    first_sentence = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return not cited_numbers(text) and bool(NOT_IN_CONTEXT.search(_normalise(first_sentence)))


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """The passage's opening, cut at a word boundary."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def to_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    return [Source(rank=rank, paper_title=c.paper_title, page=c.page, snippet=snippet(c.text),
                   full_text=c.text, score=c.score, chunk_id=c.chunk_id, paper_id=c.paper_id)
            for rank, c in enumerate(chunks, start=1)]


# ---- pipeline ----------------------------------------------------------------------------

def make_llm(config=CONFIG, model: str | None = None):
    """``ChatOllama`` at temperature 0 with the full 8k context (Ollama's default truncates)."""
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model or config.llm_model, base_url=config.ollama_base_url,
                      temperature=config.llm_temperature, seed=config.seed, num_ctx=config.llm_num_ctx)


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


class RagPipeline:
    """The LCEL chain over one retriever and one chat model."""

    def __init__(self, retriever: Retriever, llm, config=CONFIG):
        self.retriever, self.llm, self.config = retriever, llm, config
        self.model_name = getattr(llm, "model", "unknown")

        def retrieve(inputs: dict) -> list[RetrievedChunk]:
            return retriever.retrieve(inputs["question"], inputs["strategy"], inputs["k"], inputs["paper_ids"])

        generate = (
            RunnableLambda(lambda x: {"context": format_context(x["docs"]), "question": x["question"],
                                      "notice": WEAK_CONTEXT_NOTICE if retriever.is_weak(x["docs"]) else ""})
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )
        self.chain = (
            RunnableParallel(docs=RunnableLambda(retrieve), question=lambda x: x["question"])
            # No passages (e.g. a paper filter that matches nothing): refuse without calling the LLM
            | RunnablePassthrough.assign(answer=RunnableBranch((lambda x: not x["docs"], lambda x: REFUSAL_TEXT),
                                                               generate))
        )

    def strategy_used(self, strategy: str) -> str:
        return "hybrid" if strategy == "hybrid_rerank" and self.retriever.reranker is None else strategy

    def warm_up(self) -> None:
        """Load the embedder, reranker and chat model, so the first timed question is not a cold start."""
        self.retriever.retrieve("warm-up question about attention", self.strategy_used("hybrid_rerank"))
        self.llm.invoke("Reply with OK.")

    def stream(self, request: AskRequest) -> Iterator[dict]:
        """Events: ``{"event": "sources", ...}``, then ``{"event": "token", "text": ...}`` pieces,
        then ``{"event": "done", "response": AskResponse}``."""
        start = time.perf_counter()
        inputs = {"question": request.question, "strategy": request.strategy, "k": request.final_k,
                  "paper_ids": request.paper_ids}
        docs: list[RetrievedChunk] = []
        shown: list[str] = []  # the cleaned text exactly as streamed
        citation_filter, retrieval_s, first_token_s = CitationFilter(0), None, None
        for part in self.chain.stream(inputs):
            if "docs" in part:
                docs, retrieval_s = part["docs"], time.perf_counter() - start
                citation_filter = CitationFilter(len(docs))
                yield {"event": "sources", "sources": to_sources(docs), "weak_context": self.retriever.is_weak(docs),
                       "retrieval_ms": _ms(retrieval_s)}
            if part.get("answer"):
                if first_token_s is None:
                    first_token_s = time.perf_counter() - start
                text = citation_filter.feed(part["answer"])
                if text:
                    shown.append(text)
                    yield {"event": "token", "text": text}
        tail = citation_filter.flush()
        if tail:
            shown.append(tail)
            yield {"event": "token", "text": tail}

        answer_text = "".join(shown)
        refused = is_refusal(answer_text)
        response = AskResponse(
            answer=REFUSAL_TEXT if refused else answer_text.strip(),
            sources=to_sources(docs), refused=refused,
            strategy_used=self.strategy_used(request.strategy),
            latency_ms=_ms(time.perf_counter() - start),
            weak_context=self.retriever.is_weak(docs),
            retrieval_ms=_ms(retrieval_s) if retrieval_s is not None else None,
            first_token_ms=_ms(first_token_s) if first_token_s is not None else None,
            removed_citations=citation_filter.removed,
        )
        yield {"event": "done", "response": response}

    def answer(self, request: AskRequest) -> AskResponse:
        for event in self.stream(request):
            if event["event"] == "done":
                return event["response"]
        raise RuntimeError("The answer stream ended without a response")


def load_pipeline(config=CONFIG, embedding: str | None = None, llm_model: str | None = None) -> RagPipeline:
    """Build the pipeline the apps use: the chosen embedding's collection, BM25, the reranker
    when enabled, and the chat model. Loading bge-m3 and the reranker takes 10–30 s."""
    from rpab.embeddings import load_embedder
    from rpab.vectorstore import collection_name, get_client

    embedder = load_embedder(embedding or config.embedding_choice, config)
    collection = get_client(config.paths.chroma_dir).get_collection(
        collection_name(embedder.slug, config.chunk_spec.slug))
    chunks = pd.read_parquet(config.paths.chunks)
    bm25 = BM25Index.load_or_build(config.paths.bm25, chunks)
    reranker = Reranker(config.reranker_model, device=config.device) if config.use_reranker else None
    # Rephrasing with the answer model's context size keeps Ollama from reloading the model between
    # the rewrite call and the answer call; app questions get their own cache, apart from the experiments'
    rewriter = MultiQuery(replace(config, rewrite_num_ctx=config.llm_num_ctx), config.paths.multi_query_app_cache)
    retriever = Retriever(config, embedder, collection, chunks, bm25, reranker, rewriter)
    return RagPipeline(retriever, make_llm(config, llm_model), config)


@lru_cache(maxsize=1)
def default_pipeline() -> RagPipeline:
    return load_pipeline()


def answer(request: AskRequest) -> AskResponse:
    """Doc 05's service entry point, over a pipeline loaded once per process."""
    return default_pipeline().answer(request)
