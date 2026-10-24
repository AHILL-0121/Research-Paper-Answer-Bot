"""Stretch goal S3: corrective RAG with a web search fallback (doc 03, section 4).

When the user turns the fallback on and the best paper passage scores below the calibrated
weak-context threshold, the question is also sent to DuckDuckGo (``ddgs``, no API key). The
result snippets become passages marked ``origin="web"``. With a reranker, paper and web passages
are scored against the question together and the best ``k`` are kept, so a web result is used only
when it beats the papers; without one, web results cannot be compared with paper scores, so the
top paper passage is kept and the rest are web results. Web passages are labelled everywhere
(prompt, sources, UI) and never presented as paper citations. A failed search falls back to the
papers alone and says so in the response's ``web_note``.
"""

from rpab.config import CONFIG
from rpab.retrievers import Retriever
from rpab.schemas import RetrievedChunk


class WebSearch:
    """DuckDuckGo text search. The question leaves the machine, which the apps tell the user."""

    def __init__(self, max_results: int = 5, timeout_s: int = 8):
        self.max_results, self.timeout_s = max_results, timeout_s

    def search(self, question: str) -> list[RetrievedChunk]:
        from ddgs import DDGS

        hits = DDGS(timeout=self.timeout_s).text(question, max_results=self.max_results, safesearch="moderate")
        return [RetrievedChunk(chunk_id=f"web_{i}", paper_id="web", paper_title=hit.get("title") or hit["href"],
                               page=0, text=hit["body"], score=0.0, origin="web", url=hit["href"])
                for i, hit in enumerate(hits, start=1) if hit.get("body") and hit.get("href")]


def corrective_retrieve(question: str, docs: list[RetrievedChunk], retriever: Retriever, search: WebSearch,
                        k: int, config=CONFIG) -> tuple[list[RetrievedChunk], str]:
    """The passages to answer from after a web search, and a note saying what happened."""
    best, kind = retriever.confidence(docs)
    threshold = config.confidence_thresholds.get(kind)
    why = (f"The best paper passage scored {best:.2f}, below the {threshold:.2f} threshold, so the question was "
           "also sent to a web search (DuckDuckGo)." if best is not None and threshold is not None
           else "No paper passage scored high enough, so the question was also sent to a web search (DuckDuckGo).")
    try:
        web = search.search(question)
    except Exception as exc:  # network, rate limit or parsing: the papers alone still answer
        return docs, f"{why} The search failed ({exc.__class__.__name__}), so the answer uses the papers only."
    if not web:
        return docs, f"{why} It returned nothing, so the answer uses the papers only."

    if retriever.reranker is not None:
        unscored = [c for c in docs + web if c.rerank_score is None]
        scores = retriever.reranker.score(question, [_rerank_text(c) for c in unscored]) if unscored else []
        rescored = {c.chunk_id: c.model_copy(update={"score": s, "rerank_score": s}) for c, s in zip(unscored, scores)}
        candidates = [rescored.get(c.chunk_id, c) for c in docs + web]
        merged = sorted(candidates, key=lambda c: -c.rerank_score)[:k]
    else:
        merged = docs[:1] + web[: k - len(docs[:1])]

    used = sum(c.origin == "web" for c in merged)
    if not used:
        return merged, f"{why} No web result scored above the paper passages, so the answer uses the papers only."
    return merged, f"{why} {used} of the {len(merged)} passages come from the web and are marked WEB."


def _rerank_text(chunk: RetrievedChunk) -> str:
    return f"{chunk.paper_title}. {chunk.text}" if chunk.origin == "web" else chunk.text
