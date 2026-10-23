"""Step 5: retrieval strategies over one embedding model's collection and the BM25 index.

Strategies (``Retriever.retrieve``):

- ``dense``: cosine similarity in Chroma.
- ``mmr``: maximal marginal relevance over ``mmr_fetch_k`` dense candidates, trading
  relevance against redundancy with ``mmr_lambda``.
- ``bm25``: keyword search with ``rank_bm25`` (a baseline; not offered in the apps).
- ``hybrid``: BM25 and dense lists merged with weighted reciprocal rank fusion (RRF).
- ``hybrid_rerank``: hybrid's top ``retrieval_k`` candidates rescored by a cross-encoder.
- ``multi_query``: the LLM writes ``multi_query_n`` rephrasings; each query runs hybrid
  search, the lists are fused, and the result is reranked against the original question
  when the reranker is enabled.

Every strategy honours the paper filter (a Chroma ``where`` clause for dense search, a
post-filter for BM25) and deduplicates by ``chunk_id`` before reranking.
"""

import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from rpab.embeddings import CachedEmbedder
from rpab.schemas import RetrievedChunk

STRATEGIES = ("dense", "mmr", "bm25", "hybrid", "hybrid_rerank", "multi_query")

# ---- BM25 ----------------------------------------------------------------------------------

TOKENIZER_VERSION = "v1"
TOKEN = re.compile(r"[a-z0-9]+")
STOP_WORDS = frozenset(
    "a an and are as at be been but by can for from has have how in into is it its of on or "
    "our that the their then there these this to was we were what when which while who why with".split()
)


def tokenize(text: str) -> list[str]:
    """Tokenizer v1: lowercase, split on non-alphanumerics, drop a small stop word list."""
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP_WORDS]


class BM25Index:
    """``BM25Okapi`` over the chunk table, in the table's row order."""

    def __init__(self, bm25: BM25Okapi, chunk_ids: list[str]):
        self.bm25, self.chunk_ids = bm25, chunk_ids

    @classmethod
    def build(cls, chunks: pd.DataFrame) -> "BM25Index":
        return cls(BM25Okapi([tokenize(t) for t in chunks["text"]]), chunks["chunk_id"].tolist())

    @classmethod
    def load_or_build(cls, path: Path, chunks: pd.DataFrame) -> "BM25Index":
        """Reuse the saved index unless the chunk set or the tokenizer version changed."""
        path = Path(path)
        if path.exists():
            saved = joblib.load(path)
            if saved["tokenizer_version"] == TOKENIZER_VERSION and saved["chunk_ids"] == chunks["chunk_id"].tolist():
                return cls(saved["bm25"], saved["chunk_ids"])
        index = cls.build(chunks)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"bm25": index.bm25, "chunk_ids": index.chunk_ids, "tokenizer_version": TOKENIZER_VERSION}, path)
        return index

    def search(self, question: str, k: int, allowed: np.ndarray | None = None) -> list[tuple[str, float]]:
        """Top-k (chunk_id, score); ``allowed`` is a boolean mask over chunks (the paper filter)."""
        scores = self.bm25.get_scores(tokenize(question))
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top if scores[i] > 0]


# ---- reranker and query expansion ----------------------------------------------------------

class Reranker:
    """Cross-encoder that scores (question, passage) pairs; scores are probabilities (sigmoid)."""

    def __init__(self, model_name: str, device: str = "auto", max_length: int = 1024):
        import torch
        from sentence_transformers import CrossEncoder

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = CrossEncoder(model_name, device=device, max_length=max_length, model_kwargs={"dtype": dtype})
        self._sigmoid = torch.nn.Sigmoid()

    def score(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict([(question, t) for t in texts], batch_size=len(texts), activation_fn=self._sigmoid)
        return [float(s) for s in scores]


REWRITE_PROMPT = """You help search a collection of AI research papers.
Write {n} different rephrasings of the question below. Keep every technical term, vary the wording,
and make each one a complete standalone question. Reply with one rephrasing per line and nothing else.

Question: {question}"""


class MultiQuery:
    """Rephrases a question with the local LLM. Results are cached on disk with the time the LLM
    took, so experiment reruns are repeatable and still report the true latency."""

    def __init__(self, config, cache_path: Path):
        self.config, self.cache_path = config, Path(cache_path)
        self.cache = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path.exists() else {}
        self._llm = None

    @staticmethod
    def parse(reply: str, question: str, n: int) -> list[str]:
        lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"') for line in reply.splitlines()]
        unique = list(dict.fromkeys(line for line in lines if len(line) > 10 and line.lower() != question.lower()))
        return unique[:n]

    def expand(self, question: str) -> tuple[list[str], float]:
        """The original question plus up to ``multi_query_n`` rephrasings, and the LLM time in seconds."""
        n = self.config.multi_query_n
        key = f"{self.config.llm_model}|{n}|{question}"
        if key not in self.cache:
            if self._llm is None:
                from langchain_ollama import ChatOllama

                # The rewrite prompt is short, so a small context keeps the model's GPU memory low
                self._llm = ChatOllama(model=self.config.llm_model, base_url=self.config.ollama_base_url,
                                       temperature=0, seed=self.config.seed, num_ctx=self.config.rewrite_num_ctx)
                self._llm.invoke("Reply with OK.")  # loads the model, so the timed call below is generation only
            start = time.perf_counter()
            reply = self._llm.invoke(REWRITE_PROMPT.format(n=n, question=question)).content
            self.cache[key] = {"rephrasings": self.parse(reply, question, n), "seconds": time.perf_counter() - start}
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=1, ensure_ascii=False), encoding="utf-8")
        entry = self.cache[key]
        return [question] + entry["rephrasings"], entry["seconds"]


# ---- retriever ---------------------------------------------------------------------------

class Retriever:
    """All strategies over one collection. ``last_llm_seconds`` holds the multi-query LLM time
    of the most recent call, so evaluation can count it in latency even when it was cached."""

    def __init__(self, config, embedder: CachedEmbedder, collection, chunks: pd.DataFrame, bm25: BM25Index,
                 reranker: Reranker | None = None, multi_query: MultiQuery | None = None):
        self.config, self.embedder, self.collection, self.bm25 = config, embedder, collection, bm25
        self.reranker, self.multi_query = reranker, multi_query
        self.chunks = chunks.set_index("chunk_id", drop=False)
        self.paper_of = chunks["paper_id"].to_numpy()
        self.last_llm_seconds = 0.0
        if collection.metadata.get("embedding_model") != embedder.model_name:
            raise ValueError(f"{collection.name} holds {collection.metadata.get('embedding_model')} vectors, "
                             f"not {embedder.model_name}")

    def _chunk(self, chunk_id: str, **scores) -> RetrievedChunk:
        row = self.chunks.loc[chunk_id]
        return RetrievedChunk(chunk_id=chunk_id, paper_id=row["paper_id"], paper_title=row["paper_title"],
                              page=int(row["page"]), text=row["text"], **scores)

    @staticmethod
    def _where(paper_ids: list[str] | None) -> dict | None:
        return {"paper_id": {"$in": list(paper_ids)}} if paper_ids else None

    def dense(self, question: str, k: int, paper_ids: list[str] | None = None) -> list[RetrievedChunk]:
        result = self.collection.query(query_embeddings=[self.embedder.embed_query(question)], n_results=k,
                                       where=self._where(paper_ids), include=["distances"])
        return [self._chunk(cid, score=1 - d, dense_score=1 - d)
                for cid, d in zip(result["ids"][0], result["distances"][0])]

    def mmr(self, question: str, k: int, paper_ids: list[str] | None = None) -> list[RetrievedChunk]:
        query = np.array(self.embedder.embed_query(question), dtype=np.float32)
        result = self.collection.query(query_embeddings=[query], n_results=self.config.mmr_fetch_k,
                                       where=self._where(paper_ids), include=["embeddings", "distances"])
        ids, vectors = result["ids"][0], np.array(result["embeddings"][0], dtype=np.float32)
        if not ids:
            return []
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        relevance = vectors @ (query / np.linalg.norm(query))
        similarity = vectors @ vectors.T
        lam, selected, remaining = self.config.mmr_lambda, [], list(range(len(ids)))
        while remaining and len(selected) < k:
            if selected:
                gain = [lam * relevance[i] - (1 - lam) * similarity[i, selected].max() for i in remaining]
                best = remaining[int(np.argmax(gain))]
            else:
                best = remaining[int(np.argmax(relevance[remaining]))]
            selected.append(best)
            remaining.remove(best)
        return [self._chunk(ids[i], score=float(relevance[i]), dense_score=float(relevance[i])) for i in selected]

    def keyword(self, question: str, k: int, paper_ids: list[str] | None = None) -> list[RetrievedChunk]:
        allowed = np.isin(self.paper_of, paper_ids) if paper_ids else None
        return [self._chunk(cid, score=s, bm25_score=s) for cid, s in self.bm25.search(question, k, allowed)]

    def _fuse(self, ranked: list[list[RetrievedChunk]], weights: list[float], k: int) -> list[RetrievedChunk]:
        """Weighted RRF: score = sum of weight / (rrf_k + rank). Keeps each chunk's component scores."""
        fused: dict[str, dict] = {}
        for results, weight in zip(ranked, weights):
            for rank, chunk in enumerate(results, start=1):
                entry = fused.setdefault(chunk.chunk_id, {"rrf": 0.0, "dense_score": None, "bm25_score": None})
                entry["rrf"] += weight / (self.config.rrf_k + rank)
                for field in ("dense_score", "bm25_score"):
                    value = getattr(chunk, field)
                    if value is not None and (entry[field] is None or value > entry[field]):
                        entry[field] = value
        order = sorted(fused.items(), key=lambda item: -item[1]["rrf"])[:k]
        return [self._chunk(cid, score=e["rrf"], dense_score=e["dense_score"], bm25_score=e["bm25_score"])
                for cid, e in order]

    def hybrid(self, question: str, k: int, paper_ids: list[str] | None = None) -> list[RetrievedChunk]:
        n, w = self.config.retrieval_k, self.config.hybrid_dense_weight
        return self._fuse([self.dense(question, n, paper_ids), self.keyword(question, n, paper_ids)], [w, 1 - w], k)

    def rerank(self, question: str, candidates: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        if self.reranker is None:
            raise RuntimeError("Reranking needs a Reranker (disabled with RPAB_RERANKER=off)")
        unique = list({c.chunk_id: c for c in candidates}.values())  # dedupe before scoring
        scores = self.reranker.score(question, [c.text for c in unique])
        ranked = sorted(zip(unique, scores), key=lambda pair: -pair[1])[:k]
        return [c.model_copy(update={"score": s, "rerank_score": s}) for c, s in ranked]

    def retrieve(self, question: str, strategy: str = "hybrid_rerank", k: int | None = None,
                 paper_ids: list[str] | None = None) -> list[RetrievedChunk]:
        """Top-k chunks for ``question``. Without a reranker, ``hybrid_rerank`` falls back to ``hybrid``."""
        k = k or self.config.final_k
        self.last_llm_seconds = 0.0
        if strategy == "dense":
            return self.dense(question, k, paper_ids)
        if strategy == "mmr":
            return self.mmr(question, k, paper_ids)
        if strategy == "bm25":
            return self.keyword(question, k, paper_ids)
        if strategy == "hybrid" or (strategy == "hybrid_rerank" and self.reranker is None):
            return self.hybrid(question, k, paper_ids)
        if strategy == "hybrid_rerank":
            return self.rerank(question, self.hybrid(question, self.config.retrieval_k, paper_ids), k)
        if strategy == "multi_query":
            if self.multi_query is None:
                raise RuntimeError("multi_query needs a MultiQuery rewriter")
            queries, self.last_llm_seconds = self.multi_query.expand(question)
            n = self.config.retrieval_k
            candidates = self._fuse([self.hybrid(q, n, paper_ids) for q in queries], [1.0] * len(queries), n)
            return self.rerank(question, candidates, k) if self.reranker else candidates[:k]
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {', '.join(STRATEGIES)}")

    def confidence(self, results: list[RetrievedChunk]) -> tuple[float | None, str]:
        """The score the weak-context check uses: the top reranker probability when reranked,
        otherwise the top cosine similarity. BM25-only results have no calibrated score."""
        if not results:
            return None, "none"
        if results[0].rerank_score is not None:
            return results[0].rerank_score, "rerank"
        dense = [c.dense_score for c in results if c.dense_score is not None]
        return (max(dense), self.embedder.slug) if dense else (None, "none")

    def is_weak(self, results: list[RetrievedChunk]) -> bool:
        """True when the best passage scores below the threshold calibrated on dev questions."""
        score, kind = self.confidence(results)
        threshold = self.config.confidence_thresholds.get(kind)
        return score is None or (threshold is not None and score < threshold)
