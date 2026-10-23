"""Step 3: embedding models behind one interface, with a disk cache keyed by text hash.

Two models are compared:

- ``BAAI/bge-m3`` (open-source, 1024-dim), run locally on the GPU in float16.
- ``gemini-embedding-001`` (commercial, 3072-dim), called on Google's free tier with
  batching, request pacing and exponential backoff on rate-limit errors.

Both implement LangChain's ``Embeddings`` interface, so the semantic chunker and the
vector store can use either. Document vectors are cached in
``artifacts/embeddings_cache/{slug}.joblib`` and never recomputed for unchanged text;
query vectors are not cached.
"""

import copy
import re
import time
from pathlib import Path

import joblib
import numpy as np
from langchain_core.embeddings import Embeddings

from rpab.chunking import text_hash


class CachedEmbedder(Embeddings):
    """Base class: subclasses implement ``_embed``; this class adds the cache and timing."""

    model_name: str
    slug: str
    dim: int

    def __init__(self, cache_dir: Path | None = None):
        self.cache_path = Path(cache_dir) / f"{self.slug}.joblib" if cache_dir else None
        self._cache: dict[str, np.ndarray] = {}
        if self.cache_path and self.cache_path.exists():
            self._cache = joblib.load(self.cache_path)
        self.seconds = 0.0  # time spent embedding texts that were not cached
        self.embedded = 0  # number of texts actually sent to the model

    def _embed(self, texts: list[str], is_query: bool) -> np.ndarray:
        raise NotImplementedError

    def uncached(self) -> "CachedEmbedder":
        """A view sharing the loaded model but with no cache, for throwaway texts such as
        the sentence groups the semantic chunker compares."""
        view = copy.copy(self)
        view.cache_path, view._cache = None, {}
        return view

    def save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._cache, self.cache_path)

    def embed_array(self, texts: list[str]) -> np.ndarray:
        """Document vectors as a (n, dim) float32 array, using the cache where possible."""
        hashes = [text_hash(t) for t in texts]
        missing = list(dict.fromkeys(h for h in hashes if h not in self._cache))
        if missing:
            by_hash = dict(zip(hashes, texts))
            self._embed_missing([by_hash[h] for h in missing], missing)
        return np.stack([self._cache[h] for h in hashes]).astype(np.float32)

    def _embed_missing(self, texts: list[str], hashes: list[str]) -> None:
        start = time.perf_counter()
        vectors = self._embed(texts, is_query=False)
        self.seconds += time.perf_counter() - start
        self.embedded += len(texts)
        self._cache.update(zip(hashes, vectors.astype(np.float32)))
        self.save_cache()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_array(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], is_query=True)[0].tolist()


class BgeM3Embedder(CachedEmbedder):
    """BAAI/bge-m3 dense vectors via sentence-transformers. Queries need no instruction prefix."""

    model_name = "BAAI/bge-m3"
    slug = "bge_m3"
    dim = 1024

    def __init__(self, cache_dir: Path | None = None, batch_size: int = 16, max_seq_length: int = 1024):
        super().__init__(cache_dir)
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = SentenceTransformer(self.model_name, device=device, model_kwargs={"dtype": dtype})
        # bge-m3 accepts 8,192 tokens; the longest chunk is far shorter, and a cap bounds GPU memory
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def _embed(self, texts: list[str], is_query: bool) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=len(texts) > 200)


class GeminiEmbedder(CachedEmbedder):
    """gemini-embedding-001 on the free tier.

    Requests carry up to ``batch_size`` texts. Before each request the embedder waits so the
    estimated tokens sent in the last minute stay under ``tokens_per_minute``; a 429 or 5xx
    response is retried with exponential backoff, honouring the server's suggested delay.
    The cache is saved after every batch, so a run stopped by the daily quota resumes where it ended.
    """

    model_name = "models/gemini-embedding-001"
    slug = "gemini_embedding_001"
    dim = 3072
    price_per_m_tokens = 0.15  # paid-tier list price in USD; the free tier costs nothing

    def __init__(self, api_key: str, cache_dir: Path | None = None, batch_size: int = 50,
                 tokens_per_minute: int = 25_000, max_retries: int = 8):
        super().__init__(cache_dir)
        from google import genai

        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set (backend/.env.local)")
        self.client = genai.Client(api_key=api_key)
        self.batch_size = batch_size
        self.tokens_per_minute = tokens_per_minute
        self.max_retries = max_retries
        self._sent: list[tuple[float, int]] = []  # (time, estimated tokens) of recent requests
        self.tokens_sent = 0

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def _pace(self, tokens: int) -> None:
        while True:
            now = time.monotonic()
            self._sent = [(t, n) for t, n in self._sent if now - t < 60]
            if sum(n for _, n in self._sent) + tokens <= self.tokens_per_minute or not self._sent:
                return
            time.sleep(60 - (now - self._sent[0][0]) + 0.5)

    def _request(self, texts: list[str], task_type: str) -> np.ndarray:
        from google.genai import errors, types

        tokens = sum(self.estimate_tokens(t) for t in texts)
        for attempt in range(self.max_retries):
            self._pace(tokens)
            try:
                result = self.client.models.embed_content(
                    model=self.model_name, contents=texts, config=types.EmbedContentConfig(task_type=task_type))
                self._sent.append((time.monotonic(), tokens))
                self.tokens_sent += tokens
                return np.array([e.values for e in result.embeddings], dtype=np.float32)
            except (errors.ClientError, errors.ServerError) as exc:
                if getattr(exc, "code", None) not in (429, 500, 503) or attempt == self.max_retries - 1:
                    raise
                hint = re.search(r"retry in (\d+(?:\.\d+)?)s|'retryDelay': '(\d+)s'", str(exc))
                delay = float(next(g for g in hint.groups() if g)) if hint else 0.0
                time.sleep(max(delay, 2 ** attempt * 5))
        raise RuntimeError("unreachable")

    def _embed(self, texts: list[str], is_query: bool) -> np.ndarray:
        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        return np.concatenate([self._request(texts[i:i + self.batch_size], task_type)
                               for i in range(0, len(texts), self.batch_size)])

    def _embed_missing(self, texts: list[str], hashes: list[str]) -> None:
        # Embed and cache one batch at a time so progress survives a quota stop
        for i in range(0, len(texts), self.batch_size):
            super()._embed_missing(texts[i:i + self.batch_size], hashes[i:i + self.batch_size])

    @property
    def cost_usd(self) -> float:
        """What the tokens sent so far would cost on the paid tier."""
        return self.tokens_sent / 1e6 * self.price_per_m_tokens


def load_embedder(kind: str, config) -> CachedEmbedder:
    """Build the embedder named in ``config.embedding_models`` (``open`` or ``commercial``)."""
    cache_dir = config.paths.embeddings_cache
    if kind == "open":
        return BgeM3Embedder(cache_dir, batch_size=config.embed_batch_size)
    if kind == "commercial":
        return GeminiEmbedder(config.google_api_key, cache_dir)
    raise ValueError(f"Unknown embedder kind: {kind}")
