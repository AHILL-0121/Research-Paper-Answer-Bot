"""Single source of truth for paths, model names and pipeline settings.

Every other module reads settings from ``CONFIG`` instead of hard-coding them,
so an experiment changes one value here and the whole pipeline follows.
Secrets (API keys) come from ``backend/.env.local`` or ``backend/.env`` and are
never stored in this file.

Runtime choices that depend on the machine, not on experiment results, can be overridden
with ``RPAB_*`` environment variables, which ``scripts/setup.py`` writes to ``.env.local``:

    RPAB_EMBEDDING  commercial | open    embedding model used for retrieval
    RPAB_LLM        an Ollama model tag  generation model
    RPAB_RERANKER   on | off             cross-encoder reranking (slow without a GPU)
    RPAB_DEVICE     auto | cuda | cpu    where local models run
    RPAB_PROFILE    gpu | cpu            hardware profile, informational

Experiment results (chunking, k, thresholds) are deliberately not overridable this way.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
# .env.local wins over .env because load_dotenv never overwrites a variable already set
load_dotenv(BACKEND_DIR / ".env.local")
load_dotenv(BACKEND_DIR / ".env")

REFUSAL_TEXT = "I don't know based on the provided papers."


def _env_choice(name: str, default: str, allowed: tuple[str, ...] | None = None) -> str:
    """Read an ``RPAB_*`` override, failing loudly on a value the pipeline does not support."""
    value = (os.getenv(name) or default).strip()
    if allowed and value not in allowed:
        raise ValueError(f"{name}={value!r} is not one of {', '.join(allowed)} (check backend/.env.local)")
    return value


@dataclass
class Paths:
    """Filesystem locations, all relative to the backend folder."""

    data_dir: Path = BACKEND_DIR / "data"
    catalog: Path = BACKEND_DIR / "data" / "papers.json"  # hand-written titles keyed by arXiv id
    pdf_dir: Path = BACKEND_DIR / "data" / "pdfs"
    manifest: Path = BACKEND_DIR / "data" / "pdfs" / "manifest.json"
    eval_questions: Path = BACKEND_DIR / "data" / "eval" / "eval_questions.json"
    artifacts_dir: Path = BACKEND_DIR / "artifacts"
    pages: Path = BACKEND_DIR / "artifacts" / "pages.parquet"  # raw and cleaned text per page
    loader_comparison: Path = BACKEND_DIR / "artifacts" / "loader_comparison.csv"
    chroma_dir: Path = BACKEND_DIR / "artifacts" / "chroma"
    chunks: Path = BACKEND_DIR / "artifacts" / "chunks.parquet"
    chunk_grid_dir: Path = BACKEND_DIR / "artifacts" / "chunk_grid"  # one parquet per chunking experiment
    embeddings_cache: Path = BACKEND_DIR / "artifacts" / "embeddings_cache"
    bm25: Path = BACKEND_DIR / "artifacts" / "bm25.joblib"
    multi_query_cache: Path = BACKEND_DIR / "artifacts" / "multi_query_cache.json"  # LLM rephrasings per question
    multi_query_app_cache: Path = BACKEND_DIR / "artifacts" / "multi_query_app_cache.json"  # same, for app questions
    experiments: Path = BACKEND_DIR / "artifacts" / "experiments.csv"
    chat_db: Path = BACKEND_DIR / "artifacts" / "chat.db"
    answers_dir: Path = BACKEND_DIR / "artifacts" / "answers"  # generated answers and judge scores per model
    manual_scores: Path = BACKEND_DIR / "data" / "eval" / "answer_scores_manual.csv"  # hand scores, tracked


@dataclass
class Config:
    """Pipeline settings. Defaults are the starting point; experiments tune them."""

    seed: int = 42
    paths: Paths = field(default_factory=Paths)

    # Loading and cleaning
    pdf_loader: str = "pypdf"  # pypdf | pdfplumber, chosen by the loader comparison
    empty_page_chars: int = 20  # a page with fewer characters counts as empty
    repeated_line_share: float = 0.5  # header/footer = same edge line on at least this share of pages

    # Chunking
    # Chosen in notebook section 3: fixed 512-token chunks, no overlap, reference lists dropped
    chunking_strategy: str = "fixed"  # recursive | fixed | semantic
    chunk_size: int = 512  # tokens for the fixed strategy, characters for the others
    chunk_overlap: int = 0
    min_chunk_chars: int = 100  # shorter chunks are dropped
    semantic_percentile: int = 95  # breakpoint percentile, used by the semantic strategy only
    drop_references: bool = True  # drop chunks that read like a bibliography

    # Embeddings: one open-source model and one commercial model are compared
    embedding_models: dict = field(default_factory=lambda: {
        "open": "BAAI/bge-m3",
        "commercial": "models/gemini-embedding-001",
        "fallback": "nomic-embed-text",  # via Ollama, used if the Gemini free tier fails
    })
    embed_batch_size: int = 16
    # Chosen in notebook section 6, after the gold-page audit: bge-m3 (local). Gemini ties it once results
    # are reranked, so the local model wins on latency, privacy and having no API dependency.
    # Gemini stays indexed and selectable with RPAB_EMBEDDING=commercial.
    embedding_choice: str = field(default_factory=lambda: _env_choice("RPAB_EMBEDDING", "open",
                                                                       ("commercial", "open")))
    device: str = field(default_factory=lambda: _env_choice("RPAB_DEVICE", "auto", ("auto", "cuda", "cpu")))
    profile: str = field(default_factory=lambda: _env_choice("RPAB_PROFILE", "gpu", ("gpu", "cpu")))

    # Retrieval
    retrieval_k: int = 20  # candidates fetched before reranking
    final_k: int = 3  # passages given to the LLM and shown to the user
    mmr_fetch_k: int = 20
    mmr_lambda: float = 0.5
    rrf_k: int = 60  # reciprocal rank fusion constant for hybrid search
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_reranker: bool = field(default_factory=lambda: _env_choice("RPAB_RERANKER", "on", ("on", "off")) == "on")
    retrieval_strategy: str = "hybrid_rerank"  # chosen in section 6; plain hybrid when the reranker is off
    # RRF weight of the dense list (BM25 gets the rest). 0.3/0.5/0.7 differ by about one dev question,
    # so the neutral 0.5 is kept rather than a value fitted to 12 questions
    hybrid_dense_weight: float = 0.5
    # Weak-context thresholds on the top passage's score, keyed by score kind: "rerank" (cross-encoder
    # probability) or an embedding slug (cosine, used when there is no reranker). Calibrated in section 6
    # Midpoints between the highest out-of-scope and lowest answerable dev scores (3 + 12 questions)
    confidence_thresholds: dict = field(default_factory=lambda: {
        "rerank": 0.7205,  # perfect separation on dev: out-of-scope <= 0.589, answerable >= 0.852
        "bge_m3": 0.6091,  # cosine, no reranker: keeps 10 of 12 answerable, flags 3 of 3 out-of-scope
        "gemini_embedding_001": 0.7525,  # cosine, no reranker: keeps 11 of 12, flags 3 of 3
    })
    multi_query_n: int = 3
    rewrite_num_ctx: int = 2048  # the rewrite prompt is short; a small context saves GPU memory

    # Generation (local, via Ollama)
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: _env_choice("RPAB_LLM", "llama3.1:latest"))
    llm_temperature: float = 0.0
    llm_num_ctx: int = 8192  # Ollama's default context is smaller and truncates silently
    judge_model: str = "qwen2.5:7b"

    # Conversation memory (stretch goal S1)
    memory_turns: int = 6

    # Corrective RAG web fallback (stretch goal S3), used only when a request turns it on
    web_results: int = 5
    web_timeout_s: int = 8

    @property
    def chunk_spec(self):
        """The configured chunking as a ``ChunkSpec`` (imported lazily to keep config import-light)."""
        from rpab.chunking import ChunkSpec

        return ChunkSpec(self.chunking_strategy, self.chunk_size, self.chunk_overlap,
                         self.semantic_percentile, self.drop_references)

    @property
    def google_api_key(self) -> str | None:
        """Gemini key from the environment, or None when it is not set."""
        return os.getenv("GOOGLE_API_KEY") or None


CONFIG = Config()
