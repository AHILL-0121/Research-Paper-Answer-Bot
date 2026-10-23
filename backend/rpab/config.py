"""Single source of truth for paths, model names and pipeline settings.

Every other module reads settings from ``CONFIG`` instead of hard-coding them,
so an experiment changes one value here and the whole pipeline follows.
Secrets (API keys) come from ``backend/.env.local`` or ``backend/.env`` and are
never stored in this file.
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
    experiments: Path = BACKEND_DIR / "artifacts" / "experiments.csv"
    chat_db: Path = BACKEND_DIR / "artifacts" / "chat.db"


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
    # Chosen in notebook section 4: Gemini for retrieval, bge-m3 as the offline fallback
    embedding_choice: str = "commercial"
    embedding_fallback: str = "open"

    # Retrieval
    retrieval_k: int = 20  # candidates fetched before reranking
    final_k: int = 3  # passages given to the LLM and shown to the user
    mmr_fetch_k: int = 20
    mmr_lambda: float = 0.5
    rrf_k: int = 60  # reciprocal rank fusion constant for hybrid search
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    confidence_threshold: float = 0.30  # placeholder until calibrated on dev questions
    multi_query_n: int = 3

    # Generation (local, via Ollama)
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_model: str = "llama3.1:latest"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 8192  # Ollama's default context is smaller and truncates silently
    judge_model: str = "qwen2.5:7b"

    # Conversation memory (stretch goal S1)
    memory_turns: int = 6

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
