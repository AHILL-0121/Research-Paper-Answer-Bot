"""Data contracts shared by retrieval, the RAG chain and the apps (doc 05, section 6)."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """One retrieved passage with its citation metadata and the scores that ranked it."""

    chunk_id: str
    paper_id: str
    paper_title: str
    page: int
    text: str
    score: float  # final score after fusion or rerank; higher is better
    dense_score: Optional[float] = None  # cosine similarity
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None  # cross-encoder probability, 0 to 1


class Source(BaseModel):
    """A citation card, built from chunk metadata only and never parsed from the LLM's text."""

    rank: int  # 1 to final_k, the number the answer cites as [rank]
    paper_title: str
    page: int
    snippet: str  # the passage's opening, cut at a word boundary
    full_text: str
    score: float
    origin: Literal["paper", "web"] = "paper"
    url: Optional[str] = None  # web sources only (S3)
    chunk_id: Optional[str] = None  # beyond doc 05: lets evaluation check gold pages and the UI key its cards
    paper_id: Optional[str] = None


class PaperInfo(BaseModel):
    """One paper in the library, for the apps' paper list and filter."""

    paper_id: str
    paper_title: str
    year: Optional[int] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    num_pages: Optional[int] = None
    num_chunks: Optional[int] = None  # None until the index is built


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    session_id: Optional[str] = None  # S1
    strategy: Literal["dense", "mmr", "hybrid", "hybrid_rerank", "multi_query"] = "hybrid_rerank"
    paper_ids: Optional[list[str]] = None
    final_k: int = Field(default=3, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]  # always final_k when the index returned passages, refusals included
    refused: bool = False  # True when the model said it does not know
    rewritten_question: Optional[str] = None  # S1
    strategy_used: str  # differs from the request when hybrid_rerank falls back to hybrid (reranker off)
    latency_ms: int  # retrieval plus generation
    used_web_fallback: bool = False  # S3
    # Debug fields beyond doc 05's contract, shown in the apps' debug drawer
    weak_context: bool = False  # top passage scored below the calibrated threshold
    retrieval_ms: Optional[int] = None
    first_token_ms: Optional[int] = None  # from the request to the first answer token
    removed_citations: list[str] = []  # markers the sanitiser dropped, e.g. ["[4]"]
