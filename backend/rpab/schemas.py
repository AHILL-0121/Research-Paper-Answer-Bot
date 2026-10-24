"""Data contracts shared by retrieval, the RAG chain and the apps (doc 05, section 6)."""

from typing import Optional

from pydantic import BaseModel


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
