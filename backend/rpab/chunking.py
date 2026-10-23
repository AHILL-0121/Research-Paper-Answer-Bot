"""Step 2: split cleaned pages into chunks that keep their citation metadata.

Three strategies are compared:

- ``recursive``: LangChain's ``RecursiveCharacterTextSplitter``, sizes in characters.
- ``fixed``: ``TokenTextSplitter`` with the cl100k tokenizer, sizes in tokens.
- ``semantic``: ``SemanticChunker``, which starts a new chunk where the embedding distance
  between neighbouring sentences is above a percentile of all distances on the page. It has
  no size limit, so with a non-zero ``size`` any longer chunk is re-split recursively to
  ``size`` characters with ``overlap``.

Every strategy splits one page at a time, so a chunk never crosses a page boundary and
its page number is always exact. Chunks shorter than ``min_chars`` are dropped; they are
mostly stray figure labels and table fragments.
"""

import hashlib
import re
from dataclasses import dataclass

import pandas as pd
import tiktoken
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

TOKEN_ENCODING = "cl100k_base"  # same tokenizer as the EDA token counts
CHUNK_COLUMNS = ["chunk_id", "paper_id", "paper_title", "source_file", "page", "chunk_index", "text",
                 "char_count", "token_count", "chunking_strategy", "chunk_size", "chunk_overlap", "text_hash"]

# A reference-list chunk is dense with citation years and venue words
YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
VENUE = re.compile(r"\b(?:Proceedings|Conference|arXiv|preprint|Journal|Transactions|URL|https?://)", re.IGNORECASE)


@dataclass(frozen=True)
class ChunkSpec:
    """One point of the chunking grid.

    For ``semantic``, ``percentile`` sets the breakpoint and ``size``/``overlap`` (optional)
    cap the chunk length.
    """

    strategy: str
    size: int = 0
    overlap: int = 0
    percentile: int = 95
    drop_references: bool = False

    @property
    def slug(self) -> str:
        if self.strategy == "semantic":
            base = f"semantic_p{self.percentile}" + (f"_max{self.size}_{self.overlap}" if self.size else "")
        else:
            base = f"{self.strategy}_{self.size}_{self.overlap}"
        return base + ("_noref" if self.drop_references else "")


def text_hash(text: str) -> str:
    """SHA-256 of the chunk text; the embedding cache is keyed by it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def looks_like_references(text: str) -> bool:
    """True when a chunk reads like a bibliography: many years plus venue words for its length.

    Thresholds per 1,000 characters were set by reading samples: body text that cites heavily
    ("(Wang et al., 2019, 2018; Lin et al., 2018)") has years but almost never venue words.
    """
    per_kchar = 1000 / max(len(text), 1)
    return len(YEAR.findall(text)) * per_kchar >= 4 and len(VENUE.findall(text)) * per_kchar >= 1


def make_splitter(spec: ChunkSpec, embeddings: Embeddings | None = None):
    """Build the LangChain splitter for one grid point."""
    if spec.strategy == "recursive":
        return RecursiveCharacterTextSplitter(chunk_size=spec.size, chunk_overlap=spec.overlap)
    if spec.strategy == "fixed":
        return TokenTextSplitter(encoding_name=TOKEN_ENCODING, chunk_size=spec.size, chunk_overlap=spec.overlap)
    if spec.strategy == "semantic":
        from langchain_experimental.text_splitter import SemanticChunker

        if embeddings is None:
            raise ValueError("The semantic chunker needs an embedding model")
        return SemanticChunker(embeddings, breakpoint_threshold_type="percentile",
                               breakpoint_threshold_amount=spec.percentile)
    raise ValueError(f"Unknown chunking strategy: {spec.strategy}")


def chunk_pages(pages: pd.DataFrame, spec: ChunkSpec, min_chars: int,
                embeddings: Embeddings | None = None) -> pd.DataFrame:
    """Split every page and return the chunk table (one row per chunk, ``CHUNK_COLUMNS``).

    ``pages`` is the page table from ingestion (``artifacts/pages.parquet``); its
    ``clean_text`` column is split. ``chunk_index`` counts kept chunks within a page, so
    ``chunk_id`` is ``{paper_id}_p{page:03d}_c{index:03d}`` and stays the same across reruns.
    """
    splitter = make_splitter(spec, embeddings)
    cap = (RecursiveCharacterTextSplitter(chunk_size=spec.size, chunk_overlap=spec.overlap)
           if spec.strategy == "semantic" and spec.size else None)
    enc = tiktoken.get_encoding(TOKEN_ENCODING)
    rows = []
    for page in pages.sort_values(["paper_id", "page"]).itertuples(index=False):
        pieces = [p.strip() for p in splitter.split_text(page.clean_text)]
        if cap:
            pieces = [part for p in pieces for part in (cap.split_text(p) if len(p) > spec.size else [p])]
        kept = [p for p in pieces if len(p) >= min_chars and not (spec.drop_references and looks_like_references(p))]
        for index, text in enumerate(kept):
            rows.append({
                "chunk_id": f"{page.paper_id}_p{page.page:03d}_c{index:03d}",
                "paper_id": page.paper_id,
                "paper_title": page.paper_title,
                "source_file": page.source_file,
                "page": int(page.page),
                "chunk_index": index,
                "text": text,
                "char_count": len(text),
                "token_count": len(enc.encode(text, disallowed_special=())),
                "chunking_strategy": spec.strategy,
                "chunk_size": spec.size,
                "chunk_overlap": spec.overlap,
                "text_hash": text_hash(text),
            })
    chunks = pd.DataFrame(rows, columns=CHUNK_COLUMNS)
    validate_chunks(chunks)
    return chunks


def validate_chunks(chunks: pd.DataFrame) -> None:
    """Fail loudly if a chunk lost its citation metadata or an id repeats."""
    if chunks.empty:
        raise ValueError("Chunking produced no chunks")
    if chunks[["paper_title", "page", "text"]].isna().any().any():
        raise ValueError("Some chunks are missing paper_title, page or text")
    if not chunks["chunk_id"].is_unique:
        raise ValueError("chunk_id values are not unique")


def chunk_summary(chunks: pd.DataFrame) -> dict:
    """Size statistics reported for each chunking run."""
    return {
        "num_chunks": len(chunks),
        "mean_chars": round(float(chunks["char_count"].mean()), 1),
        "median_chars": float(chunks["char_count"].median()),
        "mean_tokens": round(float(chunks["token_count"].mean()), 1),
        "max_tokens": int(chunks["token_count"].max()),
        "total_tokens": int(chunks["token_count"].sum()),
    }
