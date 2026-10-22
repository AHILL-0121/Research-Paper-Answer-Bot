"""Step 1: find the paper PDFs, load them page by page and describe their quality.

Each PDF page becomes one LangChain ``Document``. Its metadata carries everything
a citation needs later: ``paper_id``, ``paper_title``, ``source_file`` and a
1-based ``page`` number that matches what a PDF viewer shows.
"""

import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain_core.documents import Document

LOADERS = {"pypdf": PyPDFLoader, "pdfplumber": PDFPlumberLoader}
ARXIV_ID = re.compile(r"arXiv[ _-]?(\d{4}\.\d{4,5})", re.IGNORECASE)
LIGATURE = re.compile("[ﬀ-ﬆ]")
HYPHEN_BREAK = re.compile(r"\w-\n\w")


def load_catalog(path: Path) -> dict[str, dict]:
    """Read the hand-written paper list and index it by arXiv id."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    return {entry["arxiv_id"]: entry for entry in entries}


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    """Return every PDF in the folder, sorted so runs are repeatable."""
    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}. Add the papers first.")
    return pdfs


def identify(pdf_path: Path, catalog: dict[str, dict]) -> dict:
    """Match a PDF to its catalog entry using the arXiv id in the file name.

    Fails loudly instead of guessing, because a wrong title would produce wrong citations.
    """
    match = ARXIV_ID.search(pdf_path.name)
    if not match or match.group(1) not in catalog:
        raise ValueError(f"{pdf_path.name}: no matching arXiv id in data/papers.json")
    return catalog[match.group(1)]


def file_sha256(path: Path) -> str:
    """Hash the file so a changed PDF can be detected and re-ingested."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pdf(pdf_path: Path, paper: dict, loader: str = "pypdf") -> list[Document]:
    """Load one PDF as a list of page Documents with citation metadata.

    Page numbers come from the page's position in the file (1-based), not from the
    loader's own metadata, so both loaders number pages the same way.
    """
    raw_pages = LOADERS[loader](str(pdf_path)).load()
    return [
        Document(
            page_content=page.page_content,
            metadata={
                "paper_id": paper["paper_id"],
                "paper_title": paper["paper_title"],
                "source_file": pdf_path.name,
                "page": number,
                "loader": loader,
            },
        )
        for number, page in enumerate(raw_pages, start=1)
    ]


def load_corpus(pdf_dir: Path, catalog_path: Path, loader: str = "pypdf") -> tuple[list[Document], list[dict]]:
    """Load every PDF and build the manifest records that describe each paper."""
    catalog = load_catalog(catalog_path)
    pages, manifest = [], []
    for pdf_path in discover_pdfs(pdf_dir):
        paper = identify(pdf_path, catalog)
        docs = load_pdf(pdf_path, paper, loader)
        pages.extend(docs)
        manifest.append({
            "paper_id": paper["paper_id"],
            "paper_title": paper["paper_title"],
            "year": paper["year"],
            "arxiv_id": paper["arxiv_id"],
            "url": f"https://arxiv.org/abs/{paper['arxiv_id']}",
            "source_file": pdf_path.name,
            "num_pages": len(docs),
            "loader_used": loader,
            "file_hash": file_sha256(pdf_path),
        })
    return pages, manifest


def write_manifest(records: list[dict], path: Path) -> None:
    """Save the manifest as pretty-printed JSON next to the PDFs."""
    Path(path).write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def text_stats(text: str, empty_chars: int) -> dict:
    """Quality signals for one page of extracted text."""
    stripped = text.strip()
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "n_chars": len(stripped),
        "n_words": len(stripped.split()),
        "is_empty": len(stripped) < empty_chars,
        "ligatures": len(LIGATURE.findall(text)),
        "cid_glyphs": text.count("(cid:"),  # glyphs the loader could not map to characters
        "hyphen_breaks": len(HYPHEN_BREAK.findall(text)),
        "non_ascii_share": sum(ord(c) > 127 for c in stripped) / max(len(stripped), 1),
        "mean_line_chars": sum(len(line) for line in lines) / max(len(lines), 1),
    }


def page_stats(docs: list[Document], empty_chars: int) -> pd.DataFrame:
    """One row of quality signals per page."""
    rows = [{"paper_id": d.metadata["paper_id"], "page": d.metadata["page"], **text_stats(d.page_content, empty_chars)}
            for d in docs]
    return pd.DataFrame(rows)


def quality_report(stats: pd.DataFrame) -> pd.DataFrame:
    """Summarise page statistics per paper."""
    return (
        stats.groupby("paper_id")
        .agg(pages=("page", "count"), empty_pages=("is_empty", "sum"), total_chars=("n_chars", "sum"),
             mean_chars_per_page=("n_chars", "mean"), min_chars_per_page=("n_chars", "min"),
             ligatures=("ligatures", "sum"), cid_glyphs=("cid_glyphs", "sum"),
             hyphen_breaks=("hyphen_breaks", "sum"), non_ascii_share=("non_ascii_share", "mean"))
        .round(3)
        .sort_values("pages", ascending=False)
    )


def compare_loaders(pdf_dir: Path, catalog_path: Path, empty_chars: int,
                    loaders: tuple[str, ...] = ("pypdf", "pdfplumber")) -> pd.DataFrame:
    """Load every paper with each loader and record speed and extraction quality."""
    catalog = load_catalog(catalog_path)
    rows = []
    for pdf_path in discover_pdfs(pdf_dir):
        paper = identify(pdf_path, catalog)
        for loader in loaders:
            start = time.perf_counter()
            docs = load_pdf(pdf_path, paper, loader)
            seconds = time.perf_counter() - start
            stats = page_stats(docs, empty_chars)
            rows.append({
                "paper_id": paper["paper_id"], "loader": loader, "seconds": round(seconds, 2),
                "pages": len(docs), "total_chars": int(stats["n_chars"].sum()),
                "empty_pages": int(stats["is_empty"].sum()), "ligatures": int(stats["ligatures"].sum()),
                "cid_glyphs": int(stats["cid_glyphs"].sum()), "hyphen_breaks": int(stats["hyphen_breaks"].sum()),
                "mean_line_chars": round(stats["mean_line_chars"].mean(), 1),
            })
    return pd.DataFrame(rows)
