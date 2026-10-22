"""Build the knowledge base from the PDFs in data/pdfs.

Current stages:
  load   Load every PDF page by page, clean the text, write data/pdfs/manifest.json
         and artifacts/pages.parquet, and print a quality report.

Later milestones add chunking, embedding and indexing stages. Run from the backend folder:

    .venv\\Scripts\\python scripts\\ingest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from rpab.cleaning import clean_documents  # noqa: E402
from rpab.config import CONFIG  # noqa: E402
from rpab.loaders import load_corpus, page_stats, quality_report, write_manifest  # noqa: E402
from rpab.utils import set_seed  # noqa: E402


def stage_load() -> pd.DataFrame:
    """Load, clean and save every page; return the per-page table."""
    paths = CONFIG.paths
    raw_pages, manifest = load_corpus(paths.pdf_dir, paths.catalog, CONFIG.pdf_loader)
    pages, page_log = clean_documents(raw_pages, CONFIG.repeated_line_share)

    stats = page_stats(pages, CONFIG.empty_page_chars)
    missing = page_log[["paper_title", "page"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{missing.sum()} pages are missing a title or page number")

    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest, paths.manifest)
    page_log.merge(stats, on=["paper_id", "page"]).to_parquet(paths.pages, index=False)

    print(f"Loaded {len(manifest)} papers, {len(pages)} pages with {CONFIG.pdf_loader}")
    print(quality_report(stats).to_string())
    empty = stats[stats["is_empty"]]
    print(f"\nEmpty pages (OCR candidates): {len(empty)}")
    print(f"Wrote {paths.manifest.relative_to(paths.data_dir.parent)} and {paths.pages.relative_to(paths.data_dir.parent)}")
    return page_log


if __name__ == "__main__":
    set_seed(CONFIG.seed)
    stage_load()
