"""Build the knowledge base from the PDFs in data/pdfs.

Stages, run in this order by default:
  load   Load every PDF page by page, clean the text, write data/pdfs/manifest.json
         and artifacts/pages.parquet, and print a quality report.
  chunk  Split the cleaned pages with the chunking set in rpab/config.py and write
         artifacts/chunks.parquet, the canonical chunk table.
  index  Embed the chunks with both embedding models, store each in its own Chroma
         collection under artifacts/chroma, and run the integrity checks.

Run from the backend folder, optionally naming the stages to run:

    .venv\\Scripts\\python scripts\\ingest.py
    .venv\\Scripts\\python scripts\\ingest.py chunk index
    .venv\\Scripts\\python scripts\\ingest.py index --models open    (one embedding model only)
"""

import argparse
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


def stage_chunk() -> pd.DataFrame:
    """Chunk the saved pages with the configured strategy and write the canonical chunk table."""
    from rpab.chunking import chunk_pages, chunk_summary
    from rpab.embeddings import load_embedder

    paths = CONFIG.paths
    spec = CONFIG.chunk_spec
    embedder = load_embedder("open", CONFIG).uncached() if spec.strategy == "semantic" else None
    chunks = chunk_pages(pd.read_parquet(paths.pages), spec, CONFIG.min_chunk_chars, embeddings=embedder)
    chunks.to_parquet(paths.chunks, index=False)
    print(f"Chunked with {spec.slug}: {chunk_summary(chunks)}")
    print(f"Wrote {paths.chunks.relative_to(paths.data_dir.parent)}")
    return chunks


def stage_index(models: tuple[str, ...] = ("open", "commercial")) -> None:
    """Embed the canonical chunks with each model and build one Chroma collection per model."""
    from rpab.embeddings import load_embedder
    from rpab.vectorstore import build_collection, get_client

    chunks = pd.read_parquet(CONFIG.paths.chunks)
    client = get_client(CONFIG.paths.chroma_dir)
    for kind in models:
        if kind == "commercial" and not CONFIG.google_api_key:
            print("Skipping the commercial model: GOOGLE_API_KEY is not set")
            continue
        embedder = load_embedder(kind, CONFIG)
        collection = build_collection(client, embedder, chunks, CONFIG.chunk_spec.slug)
        print(f"{collection.name}: {collection.count()} vectors, checks passed "
              f"({embedder.embedded} newly embedded in {embedder.seconds:.0f} s)")


STAGES = {"load": stage_load, "chunk": stage_chunk, "index": stage_index}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the knowledge base.")
    parser.add_argument("stages", nargs="*", choices=list(STAGES), default=list(STAGES))
    parser.add_argument("--models", nargs="+", choices=["open", "commercial"], default=["open", "commercial"],
                        help="embedding models to index with (index stage only)")
    args = parser.parse_args()
    set_seed(CONFIG.seed)
    for name in args.stages:
        if name == "index":
            stage_index(tuple(args.models))
        else:
            STAGES[name]()
