"""The paper library shown in the apps: one entry per ingested PDF, from the manifest and the chunk table."""

import json

import pandas as pd

from rpab.config import CONFIG
from rpab.schemas import PaperInfo


def list_papers(config=CONFIG) -> list[PaperInfo]:
    """Papers in manifest order, with their chunk counts when the index has been built.
    An empty list means ingestion has not run yet."""
    if not config.paths.manifest.exists():
        return []
    manifest = json.loads(config.paths.manifest.read_text(encoding="utf-8"))
    counts = (pd.read_parquet(config.paths.chunks, columns=["paper_id"])["paper_id"].value_counts().to_dict()
              if config.paths.chunks.exists() else {})
    return [PaperInfo(paper_id=p["paper_id"], paper_title=p["paper_title"], year=p.get("year"),
                      arxiv_id=p.get("arxiv_id"), url=p.get("url"), num_pages=p.get("num_pages"),
                      num_chunks=counts.get(p["paper_id"]))
            for p in manifest]


def unknown_paper_ids(paper_ids: list[str] | None, config=CONFIG) -> list[str]:
    """Ids in a request's paper filter that match no ingested paper."""
    known = {p.paper_id for p in list_papers(config)}
    return [pid for pid in paper_ids or [] if pid not in known]
