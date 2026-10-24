"""Retrieval metrics against the labelled eval set, and the shared experiment log.

A retrieved chunk is a hit when its page is one of the gold pages *of its own paper*:
``gold_pages`` maps each gold paper to its pages, e.g. ``{"lora": [4], "qlora": [4]}``, so a
multi-paper question cannot be hit by the right page number in the wrong paper.

Retrieval metrics score answerable questions only, because out-of-scope questions have
no gold passage. They are not ignored: their expected behaviour is a refusal, which the
weak-context threshold (M3) and the answer evaluation (M4) score explicitly.
"""

import json
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

EXPERIMENT_COLUMNS = ["run_id", "timestamp", "stage", "chunking", "chunking_strategy", "chunk_size", "chunk_overlap",
                      "embedding_model", "retrieval_strategy", "k", "split", "questions", "hit_at_1", "hit_at_3",
                      "hit_at_5", "mrr", "precision_at_3", "latency_ms", "num_chunks", "mean_chunk_chars",
                      "embed_seconds", "index_size_mb", "cost_usd", "notes"]
# Two log rows describe the same experiment when these match; a rerun replaces the old row
EXPERIMENT_KEY = ["stage", "chunking", "embedding_model", "retrieval_strategy", "k", "split"]


def validate_eval_set(questions: list[dict]) -> None:
    """Fail loudly on an inconsistent eval file rather than silently mis-scoring it."""
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("eval question ids are not unique")
    for q in questions:
        if not isinstance(q["gold_pages"], dict) or sorted(q["gold_pages"]) != sorted(q["gold_papers"]):
            raise ValueError(f"{q['id']}: gold_pages must map exactly the gold_papers to their pages")
        if q["answerable"] != bool(q["gold_papers"]) or q["split"] not in ("dev", "held_out"):
            raise ValueError(f"{q['id']}: answerable questions need gold papers, out-of-scope ones none")


def load_eval_set(path: Path, split: str | None = None, answerable: bool | None = None) -> list[dict]:
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_eval_set(questions)
    return [q for q in questions
            if (split is None or q["split"] == split) and (answerable is None or q["answerable"] == answerable)]


def is_hit(chunk: dict, question: dict) -> bool:
    return chunk["page"] in question["gold_pages"].get(chunk["paper_id"], [])


def evaluate_retrieval(questions: list[dict], search: Callable[[str], list], run_id: str = "",
                       extra_ms: Callable[[], float] | None = None) -> pd.DataFrame:
    """Run ``search`` on every answerable question; one row per question (doc 05, section 8.2).

    ``search`` takes the question text and returns ranked chunks (``RetrievedChunk`` objects or
    dicts with ``chunk_id``, ``paper_id`` and ``page``). The reciprocal rank uses every chunk it
    returns. ``extra_ms``, called after each search, adds time the search did not spend in this
    run, such as cached LLM rephrasings.
    """
    rows = []
    for q in questions:
        if not q["answerable"]:
            continue
        start = time.perf_counter()
        results = [c.model_dump() if hasattr(c, "model_dump") else c for c in search(q["question"])]
        latency_ms = (time.perf_counter() - start) * 1000 + (extra_ms() if extra_ms else 0.0)
        hits = [is_hit(c, q) for c in results]
        first = next((rank for rank, hit in enumerate(hits, start=1) if hit), None)
        rows.append({
            "run_id": run_id, "question_id": q["id"], "type": q["type"],
            "gold_papers": ",".join(q["gold_papers"]),
            "retrieved_chunk_ids": [c["chunk_id"] for c in results],
            "hit_at_1": any(hits[:1]), "hit_at_3": any(hits[:3]), "hit_at_5": any(hits[:5]),
            "reciprocal_rank": 1 / first if first else 0.0,
            "precision_at_3": sum(hits[:3]) / 3,
            "latency_ms": round(latency_ms, 1),
        })
    return pd.DataFrame(rows)


CHUNK_ID = re.compile(r"^(?P<paper>.+)_p(?P<page>\d{3})_c\d{3}$")


def rescore(per_question: pd.DataFrame, questions: list[dict]) -> pd.DataFrame:
    """Recompute hit@k, reciprocal rank and precision@3 from the stored ``retrieved_chunk_ids``
    against the current gold labels, without re-running retrieval. A chunk id encodes its paper
    and page, so this is exact; latency is kept as measured."""
    by_id = {q["id"]: q for q in questions}
    rows = []
    for row in per_question.to_dict("records"):
        q = by_id[row["question_id"]]
        chunks = [{"paper_id": m["paper"], "page": int(m["page"])}
                  for m in (CHUNK_ID.match(cid) for cid in row["retrieved_chunk_ids"])]
        hits = [is_hit(c, q) for c in chunks]
        first = next((rank for rank, hit in enumerate(hits, start=1) if hit), None)
        rows.append({**row, "hit_at_1": any(hits[:1]), "hit_at_3": any(hits[:3]), "hit_at_5": any(hits[:5]),
                     "reciprocal_rank": 1 / first if first else 0.0, "precision_at_3": sum(hits[:3]) / 3})
    return pd.DataFrame(rows)


def summarise(per_question: pd.DataFrame) -> dict:
    """Average the per-question metrics into one experiment row."""
    return {
        "questions": len(per_question),
        "hit_at_1": round(per_question["hit_at_1"].mean(), 3),
        "hit_at_3": round(per_question["hit_at_3"].mean(), 3),
        "hit_at_5": round(per_question["hit_at_5"].mean(), 3),
        "mrr": round(per_question["reciprocal_rank"].mean(), 3),
        "precision_at_3": round(per_question["precision_at_3"].mean(), 3),
        "latency_ms": round(per_question["latency_ms"].mean(), 1),
    }


def log_experiment(path: Path, row: dict) -> str:
    """Add a row to ``experiments.csv`` and return its run id.

    A row with the same ``EXPERIMENT_KEY`` is replaced and keeps its run id, so re-running
    the notebook updates the log instead of duplicating it. Fields passed as None keep
    their old value (e.g. embedding time, which a cached rerun cannot measure).
    """
    path = Path(path)
    log = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=EXPERIMENT_COLUMNS)
    row = {col: row.get(col) for col in EXPERIMENT_COLUMNS}
    row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    same = pd.Series(True, index=log.index)
    for col in EXPERIMENT_KEY:
        same &= log[col].astype(str) == str(row[col])
    if same.any():
        old = log.loc[same].iloc[0]
        row = {col: (old[col] if row[col] is None and col != "run_id" else row[col]) for col in EXPERIMENT_COLUMNS}
        row["run_id"] = old["run_id"]
        log = log[~same]
    else:
        numbers = log["run_id"].str.extract(r"(\d+)$")[0].astype(float)
        row["run_id"] = f"exp_{int(numbers.max()) + 1 if len(numbers.dropna()) else 1:03d}"
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True).sort_values("run_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return row["run_id"]


def read_experiments(path: Path, stage: str | None = None) -> pd.DataFrame:
    log = pd.read_csv(path) if Path(path).exists() else pd.DataFrame(columns=EXPERIMENT_COLUMNS)
    return log if stage is None else log[log["stage"] == stage].reset_index(drop=True)
