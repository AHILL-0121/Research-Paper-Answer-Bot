"""Experiment runners shared by the notebook and ``scripts/ingest.py``.

Each runner builds what it measures (chunks, a Chroma collection), scores dense retrieval
on the dev questions and writes one row to ``artifacts/experiments.csv``.
"""

import time

import pandas as pd

from rpab.chunking import ChunkSpec, chunk_pages, chunk_summary
from rpab.embeddings import CachedEmbedder
from rpab.evaluation import evaluate_retrieval, log_experiment, summarise
from rpab.vectorstore import build_collection, dense_search

EVAL_K = 10  # chunks retrieved per question; hit@5 and MRR are read from these

# The chunking grid from doc 02, section 3 (step 2)
CHUNK_GRID = [
    ChunkSpec("recursive", 500, 100), ChunkSpec("recursive", 500, 150),
    ChunkSpec("recursive", 800, 120), ChunkSpec("recursive", 800, 200),
    ChunkSpec("recursive", 1200, 150), ChunkSpec("recursive", 1200, 200),
    ChunkSpec("fixed", 256, 0), ChunkSpec("fixed", 256, 50),
    ChunkSpec("fixed", 512, 0), ChunkSpec("fixed", 512, 50),
    ChunkSpec("semantic", percentile=90), ChunkSpec("semantic", percentile=95),
    # Semantic chunks have no size limit; these cap them at the largest recursive size
    ChunkSpec("semantic", 1200, 150, percentile=90), ChunkSpec("semantic", 1200, 150, percentile=95),
]


def make_chunks(pages: pd.DataFrame, spec: ChunkSpec, embedder: CachedEmbedder, config) -> tuple[pd.DataFrame, float]:
    """Chunk the corpus for one grid point, save the table, and return it with the time taken."""
    start = time.perf_counter()
    helper = embedder.uncached() if spec.strategy == "semantic" else None
    chunks = chunk_pages(pages, spec, config.min_chunk_chars, embeddings=helper)
    seconds = time.perf_counter() - start
    config.paths.chunk_grid_dir.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(config.paths.chunk_grid_dir / f"{spec.slug}.parquet", index=False)
    return chunks, seconds


def run_retrieval_experiment(stage: str, spec: ChunkSpec, chunks: pd.DataFrame, embedder: CachedEmbedder,
                             client, questions: list[dict], config, notes: str = "",
                             rebuild: bool = False) -> tuple[dict, pd.DataFrame]:
    """Index ``chunks`` with ``embedder``, score dense retrieval and log the run.

    ``rebuild=True`` recreates the collection, so the embedding time is measured again.
    """
    seconds_before, embedded_before = embedder.seconds, embedder.embedded
    collection = build_collection(client, embedder, chunks, spec.slug, rebuild=rebuild)
    newly_embedded = embedder.embedded - embedded_before
    seconds = embedder.seconds - seconds_before
    per_question = evaluate_retrieval(questions, lambda q: dense_search(collection, embedder, q, EVAL_K))
    size = chunk_summary(chunks)
    row = {
        "stage": stage, "chunking": spec.slug, "chunking_strategy": spec.strategy,
        "chunk_size": spec.size, "chunk_overlap": spec.overlap, "embedding_model": embedder.model_name,
        "retrieval_strategy": "dense", "k": EVAL_K, "split": questions[0]["split"] if questions else "",
        **summarise(per_question),
        "num_chunks": size["num_chunks"], "mean_chunk_chars": size["mean_chars"],
        # Chunks already cached by an earlier run are not re-embedded, so the time for the whole set is
        # extrapolated from the chunks embedded now; a fully cached rerun keeps the logged value
        "embed_seconds": round(seconds / newly_embedded * len(chunks), 1) if newly_embedded else None,
        "index_size_mb": round(len(chunks) * embedder.dim * 4 / 1e6, 1),  # raw float32 vectors
        "cost_usd": round(getattr(embedder, "price_per_m_tokens", 0) * size["total_tokens"] / 1e6, 4),
        "notes": notes,
    }
    row["run_id"] = log_experiment(config.paths.experiments, row)
    per_question["run_id"] = row["run_id"]
    return row, per_question


def run_chunking_grid(specs: list[ChunkSpec], pages: pd.DataFrame, embedder: CachedEmbedder, client,
                      questions: list[dict], config) -> pd.DataFrame:
    """Chunk, index and score every grid point with one embedding model; return per-question results."""
    results = []
    for spec in specs:
        chunks, seconds = make_chunks(pages, spec, embedder, config)
        row, per_question = run_retrieval_experiment("chunking", spec, chunks, embedder, client, questions, config,
                                                     notes=f"chunking took {seconds:.1f} s")
        per_question["chunking"] = spec.slug
        results.append(per_question)
        print(f"{row['run_id']} {spec.slug:<22} chunks={row['num_chunks']:>5} hit@3={row['hit_at_3']:.3f} "
              f"mrr={row['mrr']:.3f}")
    return pd.concat(results, ignore_index=True)


def run_embedding_comparison(spec: ChunkSpec, chunks: pd.DataFrame, embedders: list[CachedEmbedder], client,
                             questions: list[dict], config) -> pd.DataFrame:
    """Index the same chunks with each model and score them on the same questions.

    A local model is re-embedded from scratch (through an uncached view) so its time is
    comparable with the API model's first run; the API model uses its cache, because
    re-embedding would spend free-tier quota for no new information.
    """
    results = []
    for embedder in embedders:
        local = not hasattr(embedder, "price_per_m_tokens")
        timed = embedder.uncached() if local else embedder
        row, per_question = run_retrieval_experiment("embedding", spec, chunks, timed, client, questions, config,
                                                     rebuild=local)
        per_question["embedding_model"] = embedder.model_name
        results.append(per_question)
        print(f"{row['run_id']} {embedder.model_name:<30} hit@3={row['hit_at_3']:.3f} mrr={row['mrr']:.3f} "
              f"embed={row['embed_seconds']} s")
    return pd.concat(results, ignore_index=True)
