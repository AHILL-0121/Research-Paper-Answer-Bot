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


def run_strategy_experiment(retriever, strategy: str, questions: list[dict], config, label: str | None = None,
                            notes: str = "") -> tuple[dict, pd.DataFrame]:
    """Score one retrieval strategy and log it as stage ``retrieval``.

    Two passes: an untimed warm-up pass embeds every query once (and fills the multi-query
    cache), then the timed pass reuses those vectors and is charged the time each real
    embedding and LLM rephrasing took. Latency then excludes one-time warm-up and free-tier
    pacing waits but includes everything a single user question costs.
    """
    embedder = retriever.embedder
    embedder.cache_queries = True
    answerable = [q for q in questions if q["answerable"]]
    for q in answerable:
        retriever.retrieve(q["question"], strategy, k=EVAL_K)
    retries_before = getattr(embedder, "retries", 0)
    mark = {"query_seconds": embedder.query_seconds}

    def charged_ms() -> float:
        spent = embedder.query_seconds - mark["query_seconds"]
        mark["query_seconds"] = embedder.query_seconds
        return (spent + retriever.last_llm_seconds) * 1000

    per_question = evaluate_retrieval(questions, lambda text: retriever.retrieve(text, strategy, k=EVAL_K),
                                      extra_ms=charged_ms)
    retries = getattr(embedder, "retries", 0) - retries_before
    spec = config.chunk_spec
    row = {
        "stage": "retrieval", "chunking": spec.slug, "chunking_strategy": spec.strategy, "chunk_size": spec.size,
        "chunk_overlap": spec.overlap, "embedding_model": embedder.model_name,
        "retrieval_strategy": label or strategy, "k": EVAL_K, "split": questions[0]["split"],
        **summarise(per_question), "num_chunks": len(retriever.chunks), "cost_usd": 0.0,
        "notes": "; ".join(filter(None, [notes, f"{retries} API retries in the timed pass" if retries else ""])),
    }
    row["run_id"] = log_experiment(config.paths.experiments, row)
    per_question["run_id"] = row["run_id"]
    per_question["strategy"] = label or strategy
    per_question["embedding_model"] = embedder.model_name
    return row, per_question


def confidence_table(retriever, strategy: str, questions: list[dict]) -> pd.DataFrame:
    """The top passage's score for every question, answerable or not: the input for calibrating
    the weak-context threshold. ``top_is_gold`` marks answerable questions whose top passage is a hit."""
    from rpab.evaluation import is_hit

    rows = []
    for q in questions:
        results = retriever.retrieve(q["question"], strategy, k=retriever.config.final_k)
        score, kind = retriever.confidence(results)
        rows.append({"question_id": q["id"], "split": q["split"], "type": q["type"], "answerable": q["answerable"],
                     "score": score, "kind": kind,
                     "top_is_gold": bool(results) and q["answerable"] and is_hit(results[0].model_dump(), q)})
    return pd.DataFrame(rows)


def calibrate_threshold(table: pd.DataFrame) -> dict:
    """Pick the weak-context threshold that best separates answerable from out-of-scope questions.

    Candidates are midpoints between sorted scores. The objective is balanced accuracy (the mean of
    the share of out-of-scope questions flagged and the share of answerable ones not flagged); ties
    go to the lower threshold, because wrongly flagging an answerable question costs more than
    missing a refusal the LLM is still prompted to make.
    """
    scores = sorted(table["score"].dropna().unique())
    candidates = [(a + b) / 2 for a, b in zip(scores, scores[1:])] or scores
    answerable, oos = table[table["answerable"]]["score"], table[~table["answerable"]]["score"]
    best = None
    for t in candidates:
        kept = (answerable >= t).mean()
        flagged = (oos < t).mean() if len(oos) else 0.0
        balanced = (kept + flagged) / 2
        if best is None or balanced > best["balanced_accuracy"] + 1e-12:
            best = {"threshold": round(float(t), 4), "balanced_accuracy": round(float(balanced), 3),
                    "answerable_kept": round(float(kept), 3), "out_of_scope_flagged": round(float(flagged), 3)}
    return best


def run_strategy_sweep(retriever_for, kinds: list[str], questions: list[dict], config) -> pd.DataFrame:
    """The dev comparison of notebook section 6, logged run by run.

    ``retriever_for(kind, config)`` builds a ``Retriever`` for embedding ``kind`` with the given
    config, so parameter variants (MMR lambda, hybrid weight) get their own retriever. BM25 runs
    once because it uses no embedding. Reranking and multi-query use ``config.hybrid_dense_weight``.
    """
    import dataclasses

    results = []

    def run(retriever, strategy, label=None, notes=""):
        row, per_question = run_strategy_experiment(retriever, strategy, questions, retriever.config, label, notes)
        results.append(per_question)
        print(f"{row['run_id']} {retriever.embedder.slug:<21} {label or strategy:<14} hit@1={row['hit_at_1']:.3f} "
              f"hit@3={row['hit_at_3']:.3f} mrr={row['mrr']:.3f} latency={row['latency_ms']:.0f} ms")

    for n, kind in enumerate(kinds):
        if n == 0:
            run(retriever_for(kind, config), "bm25", notes="keyword only; the embedding model is not used")
        run(retriever_for(kind, config), "dense")
        for lam in (0.5, 0.7):
            run(retriever_for(kind, dataclasses.replace(config, mmr_lambda=lam)), "mmr", f"mmr_l{lam}")
        for w in (0.3, 0.5, 0.7):
            run(retriever_for(kind, dataclasses.replace(config, hybrid_dense_weight=w)), "hybrid", f"hybrid_w{w}")
        weight = f"hybrid dense weight {config.hybrid_dense_weight}"
        run(retriever_for(kind, config), "hybrid_rerank", notes=f"{weight}; {config.retrieval_k} candidates reranked")
        run(retriever_for(kind, config), "multi_query",
            notes=f"{weight}; {config.multi_query_n} rephrasings; reranked")
    return pd.concat(results, ignore_index=True)


# The held-out plan, fixed before the held-out questions were scored: (embedding kind, strategy, role)
HELD_OUT_PLAN = [
    ("open", "hybrid_rerank", "final system"),
    ("open", "dense", "baseline"),
    ("open", "hybrid", "CPU profile (no reranker)"),
    ("commercial", "hybrid_rerank", "alternative embedding"),
    ("commercial", "dense", "alternative baseline"),
]


def run_held_out(retriever_for, questions: list[dict], config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score ``HELD_OUT_PLAN`` once on the held-out questions (reported, never used for choices),
    and check the weak-context thresholds on the final system and the CPU profile."""
    results, confidence = [], []
    for kind, strategy, role in HELD_OUT_PLAN:
        retriever = retriever_for(kind, config)
        row, per_question = run_strategy_experiment(retriever, strategy, questions, config,
                                                    notes=f"held-out, reported only: {role}")
        results.append(per_question.assign(role=role))
        if kind == config.embedding_choice and strategy in ("hybrid_rerank", "hybrid"):
            table = confidence_table(retriever, strategy, questions).assign(strategy=strategy, role=role)
            table["flagged_weak"] = [retriever.is_weak(retriever.retrieve(q["question"], strategy)) for q in questions]
            confidence.append(table)
    return pd.concat(results, ignore_index=True), pd.concat(confidence, ignore_index=True)
