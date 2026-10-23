"""Step 4: persistent Chroma collections, one per embedding model and chunking setup.

Vectors are computed by ``rpab.embeddings`` and handed to Chroma, never computed by
Chroma itself, so a collection always holds the vectors of exactly one model. The
collection name and metadata record which model and chunking produced them.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd

from rpab.embeddings import CachedEmbedder

METADATA_COLUMNS = ["paper_id", "paper_title", "source_file", "page", "chunk_index", "char_count", "chunking_strategy"]
ADD_BATCH = 4000  # Chroma rejects batches above about 5,400 records


def get_client(chroma_dir: Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(chroma_dir), settings=chromadb.Settings(anonymized_telemetry=False))


def collection_name(embedding_slug: str, chunking_slug: str) -> str:
    """``papers__{embedding}__{chunking}``, e.g. ``papers__bge_m3__recursive_800_120``."""
    return f"papers__{embedding_slug}__{chunking_slug}"


def build_collection(client: chromadb.ClientAPI, embedder: CachedEmbedder, chunks: pd.DataFrame,
                     chunking_slug: str, rebuild: bool = False) -> chromadb.Collection:
    """Embed the chunks and store them with their metadata in a cosine-distance collection.

    An existing collection with the same name and the same number of records is reused,
    because the name fixes both the model and the chunking. ``rebuild=True`` recreates it.
    """
    name = collection_name(embedder.slug, chunking_slug)
    existing = {c.name for c in client.list_collections()}
    if name in existing and not rebuild:
        collection = client.get_collection(name)
        if collection.count() == len(chunks):
            return collection
    if name in existing:
        client.delete_collection(name)

    vectors = embedder.embed_array(chunks["text"].tolist())
    first = chunks.iloc[0]
    collection = client.create_collection(name, metadata={
        "hnsw:space": "cosine",
        "embedding_model": embedder.model_name,
        "embedding_dim": int(vectors.shape[1]),
        "chunking": chunking_slug,
        "chunking_strategy": str(first["chunking_strategy"]),
        "chunk_size": int(first["chunk_size"]),
        "chunk_overlap": int(first["chunk_overlap"]),
        "num_chunks": len(chunks),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    metadatas = [{k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in row.items()}
                 for row in chunks[METADATA_COLUMNS].to_dict("records")]
    for i in range(0, len(chunks), ADD_BATCH):
        collection.add(ids=chunks["chunk_id"].iloc[i:i + ADD_BATCH].tolist(),
                       documents=chunks["text"].iloc[i:i + ADD_BATCH].tolist(),
                       embeddings=vectors[i:i + ADD_BATCH],
                       metadatas=metadatas[i:i + ADD_BATCH])
    validate_collection(collection, chunks)
    return collection


def validate_collection(collection: chromadb.Collection, chunks: pd.DataFrame) -> None:
    """The integrity checks from the backend schema (doc 05, section 13)."""
    assert collection.count() == len(chunks), f"{collection.name}: {collection.count()} vectors, {len(chunks)} chunks"
    assert chunks[["paper_title", "page", "text"]].notna().all().all(), "chunk table has missing citation fields"
    assert chunks["chunk_id"].is_unique, "chunk_id values are not unique"
    sample = collection.get(limit=5, include=["metadatas"])
    assert all("page" in m and "paper_title" in m for m in sample["metadatas"]), "collection metadata incomplete"


def dense_search(collection: chromadb.Collection, embedder: CachedEmbedder, question: str, k: int,
                 where: dict | None = None) -> list[dict]:
    """Top-k chunks by cosine similarity. ``score`` is 1 - cosine distance."""
    if collection.metadata.get("embedding_model") != embedder.model_name:
        raise ValueError(f"{collection.name} was built with {collection.metadata.get('embedding_model')}, "
                         f"not {embedder.model_name}")
    result = collection.query(query_embeddings=[embedder.embed_query(question)], n_results=k, where=where,
                              include=["documents", "metadatas", "distances"])
    return [{"chunk_id": cid, **meta, "text": doc, "score": round(1 - dist, 4)}
            for cid, meta, doc, dist in zip(result["ids"][0], result["metadatas"][0],
                                            result["documents"][0], result["distances"][0])]


def time_cold_reload(name: str, kind: str, question: str) -> dict:
    """Open a persisted collection from a new Python process, load the query embedder and run one
    query, timing each step. A separate process makes it a true cold start (no warm client or model)."""
    script = f"""
import json, sys, time
start = time.perf_counter()
sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
from rpab.config import CONFIG
from rpab.embeddings import load_embedder
from rpab.vectorstore import dense_search, get_client
collection = get_client(CONFIG.paths.chroma_dir).get_collection({name!r})
count = collection.count()
opened = time.perf_counter()
embedder = load_embedder({kind!r}, CONFIG)
loaded = time.perf_counter()
dense_search(collection, embedder, {question!r}, 3)
done = time.perf_counter()
print(json.dumps({{"vectors": count, "open_index_s": round(opened - start, 1),
                  "load_embedder_s": round(loaded - opened, 1), "first_query_s": round(done - loaded, 1),
                  "total_s": round(done - start, 1)}}))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True,
                            encoding="utf-8", errors="replace",
                            env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"})
    return {"collection": name, **json.loads(result.stdout.strip().splitlines()[-1])}
