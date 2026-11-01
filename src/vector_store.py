"""Persistent ChromaDB storage with SQLite-backed record retrieval."""
from __future__ import annotations

import logging
import hashlib
from tqdm import tqdm
import chromadb

from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_BATCH_SIZE
from .db import connect
from .gemini_service import GeminiEmbeddingService

log = logging.getLogger(__name__)


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def index_products(batch_size: int = EMBEDDING_BATCH_SIZE, limit: int | None = None, service: GeminiEmbeddingService | None = None) -> dict:
    collection = get_collection()
    service = service or GeminiEmbeddingService()
    with connect() as conn:
        query = "SELECT record_id,search_text FROM products WHERE search_text != '' ORDER BY record_id"
        if limit:
            query += f" LIMIT {int(limit)}"
        rows = conn.execute(query).fetchall()
    indexed = skipped = failed = 0
    for start in tqdm(range(0, len(rows), batch_size), desc="Embedding products"):
        batch = rows[start:start + batch_size]
        ids = [str(r["record_id"]) for r in batch]
        existing_result = collection.get(ids=ids, include=["metadatas"])
        existing_hashes = {
            rid: (metadata or {}).get("text_hash")
            for rid, metadata in zip(existing_result.get("ids", []), existing_result.get("metadatas", []))
        }
        todo = [(r["record_id"], r["search_text"]) for r in batch
                if existing_hashes.get(r["record_id"]) != hashlib.sha256(r["search_text"].encode("utf-8")).hexdigest()]
        skipped += len(batch) - len(todo)
        if not todo:
            continue
        try:
            vectors = service.embed([r[1] for r in todo], "RETRIEVAL_DOCUMENT")
            collection.upsert(ids=[r[0] for r in todo], embeddings=vectors,
                              metadatas=[{"record_id": r[0], "text_hash": hashlib.sha256(r[1].encode("utf-8")).hexdigest()} for r in todo])
            indexed += len(todo)
        except Exception:
            failed += len(todo)
            log.exception("Failed indexing batch beginning at offset %d; rerun to resume", start)
            raise
    return {"indexed": indexed, "already_indexed": skipped, "failed": failed, "total_source_rows": len(rows), "collection_count": collection.count()}


def vector_search(query: str, top_k: int = 10, service: GeminiEmbeddingService | None = None) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    service = service or GeminiEmbeddingService()
    vector = service.embed([query], "RETRIEVAL_QUERY")[0]
    result = collection.query(query_embeddings=[vector], n_results=min(max(1, top_k), collection.count()), include=["distances", "metadatas"])
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [{"record_id": rid, "score": max(0.0, 1.0 - float(distance)), "distance": float(distance)} for rid, distance in zip(ids, distances)]


def collection_stats() -> dict:
    try:
        collection = get_collection()
        return {"count": collection.count(), "name": COLLECTION_NAME}
    except Exception as exc:
        log.warning("Unable to read Chroma collection: %s", exc)
        return {"count": 0, "name": COLLECTION_NAME}
