"""Central configuration and project paths."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATASET_ID = os.getenv("HF_DATASET_ID", "revooda/indian-pharma-data")
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = ROOT / "database"
DB_PATH = Path(os.getenv("SQLITE_PATH", str(DB_DIR / "products.sqlite3")))
CHROMA_DIR = Path(os.getenv("CHROMA_PATH", str(ROOT / "chroma_db")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "pharma_products")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIMENSIONS = int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
API_REQUESTS_PER_MINUTE = int(os.getenv("GEMINI_REQUESTS_PER_MINUTE", "30"))

for _path in (RAW_DIR, PROCESSED_DIR, DB_DIR, CHROMA_DIR):
    _path.mkdir(parents=True, exist_ok=True)
