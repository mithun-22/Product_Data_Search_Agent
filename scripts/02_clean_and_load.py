"""Normalize downloaded source data and upsert it into SQLite."""
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RAW_DIR, PROCESSED_DIR
from src.dataset import read_raw
from src.data_cleaning import clean_dataset
from src.db import load_products
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
if __name__ == "__main__":
    raw_path = RAW_DIR / "indian_pharma_data.parquet"
    if not raw_path.exists():
        raise SystemExit("Raw dataset missing. Run: python scripts/01_download_dataset.py")
    cleaned = clean_dataset(read_raw(raw_path))
    processed_path = PROCESSED_DIR / "indian_pharma_data_clean.jsonl"
    cleaned.to_json(processed_path, orient="records", lines=True, force_ascii=False)
    count = load_products(cleaned)
    print(f"Cleaned records upserted: {count:,}\nProcessed data: {processed_path}")
