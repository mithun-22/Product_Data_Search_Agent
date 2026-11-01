"""Build/resume the persistent Gemini + Chroma vector index."""
import argparse
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vector_store import index_products
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Index at most this many records (smoke run)")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    print(index_products(batch_size=args.batch_size or 16, limit=args.limit))
