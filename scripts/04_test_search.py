"""Check SQLite exact-ID lookup; add --semantic to query the Gemini index."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.db import stats, exact_id
from src.search import search_products
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="101")
    parser.add_argument("--semantic", action="store_true")
    args = parser.parse_args()
    print("SQLite:", stats())
    if args.semantic:
        for result in search_products(args.query, top_k=5):
            p = result["product"]
            print(f"{result['score']:.3f} | {p.get('brand_name')} | {p.get('primary_ingredient')} | {result['explanation']}")
    else:
        print("Exact ID:", exact_id(args.query))
