"""Download and inspect the configured Hugging Face dataset."""
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import load_pharma_dataset, save_raw
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
if __name__ == "__main__":
    frame = load_pharma_dataset()
    output = save_raw(frame)
    print(f"Rows: {len(frame):,}\nColumns ({len(frame.columns)}): {list(frame.columns)}\nSaved: {output}")
