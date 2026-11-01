"""Hugging Face dataset acquisition and schema inspection."""
from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd
from datasets import load_dataset

from .config import DATASET_ID, RAW_DIR

log = logging.getLogger(__name__)


def load_pharma_dataset(dataset_id: str = DATASET_ID, split: str = "train") -> pd.DataFrame:
    """Download/load a HF dataset and return its observed columns as a DataFrame."""
    dataset = load_dataset(dataset_id, split=split)
    frame = dataset.to_pandas()
    log.info("Loaded %s rows from %s[%s]; columns=%s", len(frame), dataset_id, split, list(frame.columns))
    return frame


def save_raw(frame: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or RAW_DIR / "indian_pharma_data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def read_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW_DIR / "indian_pharma_data.parquet"
    return pd.read_parquet(path)
