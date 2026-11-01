"""Schema-aware normalization while retaining all source attributes."""
from __future__ import annotations

import ast
import json
import logging
from typing import Any
import pandas as pd

from .serializer import clean_value

log = logging.getLogger(__name__)
REQUIRED_SOURCE_FIELDS = {"product_id", "brand_name", "manufacturer"}


def _text(value: Any) -> str | None:
    value = clean_value(value)
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _ingredients(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return clean_value(value)
    raw = _text(value)
    if raw is None:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
            if isinstance(parsed, (list, dict)):
                return clean_value(parsed)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return raw


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate observed schema, sanitize nulls, and add normalized search fields."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    if frame.empty:
        return pd.DataFrame(columns=[*frame.columns, "record_id", "product_name", "search_text", "source_json"])
    missing = REQUIRED_SOURCE_FIELDS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required identity/search fields: {sorted(missing)}; observed={list(frame.columns)}")

    out = frame.copy()
    for col in out.columns:
        # Per-cell conversion avoids pandas scalar fillna pitfalls and handles nullable extension dtypes.
        out[col] = out[col].map(clean_value)
    out["record_id"] = out["product_id"].map(lambda x: f"hf:{x}" if x is not None else None)
    out["product_name"] = out["brand_name"].map(_text)
    if "active_ingredients" in out:
        out["active_ingredients"] = out["active_ingredients"].map(_ingredients)

    def row_text(row: pd.Series) -> str:
        labels = {
            "product_name": "Product", "brand_name": "Brand", "manufacturer": "Manufacturer",
            "manufacturer_raw": "Manufacturer", "primary_ingredient": "Active ingredient",
            "active_ingredients": "Active ingredients", "primary_strength": "Strength",
            "dosage_form": "Dosage form", "pack_size": "Pack size", "pack_unit": "Pack unit",
            "price_inr": "Price INR", "therapeutic_class": "Therapeutic class",
            "packaging_raw": "Packaging", "is_discontinued": "Discontinued",
        }
        parts = []
        for col, label in labels.items():
            if col in row.index and row[col] is not None:
                value = row[col]
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                parts.append(f"{label}: {value}")
        if row.get("product_id") is not None:
            parts.insert(0, f"Product ID: {row['product_id']}")
        return " | ".join(parts)

    out["search_text"] = out.apply(row_text, axis=1)
    out["source_json"] = out.apply(lambda row: json.dumps({k: clean_value(v) for k, v in row.items() if k not in {"source_json", "search_text"}}, ensure_ascii=False), axis=1)
    duplicate = out["record_id"].duplicated(keep="first")
    if duplicate.any():
        log.warning("Dropping %d duplicate product IDs", int(duplicate.sum()))
        out = out.loc[~duplicate].copy()
    return out
