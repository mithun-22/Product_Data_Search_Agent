"""Hybrid exact, lexical, and semantic product search."""
from __future__ import annotations

import re
from typing import Any

from . import db
from .gemini_service import GeminiEmbeddingService
from .vector_store import vector_search


def _record_id(record: dict) -> str:
    return str(record.get("record_id") or f"hf:{record.get('product_id')}")


def _matches_filters(record: dict, filters: dict[str, Any] | None) -> bool:
    filters = filters or {}
    aliases = {"manufacturer": "manufacturer", "dosage_form": "dosage_form", "therapeutic_class": "therapeutic_class", "primary_ingredient": "primary_ingredient"}
    for key, val in filters.items():
        if not val or key not in aliases:
            continue
        actual = str(record.get(aliases[key]) or "").casefold()
        if isinstance(val, (list, tuple, set)):
            if val and not any(str(v).casefold() == actual for v in val):
                return False
        elif actual != str(val).casefold():
            return False
    return True


def search_products(query: str, top_k: int = 10, filters: dict | None = None,
                    service: GeminiEmbeddingService | None = None) -> list[dict]:
    query = query.strip()
    if not query:
        return []
    if re.fullmatch(r"(?:hf:)?\d+", query):
        record = db.exact_id(query.removeprefix("hf:"))
        return ([{"product": record, "score": 1.0, "explanation": "Exact product ID match"}] if record and _matches_filters(record, filters) else [])

    # Broader initial retrieval allows the user-selected filters to be applied after SQLite fetch.
    exact = db.exact_text(query, limit=max(top_k * 12, 100))
    by_id: dict[str, dict] = {}
    for item in exact:
        rid = _record_id(item)
        if _matches_filters(item, filters):
            by_id[rid] = {"product": item, "score": 0.85, "explanation": "Product attribute or text match"}
    if len(by_id) < top_k:
        try:
            hits = vector_search(query, max(top_k * 8, 40), service)
            records = db.get_products([h["record_id"] for h in hits])
            record_map = {_record_id(p): p for p in records}
            for hit in hits:
                item = record_map.get(hit["record_id"])
                if not item or not _matches_filters(item, filters):
                    continue
                semantic = max(0.0, min(1.0, hit["score"]))
                existing = by_id.get(hit["record_id"])
                if existing:
                    existing["score"] = min(1.0, max(existing["score"], semantic + 0.08))
                    existing["explanation"] = "Attribute and semantic match"
                else:
                    by_id[hit["record_id"]] = {"product": item, "score": semantic, "explanation": "Semantic match to the product description"}
        except RuntimeError:
            if not by_id:
                raise
    return sorted(by_id.values(), key=lambda x: x["score"], reverse=True)[:max(1, min(top_k, 100))]


def similar_products(product: dict | str, top_k: int = 5, service: GeminiEmbeddingService | None = None) -> list[dict]:
    if isinstance(product, str):
        product = db.exact_id(product) or next(iter(db.exact_text(product, limit=1)), None)
    if not product:
        return []
    rid = _record_id(product)
    query = product.get("search_text") or " | ".join(str(product.get(k)) for k in ("brand_name", "primary_ingredient", "primary_strength", "dosage_form", "therapeutic_class") if product.get(k))
    hits = vector_search(query, top_k + 1, service)
    others = [h for h in hits if h["record_id"] != rid]
    records = db.get_products([h["record_id"] for h in others])
    mapped = {_record_id(r): r for r in records}
    return [{"product": mapped[h["record_id"]], "score": h["score"], "explanation": "Similar ingredient, product attributes, or therapeutic description"} for h in others if h["record_id"] in mapped][:top_k]
