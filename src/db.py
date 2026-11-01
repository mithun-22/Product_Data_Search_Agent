"""SQLite persistence; original source JSON remains authoritative."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable
import pandas as pd

from .config import DB_PATH
from .serializer import dumps, loads


@contextmanager
def connect(path: Path | str = DB_PATH):
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize(path: Path | str = DB_PATH) -> None:
    with connect(path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS products (
            record_id TEXT PRIMARY KEY, product_id TEXT, product_name TEXT, brand_name TEXT,
            manufacturer TEXT, primary_ingredient TEXT, primary_strength TEXT, dosage_form TEXT,
            therapeutic_class TEXT, price_inr REAL, pack_size REAL, pack_unit TEXT,
            is_discontinued INTEGER, search_text TEXT NOT NULL, source_json TEXT NOT NULL
        )""")
        for name, col in (("idx_products_name", "product_name"), ("idx_products_brand", "brand_name"),
                          ("idx_products_ingredient", "primary_ingredient"), ("idx_products_manufacturer", "manufacturer"),
                          ("idx_products_id", "product_id")):
            conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON products({col})")


def load_products(frame: pd.DataFrame, path: Path | str = DB_PATH, chunk_size: int = 1000) -> int:
    initialize(path)
    cols = ["record_id", "product_id", "product_name", "brand_name", "manufacturer", "primary_ingredient",
            "primary_strength", "dosage_form", "therapeutic_class", "price_inr", "pack_size", "pack_unit",
            "is_discontinued", "search_text", "source_json"]
    rows = []
    for _, row in frame.iterrows():
        def val(k: str):
            v = row.get(k)
            return None if v is None else v
        rows.append((str(val("record_id")), str(val("product_id")) if val("product_id") is not None else None,
                     val("product_name"), val("brand_name"), val("manufacturer"), val("primary_ingredient"),
                     val("primary_strength"), val("dosage_form"), val("therapeutic_class"), val("price_inr"),
                     val("pack_size"), val("pack_unit"), None if val("is_discontinued") is None else int(bool(val("is_discontinued"))),
                     val("search_text") or "", val("source_json") or "{}"))
    sql = f"INSERT INTO products ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)}) ON CONFLICT(record_id) DO UPDATE SET " + ",".join(f"{c}=excluded.{c}" for c in cols[1:])
    with connect(path) as conn:
        for start in range(0, len(rows), chunk_size):
            conn.executemany(sql, rows[start:start + chunk_size])
    return len(rows)


def get_product(record_id: str, path: Path | str = DB_PATH) -> dict | None:
    with connect(path) as conn:
        row = conn.execute("SELECT source_json FROM products WHERE record_id=?", (str(record_id),)).fetchone()
    return loads(row["source_json"], {}) if row else None


def get_products(record_ids: Iterable[str], path: Path | str = DB_PATH) -> list[dict]:
    ids = list(dict.fromkeys(map(str, record_ids)))
    if not ids:
        return []
    with connect(path) as conn:
        rows = conn.execute(f"SELECT record_id,source_json FROM products WHERE record_id IN ({','.join('?' for _ in ids)})", ids).fetchall()
    mapped = {r["record_id"]: loads(r["source_json"], {}) for r in rows}
    return [mapped[i] for i in ids if i in mapped]


def exact_id(product_id: str, path: Path | str = DB_PATH) -> dict | None:
    with connect(path) as conn:
        row = conn.execute("SELECT source_json FROM products WHERE product_id=? OR record_id=? LIMIT 1", (str(product_id), f"hf:{product_id}")).fetchone()
    return loads(row["source_json"], {}) if row else None


def exact_text(query: str, limit: int = 100, path: Path | str = DB_PATH) -> list[dict]:
    pattern = f"%{query.strip()}%"
    with connect(path) as conn:
        rows = conn.execute("SELECT source_json FROM products WHERE product_name LIKE ? COLLATE NOCASE OR brand_name LIKE ? COLLATE NOCASE OR primary_ingredient LIKE ? COLLATE NOCASE OR manufacturer LIKE ? COLLATE NOCASE OR search_text LIKE ? COLLATE NOCASE LIMIT ?", (pattern, pattern, pattern, pattern, pattern, limit)).fetchall()
    return [loads(r[0], {}) for r in rows]


def stats(path: Path | str = DB_PATH) -> dict:
    if not Path(path).exists():
        return {"count": 0}
    with connect(path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        manufacturers = conn.execute("SELECT COUNT(DISTINCT manufacturer) FROM products WHERE manufacturer IS NOT NULL").fetchone()[0]
        categories = conn.execute("SELECT COUNT(DISTINCT therapeutic_class) FROM products WHERE therapeutic_class IS NOT NULL").fetchone()[0]
    return {"count": count, "manufacturers": manufacturers, "categories": categories}


def filter_options(column: str, path: Path | str = DB_PATH, limit: int = 300) -> list[str]:
    allowed = {"manufacturer", "dosage_form", "therapeutic_class", "primary_ingredient"}
    if column not in allowed:
        raise ValueError("Unsupported filter column")
    with connect(path) as conn:
        return [r[0] for r in conn.execute(f"SELECT DISTINCT {column} FROM products WHERE {column} IS NOT NULL ORDER BY {column} LIMIT ?", (limit,)).fetchall()]
