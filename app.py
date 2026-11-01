"""Streamlit UI for Product Data Search."""
from __future__ import annotations
import logging
import pandas as pd
import streamlit as st
from src.config import DB_PATH
from src.db import stats as db_stats, filter_options
from src.vector_store import collection_stats
from src.search import search_products, similar_products
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Product Data Search", page_icon="💊", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1400px}.hero{padding:1.35rem 1.6rem;border-radius:16px;background:linear-gradient(120deg,#102b46,#146c75);color:white;margin-bottom:1.2rem}
.hero h1{margin:0;color:white}.hero p{margin:.4rem 0 0;color:#d5eef0;font-size:1.05rem}div[data-testid="stMetric"]{background:#f4f7fa;padding:12px 16px;border-radius:12px}
</style><div class="hero"><h1>Product Data Search</h1><p>AI-powered pharmaceutical product discovery</p></div>""", unsafe_allow_html=True)
stats = db_stats()
vstats = collection_stats()
with st.sidebar:
    st.header("Search settings")
    top_k = st.slider("Results to show", 1, 30, 10)
    filters = {}
    for key, label in (("manufacturer", "Manufacturer"), ("dosage_form", "Dosage form"), ("therapeutic_class", "Therapeutic class"), ("primary_ingredient", "Active ingredient")):
        try:
            selected = st.multiselect(label, filter_options(key), key=f"filter_{key}")
            if selected:
                filters[key] = selected
        except Exception:
            logging.exception("Unable to load filter options for %s", key)
    st.divider()
    st.subheader("Dataset and index")
    st.metric("Products in SQLite", f"{stats.get('count', 0):,}")
    st.metric("Vector embeddings", f"{vstats.get('count', 0):,}")
    st.caption("SQLite stores complete source records. ChromaDB stores vectors and retrieval IDs.")
if not DB_PATH.exists() or stats.get("count", 0) == 0:
    st.info("No product data loaded yet. Run the dataset download and SQLite load scripts from the README.")
if "results" not in st.session_state:
    st.session_state.results = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
with st.form("search_form"):
    query = st.text_input("Search pharmaceutical products", value=st.session_state.last_query, placeholder="e.g. medicines containing paracetamol, Cipla, 500 mg tablets")
    col_search, col_clear = st.columns(2)
    submitted = col_search.form_submit_button("Search", type="primary", use_container_width=True)
    clear = col_clear.form_submit_button("Clear search", use_container_width=True)
if clear:
    st.session_state.results = []
    st.session_state.last_query = ""
    st.rerun()
if submitted:
    if not query.strip():
        st.warning("Enter a product name, ingredient, manufacturer, product ID, or natural-language query.")
    else:
        st.session_state.last_query = query.strip()
        with st.spinner("Searching product records…"):
            try:
                st.session_state.results = search_products(query, top_k=top_k, filters=filters)
            except Exception as exc:
                st.error(f"Search could not be completed: {exc}")
                st.session_state.results = []
results = st.session_state.results
if results:
    st.subheader(f"Results for “{st.session_state.last_query}”")
    table = []
    for item in results:
        p = item["product"]
        table.append({"Product ID": p.get("product_id"), "Product": p.get("brand_name"), "Active ingredient": p.get("primary_ingredient"), "Strength": p.get("primary_strength"), "Form": p.get("dosage_form"), "Manufacturer": p.get("manufacturer"), "Therapeutic class": p.get("therapeutic_class"), "Price (INR)": p.get("price_inr"), "Relevance": f"{item['score']:.0%}", "Match reason": item["explanation"]})
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    labels = [f"{r['product'].get('brand_name')} · ID {r['product'].get('product_id')}" for r in results]
    chosen = st.selectbox("Product details", range(len(results)), format_func=lambda i: labels[i])
    selected = results[chosen]
    with st.expander("Selected product record", expanded=True):
        st.caption(f"Match explanation: {selected['explanation']} · Relevance: {selected['score']:.1%}")
        st.json(selected["product"], expanded=True)
    if vstats.get("count", 0):
        with st.expander("Similar products"):
            try:
                with st.spinner("Finding similar products…"):
                    similar = similar_products(selected["product"], top_k=5)
                if similar:
                    st.dataframe(pd.DataFrame([{"Product": s["product"].get("brand_name"), "Ingredient": s["product"].get("primary_ingredient"), "Strength": s["product"].get("primary_strength"), "Manufacturer": s["product"].get("manufacturer"), "Similarity": f"{s['score']:.0%}"} for s in similar]), use_container_width=True, hide_index=True)
                else:
                    st.caption("No similar products found.")
            except Exception as exc:
                st.error(f"Similar product search failed: {exc}")
elif st.session_state.last_query:
    st.info("No matching products found. Try a shorter query or remove a filter.")
else:
    st.markdown("Try **Medicines containing paracetamol**, **Products manufactured by Cipla**, or **500 mg pain relief tablets**.")
