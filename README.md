# Product Data Search

AI-powered pharmaceutical product discovery over the Hugging Face dataset `revooda/indian-pharma-data`.

The application keeps complete source records in SQLite and uses persistent ChromaDB only for vectors and the minimal `record_id` retrieval key. Gemini creates document and query embeddings with `gemini-embedding-001`. Product attributes are not generated: available source values are preserved, and `brand_name` is exposed as the normalized product label because this dataset does not provide a separate `product_name` field.

## Verified dataset schema

The Hugging Face dataset viewer currently reports one `train` split with **253,973 rows** and these observed fields: `product_id`, `brand_name`, `manufacturer`, `price_inr`, `is_discontinued`, `dosage_form`, `pack_size`, `pack_unit`, `num_active_ingredients`, `primary_ingredient`, `primary_strength`, `active_ingredients`, `therapeutic_class`, `packaging_raw`, and `manufacturer_raw`. The ingestion code validates its required identity/search fields, logs the observed columns and row count, supports missing optional columns, sanitizes null/non-finite values, retains every source field in `source_json`, and parses structured ingredient values where possible. The ID from the source dataset becomes the stable key `hf:<product_id>`.

Google currently lists `gemini-embedding-001` as a stable text embedding model. It supports `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY`, and 768 dimensions are an available recommended output size. The Python SDK used here is `google-genai` (`from google import genai`). Check [Google's embedding model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-001) and [embedding guide](https://ai.google.dev/gemini-api/docs/embeddings) for current service limits and model lifecycle.

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.11 or newer (64-bit recommended)
- Internet access for Hugging Face download and Gemini embedding/indexing
- A Gemini API key for vector indexing and semantic search; get one from [Google AI Studio](https://aistudio.google.com/app/apikey)
- About 73 MB for the source dataset plus SQLite and vector index space. Full initial embedding uses Gemini API quota/billing and can take time.

## Windows installation

Open PowerShell in this project directory:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Set `GEMINI_API_KEY` in `.env` to your own key. Keep `.env` private; it is excluded by `.gitignore`. The embedding model, output dimension, request pacing, and batch size have working defaults in the example file. `gemini-embedding-001` is configured by default.

## Download and load the dataset

Run from the project root with the virtual environment active:

```powershell
python scripts/01_download_dataset.py
python scripts/02_clean_and_load.py
```

Step 1 downloads the dataset through Hugging Face `datasets`, prints the actual observed column names and row count, and saves a Parquet snapshot under `data/raw/`. Step 2 normalizes it, writes a processed JSON Lines snapshot to `data/processed/`, initializes `database/products.sqlite3`, and idempotently upserts all records. To refresh from the current HF revision, rerun both steps. The scripts do not fabricate unavailable fields.

## Build the vector index

```powershell
python scripts/03_build_vector_index.py
```

This embeds product text in batches and stores persistent vectors in `chroma_db/`. Progress and failures are logged. Chroma IDs are the SQLite record IDs; metadata stores the ID and a content hash to detect source changes. The script skips unchanged IDs and re-embeds changed/new records, so rerunning resumes after an interruption. Re-run after refreshing SQLite to update the index. To verify a small subset first, use `python scripts/03_build_vector_index.py --limit 20`.

The implementation uses modest request pacing and exponential retry for transient API errors. Invalid credentials/model requests fail promptly. On quota exhaustion, indexing stops with the failed batch logged; after quota availability returns, rerun to continue. Adjust `GEMINI_REQUESTS_PER_MINUTE` and `EMBEDDING_BATCH_SIZE` to fit your account's current limits.

## Run the application

```powershell
streamlit run app.py
```

The app offers natural language and attribute search, exact product ID lookup, manufacturer/dosage/category/ingredient filters, configurable top-K, relevance explanations, full source-record details, similar product discovery, and SQLite/vector index counts. Natural-language search and similar-product recommendations use Gemini vectors. Exact IDs and text/attribute matches can also be found through SQLite. For semantic fallback, make sure the index is built and `.env` contains a valid key.

## Search and data checks

Check SQLite initialization and exact ID lookup without making a Gemini request:

```powershell
python scripts/04_test_search.py 101
```

Test a semantic query after indexing:

```powershell
python scripts/04_test_search.py "Medicines containing paracetamol" --semantic
```

Search results are product discovery results based on source fields and textual similarity. They are not medical advice, treatment recommendations, or a verification of product safety, availability, or regulatory status. Confirm pharmaceutical decisions with appropriate qualified sources.

## Storage layout

- `data/raw/indian_pharma_data.parquet`: downloaded source snapshot
- `data/processed/indian_pharma_data_clean.jsonl`: normalized snapshot retaining source attributes
- `database/products.sqlite3`: complete product records and searchable columns; source of truth
- `chroma_db/`: persistent semantic vectors and retrieval IDs

Runtime data and credentials are intentionally excluded from version control.

## Troubleshooting

- **`GEMINI_API_KEY is missing`**: make sure `.env` exists in the project root and contains `GEMINI_API_KEY=...`; restart Streamlit after editing.
- **401/403 from Gemini**: check that the key is valid and the Gemini API is enabled for its project.
- **429/quota or rate limit**: wait for quota reset, lower requests per minute or batch size, then rerun indexing. Existing embeddings are kept.
- **Dataset download/network error**: retry `01_download_dataset.py`; Hugging Face may be temporarily unavailable. The script prints the actual schema when successful.
- **Missing raw file**: run the download step before clean/load.
- **No semantic hits**: check the sidebar vector count and run `03_build_vector_index.py`.
- **Chroma dimension mismatch**: do not change embedding dimensions or model against an existing collection. Use a fresh collection name in `.env` and rebuild, or remove the old Chroma collection only if you intend to discard its embeddings.
- **SQLite refresh**: rerun download and clean/load, then rebuild indexing. Existing records are updated in SQLite and new IDs are indexed idempotently.

## Dependency notes

`requirements.txt` constrains the current major ranges for Streamlit, Hugging Face Datasets, Pandas, PyArrow, ChromaDB, `google-genai`, python-dotenv, and tqdm. Install into Python 3.11+; if deploying, pin the resolved versions from your tested environment for repeatable builds.
