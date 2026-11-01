"""Gemini embedding client with pacing, retries, and task-specific embeddings."""
from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque

from google import genai
from google.genai import types

from .config import GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, API_REQUESTS_PER_MINUTE

log = logging.getLogger(__name__)


class GeminiEmbeddingService:
    def __init__(self, api_key: str | None = None, model: str = EMBEDDING_MODEL, dimensions: int = EMBEDDING_DIMENSIONS,
                 requests_per_minute: int = API_REQUESTS_PER_MINUTE, client=None):
        key = api_key or GEMINI_API_KEY
        if not client and not key:
            raise RuntimeError("GEMINI_API_KEY is missing. Set it in the project .env file.")
        if model != "gemini-embedding-001":
            raise ValueError(f"Unsupported embedding model {model!r}; this implementation is validated for gemini-embedding-001")
        if not 128 <= dimensions <= 3072:
            raise ValueError("Embedding dimensions must be between 128 and 3072")
        self.model, self.dimensions = model, dimensions
        self.client = client or genai.Client(api_key=key)
        self.rpm = max(1, requests_per_minute)
        self._requests = deque()
        self._lock = threading.Lock()

    def _pace(self):
        with self._lock:
            now = time.monotonic()
            while self._requests and now - self._requests[0] >= 60:
                self._requests.popleft()
            if len(self._requests) >= self.rpm:
                delay = 60 - (now - self._requests[0])
                if delay > 0:
                    time.sleep(delay)
            self._requests.append(time.monotonic())

    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT", retries: int = 6) -> list[list[float]]:
        if not texts:
            return []
        for attempt in range(retries):
            self._pace()
            try:
                result = self.client.models.embed_content(
                    model=self.model, contents=texts,
                    config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dimensions),
                )
                vectors = [list(map(float, emb.values)) for emb in result.embeddings]
                if len(vectors) != len(texts) or any(len(v) != self.dimensions for v in vectors):
                    raise RuntimeError("Gemini returned an unexpected embedding count or dimension")
                return vectors
            except Exception as exc:
                # Do not retry authentication, malformed request, or invalid model failures.
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                message = str(exc).lower()
                permanent = status in (400, 401, 403, 404) or any(x in message for x in ("api key not valid", "permission denied", "not found", "invalid argument"))
                if permanent or attempt == retries - 1:
                    raise RuntimeError(f"Gemini embedding failed: {exc}") from exc
                wait = min(60, 2 ** attempt) + random.random()
                log.warning("Gemini embedding request failed (attempt %d/%d); retrying in %.1fs: %s", attempt + 1, retries, wait, exc)
                time.sleep(wait)
        raise RuntimeError("Gemini embedding retries exhausted")
