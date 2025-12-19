"""Embedding generation supporting both local models and Voyage AI API."""

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import time

from .config import (
    EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_DIMENSIONS,
    VOYAGE_API_KEY,
    VOYAGE_MODEL,
    VOYAGE_DIMENSIONS,
    VOYAGE_MAX_BATCH,
    VOYAGE_MAX_WORKERS,
    QUERY_PREFIX,
    DOCUMENT_PREFIX,
)


class VoyageEmbedder:
    """Generate embeddings using Voyage AI API with parallel requests."""

    API_URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(
        self,
        api_key: str = VOYAGE_API_KEY,
        model: str = VOYAGE_MODEL,
        dimensions: int = VOYAGE_DIMENSIONS,
        max_workers: int = VOYAGE_MAX_WORKERS,
        max_batch: int = VOYAGE_MAX_BATCH,
    ):
        self.api_key = api_key
        self.model = model
        self._dimensions = dimensions
        self.max_workers = max_workers
        self.max_batch = max_batch
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _embed_batch(
        self, texts: list[str], input_type: str = "document", max_retries: int = 3
    ) -> list[list[float]]:
        """Embed a single batch of texts with retry logic.

        Args:
            texts: List of texts to embed
            input_type: 'document' or 'query'
            max_retries: Number of retries on failure

        Returns:
            List of embedding vectors
        """
        payload = {
            "input": texts,
            "model": self.model,
            "input_type": input_type,
            "output_dimension": self._dimensions,
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.API_URL, headers=self.headers, json=payload, timeout=120
                )

                if response.status_code == 400:
                    # Bad request - might be too many tokens, try smaller batch
                    error_detail = response.json().get("detail", "Unknown error")
                    raise ValueError(f"Voyage API 400 error: {error_detail}")

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

                data = response.json()
                # Sort by index to maintain order
                embeddings = sorted(data["data"], key=lambda x: x["index"])
                return [e["embedding"] for e in embeddings]

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue

        raise last_error or Exception("Failed to embed batch after retries")

    def _embed_batch_adaptive(
        self, texts: list[str], input_type: str = "document", current_size: int = None
    ) -> list[list[float]]:
        """Embed a batch with adaptive sizing - reduces batch size on failure.

        Fallback sequence: 100 → 50 → 32 → 16 → 8 → 4 → 2 → 1
        """
        if current_size is None:
            current_size = len(texts)

        # Process in chunks of current_size
        if len(texts) <= current_size:
            try:
                return self._embed_batch(texts, input_type)
            except ValueError as e:
                # Batch too large - try smaller size
                if current_size <= 1:
                    raise  # Can't reduce further

                # Fallback sizes: halve, but floor to nice numbers
                next_size = max(1, current_size // 2)
                if current_size > 50:
                    next_size = 50
                elif current_size > 32:
                    next_size = 32
                elif current_size > 16:
                    next_size = 16

                return self._embed_batch_adaptive(texts, input_type, next_size)

        # Split into chunks of current_size and process each
        results = []
        for i in range(0, len(texts), current_size):
            batch = texts[i:i + current_size]
            batch_results = self._embed_batch_adaptive(batch, input_type, current_size)
            results.extend(batch_results)
        return results

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = None,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Generate embeddings for documents using parallel API calls.

        Uses adaptive batch sizing - if a batch fails due to token limits,
        it automatically splits into smaller batches.

        Args:
            texts: List of texts to embed
            batch_size: Texts per API call (default: max_batch)
            show_progress: Print progress updates

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        batch_size = batch_size or self.max_batch
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        if show_progress:
            print(f"Embedding {len(texts)} texts in {len(batches)} batches...")

        all_embeddings = [None] * len(batches)
        start_time = time.time()

        # Process batches with adaptive sizing (sequential for better error handling)
        for idx, batch in enumerate(batches):
            try:
                all_embeddings[idx] = self._embed_batch_adaptive(batch, "document")
            except Exception as e:
                raise RuntimeError(f"Failed to embed batch {idx + 1}/{len(batches)}: {e}")

            if show_progress:
                elapsed = time.time() - start_time
                print(f"  Batch {idx + 1}/{len(batches)} done ({elapsed:.1f}s)")

        # Flatten batched results
        result = []
        for batch_embeddings in all_embeddings:
            result.extend(batch_embeddings)

        if show_progress:
            total_time = time.time() - start_time
            print(f"Completed {len(texts)} embeddings in {total_time:.1f}s")

        return result

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        result = self._embed_batch([query], input_type="query")
        return result[0]

    def embed_for_similarity(self, text: str) -> list[float]:
        """Generate embedding for finding similar chunks."""
        result = self._embed_batch([text], input_type="document")
        return result[0]

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        return self._dimensions

    def get_model_info(self) -> dict:
        """Get information about the embedding configuration."""
        return {
            "provider": "voyage",
            "model_name": self.model,
            "dimensions": self._dimensions,
            "max_batch": self.max_batch,
            "max_workers": self.max_workers,
        }


class LocalEmbedder:
    """Generate embeddings using local sentence-transformers model."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None

    def load_model(self) -> None:
        """Load the embedding model. Call this once at startup."""
        if self.model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
            self.model.max_seq_length = 8192

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Generate embeddings for documents."""
        if self.model is None:
            self.load_model()

        prefixed_texts = [f"{DOCUMENT_PREFIX}{text}" for text in texts]

        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        if self.model is None:
            self.load_model()

        prefixed_query = f"{QUERY_PREFIX}{query}"

        embedding = self.model.encode(
            prefixed_query, normalize_embeddings=True, convert_to_numpy=True
        )

        return embedding.tolist()

    def embed_for_similarity(self, text: str) -> list[float]:
        """Generate embedding for finding similar chunks."""
        if self.model is None:
            self.load_model()

        prefixed_text = f"{DOCUMENT_PREFIX}{text}"

        embedding = self.model.encode(
            prefixed_text, normalize_embeddings=True, convert_to_numpy=True
        )

        return embedding.tolist()

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        return LOCAL_EMBEDDING_DIMENSIONS

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "provider": "local",
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "max_seq_length": self.model.max_seq_length if self.model else 8192,
            "loaded": self.model is not None,
        }


# Type alias for embedder
Embedder = VoyageEmbedder | LocalEmbedder

# Global embedder instance
_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Get or create the global embedder instance based on config."""
    global _embedder
    if _embedder is None:
        if EMBEDDING_PROVIDER == "voyage":
            _embedder = VoyageEmbedder()
        else:
            _embedder = LocalEmbedder()
    return _embedder
