"""IVARA Embedding Engine.

Provides offline-safe loading of SentenceTransformer embeddings from the local cache.
Ensures zero network latency, strict cache enforcement, and clear actionable errors
if the expected model is unavailable.
"""

import os
from typing import Optional, Any
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_model_instance: Optional[SentenceTransformer] = None


class EmbeddingModelUnavailableError(RuntimeError):
    """Raised when the requested embedding model is not available in local cache or cannot be loaded."""
    pass


def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    local_files_only: Optional[bool] = None,
) -> SentenceTransformer:
    """Load the sentence embedding model safely.
    
    In offline/isolated environments, loads strictly from the local cache.
    If the model is not available in cache, raises an actionable error rather than
    silently substituting another model or hanging on network calls.
    """
    if local_files_only is None:
        # Check if offline is explicitly disabled via IVARA_ALLOW_ONLINE_EMBEDDING=1
        allow_online = os.environ.get("IVARA_ALLOW_ONLINE_EMBEDDING", "0").lower() in ("1", "true")
        # Default to local cache first to prevent network stalls
        local_files_only = not allow_online or (
            os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        )

    try:
        # Prefer loading from local cache
        return SentenceTransformer(model_name, local_files_only=local_files_only)
    except Exception as e:
        # If local_files_only was attempted and online access is explicitly allowed, retry
        if local_files_only and os.environ.get("IVARA_ALLOW_ONLINE_EMBEDDING", "0").lower() in ("1", "true"):
            try:
                return SentenceTransformer(model_name, local_files_only=False)
            except Exception as online_err:
                raise EmbeddingModelUnavailableError(
                    f"Embedding model '{model_name}' could not be loaded from cache or network: {online_err}"
                ) from online_err

        # Produce a clear, actionable error without silent substitution
        raise EmbeddingModelUnavailableError(
            f"Embedding model '{model_name}' is not available in local cache. "
            f"In offline mode, IVARA requires '{model_name}' to be pre-cached in HuggingFace cache. "
            f"Model substitution is strictly prohibited to preserve deterministic embedding dimensions and retrieval parity. "
            f"Underlying error: {type(e).__name__}: {e}"
        ) from e


def get_model() -> SentenceTransformer:
    """Get or initialize the singleton embedding model."""
    global _model_instance
    if _model_instance is None:
        _model_instance = load_embedding_model()
    return _model_instance


def get_embedding(text):
    """Compute dense embedding vector for the provided text."""
    return get_model().encode(text)


def get_embeddings(texts: list[str], batch_size: int = 64) -> list[Any]:
    """Compute dense embedding vectors for a batch of texts efficiently."""
    if not texts:
        return []
    return get_model().encode(texts, batch_size=batch_size, show_progress_bar=False)


def __getattr__(name: str):
    """Maintain backward compatibility for callers importing `src.embedding.model`."""
    if name == "model":
        return get_model()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
