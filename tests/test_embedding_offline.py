"""Tests for offline-safe embedding model loading and error handling."""

import os
import pytest
from src.embedding import (
    load_embedding_model,
    get_model,
    get_embedding,
    EmbeddingModelUnavailableError,
    DEFAULT_MODEL_NAME,
)
import src.embedding as embedding_mod


def test_embedding_model_loads_from_cache_offline():
    """Verify default model loads offline from local cache without network."""
    # Ensure offline flags
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    model = load_embedding_model(DEFAULT_MODEL_NAME, local_files_only=True)
    assert model is not None
    emb = model.encode("IVARA test embedding")
    assert len(emb) == 384


def test_get_embedding_and_singleton():
    """Verify get_embedding produces valid vectors and get_model returns the singleton."""
    emb = get_embedding("Deterministic offline embedding test")
    assert len(emb) == 384
    
    m1 = get_model()
    m2 = embedding_mod.model  # PEP 562 attribute access
    assert m1 is m2


def test_unavailable_model_raises_actionable_error():
    """Verify requesting an unavailable model raises actionable error without substitution."""
    with pytest.raises(EmbeddingModelUnavailableError) as exc_info:
        load_embedding_model("nonexistent-unreachable-model-9999", local_files_only=True)
    
    msg = str(exc_info.value)
    assert "nonexistent-unreachable-model-9999" in msg
    assert "not available in local cache" in msg
    assert "substitution is strictly prohibited" in msg

