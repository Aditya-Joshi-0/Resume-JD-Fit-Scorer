import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Union
import streamlit as st
from core.config import Settings

settings = Settings.from_env()
MODEL_NAME = settings.embedding_model


@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    """Load and cache the embedding model across Streamlit reruns."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Embed a list of strings.
    Returns a 2D numpy array of shape (n_texts, embedding_dim).
    """
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Pre-normalise for fast cosine via dot product
        show_progress_bar=False,
    )
    return embeddings


def embed_single(text: str, model: SentenceTransformer) -> np.ndarray:
    """Embed a single string. Returns 1D array."""
    return embed_texts([text], model)[0]
