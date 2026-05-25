"""Embedding clients.

Voyage primary (`voyage-3-large` by default), OpenAI fallback.
All clients return `np.ndarray` of shape (n, dim), L2-normalized so cosine
similarity == inner product (we use FAISS `IndexFlatIP`).
"""

from __future__ import annotations

import asyncio
import os
from typing import Protocol

import numpy as np

from ..config import Config


class Embedder(Protocol):
    model: str
    dim: int

    async def embed(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (v / norms).astype("float32")


# --------------------------------------------------------------------------- #
# Voyage


class VoyageEmbedder:
    def __init__(self, cfg: Config) -> None:
        self.model = cfg.embeddings.model
        self.dim = cfg.embeddings.dim
        self._cfg = cfg

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        api_key = self._cfg.secrets.VOYAGE_API_KEY or os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set; cannot use VoyageEmbedder")
        # voyageai is sync; offload to a thread to keep the loop responsive.
        import voyageai

        client = voyageai.Client(api_key=api_key)

        def _call() -> list[list[float]]:
            res = client.embed(texts, model=self.model, input_type="document")
            return res.embeddings

        vecs = await asyncio.to_thread(_call)
        arr = np.asarray(vecs, dtype="float32")
        return _l2_normalize(arr)


# --------------------------------------------------------------------------- #
# OpenAI fallback


class OpenAIEmbedder:
    def __init__(self, cfg: Config) -> None:
        # Override dim if running OpenAI's text-embedding-3-small (1536) or -large (3072).
        self.model = cfg.embeddings.model if cfg.embeddings.provider == "openai" else "text-embedding-3-small"
        # text-embedding-3-small native dim is 1536, but the API supports a `dimensions` parameter
        # to shrink. We keep the configured dim and pass it explicitly.
        self.dim = cfg.embeddings.dim
        self._cfg = cfg

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        api_key = self._cfg.secrets.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; cannot use OpenAIEmbedder")

        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install openai (or co-scientist[openai]) to use the fallback") from e

        client = openai.AsyncOpenAI(api_key=api_key)
        # OpenAI supports batches of up to ~2048 entries; chunk conservatively.
        batches = [texts[i : i + 256] for i in range(0, len(texts), 256)]
        out: list[list[float]] = []
        for batch in batches:
            resp = await client.embeddings.create(
                model=self.model, input=batch, dimensions=self.dim
            )
            out.extend(d.embedding for d in resp.data)
        return _l2_normalize(np.asarray(out, dtype="float32"))


# --------------------------------------------------------------------------- #
# Resolver


def make_embedder(cfg: Config) -> Embedder:
    provider = cfg.embeddings.provider.lower()
    if provider == "voyage":
        return VoyageEmbedder(cfg)
    if provider == "openai":
        return OpenAIEmbedder(cfg)
    raise ValueError(f"unknown embeddings provider: {provider}")
