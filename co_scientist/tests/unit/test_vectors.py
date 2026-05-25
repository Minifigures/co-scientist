"""Tests for the FAISS store. Embedder is network-bound; we feed fake vectors."""

from __future__ import annotations

import numpy as np
import pytest

from co_scientist.vectors.store import FaissStore


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype("float32")
    return v / np.linalg.norm(v)


@pytest.mark.asyncio
async def test_faiss_store_add_search_persist(tmp_cfg) -> None:
    store = FaissStore(tmp_cfg, "ses_v", dim=8)
    await store.load_or_create()
    assert store.n == 0

    o1 = await store.add("hyp_1", _vec(1))
    o2 = await store.add("hyp_2", _vec(2))
    assert (o1, o2) == (0, 1)
    assert store.n == 2

    # k-NN should find itself first
    results = await store.search(_vec(1), k=2)
    assert results[0][0] == "hyp_1"
    assert results[0][1] == pytest.approx(1.0, abs=1e-3)

    # cosine matrix is 2x2 with 1s on diagonal
    m = await store.cosine_matrix()
    assert m.shape == (2, 2)
    assert m[0, 0] == pytest.approx(1.0, abs=1e-3)

    # Persist, then re-open
    await store.save()

    store2 = FaissStore(tmp_cfg, "ses_v", dim=8)
    await store2.load_or_create()
    assert store2.n == 2
    assert store2.hypothesis_at(0) == "hyp_1"
    assert store2.hypothesis_at(1) == "hyp_2"


@pytest.mark.asyncio
async def test_faiss_offset_lookup(tmp_cfg) -> None:
    store = FaissStore(tmp_cfg, "ses_v2", dim=4)
    await store.load_or_create()
    await store.add("a", _vec(1, 4))
    await store.add("b", _vec(2, 4))
    assert store.offset_of("a") == 0
    assert store.offset_of("b") == 1
    assert store.offset_of("missing") is None
