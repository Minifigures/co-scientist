"""Tests for model routing + downgrade chain + cost estimation."""

from __future__ import annotations

from co_scientist.config import Config
from co_scientist.llm.routing import (
    NEVER_DEGRADE,
    estimate_cost_usd,
    route,
    thinking_budget_for,
)


def test_default_routes_use_opus_for_heavy_modes() -> None:
    cfg = Config()
    assert route(cfg, "generation", "literature").model == cfg.models.generation
    assert route(cfg, "reflection", "verification").model == cfg.models.reflection
    assert route(cfg, "metareview", "final").model == cfg.models.metareview_final


def test_thinking_only_on_opus() -> None:
    cfg = Config()
    r = route(cfg, "reflection", "verification")
    if r.model.startswith("claude-opus"):
        assert r.thinking_tokens == cfg.thinking.reflection_verification
    else:
        assert r.thinking_tokens == 0


def test_degrade_walks_chain_once() -> None:
    cfg = Config()
    r1 = route(cfg, "generation", "literature", degraded=False)
    r2 = route(cfg, "generation", "literature", degraded=True)
    # Should be different unless never-degrade
    assert "generation.literature" not in NEVER_DEGRADE
    assert r1.model != r2.model


def test_never_degrade_modes_stay_put() -> None:
    cfg = Config()
    r1 = route(cfg, "reflection", "verification", degraded=False)
    r2 = route(cfg, "reflection", "verification", degraded=True)
    assert r1.model == r2.model


def test_thinking_budget_lookup() -> None:
    cfg = Config()
    assert thinking_budget_for(cfg, "reflection.verification") == cfg.thinking.reflection_verification
    assert thinking_budget_for(cfg, "ranking.pairwise") == cfg.thinking.ranking_pairwise
    assert thinking_budget_for(cfg, "made_up.mode") == 0


def test_cache_reads_are_cheaper_than_uncached_input() -> None:
    """Same total 10k-token context: all-uncached is more expensive than
    mostly-cached (2k uncached + 8k cache reads)."""
    all_uncached = estimate_cost_usd(
        model="claude-opus-4-7", input_tokens=10_000, output_tokens=1_000
    )
    mostly_cached = estimate_cost_usd(
        model="claude-opus-4-7",
        input_tokens=2_000, output_tokens=1_000,
        cache_read=8_000, cache_write=0,
    )
    assert mostly_cached < all_uncached
