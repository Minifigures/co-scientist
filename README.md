# AI Co-Scientist

A multi-agent system for tournament-style scientific hypothesis generation, ranking, and synthesis. Built on the architecture described in [`reference/`](reference/) (Google's Co-Scientist), implemented in Python on top of the raw Anthropic SDK.

The system takes a natural-language research goal, runs six specialized LLM agents in a coordinated loop, and produces a *Research Overview* of the top-ranked hypotheses:

- **Generation** — proposes hypotheses via literature review and simulated scientific debate
- **Reflection** — reviews hypotheses for novelty, correctness, and testability; deep-verifies assumptions
- **Ranking** — runs an Elo tournament with simulated debates between hypotheses
- **Evolution** — combines, simplifies, and reimagines top-ranked hypotheses
- **Proximity** — embeds and clusters hypotheses to drive dedup and informative pairings
- **Meta-review** — synthesizes system-wide feedback and the final research overview

A **Supervisor** schedules agents via a durable task queue (SQLite-backed) with bounded concurrency. The full design is in [`/Users/kuan-linhuang/.claude/plans/based-on-these-txt-unified-pearl.md`](../../.claude/plans/based-on-these-txt-unified-pearl.md).

## Status

**Through M3 — first vertical slice runnable.** Shipped so far:

- **M0 — Skeleton.** Package structure, configuration loader, SQLite schema + migrations (12 tables incl. spans/events/elo_journal), ULID + deterministic-hash IDs, structlog JSONL logging, Typer CLI scaffold.
- **M1 — Storage, vectors, tools.** Repos for sessions/hypotheses/reviews/tasks/tournaments/transcripts/feedback/embeddings/events/spans; Voyage+OpenAI embedders; FAISS `IndexFlatIP` per-session store; built-in tools (`web_search`, `web_fetch`, `pubmed_search`, `arxiv_search`, `europe_pmc_search`); science-skills bridge that parses `SKILL.md` + shells out to scripts; `co-scientist tools list` lists everything per-agent.
- **M2 — Anthropic SDK layer.** All 14 prompt templates (verbatim from `reference/9` where they exist; synthesized where the reference is silent); Jinja2 loader; retry policy honoring Retry-After for 429/529; `TokenBudget` with per-agent shares and async-locked admission; model routing with never-degrade list (`reflection.verification`, `metareview.final`); `AnthropicClient` with 4-tier `cache_control` breakpoints, retry, transcript persistence, USD accounting; tool-loop driver with parallel cap and URL tracking for citation honesty; `UNTRUSTED_SOURCE` quoting for prompt-injection defense.
- **M3 — Generation + Reflection vertical slice.** `BaseAgent`; `GenerationAgent` (literature strategy) with `record_hypothesis` tool, dedup via FAISS, citation filter against tool-loop URLs; `ReflectionAgent` (full review) with `record_review` tool, evidence URL filter; minimal `Supervisor` that parses the goal, enqueues N Generation tasks, drains the queue with a bounded asyncio worker pool, runs one Reflection per new hypothesis, and writes a stub overview; `co-scientist run "goal"` end-to-end.

**45 unit tests passing.** No network calls in CI: Anthropic + embeddings + web tools are exercised via dedicated tests where possible and gated behind a manual smoke test otherwise.

### What's still ahead (per plan)
- **M4** — Ranking + Elo tournament (`AddToTournament`, `RunTournamentBatch`, pair selection, debate-vs-pairwise mode switching, `elo_journal` idempotency at the agent level).
- **M5** — Full Supervisor scheduling: `decide_next_steps`, follow-up rules, lease/heartbeat, crash recovery, dead-letter, pause/resume/abort.
- **M6** — Evolution + Proximity + Meta-review (system feedback periodic + final overview synthesis).
- **M7** — FastAPI + htmx + SSE web UI.
- **M8** — Safety classifier, citation verifier, spans/metrics, eval scaffolding.
- **M9** — Batch API for sub-decile tournament matches; cost estimator pre-flight; resume hardening.

## Install

```bash
# Recommended: Python 3.11–3.13 (FAISS wheel availability)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# fill in ANTHROPIC_API_KEY at minimum
```

## Initialize

```bash
co-scientist init
co-scientist list
```

`init` creates `data/` (artifacts, vectors, logs) and applies migrations to `data/co_scientist.db`.

## Configuration

Layered: [`config/default.toml`](config/default.toml) → `~/.co-scientist/config.toml` → `./co-scientist.toml` → `--config <path>`. Secrets come from environment only (see [`.env.example`](.env.example)).

## Repository layout

```
co_scientist/
  agents/       # supervisor + 6 specialized agents (M3+)
  llm/          # Anthropic client wrapper, tool loop, budgets, routing (M2)
  storage/      # SQLite schema, db connection, repos (M0/M1)
  tools/        # tool registry; web/search, science-skills, code exec (M1)
  vectors/      # embeddings + FAISS index (M1)
  orchestrator/ # task queue, worker pool, termination, event bus (M5)
  safety/       # injection quoting, classifier, citation verifier (M8)
  obs/          # spans, metrics (M8)
  web/          # FastAPI + htmx + SSE UI (M7)
  evals/        # per-agent + e2e + regression evals (M8)
  tests/        # unit, fixtures, smoke
config/
  default.toml
  prompts/      # Jinja2 templates per agent.mode (from reference/9)
reference/      # input materials (pseudocode, prompts, diagrams)
data/           # gitignored; runtime artifacts
vendor/         # gitignored; pinned clone of google-deepmind/science-skills
```

## License

Apache-2.0.
