# CLAUDE.md

The operating rules for this wiki live in **[AGENTS.md](AGENTS.md)** — read it first, every session.
It is the single source of truth for both Claude Code and Codex. This file exists only so Claude Code
picks up the same contract.

**This is rl-betfair's in-repo copy of `llm-wiki-v3`** — the deep, pull-on-demand knowledge layer for
this repo (granular IDs, claim provenance, typed links, an embedded queryable projection, a dialog
record). Provenance + what was copied/dropped: [`VENDORED_FROM.md`](VENDORED_FROM.md). Why it exists:
`../plans/memory-improvements/`. The **binding invariants** live at the bottom of [AGENTS.md](AGENTS.md)
— read them before touching the engine. In short: files are the system of record; the projection DB is
derived/regenerable/gitignored; the zero-dependency core always works; deterministic code extracts
facts, the LLM only synthesizes/suggests; with the LLM off the wiki is still queryable (the
dependency-inversion test); capabilities run across two surfaces here — the standalone CLI and the
read-only MCP server.

Quick reminders (full detail in AGENTS.md):
- When the user says `open` / `start` (or greets you), run the session-start **opener**: report active
  cloud, counts, and anything pending, then ask what to ingest or answer. Don't ingest until confirmed.
- **One cloud: `shared/`** (rl-betfair is all "work"; `personal/` unused). Sources are **referenced,
  not copied** — for the repo-scan path the source IS the in-repo file.
- Two ingest entry points: **repo-scan** (`scripts/scan_repo.py` over `plans/**`, `docs/**`, root
  knowledge docs) and the **inbox** (external ML docs you drop in). One source at a time, read fully;
  large docs go through the anti-shortcut extraction engine.
- The graph is the product — every note reaches a hub; no orphans.
- Finalize every ingest with `python scripts/wiki_tool.py finalize-ingest`.
- **"Process everything" means finish the job** — clear the coverage floor on every heavy source
  before closing it. No "TBD" deferrals, no asking which to prioritise.
- Repeatable workflows live in `skills/<name>/SKILL.md` — `ingest, extract, query, maintain, audit, dream, batch, static, gdrive-ingest`. Open the matching one when a request fits (e.g. "dream" → `skills/dream/SKILL.md`).
