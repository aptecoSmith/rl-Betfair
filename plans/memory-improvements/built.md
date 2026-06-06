# Built — the in-repo knowledge wiki (2026-06-06)

Prong 2 (the deep, pull-on-demand layer) from `design.md` is **built**. `_kickoff.md`
executed end to end. Four revertible commits on `pbt-gpu-forward`:

| commit | what |
|---|---|
| `3ff8f5b` | Vendored `llm-wiki-v3` → `wiki/` (empty vault), adapted AGENTS/CLAUDE/README, gitignore, provenance, subdir patch |
| `23691ec` | Wired the read-only MCP server (`.mcp.json` + `.claude/settings.json`) |
| `c8ae3e7` | Proof-first ingest: the reward-shaping supersession cluster |
| `2d06ded` | `scan_repo.py` repo-scan ingester + registered/queued the markdown corpus |
| `c5de08b` | Refiltered the scan to **knowledge-core** (185 files; process scaffolding dropped) |

## What exists now
- **`wiki/`** — a self-contained, working v3 (engine + MCP + skills + schema), zero
  vendored notes. Operating rules: `wiki/AGENTS.md`. Provenance + keep/drop list +
  the one local patch: `wiki/VENDORED_FROM.md`.
- **Query surface (CLI, LLM-off):** `python wiki/scripts/wiki_tool.py {search,query,view}`.
  **MCP surface (read-only):** server `rl-betfair-wiki` in `.mcp.json` — exposes
  `search_pages` / `query_wiki` / `render_view` etc. **Restart Claude Code** to load it
  (`mcp` is installed in `.venv`).
- **Proof-first cluster** (8 notes, 14 claims, 3 supersession chains, 3 decisions):
  equal-exposure→equal-profit sizing; CLOSE_SIGNAL_BONUS 1.0→0.5→0; aggregate→per-pair
  naked penalty + the raw/shaped anchor + synthesis hub. Demonstrated working:
  `query unverified-claims`, `query supersedes`, `view supersession-timeline`,
  `view dialog-trail`; 1 claim cross-source-verified.
- **Repo-scan:** `python wiki/scripts/scan_repo.py` registered + queued **185 knowledge files**
  (`purpose`/`lessons_learnt`/`findings`/`design` under `plans/` + all of `docs/` + the named logs
  EXPERIMENTS/EXPLORATIONS/CLAUDE/genes_census; process scaffolding —
  master_todo/progress/session_prompt(s)/hard_constraints — deliberately excluded).
  `python wiki/scripts/batch.py --name repo-md status` shows 3 done / 182 pending.

## Decisions taken at build time (deviations worth knowing)
- **MCP lives in `.mcp.json`, not `settings.json`.** Current Claude Code reads MCP
  servers from `.mcp.json` (project) / `~/.claude.json` (user) — not `settings.json`
  (which got the read-only allowlist + server pre-approval). The kickoff said
  settings.json; this is the corrected mechanism.
- **Absolute machine paths.** Claude Code MCP config has no project-relative/`cwd`
  support, so `.mcp.json` uses absolute paths (`.venv` python + `PYTHONPATH=wiki`);
  source locations in `Schema/sources.jsonl` are absolute too (forward-slash, the
  registry is machine-keyed by design + exempt from the audit path-check). Adjust per
  machine or move MCP to `~/.claude.json` if paths differ.
- **One engine patch** (`wiki/scripts/wiki_tool.py::git_changes` + `_repo_prefix`): makes
  `finalize`'s junk-guard + `git add` work with the wiki as a *subdir* (strips the
  `git rev-parse --show-prefix`). No-op when wiki == repo root. Re-apply on any re-copy.

## What's deferred (not this session)
- **Compiling the 182 pending sources** into notes — the resumable `batch` skill's job
  (`python wiki/scripts/batch.py --name repo-md next` → ingest → finalize → done), one source
  at a time (anti-shortcut: a note per concept/finding/decision). Multi-session effort; the
  proof-first cluster validated the quality bar.
- **Ingesting `.py` code** (markdown knowledge first).
- **A maintenance cadence** (`audit`/`maintain`/`dream`) once the corpus lands.
- The `.runtime/batch/repo-md.json` ledger is gitignored (machine-local progress).
