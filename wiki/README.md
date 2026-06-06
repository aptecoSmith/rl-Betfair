# rl-betfair knowledge wiki (in-repo `llm-wiki-v3`)

**What this is.** rl-betfair's own copy of the operator's `llm-wiki-v3` — an Engelbart/CoDIAK-grounded
knowledge engine — vendored into this repo as the **deep, pull-on-demand knowledge layer**. The
always-loaded surface (`../CLAUDE.md` + the `memory/` index) stays thin; the accreted domain depth
(claims + provenance, supersession chains, the decision/dialog record) lives here and is pulled on
demand — by the CLI below or the read-only MCP server. Why: `../plans/memory-improvements/`. What was
copied/dropped from upstream, and the one local patch: [`VENDORED_FROM.md`](VENDORED_FROM.md).

Operating rules (read first, every session): **[AGENTS.md](AGENTS.md)**.

## Running it — two surfaces, one engine

```bash
# 1. Standalone CLI (zero-dependency core — pure stdlib):
python scripts/wiki_tool.py doctor          # status
python scripts/wiki_tool.py project         # build the SQLite projection from the files
python scripts/wiki_tool.py query --list    # structured queries (claims, provenance, supersedes, …) — LLM off
python scripts/wiki_tool.py view --list     # generated views (supersession-timeline, dialog-trail, …)

# 2. MCP server (read-only; the same engine, exposed as tools — registered in ../.mcp.json):
python -m mcp_server                         # stdio
#   needs the `mcp` package:  pip install -r mcp_server/requirements.txt   (the core CLI needs nothing installed)
```
The projection (`.runtime/projection.db`) and views (`out/`) are **derived and gitignored** — delete
and rebuild from the files anytime. The same query returns identical results from CLI and MCP.

## Ingesting rl-betfair's knowledge — two entry points
- **Repo-scan (bulk):** `python scripts/scan_repo.py` walks `plans/**/*.md`, `docs/**/*.md`, and the
  root knowledge docs (`CLAUDE.md`, `genes_census.md`, `plans/EXPERIMENTS.md`, `plans/EXPLORATIONS.md`),
  registers each as a referenced source, and queues it for the `batch` skill (one source at a time;
  a note per concept/finding/decision, not per file). Markdown first; code is a later option.
- **Inbox drop (external ML docs):** drop a file into `inbox/pending/Files/`, a URL into
  `inbox/pending/urls.md`, or a web-clip into `Clippings/`, then run the `ingest` skill.

## The one-line thesis
Keep human-readable files as the *system of record*; add stable object IDs, claim-level provenance,
typed links, a captured reasoning record, and an **embedded queryable index projected from the files**
— so that *with the LLM switched off, the knowledge base is still structured, addressable, and
queryable* (the **dependency-inversion test**).

## Relationship to upstream
The engine is **vendored, not developed here** — bugfixes/improvements belong upstream in
`llm-wiki-v3` and are pulled in by re-copy (re-applying the documented subdir patch). See
[`VENDORED_FROM.md`](VENDORED_FROM.md) for the source commit and the keep/drop list.
