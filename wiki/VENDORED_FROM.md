# Vendored from llm-wiki-v3

This `wiki/` is rl-betfair's **own in-repo copy** of the operator's `llm-wiki-v3`
knowledge engine — copied as a working machine, started with an **empty vault**.
It is self-contained: the knowledge versions with the code, no cross-repo dependency.

| | |
|---|---|
| **Source repo** | `C:\Users\jsmit\source\repos\llm-wiki-v3` |
| **Source commit** | `08145d5ead5646cf3da255914b61cba125a916b6` |
| **Copied on** | 2026-06-06 |
| **Design / decision** | `plans/memory-improvements/{purpose,current_state,design}.md`, `_kickoff.md` |

## What was copied (the machine)
- `scripts/` — the deterministic engine (`wiki_tool.py` + siblings: `projection`,
  `claims`, `dialog`, `views`, `bootstrap`, `doctypes`, `intake`, `vectorstore`,
  `scrape`, `audit_public`, `dream`). Zero-dependency core (stdlib only).
- `extractors/` — optional dense-doc text extraction (PDF/DOCX/PPTX/XLSX); the
  engine imports `extract` from here for the `extract`/`ingest` skills.
- `mcp_server/` — the read-only MCP surface (`core.py` is stdlib; `server.py`
  needs the `mcp` package — see `requirements.txt`).
- `Schema/` — the contracts (`*.md`) + `tool-version.json`. All `*.jsonl`
  (`sources`, `ingest-log`, `catalog`, `open-questions`) **reset to empty**.
- `templates/`, `skills/` — note templates + skill playbooks, verbatim.
- `inbox/` + `Clippings/` — the external-doc drop folders, **emptied** to a clean
  `pending/` + `static/` structure (operator content not copied).
- `requirements.txt`, `AGENTS.md` (adapted for rl-betfair), `CLAUDE.md`
  (thin mirror), `README.md` (rewritten), `.gitattributes`.

## What was dropped (not copied)
- **Content** — all notes (`shared/`, `personal/`) and the operator's inbox docs:
  the vault starts EMPTY.
- **v3-development machinery** — `benchmarking/`, `comparison/`, `ci/`, `docs/`
  (build_docs / session-prompts / v2-vs-v3), `tests/` (engine dev tests).
- **Unused surfaces / ops** — the Docker stack (`Dockerfile`, `docker-compose.yml`,
  `.dockerignore`), `automation/` (inbox watcher — deferred), `start_*.bat`,
  `to-ingest/`, `inbox_personal/` (single shared cloud; `personal/` unused).
- **Local / derived** — `.git`, `.venv`, `.runtime/`, `.obsidian/`,
  `.wiki-local.json`, stray `concepts.md`.

## rl-betfair-specific patch (re-apply on any re-copy)
`scripts/wiki_tool.py` — `git_changes()` + new `_repo_prefix()`. Upstream assumes
the wiki **is** the git repo root; here it's a subdir of rl-betfair. `git status
--porcelain` reports repo-root-relative paths, so `finalize`'s junk-guard and
`git add` are made prefix-aware (strip `git rev-parse --show-prefix`, scope status
to the wiki subtree). The patch is a **no-op when wiki == repo root**
(byte-identical to upstream), so it's safe to keep across re-copies. `--push` is
never passed (rl-betfair pushes only on request, via `git push all`).

## Re-copying (only if a v3 engine improvement is ever wanted)
Low priority — the engine is stable. To refresh: re-copy the `scripts/`,
`extractors/`, `mcp_server/`, `templates/`, `skills/`, `Schema/*.md` machinery from
a newer `llm-wiki-v3` commit, **preserve this repo's vault** (`shared/`,
`Schema/*.jsonl`), and re-apply the `git_changes()` patch above. Update this file's
commit/date.
