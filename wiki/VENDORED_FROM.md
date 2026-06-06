# Vendored from llm-wiki-v3

This `wiki/` is rl-betfair's **own in-repo copy** of the operator's `llm-wiki-v3`
knowledge engine — copied as a working machine, started with an **empty vault**.
It is self-contained: the knowledge versions with the code, no cross-repo dependency.

| | |
|---|---|
| **Source repo** | `../llm-wiki-v3` (operator's repo, sibling checkout) |
| **Source commit** | `7873a29feb2582a9afeb34087be9fddac986aadc` |
| **Copied on** | 2026-06-06 (initial copy `08145d5`); synced to `11cf011` 2026-06-06 |
| **Design / decision** | `plans/memory-improvements/{purpose,current_state,design}.md`, `_kickoff.md` |

## Sync history
- **2026-06-07 → `7873a29`** (pre-commit quality-gate hook + `gate` command). Ported the read-only
  **`gate`** command into `wiki/scripts/wiki_tool.py` (validate + connectivity + claims-lint + coverage;
  exit 1 on any ERROR — the gates even outside `finalize-ingest`). Installed an **adapted repo-root
  `.githooks/pre-commit`** (rl-betfair, not `wiki/.githooks`): it greps `wiki/(shared|personal)/` +
  `wiki/**/*.claims.jsonl` so only wiki-note commits run the gate (code/training commits are
  untouched), and calls `wiki/scripts/wiki_tool.py gate`. Activated with `git config core.hooksPath
  .githooks` (local, per-clone). Did NOT port upstream's `cmd_init` hook auto-install (it assumes
  wiki == repo root); skipped the upstream test. Override a flagged commit with `git commit --no-verify`.
- **2026-06-07 — checked `5ad25e4`, no port.** Upstream `11cf011..5ad25e4` was vault *content*
  only (an ingest + relocating their source docs into `inbox/pending/Files`); the engine, skills,
  schema contracts, templates, MCP and extractors are byte-identical to `11cf011`. The inbox-drop
  workflow fix doesn't apply to rl-betfair's repo-scan (references-in-place) model. Engine verified
  current through `5ad25e4`.
- **2026-06-06 → `11cf011`** (anti-light-note enforcement). Ported `scripts/wiki_tool.py`
  (new `coverage` ERRORs — **under-extraction** vs the source's real size, and **claimless**
  substantive notes — now counted in the finalize error gate; `finalize-ingest` **strict by
  default**, `--no-strict` to override) + the `batch`/`curate`/`extract` skill updates. Skipped the
  upstream *content* reset (their `Schema/*.jsonl` corpus). Re-applied the local `git_changes`/
  `_repo_prefix` subdir patch on the new `wiki_tool.py`. Local-only files (`scan_repo.py`,
  `tag_superseded.py`, the rl-betfair AGENTS/skills edits, the `superseded` tag) preserved.
- **2026-06-06 → `08145d5`** initial vendor (empty vault).

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
