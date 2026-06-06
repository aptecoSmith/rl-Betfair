# Current state — what's done + the assets to build on (2026-06-06)

## 1. Prong 1 (curation) — DONE this session
- **CLAUDE.md slimmed 1634 → 421 lines** (commit `0c1ebe8` on `pbt-gpu-forward`).
  Kept every load-bearing invariant, "do not re-break" rule, regression-test
  pointer, and `plans/` pointer; moved dated change-sequences, worked examples,
  and per-section scoreboard-comparability prose into the cited plan folders;
  de-duplicated the byte-identity / comparability boilerplate into one
  cross-cutting note. `git revert 0c1ebe8` to roll back.
- **Memory consolidated 52 → 46 topic files** (index `MEMORY.md` 50→46 entries,
  in sync). Retired 8 done/dated; reconciled a stale fact (`always_gpu` —
  "always cuda" was contradicted by the multiprocess-CPU fast path); merged the
  two process-kill notes; folded the 2026-05-20/21 handoff's deployment caveats
  into `project_force_close_train_vs_deploy`; added `feedback_slim_always_loaded_context`
  + `project_tick_tock`; indexed an orphan. Backup: `C:\tmp\memory_backup_pre_consolidate`.

## 2. The memory system today (the always-loaded layer)
- Lives in the `.claude` profile (`…/memory/`), **auto-loaded by the harness**
  each session via `MEMORY.md` (one line per fact). Types: `user`, `feedback`,
  `project`, `reference`. ~46 entries after consolidation.
- This is a *push* store with a hard size budget (it's resident context). It is
  NOT the place for deep technical depth — that's what prong 2 is for. Keep it to
  user/feedback/orientation facts; push domain depth to the wiki.

## 3. llm-wiki-v3 — the deep-layer asset (`../../../llm-wiki-v3`)
An Engelbart/CoDIAK-grounded knowledge wiki (successor to v2). **Built, Phases
0–I complete.** Read its `README.md` + `AGENTS.md` + `docs/build_docs/v3-engelbart-plan.md`.
The properties that matter here:

- **Files are the system of record** — plain-text markdown notes in an Obsidian
  vault; portable, git, unzip-and-go. No server-as-store, no Postgres.
- **Granular IDs** (ULID frontmatter) + ID-based `[[wiki-links]]` (rename-safe).
- **Claims are the atom** — `*.claims.jsonl` sidecars carry `source-id` +
  locator + author. **Sources are referenced, not copied** (`Schema/sources.jsonl`).
- **Typed links** (epistemic + entity edges).
- **Embedded projection** — `wiki_tool.py project` builds a derived SQLite lens
  from the files; `query` answers structured questions **with the LLM off**
  (derived/gitignored/regenerable — never the truth).
- **CoDIAK dialog record** (`Schema/dialog.jsonl`) — the append-only reasoning
  trail (questions, decisions, contradictions + resolutions, supersessions),
  each entry linked to the claim/note/source ids it concerns.
- **Agency marking** — claims carry `asserted_by` / `verified_by`; `verify`
  promotes asserted→verified; `query unverified-claims` surfaces unverified ones.
- **The dependency-inversion test** (north star): with the LLM off, the wiki is
  still structured, addressable, queryable.
- **Note types:** `topic, concept, entity, project, synthesis, log, query`;
  context tags `work, home, research, meetings, lessons`. **Two clouds:**
  `shared/` (incl. work) and `personal/` (`shared` must never depend on `personal`).
- **Skills** (playbooks in `skills/<name>/SKILL.md`): `curate, ingest, extract,
  query, maintain, audit, dream, batch, static, gdrive-ingest`. Anti-shortcut
  extraction for large/dense docs (≥1 note/concept, not per file).

### Three runtime surfaces, one engine
Standalone CLI (`python scripts/wiki_tool.py …`), **MCP server** (`python -m
mcp_server`, read-only, shared-cloud-only by default), and a Docker stack — all
over the same file record + deterministic engine.

### The MCP read surface (how a rl-betfair session would query it)
`get_index` · `read_page` · `get_page_metadata` · `list_section` ·
**`search_pages`** (BM25 → path/title/score/snippet) · `resolve_links` ·
`get_source` · `list_queries` / **`query_wiki`** (`claims-by-source`,
**`unverified-claims`**, `relations`, **`supersedes`**, `contradictions`,
`sources`, `tag`, `note-claims`) · `list_views` / **`render_view`**
(`verified-only`, `unverified-claims`, **`dialog-trail`**,
**`supersession-timeline`**, `draft-vs-stable`).

Already in use for the operator's other domains (FEWS, ISTQB, clipped articles)
— rl-betfair would be a new domain in the same shared cloud (or a dedicated
vault — see `design.md` open decisions).
