# Kickoff — build rl-betfair's in-repo knowledge wiki

Open this session **in the rl-betfair repo** (`C:\Users\jsmit\source\repos\rl-betfair`).
You're executing the BUILD of the in-repo knowledge wiki. The design is settled —
this session implements it. It is **independent of the tick-tock work**; don't
touch the running pbt campaign or the training code.

## Read first (the settled design)
- `plans/memory-improvements/purpose.md` — why (slim the always-loaded surface;
  add a deep, pull-on-demand retrievable layer).
- `plans/memory-improvements/current_state.md` — what's done (CLAUDE.md slimmed
  1634→421; memory consolidated 52→46) + what `llm-wiki-v3` is (the engine you copy).
- `plans/memory-improvements/design.md` — **your spec.** §"Architecture — SETTLED"
  + §"Ingestion — two entry points".

## The decision (made — do not relitigate)
Take a COPY of the operator's existing `llm-wiki-v3`
(`C:\Users\jsmit\source\repos\llm-wiki-v3`) and make it rl-betfair's own, in-repo
at `rl-betfair/wiki/`. **Copy the machine, not the knowledge — start EMPTY.**
rl-betfair ends up with a self-contained, working v3 it queries via MCP.

## Build steps (commit in revertible chunks)
1. **Copy + empty.** Copy `llm-wiki-v3` → `rl-betfair/wiki/`, then:
   - Delete all content notes (`shared/`, `personal/`) and reset `Schema/*.jsonl`
     (`sources`, `dialog`, `catalog`, `ingest-log`) to empty.
   - Remove `wiki/.git` (it joins rl-betfair's git) and `wiki/.venv` (don't vendor it).
   - Drop the v3-*development* machinery: `benchmarking/`, `comparison/`, `ci/`,
     `docs/build_docs/`.
   - KEEP the system: `scripts/wiki_tool.py`, `mcp_server/` (+ `core.py`),
     `Schema/` templates, `templates/`, `skills/`, `inbox/` + `Clippings/`,
     `requirements.txt`, `AGENTS.md`.
   - Record provenance: `wiki/VENDORED_FROM.md` = the `llm-wiki-v3` git SHA + date
     you copied from.
2. **Adapt `wiki/AGENTS.md`** for rl-betfair: single `shared` cloud (all "work";
   `personal/` unused), the inbox is for external ML docs, and add the repo-scan
   path (step 7). Keep v3's binding invariants verbatim (files-as-record,
   zero-dependency core, dependency-inversion, deterministic-extracts-facts).
3. **Gitignore** the derived artifacts in rl-betfair's `.gitignore`:
   `wiki/.runtime/`, `wiki/out/`.
4. **Verify the empty engine runs** (the zero-dep core must work standalone):
   `python wiki/scripts/wiki_tool.py doctor`; `init` if it asks; then `project`
   and `query --list`. You want a clean, queryable, empty wiki.
5. **Wire MCP** into `.claude/settings.json` (use the `update-config` skill): add
   the wiki's MCP server **read-only** so a rl-betfair session can `search_pages`
   / `query_wiki` / `render_view` against the in-repo wiki. The core is stdlib (no
   install); the MCP server needs the `mcp` package — install `wiki/requirements.txt`
   into whatever env runs it.
6. **Proof-first ingest** (validate the fit BEFORE bulk): hand-ingest the
   **reward-shaping supersession cluster** — the dated history cut from CLAUDE.md
   (CLOSE_SIGNAL_BONUS 1.0→0.5→0.0; equal-exposure→equal-profit sizing; the
   raw/shaped reward revisions). Sources = the relevant `plans/` folders + the git
   history of `CLAUDE.md`. Produce `concept`/`synthesis` notes, `claims` with
   provenance, `dialog` entries for the decisions, and **demonstrate**
   `render_view supersession-timeline` + `query unverified-claims` working. If the
   fit is good, proceed; if the schema fights rl-betfair's shape, note it and adjust.
7. **Build the repo-scan ingester** (the "scan the whole folder" requirement): a
   thin wrapper over v3's `register` + `batch` that walks `plans/**/*.md`,
   `docs/**/*.md`, and root knowledge docs (`CLAUDE.md`, `genes_census.md`,
   `plans/EXPERIMENTS.md`, `plans/EXPLORATIONS.md`), registers each as a source
   (referenced, not copied), and queues them for batch ingest. Scope v1 to
   markdown knowledge (NOT `.py` code). Then run it to ingest the rest.

## Guardrails
- **The vault starts EMPTY** — copy machinery, not notes.
- Don't break v3's invariants: files are the system of record; the projection DB
  is derived/gitignored/regenerable; the zero-dep core always works; with the LLM
  off the wiki is still structured/addressable/queryable.
- **Anti-shortcut ingestion:** a note per concept/finding/decision, not per file
  (v3's `extract` skill for dense docs).
- This copy is rl-betfair's OWN — rl-betfair-specific additions (the repo-scanner)
  belong inside it.

## Out of scope / later
- Ingesting `.py` code (markdown knowledge first).
- A scheduled auto-ingest / maintenance routine (`audit`/`dream` cadence) — set up
  after the corpus lands and the fit is proven.
