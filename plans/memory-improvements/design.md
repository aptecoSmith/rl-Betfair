# Design — rl-betfair's deep knowledge layer IS llm-wiki-v3

The "wiki" in prong 2 is **not a new thing**. It is the operator's existing
`llm-wiki-v3`, with rl-betfair's knowledge ingested as a domain and queried via
its MCP server. This doc makes the case for why v3 fits, the retrieval path, the
ingestion approach, and the decisions still open.

---

## Why rl-betfair's knowledge is an unusually good fit for v3

v3's spine was built for exactly the shape rl-betfair knowledge already has:

| rl-betfair reality | v3 mechanism |
|---|---|
| A finding is a **claim sourced to a specific run** ("lay_price_max=20 → +£0.098/£ EV on the 2026-05-14 held-out re-probe") | **claims + claim-level provenance** (`*.claims.jsonl`, `source-id` + locator); `query claims-by-source` |
| Reward shaping is a **chain of supersessions** (CLOSE_SIGNAL_BONUS 1.0 → 0.5 → 0.0; equal-exposure → equal-profit sizing) | **supersession** (`query supersedes`, `render_view supersession-timeline`) — the exact archaeology we just cut from CLAUDE.md |
| The **why** behind each change (the lessons_learnt) | **CoDIAK dialog record** (`Schema/dialog.jsonl`) — decisions, contradictions + resolutions, linked to the claim ids; `render_view dialog-trail` |
| **"candidate to test" vs "held-out proven"** — THE cardinal discipline | **agency marking** (`asserted_by` / `verified_by`); `query unverified-claims` = *which findings are NOT held-out-confirmed yet* |
| Cross-cohort **contradictions** (a recipe that won in-sample, lost held-out) | `query contradictions` |
| The whole point: **reduce my context dependency** | **dependency-inversion** — query the knowledge with the LLM off |

The phenotype/tick-tock loop strengthens this further: each Tock produces a
hypothesis (an *asserted* claim) that its held-out summary either *verifies* or
*contradicts* — a native v3 lifecycle. The wiki could become where hypotheses and
their verdicts accumulate as first-class, queryable claims.

## The retrieval path (concrete)

Connect the v3 **MCP server** to the rl-betfair Claude session (read-only,
shared-cloud-only). Then, instead of grepping `plans/` or reloading a fat
CLAUDE.md, a session pulls what it needs:

- `search_pages("naked variance per leg")` → the synthesis note + snippet.
- `query_wiki("supersedes", of="close-signal-bonus")` → the reward-change chain,
  **LLM off**.
- `query_wiki("unverified-claims")` → rl-betfair findings asserted but not yet
  held-out-verified — a standing guard against trusting in-sample results.
- `render_view("dialog-trail", of=<decision-id>)` → *why* a knob was changed.
- `render_view("supersession-timeline")` → the CLAUDE.md archaeology we removed,
  on demand, structured.

This is the payoff: the always-loaded surface stays thin, and the depth is one
MCP call away — addressable by topic, with provenance, with the LLM off.

## Ingestion — two entry points

**1. Repo-scan (bulk — the rl-betfair corpus).** A `scan-repo` step walks the
rl-betfair tree (e.g. `plans/**/*.md`, `docs/**/*.md`, and root knowledge docs:
`CLAUDE.md`, `genes_census.md`, `plans/EXPERIMENTS.md`, `plans/EXPLORATIONS.md`),
**registers each as a v3 source** (referenced, not copied — the files stay in the
repo), and queues them for the **`batch`** skill to ingest resumably (one source
at a time; anti-shortcut: a note per *concept / finding / decision*, not per
file). This is the operator's "point it at the whole folder and ingest"
requirement — a thin **rl-betfair-specific wrapper over v3's existing `register` +
`batch`**, added to the copied wiki (the copy is rl-betfair's own, so a
repo-aware scanner belongs in it). Scope v1 to the knowledge *markdown*;
ingesting code is a later option.

**2. Inbox drop (external docs you find).** Keep v3's native inbox: drop an ML
paper / doc / URL into `wiki/inbox/pending/` (or `wiki/Clippings/` for
web-clips), and the wiki ingests it via the **`ingest`** skill, then moves it to
`processed/`. This is v3-as-shipped — **preserved by the copy, no new work**.

**Note shapes (both paths):** a `concept` note per mechanism (the matcher,
equal-profit sizing, naked variance, the held-out protocol), a `synthesis` note
per cross-cohort lesson, a `project` note per campaign, `claims` carrying the
numbers + their source run, `dialog` entries for the decisions.

**Proof-first sequence:** before turning the repo-scan loose on everything,
hand-ingest the **reward-shaping supersession cluster** to validate the fit (it's
the bulk of what we cut from CLAUDE.md and the best `supersession-timeline` demo),
then bulk-scan the rest.

## Architecture — SETTLED (2026-06-06)

**Take a copy of the existing `llm-wiki-v3` and make it rl-betfair's own, in the
repo as `rl-betfair/wiki/`** — a real, working v3 (engine + MCP server + skills +
schema), started fresh for rl-betfair's knowledge. Operator decision: *"take a
copy of the existing v3."*

Why a copy (not the shared vault): the engine resolves its vault relative to its
own location (`ROOT = scripts/`'s parent — confirmed in `wiki_tool.py`), so "v3
in the repo" means co-locating engine + vault. A copy keeps rl-betfair
**self-contained** — the knowledge versions with the code, there's no cross-repo
dependency, and it matches v3's own portable / unzip-and-go design (the core is
stdlib-only; only the MCP layer needs the `mcp` package).

The build (deferred — see Non-goals) will:
- **Copy the machine, not the knowledge → an EMPTY vault.** Copy v3 →
  `rl-betfair/wiki/`, then remove ALL content notes (`shared/`, `personal/`) and
  reset `Schema/*.jsonl` (sources, dialog, catalog) to empty, and drop the
  v3-*development* machinery (`benchmarking/`, `comparison/`, `ci/`,
  `docs/build_docs/`). Keep the *system*: `scripts/wiki_tool.py`, `mcp_server/`,
  `Schema/` templates, `templates/`, `skills/`, **`inbox/` + `Clippings/`**, and
  an `AGENTS.md` adapted for rl-betfair. Result: a clean, working v3 with zero
  notes, ready to ingest rl-betfair's own knowledge.
- Single `shared/` cloud (rl-betfair is all "work"; `personal/` unused).
- Gitignore the derived `.runtime/` + `out/`; commit the markdown + `Schema/*.jsonl`.
- Wire `wiki/`'s MCP server into rl-betfair's `.claude/settings.json` (read-only)
  so a session queries its own in-repo wiki.
- Record the source v3 commit in `wiki/VENDORED_FROM.md`; re-copy only if a v3
  engine improvement is ever wanted (low priority — the engine is stable).

## Remaining minor calls (decide at build time)
- **memory/ ↔ wiki boundary:** `memory/` stays the thin *always-loaded*
  user/feedback layer; technical domain depth goes to the wiki — a memory entry
  may *point* at a wiki note, not restate it.
- **Maintenance cadence (load-bearing):** run v3's `audit` / `maintain` / `dream`
  skills on a cadence so the wiki doesn't re-bloat the way CLAUDE.md did. The
  artifact is easy; the discipline is the point.
- **First ingest target:** the reward-shaping supersession cluster (best
  provenance + `supersession-timeline` demo).

## Non-goals / deferred
- Building any of this now — **tick-tock first.** This is design only.
- Migrating `memory/` wholesale into v3 (it has a structural always-loaded role
  the pull-only wiki can't replace).
- Automating ingestion before the manual loop has proven the fit.
