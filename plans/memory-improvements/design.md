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

## Ingestion approach

- **Source = the `plans/` folders + the slimmed-out CLAUDE.md history.** Register
  them as v3 sources (referenced, not copied — `wiki_tool.py register`), then run
  the **`ingest`** / **`extract`** skills (anti-shortcut: one note per *concept /
  finding / decision*, not per file; cross-link into hubs; lift key numbers into
  the notes so they survive even if a plan is archived).
- Likely note shape: a `concept` note per mechanism (the matcher, equal-profit
  sizing, naked variance, the held-out protocol), a `synthesis` note per
  cross-cohort lesson, a `project` note per campaign, `claims` carrying the
  numbers + their source run, `dialog` entries for the decisions.
- Start with the **densest, highest-supersession cluster** as the proof: the
  reward-shaping history (it's the bulk of what we cut from CLAUDE.md and the best
  supersession-timeline demo).

## Open decisions (the operator's calls)

1. **One shared vault vs a dedicated rl-betfair vault.** Lean **shared** —
   rl-betfair becomes a `work`-tagged domain in the existing v3 shared cloud
   (v3-native: one vault, many domains, tags+search partition; reuses all infra).
   A dedicated vault gives cleaner separation but is a second vault to maintain.
2. **Manual vs automated ingestion.** Lean **manual-first** (the `ingest` skill,
   operator-driven) to seed and prove the loop; automate later (a script or a
   tick-tock-style routine that files new findings/hypotheses as claims).
3. **memory/ ↔ wiki boundary.** Keep `memory/` as the thin *always-loaded*
   user/feedback/orientation layer; push *technical domain depth* to the wiki.
   Don't duplicate — a memory entry may *point* at a wiki note, not restate it.
4. **MCP wiring.** Add the v3 MCP server (`python -m mcp_server` in
   `llm-wiki-v3`) to the rl-betfair session's MCP config (read-only). Confirm the
   shared-cloud-only default is what we want from a rl-betfair session.
5. **Maintenance cadence — the load-bearing one.** CLAUDE.md bloated because
   nothing pruned. The wiki re-bloats the same way without a recurring pass: v3's
   **`audit`** (contradictions / stale claims / gaps), **`maintain`** (health),
   and **`dream`** (reflect + propose questions) skills are the tools; pick a
   cadence (manual, or a scheduled routine). The artifact is easy; the discipline
   is the point.
6. **First ingest target** to validate the fit: the reward-shaping supersession
   cluster (per "Ingestion approach").

## Non-goals / deferred
- Building any of this now — **tick-tock first.** This is design only.
- Migrating `memory/` wholesale into v3 (it has a structural always-loaded role
  the pull-only wiki can't replace).
- Automating ingestion before the manual loop has proven the fit.
