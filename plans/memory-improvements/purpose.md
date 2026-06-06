# Memory improvements — slim the always-loaded surface, deepen the retrievable layer

**New here? Read in order:** `purpose.md` (this) → `current_state.md` (what's
done + the assets, incl. llm-wiki-v3) → `design.md` (the target architecture +
open decisions).

---

## The problem

rl-betfair's knowledge has accreted over months. The operator's concern
(2026-06-06): *"is Claude getting dumber because there are too many docs to look
over?"* — a real symptom, correctly felt.

**The cost is the *always-loaded* surface, not the existence of knowledge.** Two
things load into context every session and dilute focus / eat budget:
`CLAUDE.md` (had grown to 1634 lines, mostly dated reward-shaping archaeology)
and the `MEMORY.md` index (50 entries, some stale/contradictory). The `plans/`
folders are *pull-not-push* — they cost search noise, not attention budget.

So the fix is **curation, not relocation** — and a new repo would make it worse
(sharper on plumbing, dumber on the science, which needs the accreted domain
knowledge). See memory `feedback_slim_always_loaded_context`.

## The two-pronged fix

1. **Slim the always-loaded surface** *(done this session — see
   `current_state.md`)*. `CLAUDE.md` → current invariants + "do not re-break"
   rules + test-guard pointers + plan pointers (1634→421). Memory consolidated
   (52→46 files; stale facts fixed; done/dated entries retired). The discipline,
   not the artifact, is the thing — this needs a recurring pruning cadence or it
   re-bloats.

2. **Stand up a deep, retrievable layer** so the slimmed-out depth isn't lost —
   it's *pulled on demand* instead of *resident*. **This layer is
   [`llm-wiki-v3`](../../../llm-wiki-v3)** (the operator's existing
   Engelbart/CoDIAK knowledge wiki, MCP-served), NOT a new bespoke wiki. rl-betfair's
   knowledge is an unusually good fit for v3's spine (claims + claim-level
   provenance, supersession, a captured decision/dialog record, asserted→verified
   agency, and the dependency-inversion property — *queryable with the LLM off*).
   The mapping is the heart of `design.md`.

## The reframe: four stores, clear roles

| Store | Role | Loaded |
|---|---|---|
| `CLAUDE.md` | current invariants + the rules | always (thin) |
| `memory/` | user facts + cross-session feedback | always (thin index) |
| `plans/` | raw per-experiment history (archaeology) | pull |
| **`llm-wiki-v3`** | **topic-keyed synthesis: claims/provenance/dialog, MCP-served, LLM-off queryable** | **pull (deep)** |

Without the fourth store the slimmed depth has nowhere to live but `plans/`
(chronological, hard to retrieve by topic). v3 is the topic-keyed synthesis layer
over `plans/` — *not a copy of it*.

## Scope

This plan is the **design**. The build is **deferred** — tick-tock
(`plans/tick-tock/`) is the live priority. Prong 1 (curation) is already done;
prong 2 (v3 integration) is designed here and built later. Memory pointer:
`project_tick_tock`, `feedback_slim_always_loaded_context`.
