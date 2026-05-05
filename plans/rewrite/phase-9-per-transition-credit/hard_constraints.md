---
plan: rewrite/phase-9-per-transition-credit
---

# Hard constraints

§1  The per-slot label broadcast path MUST remain available as a fallback.
    Controlled by `training.per_transition_credit: bool` in config (default
    `false`). When `false`, all behaviour is byte-identical to the Phase 7
    baseline — same update statistics, same gradients. This enables clean
    A/B probing.

§2  Do not modify `env/betfair_env.py`, `env/bet_manager.py`, the reward path,
    or any v1 code. Per-transition credit is purely a label-assignment change
    inside `training_v2/`. No env changes.

§3  Open-step tracking uses the collector-side diff (Option C from purpose.md):
    snapshot `len(env.bet_manager.bets)` before each `env.step()` call;
    newly added bets are those at indices `[old_len:]` after the step.
    Do not add `step_index` to `Bet` or modify `BetfairEnv.step()`.

§4  Per-transition labels are assigned ONLY at the opening step of each pair
    (the transition where the aggressive first leg was placed). The closing
    transition is NOT labelled — the mature/naked outcome is credit-assigned
    to the DECISION to open, not the tick where the second leg happened to
    fill (which the agent didn't control).

§5  Per-episode JSONL gains `per_transition_credit_active: bool`. Pre-plan rows
    missing this field → treated as `false`. Do not break existing JSONL
    consumers (add field, don't rename or remove).

§6  Regression guard required: `test_per_slot_path_byte_identical_when_disabled`.
    With `per_transition_credit = false`, a full training episode produces
    bit-for-bit identical update statistics (`policy_loss`, `value_loss`,
    `approx_kl`) to a Phase 7 baseline run at the same seed and config.

§7  apply per-transition credit to `mature_prob` ONLY in this phase.
    `fill_prob` has a broken label (purpose.md — GA votes it to 0; per-
    transition credit doesn't fix the label content). `risk_head` NLL is
    left on the per-slot path (secondary effect; defer to follow-on).
    Do not extend to fill_prob or risk in this phase.
