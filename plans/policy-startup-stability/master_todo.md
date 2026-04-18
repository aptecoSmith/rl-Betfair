# Master TODO — Policy Startup Stability

Two sessions. Session 01 lands the fix + tests; Session 02 is
the documentation + activation-plan reset. Both are designed to
be runnable unattended — clear deliverables, clear exit
criteria, no manual decision points mid-session that an agent
can't resolve from the brief alone.

## Session 01 — Per-batch advantage normalisation + tests

**Status:** pending

**Deliverables:**
- `agents/ppo_trainer.py`: per-mini-batch advantage
  normalisation in the surrogate-loss branch. One function,
  one localised change. Use the literature-standard recipe
  (`(adv - mean) / (std + 1e-8)`).
- New synthetic test in `tests/test_ppo_trainer.py` (or new
  file `tests/test_ppo_advantage_normalisation.py` if the
  trainer test file is already large) that demonstrates
  spike-prevention per `hard_constraints.md §16`.
- Smoke test: 1-agent, 5-episode training run. Confirm
  `episodes.jsonl` for that agent shows NO `policy_loss`
  exceeding 100 on episode 1. If it does, ship the optional
  first-update LR warmup defence-in-depth (per
  `purpose.md` §3 and the Session 01 prompt's stretch goal).

**Exit criteria:**
- `pytest tests/test_ppo_*.py -q` green.
- New advantage-normalisation test green.
- Pre-existing
  `test_invariant_raw_plus_shaped_equals_total_reward` green.
- Full `pytest tests/ -q` green.
- Smoke run produces an `episodes.jsonl` row for ep 1 with
  `policy_loss < 100` (and ideally `< 5`).

**Acceptance:** the new synthetic test ASSERTS that the
post-update action_head mean shift is materially smaller in
the normalised case than the un-normalised case, on the same
fake rollout. This is the principled check that the fix
prevents collapse, not just dampens it.

**Commit:** one commit, type `fix(agents)`. First line names
the change. Body explains the failure mode (with a one-line
reference to the agent `3e37822e-c9fa` evidence from
`purpose.md`) and the literature precedent (Engstrom et al.
2020).

## Session 02 — CLAUDE.md + activation-plan reset

**Status:** pending

**Deliverables:**
- CLAUDE.md gains a paragraph under "Reward function: raw vs
  shaped" (or as a new sub-section if it fits better — exercise
  judgement when reading the existing structure) explaining:
  - That advantage normalisation is in effect in the PPO
    update loop.
  - Why it's load-bearing: large-magnitude rewards (typical of
    scalping) without normalisation produce gradient updates
    on the first rollout that saturate action heads.
  - One sentence cross-link to this plan.
- `progress.md` of this plan gets a Session 02 entry
  cross-linking to Session 01's commit.
- All four activation plans (`activation-A-baseline`,
  `B-001/010/100`) reset to draft — same JSON-edit pattern
  used in `scalping-naked-asymmetry` Session 02 and earlier:
  `status='draft'`, `started_at=None`, `completed_at=None`,
  `current_generation=None`, `current_session=0`,
  `outcomes=[]`. Verify post-edit by listing the four plans'
  `status` and `outcomes`.
- (Optional, operator preference) prune the non-garaged
  models from the now-stopped 2026-04-18 morning run via
  `scripts/prune_non_garaged.py`. Not strictly required —
  pruning is independent — but a clean registry makes the
  next run's learning-curves panel start from a clean slate.

**Exit criteria:**
- Prose merge for CLAUDE.md (commit type `docs`).
- Activation plans confirmed draft (manual `ls` + `python -c`
  loop, or via the API).
- `git status` clean except for the gitignored archive folders.

**Acceptance:** opening `plans/INDEX.md` and CLAUDE.md cold,
the reader can:
- Identify this plan as the latest.
- Find the advantage-normalisation note in CLAUDE.md without
  searching.
- See that all four activation plans are ready to launch.

**Commit:** one commit, type `docs`. Cross-references Session
01's commit hash.

---

## After Session 02: re-run activation-A-baseline

The fix is complete on the operator side. To validate:

1. Operator launches activation-A-baseline.
2. Watch the learning-curves panel. Expected differences from
   the 2026-04-18 morning run:
   - `policy_loss` series in the per-agent panel shows NO
     ep-1 spikes above 100 across the population.
   - `arbs_closed > 0` on multiple agents, not just episode
     1 of one agent.
   - `best_fitness` per generation moves across the GA run
     (not frozen at one value).
3. Capture findings in this plan's `progress.md` under a
   "Validation" entry following the convention used in
   `scalping-equal-profit-sizing/progress.md`.

If validation succeeds: green light to run the activation
playbook's B sweeps. If the same frozen-fitness pattern still
appears with `arbs_closed=0`, the next layer is action-head
initialisation — opens a fresh plan. (Don't fold action-head
init changes into this plan; they're a different fix with
their own constraints.)
