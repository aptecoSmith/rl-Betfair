# Arb Signal Cleanup — Session 01 prompt

Current session: **Session 01 — Force-close at T−N +
entropy-velocity gene**.

Detailed brief:
[`session_prompts/01_force_close_and_entropy_velocity.md`](session_prompts/01_force_close_and_entropy_velocity.md).

Before starting, read:

- [`purpose.md`](purpose.md) — the three-mechanism
  diagnosis from the 2026-04-21 `arb-curriculum-probe`
  Validation (3/5 pass, C1 + C4 fail). In particular the
  "Why these three together, not sequential" subsection
  that scopes this plan as a single combined probe, not
  a sequence.
- [`hard_constraints.md`](hard_constraints.md) — 41
  non-negotiables. §9–§14 (force-close mechanics),
  §15–§18 (entropy velocity gene), §28–§29 (telemetry +
  invariant), §30–§33 (testing).
- [`master_todo.md`](master_todo.md) — three-session
  scope and per-session exit criteria.
- `plans/arb-curriculum/progress.md` — the Validation
  entry on `arb-curriculum-probe` that motivated this
  plan. Read the "Criteria results", "BC diagnostics",
  and "Naked vs force-closed rate" columns so you know
  what signal this plan's validator is looking for.
- `plans/arb-curriculum/purpose.md` — the five success
  criteria this plan's validator reuses. They haven't
  changed; this plan asks the same question of a
  different probe.
- `env/betfair_env.py` — the file being edited for
  force-close. Locate the step loop around line 1458
  (where `time_to_off` is computed) and
  `_settle_current_race` around line 2190. Also
  `_attempt_close` around line 1961 — the execution
  mechanism force-close reuses.
- `env/exchange_matcher.py` — understand the junk filter
  and LTP guard that force-close inherits. Do NOT
  modify this file.
- `env/bet_manager.py` — the Bet dataclass gains a
  `force_close: bool = False` attribute and BetManager
  may gain a `scalping_arbs_force_closed` counter.
- `agents/ppo_trainer.py` — the file being edited for
  entropy velocity. Locate `_alpha_optimizer`
  construction (search for `self._log_alpha` and
  `SGD`). Also locate the trainer-side gene override
  path — if `_TRAINER_GENE_MAP` doesn't exist, Session
  01 establishes it.
- `CLAUDE.md` "Order matching", "Bet accounting",
  "Reward function: raw vs shaped", "Entropy control"
  sections. The accounting invariants there are the
  ones we're preserving.

## Do NOT

- Do NOT touch the matcher. Force-close calls existing
  matcher code; the matcher itself is unchanged.
- Do NOT count force-closes as matured arbs. Matured-arb
  bonus stays at `completed + closed` (agent-initiated).
- Do NOT run the full pytest suite during active
  training. Operator directive from prior plans.
- Do NOT bundle Session 02 changes into this session. The
  shaped-penalty warmup is a separate commit.
- Do NOT set the default values to non-zero. Both new
  knobs default to 0 so existing runs are byte-identical.
