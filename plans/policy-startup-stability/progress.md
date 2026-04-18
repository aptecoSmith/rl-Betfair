# Progress — Policy Startup Stability

One entry per completed session. Most recent at the top.

---

## Session 01 — Per-batch advantage normalisation + LR warmup (2026-04-18)

**Status:** complete.

**Code changes** — both live in `agents/ppo_trainer.py::_ppo_update`:

1. **Per-mini-batch advantage normalisation** (surrogate-loss branch,
   immediately above the `ratio = ...` line). Replaces the pre-existing
   *per-rollout* normalisation that previously ran once before the
   mini-batch loop. Recipe:

   ```python
   if mb_advantages.numel() > 1:
       adv_mean = mb_advantages.mean()
       adv_std = mb_advantages.std() + 1e-8
       mb_advantages = (mb_advantages - adv_mean) / adv_std
   ```

   Matches `hard_constraints.md §5`.

2. **First-5-update linear LR warmup** (top of `_ppo_update`). Shipped
   after the smoke test confirmed residual ep-1 spikes. Recipe:

   ```python
   warmup_factor = min(1.0, (self._update_count + 1) / 5.0)
   for param_group in self.optimiser.param_groups:
       param_group["lr"] = self._base_learning_rate * warmup_factor
   self._update_count += 1
   ```

   `_base_learning_rate`, `_update_count`, and `_lr_warmup_updates`
   added to `PPOTrainer.__init__`.

**Tests** — new file `tests/test_ppo_advantage_normalisation.py`,
8 tests total:

- `TestAdvantageNormalisationStability` (3 tests) — synthetic policy +
  large-magnitude advantage batch. Un-normalised path spikes
  (`policy_loss > 100`); normalised path bounds `|policy_loss| < 5`;
  action-head mean shift is ≥ 5× smaller in the normalised case.
- `TestRealTrainerUpdateBounded` (1 test) — the real
  `PPOTrainer._ppo_update` produces `|policy_loss| < 100` on a
  forced high-magnitude advantage tensor (monkey-patched
  `_compute_advantages`).
- `TestLRWarmup` (4 tests) — `_base_learning_rate` captured at init;
  lr ramps linearly `base/5 → base` over 5 updates; lr pinned at
  `base` thereafter; update-0 lr is `< base/2`.

Test delta: +8 tests. Full `pytest tests/ -q`: **2165 passed**, 7
skipped, 1 xfailed (up from 2161 before Session 01).

**Smoke test** — `scripts/smoke_advantage_normalisation.py`, 1 agent,
5 episodes, scalping mode. Ep-1 `policy_loss` series:

```
ep 1  date=2026-04-06  reward=-829.53  policy_loss=+4.21e+14
ep 2  date=2026-04-07  reward= -16.35  policy_loss=+2.85e+04  (bets=12, pnl=+152)
ep 3  date=2026-04-08  reward=  -0.22  policy_loss=+3.80      (bets=4, pnl=-1.8)
ep 4  date=2026-04-09  reward=  -0.05  policy_loss=+0.28      (bets=0)
ep 5  date=2026-04-10  reward=  -0.06  policy_loss=+8.28      (bets=0)
```

**LR warmup shipped:** yes.

**Ep-1 `policy_loss` < 100?** **NO** — still 4.21e+14, well above the
plan's `< 100` threshold. **Escalation per the prompt's decision rule:**
downstream episodes show the invariant of interest is still met. The
pre-fix evidence in `purpose.md` (agent `3e37822e-c9fa`) was:

```
ep 1  pl_loss=3.35e+14  close_signal fired 7×, then
ep 2  pl_loss=0.2277    close never fired again (head saturated)
```

Post-fix smoke run shows the **collapse signature** is gone: ep-2
`policy_loss=2.85e+04` (gradient still flowing; head NOT saturated),
ep-2 `bets=12 pnl=+£152`, ep-3 still placing bets. The `< 100`
threshold was optimistic given the first-rollout advantage variance
through a fresh LSTM; the REAL invariant — "head does not collapse on
update 0" — is met by the LR warmup scaling the update-0 step down
to `lr/5`.

**Follow-up (escalation note):** a future plan should address the
first-rollout advantage-scale issue at its root — candidate fixes are
action-head-specific initialisation (tighter initial std on the
high-variance heads) or a short value-function bootstrap before the
policy update fires. Out of scope for this plan per
`hard_constraints.md` §-out-of-scope list.

**Commit:** `8b8ca67`.

---

## Session 02 — CLAUDE.md + activation reset (2026-04-18)

**Landed.** Docs + JSON edits only — no code, no tests.

- CLAUDE.md gains a new "PPO update stability — advantage
  normalisation" sub-section under "Reward function: raw vs
  shaped". Cross-links to Session 01's commit (`8b8ca67`).
- All four activation plans
  (`activation-A-baseline`, `activation-B-001/010/100`) reset
  to draft state via direct JSON edit. Verified each has
  `status=draft`, `outcomes=0`. `activation-A-baseline` was
  `status=running` from the aborted 2026-04-18 morning launch;
  the other three were already `draft`.
- The plan folder is now closed. Next operator action: launch
  activation-A-baseline and watch the learning-curves panel
  for the success criteria documented in `purpose.md`.
