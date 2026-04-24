# Post-KL-fix PARTIAL run archive (2026-04-24 22:00 UTC)

Launch of `post-kl-fix-reference` plan (`dcb97886-…`). The run was
interrupted mid-flight — 12 weight files + ~1,600 episode rows in
`logs/training/episodes.jsonl` were produced before the kill.

## What this data tells us (and why it's worth keeping)

**The PPO KL fix works, but not far enough.**

Sampled `approx_kl` values from worker.log on this run:

```
Episode 2 ep1 approx_kl = 3.9392
Episode 3 ep1 approx_kl = 18.8719
```

Pre-fix the same location showed median 12,740 (see
`plans/ppo-stability-and-force-close-investigation/findings.md`).
So the rollout↔update state-mismatch fix dropped KL by **~1,000×**.

But the 0.03 early-stop threshold is still tripping. Why: the check
runs **once per PPO epoch** after the full mini-batch sweep (~200
mini-batches per rollout at 10k+ transitions). Each mini-batch
takes a small gradient step; 200 small steps accumulate enough KL
drift to exceed 0.03 even though each individual step is tiny.
Classic PPO recipes (stable-baselines3) check KL **per mini-batch**
and stop the current epoch on the first mini-batch that breaches —
our `_ppo_update` at `agents/ppo_trainer.py:1996` waits until the
end of the epoch.

## Follow-on not yet implemented

Moving the KL check inside the mini-batch loop + bumping the
threshold to SB3's standard `target_kl × 1.5` would complete the
fix. Tracked as a residual in `plans/ppo-kl-fix/` (needs a
Session 02 entry in `lessons_learnt.md` + implementation).

## Contents

- `models.db` — 12 models' training + evaluation records (76 KB).
- `weights/` — 12 partial-training `.pt` files.
- No `training_plans/` — `dcb97886-…json` is still live (reset to
  status=draft for re-launch).

## Why not delete

Same reasons as `registry/archive_pre-kl-fix-20260424T211800Z/`:
- Data point for "what partial KL-fix looks like" when designing
  the Session 02 per-mini-batch fix.
- Scoreboard outcomes for the 12 partial agents are a useful
  baseline once the full fix lands.

Don't cross-load any of these weights into a post-Session-02 run —
the gradient pathway changed, so policy-state is not transferable.
