# post-kl-fix-reference, gen-0 complete + gen-1 partial (2026-04-25 02:13Z)

Plan: `post-kl-fix-reference` (`dcb97886…`), run_id `cc2bd0ba`.
Wall time: 3,256 seconds (54 min).

## Outcome

- **Gen 0**: 12 agents trained successfully through 18 episodes
  each. best_fitness = 0.3554 (eea5080e, ppo_time_lstm_v1, pnl
  +£2.51, win_rate 83%). mean_fitness = −0.118.
- **Gen 1**: collapsed to 4 surviving agents due to a state_dict
  load bug (see "Bugs found" below). mean_fitness = −0.176.

Plan status finalised as `failed` because gen-1 didn't complete.
Reset to `draft` after archive for re-launch under the post-fix
trainer.

## What this run validated

**The Session-02 KL fix mechanically works.**

| metric | pre-fix | session-01 | this run |
|---|---|---|---|
| median `approx_kl` | 12,740 | 3–20 | **0.043** |
| max `approx_kl` | 4.6M | ~80k | **0.60** |

KL is now measured in the literature-normal range. The
rollout↔update state-mismatch is gone.

## What this run revealed

**Two follow-on bugs found:**

1. **Threshold-too-tight starvation.** With `kl_early_stop_
   threshold=0.03`, natural per-mini-batch drift of 0.03–0.07
   tripped the check on the 1st or 2nd mini-batch every update.
   PPO ran 3–13 mini-batches per update out of the ~600 budget
   (`ppo_epochs × mini_batches_per_epoch`). Bumping the default
   to 0.15 lands in commit alongside this archive.

2. **State_dict drift on gen-1 survivor load.**
   `backfill_hyperparameters` set `lstm_layer_norm=True` (int_choice
   midpoint default) on every gen-0 record at gen-1 start. Gen-0
   weights were trained with `lstm_layer_norm=False` → no
   LayerNorm params in state_dict → "Missing key:
   lstm_output_norm.weight" on rebuild. 8/12 agents skipped at
   gen-1, gen-1 collapsed to 4. Fix: extend
   `infer_arch_hp_from_state_dict` to detect `lstm_layer_norm`
   from the presence/absence of `lstm_output_norm.weight`, plus
   `strict=False` retry as a belt-and-braces safety net.

## What this run revealed about the agent's behaviour

Even under fixed PPO, the agent did NOT learn to be selective on
its own:

| metric | pre-fix gen-1 | this run gen-0 |
|---|---|---|
| pairs opened / race | ~243 | ~620 |
| force-close rate | ~75 % | 77 % |
| matured / race | 23 | 58 |
| `scalping_force_closed_pnl` | −£213 | −£390 |
| total_pnl / race | −£126 | −£362 |

The agent opens MORE pairs at the same selectivity ratio, so
absolute force-close cost goes UP. **`plans/selective-open-shaping/`
is justified** — needs an open-time cost with refund-on-mature
to teach selectivity.

## Why kept

- Gen-0 data is the cleanest post-Session-02 baseline; usable for
  the `selective-open-shaping` Session 02 gene-sweep design.
- Worker.log around line 1170310 is the diagnostic trail for both
  the threshold-too-tight observation AND the state_dict crash —
  preserve for future-me debugging.
- The 12 gen-0 weights are valid checkpoints under the post-fix
  trainer; could be loaded as references in the next probe.

## Don't reuse the weights

Don't cross-load these into a post-threshold-bump run — different
gradient regime. The next clean baseline run starts fresh with
the bumped threshold + the state_dict fix.
