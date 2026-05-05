---
plan: rewrite/phase-11-bc-gene-exploration
---

# Hard constraints

_(to be expanded; placeholder so the plan folder has the standard
shape.)_

§1  Don't open this plan until Phase 8 S03 has shipped and either
    passed or failed its gate. Tuning a mechanism we haven't
    validated risks chasing noise.

§2  Whatever option is chosen (manual sweep vs GA-evolvable),
    `bc_pretrain_steps = 0` must remain byte-identical to a no-BC
    run. The Phase 8 S02 §7 contract carries forward.

§3  No env edits. No reward-shape changes. This plan tunes only the
    BC knobs in `training_v2/discrete_ppo/bc_pretrain.py` and the
    trainer's warmup arithmetic.
