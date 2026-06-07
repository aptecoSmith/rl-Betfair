---
id: 01KTGJS2NNZKGT33S8ZBG10Z2R
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-184d90]
aliases: [shared config aliasing, override mutates shared dict, config copy, sleeper bug]
---

# Shared-config mutation trap

A sleeper aliasing bug: merging per-agent reward overrides directly into `config["reward"]` mutates a
config object that **multiple agents share**, so one agent's overrides silently leak into the others.

## What it is

"Shared-config mutation is an easy trap." The first draft of `BetfairEnv.__init__` merged overrides into
`config["reward"]` directly — a horrible sleeper bug the moment two agents share a config object, "which
is exactly what `PPOTrainer(... config=self.config ...)` does." The fix is to copy before mutating, locked
in by a regression test (`test_env_overrides_do_not_mutate_shared_config`) so it can't be reintroduced.

## Why it matters

A general hazard wherever per-instance customisation is applied to a shared mutable default (config dicts,
default args, class-level containers): mutate-in-place aliases every holder of the reference. The cohort
makes it especially dangerous because agents are *meant* to differ — silent cross-contamination would
invalidate every per-agent comparison. Same "diff the call sites / who shares this object" discipline as
the forked-path silent-drop family ([[batched-path-silent-drops]]); the cure is copy-on-write plus a
no-mutation regression test.

## Sources
- `src-184d90` lessons_learnt.md (js_desktop:present)
