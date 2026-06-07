---
id: 01KTFXVRENRH4980GDFBDPT5RJ
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-094c38]
aliases: [H1-H5 refuted, KL hypotheses, correlation refutation]
---

# KL-explosion hypotheses refuted by correlation

All five prompt hypotheses (H1 advantage-norm off, H2 reward-centering units, H3/H4/H5) were **refuted
by the matched (KL ↔ episode-row) correlation dataset** — a worked example of disproving suspects with
the evidence before fixing.

## What it is

The refutations: advantage normalisation **is** wired on the live mini-batch path (H1); reward
centering passes the per-step mean, not the episode sum, and `value_loss` sits in 0..33 (median 5.5) —
**three orders of magnitude below** the 6.8e+08 signature of the historical units bug, so H2 is out;
force-close magnitude correlates with KL at the **wrong sign** (ρ −0.239; more force-closes → *lower*
KL, refuting H4); entropy coefficient α correlates at the wrong sign too (H5). Only the true cause —
the stateful/stateless mismatch — predicts the observed KL-grows-with-episode signature. The method:
join `worker.log` approx_kl lines to `episodes.jsonl` on `(model_id, day)` and rank-correlate the
suspects.

## Why it matters

Don't fix the named suspect; correlate it first. A zero or wrong-signed correlation cheaply kills a
hypothesis. Companion to [[ppo-kl-stateful-stateless-mismatch]] (the surviving cause).

## Sources
- `src-094c38` findings.md (js_desktop:present)
