---
id: 01KTG846Y5YPC19KPZJJPBJHPA
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0e88d0]
aliases: [upper-tail instability, day-to-day variance, box reliability shifts]
---

# Prediction upper-tail instability (day-to-day)

Phase-15's predictor-as-feature architecture works, but the **upper tail isn't stable across days**:
the gate at the top of predictions delivers selective trading, yet its reliability shifts day-to-day
and the predictor alone can't tell which kind of day it's on.

## What it is

In the phase-15 big run, one agent at T=0.75 hit a 40% mature rate across 3 eval days (above the 34.8%
break-even) while another at the same threshold hit 9%; day 1 was bet-active but loss-making, day 3 was
NOOP for most agents — only 1/12 agents cleared break-even, and v8's positive *single-day* pnl did not
generalise to a 3-day window. The strategic model ("the box stays dark; bet rarely when it lights") is
right, but the box's reliability varies and the predictor has no sense of the day's character. This is
also why the **headline metric is mature rate aggregated across multiple eval days, not single-day
pnl** — per-day pnl variance is huge and **single-day signals fool you**.

## Why it matters

Motivates phase-16 (consensus + market-context features): a calibrated per-runner predictor
([[direction-head-feature-slice]]) still can't read the day. Same single-day-noise caution as
[[eval-sampling-variance-dominates]]. Fixes: [[ensemble-consensus-uncertainty]] and
[[market-state-and-cross-runner-features]].

## Sources
- `src-0e88d0` purpose.md (js_desktop:present)
