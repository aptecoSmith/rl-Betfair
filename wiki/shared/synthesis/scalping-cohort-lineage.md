---
id: 01KTGC1SKBW3SHHGHEDMD4GR1E
type: synthesis
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [scalping scoreboard, cohort lineage, band verdicts, layq null, E3 strong band]
---

# Scalping cohort lineage (the band-verdict scoreboard)

The chronological backbone of the scalping campaign, read off `plans/EXPERIMENTS.md`. Each cohort
inherited the prior's gate stack and was graded by band (MODEST / STRONG / REGRESSION / REJECTED). Two
load-bearing reference points recur across the project: the **layq null** (the deployment baseline) and
the **E3 cohort** (the first STRONG-band cohort in the project's history).

## What it is

The progression:
- **scalping-close-signal** (2026-04-17) — foundational `close_signal` action; the substrate every later
  plan ran on ([[close-signal-bonus]]).
- **selective-open-shaping** cohort-O/O2 (2026-04-25) — `open_cost` gene; mechanism delivered but the
  policy couldn't respond (no per-runner forecast in the action input).
- **scalping-pwin-gate** (2026-05-12) → **race-confidence-gate** (MODEST, +£39/d) → **lay-quality-gate**
  (STRONG, +£192/d fc=0 / +£26/d fc=120) — the **layq null** that becomes the baseline for everything
  after.
- **tnv → tnv2 (REGRESSION) → tnv3 (REJECTED on mechanism)** — variance-aware selection; rejected
  because [[selection-vs-measurement-signal]] (GA selection can't fix a reward-side problem).
- **fc-cost probes A/B/C/O/A2/H/D + E1/E2** — all NO BITE ([[gradient-delivered-ppo-unresponsive]]).
- **E3 close-feasibility gate** (2026-05-18) — BITES; full cohort = **STRONG band**, +£55/d fc=120, first
  strong-band cohort ever ([[close-feasibility-open-gate]]).
- **R-series + Sortino** — promising at probe, regress under full breeding ([[probe-to-cohort-regression]]).
- **recipe-expansion fc=0 mirage** — huge in-sample, collapses held-out ([[fc0-insample-mirage]]).

## Why it matters

The scoreboard is only comparable on the right axis — a reward-shape change moves `shaped_bonus` but
leaves `raw_pnl_reward` meaning-stable, and naked P&L is zero-EV variance, so cohorts are compared on
LOCKED + mat% + the fc=120 deployable cell, never on raw day_pnl over a short window. The one
intervention class that scaled cleanly was **env priors that REMOVE bad decisions** (E3-style), not
reward-shaping or selector changes — the recurring lesson of this whole lineage.

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
