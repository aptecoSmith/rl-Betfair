---
id: 01KTG846Y40153CK0N7AZYN58A
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-0e88d0]
aliases: [market-state features, cross-runner features, where is the action]
---

# Market-state + cross-runner features

Two feature-engineering additions that let the per-runner predictor see beyond a single runner's own
slice: **market-state** features (broadcast across runners) and **cross-runner** features (per-runner,
relative to the field).

## What it is

Phase-15's predictor reads only the runner's own 125-dim slice, so it can't see **what today's market
is like overall** (volatile? thin book? unusual volume?) or what other runners are doing. **Market-state
(S02, broadcast):** rolling LTP volatility across the field, rolling traded volume vs day average, mean
top-of-book ladder depth, mean spread width. **Cross-runner (S03, per-runner relative):**
`volume_rank_in_field`, `volume_share_in_field`, `ltp_velocity_zscore` (vs field mean), and
`field_concentration` (HHI of money across the field). These **force every per-runner prediction to
incorporate** "where is the action actually happening?" — so horse A's predictor, seeing horse C with
volume rank 1 and 60% share, lowers its confidence on A and the gate stays dark for A.

## Why it matters

Addresses the other half of [[prediction-tail-instability]] (alongside [[ensemble-consensus-uncertainty]]):
a per-runner-only view ([[lstm-compression-bottleneck]] gave it the raw slice; this gives it field
context). A pure feature addition — RUNNER_KEYS expands, OBS_SCHEMA_VERSION bumps, oracle/direction
caches re-scan.

## Sources
- `src-0e88d0` purpose.md (js_desktop:present)
