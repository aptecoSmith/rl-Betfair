---
id: 01KTGP9R7FNW2YRZGE7EZ40C9S
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-1b344c]
aliases: [clamp log-var in forward, clamp at source, single clamp site, safe value to every consumer]
---

# Clamp at the source, not at each consumer

A design lesson from the risk head: clamp a numerically-dangerous quantity (log-variance) **inside the
forward pass** so `PolicyOutput` exposes an already-safe value — rather than leaving every downstream
consumer (the NLL, the parquet writer, the UI) to defend itself.

## What it is

The initial sketch returned raw log-var and made the NLL responsible for keeping `exp(log_var)` finite;
the switch to clamping inside each architecture's `forward` gave three benefits at once: parquet can't
store a silently-NaN stddev (`exp(0.5·log_var)` is well-defined), the NLL helper stays minimal (no
defensive clamping duplicated across modules), and `test_log_var_clamped_in_forward` "is a direct test of
the single clamp site." One clamp, one test, every consumer safe.

## Why it matters

A general robustness principle: validate/clamp at the point of production, not at N points of consumption —
it removes whole classes of "one consumer forgot to guard" bugs and gives a single test site. The same
spirit as the value-domain checks ([[value-domain-feature-assertions]]) and a relative of
[[shared-config-mutation-trap]] (fix the hazard where it originates, not at every reader). Especially
important when one of the consumers is a persisted file (a NaN stddev in parquet poisons the UI silently).

## Sources
- `src-1b344c` lessons_learnt.md (js_desktop:present)
