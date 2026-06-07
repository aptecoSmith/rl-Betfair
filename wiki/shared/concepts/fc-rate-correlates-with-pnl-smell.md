---
id: 01KTGJG33WC7P44H2NKY6GW418
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-1663dd]
aliases: [fc rate correlates with pnl, safety net masks failure, cash buckets reconcile, the smell]
---

# fc rate positively correlated with eval P&L (the smell)

The diagnostic that exposed force-close as a crutch: in AMBER v2 the **two profitable agents have the
*highest* fc rates (0.850, 0.862)** — within this architecture, fc rate is positively correlated with eval
P&L. A safety net that correlates with success is masking the failure mode, not fixing it.

## What it is

A policy that can't close on its own settles via the crutch, and "we've stopped seeing that as a problem
because the cash buckets reconcile" — the force-closed pairs still settle to a P&L number, so the accounting
looks fine while the underlying capability (active closing) is absent. The tell is the *direction* of the
correlation: if relying on the backstop more makes an agent look better, the backstop is substituting for
the skill you wanted the policy to learn.

## Why it matters

A reusable audit heuristic: when a fallback mechanism correlates *positively* with your success metric,
suspect it's masking a missing capability rather than providing a real one — the reconciled accounting
hides the gap. This is the evidence base for [[force-close-is-a-crutch]] and a concrete instance of why
[[remove-decisions-beats-teaching]] / capability-vs-crutch distinctions matter when reading cohort
scoreboards.

## Sources
- `src-1663dd` purpose.md (js_desktop:present)
