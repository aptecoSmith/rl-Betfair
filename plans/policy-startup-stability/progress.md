# Progress — Policy Startup Stability

One entry per completed session. Most recent at the top.

---

_(No completed sessions yet. Plan folder created 2026-04-18 in
response to the post-`scalping-equal-profit-sizing` activation-
A-baseline run, where agent `3e37822e-c9fa`'s trajectory
revealed the same pattern observed in two previous runs: a
catastrophic policy_loss spike on episode 1 saturates the
close_signal action head, after which the action never fires
again. Fixes #12, #13, #14 add mechanism / shape / math; this
plan addresses the training-loop stability that prevents those
mechanisms from being properly exercised by the GA. See
`purpose.md` for the agent-3e37822e trajectory + literature
reference (Engstrom et al. 2020)._)
