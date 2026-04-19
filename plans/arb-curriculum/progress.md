# Progress — Arb Curriculum

One entry per completed session. Most recent at the top.
Include commit hash, what landed, what's not changed, and
any gotchas.

Format per session follows
`plans/reward-densification/progress.md` — "What landed",
"Not changed", "Gotchas", "Test suite", "Next".

---

_Plan folder created 2026-04-19. See `purpose.md` for the
structural diagnosis flowing from the
`reward-densification-probe` 2026-04-19 failure and the
gene-sweep currently running: the policy finds "arb less"
before "arb better" because random arbing is expected-
negative. This plan attacks the local minimum via four
coordinated interventions: offline oracle scan, matured-arb
bonus, naked-loss annealing, BC pretrain + curriculum day
ordering._

_Operator observation 2026-04-19: one training day had
only 3 possible arbs across the whole day. That's bad
curriculum material at agent init — nothing for the policy
to imitate or exploit, but still ambient cost on any
random arbing. Curriculum day ordering (Session 05) uses
Session 01's oracle density to front-load arb-rich days._
