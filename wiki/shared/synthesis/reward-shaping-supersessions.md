---
id: 01KTF937MECZT970TJZ4B2SS10
type: synthesis
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-bf778c, src-3f548f, src-04294a]
aliases: [reward shaping supersessions, reward shaping archaeology]
---

# Reward-shaping supersessions

The rl-betfair scalping reward did not arrive fully formed — it is a **chain of
supersessions**, each correcting a perverse gradient the previous shape produced.
This note is the topic-keyed map of that archaeology (the dated history kept out of
the always-loaded `CLAUDE.md`).

## Summary

Every change here edits one of the two buckets in [[raw-vs-shaped-reward]]. Because
each is a reward-scale change, rows trained across one of these boundaries are
**not comparable on `shaped_bonus`/`total_reward`** — compare on `raw_pnl_reward`,
which stays meaning-stable. Pre-change garaged models remain valid *as pre-change
references only*.

## The chains

| Mechanism | Superseded form | Current form | When |
|---|---|---|---|
| Pair sizing | [[equal-exposure-sizing]] (`S_back·P_back/P_lay`, exposure-equal) | [[equal-profit-sizing]] (profit-equal after commission) | commit `f7a09fc`, 2026-04-18 |
| Close reward | [[close-signal-bonus-legacy]] (£1 → £0.5) | [[close-signal-bonus]] (= 0.0) | `f193e41` (halve) → `5d57a91` (zero) |
| Naked penalty | [[naked-asymmetry-aggregate]] (`min(0, Σ naked)`) | [[naked-asymmetry-per-pair]] (`Σ min(0, naked)`) | 2026-04-17/18 |

## The recurring lesson

Two of the three were fixed by **removing or correcting** a term that created a
perverse gradient, not by re-tuning it: the close bonus competed with natural
maturation (so it was deleted), and the aggregate naked term let luck launder
losses (so the `min` moved inside the per-pair sum). The sizing fix was a
correctness fix — the old formula only equalised P&L at zero commission. The
through-line: at probe scale, removing a bad decision beats teaching the policy to
make it better.

## Sources
- `src-bf778c` purpose.md (js_desktop:present)
- `src-3f548f` CLAUDE.md (js_desktop:present)
- `src-04294a` purpose.md (js_desktop:present)
