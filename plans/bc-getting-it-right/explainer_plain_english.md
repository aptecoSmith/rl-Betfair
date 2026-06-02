# BC getting-it-right — plain-English explainer

A non-jargon companion to `purpose.md` / `hard_constraints.md`. Added
2026-05-31 at the operator's request. The technical docs are the source
of truth; this is the "what are we actually doing and why" version.

## What we're training

A model (the "policy") looks at the market at one moment in time (a
"tick") and, for **each runner**, picks one of four actions: do nothing,
open a back trade, open a lay trade, or close a trade. Per-runner,
per-tick. That's it — the model's whole job is to make those calls.

## What "mature" means (and why mat% matters)

When the model opens a back trade, it also drops a resting **lay order** a
few ticks better than where it backed. If the market trades through and
that lay fills, the pair has **matured** — we've locked in a small
guaranteed profit (the "scalp"). If the lay never fills before the race
starts, we have to bail out at the off (a "force-close"), usually at a
small loss.

So **mat% = of the trades the model opened, how many actually locked the
scalp.** It's the natural scoreboard: high mat% = the model's "open this
trade" calls were good. mat% is and stays a primary metric.

## Why mat% alone wasn't enough to DEBUG the model

Two reasons — both about *fixing* the model, not about whether mat% is the
right goal:

1. **When mat% is bad, it doesn't tell you WHY.** Our first trained model
   scored 4% mat%. That could mean either (a) it picks bad runners, or
   (b) it picks okay runners but opens *far too many* — firing on nearly
   every tick, so the good picks drown in junk. Those need opposite
   fixes. mat% alone can't separate them. (A separate test showed it was
   (b): the model *can* pick well, it just wasn't being choosy.)

2. **We were accidentally training it on the wrong question.** The way the
   training data was built, the model mostly learned the EASY question —
   *"is there any tradeable spread on this runner right now?"* — when the
   question that actually makes money is the HARD one: *"will THIS
   particular trade's lay actually fill, or will I get stuck and have to
   bail?"* The two look almost identical the instant you open; only the
   second matters. We already have the data to teach the hard question;
   we just weren't using it that way.

## What this plan changes (in plain terms)

- **Keep mat% as the scoreboard.** It's the goal.
- **Add one companion question** the model should be able to answer:
  *"when you're most confident a trade will work, how often does it
  actually work?"* Reference point: a simple statistics model on the same
  data got its *confident* picks right **30% of the time vs 12% for random
  guessing** — so the signal is real and learnable.
- **Train it on the HARD question** (will this specific trade's lay fill)
  instead of the easy one, and **let it be choosy** (skip the iffy ones
  instead of opening everything).

## A note on jargon you may see in the technical docs

- **"AUC"** = just the academic name for that companion question ("when
  confident, how often right?", measured across all confidence levels).
  We track it internally; results are reported in plain mat% terms.
- **"mature_prob head"** = the part of the model whose job is to output,
  per runner, *how likely this trade is to mature.* The choosiness comes
  from only opening when that number is high enough.
- **"hard negatives"** = teaching examples of trades that looked good but
  did NOT mature (got stuck and bailed) — so the model learns the hard
  distinction, not just "spread vs. no spread."
- **"input normalization"** = a fix so a few huge raw numbers in the
  model's inputs (trade volumes in the hundreds of thousands) don't drown
  out everything else. Already done.
