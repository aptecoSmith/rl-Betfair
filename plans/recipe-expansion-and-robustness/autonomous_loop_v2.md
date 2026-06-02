# Autonomous loop v2 — held-out + mat%-selection regime

Standing instructions for the autonomous experiment loop, rewritten
2026-05-28 after the held-out reckoning. Supersedes
goal_directed_instructions.md (which chased total day_pnl and led to
the fc=0 overfitting mirage).

## The hard-won lessons (do not relitigate)

1. **ALWAYS eval on held-out days. NEVER train on them.** Every
   experiment: train on 2026-04-06/08/09; eval on held-out May days.
   Iteration-eval = 7 odd days (05-07,09,11,13,15,17,19). Final-test =
   7 even days (05-08,10,...,20), reserved, touched once at the end.
2. **Naked P&L is zero-EV directional variance.** It looked like a
   +£287 edge in-sample; held-out it was -£175. Ignore total day_pnl
   when it's naked-dominated. fc=120 stays ON (safety rail).
3. **The real edge is LOCKED P&L from matured scalps.** Held-out,
   each matured scalp locks +£3-6 after commission. The mechanic
   works. THE BOTTLENECK IS mat% — only 2-3% of opened pairs mature.
4. **Selection, not quality, is the problem.** The agent opens
   100-160 pairs/day; ~97% never fill both legs and get force-closed
   (-£106 to -£170/day). For net-positive, mat% must rise from ~3%
   toward 30-50%, OR opens must drop drastically to only the
   high-fill-probability pairs.

## The objective

Find a recipe with, ON HELD-OUT DAYS:
- **mat% materially above ~3%** (the headline target — scalp selection)
- **locked-per-matured-pair staying positive** (+£3+)
- **total locked P&L exceeding force-close + close cost** → day_pnl ≥ 0
- reasonable opens (not collapsed to ~0)

Rank candidates on **held-out mat% and held-out locked P&L**, NOT
total day_pnl (which is naked-noise on a short window).

## Levers to push mat% (try these, measure on held-out)

1. **Tighter target spread** — `arb_spread_target_lock_pct` low
   (0.005 or below). Passive sits closer to LTP → fills more often →
   higher mat%. Trade-off: smaller locked per pair. Find the knee.
2. **Fill/mature-prob gating** — only open when the policy's
   `mature_prob_head` (or `fill_prob_head`) predicts the pair will
   fill. We HAVE these heads but have never gated OPENs on them.
   Needs a small env/action-space change (a mature-prob open gate)
   — analogous to the pwin_back gate. ~30-60 min eng.
3. **Drastic selectivity** — push opens from ~130 down to ~20-40 by
   stacking gates, betting that fewer-but-better opens mature more.
4. **Force-close window** — moderate changes (T-90, T-150) give pairs
   more/less time to fill. Already partly explored; revisit on held-out.
5. **Higher BC dose is NOT the answer** — BC makes the agent open
   MORE (worse selection). Lower or zero BC may help selection.

## Loop protocol (each wake)

1. Read this file + monitoring_notes.md. Check the active wrapper's
   log for completion.
2. If the active round finished: pull held-out metrics (mat%, locked,
   locked/matured, day_pnl) per cell. Append a dated entry to
   monitoring_notes.md: cells done, held-out leader by mat% + locked,
   what it implies.
3. Design the NEXT round targeting mat%, based on what just landed:
   - If a lever lifted held-out mat%, sweep around it.
   - If a lever did nothing, drop it.
   - If selection needs an architectural gate (mature-prob open gate),
     build it (small env change + test) then sweep it.
4. Write the round wrapper (train on train days, **eval on the 7
   held-out odd days**), launch it detached via
   `nohup bash <wrapper> &` (the reliable launch path — NOT the
   babysit Popen path, which dies under job objects).
5. Update EXPERIMENTS.md with the round's intention + result when it
   completes.
6. Reschedule the next wake (~30-40 min, one cell cycle; longer if a
   multi-day held-out eval round is running, since those cells are
   ~30-40 min each).

## Launch reliability (learned the hard way)

- Launch wrappers with `nohup bash <wrapper> > /dev/null 2>&1 &` then
  `disown`, from a normal Bash tool call. This survives.
- The babysit_loop.py Popen launcher is unreliable under Windows job
  objects — do NOT rely on it for launches.
- Chain wrappers (poll upstream log for "fan-out complete", then run
  next) survive session close and are the robust way to queue
  multiple rounds.
- Kill processes via psutil (Windows PID namespace), not git-bash
  `kill` (different namespace, silently no-ops).

## Stop conditions

- A recipe shows held-out mat% > ~15% AND held-out day_pnl ≥ 0 across
  ≥3 seeds → strong candidate, document and tell the operator.
- Exhausted the obvious mat% levers with no held-out improvement →
  write a findings summary recommending the next structural idea
  (e.g. a different selection signal, or a redesign of the open
  decision), and tell the operator.
- Otherwise keep looping.
