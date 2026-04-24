# Lessons learnt — Arb Signal Cleanup

One entry per surprise (bug, design miss, behavioural
finding) uncovered during this plan's sessions. Dated.
Most recent first.

Format per entry: **Date · Session · Title**, then a
short paragraph on what surprised us, what was wrong,
and what the fix or policy change is.

---

## 2026-04-21 · Pre-plan · Transformer architecture has bounded memory (goldfish confirmed)

Operator asked whether the three architectures carry a
"steadily increasing view" of the race or a rolling
window. Audit of
`agents/policy_network.py::PPOTransformerPolicy` (lines
1216–1235) confirmed the transformer is EXPLICITLY a
rolling-window model: its "hidden_state" slot is
repurposed as a buffer of the last
`transformer_ctx_ticks` fused embeddings, with
`ctx_ticks ∈ {32, 64, 128}` and default 32. Anything
older is unrecoverable to the model.

Scale check: real-data races average ~150–250 ticks
(e.g. 2026-03-31 had 9 268 ticks / 39 races = 238
ticks/race). At default 32, the transformer sees ~13 %
of a race; at the max 128, ~54 %. Across races the
buffer rolls past any boundary the same as any tick
boundary — no inter-race memory.

The LSTM variants (`ppo_lstm_v1`, `ppo_time_lstm_v1`)
do NOT have this problem — their hidden state is
initialised once per rollout at
`agents/ppo_trainer.py:936` and threaded through every
step of the full training day.

Decision for this plan: **widen the arch's allowed
`ctx_ticks` set from `{32, 64, 128}` to
`{32, 64, 128, 256}` and pin at 256** for the probe
(all cohorts). 256 covers the full race for the median
case (~238 ticks). The widening is purely a
range-enumeration edit — `position_embedding` and
`causal_mask` already size off the gene value; no
architectural machinery needs touching. Session 01
handles the widening + a build-and-forward smoke test.
Session 03 pins at 256. Longer races (300+ ticks, e.g.
Wolverhampton multi-race parades) still truncate at the
earliest ticks; this is a strict improvement on 128,
not a full-race guarantee for every race. See
`hard_constraints.md` §14a–§14d and §24. If
transformers STILL underperform LSTMs on C1/C4 after
this fix, the follow-on is architectural (raise the
ceiling beyond 256, or drop the transformer from the
arch mix for the scale run).

Compute cost of the raise: transformer attention is
O(n²). ctx=128 → 256 is ~4× attention FLOPs per step.
On a 3090 with `d_model=256` and `depth ∈ {1,2,3}` the
absolute cost stays small; expected +10–20 % wall-clock
for transformer-only agents, negligible for LSTM
cohorts.

Not addressed here but worth noting: the POSITION_DIM
obs carries current exposure but NOT entry prices for
existing bets — "I'm long at price 5.0 vs price 3.0"
is something the LSTM has to remember from when the
bet was placed. If C4 still fails despite everything
in this plan, entry-price-in-obs is a strong next
candidate (schema bump; lives in
`observation-space-audit` if that plan opens).

## 2026-04-21 · Pre-plan · `arb-curriculum-probe` "crash" was a race, not a crash

The 277bbf49 probe was reported as having "crashed" at
the gen_0→gen_1 re-evaluation with a VCRUNTIME140 fault.
On inspection the fault lines were from an unrelated
April-11 run; the current probe actually completed two
generations successfully. The plan status was set to
`failed` by `TrainingWorker._check_dead_thread` racing
with `_AsyncBridgeQueue.put_nowait`'s
`call_soon_threadsafe` delivery of the terminal
`run_complete` event: thread dies first, watchdog polls
before main loop processes the event, marks plan failed
and clears `_active_plan_id`, then `_handle_event` fires
but can no longer auto-continue because the plan id is
gone. Fix: watchdog now requires two consecutive polls
of (thread-dead + running=True) before marking failed,
giving the scheduled callback time to run. Guard against
confusing old log entries with current-run symptoms when
triaging plan failures.

## 2026-04-21 · Pre-plan · Inherit BC, force-close, and warmup — don't re-derive them

`plans/arb-curriculum/` scoped BC pretrain carefully
(per-agent, never shared; signal + arb_spread heads only;
separate optimiser) and the curriculum ordering (density
first). Those decisions landed, were validated as
working in the 277bbf49 probe (BC worked on 66/66
agents, density ordering was uniformly applied), and
should NOT be reopened here. This plan only adds three
new mechanisms on top of the already-landed arb-
curriculum scaffolding.

## 2026-04-21 · Pre-plan · Post-BC penalty shape punishes exploration

Observed in the 277bbf49 Validation: 7 agents ended the
probe with positive cumulative cash P&L, but only 1
agent had any episode with positive `total_reward`. The
failure shape was consistent: large positive `raw_pnl`
dominated by larger negative `shaped_bonus` driven by
`efficiency_cost × bet_count` and the centred
`(precision − 0.5) × precision_bonus` term. Post-BC the
policy is confident on oracle targets but hasn't yet
calibrated precision or learned to gate by high-
confidence opportunities — so it explores at
exploration-level bet counts and mediocre precision.
Both penalties are calibrated for late-training,
disciplined policies. Applying them at full strength
from ep 1 teaches the agent to stop betting. Hence the
shaped-penalty warmup in this plan's Session 02.

## 2026-04-21 · Pre-plan · Naked variance dominates the training signal

The 277bbf49 single C4-passing agent (2ac21f95) reached
+31.51 peak `total_reward` via 95 naked outcomes and 0
close_signal uses — a directional accident, not
controlled arbing. Agents that DID run controlled arbs
(5b93d7b3 with arb rate 9 %→35 % trajectory, +£2 544
cumulative cash P&L) were outscored in the validator
because naked winners at `naked_loss_scale = 1.0` move
reward by ±£100s per race while the matured-arb bonus is
capped at ±£10. The reward signal tracks luck, not
skill. Hence force-close in this plan's Session 01 — not
because nakedness is always wrong, but because its
variance dominates everything else we're trying to
teach.
