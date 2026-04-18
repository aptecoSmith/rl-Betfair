# Hard constraints — Naked-Windfall Clip & Training Stability

Non-negotiable rules. Anything that violates one gets rejected in
review before destabilising the next training run.

## Scope

**§1** This plan makes five coordinated changes, described in
`purpose.md`:

1. Reward shape (naked full-cash in raw + 95% winner clip in
   shaped + close bonus in shaped, softener removed).
2. PPO stability (KL early-stop, ratio clamp, per-arch initial
   LR, warmup coverage check).
3. Entropy control (halve `entropy_coefficient` default, reward
   centering).
4. Smoke-test gate (UI tickbox + assertion harness).
5. Full registry reset + activation-plan redraft + relaunch.

Anything NOT in that list is out of scope — bugfix, refactor,
"while we're here" cleanup all get routed to a separate plan
folder. Examples explicitly out of scope: matcher changes
(`env/exchange_matcher.py`), action/obs schema bumps, gene-range
edits, GA selection pressure changes, new shaped terms beyond
the close-bonus, pair-sizing changes.

**§2** No new shaped-reward terms beyond the 95% naked-winner
clip and the per-close bonus. Any "let's also add a …" idea gets
routed to a separate plan. The point of this plan is to land a
single coherent fix, not to layer five co-dependent tweaks.

## Reward semantics

**§3** The `raw + shaped ≈ total_reward` invariant (CLAUDE.md →
"Reward function: raw vs shaped") MUST hold post-fix. The
pre-existing test
`test_invariant_raw_plus_shaped_equals_total_reward` in
`tests/test_forced_arbitrage.py` stays green. If it fails after
Session 01, the per-pair naked term has leaked across the
channels — fix the plumbing, don't relax the test.

**§4** Raw channel reports actual cash. After this plan's
Session 01b refinement, `info["raw_pnl_reward"]` for a race
equals:

```
race_pnl
```

— i.e. the full whole-race cashflow: `scalping_locked_pnl +
scalping_closed_pnl + sum(per_pair_naked_pnl)`. Every £ that
moved in or out of the wallet lands in raw. The previous
asymmetric hiding of naked winners (`min(0, …)`) moves OUT of
raw and into shaped.

**§4a** (Session 01 landing note.) Session 01's initial draft
set `raw = scalping_locked_pnl + sum(per_pair_naked_pnl)`
which silently excluded `scalping_closed_pnl`. A pair closed
at a loss via `close_signal` would contribute `raw=0,
shaped=+£1 (close bonus), net=+£1` — rewarding the agent for
a trade that actually lost real cash. Session 01b changes
raw to `race_pnl`, making the loss-closed case `raw=−£loss,
shaped=+£1, net=−(loss−1)` — correctly negative. If Session
01 has already landed with the §4-draft formula when Session
01b begins, Session 01b treats §4 as its target spec and
refines the implementation accordingly; if Session 01 lands
directly with the final formula, Session 01b becomes a
docs-and-tests-only commit.

**§5** The 95% naked-winner clip lives in the shaped channel.
Specifically:

```
shaped += −0.95 × sum(max(0, per_pair_naked_pnl))
```

**§6** The per-close bonus lives in the shaped channel.
Specifically:

```
shaped += +1.00 × n_close_signal_successes_this_race
```

A "close_signal success" is a close_signal action that actually
reduced naked exposure (i.e. a pair moved from `incomplete`→
`complete` as a direct consequence). The existing
`scalping_closed_pnl` accumulator in `_settle_current_race` is
the right signal-source — count its underlying pairs, not
close_signal action emissions (which may no-op if there's
nothing to close).

**§7** The "no reward for directional luck" invariant per
CLAUDE.md remains the design intent. Under the new shape, a
random policy that happens to win £100 on a naked receives only
£5 of net reward (raw +100, shaped −95). That's small enough
that the invariant holds *in expectation over the training
signal*, even though raw is no longer asymmetrically masking
winners. This is a deliberate shift — raw reports actual cash,
shaped neutralises the training incentive for luck. Document
the shift in CLAUDE.md per §15.

**§8** The `locked_pnl` floor (`max(0, min(win, lose))`) stays.
The equal-profit pair-sizing formula (`scalping-equal-profit-sizing`,
commit `f7a09fc`) stays. The per-pair naked aggregation
(`scalping-naked-asymmetry`, commit `d59a507`) stays — Session
01 uses its accessor directly.

## PPO stability

**§9** KL early-stop is applied at PPO-epoch granularity, NOT
mini-batch granularity. After each full epoch sweep of
mini-batches, compute approximate KL across the epoch
(`mean(old_logp - new_logp)`); if it exceeds a threshold
(default `0.03`, literature standard), break out of the
remaining epochs for this rollout. Do NOT break mid-epoch — it
leaves the mini-batches unevenly weighted.

**§10** Ratio clamp is `log_ratio = torch.clamp(new_logp -
old_logp, -20, +20)` applied BEFORE `.exp()`. Numerical
backstop for when KL early-stop hasn't yet caught a runaway
update within the first epoch. Must not change gradients in
the common case (|log_ratio| ≪ 20 under normal PPO updates).

**§11** Per-architecture initial LR. The transformer
(`ppo_transformer_v1`) gets half the LR of the LSTMs. Encoded
in the architecture's default hyperparameters — NOT as an
operator-visible knob. The GA can still mutate LR around the
new midpoint.

**§12** The existing 5-update linear LR warmup
(`agents/ppo_trainer.py:1114`) stays. Session 02 verifies it
fires for all three architectures (transformer, LSTM,
time-LSTM). Extending the warmup to 10 updates is OPTIONAL
defence-in-depth — only ship if the smoke test fails with 5.

## Entropy control

**§13** `entropy_coefficient` default halves from `0.01` to
`0.005`. The GA gene range for entropy coefficient is NOT
changed — only the default that fresh-generation agents start
from. Gene-expressed values for inherited agents carry through
unchanged (reset handles the rest).

**§14** Reward centering subtracts a running-mean baseline from
`total_reward` before advantage computation. The running mean
is per-trainer, exponentially-smoothed (α=0.01), initialised at
0, updated once per rollout. Implementation lands in
`agents/ppo_trainer.py` alongside the existing advantage
normalisation. Centering MUST NOT change advantage ordering
within a rollout — advantages are derived from
`reward - baseline + gamma × V(s') - V(s)`, and the constant
subtraction is a pure translation of returns, so advantages
(after per-mini-batch normalisation) are numerically
equivalent in expectation. A unit test asserts equivalence on
synthetic rollouts (Session 03 deliverable).

## Smoke-test gate

**§15** The smoke-test gate is a UI-controllable pre-flight
for training launch. Default ON. Semantics:

- **Input:** the training-launch request from the UI (same
  payload shape as today).
- **When ON:** backend runs a 2-agent × 3-episode probe
  BEFORE starting the full population. Probe uses one
  transformer and one LSTM, default hyperparameters.
- **Assertions run after probe finishes:**
  - `ep1.policy_loss < 100` on both probe agents
  - `ep3.entropy <= ep1.entropy` on both probe agents
    (monotone non-increasing across the probe)
  - `max(ep1..ep3.arbs_closed) >= 1` on at least one probe
    agent
- **Pass:** full population launches immediately.
- **Fail:** full population does NOT launch. Failure
  diagnostics surface in the UI with a Launch-Anyway override
  button (confirmation modal) and a Re-run-Smoke-Test button.
- **When OFF:** legacy behaviour. Full population launches
  with no probe.

**§16** The smoke-test probe must write its episodes to the
same `logs/training/episodes.jsonl` stream as a normal run, so
the live-training panel can show it. The probe's model_ids are
tagged with a `smoke_test: true` flag in the row so the
learning-curves panel can filter or colour them distinctly
(surface design lives in Session 04's UI work, but the flag is
non-negotiable — downstream tooling needs it).

## Registry reset

**§17** The full registry reset archives:

- `registry/models.db` → `registry/archive_<isodate>Z/models.db`
- `registry/weights/` → `registry/archive_<isodate>Z/weights/`
- `logs/training/episodes.jsonl` →
  `logs/training/episodes.pre-naked-clip-stability-<isodate>.jsonl`

The reset script is `scripts/reset_registry.py` if it exists,
or manual `mv` + database init otherwise. Session 05 checks
first and uses the script if available.

**§18** After reset, activation plans are redrafted via the
same JSON-edit pattern used twice before
(`scalping-naked-asymmetry` Session 02,
`policy-startup-stability` Session 02):
`status='draft'`, `started_at=None`, `completed_at=None`,
`current_generation=None`, `current_session=0`, `outcomes=[]`
on all four of `activation-A-baseline`, `B-001/010/100`.

**§19** Do NOT delete the archived `models.db` or
`episodes.jsonl`. Keep them in `registry/archive_*` so post-
mortem comparisons remain possible.

## Testing

**§20** Each session commit ships with new tests (numbered in
the session prompts). Full `pytest tests/ -q` MUST be green on
every session commit — not just at plan completion. No
"we'll fix the test in the next session" — that hides
regressions behind the session boundary.

**§21** Frontend `ng test --watch=false` must be green on
Session 04's commit (UI change). Not required on other
sessions.

**§22** The PPO stability session (02) includes a synthetic
high-KL test: fabricate a rollout where the optimal policy
update exceeds the KL threshold; assert the early-stop fires
and subsequent epochs are skipped.

**§23** The reward-shape session (01) includes a worked-example
test covering every row of the `purpose.md` outcome table.
Inputs are hand-authored per-pair naked P&L; the assertion is
on the exact `raw` and `shaped` contributions.

## Reward-scale change protocol

**§24** Per CLAUDE.md and the convention from prior reward
changes, this is a reward-scale change. Session 01's commit
message MUST:

- Name the change in the first line.
- Include a worked numerical example showing old vs new
  contributions for (a) a naked winner and (b) a naked loser.
- State that post-fix scoreboard rows are not directly
  comparable to pre-fix rows.

**§25** CLAUDE.md "Reward function: raw vs shaped" gets a new
dated paragraph. Historical entries (2026-04-15, 2026-04-18)
are preserved. Format matches the existing narrative pattern.

## Cross-session

**§26** Sessions land as separate commits, in order 01 → 05.
Session 05 (registry reset + launch) is a **manual operator
step**, NOT something an agent runs autonomously. The plan
folder's Session 05 prompt is instructional; execution is
operator-gated. Same rule as `scalping-naked-asymmetry`
Session 02.

**§27** If any earlier session fails to land cleanly, later
sessions block. Specifically: if Session 01 fails the
raw+shaped invariant, Session 02 does not start — the
stability fixes are meaningless if the reward signal is
broken.

**§28** Do NOT bundle the re-launch into the Session 05
commit. Session 05 is "archive + reset + docs"; the launch is
a follow-on operator action that writes back into
`progress.md` as a Validation entry.
