# Session prompt — Phase 1, Session 02: discrete policy + per-runner value + smoke test

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

Implement the discrete-action policy class with masked categorical
action head, optional small continuous heads (stake, arb_spread),
and a **per-runner value head**. Wire it through Session 01's shim
and run an end-to-end smoke test on a real day's data with random
weights. **No training. No optimiser. No loss function.**

Output:

- `agents_v2/discrete_policy.py` with `BaseDiscretePolicy` +
  `DiscreteLSTMPolicy`. (Transformer / TimeLSTM variants OPTIONAL
  — ship LSTM as the baseline, file the others as Phase 1 follow-on
  if time-constrained.)
- `agents_v2/smoke_test.py` — runs 1000 env steps end-to-end via
  the shim, asserts the success-bar conditions, writes one
  `episodes.jsonl` row.
- Tests in `tests/test_agents_v2_discrete_policy.py` and
  `tests/test_agents_v2_smoke.py`.
- Findings writeup at
  `plans/rewrite/phase-1-policy-and-env-wiring/findings.md`.

## What you need to read first

1. `plans/rewrite/README.md` — rewrite plan overview, hard
   constraints.
2. `plans/rewrite/phase-1-policy-and-env-wiring/purpose.md` —
   locked architecture: discrete-categorical action head with
   masking, two small continuous heads (stake, arb_spread —
   default-off per Phase 0 finding that arb_ticks=20 dominates),
   per-runner value head outputting `R^max_runners`.
3. `plans/rewrite/phase-1-policy-and-env-wiring/session_01_findings.md`
   — Session 01's shim contract. Read the action-space size
   formula and obs_dim formula; your policy's input/output
   shapes must match.
4. `plans/rewrite/phase-0-supervised-scorer/findings.md` — the
   scorer's calibrated probability is in the obs (positions
   `[base_dim + 2*i + side_idx]` for runner `i` × side ∈
   {back=0, lay=1}). Optional: nothing forces the policy to
   look at those features more than any other obs dim — but
   it'd be a rookie miss to weight-init the input layer in a
   way that ignores them. Default `nn.Linear` init is fine.
5. `agents/policy_network.py` — v1 policy. Use as a reference for
   the LSTM backbone shape and the recurrent-state contract
   (CLAUDE.md "Recurrent PPO: hidden-state protocol on update").
   **Do not import or subclass v1 classes.** Parallel tree (hard
   constraint #4 in the rewrite README).
6. `CLAUDE.md` sections "Recurrent PPO: hidden-state protocol on
   update", "fill_prob feeds actor_head" — the second one is for
   AVOIDING that pattern; v2 reads scorer outputs from obs only.

## What to do

### 1. `agents_v2/discrete_policy.py::BaseDiscretePolicy` (~60 min)

Abstract base. Defines the forward-pass output contract:

```python
@dataclass(frozen=True)
class DiscretePolicyOutput:
    logits: torch.Tensor              # (batch, action_space.n)
    masked_logits: torch.Tensor       # logits with -inf at masked indices
    action_dist: torch.distributions.Categorical
    stake_alpha: torch.Tensor         # (batch,) Beta α
    stake_beta: torch.Tensor          # (batch,) Beta β
    value_per_runner: torch.Tensor    # (batch, max_runners)
    new_hidden_state: tuple[torch.Tensor, ...]  # arch-specific shape


class BaseDiscretePolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_space: DiscreteActionSpace,
        hidden_size: int = 128,
    ) -> None: ...

    def init_hidden(self, batch: int = 1) -> tuple[torch.Tensor, ...]:
        """Per-arch zero hidden state. Same protocol v1 used."""

    def forward(
        self,
        obs: torch.Tensor,                # (batch, ctx, obs_dim)
        hidden_state: tuple[torch.Tensor, ...],
        mask: torch.Tensor | None = None, # (batch, action_space.n)
    ) -> DiscretePolicyOutput: ...

    @staticmethod
    def pack_hidden_states(states: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]: ...
    @staticmethod
    def slice_hidden_states(packed, indices) -> tuple[torch.Tensor, ...]: ...
```

The `pack_hidden_states` / `slice_hidden_states` contract is
exactly what v1 uses for PPO hidden-state batching (CLAUDE.md
"Recurrent PPO: hidden-state protocol on update"). LSTM packs
`(h, c)` along dim=1 (batch axis); transformer packs `(buffer,
valid_count)` along dim=0. Phase 2's PPO update needs this; ship
it now even though we're not training yet.

### 2. `agents_v2/discrete_policy.py::DiscreteLSTMPolicy` (~60 min)

```python
class DiscreteLSTMPolicy(BaseDiscretePolicy):
    """Single-layer LSTM, hidden=128.

    Backbone:  Linear(obs_dim → hidden) → ReLU → LSTM(hidden, hidden)
    Heads:
      - logits_head:        Linear(hidden → action_space.n)
      - stake_alpha_head:   Linear(hidden → 1) → softplus + 1
      - stake_beta_head:    Linear(hidden → 1) → softplus + 1
      - value_head:         Linear(hidden → max_runners)
    """
```

Implementation notes:

- **Mask application:** `masked_logits = logits.masked_fill(~mask, float('-inf'))`,
  then `Categorical(logits=masked_logits)`. Sampling from this
  distribution NEVER returns a masked index. Verify in tests.
- **Beta heads:** `softplus(x) + 1.0` keeps `α, β > 1` so the Beta
  is unimodal (no hairpin pathologies near 0/1). The action
  caller draws `s = Beta(α, β).sample()` then re-scales to
  `[MIN_BET_STAKE, max_stake_cap]` outside the policy.
- **Value head:** plain linear, no activation. Phase 2's GAE
  consumes per-runner returns directly.
- **Hidden state:** `(h, c)` each shape `(num_layers=1, batch,
  hidden_size)`. Same protocol v1 uses.

### 3. `agents_v2/smoke_test.py` (~45 min)

```python
def main(
    *,
    day_path: Path,
    n_steps: int = 1000,
    seed: int = 42,
    out_path: Path = REPO_ROOT / "logs" / "agents_v2_smoke" / "smoke.jsonl",
) -> int:
    """Random-init LSTM policy → shim → env, n_steps steps.

    Asserts:
      1. No exceptions raised.
      2. At least one row in `episodes.jsonl`-style output.
      3. Discrete action histogram covers ALL of NOOP, OPEN_BACK,
         OPEN_LAY, CLOSE at least once.
      4. Action mask never produced an illegal action (refusal
         counter stays at 0).
      5. Scorer prediction matches standalone booster on at least
         one captured (obs, runner, side) tuple — done by
         re-loading the booster + calibrator and re-predicting.
    """
```

CLI: `python -m agents_v2.smoke_test --day data/processed/2026-04-23.parquet`
(or whatever path the implementer prefers).

The script writes one `episodes.jsonl`-shaped row at the end.
The schema can match v1 loosely — the UI adapter is Phase 3, not
Phase 1.

### 4. Tests (~75 min)

`tests/test_agents_v2_discrete_policy.py`:

- `test_forward_shapes` — given `obs_dim=N, max_runners=14`,
  random `(batch=4, ctx=8, N)` input → `logits.shape == (4,
  action_space.n)`, `value_per_runner.shape == (4, 14)`,
  `stake_alpha.shape == (4,)`, `stake_beta.shape == (4,)`.
- `test_masked_categorical_assigns_zero_probability_to_masked_actions`
  — set every other action to masked, sample 1000 times, assert
  no sample lands on a masked index.
- `test_init_hidden_zero` — fresh hidden state is all zeros.
- `test_pack_slice_round_trip_lstm` — pack a list of 4 hidden
  states, slice with `indices=[0, 2]`, assert the result equals
  the original states 0 and 2 (`torch.allclose`).
- `test_backward_produces_gradients_on_all_params` — sum the
  outputs, `.backward()`, assert every `requires_grad=True`
  parameter has a non-None `.grad`. Catches accidentally
  `detach()`'d heads (the Phase 0 lesson from
  `plans/fill-prob-in-actor` applies — gradient must flow
  through the value head AND the action head).
- `test_value_head_outputs_per_runner_not_scalar` — explicit
  shape guard. The whole point of the rewrite is per-runner
  credit; a regression that returned `(batch, 1)` would silently
  collapse Phase 2.

`tests/test_agents_v2_smoke.py`:

- `@pytest.mark.slow` end-to-end smoke matching success bar #1.
- Skip with a `pytest.skip` if `models/scorer_v1/` doesn't exist
  (Phase 0 wasn't run).
- Use a tiny fixture day (1–2 races, ~50 ticks) to keep the
  test under the project-wide 60s timeout. The full 1000-step
  smoke run is the CLI command, not the test.

### 5. Findings writeup (~30 min)

`plans/rewrite/phase-1-policy-and-env-wiring/findings.md`:

- Success bar table (1–5 from purpose.md PASS/FAIL).
- Smoke run output: action histogram across the 1000 steps
  (random policy will distribute roughly uniformly across
  unmasked actions). Note any surprises.
- **Action mask refusal counter** — must be 0. If anything else,
  the shim/policy is producing illegal actions through some path
  the mask doesn't cover.
- Forward-pass wall time on CPU per step (rough — Phase 2 cares
  about PPO-update wall, not this).
- Phase 2 implications: anything in the policy class that you
  expect to need to change once the trainer's KL controller and
  per-runner GAE land. Specifically: does the masking interact
  cleanly with PPO's importance-sampling ratio? (Yes — masked
  log-probs are well-defined as `-inf` and PPO's clip-and-mean
  still works; flag for Phase 2's awareness.)

## Stop conditions

- All 5 success-bar conditions PASS → write findings.md GREEN,
  message operator "Phase 1 GREEN, ready for Phase 2", **stop**.
- Any of bars 1–4 fails → write findings.md, identify which session
  needs revisiting (Session 01 if it's a shim issue, this session
  if it's a policy issue), **stop**.
- Bar 5 (env unchanged) fails → revert env changes immediately,
  file as a Phase −1 follow-on, **stop**.

## Hard constraints

- **No training code.** No optimiser, no loss, no GAE, no value
  loss, no entropy bonus, no PPO clip. The forward pass is the
  whole policy in Phase 1.
- **No env edits.** Same as Session 01.
- **No re-import of v1 classes.** Parallel tree. If you need v1
  as a reference, READ the file, don't import.
- **Per-runner value head is non-negotiable.** A scalar value
  head would force Phase 2 back into v1's credit-assignment
  pathology. If you find a reason it's hard, file as a finding
  and stop — don't ship a scalar fallback.
- **Mask at the logits level, not by post-hoc rejection.** The
  Categorical must see `-inf` on masked actions. Anything else
  (sampling-and-retry, action-validity check after sampling)
  loses entropy in a way Phase 2 will struggle to compensate for.
- **No new shaped rewards** (rewrite hard constraint #5; Phase 1
  doesn't shape rewards anyway, but flag it: the smoke test
  must NOT add a "completion bonus" or anything similar).
- **No hyperparameter search.** Pick `hidden_size=128`,
  `lr=N/A` (no training), `gamma=N/A`. Phase 2 will sweep the
  ones that matter.

## Out of scope

- PPO trainer (Phase 2).
- Per-runner GAE (Phase 2).
- Frontend wiring (Phase 3).
- Transformer / TimeLSTM variants (OPTIONAL in this session;
  file as Phase 1 follow-on if you ship LSTM only).
- Smoke run on a full GA cohort (Phase 3).
- Comparing performance to v1 (Phase 3).
- Profiling forward-pass speed (irrelevant — env step dominates).

## Useful pointers

- `agents_v2/action_space.py`, `agents_v2/env_shim.py` — Session
  01's deliverables; import and use.
- `agents/policy_network.py` — v1 reference for LSTM hidden-state
  protocol. Read, don't import.
- `CLAUDE.md`, sections:
  - "Recurrent PPO: hidden-state protocol on update" — the
    `pack_hidden_states` / `slice_hidden_states` contract.
  - "fill_prob feeds actor_head" — DON'T do this in v2; the
    scorer is in obs, not actor input.
  - "PPO update stability — advantage normalisation" — Phase 2
    will need this; not yours.
- `torch.distributions.Categorical(logits=masked_logits)` —
  PyTorch's masked categorical. Sampling from `-inf` logits is
  well-defined and stable as long as at least one logit is
  finite (no-op is always legal, so this holds).
- `models/scorer_v1/` — Phase 0 artefacts, loaded by the shim.

## Estimate

3–5 hours.

- 60 min: `BaseDiscretePolicy` + abstract contract.
- 60 min: `DiscreteLSTMPolicy` concrete implementation.
- 45 min: `smoke_test.py`.
- 75 min: tests.
- 30 min: findings writeup.

If past 6 hours, stop and check scope. The most likely overrun
is the masked-categorical sample-no-illegal test (PyTorch's
behaviour is documented but the corner cases — all-but-one
masked, all masked — are worth verifying carefully). If the
test infrastructure for the env shim is awkward, file as a
Session 01 finding and ship the policy tests in isolation.
