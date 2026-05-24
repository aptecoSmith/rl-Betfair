# Candidate architectures to test

Five candidates ranked from minimal change → biggest change. Each
gets a `candidate_<name>_results.md` writeup after the probe cohort
runs.

## Baseline (no change)

```python
direction_prob_head = nn.Linear(hidden_size, max_runners)
# input: lstm_last (batch, hidden_size)
# output: (batch, max_runners) logits
```

The architecture in master at this plan's creation.

Phase-15 cohort 1779613306 (Stopped after agent 1) saw this stay
at the pos-weighted uniform-0.5 random floor for 16 days. The
backbone probe (`tools/backbone_signal_probe.py`) showed
`lstm_last` carries ~0% direction signal.

## Candidate 1: residual obs path

**The simplest possible fix.**

```python
# Slice the 12 dir_* obs columns per runner: shape (batch, max_runners, 12)
# Flatten to (batch, max_runners * 12 = 168) for max_runners=14.
dir_obs = extract_dir_columns(obs)  # (batch, max_runners * 12)
direction_input = torch.cat([lstm_last, dir_obs], dim=-1)
direction_prob_head = nn.Linear(hidden_size + max_runners * 12, max_runners)
```

* Input dim: `hidden_size + max_runners * 12` (= 424 for
  hidden_size=256, max_runners=14)
* Output dim: max_runners
* Param overhead: ~+59k weights per side (back+lay)
* Adds 168 features that bypass the backbone's policy-optimised
  squashing.

**Hypothesis:** the head will descend BCE because it now has
direct access to the same raw features the 11-19% logreg used.

## Candidate 2: per-runner residual obs path

Same idea but each runner's head reads only ITS OWN runner's 12
columns:

```python
# Per-runner: lstm_last + that runner's 12 obs columns + the runner_emb
# = hidden_size + 12 + runner_embed_dim per slot
# Then a small per-runner MLP (shared weights across runners).
direction_prob_head = nn.Sequential(
    nn.Linear(hidden_size + 12 + runner_embed_dim, head_hidden),
    nn.ReLU(),
    nn.Linear(head_hidden, 1),  # one logit per runner
)
# Applied per slot, output stacked to (batch, max_runners).
```

Cleaner inductive bias: the head for runner i shouldn't care about
runner j's direction columns. Less risk of cross-runner leakage in
the linear weights.

**Hypothesis:** equivalent or slightly better than C1; lower
overall parameters because shared per-runner weights.

## Candidate 3: separate mini-LSTM for direction

A dedicated tiny LSTM that reads ONLY the per-tick predictor obs
columns through time. Direction head reads its hidden state, not
the main backbone's.

```python
# Per-tick predictor block (across all runners): 12 * max_runners = 168
# Mini-LSTM: hidden 64, 1 layer
direction_lstm = nn.LSTM(input_size=12 * max_runners, hidden_size=64, num_layers=1)
direction_prob_head = nn.Linear(64, max_runners)
# Forward:
#   direction_hidden = direction_lstm(predictor_columns_over_ctx)[:, -1, :]
#   direction_logits = direction_prob_head(direction_hidden)
```

Maximum decoupling: PPO can't reshape this LSTM because no value /
policy gradient flows through it. Only the aux BCE gradient
trains it.

**Hypothesis:** strongest signal preservation, but slowest to
train (LSTM needs to learn temporal patterns from scratch on a
narrow input).

## Candidate 4: detached backbone + raw obs

```python
direction_input = torch.cat([lstm_last.detach(), obs_dir_columns], dim=-1)
direction_prob_head = nn.Linear(hidden_size + max_runners * 12, max_runners)
```

Same as C1 but breaks the gradient flow from the BCE loss back
into the backbone. The aux head's gradient stays local to the
head; PPO is the sole optimiser of the backbone.

**Hypothesis:** removes one source of head/policy gradient
conflict. May ALSO help (or hurt — the auxiliary loss can no
longer pull the backbone toward direction-relevant features even
when that would help the actor).

## Candidate 5: deeper head + residual obs

```python
direction_prob_head = nn.Sequential(
    nn.Linear(hidden_size + max_runners * 12, 128),
    nn.GELU(),
    nn.Linear(128, 64),
    nn.GELU(),
    nn.Linear(64, max_runners),
)
```

Same input as C1, but more head capacity for non-linear
combinations of the predictor columns. Could capture interactions
between back/lay quantiles + champion p_win that a single linear
layer misses.

**Hypothesis:** if C1 isn't enough capacity, C5 will close the
gap. Risk: overfitting on the smaller probe-cohort training
budget.

## Tie-breaker rules

If two candidates pass §7 within rounding error, prefer the
SIMPLER architecture (C1 > C2 > C5 > C4 > C3). Simpler =
fewer parameters, fewer non-linearities, easier to reason about.
