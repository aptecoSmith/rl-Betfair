# Session prompt — Phase 4 Session 03: drop per-tick distribution wrapper construction

Use this prompt to open a new session in a fresh context. Self-
contained.

---

## The task

Every tick of the rollout, a fresh `torch.distributions.Beta`
wrapper is constructed for one `.sample()` and one `.log_prob()`
call:

```python
# (rollout.py lines 230-233)
stake_dist = torch.distributions.Beta(
    out.stake_alpha, out.stake_beta,
)
stake_unit_t = stake_dist.sample()              # (1,)
```

`Beta.__init__` does parameter validation, broadcasting checks,
and pre-allocates internal tensors (`_dirichlet`,
`concentration1`, `concentration0`). At 12 k ticks/episode this
is 12 k Python class instantiations + 12 k validations.

Same shape applies to `Categorical` in
`agents_v2/discrete_policy.py::DiscreteLSTMPolicy.forward`
(line 308) — Session 07+ deals with the policy-side instance.
This session is **rollout-side only**: the `Beta` constructed in
`_collect`.

**Replace with a direct functional path on the rollout hot
path.** Two viable shapes:

**Shape A (recommended): disable validation + cache the
wrapper.** `torch.distributions.Distribution.set_default_
validate_args(False)` once at module load disables the per-
init validation. Construct `Beta` with the per-tick
parameters but skip the validation cost; this is
bit-identical to today (the validations were never expected to
fail; they're guards). Lowest-risk win, smallest payoff.

**Shape B: inline the sample formula.** `Beta(α, β).sample()`
under the hood is `_standard_gamma(α) / (_standard_gamma(α) +
_standard_gamma(β))`. Calling `torch._standard_gamma` directly
on the parameter tensors and computing the ratio inline skips
the whole Distribution machinery. **Bit-identity preserved iff
the same RNG sequence is consumed in the same order** — which
holds because `Beta.sample` calls `_standard_gamma(α)` then
`_standard_gamma(β)` in that order.

For `log_prob` at PPO update time, the wrapper STAYS — the
update path computes `Beta(stored_α, stored_β).log_prob(stored_
sample)` once per mini-batch, not once per tick, so its overhead
is negligible. Only the per-tick rollout path is touched here.

End-of-session bar:

1. **CPU bit-identity preserved.** `stake_unit` values match
   pre-Session-03 byte-for-byte on a fixed seed. This is the
   strictest guard in this phase — `Beta.sample` is an
   RNG-consuming op and any reordering shows up immediately.
2. **`log_prob_stake` values bit-identical** at PPO-update time
   (the consumer side reconstructs the wrapper; verify the
   stored tensor inputs are identical).
3. **All pre-existing v2 tests pass on CPU.**
4. **ms/tick measurement** vs Session 02's baseline.
5. **Verdict** GREEN / PARTIAL / FAIL.

## What you need to read first

1. `plans/rewrite/phase-4-restore-speed/purpose.md` and
   `findings.md` (Sessions 01 + 02 rows).
2. `training_v2/discrete_ppo/rollout.py` lines 226–262 — the
   `Beta` construction + `sample` + `log_prob` block.
3. PyTorch source for `Beta`:
   `torch.distributions.beta.Beta`. Know what `__init__` and
   `sample` actually do. The two interesting calls are:
   - `__init__`: stores `concentration1` / `concentration0`,
     creates an internal `_dirichlet = Dirichlet(stack([α, β]))`.
   - `sample`: calls `_dirichlet.sample()` which calls
     `_standard_gamma`.
4. PyTorch's Distribution validation toggle:
   `torch.distributions.Distribution.set_default_validate_args`.
   Read the docstring; confirm "False" disables per-init
   validation globally (worth checking — there's a per-instance
   `validate_args` kwarg too).
5. `tests/test_v2_rollout_*.py` — the existing tests that touch
   `stake_unit` / `log_prob_stake`.

## Implementation sketch (Shape A, recommended)

In `training_v2/discrete_ppo/rollout.py` (or wherever the
collector is constructed):

```python
import torch.distributions
torch.distributions.Distribution.set_default_validate_args(False)
```

Apply once, at module load. The Beta construction in `_collect`
keeps its current form but no longer pays the validation cost.
Document the rationale in a comment so the global toggle isn't
mysterious.

If Shape A doesn't deliver enough win, switch to Shape B:

```python
# (in _collect, replacing lines 230-236)
gamma_a = torch._standard_gamma(out.stake_alpha)
gamma_b = torch._standard_gamma(out.stake_beta)
stake_unit_t = gamma_a / (gamma_a + gamma_b)
stake_unit = float(stake_unit_t.item())

# log_prob inlined too, since we already have the parameters:
if action_uses_stake(self.action_space, action_idx):
    log_prob_stake_t = (
        (out.stake_alpha - 1) * torch.log(stake_unit_t)
        + (out.stake_beta - 1) * torch.log1p(-stake_unit_t)
        - torch.lgamma(out.stake_alpha)
        - torch.lgamma(out.stake_beta)
        + torch.lgamma(out.stake_alpha + out.stake_beta)
    ).detach().squeeze()
else:
    log_prob_stake_t = torch.zeros(...)
```

Shape B reorders one operation (`stack` inside Dirichlet vs
sequential gamma calls) — verify bit-identity is preserved
empirically before shipping. If it isn't, fall back to Shape A
+ a docstring noting why Shape B was rejected.

## Tests to add

In `tests/test_v2_rollout_distributions.py` (new file):

1. `test_stake_sample_bit_identical_to_pre_session_03_on_fixed_seed`
   — capture the full `[stake_unit for tick in episode]` from a
   1-day CPU rollout pre-change. Re-run post-change and assert
   byte-for-byte equality.

2. `test_log_prob_stake_bit_identical_to_pre_session_03_on_fixed_seed`
   — same shape on `log_prob_stake_t` materialised at end-of-
   episode.

3. `test_distribution_validation_disabled_globally` — assert
   `torch.distributions.Distribution._validate_args` (or
   equivalent attribute that the toggle sets) reflects the
   disabled state after the collector module is imported. Catches
   a future regression where someone imports the collector but
   the toggle doesn't take effect.

4. (Shape B only) `test_inline_beta_sample_matches_distributions_beta`
   — at fixed RNG state, the inline gamma-ratio formula produces
   bit-identical samples to `Beta(α, β).sample()` for a sweep of
   α / β values.

## Hard constraints

1. **Bit-identity is the load-bearing correctness guard.** A 1
   bit difference in `stake_unit` propagates through
   `stake_pounds` → `env.step` → bet placement → race outcome →
   reward → next-tick obs → policy forward → next sample. The
   whole episode diverges. Strict equality, not "close".
2. **Disable validation globally only with a docstring.** The
   `set_default_validate_args(False)` toggle is a process-wide
   change. If this is undesirable, switch to per-instance
   `validate_args=False` on the Beta construction (loses the
   speed win on Categorical / other distributions but is more
   surgical). Document the choice.
3. **Don't touch `Categorical` in `discrete_policy.py`.**
   That's Session 07+. Bundling them here makes the bit-
   identity test ambiguous about which change caused which
   regression.
4. **Don't introduce torch ops that weren't already in the hot
   path.** Shape A is the safer choice; Shape B requires
   verifying that `_standard_gamma` is the actual underlying
   call (PyTorch sometimes refactors internal sampling paths
   between minor versions). If in doubt, ship Shape A.

## Deliverables

- `training_v2/discrete_ppo/rollout.py` — Shape A or Shape B.
- Module-level docstring explaining the validation toggle (if
  Shape A) or the inlined formula (if Shape B).
- `tests/test_v2_rollout_distributions.py` (new) with the
  tests above.
- `findings.md` updated with this session's row.
- Commit: `feat(rewrite): phase-4 S03 (GREEN|PARTIAL) - drop
  per-tick Beta wrapper construction`.

## Estimate

~1.5 h for Shape A; ~2.5 h for Shape B (the inline formula
needs the per-element bit-identity sweep). If past 3 h, stop —
fall back to Shape A and document why Shape B was rejected.
