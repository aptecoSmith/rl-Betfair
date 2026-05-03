"""Phase 4 Session 07 regression guards for ``DiscreteLSTMPolicy._apply_mask``.

Pre-S07 the mask path constructed a fresh
``torch.tensor(float("-inf"), ...)`` per call and used
``torch.where(mask, logits, neg_inf)``. Post-S07 it uses
``logits.masked_fill(~mask, float("-inf"))`` — same numerical result,
no per-call allocation. These tests pin both properties.
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from agents_v2.action_space import DiscreteActionSpace
from agents_v2.discrete_policy import DiscreteLSTMPolicy


def _legacy_apply_mask(
    logits: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Pre-Session-07 mask path. Used as a bit-identity reference."""
    if mask is None:
        return logits
    if mask.shape != logits.shape:
        if mask.dim() == 1 and mask.shape[0] == logits.shape[1]:
            mask = mask.unsqueeze(0).expand_as(logits)
    if mask.dtype != torch.bool:
        mask = mask.bool()
    neg_inf = torch.tensor(
        float("-inf"), dtype=logits.dtype, device=logits.device,
    )
    return torch.where(mask, logits, neg_inf)


def _make_policy(seed: int = 0) -> DiscreteLSTMPolicy:
    torch.manual_seed(seed)
    space = DiscreteActionSpace(max_runners=14)
    return DiscreteLSTMPolicy(
        obs_dim=64, action_space=space, hidden_size=32,
    )


def _mask_inputs():
    """Sweep of (logits, mask) inputs covering the rollout shapes."""
    torch.manual_seed(123)
    cases = []
    space = DiscreteActionSpace(max_runners=14)
    n = space.n

    # Batch=1 (rollout shape), full mask.
    logits1 = torch.randn(1, n)
    mask1 = torch.ones(1, n, dtype=torch.bool)
    mask1[0, 5] = False
    mask1[0, 7] = False
    cases.append((logits1, mask1))

    # Batch=4, half-masked.
    logits2 = torch.randn(4, n)
    mask2 = torch.ones(4, n, dtype=torch.bool)
    for b in range(4):
        for k in range(2, n, 2):
            mask2[b, k] = False
    cases.append((logits2, mask2))

    # 1-D mask broadcast over batch=3.
    logits3 = torch.randn(3, n)
    mask3 = torch.ones(n, dtype=torch.bool)
    mask3[1] = False
    mask3[3] = False
    cases.append((logits3, mask3))

    # Only NOOP legal — the extreme rollout case.
    logits4 = torch.randn(2, n)
    mask4 = torch.zeros(2, n, dtype=torch.bool)
    mask4[:, 0] = True
    cases.append((logits4, mask4))

    # Non-bool mask (int) — exercises the bool() promotion branch.
    logits5 = torch.randn(2, n)
    mask5_bool = torch.ones(2, n, dtype=torch.bool)
    mask5_bool[:, 9] = False
    cases.append((logits5, mask5_bool.to(torch.uint8)))

    return cases


# ── 1. Bit-identity on _apply_mask output ──────────────────────────────────


def test_apply_mask_bit_identical_to_pre_session_07():
    """Per-call output equals the pre-S07 ``torch.where`` form, byte-equal."""
    policy = _make_policy(seed=42)
    for logits, mask in _mask_inputs():
        post = policy._apply_mask(logits, mask)
        ref = _legacy_apply_mask(logits, mask)
        assert torch.equal(post, ref), (
            f"_apply_mask diverged from pre-S07 form for logits "
            f"shape {tuple(logits.shape)} mask shape {tuple(mask.shape)}"
        )


# ── 2. Categorical sample / log_prob bit-identity (consumer-side) ─────────


def test_categorical_sample_bit_identical_to_pre_session_07_on_fixed_rng_state():
    """Sampling from the masked logits is byte-equal across the two forms."""
    policy = _make_policy(seed=7)
    for logits, mask in _mask_inputs():
        post = policy._apply_mask(logits, mask)
        ref = _legacy_apply_mask(logits, mask)

        torch.manual_seed(99)
        samples_post = Categorical(logits=post).sample(torch.Size([16]))
        torch.manual_seed(99)
        samples_ref = Categorical(logits=ref).sample(torch.Size([16]))
        assert torch.equal(samples_post, samples_ref)

        lp_post = Categorical(logits=post).log_prob(samples_post)
        lp_ref = Categorical(logits=ref).log_prob(samples_ref)
        assert torch.equal(lp_post, lp_ref)


# ── 3. No per-call -inf tensor allocation ──────────────────────────────────


def test_neg_inf_tensor_not_allocated_per_call():
    """Patch ``torch.tensor`` and confirm ``_apply_mask`` doesn't call it.

    The pre-S07 form allocated a fresh 0-d ``-inf`` tensor on every
    invocation. Post-S07 ``masked_fill`` accepts a Python float scalar
    so the constructor is never reached on the mask path.
    """
    policy = _make_policy(seed=1)
    space = policy.action_space
    n_calls = 1000

    real_torch_tensor = torch.tensor
    call_count = {"n": 0}

    def counting_tensor(*args, **kwargs):
        call_count["n"] += 1
        return real_torch_tensor(*args, **kwargs)

    logits = torch.randn(1, space.n)
    mask = torch.ones(1, space.n, dtype=torch.bool)
    mask[0, 4] = False

    torch.tensor = counting_tensor  # type: ignore[assignment]
    try:
        for _ in range(n_calls):
            policy._apply_mask(logits, mask)
    finally:
        torch.tensor = real_torch_tensor  # type: ignore[assignment]

    # Pre-S07 this would be 1000 (one per call). Post-S07 it should be 0.
    assert call_count["n"] <= 1, (
        f"_apply_mask called torch.tensor {call_count['n']} times across "
        f"{n_calls} invocations; expected ≤ 1 (one-time init or none)."
    )


# ── 4. Out-of-place: input logits unchanged ────────────────────────────────


def test_masked_fill_does_not_mutate_logits_input():
    """Out-of-place ``masked_fill`` (not ``masked_fill_``)."""
    policy = _make_policy(seed=2)
    space = policy.action_space
    logits = torch.randn(2, space.n)
    logits_clone = logits.clone()

    mask = torch.ones(2, space.n, dtype=torch.bool)
    mask[:, 3] = False
    mask[:, 5] = False

    masked = policy._apply_mask(logits, mask)
    assert torch.equal(logits, logits_clone), (
        "_apply_mask mutated its input — masked_fill_ used instead of masked_fill"
    )
    # And the returned tensor is a different object.
    assert masked.data_ptr() != logits.data_ptr()


# ── 5. dtype / device preservation ─────────────────────────────────────────


def test_apply_mask_preserves_dtype_and_device():
    """fp32 input → fp32 output; CPU input → CPU output."""
    policy = _make_policy(seed=3)
    space = policy.action_space
    logits = torch.randn(1, space.n, dtype=torch.float32)
    mask = torch.ones(1, space.n, dtype=torch.bool)
    mask[0, 6] = False

    masked = policy._apply_mask(logits, mask)
    assert masked.dtype == torch.float32
    assert masked.device == logits.device
    # Also: a masked entry is exactly -inf (not a promoted-then-cast value).
    assert masked[0, 6].item() == float("-inf")
