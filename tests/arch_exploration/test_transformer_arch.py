"""Session 6 — ``ppo_transformer_v1`` architecture.

CPU-only tests covering the transformer policy, its three new genes,
the rolling tick-context buffer, causal masking, the ``arch_change_cooldown``
mechanism in ``PopulationManager.mutate``, and the per-architecture
``learning_rate`` override in ``TrainingPlan``.

No GPU, no training loops — see ``plans/arch-exploration/testing.md``.
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import pytest
import torch
import yaml

from agents.architecture_registry import REGISTRY, create_policy
from agents.policy_network import (
    AGENT_STATE_DIM,
    MARKET_DIM,
    POSITION_DIM,
    RUNNER_DIM,
    VELOCITY_DIM,
    PPOTransformerPolicy,
)
from agents.population_manager import (
    HyperparamSpec,
    PopulationManager,
    parse_search_ranges,
    sample_hyperparams,
)
from training.training_plan import TrainingPlan


MAX_RUNNERS = 12


def _obs_dim(max_runners: int = MAX_RUNNERS) -> int:
    return (
        MARKET_DIM
        + VELOCITY_DIM
        + RUNNER_DIM * max_runners
        + AGENT_STATE_DIM
        + POSITION_DIM * max_runners
    )


def _action_dim(max_runners: int = MAX_RUNNERS) -> int:
    return max_runners * 2


@pytest.fixture
def real_search_ranges() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)["hyperparameters"]["search_ranges"]


def _tiny_hp(
    heads: int = 2,
    depth: int = 1,
    ctx_ticks: int = 32,
    d_model: int = 32,
) -> dict:
    """Tiny CPU-friendly hyperparameter dict for the transformer."""
    return {
        "lstm_hidden_size": d_model,
        "mlp_hidden_size": 16,
        "mlp_layers": 1,
        "transformer_heads": heads,
        "transformer_depth": depth,
        "transformer_ctx_ticks": ctx_ticks,
    }


# ── 1. Registry lookup ──────────────────────────────────────────────────────


def test_transformer_architecture_registered():
    assert "ppo_transformer_v1" in REGISTRY
    cls = REGISTRY["ppo_transformer_v1"]
    assert cls is PPOTransformerPolicy
    assert cls.architecture_name == "ppo_transformer_v1"
    assert cls.description  # non-empty human-readable string

    # The registry factory should return an instance of the transformer
    # class when called with the public name.
    policy = create_policy(
        name="ppo_transformer_v1",
        obs_dim=_obs_dim(),
        action_dim=_action_dim(),
        max_runners=MAX_RUNNERS,
        hyperparams=_tiny_hp(),
    )
    assert isinstance(policy, PPOTransformerPolicy)


# ── 2. Genes sampled and in range ──────────────────────────────────────────


def test_sampler_emits_transformer_genes_in_range(real_search_ranges):
    specs = parse_search_ranges(real_search_ranges)
    rng = random.Random(7)
    seen_heads: set[int] = set()
    seen_depth: set[int] = set()
    seen_ctx: set[int] = set()

    for _ in range(300):
        hp = sample_hyperparams(specs, rng)
        assert hp["transformer_heads"] in (2, 4, 8)
        assert hp["transformer_depth"] in (1, 2, 3)
        assert hp["transformer_ctx_ticks"] in (32, 64, 128)
        seen_heads.add(hp["transformer_heads"])
        seen_depth.add(hp["transformer_depth"])
        seen_ctx.add(hp["transformer_ctx_ticks"])

    # 300 samples of a uniform 3-way choice should exercise every value.
    assert seen_heads == {2, 4, 8}
    assert seen_depth == {1, 2, 3}
    assert seen_ctx == {32, 64, 128}


# ── 3. Instantiation grid + two-call forward pass ──────────────────────────


_HEADS = (2, 4)
_DEPTH = (1, 2)
_CTX = (32, 64)


@pytest.mark.parametrize(
    "heads,depth,ctx_ticks",
    list(itertools.product(_HEADS, _DEPTH, _CTX)),
)
def test_policy_instantiates_and_forwards_twice(heads, depth, ctx_ticks):
    obs_dim = _obs_dim()
    action_dim = _action_dim()
    # d_model must be divisible by nhead. Smallest divisible by 2 and 4
    # that leaves room for runner + market embeddings is 32.
    policy = create_policy(
        name="ppo_transformer_v1",
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=MAX_RUNNERS,
        hyperparams=_tiny_hp(heads=heads, depth=depth, ctx_ticks=ctx_ticks),
    )
    policy.eval()

    # Structural attribute round-trip — proves the gene reached the module.
    assert policy.transformer_heads == heads
    assert policy.transformer_depth == depth
    assert policy.ctx_ticks == ctx_ticks
    assert len(policy.transformer_encoder.layers) == depth

    obs = torch.zeros(2, obs_dim)

    # Call 1: fresh hidden state (None → zero buffer).
    out1 = policy(obs)
    assert out1.action_mean.shape == (2, action_dim)
    assert out1.value.shape == (2, 1)
    buffer1, valid1 = out1.hidden_state
    assert buffer1.shape == (2, ctx_ticks, 32)
    assert valid1.shape == (2,)
    assert torch.equal(valid1, torch.tensor([1, 1]))

    # Call 2: carry the buffer forward — this is what a rollout does.
    out2 = policy(obs, out1.hidden_state)
    assert out2.action_mean.shape == (2, action_dim)
    buffer2, valid2 = out2.hidden_state
    assert buffer2.shape == (2, ctx_ticks, 32)
    assert torch.equal(valid2, torch.tensor([2, 2]))


# ── 4. Causal masking ───────────────────────────────────────────────────────


def test_transformer_respects_causal_mask():
    """Two sequences identical up to index T but differing at T+1 must
    produce identical encoder output at index T (before the differing
    tick can influence earlier positions via attention).
    """
    torch.manual_seed(0)
    obs_dim = _obs_dim()
    action_dim = _action_dim()
    policy = create_policy(
        name="ppo_transformer_v1",
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=MAX_RUNNERS,
        hyperparams=_tiny_hp(heads=4, depth=2, ctx_ticks=32),
    )
    policy.eval()

    # Feed a 4-tick sequence. Positions in the rolling buffer will be
    # [0-pad × 28, tick0, tick1, tick2, tick3].
    seq_a = torch.randn(1, 4, obs_dim)
    seq_b = seq_a.clone()
    # Differ at T+1 (index 3) — the most recent tick.
    seq_b[0, 3] = torch.randn(obs_dim)

    with torch.no_grad():
        enc_a = policy.encode_sequence(seq_a)
        enc_b = policy.encode_sequence(seq_b)

    assert enc_a.shape == (1, 32, 32)

    # Position -2 in the buffer corresponds to tick2 (the "T" position).
    # Causal mask blocks attention to position -1, where the differing
    # tick lives, so encoded outputs at position -2 must be bit-for-bit
    # equal between the two runs. Mild tolerance accounts for layer-norm
    # numerics.
    assert torch.allclose(
        enc_a[:, -2, :], enc_b[:, -2, :], atol=1e-6,
    )
    # Sanity: position -1 (tick3) SHOULD differ between the two runs.
    assert not torch.allclose(
        enc_a[:, -1, :], enc_b[:, -1, :], atol=1e-6,
    )


# ── 5. Rolling-buffer overflow ──────────────────────────────────────────────


def test_rolling_buffer_retains_most_recent_ctx_ticks():
    """Feeding ctx_ticks + 5 steps must not crash, must clamp
    ``valid_count`` at ``ctx_ticks``, and must keep only the most recent
    ``ctx_ticks`` fused fingerprints — the oldest 5 should have rolled
    off the start of the buffer.
    """
    ctx_ticks = 32
    obs_dim = _obs_dim()
    action_dim = _action_dim()
    policy = create_policy(
        name="ppo_transformer_v1",
        obs_dim=obs_dim,
        action_dim=action_dim,
        max_runners=MAX_RUNNERS,
        hyperparams=_tiny_hp(heads=2, depth=1, ctx_ticks=ctx_ticks),
    )
    policy.eval()

    # Known-per-step "fingerprint" inputs. ``float(t+1)`` avoids the
    # all-zero first tick colliding with the buffer's zero padding.
    obs_list = [
        torch.full((1, obs_dim), float(t + 1)) for t in range(ctx_ticks + 5)
    ]

    hidden = None
    with torch.no_grad():
        for o in obs_list:
            out = policy(o, hidden)
            hidden = out.hidden_state

    buffer, valid = hidden
    assert buffer.shape == (1, ctx_ticks, 32)
    assert int(valid.item()) == ctx_ticks  # clamped, not overflowed

    # The first 5 feeds should have rolled off. Verify by re-encoding
    # each obs in isolation and comparing the buffer contents slot-by-slot.
    expected_indices = range(5, ctx_ticks + 5)  # obs_list[5..36]
    with torch.no_grad():
        for buf_idx, obs_idx in enumerate(expected_indices):
            fresh_fused, _ = policy._encode_ticks(obs_list[obs_idx])
            assert torch.allclose(
                buffer[0, buf_idx, :], fresh_fused[0], atol=1e-5
            ), f"buffer slot {buf_idx} should contain fingerprint of obs {obs_idx}"

    # The entries that rolled off (obs_list[0..4]) must NOT still be in
    # the buffer — this catches a "shift but don't drop" regression.
    with torch.no_grad():
        first_fused, _ = policy._encode_ticks(obs_list[0])
    for buf_idx in range(ctx_ticks):
        assert not torch.allclose(
            buffer[0, buf_idx, :], first_fused[0], atol=1e-5
        ), f"obs 0's fingerprint should not still live at slot {buf_idx}"


# ── 6. Architecture cooldown in PopulationManager.mutate ───────────────────


def _mk_population_manager_with_three_arches() -> PopulationManager:
    """Construct a minimal ``PopulationManager`` that knows about the
    three live architectures, so mutation of ``architecture_name`` is
    meaningful.
    """
    config = {
        "population": {"size": 3},
        "training": {
            "architecture": "ppo_lstm_v1",
            "max_runners": MAX_RUNNERS,
        },
        "hyperparameters": {
            "search_ranges": {
                "learning_rate": {
                    "type": "float_log", "min": 1e-5, "max": 5e-4,
                },
                "architecture_name": {
                    "type": "str_choice",
                    "choices": [
                        "ppo_lstm_v1",
                        "ppo_time_lstm_v1",
                        "ppo_transformer_v1",
                    ],
                },
            },
        },
    }
    return PopulationManager(config, model_store=None)


def test_arch_cooldown_blocks_architecture_mutation():
    pm = _mk_population_manager_with_three_arches()

    hp = {
        "learning_rate": 1e-4,
        # Starting from the middle index — both ±1 mutation directions
        # land on a different arch, so mutation_rate=1.0 ALWAYS changes
        # the arch when cooldown is 0. That makes the test unambiguous.
        "architecture_name": "ppo_time_lstm_v1",
        "arch_change_cooldown": 1,
    }
    original_arch = hp["architecture_name"]

    # Cooldown > 0 → arch mutation is blocked this generation.
    pm.mutate(hp, mutation_rate=1.0, rng=random.Random(0))
    assert hp["architecture_name"] == original_arch
    # Cooldown decrements toward zero.
    assert hp["arch_change_cooldown"] == 0

    # Re-arm: with cooldown back at 0 the next mutation should change
    # the arch (same mutation_rate + guaranteed-adjacent str_choice).
    pm.mutate(hp, mutation_rate=1.0, rng=random.Random(0))
    assert hp["architecture_name"] != original_arch
    # A successful arch mutation arms the cooldown for NEXT generation.
    assert hp["arch_change_cooldown"] == 1

    # And the newly armed cooldown blocks another arch change on the
    # very next mutate() call — end-to-end round trip.
    locked_arch = hp["architecture_name"]
    pm.mutate(hp, mutation_rate=1.0, rng=random.Random(1))
    assert hp["architecture_name"] == locked_arch
    assert hp["arch_change_cooldown"] == 0


# ── 7. Planner per-arch learning_rate override ─────────────────────────────


def test_training_plan_arch_specific_lr_range_applied_at_gen0():
    """TrainingPlan may override ``learning_rate`` per architecture.
    Agents of the overridden arch must sample from the override range;
    all other agents fall back to the global range.
    """
    # Global LR range — anywhere below the override's lower bound is
    # guaranteed to be out of the override's [2e-4, 5e-4] window.
    global_lr = {"type": "float_log", "min": 1.0e-5, "max": 1.0e-4}
    override_lr = {"type": "float_log", "min": 2.0e-4, "max": 5.0e-4}

    hp_ranges = {
        "learning_rate": global_lr,
        "architecture_name": {
            "type": "str_choice",
            "choices": ["ppo_lstm_v1", "ppo_transformer_v1"],
        },
        # A couple of filler genes so the transformer policy instantiates
        # correctly -- the planner passes the hp dict to ``create_policy``.
        "lstm_hidden_size": {
            "type": "int_choice", "choices": [32],
        },
        "mlp_hidden_size": {
            "type": "int_choice", "choices": [16],
        },
        "mlp_layers": {"type": "int", "min": 1, "max": 1},
        "transformer_heads": {"type": "int_choice", "choices": [2]},
        "transformer_depth": {"type": "int_choice", "choices": [1]},
        "transformer_ctx_ticks": {"type": "int_choice", "choices": [32]},
    }

    plan = TrainingPlan.new(
        name="arch-lr-override",
        population_size=20,
        architectures=["ppo_lstm_v1", "ppo_transformer_v1"],
        hp_ranges=hp_ranges,
        seed=99,
        arch_mix={"ppo_lstm_v1": 10, "ppo_transformer_v1": 10},
        min_arch_samples=5,
        arch_lr_ranges={"ppo_transformer_v1": override_lr},
    )

    config = {
        "population": {"size": plan.population_size},
        "training": {
            "architecture": "ppo_lstm_v1",
            "max_runners": MAX_RUNNERS,
        },
        "hyperparameters": {"search_ranges": hp_ranges},
    }
    pm = PopulationManager(config, model_store=None)

    agents = pm.initialise_population(generation=0, seed=7, plan=plan)
    assert len(agents) == plan.population_size

    lstm_lrs = [
        a.hyperparameters["learning_rate"]
        for a in agents
        if a.architecture_name == "ppo_lstm_v1"
    ]
    xf_lrs = [
        a.hyperparameters["learning_rate"]
        for a in agents
        if a.architecture_name == "ppo_transformer_v1"
    ]
    assert len(lstm_lrs) == 10
    assert len(xf_lrs) == 10

    # LSTM agents stay inside the global range.
    for lr in lstm_lrs:
        assert global_lr["min"] <= lr <= global_lr["max"], lr

    # Transformer agents come from the override range.
    for lr in xf_lrs:
        assert override_lr["min"] <= lr <= override_lr["max"], lr

    # And importantly, the override range sits ABOVE the global range,
    # so no LSTM value can accidentally look like a transformer value.
    assert max(lstm_lrs) < min(xf_lrs)
