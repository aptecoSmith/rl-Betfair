"""DiscreteTransformerPolicy — pbt-breeding Step 1b architecture port.

Gates the v2 transformer backbone: it must satisfy the SAME
``BaseDiscretePolicy`` contract as the LSTM (forward shapes, the shared
head stack, strict checkpoint round-trip) AND train end-to-end through
the existing v2 PPO trainer with its ``(buffer, valid_count)`` hidden
state — the sharp edge, since that state's rank-1 ``valid_count`` slot
must survive the rollout collector's pre-allocated capture buffers and
the PPO update's pack/slice helpers (which the transformer restores to
the dim-0 BasePolicy defaults).

The end-to-end ``train_episode`` test reuses the tiny synthetic day from
``test_discrete_ppo_trainer`` so the transformer runs the identical
trainer path the LSTM does.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

from agents_v2.action_space import DiscreteActionSpace
from agents_v2.discrete_policy import (  # noqa: F401
    DiscreteLSTMPolicy,
    DiscreteTransformerPolicy,
    _apply_rope,
    _rope_cos_sin,
    _rotate_half,
)
from training_v2.cohort.worker import load_warm_start_weights

# Reuse the synthetic-day + scalping-config helpers the LSTM trainer
# tests use, so the transformer exercises the identical trainer path.
# (Sibling test modules aren't importable by default — put tests/ on the
# path so ``import test_discrete_ppo_trainer`` resolves.)
sys.path.insert(0, os.path.dirname(__file__))
from test_discrete_ppo_trainer import _make_day, _scalping_config  # noqa: E402


_SIZINGS = [
    # (hidden_size/d_model, n_heads, depth, ctx_ticks)
    (64, 4, 2, 32),
    (128, 8, 1, 64),
    (256, 2, 3, 32),
    (64, 2, 1, 128),
]


def _space() -> DiscreteActionSpace:
    return DiscreteActionSpace(max_runners=14)


class TestTransformerForwardContract:
    @pytest.mark.parametrize("h,nh,d,ctx", _SIZINGS)
    def test_forward_shapes(self, h, nh, d, ctx):
        sp = _space()
        p = DiscreteTransformerPolicy(
            obs_dim=64, action_space=sp, hidden_size=h,
            depth=d, n_heads=nh, ctx_ticks=ctx, input_norm=True,
        )
        out = p(torch.randn(3, 64), hidden_state=p.init_hidden(3))
        assert out.logits.shape == (3, sp.n)
        assert out.value_per_runner.shape == (3, 14)
        assert out.stake_alpha.shape == (3,)
        buf, vc = out.new_hidden_state
        assert buf.shape == (3, ctx, h)
        assert vc.shape == (3,)
        assert vc.dtype == torch.long

    def test_d_model_not_divisible_by_heads_raises(self):
        # 64 % 5 != 0
        with pytest.raises(ValueError):
            DiscreteTransformerPolicy(
                obs_dim=64, action_space=_space(), hidden_size=64,
                n_heads=5,
            )

    def test_checkpoint_round_trips_strict_and_has_no_lstm_keys(self):
        sp = _space()
        p = DiscreteTransformerPolicy(
            obs_dim=64, action_space=sp, hidden_size=128,
            depth=2, n_heads=4, ctx_ticks=32, input_norm=True,
        )
        sd = p.state_dict()
        assert not any(k.startswith("lstm.") for k in sd), \
            [k for k in sd if k.startswith("lstm.")]
        assert any(k.startswith("transformer_encoder.") for k in sd)
        assert any(k.startswith("position_embedding.") for k in sd)
        p2 = DiscreteTransformerPolicy(
            obs_dim=64, action_space=sp, hidden_size=128,
            depth=2, n_heads=4, ctx_ticks=32, input_norm=True,
        )
        p2.load_state_dict(sd, strict=True)  # must not raise

    def test_shares_head_stack_keys_with_lstm(self):
        """The transformer must carry the EXACT same head modules as the
        LSTM (only the backbone differs) — guards the shared-stack claim."""
        sp = _space()
        lstm = DiscreteLSTMPolicy(obs_dim=64, action_space=sp, hidden_size=64)
        tr = DiscreteTransformerPolicy(
            obs_dim=64, action_space=sp, hidden_size=64, n_heads=4,
        )
        head_prefixes = (
            "fill_prob_head.", "mature_prob_head.", "risk_head.",
            "direction_prob_head.", "runner_slot_embedding.",
            "actor_head.", "noop_head.", "stake_alpha_head.",
            "stake_beta_head.", "value_head.", "input_proj.",
        )
        for pre in head_prefixes:
            lk = {k for k in lstm.state_dict() if k.startswith(pre)}
            tk = {k for k in tr.state_dict() if k.startswith(pre)}
            assert lk == tk and lk, f"head {pre} differs: {lk} vs {tk}"


class TestTransformerBufferProtocol:
    """The (buffer, valid_count) state must pack/slice the dim-0 way."""

    def test_pack_slice_round_trip(self):
        sp = _space()
        p = DiscreteTransformerPolicy(
            obs_dim=64, action_space=sp, hidden_size=64, n_heads=4,
            ctx_ticks=32,
        )
        # Collect 5 per-tick (batch=1) states by chaining forwards.
        states = []
        hid = p.init_hidden(1)
        for _ in range(5):
            states.append(tuple(t.clone() for t in hid))
            out = p(torch.randn(1, 64), hidden_state=hid)
            hid = out.new_hidden_state
        packed = p.pack_hidden_states(states)  # dim-0 cat → (5, ...)
        assert packed[0].shape == (5, 32, 64)
        assert packed[1].shape == (5,)
        idx = torch.tensor([0, 2, 4])
        sliced = p.slice_hidden_states(packed, idx)
        assert sliced[0].shape == (3, 32, 64)
        # A forward conditioned on the sliced batch state runs cleanly.
        out = p(torch.randn(3, 64), hidden_state=sliced)
        assert out.logits.shape == (3, sp.n)


class TestTransformerTrainsEndToEnd:
    """The load-bearing gate: a transformer trains through the REAL v2
    PPO trainer (rollout + update) without exception, and its weights
    move — proving the buffer hidden state survives the whole path."""

    def test_train_episode_runs_and_updates_weights(self):
        from agents_v2.env_shim import DiscreteActionShim
        from env.betfair_env import BetfairEnv
        from training_v2.discrete_ppo.trainer import DiscretePPOTrainer

        torch.manual_seed(0)
        np.random.seed(0)
        env = BetfairEnv(
            _make_day(n_races=2, n_pre_ticks=10, n_inplay_ticks=2),
            _scalping_config(),
        )
        shim = DiscreteActionShim(env)
        policy = DiscreteTransformerPolicy(
            obs_dim=shim.obs_dim,
            action_space=shim.action_space,
            hidden_size=32,
            depth=1,
            n_heads=4,
            ctx_ticks=16,
        )
        before = {k: v.clone() for k, v in policy.state_dict().items()
                  if v.dtype.is_floating_point}
        trainer = DiscretePPOTrainer(
            policy=policy, shim=shim, learning_rate=3e-4,
            mini_batch_size=32, ppo_epochs=2, device="cpu",
        )
        stats = trainer.train_episode()  # rollout + PPO update, no raise
        assert stats.n_steps > 0
        # At least one transformer-backbone param moved (gradient flowed
        # through the encoder + the buffer protocol).
        after = policy.state_dict()
        moved = [
            k for k, v in before.items()
            if k.startswith(("transformer_encoder.", "position_embedding.",
                             "input_proj."))
            and not torch.equal(v, after[k])
        ]
        assert moved, "no transformer-backbone weight changed after a PPO update"


class TestTransformerWarmStart:
    """Step 1 warm-start composes with the transformer (Step 1b)."""

    def test_warm_start_reproduces_transformer_forward(self, tmp_path):
        sp = _space()
        parent = DiscreteTransformerPolicy(
            obs_dim=64, action_space=sp, hidden_size=64, n_heads=4,
            ctx_ticks=32, input_norm=True,
        )
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for prm in parent.parameters():
                prm.copy_(torch.randn(prm.shape, generator=g))
            parent.set_input_norm_stats(
                torch.randn(64, generator=g).numpy(),
                (torch.rand(64, generator=g) + 0.5).numpy(),
            )
        obs = torch.randn(3, 64, generator=torch.Generator().manual_seed(1))
        hid = parent.init_hidden(3)
        parent.eval()
        with torch.no_grad():
            p_out = parent(obs, hidden_state=hid)

        path = tmp_path / "tr.pt"
        torch.save(parent.state_dict(), str(path))

        child = DiscreteTransformerPolicy(
            obs_dim=64, action_space=sp, hidden_size=64, n_heads=4,
            ctx_ticks=32, input_norm=True,
        )
        load_warm_start_weights(child, path)
        child.eval()
        with torch.no_grad():
            c_out = child(obs, hidden_state=hid)
        assert torch.equal(c_out.logits, p_out.logits)
        assert torch.equal(c_out.value_per_runner, p_out.value_per_runner)


class TestRoPEMath:
    """The load-bearing correctness signature for rotary positional
    embedding: a wrong rotation / reshape silently degrades the transformer,
    so assert the two defining properties directly."""

    def test_relative_position_invariance(self):
        # <rope(q, m), rope(k, n)> depends ONLY on (m - n). Slide a fixed
        # (q, k) pair across all absolute positions at a fixed gap and the
        # dot product must not move.
        ctx, head_dim, gap = 64, 32, 5
        cos, sin = _rope_cos_sin(ctx, head_dim)
        g = torch.Generator().manual_seed(0)
        qv = torch.randn(head_dim, generator=g)
        kv = torch.randn(head_dim, generator=g)

        def rot(vec, pos):
            v = vec.view(1, 1, 1, head_dim)
            return _apply_rope(
                v, cos[pos:pos + 1], sin[pos:pos + 1],
            ).view(head_dim)

        dots = [
            torch.dot(rot(qv, m), rot(kv, m - gap)).item()
            for m in range(gap, ctx)
        ]
        assert max(dots) - min(dots) < 1e-4, "RoPE not relative-position"

    def test_norm_preserving_and_pos0_identity(self):
        cos, sin = _rope_cos_sin(32, 16)
        x = torch.randn(1, 2, 32, 16, generator=torch.Generator().manual_seed(1))
        xr = _apply_rope(x, cos, sin)
        # A rotation preserves the per-position vector norm.
        assert torch.allclose(
            x.norm(dim=-1), xr.norm(dim=-1), atol=1e-5,
        )
        # Position 0 is a zero-angle rotation -> identity.
        assert torch.allclose(xr[:, :, 0, :], x[:, :, 0, :], atol=1e-6)

    def test_rotate_half_partners_match_duplicated_freqs(self):
        # rotate_half maps dim i -> dim i+head_dim/2, which must be the SAME
        # pair the cat([freqs, freqs]) cos/sin layout rotates together.
        x = torch.arange(8, dtype=torch.float32)
        r = _rotate_half(x.view(1, 8))
        assert torch.equal(r.view(8), torch.tensor(
            [-4.0, -5.0, -6.0, -7.0, 0.0, 1.0, 2.0, 3.0]))

    def test_even_head_dim_required(self):
        with pytest.raises(ValueError):
            _rope_cos_sin(16, 15)  # odd head_dim


class TestRoPETransformer:
    """The RoPE positional-encoding variant of the transformer backbone
    (pbt-gpu-forward task #8)."""

    @pytest.mark.parametrize("h,nh,d,ctx", _SIZINGS)
    def test_forward_shapes(self, h, nh, d, ctx):
        sp = _space()
        p = DiscreteTransformerPolicy(
            obs_dim=64, action_space=sp, hidden_size=h,
            depth=d, n_heads=nh, ctx_ticks=ctx, pos_encoding="rope",
        )
        out = p(torch.randn(3, 64), hidden_state=p.init_hidden(3))
        assert out.logits.shape == (3, sp.n)
        assert out.value_per_runner.shape == (3, 14)
        buf, vc = out.new_hidden_state
        assert buf.shape == (3, ctx, h)

    def test_state_dict_has_rope_layers_not_learned_keys(self):
        p = DiscreteTransformerPolicy(
            obs_dim=64, action_space=_space(), hidden_size=64, depth=2,
            n_heads=4, ctx_ticks=32, pos_encoding="rope",
        )
        keys = list(p.state_dict().keys())
        assert any(k.startswith("rope_layers.") for k in keys)
        assert not any(
            "transformer_encoder" in k or "position_embedding" in k
            for k in keys
        )
        # Deterministic cos/sin tables are non-persistent -> NOT in state_dict.
        assert not any("rope_cos" in k or "rope_sin" in k for k in keys)
        assert len(p.rope_layers) == 2

    def test_ffn_mult_widens_rope_ffn(self):
        p = DiscreteTransformerPolicy(
            obs_dim=64, action_space=_space(), hidden_size=128, depth=1,
            n_heads=4, ctx_ticks=32, pos_encoding="rope", ffn_mult=4,
        )
        assert p.rope_layers[0].ff[0].out_features == 512  # 128 * 4

    def test_warm_start_reproduces_rope_forward(self, tmp_path):
        sp = _space()
        kw = dict(
            obs_dim=64, action_space=sp, hidden_size=64, depth=2, n_heads=4,
            ctx_ticks=32, pos_encoding="rope", input_norm=True,
        )
        parent = DiscreteTransformerPolicy(**kw)
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for prm in parent.parameters():
                prm.copy_(torch.randn(prm.shape, generator=g))
            parent.set_input_norm_stats(
                torch.randn(64, generator=g).numpy(),
                (torch.rand(64, generator=g) + 0.5).numpy(),
            )
        obs = torch.randn(3, 64, generator=torch.Generator().manual_seed(2))
        hid = parent.init_hidden(3)
        parent.eval()
        with torch.no_grad():
            p_out = parent(obs, hidden_state=hid)
        path = tmp_path / "rope.pt"
        torch.save(parent.state_dict(), str(path))
        child = DiscreteTransformerPolicy(**kw)
        load_warm_start_weights(child, path)
        child.eval()
        with torch.no_grad():
            c_out = child(obs, hidden_state=hid)
        assert torch.equal(c_out.logits, p_out.logits)
        assert torch.equal(c_out.value_per_runner, p_out.value_per_runner)

    def test_train_episode_runs_and_updates_rope_weights(self):
        from agents_v2.env_shim import DiscreteActionShim
        from env.betfair_env import BetfairEnv
        from training_v2.discrete_ppo.trainer import DiscretePPOTrainer

        torch.manual_seed(0)
        np.random.seed(0)
        env = BetfairEnv(
            _make_day(n_races=2, n_pre_ticks=10, n_inplay_ticks=2),
            _scalping_config(),
        )
        shim = DiscreteActionShim(env)
        policy = DiscreteTransformerPolicy(
            obs_dim=shim.obs_dim, action_space=shim.action_space,
            hidden_size=32, depth=1, n_heads=4, ctx_ticks=16,
            pos_encoding="rope",
        )
        before = {k: v.clone() for k, v in policy.state_dict().items()
                  if v.dtype.is_floating_point}
        trainer = DiscretePPOTrainer(
            policy=policy, shim=shim, learning_rate=3e-4,
            mini_batch_size=32, ppo_epochs=2, device="cpu",
        )
        stats = trainer.train_episode()  # rollout + PPO update, no raise
        assert stats.n_steps > 0
        after = policy.state_dict()
        moved = [
            k for k, v in before.items()
            if k.startswith(("rope_layers.", "input_proj."))
            and not torch.equal(v, after[k])
        ]
        assert moved, "no rope-backbone weight changed after a PPO update"
