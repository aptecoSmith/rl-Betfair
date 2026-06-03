"""build_policy — the single genome→policy factory (pbt-breeding HC#11).

One constructor for both the worker and the held-out re-eval tool, so a
trained policy and its re-evaluation are bit-for-bit the same module.
Guards: correct dispatch on ``genes.architecture``; the LSTM path is
byte-identical to a direct ``DiscreteLSTMPolicy(...)`` build (so existing
cohorts are unchanged); the factory composes with Step 1 warm-start for
BOTH architectures; ``policy_arch_name`` discriminates the two.
"""

from __future__ import annotations

import pytest
import torch

from agents_v2.action_space import DiscreteActionSpace
from agents_v2.discrete_policy import (
    DiscreteLSTMPolicy,
    DiscreteTransformerPolicy,
)
from agents_v2.policy_factory import build_policy, policy_arch_name
from training_v2.cohort.genes import CohortGenes
from training_v2.cohort.worker import load_warm_start_weights


def _genes(architecture="lstm", hidden_size=64, **kw) -> CohortGenes:
    return CohortGenes(
        learning_rate=3e-4, entropy_coeff=0.01, clip_range=0.2,
        gae_lambda=0.95, value_coeff=0.5, mini_batch_size=64,
        hidden_size=hidden_size, architecture=architecture, **kw,
    )


def _space() -> DiscreteActionSpace:
    return DiscreteActionSpace(max_runners=14)


class TestDispatch:
    def test_builds_lstm(self):
        p = build_policy(
            _genes("lstm", 64), obs_dim=64, action_space=_space(),
        )
        assert isinstance(p, DiscreteLSTMPolicy)
        assert not isinstance(p, DiscreteTransformerPolicy)

    def test_builds_transformer(self):
        p = build_policy(
            _genes("transformer", 128, transformer_depth=2,
                   transformer_heads=8, transformer_ctx_ticks=64),
            obs_dim=64, action_space=_space(),
        )
        assert isinstance(p, DiscreteTransformerPolicy)
        assert p.d_model == 128 and p.depth == 2
        assert p.n_heads == 8 and p.ctx_ticks == 64

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError):
            build_policy(_genes("rnn"), obs_dim=64, action_space=_space())

    @pytest.mark.parametrize("arch,hs,extra", [
        ("lstm", 64, {}),
        ("lstm", 256, {}),
        ("transformer", 64, dict(transformer_heads=4, transformer_depth=1,
                                 transformer_ctx_ticks=32)),
        ("transformer", 256, dict(transformer_heads=8, transformer_depth=3,
                                  transformer_ctx_ticks=128)),
    ])
    def test_forward_runs_for_every_arch_sizing(self, arch, hs, extra):
        p = build_policy(
            _genes(arch, hs, **extra), obs_dim=64, action_space=_space(),
            input_norm=True,
        )
        out = p(torch.randn(2, 64), hidden_state=p.init_hidden(2))
        assert out.logits.shape == (2, _space().n)


class TestLstmPathByteIdentical:
    """The LSTM factory path must build the SAME module as the direct
    constructor (with the same runtime args) — so routing the worker
    through the factory leaves existing cohorts byte-identical."""

    def test_factory_lstm_forward_equals_direct(self):
        sp = _space()
        g = _genes("lstm", 128)
        torch.manual_seed(0)
        direct = DiscreteLSTMPolicy(
            obs_dim=64, action_space=sp, hidden_size=128,
            direction_gate_enabled=False, direction_gate_threshold=0.5,
            mature_prob_open_threshold=0.0, enable_fc_prob_head=False,
            runner_dim=None, frozen_direction_head_path=None,
            input_norm=True,
        )
        torch.manual_seed(0)
        viafac = build_policy(
            g, obs_dim=64, action_space=sp, runner_dim=None,
            input_norm=True,
        )
        # Same seed ⇒ same init ⇒ identical state_dict keys + values.
        assert direct.state_dict().keys() == viafac.state_dict().keys()
        for k in direct.state_dict():
            assert torch.equal(direct.state_dict()[k], viafac.state_dict()[k])


class TestArchName:
    def test_lstm_name_unchanged(self):
        assert policy_arch_name(_genes("lstm", 128)) == \
            "v2_discrete_ppo_lstm_h128"

    def test_transformer_name_encodes_structure(self):
        name = policy_arch_name(_genes(
            "transformer", 256, transformer_depth=3, transformer_heads=8,
            transformer_ctx_ticks=128,
        ))
        assert name == "v2_discrete_ppo_transformer_d256_L3_h8_ctx128"

    def test_two_architectures_get_distinct_names(self):
        a = policy_arch_name(_genes("lstm", 128))
        b = policy_arch_name(_genes("transformer", 128))
        assert a != b


class TestFactoryComposesWithWarmStart:
    """Build → save → build → warm-start, through the factory, for BOTH
    architectures: the forward reproduces (Step 1 ⊕ Step 1b)."""

    @pytest.mark.parametrize("arch,extra", [
        ("lstm", {}),
        ("transformer", dict(transformer_heads=4, transformer_depth=2,
                             transformer_ctx_ticks=32)),
    ])
    def test_warm_start_through_factory(self, tmp_path, arch, extra):
        sp = _space()
        g = _genes(arch, 64, **extra)
        parent = build_policy(g, obs_dim=64, action_space=sp, input_norm=True)
        gg = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for prm in parent.parameters():
                prm.copy_(torch.randn(prm.shape, generator=gg))
        obs = torch.randn(2, 64, generator=torch.Generator().manual_seed(9))
        hid = parent.init_hidden(2)
        parent.eval()
        with torch.no_grad():
            p_out = parent(obs, hidden_state=hid)

        path = tmp_path / f"{arch}.pt"
        torch.save(parent.state_dict(), str(path))
        child = build_policy(g, obs_dim=64, action_space=sp, input_norm=True)
        load_warm_start_weights(child, path)
        child.eval()
        with torch.no_grad():
            c_out = child(obs, hidden_state=hid)
        assert torch.equal(c_out.logits, p_out.logits)
