"""Throughput-fix Session 02 — batched cohort forward regression guards.

Five tests, mirroring the spec in
``plans/rewrite/phase-3-followups/throughput-fix/session_prompts/
02_batched_cohort_forward.md`` §5.

The first two are the load-bearing correctness guards. The remaining
three exercise structural invariants: active-set bookkeeping,
architecture clustering, and the N=1 reduction to the Session 01
single-agent path.

Note on the design (c) fallback. The Session 02 prompt's preferred
shape was vmap-based design (b) — clustering by architecture and
running one batched forward per cluster via
``torch.func.functional_call``. PyTorch 2.11 does not implement a
batching rule for ``aten::lstm.input``, so vmap fails on
``DiscreteLSTMPolicy.forward``. The implementation falls back to
design (c) — per-agent forward in a Python loop within
``BatchedRolloutCollector``. Per-agent self-parity (test #1) and
N=1 byte-identity (test #5) are preserved by construction since
each agent's forward uses the same op order as solo. The speed
benefit is bounded by what Session 01's deferred-sync pattern
already achieved per agent — see this plan's findings.md for the
verdict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from env.betfair_env import BetfairEnv

from agents_v2.discrete_policy import DiscreteLSTMPolicy
from tests.test_betfair_env import _make_day
from training_v2.discrete_ppo.batched_rollout import (
    BatchedRolloutCollector,
    cluster_agents_by_arch,
)
from training_v2.discrete_ppo.rollout import RolloutCollector


REPO_ROOT = Path(__file__).resolve().parents[1]
SCORER_DIR = REPO_ROOT / "models" / "scorer_v1"


def _scorer_runtime_available() -> tuple[bool, str]:
    if not (SCORER_DIR / "model.lgb").exists():
        return False, (
            f"Scorer artefacts missing under {SCORER_DIR}; "
            "run `python -m training_v2.scorer.train_and_evaluate` first."
        )
    try:
        import lightgbm  # noqa: F401
    except Exception as exc:
        return False, f"lightgbm not importable: {exc!r}"
    try:
        import joblib  # noqa: F401
    except Exception as exc:
        return False, f"joblib not importable: {exc!r}"
    return True, ""


_runtime_ok, _runtime_reason = _scorer_runtime_available()


def _scalping_config(max_runners: int = 4) -> dict:
    return {
        "training": {
            "max_runners": max_runners,
            "starting_budget": 100.0,
            "max_bets_per_race": 20,
            "scalping_mode": True,
            "betting_constraints": {
                "max_back_price": 50.0,
                "max_lay_price": None,
                "min_seconds_before_off": 0,
                "force_close_before_off_seconds": 0,
            },
        },
        "actions": {"force_aggressive": True},
        "reward": {
            "early_pick_bonus_min": 1.2,
            "early_pick_bonus_max": 1.5,
            "early_pick_min_seconds": 300,
            "efficiency_penalty": 0.01,
            "commission": 0.05,
            "mark_to_market_weight": 0.0,
        },
    }


def _build_solo(seed: int = 0, n_races: int = 2, hidden_size: int = 32):
    """Build one (env, shim, policy, RolloutCollector) under the given seed."""
    from agents_v2.env_shim import DiscreteActionShim

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = BetfairEnv(
        _make_day(n_races=n_races, n_pre_ticks=10, n_inplay_ticks=2),
        _scalping_config(),
    )
    shim = DiscreteActionShim(env)
    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=hidden_size,
    )
    collector = RolloutCollector(shim=shim, policy=policy, device="cpu")
    return env, shim, policy, collector


def _build_batched_cluster(
    seeds: list[int],
    n_races: int = 2,
    hidden_size: int = 32,
):
    """Build a list of (shim, policy) pairs and a BatchedRolloutCollector.

    Each agent is constructed under its own ``torch.manual_seed`` so
    initial weights are deterministic per-seed (mirroring solo's
    construction in ``_build_solo``).
    """
    from agents_v2.env_shim import DiscreteActionShim

    shims = []
    policies = []
    for s in seeds:
        torch.manual_seed(int(s))
        np.random.seed(int(s))
        env = BetfairEnv(
            _make_day(n_races=n_races, n_pre_ticks=10, n_inplay_ticks=2),
            _scalping_config(),
        )
        shim = DiscreteActionShim(env)
        policy = DiscreteLSTMPolicy(
            obs_dim=shim.obs_dim,
            action_space=shim.action_space,
            hidden_size=hidden_size,
        )
        shims.append(shim)
        policies.append(policy)

    collector = BatchedRolloutCollector(
        shims=shims, policies=policies, device="cpu", seeds=seeds,
    )
    return shims, policies, collector


# ── Test 1: per-agent self-parity (batched-vs-solo) ─────────────────────────


@pytest.mark.slow
@pytest.mark.skipif(not _runtime_ok, reason=_runtime_reason)
@pytest.mark.timeout(120)
def test_per_agent_self_parity_batched_vs_solo():
    """Running agent A inside a batch of N produces transitions
    bit-identical to running A alone at the same seed.

    Build a 4-agent cluster with seeds [42, 43, 44, 45]. Run via
    BatchedRolloutCollector and keep agent 0's transition list.
    Then build a solo agent on the same synthetic day with the
    same weight-init seed and re-seed the global RNG before the
    rollout so solo's state-at-rollout matches the per-agent
    state captured by ``BatchedRolloutCollector(seeds=[42, ...])``.

    Bit-identity is preserved by construction in design (c): each
    agent's forward in the batched path is the same call as solo,
    and the per-agent RNG save/restore inside
    ``BatchedRolloutCollector`` makes the random samples
    indistinguishable from solo's torch.manual_seed(42)-driven
    sequence.
    """
    seeds = [42, 43, 44, 45]
    _shims, _policies, batched = _build_batched_cluster(seeds=seeds)
    transitions_per_agent = batched.collect_episode_batch()
    transitions_batched_0 = transitions_per_agent[0]

    # Solo uses the same weight-init seed (42) AND re-seeds global
    # RNG immediately before the rollout, so its state at start of
    # ``collect_episode`` equals what BatchedRolloutCollector
    # captured for agent 0 (``torch.manual_seed(42)`` then
    # ``torch.get_rng_state()``).
    _env_solo, _shim_solo, _policy_solo, solo = _build_solo(seed=42)
    torch.manual_seed(42)
    transitions_solo = solo.collect_episode()

    assert len(transitions_solo) > 0
    assert len(transitions_batched_0) == len(transitions_solo), (
        f"step count mismatch: batched={len(transitions_batched_0)} "
        f"solo={len(transitions_solo)}"
    )
    for t, (tb, ts) in enumerate(zip(transitions_batched_0, transitions_solo)):
        assert tb.action_idx == ts.action_idx, (
            f"tick {t}: action_idx batched={tb.action_idx!r} "
            f"solo={ts.action_idx!r}"
        )
        assert tb.stake_unit == ts.stake_unit, (
            f"tick {t}: stake_unit batched={tb.stake_unit!r} "
            f"solo={ts.stake_unit!r}"
        )
        assert tb.log_prob_action == ts.log_prob_action, (
            f"tick {t}: log_prob_action batched={tb.log_prob_action!r} "
            f"solo={ts.log_prob_action!r}"
        )
        assert tb.log_prob_stake == ts.log_prob_stake, (
            f"tick {t}: log_prob_stake batched={tb.log_prob_stake!r} "
            f"solo={ts.log_prob_stake!r}"
        )
        assert np.array_equal(tb.value_per_runner, ts.value_per_runner), (
            f"tick {t}: value_per_runner batched={tb.value_per_runner!r} "
            f"solo={ts.value_per_runner!r}"
        )


# ── Test 2: per-agent RNG independence in a batch ───────────────────────────


@pytest.mark.slow
@pytest.mark.skipif(not _runtime_ok, reason=_runtime_reason)
@pytest.mark.timeout(180)
def test_per_agent_rng_independence_in_batch():
    """Switching agent 0's seed must NOT change agents 1-3's actions.

    Two batched runs over 4 agents. Run 1: seeds [42, 43, 44, 45].
    Run 2: seeds [99, 43, 44, 45]. Agents 1-3 must produce
    bit-identical transition lists across the two runs (their
    private RNG state and policy weights are unchanged); agent 0
    diverges by construction.

    The save/restore mechanism in :class:`BatchedRolloutCollector`
    is the load-bearing primitive — agent i's tick reads/writes
    only ``per_agent_cpu_states[i]`` and never leaks across
    agents.
    """
    _s_a, _p_a, batched_a = _build_batched_cluster(seeds=[42, 43, 44, 45])
    transitions_a = batched_a.collect_episode_batch()

    _s_b, _p_b, batched_b = _build_batched_cluster(seeds=[99, 43, 44, 45])
    transitions_b = batched_b.collect_episode_batch()

    # Agents 1, 2, 3 unchanged.
    for i in (1, 2, 3):
        assert len(transitions_a[i]) == len(transitions_b[i]), (
            f"agent {i}: step count drift {len(transitions_a[i])} "
            f"vs {len(transitions_b[i])}"
        )
        for t, (ta, tb) in enumerate(zip(transitions_a[i], transitions_b[i])):
            assert ta.action_idx == tb.action_idx, (
                f"agent {i} tick {t}: action drift after agent 0 reseed "
                f"(a={ta.action_idx!r} b={tb.action_idx!r})"
            )
            assert ta.log_prob_action == tb.log_prob_action, (
                f"agent {i} tick {t}: log_prob_action drift after "
                f"agent 0 reseed"
            )
            assert np.array_equal(ta.value_per_runner, tb.value_per_runner), (
                f"agent {i} tick {t}: value_per_runner drift after "
                f"agent 0 reseed"
            )


# ── Test 3: active set shrinks when an agent terminates ─────────────────────


@pytest.mark.slow
@pytest.mark.skipif(not _runtime_ok, reason=_runtime_reason)
@pytest.mark.timeout(120)
def test_active_set_shrinks_when_agent_terminates_mid_batch():
    """When one agent's episode ends earlier than another's, the
    active set shrinks and the ended agent stops consuming forwards.

    Construct a 2-agent cluster where agent 0 sees a short day
    (1 race × small ticks) and agent 1 sees a longer day (3 races).
    Verify that at the end of collection:
    - Agent 0's transition list is shorter than agent 1's.
    - Agent 0's final transition is ``done = True`` and the
      remaining transitions on agent 1 do not appear in agent 0's
      list (the active-set bookkeeping correctly excluded a
      terminated agent from subsequent ticks).
    """
    from agents_v2.env_shim import DiscreteActionShim

    torch.manual_seed(7)
    np.random.seed(7)
    env_short = BetfairEnv(
        _make_day(n_races=1, n_pre_ticks=10, n_inplay_ticks=2),
        _scalping_config(),
    )
    shim_short = DiscreteActionShim(env_short)
    p0 = DiscreteLSTMPolicy(
        obs_dim=shim_short.obs_dim,
        action_space=shim_short.action_space,
        hidden_size=32,
    )

    torch.manual_seed(8)
    np.random.seed(8)
    env_long = BetfairEnv(
        _make_day(n_races=3, n_pre_ticks=10, n_inplay_ticks=2),
        _scalping_config(),
    )
    shim_long = DiscreteActionShim(env_long)
    p1 = DiscreteLSTMPolicy(
        obs_dim=shim_long.obs_dim,
        action_space=shim_long.action_space,
        hidden_size=32,
    )

    collector = BatchedRolloutCollector(
        shims=[shim_short, shim_long],
        policies=[p0, p1],
        device="cpu",
        seeds=[7, 8],
    )
    transitions = collector.collect_episode_batch()

    assert len(transitions[0]) > 0
    assert len(transitions[1]) > 0
    assert len(transitions[0]) < len(transitions[1]), (
        "expected agent 0 (1 race) to terminate before agent 1 (3 races); "
        f"got len(0)={len(transitions[0])} len(1)={len(transitions[1])}"
    )
    # Agent 0's last transition is the terminal one.
    assert transitions[0][-1].done is True
    # Intermediate transitions are non-done.
    assert all(t.done is False for t in transitions[0][:-1])
    # Agent 1 also terminates eventually.
    assert transitions[1][-1].done is True


# ── Test 4: cluster_agents_by_arch groups compatible archs ──────────────────


def test_cluster_key_groups_compatible_archs():
    """Agents with different ``hidden_size`` land in different clusters.

    Pure unit test on :func:`cluster_agents_by_arch` — no env, no
    rollout. Constructs a small mix of policies and asserts the
    grouping shape.
    """
    from agents_v2.action_space import DiscreteActionSpace

    action_space = DiscreteActionSpace(max_runners=4)

    p_h32_a = DiscreteLSTMPolicy(
        obs_dim=20, action_space=action_space, hidden_size=32,
    )
    p_h32_b = DiscreteLSTMPolicy(
        obs_dim=20, action_space=action_space, hidden_size=32,
    )
    p_h64 = DiscreteLSTMPolicy(
        obs_dim=20, action_space=action_space, hidden_size=64,
    )
    p_h32_c = DiscreteLSTMPolicy(
        obs_dim=20, action_space=action_space, hidden_size=32,
    )

    clusters = cluster_agents_by_arch([p_h32_a, p_h32_b, p_h64, p_h32_c])
    # Two clusters: hidden=32 (3 agents) and hidden=64 (1 agent).
    assert len(clusters) == 2
    # Find the hidden=32 cluster — its key contains hidden_size=32.
    h32_key = next(k for k in clusters.keys() if k[1] == 32)
    h64_key = next(k for k in clusters.keys() if k[1] == 64)
    assert clusters[h32_key] == [0, 1, 3]  # original order preserved
    assert clusters[h64_key] == [2]


def test_cluster_key_groups_obs_dim_separately():
    """Different ``obs_dim`` is its own cluster axis."""
    from agents_v2.action_space import DiscreteActionSpace

    action_space = DiscreteActionSpace(max_runners=4)
    p_a = DiscreteLSTMPolicy(
        obs_dim=20, action_space=action_space, hidden_size=32,
    )
    p_b = DiscreteLSTMPolicy(
        obs_dim=30, action_space=action_space, hidden_size=32,
    )
    clusters = cluster_agents_by_arch([p_a, p_b])
    assert len(clusters) == 2


def test_batched_collector_rejects_mixed_archs():
    """Constructing a BatchedRolloutCollector across mixed archs errors."""
    from agents_v2.action_space import DiscreteActionSpace
    from agents_v2.env_shim import DiscreteActionShim

    if not _runtime_ok:
        pytest.skip(_runtime_reason)

    torch.manual_seed(0)
    env_a = BetfairEnv(
        _make_day(n_races=1, n_pre_ticks=5, n_inplay_ticks=1),
        _scalping_config(),
    )
    shim_a = DiscreteActionShim(env_a)
    env_b = BetfairEnv(
        _make_day(n_races=1, n_pre_ticks=5, n_inplay_ticks=1),
        _scalping_config(),
    )
    shim_b = DiscreteActionShim(env_b)

    p_h32 = DiscreteLSTMPolicy(
        obs_dim=shim_a.obs_dim,
        action_space=shim_a.action_space,
        hidden_size=32,
    )
    p_h64 = DiscreteLSTMPolicy(
        obs_dim=shim_b.obs_dim,
        action_space=shim_b.action_space,
        hidden_size=64,
    )
    with pytest.raises(ValueError, match="cluster key"):
        BatchedRolloutCollector(
            shims=[shim_a, shim_b],
            policies=[p_h32, p_h64],
            device="cpu",
            seeds=[1, 2],
        )


# ── Test 5: N=1 falls back to Session 01 path ──────────────────────────────


@pytest.mark.slow
@pytest.mark.skipif(not _runtime_ok, reason=_runtime_reason)
@pytest.mark.timeout(600)
def test_train_cluster_batched_runs_end_to_end_on_synthetic_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 2-agent cluster runs train + eval through ``train_cluster_batched``.

    Mirrors ``test_train_one_agent_runs_end_to_end_on_synthetic_data``
    for the batched path. Validates: per-agent :class:`AgentResult`
    is populated, registry round-trips for each, and both agents
    in the cluster share the same architecture cluster key.
    """
    from registry.model_store import ModelStore
    from training_v2.cohort.batched_worker import train_cluster_batched
    from training_v2.cohort.genes import CohortGenes
    from training_v2.cohort import batched_worker as bw_mod
    from training_v2.cohort import worker as worker_mod

    days = {
        "2026-04-23": _make_day(n_races=1, n_pre_ticks=4, n_inplay_ticks=2),
        "2026-04-24": _make_day(n_races=1, n_pre_ticks=4, n_inplay_ticks=2),
    }

    def _fake_load_day(date_str, data_dir):
        return days[date_str]

    monkeypatch.setattr(worker_mod, "load_day", _fake_load_day)
    monkeypatch.setattr(bw_mod, "load_day", _fake_load_day, raising=False)

    base = dict(
        learning_rate=3e-4,
        entropy_coeff=0.01,
        clip_range=0.2,
        gae_lambda=0.95,
        value_coeff=0.5,
        mini_batch_size=32,
        hidden_size=64,
    )
    g0 = CohortGenes(**base)
    g1 = CohortGenes(**base)

    store = ModelStore(
        db_path=tmp_path / "models.db",
        weights_dir=tmp_path / "weights",
        bet_logs_dir=tmp_path / "bet_logs",
    )

    results = train_cluster_batched(
        agent_ids=["batched-0", "batched-1"],
        genes_list=[g0, g1],
        days_to_train=["2026-04-23"],
        eval_day="2026-04-24",
        data_dir=tmp_path,
        device="cpu",
        seeds=[42, 43],
        model_store=store,
        scorer_dir=SCORER_DIR,
        generation=0,
    )

    assert len(results) == 2
    for r in results:
        assert r.architecture_name == "v2_discrete_ppo_lstm_h64"
        assert r.train.n_days == 1
        assert r.train.total_steps > 0
        assert isinstance(r.eval.total_reward, float)
        assert r.eval.eval_day == "2026-04-24"
        assert r.eval.n_steps > 0
        rec = store.get_model(r.model_id)
        assert rec is not None
        assert Path(r.weights_path).exists()

    # The two agents must have different model_ids (per-agent
    # registry rows) but share the architecture name.
    assert results[0].model_id != results[1].model_id


@pytest.mark.slow
@pytest.mark.skipif(not _runtime_ok, reason=_runtime_reason)
@pytest.mark.timeout(120)
def test_batched_collector_falls_back_to_n1_session01_path():
    """At N=1 the batched collector produces transitions byte-identical
    to Session 01's RolloutCollector at the same seed.

    The degenerate-batch case must reduce to Session 01's path —
    no semantic drift introduced by the batched code path. The
    BatchedRolloutCollector at N=1 with ``seeds=None`` skips the
    per-agent RNG save/restore and uses the global RNG, exactly
    like RolloutCollector.
    """
    seed = 7
    _e_solo, _s_solo, _p_solo, solo = _build_solo(seed=seed)
    t_solo = solo.collect_episode()

    # Build a single-agent batched collector at the same seed. We
    # use seeds=None to take the global-RNG path, matching solo's
    # ``torch.manual_seed(seed)`` setup.
    from agents_v2.env_shim import DiscreteActionShim

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = BetfairEnv(
        _make_day(n_races=2, n_pre_ticks=10, n_inplay_ticks=2),
        _scalping_config(),
    )
    shim = DiscreteActionShim(env)
    policy = DiscreteLSTMPolicy(
        obs_dim=shim.obs_dim,
        action_space=shim.action_space,
        hidden_size=32,
    )
    batched = BatchedRolloutCollector(
        shims=[shim], policies=[policy], device="cpu", seeds=None,
    )
    t_batched = batched.collect_episode_batch()[0]

    assert len(t_solo) == len(t_batched)
    assert len(t_solo) > 0
    for i, (tb, ts) in enumerate(zip(t_batched, t_solo)):
        assert tb.action_idx == ts.action_idx, f"tick {i}: action drift"
        assert tb.stake_unit == ts.stake_unit, f"tick {i}: stake drift"
        assert tb.log_prob_action == ts.log_prob_action, (
            f"tick {i}: log_prob_action batched={tb.log_prob_action!r} "
            f"solo={ts.log_prob_action!r}"
        )
        assert tb.log_prob_stake == ts.log_prob_stake, (
            f"tick {i}: log_prob_stake drift"
        )
        assert np.array_equal(tb.value_per_runner, ts.value_per_runner), (
            f"tick {i}: value_per_runner drift"
        )
