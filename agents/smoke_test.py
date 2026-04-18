"""Smoke-test gate — 2-agent × 3-episode probe before full GA launch.

Session 04 of ``plans/naked-clip-and-stability/``. A pre-flight that
catches the three pathologies Sessions 01–03 fix (policy-loss blow-up,
rising entropy, naked-windfall / arbs-closed collapse) before burning
the eight hours it takes to train 16 agents from scratch.

The public surface is deliberately narrow:

- :func:`evaluate_probe_episodes` — pure assertion evaluator. Takes a
  list of episode rows (same shape as ``logs/training/episodes.jsonl``)
  and returns a :class:`SmokeResult`. Unit-testable without any
  training machinery.
- :func:`run_smoke_test` — orchestrator that creates two fresh probe
  agents (one transformer, one LSTM), trains them for 3 episodes via
  the real ``PPOTrainer`` harness, writes the resulting rows to the
  main ``episodes.jsonl`` stream tagged ``smoke_test: true``, and
  evaluates them. The real launch path calls this; tests mostly call
  ``evaluate_probe_episodes`` directly with fabricated rows.

Assertions (hard_constraints.md §15):

1. ``ep1.policy_loss < 100`` on both probe agents.
2. ``ep3.entropy <= ep1.entropy`` on both probe agents (non-increasing).
3. ``max(ep1..ep3.arbs_closed) >= 1`` on AT LEAST ONE probe agent.

All assertion thresholds live as module-level constants so the GA /
probe runner cannot drift them silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


# Hard-constraint §15 thresholds. Changing any of these breaks the gate
# — do not mutate them per-call.
EP1_POLICY_LOSS_MAX = 100.0
ARBS_CLOSED_MIN = 1
PROBE_EPISODE_COUNT = 3


@dataclass(frozen=True)
class SmokeAssertionResult:
    """Outcome of one of the three gate assertions.

    ``observed`` / ``threshold`` are reported in whichever unit the
    assertion uses (e.g. policy_loss magnitude, entropy delta, arbs
    closed count). ``detail`` is a human-readable line the UI modal
    renders verbatim.
    """

    name: str
    passed: bool
    observed: float
    threshold: float
    detail: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SmokeResult:
    """Aggregate probe outcome.

    ``passed`` is the AND of every ``SmokeAssertionResult.passed`` in
    ``assertions``; the constructor does not validate this — the
    evaluator is responsible for producing consistent output.

    ``probe_model_ids`` lists the ephemeral probe agents so a post-hoc
    reader of ``episodes.jsonl`` can locate the smoke-test rows even if
    the ``smoke_test: true`` tag is dropped by a downstream consumer.
    """

    passed: bool
    assertions: list[SmokeAssertionResult]
    probe_model_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "assertions": [a.to_dict() for a in self.assertions],
            "probe_model_ids": list(self.probe_model_ids),
        }


def _rows_by_agent(rows: Iterable[dict]) -> dict[str, list[dict]]:
    """Group episode rows by ``model_id`` and sort each agent's rows by
    episode index.

    Rows missing ``model_id`` or ``episode`` are dropped — they can't
    participate in an ep1 / ep3 comparison.
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        mid = row.get("model_id")
        ep = row.get("episode")
        if mid is None or ep is None:
            continue
        grouped.setdefault(mid, []).append(row)
    for mid in grouped:
        grouped[mid].sort(key=lambda r: r.get("episode", 0))
    return grouped


def evaluate_probe_episodes(rows: list[dict]) -> SmokeResult:
    """Evaluate the three gate assertions against a list of probe rows.

    Expected ``rows`` shape mirrors ``logs/training/episodes.jsonl``
    (see ``PPOTrainer._log_episode``): each row carries ``episode``,
    ``model_id``, ``policy_loss``, ``entropy``, ``arbs_closed``. Rows
    outside the probe are caller-filtered; this function does not look
    at the ``smoke_test`` tag.

    Returns a :class:`SmokeResult` even when ``rows`` is empty — the
    assertions all fail with ``observed=0`` and ``detail`` naming the
    missing data, so the UI modal has something meaningful to show.
    """
    grouped = _rows_by_agent(rows)
    probe_ids = sorted(grouped.keys())

    assertions: list[SmokeAssertionResult] = []

    # Assertion 1 — ep1.policy_loss < 100 on BOTH agents.
    ep1_losses: list[tuple[str, float]] = []
    for mid in probe_ids:
        first = grouped[mid][0] if grouped[mid] else None
        if first is None or first.get("episode") != 1:
            continue
        ep1_losses.append((mid, float(first.get("policy_loss", 0.0))))
    if not ep1_losses:
        assertions.append(SmokeAssertionResult(
            name="ep1_policy_loss",
            passed=False,
            observed=0.0,
            threshold=EP1_POLICY_LOSS_MAX,
            detail="No ep1 rows found from probe agents.",
        ))
    else:
        worst_mid, worst = max(ep1_losses, key=lambda t: t[1])
        passed = all(v < EP1_POLICY_LOSS_MAX for _, v in ep1_losses)
        assertions.append(SmokeAssertionResult(
            name="ep1_policy_loss",
            passed=passed,
            observed=worst,
            threshold=EP1_POLICY_LOSS_MAX,
            detail=(
                f"ep1 policy_loss: worst = {worst:.4f} "
                f"(agent {worst_mid[:8]}), threshold < {EP1_POLICY_LOSS_MAX}"
            ),
        ))

    # Assertion 2 — ep3.entropy <= ep1.entropy on BOTH agents.
    entropy_diffs: list[tuple[str, float, float, float]] = []  # mid, ep1, ep3, delta
    for mid in probe_ids:
        eps = grouped[mid]
        first = next((r for r in eps if r.get("episode") == 1), None)
        third = next(
            (r for r in eps if r.get("episode") == PROBE_EPISODE_COUNT), None,
        )
        if first is None or third is None:
            continue
        e1 = float(first.get("entropy", 0.0))
        e3 = float(third.get("entropy", 0.0))
        entropy_diffs.append((mid, e1, e3, e3 - e1))
    if not entropy_diffs:
        assertions.append(SmokeAssertionResult(
            name="entropy_non_increasing",
            passed=False,
            observed=0.0,
            threshold=0.0,
            detail=(
                f"Missing ep1 or ep{PROBE_EPISODE_COUNT} entropy rows."
            ),
        ))
    else:
        worst_mid, e1, e3, delta = max(entropy_diffs, key=lambda t: t[3])
        passed = all(d <= 0.0 for _, _, _, d in entropy_diffs)
        assertions.append(SmokeAssertionResult(
            name="entropy_non_increasing",
            passed=passed,
            observed=delta,
            threshold=0.0,
            detail=(
                f"ep3−ep1 entropy: worst Δ = {delta:+.4f} "
                f"(agent {worst_mid[:8]}: {e1:.3f} → {e3:.3f}), "
                f"threshold <= 0"
            ),
        ))

    # Assertion 3 — max arbs_closed >= 1 on AT LEAST ONE probe agent.
    per_agent_max: list[tuple[str, int]] = []
    for mid in probe_ids:
        eps = grouped[mid]
        if not eps:
            continue
        # Only consider the first PROBE_EPISODE_COUNT rows — the probe
        # is sized at 3 episodes; extra rows (e.g. inherited from a
        # partially-recycled model_id) would bias this upward.
        window = [r for r in eps if 1 <= r.get("episode", 0) <= PROBE_EPISODE_COUNT]
        peak = max((int(r.get("arbs_closed", 0)) for r in window), default=0)
        per_agent_max.append((mid, peak))
    if not per_agent_max:
        assertions.append(SmokeAssertionResult(
            name="arbs_closed_any_agent",
            passed=False,
            observed=0.0,
            threshold=float(ARBS_CLOSED_MIN),
            detail="No probe rows found — cannot evaluate arbs_closed.",
        ))
    else:
        best_mid, best = max(per_agent_max, key=lambda t: t[1])
        passed = best >= ARBS_CLOSED_MIN
        assertions.append(SmokeAssertionResult(
            name="arbs_closed_any_agent",
            passed=passed,
            observed=float(best),
            threshold=float(ARBS_CLOSED_MIN),
            detail=(
                f"max arbs_closed across probe: {best} "
                f"(best agent {best_mid[:8]}), threshold >= {ARBS_CLOSED_MIN}"
            ),
        ))

    all_passed = all(a.passed for a in assertions)
    return SmokeResult(
        passed=all_passed,
        assertions=assertions,
        probe_model_ids=probe_ids,
    )


# ── Probe runner ─────────────────────────────────────────────────────
#
# ``run_smoke_test`` is the production caller. Factored as a thin
# orchestration wrapper so unit tests can mock it via dependency
# injection and focus on ``evaluate_probe_episodes``.
#
# The runner delegates the actual training loop to the same
# ``PPOTrainer`` the full run uses — "the probe must reuse the full
# training harness" (session prompt §1). The probe's episode rows land
# in the same ``episodes.jsonl`` file tagged ``smoke_test: true`` so
# the live learning-curves panel can colour them distinctly (§16).


def run_smoke_test(
    config: dict,
    train_days: list,
    *,
    probe_architectures: tuple[str, str] = ("ppo_transformer_v1", "ppo_lstm_v1"),
    n_episodes: int = PROBE_EPISODE_COUNT,
    progress_queue=None,
) -> SmokeResult:
    """Run the 2-agent × 3-episode probe and evaluate the gate.

    The probe intentionally ignores the population size, architecture
    mix, and hyperparameter ranges from the training plan — it always
    runs with one transformer and one LSTM at each architecture's
    default hyperparameters. The point is to catch failures in the
    default code path, not to predict plan-specific behaviour.

    Each probe agent trains for ``n_episodes`` (default 3) episodes
    through the real ``PPOTrainer`` harness. The runner slices
    ``train_days`` down to ``n_episodes`` items — callers supplying
    fewer days get ``ValueError`` rather than a silently-short probe.

    Probe agents are ephemeral: their weights are NOT written to the
    model store. They exist only long enough to produce the
    ``episodes.jsonl`` rows the gate evaluates, then garbage-collect.

    Deferred imports guard the API process — torch and the policy
    registry are avoided at module-load time so ``from agents.smoke_test
    import evaluate_probe_episodes`` stays cheap.
    """
    if len(train_days) < n_episodes:
        raise ValueError(
            f"smoke-test probe needs at least {n_episodes} training days, "
            f"got {len(train_days)}"
        )
    probe_days = list(train_days[:n_episodes])

    # Deferred imports — the API process avoids torch at module-load time.
    from agents.architecture_registry import REGISTRY, create_policy
    from agents.ppo_trainer import PPOTrainer
    from env.betfair_env import BetfairEnv

    probe_model_ids: list[str] = []
    probe_episode_rows: list[dict] = []

    # One representative env is used only to read dimensions for
    # policy construction; the actual training loop creates a fresh
    # BetfairEnv per day inside ``PPOTrainer._rollout`` (see
    # ``agents/ppo_trainer.py:769``). ``BetfairEnv`` takes a single
    # positional ``day``, not a list — reusing the first probe day
    # here is equivalent to how ``PopulationManager`` derives
    # ``obs_dim`` / ``action_dim`` from env constants.
    sample_env = BetfairEnv(probe_days[0], config)
    obs_dim = int(sample_env.observation_space.shape[0])
    action_dim = int(sample_env.action_space.shape[0])
    max_runners = sample_env.max_runners

    for arch_name in probe_architectures:
        if arch_name not in REGISTRY:
            raise ValueError(
                f"smoke-test probe architecture '{arch_name}' not in registry"
            )
        model_id = f"smoke-{arch_name}"
        probe_model_ids.append(model_id)

        # Default hyperparameters only — the per-architecture LR
        # overrides (Session 02) fire automatically because PPOTrainer
        # reads ``type(policy).default_learning_rate`` when
        # ``hyperparams.learning_rate`` is absent.
        policy = create_policy(
            name=arch_name,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_runners=max_runners,
            hyperparams={},
        )
        trainer = PPOTrainer(
            policy=policy,
            config=config,
            hyperparams={},
            progress_queue=progress_queue,
            model_id=model_id,
            architecture_name=arch_name,
        )
        trainer.smoke_test_tag = True  # picked up by _log_episode

        trainer.train(days=probe_days, n_epochs=1)

        # Collect the probe rows by model_id from the shared log.
        rows = _tail_probe_rows(trainer.log_dir / "episodes.jsonl", model_id)
        probe_episode_rows.extend(rows)

    result = evaluate_probe_episodes(probe_episode_rows)
    return SmokeResult(
        passed=result.passed,
        assertions=result.assertions,
        probe_model_ids=probe_model_ids,
    )


def _tail_probe_rows(log_path, model_id: str) -> list[dict]:
    """Read ``episodes.jsonl`` and return rows matching ``model_id``.

    Defensive against a partially-written file at the tail (invalid
    JSON lines are skipped).
    """
    import json
    from pathlib import Path

    path = Path(log_path)
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("model_id") == model_id:
                out.append(row)
    return out
