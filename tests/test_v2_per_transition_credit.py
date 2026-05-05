"""Tests for Phase 9 Session 01 — per-transition mature_prob credit.

Covers the two pure-helper deliverables:

- :class:`PairOpenRecord` + :func:`collect_pair_open_records_from_step`
  — collector-side filter that turns newly placed bets into
  open-step records, deduping passive second legs and skipping
  agent close legs.
- :func:`assign_per_transition_labels` — turns a list of
  ``PairOpenRecord`` + the rollout's matched bets into
  ``(mature_label, mature_mask)`` arrays whose only ``mask=True``
  positions are the OPEN steps.

The trainer wiring + a real-rollout integration test are S02 / S03
deliverables. These tests exercise the label-assignment logic
directly with synthetic ``Bet`` fixtures so the algebra is anchored
without spinning up an env.
"""

from __future__ import annotations

import numpy as np
import torch

from agents_v2.action_space import DiscreteActionSpace
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from env.bet_manager import Bet, BetSide
from training_v2.discrete_ppo.aux_labels import (
    PairOpenRecord,
    PerRunnerAuxLabels,
    assign_per_transition_labels,
    collect_pair_open_records_from_step,
    compute_per_runner_aux_labels,
)
from training_v2.discrete_ppo.trainer import DiscretePPOTrainer
from training_v2.discrete_ppo.transition import RolloutBatch


# ── Fixture helper (mirrors test_v2_aux_heads._make_bet shape) ───────────


def _make_bet(
    *,
    selection_id: int,
    side: BetSide,
    matched_stake: float = 5.0,
    average_price: float = 3.0,
    market_id: str = "M",
    pair_id: str | None = "P",
    force_close: bool = False,
    close_leg: bool = False,
) -> Bet:
    return Bet(
        selection_id=int(selection_id),
        side=side,
        requested_stake=float(matched_stake),
        matched_stake=float(matched_stake),
        average_price=float(average_price),
        market_id=str(market_id),
        pair_id=pair_id,
        force_close=bool(force_close),
        close_leg=bool(close_leg),
    )


def _matured_pair(*, pair_id: str, selection_id: int) -> list[Bet]:
    """BACK + LAY both matched, no force_close — the strict-mature class."""
    return [
        _make_bet(
            selection_id=selection_id, side=BetSide.BACK,
            average_price=3.0, pair_id=pair_id,
        ),
        _make_bet(
            selection_id=selection_id, side=BetSide.LAY,
            average_price=2.9, pair_id=pair_id,
        ),
    ]


def _force_closed_pair(
    *, pair_id: str, selection_id: int,
) -> list[Bet]:
    """Open + force-closed close leg → strict-mature negative class."""
    return [
        _make_bet(
            selection_id=selection_id, side=BetSide.BACK,
            average_price=3.0, pair_id=pair_id,
        ),
        _make_bet(
            selection_id=selection_id, side=BetSide.LAY,
            average_price=4.0, pair_id=pair_id,
            force_close=True, close_leg=True,
        ),
    ]


def _naked_pair(*, pair_id: str, selection_id: int) -> list[Bet]:
    """Only the open leg matched — naked → strict-mature negative class."""
    return [
        _make_bet(
            selection_id=selection_id, side=BetSide.BACK,
            average_price=3.0, pair_id=pair_id,
        ),
    ]


# Two-race fixture: race-1 uses market "R1" with selection 100 on slot 0;
# race-2 uses market "R2" with selection 200 on slot 1. Slot indices
# differ across races deliberately — the per-transition path must not
# rely on a single global slot mapping.
_TWO_RACE_RUNNER_MAP = {
    "R1": {100: 0, 101: 1},
    "R2": {200: 0, 201: 1},
}
_MAX_RUNNERS = 4
_N_STEPS = 200


# ── assign_per_transition_labels ──────────────────────────────────────────


def test_matured_pair_label_1_at_open_step():
    """Two-race rollout, race-1 opens a pair on slot 0 at step 40 that
    matures cleanly. Step 40 receives ``label = 1.0`` and ``mask = True``.
    """
    legs = _matured_pair(pair_id="p1", selection_id=100)
    records = [
        PairOpenRecord(pair_id="p1", step_index=40, runner_slot=0),
    ]
    label, mask = assign_per_transition_labels(records, legs, _N_STEPS)
    assert label[40] == 1.0
    assert mask[40] is np.True_ or mask[40] == True  # noqa: E712


def test_force_closed_pair_label_0_at_open_step():
    """Pair opened at step 40, second leg has ``force_close=True`` →
    step 40 carries ``label = 0.0`` and ``mask = True`` (the OPEN
    decision is still where credit lands; the outcome class is just
    negative).
    """
    legs = _force_closed_pair(pair_id="p1", selection_id=100)
    records = [
        PairOpenRecord(pair_id="p1", step_index=40, runner_slot=0),
    ]
    label, mask = assign_per_transition_labels(records, legs, _N_STEPS)
    assert label[40] == 0.0
    assert mask[40]


def test_naked_pair_label_0_at_open_step():
    """Pair opened at step 40, only one leg ever matched → naked →
    step 40 carries ``label = 0.0`` and ``mask = True`` (same shape
    as the force-closed case but a different outcome class).
    """
    legs = _naked_pair(pair_id="p1", selection_id=100)
    records = [
        PairOpenRecord(pair_id="p1", step_index=40, runner_slot=0),
    ]
    label, mask = assign_per_transition_labels(records, legs, _N_STEPS)
    assert label[40] == 0.0
    assert mask[40]


def test_non_open_steps_have_mask_false():
    """200-step rollout, one pair opened at step 40 → the only mask
    position that's True is index 40. Every other tick gets BCE skipped.
    """
    legs = _matured_pair(pair_id="p1", selection_id=100)
    records = [
        PairOpenRecord(pair_id="p1", step_index=40, runner_slot=0),
    ]
    label, mask = assign_per_transition_labels(records, legs, _N_STEPS)
    assert mask.sum() == 1
    assert mask[40]
    # Off-step labels are 0.0 (the placeholder; trainer must not consume).
    assert label[39] == 0.0
    assert label[41] == 0.0


def test_multiple_pairs_same_step_max_label():
    """Two pairs opened at the same step, one matures + one goes naked
    → the step takes the ``max`` label = 1.0. Mirrors the env case
    where two runners are signalled on the same tick.
    """
    legs: list[Bet] = []
    legs.extend(_matured_pair(pair_id="p_mat", selection_id=100))
    legs.extend(_naked_pair(pair_id="p_naked", selection_id=101))
    records = [
        PairOpenRecord(pair_id="p_mat", step_index=40, runner_slot=0),
        PairOpenRecord(pair_id="p_naked", step_index=40, runner_slot=1),
    ]
    label, mask = assign_per_transition_labels(records, legs, _N_STEPS)
    assert label[40] == 1.0
    assert mask[40]


def test_close_legs_not_tracked():
    """``collect_pair_open_records_from_step`` must skip ``close_leg=True``
    bets — agent close_signal legs are an outcome, not an OPEN
    decision. The close tick stays unmasked downstream of
    ``assign_per_transition_labels`` because no record points there.
    """
    seen: set[str] = set()

    # Step 40 — agent places aggressive open of pair "p1".
    open_leg = _make_bet(
        selection_id=100, side=BetSide.BACK, pair_id="p1",
        market_id="R1",
    )
    open_records = collect_pair_open_records_from_step(
        [open_leg],
        step_index=40,
        market_to_runner_map=_TWO_RACE_RUNNER_MAP,
        max_runners=_MAX_RUNNERS,
        seen_pair_ids=seen,
    )
    assert len(open_records) == 1
    assert open_records[0].pair_id == "p1"
    assert open_records[0].step_index == 40
    assert open_records[0].runner_slot == 0

    # Step 100 — agent fires close_signal on the same pair. The close
    # leg has the same pair_id but ``close_leg=True``; the collector
    # must not turn this into a new record.
    close_leg = _make_bet(
        selection_id=100, side=BetSide.LAY, pair_id="p1",
        close_leg=True, average_price=2.9, market_id="R1",
    )
    close_records = collect_pair_open_records_from_step(
        [close_leg],
        step_index=100,
        market_to_runner_map=_TWO_RACE_RUNNER_MAP,
        max_runners=_MAX_RUNNERS,
        seen_pair_ids=seen,
    )
    assert close_records == []

    # End-to-end: feed the OPEN record (no close record) plus the full
    # bet history into the assignment function and confirm step 100
    # carries no mask. Pair has 2 legs but the one we want to label
    # is the OPEN at step 40, not the close at step 100.
    label, mask = assign_per_transition_labels(
        open_records + close_records,
        [open_leg, close_leg],
        _N_STEPS,
    )
    assert mask[40]
    assert not mask[100]
    # The pair classifies as ``label = 1.0`` (matured-or-agent-closed —
    # close_leg=True without force_close is the agent-closed case, in
    # the positive class per CLAUDE.md §"mature_prob_head feeds
    # actor_head"). That label lands at the OPEN step, not the close
    # step.
    assert label[40] == 1.0
    assert label[100] == 0.0


def test_empty_rollout_no_pairs():
    """Rollout with no opens → all-zero label, all-False mask, no crash.

    ``n_steps`` is positive (not a degenerate empty episode); the
    point of the test is that the no-records path doesn't divide by
    zero or raise.
    """
    label, mask = assign_per_transition_labels([], [], _N_STEPS)
    assert label.shape == (_N_STEPS,)
    assert mask.shape == (_N_STEPS,)
    assert not mask.any()
    assert (label == 0.0).all()


# ── Bonus collector-side coverage (passive-leg dedup) ────────────────────


def test_passive_second_leg_does_not_create_second_record():
    """The aggressive leg of a pair lands on the OPEN tick; the passive
    leg of the same pair_id may match many ticks later. The collector
    must dedupe on ``pair_id`` so each pair has at most one record.
    """
    seen: set[str] = set()
    aggressive = _make_bet(
        selection_id=100, side=BetSide.BACK, pair_id="p1",
        average_price=3.0, market_id="R1",
    )
    passive = _make_bet(
        selection_id=100, side=BetSide.LAY, pair_id="p1",
        average_price=2.9, market_id="R1",
    )

    open_records = collect_pair_open_records_from_step(
        [aggressive],
        step_index=40,
        market_to_runner_map=_TWO_RACE_RUNNER_MAP,
        max_runners=_MAX_RUNNERS,
        seen_pair_ids=seen,
    )
    assert len(open_records) == 1
    later_records = collect_pair_open_records_from_step(
        [passive],
        step_index=70,
        market_to_runner_map=_TWO_RACE_RUNNER_MAP,
        max_runners=_MAX_RUNNERS,
        seen_pair_ids=seen,
    )
    assert later_records == []


# ── Trainer-level integration tests (Phase 9 S02) ────────────────────────


class _StubBM:
    """Minimal :class:`env.bet_manager.BetManager` stand-in for tests."""

    def __init__(self, bets: list[Bet]) -> None:
        self.bets = bets


class _StubEnv:
    """Minimal :class:`env.betfair_env.BetfairEnv` stand-in.

    The trainer's per-transition path reads
    ``env._settled_bets`` + ``env.bet_manager.bets`` to classify pair
    outcomes. Tests construct one of these with the bets that should
    be visible to ``assign_per_transition_labels``.
    """

    def __init__(
        self,
        settled_bets: list[Bet] | None = None,
        live_bets: list[Bet] | None = None,
    ) -> None:
        self._settled_bets = list(settled_bets or [])
        self.bet_manager = _StubBM(list(live_bets or []))


class _StubShim:
    """Minimal shim used by trainer tests that don't run a real rollout."""

    def __init__(
        self,
        space: DiscreteActionSpace,
        env: _StubEnv | None = None,
        obs_dim: int = 16,
    ) -> None:
        self.action_space = space
        self.max_runners = space.max_runners
        self.obs_dim = obs_dim
        self.env = env or _StubEnv()


def _build_trainer_with_stub(
    *,
    hp: dict,
    env: _StubEnv | None = None,
    obs_dim: int = 16,
    hidden_size: int = 8,
    seed: int = 0,
    mini_batch_size: int = 8,
    ppo_epochs: int = 1,
) -> DiscretePPOTrainer:
    """Construct a trainer at fixed seed with a stub shim+env.

    ``ppo_epochs=1`` keeps the test cheap; the per-transition logic
    is exercised once per epoch so a single epoch suffices to catch
    regressions.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    space = DiscreteActionSpace(max_runners=4)
    shim = _StubShim(space, env=env, obs_dim=obs_dim)
    policy = DiscreteLSTMPolicy(
        obs_dim=obs_dim, action_space=space, hidden_size=hidden_size,
    )
    trainer = DiscretePPOTrainer(
        policy=policy, shim=shim, hp=hp, device="cpu",
        ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size,
        learning_rate=1e-4,
        # Synthetic obs / random advantages drive ratio approx_kl very
        # high on the first step; lift the threshold so the per-epoch
        # mini-batch loop completes and the n_mature_targets counter
        # reflects every mini-batch, not just the one before early-stop.
        kl_early_stop_threshold=1e9,
    )
    return trainer


def _make_synthetic_batch(
    *,
    trainer: DiscretePPOTrainer,
    n_steps: int = 32,
    pair_open_records: list[PairOpenRecord] | None = None,
    aux_labels: PerRunnerAuxLabels | None = None,
    seed: int = 1,
) -> RolloutBatch:
    """Synthesise a minimal RolloutBatch to drive ``_update_from_batch``."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    R = trainer.max_runners
    action_n = trainer.action_space.n
    obs_dim = trainer.shim.obs_dim
    hidden = trainer.policy.hidden_size
    num_layers = trainer.policy.num_layers

    obs = np.random.randn(n_steps, obs_dim).astype(np.float32)
    mask = np.ones((n_steps, action_n), dtype=bool)
    action_idx = np.zeros(n_steps, dtype=np.int64)  # all NOOP — sidesteps stake
    stake_unit = np.full(n_steps, 0.5, dtype=np.float32)
    log_prob_action = np.full(n_steps, -1.0, dtype=np.float32)
    log_prob_stake = np.zeros(n_steps, dtype=np.float32)
    value_per_runner = np.random.randn(n_steps, R).astype(np.float32) * 0.1
    per_runner_reward = np.random.randn(n_steps, R).astype(np.float32) * 0.1
    done = np.zeros(n_steps, dtype=bool)
    done[-1] = True

    h = torch.zeros((n_steps, num_layers, 1, hidden), dtype=torch.float32)
    c = torch.zeros((n_steps, num_layers, 1, hidden), dtype=torch.float32)

    if aux_labels is None:
        aux_labels = PerRunnerAuxLabels(
            fill_label=np.zeros(R, dtype=np.float32),
            mature_label=np.zeros(R, dtype=np.float32),
            risk_label=np.full(R, np.nan, dtype=np.float32),
            runner_mask=np.zeros(R, dtype=bool),
            risk_mask=np.zeros(R, dtype=bool),
        )

    return RolloutBatch(
        obs=obs,
        hidden_state_in=(h, c),
        mask=mask,
        action_idx=action_idx,
        stake_unit=stake_unit,
        log_prob_action=log_prob_action,
        log_prob_stake=log_prob_stake,
        value_per_runner=value_per_runner,
        per_runner_reward=per_runner_reward,
        done=done,
        n_steps=n_steps,
        aux_labels=aux_labels,
        pair_open_records=pair_open_records,
    )


def _run_one_update(
    trainer: DiscretePPOTrainer,
    batch: RolloutBatch,
    *,
    update_seed: int = 100,
):
    """Drive ``_update_from_batch`` at a fixed seed → return EpisodeStats."""
    torch.manual_seed(update_seed)
    np.random.seed(update_seed)
    return trainer._update_from_batch(
        batch=batch, last_info={}, t0=0.0,
    )


# ── Test 8: byte-identity when disabled ───────────────────────────────────


def test_per_slot_path_byte_identical_when_disabled():
    """Hard_constraints.md §6 — load-bearing regression guard.

    With ``per_transition_credit=False`` the trainer's gradient path
    must be byte-identical to the Phase 7 baseline: feeding the same
    batch (with or without ``pair_open_records`` populated) at the
    same seed must produce identical update statistics.

    The strict version of the guard: a batch carrying populated
    ``pair_open_records`` is silently ignored when the flag is off.
    Catches a future refactor that accidentally consumes the records
    on the disabled path (e.g. unconditional ``assign_per_transition_labels``
    call before the flag check).
    """
    hp = {
        "fill_prob_loss_weight": 0.5,
        "mature_prob_loss_weight": 0.5,
        "risk_loss_weight": 0.0,
        "per_transition_credit": False,
    }

    # Trainer A — batch with pair_open_records=None (legacy shape).
    trainer_a = _build_trainer_with_stub(hp=hp, seed=42)
    batch_a = _make_synthetic_batch(
        trainer=trainer_a, pair_open_records=None, seed=7,
    )
    stats_a = _run_one_update(trainer_a, batch_a, update_seed=99)

    # Trainer B — same seed/shape but batch carries populated records.
    # Per_transition_credit=False ⇒ records must be ignored.
    trainer_b = _build_trainer_with_stub(hp=hp, seed=42)
    populated_records = [
        PairOpenRecord(pair_id="p1", step_index=5, runner_slot=0),
        PairOpenRecord(pair_id="p2", step_index=10, runner_slot=1),
    ]
    batch_b = _make_synthetic_batch(
        trainer=trainer_b, pair_open_records=populated_records, seed=7,
    )
    stats_b = _run_one_update(trainer_b, batch_b, update_seed=99)

    assert stats_a.policy_loss_mean == stats_b.policy_loss_mean
    assert stats_a.value_loss_mean == stats_b.value_loss_mean
    assert stats_a.approx_kl_mean == stats_b.approx_kl_mean
    assert stats_a.mature_prob_bce_mean == stats_b.mature_prob_bce_mean
    # And the diagnostic confirms the disabled path is in effect.
    assert stats_a.per_transition_credit_active is False
    assert stats_b.per_transition_credit_active is False
    assert stats_a.n_mature_targets == 0
    assert stats_b.n_mature_targets == 0


# ── Test 9: n_mature_targets == 0 for naked-only rollout ──────────────────


def test_n_mature_targets_is_zero_for_naked_only_rollout():
    """When the rollout has no opens (or every open already settled
    elsewhere), ``n_mature_targets=0`` even with the flag on. The
    losses must remain finite — no NaN from division-by-zero.
    """
    hp = {
        "fill_prob_loss_weight": 0.0,
        "mature_prob_loss_weight": 0.5,
        "risk_loss_weight": 0.0,
        "per_transition_credit": True,
    }
    trainer = _build_trainer_with_stub(hp=hp, seed=42)
    batch = _make_synthetic_batch(
        trainer=trainer, pair_open_records=[], seed=7,
    )
    stats = _run_one_update(trainer, batch)
    assert stats.per_transition_credit_active is True
    assert stats.n_mature_targets == 0
    assert np.isfinite(stats.mature_prob_bce_mean)
    assert np.isfinite(stats.policy_loss_mean)
    assert np.isfinite(stats.value_loss_mean)


# ── Test 10: n_mature_targets > 0 when pairs mature ──────────────────────


def test_n_mature_targets_nonzero_when_pairs_mature():
    """Synthetic rollout with several open records pointing at
    matured pairs — the per-transition path fires and the diagnostic
    counter exceeds zero.
    """
    hp = {
        "fill_prob_loss_weight": 0.0,
        "mature_prob_loss_weight": 0.5,
        "risk_loss_weight": 0.0,
        "per_transition_credit": True,
    }

    # Build a stub env that classifies p1, p2, p3 all as MATURED.
    settled = (
        _matured_pair(pair_id="p1", selection_id=100)
        + _matured_pair(pair_id="p2", selection_id=101)
        + _matured_pair(pair_id="p3", selection_id=100)
    )
    env = _StubEnv(settled_bets=settled, live_bets=[])

    trainer = _build_trainer_with_stub(
        hp=hp, env=env, seed=42, ppo_epochs=2, mini_batch_size=8,
    )
    records = [
        PairOpenRecord(pair_id="p1", step_index=4, runner_slot=0),
        PairOpenRecord(pair_id="p2", step_index=12, runner_slot=1),
        PairOpenRecord(pair_id="p3", step_index=20, runner_slot=0),
    ]
    batch = _make_synthetic_batch(
        trainer=trainer, n_steps=32, pair_open_records=records, seed=7,
    )
    stats = _run_one_update(trainer, batch)

    assert stats.per_transition_credit_active is True
    # 3 opens × 2 epochs = 6 expected target hits across the update.
    # Each open transition lands in exactly one mini-batch per epoch.
    assert stats.n_mature_targets > 0
    assert stats.n_mature_targets == 3 * trainer.ppo_epochs


# ── Test 11: BCE targets concentrated, not broadcast ─────────────────────


def test_bce_targets_concentrated_not_broadcast():
    """The whole point of per-transition credit is that the BCE label
    lands on a TINY fraction of the rollout's transitions — not on
    every tick (purpose.md §"The fix"). With 5 opens in a 256-step
    rollout the masked-entry total across the update should be
    ``5 × ppo_epochs`` (each open appears once per epoch's shuffled
    mini-batch pass), and definitely much smaller than the ~256 ×
    ppo_epochs entries the per-slot path would touch.
    """
    hp = {
        "fill_prob_loss_weight": 0.0,
        "mature_prob_loss_weight": 0.5,
        "risk_loss_weight": 0.0,
        "per_transition_credit": True,
    }
    settled = (
        _matured_pair(pair_id="p1", selection_id=100)
        + _matured_pair(pair_id="p2", selection_id=101)
        + _matured_pair(pair_id="p3", selection_id=100)
        + _matured_pair(pair_id="p4", selection_id=101)
        + _matured_pair(pair_id="p5", selection_id=100)
    )
    env = _StubEnv(settled_bets=settled, live_bets=[])
    trainer = _build_trainer_with_stub(
        hp=hp, env=env, seed=42, ppo_epochs=2, mini_batch_size=16,
    )
    n_steps = 256
    records = [
        PairOpenRecord(pair_id="p1", step_index=10, runner_slot=0),
        PairOpenRecord(pair_id="p2", step_index=50, runner_slot=1),
        PairOpenRecord(pair_id="p3", step_index=120, runner_slot=0),
        PairOpenRecord(pair_id="p4", step_index=180, runner_slot=1),
        PairOpenRecord(pair_id="p5", step_index=240, runner_slot=0),
    ]
    batch = _make_synthetic_batch(
        trainer=trainer, n_steps=n_steps, pair_open_records=records, seed=7,
    )
    stats = _run_one_update(trainer, batch)

    n_pairs_opened = len(records)
    expected_total = n_pairs_opened * trainer.ppo_epochs
    # Each open lands in exactly one mini-batch per epoch (the shuffled
    # permutation has no duplicates), so the count should equal the
    # expected total. We allow a tiny slack only for the prompt's
    # "shouldn't happen but guard for it" note about boundary splits.
    assert stats.n_mature_targets == expected_total
    # The whole point: << total transitions touched by the per-slot
    # path. A per-slot BCE would consume mb_size × ppo_epochs ×
    # (n_steps // mb_size) = n_steps × ppo_epochs entries.
    per_slot_total = n_steps * trainer.ppo_epochs
    assert stats.n_mature_targets * 20 < per_slot_total  # ≥ 20× concentration


def test_records_skip_unpaired_and_unmapped_bets():
    """``pair_id is None`` and unknown ``market_id`` both → no record.

    Defensive guards against legacy unpaired bets and missing
    runner-map entries — neither has a meaningful pair-level outcome
    to credit.
    """
    seen: set[str] = set()
    unpaired = _make_bet(
        selection_id=100, side=BetSide.BACK, pair_id=None,
        market_id="R1",
    )
    unknown_market = _make_bet(
        selection_id=999, side=BetSide.BACK, pair_id="p_unknown",
        market_id="UNKNOWN",
    )
    records = collect_pair_open_records_from_step(
        [unpaired, unknown_market],
        step_index=10,
        market_to_runner_map=_TWO_RACE_RUNNER_MAP,
        max_runners=_MAX_RUNNERS,
        seen_pair_ids=seen,
    )
    assert records == []
    assert seen == set()
