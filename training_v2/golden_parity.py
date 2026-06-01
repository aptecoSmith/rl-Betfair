"""Bit-identical golden-trajectory harness — training-speedup-v2 Step 1.

THE keystone deliverable. Captures the per-tick stream the CURRENT
(canonical, sequential) env produces and gives a comparator that every
later speedup stage is gated on. The whole plan exists because a silent
divergence (``--batched`` dropping BC) sailed in twice; this harness
turns "two weeks in, a basic error" into "the divergence trips the
golden gate the instant it appears".

Design (per hard_constraints #1, #7):

* **Golden = the canonical sequential path** — :class:`RolloutCollector`
  driving the predictors-ON :class:`BetfairEnv`. "Current env is golden"
  (HC#7): when a fast path disagrees, golden is right by definition.
* **Capture on CPU.** The env is pure CPU; the LSTM forward is
  deterministic on CPU. So golden re-capture is bit-identical, which is
  what makes GATE (a) (self-parity) a clean signal and lets the
  per-quantity tolerances be tight.
* **Per-quantity tolerance** distinguishes acceptable float reordering
  from logic drift:
    - discrete quantities (actions, done, bet count, pair_ids,
      side/outcome/force-close/stop/close classifications, settle
      counts) — **EXACT**;
    - continuous (obs, reward, value, log-prob, hidden state, price,
      stake, P&L) — within a declared atol/rtol justified as
      float-reordering only (e.g. a GPU reduction reorders a sum). A
      one-tick logic perturbation (a price off by a tick ≈ 1-5 %, or a
      flipped action) blows past these by orders of magnitude → caught.

Used by ``tests/test_env_golden_parity.py`` (the load-bearing regression
guard) and the per-stage comparators in Steps 2 / 3A / 3B / 3C.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agents_v2.discrete_policy import BaseDiscretePolicy
from agents_v2.env_shim import DiscreteActionShim
from training_v2.discrete_ppo.rollout import RolloutCollector


__all__ = [
    "GoldenStream",
    "Mismatch",
    "TOLERANCES",
    "capture_golden",
    "compare_streams",
    "save_stream",
    "load_stream",
]


# ── Per-quantity tolerances (rtol, atol). Documented justification. ───────
#
# Discrete quantities are NOT here — they are compared with ==.
# GPU-touched continuous quantities (the policy forward's log-probs,
# values, hidden state) get 1e-4, matching the repo's existing
# rollout attribution tolerance (``rollout._ATTRIBUTION_TOLERANCE``).
# Pure-CPU env quantities (obs assembled by numpy, reward, P&L) get a
# tighter atol since no float reordering happens on the same device.
TOLERANCES: dict[str, tuple[float, float]] = {
    # per-tick
    "obs": (0.0, 1e-5),
    "stake_unit": (0.0, 1e-6),
    "log_prob_action": (1e-4, 1e-4),
    "log_prob_stake": (1e-4, 1e-4),
    "value_per_runner": (1e-4, 1e-4),
    "per_runner_reward": (0.0, 1e-4),
    "hidden_in": (1e-4, 1e-4),
    # bet ledger (continuous)
    "bet.requested_stake": (0.0, 1e-4),
    "bet.matched_stake": (0.0, 1e-4),
    "bet.average_price": (0.0, 1e-4),
    "bet.pnl": (0.0, 1e-3),
    "bet.ltp_at_placement": (0.0, 1e-4),
    "bet.fill_prob_at_placement": (0.0, 1e-5),
    "bet.mature_prob_at_placement": (0.0, 1e-5),
    "bet.direction_back_prob_at_placement": (0.0, 1e-5),
    "bet.direction_lay_prob_at_placement": (0.0, 1e-5),
    # day-level info (continuous)
    "info": (0.0, 1e-3),
}

# Bet fields compared EXACTLY (discrete identity / classification).
# NOTE: ``pair_id`` is deliberately ABSENT — it is a random
# ``uuid4().hex[:12]`` (env/betfair_env.py:3739), nondeterministic
# run-to-run by design. What is load-bearing is the pairing STRUCTURE
# (which bets share a pair), compared separately via _pairing_groups.
_BET_DISCRETE = (
    "selection_id", "market_id", "side", "outcome", "tick_index",
    "close_leg", "force_close", "stop_close", "is_each_way",
)
_BET_CONTINUOUS = (
    "requested_stake", "matched_stake", "average_price", "pnl",
    "ltp_at_placement", "fill_prob_at_placement",
    "mature_prob_at_placement", "direction_back_prob_at_placement",
    "direction_lay_prob_at_placement",
)

# Day-level info keys to capture. Discrete counts compared exactly;
# continuous P&L compared within "info" tolerance.
_INFO_DISCRETE = (
    "bet_count", "winning_bets", "races_completed",
    "arbs_completed", "arbs_naked", "arbs_closed",
    "arbs_force_closed", "arbs_stop_closed", "pairs_opened",
    "direction_gate_refusals", "pwin_back_gate_refusals",
    "pwin_lay_gate_refusals",
)
_INFO_CONTINUOUS = (
    "day_pnl", "raw_pnl_reward", "shaped_bonus",
    "locked_pnl", "naked_pnl", "closed_pnl",
    "force_closed_pnl", "stop_closed_pnl",
)


@dataclass
class GoldenStream:
    """Per-tick + end-of-episode snapshot of one (env, policy, seed) run."""

    # provenance (not compared; for fixture identification)
    case: str = ""
    seed: int = 0
    hidden_size: int = 0
    obs_dim: int = 0
    n_steps: int = 0

    # per-tick arrays (length n_steps)
    obs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    action_idx: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    stake_unit: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    log_prob_action: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    log_prob_stake: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    value_per_runner: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    per_runner_reward: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    done: np.ndarray = field(default_factory=lambda: np.zeros((0,), bool))
    hidden_in: tuple[np.ndarray, ...] = ()

    # end-of-episode
    bets: list[dict] = field(default_factory=list)
    info_discrete: dict[str, int] = field(default_factory=dict)
    info_continuous: dict[str, float] = field(default_factory=dict)


# ── Capture ───────────────────────────────────────────────────────────────


def capture_golden(
    shim: DiscreteActionShim,
    policy: BaseDiscretePolicy,
    *,
    seed: int,
    case: str = "",
    device: str = "cpu",
    deterministic: bool = False,
) -> GoldenStream:
    """Run the canonical :class:`RolloutCollector` once and snapshot it.

    The caller owns env/policy construction (so the same builder can make
    golden and candidate envs). Seeding is done HERE, immediately before
    ``collect_episode``, so two captures with the same seed + same policy
    weights + same env are bit-identical (the env is a deterministic
    replay; the only RNG consumer is action/stake sampling).
    """
    torch.manual_seed(int(seed) & 0x7FFFFFFF)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) & 0x7FFFFFFF)

    collector = RolloutCollector(shim=shim, policy=policy, device=device)
    batch = collector.collect_episode(deterministic=deterministic)
    env = shim.env

    hidden_in = tuple(
        np.asarray(t.detach().cpu().numpy(), dtype=np.float32)
        for t in batch.hidden_state_in
    )

    bets = [_bet_to_dict(b) for b in env.all_settled_bets]

    info = collector.last_info or {}
    info_discrete = {k: int(info.get(k, 0)) for k in _INFO_DISCRETE if k in info}
    info_continuous = {
        k: float(info.get(k, 0.0)) for k in _INFO_CONTINUOUS if k in info
    }

    return GoldenStream(
        case=case,
        seed=int(seed),
        hidden_size=int(getattr(policy, "hidden_size", 0)),
        obs_dim=int(shim.obs_dim),
        n_steps=int(batch.n_steps),
        obs=np.asarray(batch.obs, dtype=np.float32).copy(),
        action_idx=np.asarray(batch.action_idx, dtype=np.int64).copy(),
        stake_unit=np.asarray(batch.stake_unit, dtype=np.float32).copy(),
        log_prob_action=np.asarray(batch.log_prob_action, dtype=np.float32).copy(),
        log_prob_stake=np.asarray(batch.log_prob_stake, dtype=np.float32).copy(),
        value_per_runner=np.asarray(batch.value_per_runner, dtype=np.float32).copy(),
        per_runner_reward=np.asarray(batch.per_runner_reward, dtype=np.float32).copy(),
        done=np.asarray(batch.done, dtype=bool).copy(),
        hidden_in=hidden_in,
        bets=bets,
        info_discrete=info_discrete,
        info_continuous=info_continuous,
    )


def _bet_to_dict(bet: Any) -> dict:
    d: dict[str, Any] = {}
    for f in _BET_DISCRETE:
        v = getattr(bet, f, None)
        # Enums (BetSide/BetOutcome) → their str value for stable compare.
        d[f] = getattr(v, "value", v)
    for f in _BET_CONTINUOUS:
        v = getattr(bet, f, None)
        d[f] = None if v is None else float(v)
    return d


# ── Compare ─────────────────────────────────────────────────────────────


@dataclass
class Mismatch:
    quantity: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.quantity}] {self.detail}"


def _cmp_array(name: str, a: np.ndarray, b: np.ndarray,
               *, exact: bool, out: list[Mismatch]) -> None:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        out.append(Mismatch(name, f"shape {a.shape} vs {b.shape}"))
        return
    if a.size == 0:
        return
    if exact:
        if not np.array_equal(a, b):
            n = int((a != b).sum())
            idx = int(np.argmax((a != b).reshape(-1)))
            out.append(Mismatch(
                name,
                f"{n} discrete mismatches; first flat idx {idx}: "
                f"{a.reshape(-1)[idx]!r} vs {b.reshape(-1)[idx]!r}",
            ))
        return
    rtol, atol = TOLERANCES.get(name, (1e-4, 1e-4))
    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        # ignore nan-vs-nan
        with np.errstate(invalid="ignore"):
            mask = ~(np.isnan(a) & np.isnan(b))
        maxd = float(np.nanmax(np.where(mask, diff, 0.0)))
        idx = int(np.nanargmax(np.where(mask, diff, 0.0)))
        out.append(Mismatch(
            name,
            f"max|Δ|={maxd:.3e} > atol={atol:.1e} rtol={rtol:.1e} "
            f"at flat idx {idx}: {a.reshape(-1)[idx]!r} vs "
            f"{b.reshape(-1)[idx]!r}",
        ))


def compare_streams(
    golden: GoldenStream, candidate: GoldenStream, *, label: str = "",
) -> list[Mismatch]:
    """Diff two streams per hard-constraint #1. Empty list ⇒ PASS."""
    out: list[Mismatch] = []

    if golden.n_steps != candidate.n_steps:
        out.append(Mismatch(
            "n_steps", f"{golden.n_steps} vs {candidate.n_steps}",
        ))
        # length divergence makes per-tick array compare meaningless
        return out

    # per-tick — discrete
    _cmp_array("action_idx", golden.action_idx, candidate.action_idx,
               exact=True, out=out)
    _cmp_array("done", golden.done, candidate.done, exact=True, out=out)
    # per-tick — continuous
    for name in ("obs", "stake_unit", "log_prob_action", "log_prob_stake",
                 "value_per_runner", "per_runner_reward"):
        _cmp_array(name, getattr(golden, name), getattr(candidate, name),
                   exact=False, out=out)
    # hidden state tuple
    if len(golden.hidden_in) != len(candidate.hidden_in):
        out.append(Mismatch(
            "hidden_in", f"tuple len {len(golden.hidden_in)} vs "
            f"{len(candidate.hidden_in)}",
        ))
    else:
        for k, (ga, ca) in enumerate(zip(golden.hidden_in, candidate.hidden_in)):
            _cmp_array("hidden_in", ga, ca, exact=False, out=out)

    # bet ledger
    _compare_bets(golden.bets, candidate.bets, out)

    # info
    for k, gv in golden.info_discrete.items():
        cv = candidate.info_discrete.get(k)
        if cv != gv:
            out.append(Mismatch(f"info.{k}", f"{gv} vs {cv} (exact)"))
    _, atol = TOLERANCES["info"]
    for k, gv in golden.info_continuous.items():
        cv = candidate.info_continuous.get(k)
        if cv is None or abs(float(gv) - float(cv)) > atol:
            out.append(Mismatch(f"info.{k}", f"{gv} vs {cv} (atol={atol:.1e})"))

    return out


def _pairing_groups(bets: list[dict]) -> list[int]:
    """Canonicalise opaque random pair_ids to structural group indices.

    Maps each bet's ``pair_id`` to an integer by first-appearance order
    (0, 1, 2, …); ``None`` (unpaired) → -1. Two runs that pair the SAME
    bets in the same order produce identical group sequences regardless
    of the random uuid strings, so this compares pairing STRUCTURE.
    """
    groups: list[int] = []
    seen: dict[str, int] = {}
    for b in bets:
        pid = b.get("pair_id")
        if pid is None:
            groups.append(-1)
        else:
            groups.append(seen.setdefault(pid, len(seen)))
    return groups


def _compare_bets(golden: list[dict], cand: list[dict],
                  out: list[Mismatch]) -> None:
    if len(golden) != len(cand):
        out.append(Mismatch(
            "bet_count", f"{len(golden)} vs {len(cand)} bets",
        ))
        return
    # Structural pairing comparison (ignores random uuid strings).
    gg, cg = _pairing_groups(golden), _pairing_groups(cand)
    if gg != cg:
        first = next((i for i in range(len(gg)) if gg[i] != cg[i]), -1)
        out.append(Mismatch(
            "pairing_structure",
            f"group sequence diverges at bet[{first}]: "
            f"{gg[first]} vs {cg[first]}",
        ))
    for i, (gb, cb) in enumerate(zip(golden, cand)):
        for f in _BET_DISCRETE:
            if gb.get(f) != cb.get(f):
                out.append(Mismatch(
                    f"bet[{i}].{f}", f"{gb.get(f)!r} vs {cb.get(f)!r} (exact)",
                ))
        for f in _BET_CONTINUOUS:
            gv, cv = gb.get(f), cb.get(f)
            if gv is None and cv is None:
                continue
            if gv is None or cv is None:
                out.append(Mismatch(f"bet[{i}].{f}", f"{gv!r} vs {cv!r}"))
                continue
            _, atol = TOLERANCES.get(f"bet.{f}", (0.0, 1e-4))
            if abs(float(gv) - float(cv)) > atol:
                out.append(Mismatch(
                    f"bet[{i}].{f}",
                    f"|Δ|={abs(float(gv)-float(cv)):.3e} > atol={atol:.1e}",
                ))


# ── Serialization ────────────────────────────────────────────────────────


def save_stream(path: str | Path, stream: GoldenStream) -> None:
    """Persist a stream as a ``.npz`` (arrays) + ``.json`` (bets/info/meta)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "obs": stream.obs,
        "action_idx": stream.action_idx,
        "stake_unit": stream.stake_unit,
        "log_prob_action": stream.log_prob_action,
        "log_prob_stake": stream.log_prob_stake,
        "value_per_runner": stream.value_per_runner,
        "per_runner_reward": stream.per_runner_reward,
        "done": stream.done,
    }
    for k, h in enumerate(stream.hidden_in):
        arrays[f"hidden_in_{k}"] = h
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    meta = {
        "case": stream.case, "seed": stream.seed,
        "hidden_size": stream.hidden_size, "obs_dim": stream.obs_dim,
        "n_steps": stream.n_steps, "n_hidden": len(stream.hidden_in),
        "bets": stream.bets,
        "info_discrete": stream.info_discrete,
        "info_continuous": stream.info_continuous,
    }
    path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def load_stream(path: str | Path) -> GoldenStream:
    path = Path(path)
    npz = np.load(path.with_suffix(".npz"))
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    n_hidden = int(meta.get("n_hidden", 0))
    hidden_in = tuple(npz[f"hidden_in_{k}"] for k in range(n_hidden))
    return GoldenStream(
        case=meta["case"], seed=int(meta["seed"]),
        hidden_size=int(meta["hidden_size"]), obs_dim=int(meta["obs_dim"]),
        n_steps=int(meta["n_steps"]),
        obs=npz["obs"], action_idx=npz["action_idx"],
        stake_unit=npz["stake_unit"], log_prob_action=npz["log_prob_action"],
        log_prob_stake=npz["log_prob_stake"],
        value_per_runner=npz["value_per_runner"],
        per_runner_reward=npz["per_runner_reward"], done=npz["done"],
        hidden_in=hidden_in,
        bets=meta["bets"], info_discrete=meta["info_discrete"],
        info_continuous=meta["info_continuous"],
    )
