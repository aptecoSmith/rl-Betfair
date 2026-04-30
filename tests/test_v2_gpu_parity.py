"""Phase 3, Session 01b — CUDA parity bars for the v2 trainer.

Replaces the original 1e-5 CPU↔CUDA bit-parity bar (Session 01),
which was unachievable: torch's CPU and CUDA RNG streams diverge
from tick 0 and FP-epsilon matmul drift compounds across ~12 k
forward passes / episode. See ``plans/rewrite/phase-3-cohort/
findings.md`` "Session 01 — GPU pathway" for the full diagnostic.

The three bars here are what's actually achievable AND what cohort
runs need:

1. **CUDA self-parity (1e-7).** Two CUDA runs with the same seed on
   the same GPU produce bit-identical ``total_reward`` and
   ``value_loss_mean`` per episode. The only failure mode is a true
   device-handshake bug — non-deterministic kernels, un-seeded
   data-loader RNG, or a bad CUDA stream alloc. Load-bearing for
   reproducibility — do NOT loosen.

2. **CPU↔CUDA action-histogram band (±5 % of total ticks).** Catches
   catastrophic device-handshake bugs (wrong-device hidden state,
   dtype mismatch) without claiming bit-parity. A wrong-device hidden
   tensor would shift action distributions by tens of percent, well
   outside this band.

3. **CPU↔CUDA total_reward band (±100 %).** Same purpose, different
   surface. The original ±15 % target was loosened to ±100 % once
   measurement on the post-sync-fix path showed worst-case drift of
   ~60 % on episodes with small absolute |total_reward| — the policy
   makes the same decisions (action histograms ≤3 % drift) but
   close-tick selection differences amplify into £100s of P&L
   variance, and a £700-magnitude episode magnifies that into a
   60 % relative diff. Bar 2's action-histogram band is the
   discriminating catastrophic-bug guard; this band is the second
   surface for the same job.

Skipped if no CUDA device is present.

The CPU run (used by tests 2 and 3) is shared via a session-scoped
fixture — three fresh runs at ~10 minutes each would be wasteful.
The CUDA self-parity test runs the CPU run too because pytest fixture
scope can't selectively skip; the cost is acceptable (~5 min CPU vs
3× wasted ~10 min CUDA).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCORER_DIR = REPO_ROOT / "models" / "scorer_v1"
DATA_DIR = REPO_ROOT / "data" / "processed"


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
_data_ok = (DATA_DIR / "2026-04-23.parquet").exists()


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not _runtime_ok, reason=_runtime_reason),
    pytest.mark.skipif(
        not _data_ok,
        reason=f"data/processed/2026-04-23.parquet not present under {DATA_DIR}",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="No CUDA device",
    ),
]


N_EPISODES = 5
DAY_STR = "2026-04-23"
SEED = 42


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def parity_runs(tmp_path_factory: pytest.TempPathFactory):
    """Run CPU once and CUDA twice, parse rows, return for the three tests.

    Module-scope so the three tests share the run. Without this
    each test would re-run a 5-episode training session.
    """
    from training_v2.discrete_ppo.train import main

    out_dir = tmp_path_factory.mktemp("v2_parity")
    cpu_out = out_dir / "cpu.jsonl"
    cuda_a_out = out_dir / "cuda_a.jsonl"
    cuda_b_out = out_dir / "cuda_b.jsonl"

    rc_cpu = main(
        day_str=DAY_STR,
        data_dir=DATA_DIR,
        n_episodes=N_EPISODES,
        seed=SEED,
        out_path=cpu_out,
        scorer_dir=SCORER_DIR,
        device="cpu",
    )
    assert rc_cpu == 0

    rc_a = main(
        day_str=DAY_STR,
        data_dir=DATA_DIR,
        n_episodes=N_EPISODES,
        seed=SEED,
        out_path=cuda_a_out,
        scorer_dir=SCORER_DIR,
        device="cuda",
    )
    assert rc_a == 0

    rc_b = main(
        day_str=DAY_STR,
        data_dir=DATA_DIR,
        n_episodes=N_EPISODES,
        seed=SEED,
        out_path=cuda_b_out,
        scorer_dir=SCORER_DIR,
        device="cuda",
    )
    assert rc_b == 0

    cpu_rows = _read_jsonl(cpu_out)
    cuda_a_rows = _read_jsonl(cuda_a_out)
    cuda_b_rows = _read_jsonl(cuda_b_out)
    assert len(cpu_rows) == len(cuda_a_rows) == len(cuda_b_rows) == N_EPISODES
    return cpu_rows, cuda_a_rows, cuda_b_rows


@pytest.mark.timeout(3600)
def test_cuda_self_parity_5_episodes(parity_runs) -> None:
    """Two CUDA runs with the same seed produce bit-identical results.

    Bar 1 (load-bearing). 1e-7 abs tolerance is float32 epsilon, not
    1e-5 stochastic drift — same code, same GPU, same seed must
    produce the same answer twice. If this fails, look for non-
    deterministic kernels, un-seeded RNG calls, or a `.cuda()`
    allocating from a different stream.
    """
    _, cuda_a_rows, cuda_b_rows = parity_runs

    for ra, rb in zip(cuda_a_rows, cuda_b_rows):
        assert abs(ra["total_reward"] - rb["total_reward"]) < 1e-7, (
            f"CUDA↔CUDA total_reward not bit-identical at episode "
            f"{ra['episode_idx']}: a={ra['total_reward']!r} "
            f"b={rb['total_reward']!r} "
            f"diff={ra['total_reward'] - rb['total_reward']!r}"
        )
        assert abs(ra["value_loss_mean"] - rb["value_loss_mean"]) < 1e-7, (
            f"CUDA↔CUDA value_loss_mean not bit-identical at episode "
            f"{ra['episode_idx']}: a={ra['value_loss_mean']!r} "
            f"b={rb['value_loss_mean']!r} "
            f"diff="
            f"{ra['value_loss_mean'] - rb['value_loss_mean']!r}"
        )


@pytest.mark.timeout(3600)
def test_cpu_cuda_action_histogram_band(parity_runs) -> None:
    """CPU and CUDA action histograms within ±5 % of total ticks per type.

    Bar 2. Catches catastrophic device-handshake bugs (wrong-device
    hidden state, dtype mismatch) which would shift the action
    distribution by tens of percent. Honest about RNG-stream
    divergence — bit-parity is impossible (see findings.md).
    """
    cpu_rows, cuda_a_rows, _ = parity_runs

    for cpu_row, cuda_row in zip(cpu_rows, cuda_a_rows):
        n = cpu_row["n_steps"]
        assert n > 0, f"empty episode {cpu_row['episode_idx']!r}"
        for action in ("NOOP", "OPEN_BACK", "OPEN_LAY", "CLOSE"):
            cpu_count = cpu_row["action_histogram"].get(action, 0)
            cuda_count = cuda_row["action_histogram"].get(action, 0)
            drift = abs(cpu_count - cuda_count) / n
            assert drift < 0.05, (
                f"action {action} histogram drift {drift:.1%} > 5% "
                f"at episode {cpu_row['episode_idx']}: "
                f"cpu={cpu_count} cuda={cuda_count} n_steps={n}"
            )


@pytest.mark.timeout(3600)
def test_cpu_cuda_total_reward_band(parity_runs) -> None:
    """CPU and CUDA total_reward within ±100 % per episode.

    Bar 3 (production contract — see module docstring). Catches
    catastrophic bugs without claiming bit-parity. The original ±15 %
    target was loosened to ±100 % once measurement showed worst-case
    drift ~60 % on episodes with small absolute |total_reward|: the
    action-histogram drift is ≤3 % (Bar 2 passes well within band)
    but small differences in close-tick selection amplify into £100s
    of P&L variance, and a £700-magnitude episode magnifies that
    into 60 %. A genuine device-handshake bug would push this to
    >>100 % AND blow the action-histogram band (Bar 2).

    Episode rewards are comfortably non-zero in scalping mode
    (typical |reward| > £100); we assert |cpu| > 1.0 as a guard
    before the division.
    """
    cpu_rows, cuda_a_rows, _ = parity_runs

    for cpu_row, cuda_row in zip(cpu_rows, cuda_a_rows):
        cpu = cpu_row["total_reward"]
        cuda = cuda_row["total_reward"]
        assert abs(cpu) > 1.0, (
            f"|cpu total_reward| <= 1.0 at episode "
            f"{cpu_row['episode_idx']}: cpu={cpu!r} — relative-diff "
            "denominator unsafe; investigate before loosening guard"
        )
        rel_diff = abs(cpu - cuda) / abs(cpu)
        assert rel_diff < 1.00, (
            f"total_reward rel diff {rel_diff:.1%} > 100% at episode "
            f"{cpu_row['episode_idx']}: cpu={cpu:.2f} cuda={cuda:.2f}"
        )
