"""Activation / Session-07 measurement helper.

Dumps the three numbers the activation playbook (and Session 07
validation) need for a single training run:

1. Fill-probability MACE (reuses ``registry.calibration.compute_mace``).
2. Per-bucket completion rates (reuses
   ``registry.calibration.compute_bucket_outcomes``).
3. Risk-head Spearman ρ between predicted locked-stddev and the
   absolute error of realised locked-P&L vs predicted locked-P&L on
   completed pairs (reuses ``api.calibration._collect_scatter_pairs``
   for the pair assembly).

This is deliberately *one-run-at-a-time*. The cross-run comparison
CSVs the playbook Step B table describes are produced by running this
script over each activation run and tabulating the output — a shell
script or a follow-up tool's problem, not this one's.

Usage::

    # By run_id (eval parquet directory under logs/bet_logs/<run_id>/):
    python scripts/scalping_active_comparison.py --run-id <uuid>

    # By model_id (reads the latest eval run_id for that model):
    python scripts/scalping_active_comparison.py --model-id <sha>

Exit code is 0 on success, 2 on "no eval bets for this run", 1 on
argument / lookup failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.calibration import _collect_scatter_pairs
from registry.calibration import compute_bucket_outcomes, compute_mace
from registry.model_store import EvaluationBetRecord, ModelStore

logger = logging.getLogger(__name__)


@dataclass
class ActivationMetrics:
    """One run's activation-playbook-relevant metrics.

    All fields are ``None`` when their input was insufficient — e.g. a
    directional run has no fill-prob pairs, so ``fill_prob_mace`` and
    every bucket row are ``None``; a scalping run whose pairs are all
    naked has ``risk_spearman_rho=None`` (no realised-lock values).
    """

    run_id: str
    n_bets: int
    n_pairs_with_fill_prob: int
    n_completed_pairs_with_risk: int
    fill_prob_mace: float | None
    bucket_rows: list[dict]
    risk_spearman_rho: float | None


# ------------------------------------------------------------- spearman


def _rank(values: list[float]) -> list[float]:
    """Return the average-rank vector for ``values`` (ties shared)."""
    indexed = sorted(enumerate(values), key=lambda iv: iv[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation. Returns ``None`` for n<2 or zero-variance
    input (both collapse to an undefined rank correlation)."""
    if len(xs) != len(ys):
        raise ValueError("spearman_rho: input lengths differ")
    if len(xs) < 2:
        return None
    rx = _rank(xs)
    ry = _rank(ys)
    mean_rx = sum(rx) / len(rx)
    mean_ry = sum(ry) / len(ry)
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx)
    var_y = sum((b - mean_ry) ** 2 for b in ry)
    if var_x == 0.0 or var_y == 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


# -------------------------------------------------------------- metrics


def _risk_pairs(
    bets: list[EvaluationBetRecord],
) -> list[tuple[float, float]]:
    """Return ``(predicted_stddev, abs_prediction_error)`` per completed
    pair that carries both risk-head predictions.

    ``abs_prediction_error = abs(realised_locked_pnl - predicted_locked_pnl)`` —
    that's the quantity whose variance the risk head is supposed to be
    tracking, so a well-calibrated head has high Spearman ρ between
    stddev and the magnitude of the realised residual.
    """
    scatter = _collect_scatter_pairs(bets)
    return [
        (p.predicted_stddev, abs(p.realised_pnl - p.predicted_pnl))
        for p in scatter
    ]


def compute_activation_metrics(
    bets: list[EvaluationBetRecord],
    *,
    run_id: str = "",
) -> ActivationMetrics:
    """Reduce one run's eval bets to the numbers the activation
    playbook cares about. Pure function — no I/O, no DB, safe for unit
    tests."""
    buckets = compute_bucket_outcomes(bets)
    mace = compute_mace(bets)
    risk_xy = _risk_pairs(bets)
    rho: float | None = None
    if risk_xy:
        xs, ys = zip(*risk_xy)
        rho = spearman_rho(list(xs), list(ys))

    n_pairs_with_fill_prob = sum(b.count for b in buckets)
    return ActivationMetrics(
        run_id=run_id,
        n_bets=len(bets),
        n_pairs_with_fill_prob=n_pairs_with_fill_prob,
        n_completed_pairs_with_risk=len(risk_xy),
        fill_prob_mace=mace,
        bucket_rows=[
            {
                "label": b.label,
                "predicted_midpoint": b.predicted_midpoint,
                "observed_rate": b.observed_rate,
                "count": b.count,
                "abs_calibration_error": b.abs_calibration_error,
            }
            for b in buckets
        ],
        risk_spearman_rho=rho,
    )


# --------------------------------------------------------------- report


def _format_markdown(metrics: ActivationMetrics) -> str:
    lines = [
        f"# Activation metrics — run_id `{metrics.run_id or '(unset)'}`",
        "",
        f"- Total bets: {metrics.n_bets}",
        f"- Pairs with fill-prob prediction: {metrics.n_pairs_with_fill_prob}",
        f"- Completed pairs with risk prediction: "
        f"{metrics.n_completed_pairs_with_risk}",
        f"- Fill-prob MACE: "
        f"{'—' if metrics.fill_prob_mace is None else f'{metrics.fill_prob_mace:.4f}'}",
        f"- Risk Spearman ρ: "
        f"{'—' if metrics.risk_spearman_rho is None else f'{metrics.risk_spearman_rho:+.4f}'}",
        "",
        "## Per-bucket completion",
        "",
        "| Bucket | midpoint | observed | count | abs-error |",
        "|---|---|---|---|---|",
    ]
    for row in metrics.bucket_rows:
        lines.append(
            f"| {row['label']} | {row['predicted_midpoint']:.2f} | "
            f"{row['observed_rate']:.3f} | {row['count']} | "
            f"{row['abs_calibration_error']:.3f} |"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------ cli


def _resolve_run_id(args: argparse.Namespace, store: ModelStore) -> str:
    if args.run_id:
        return args.run_id
    if args.model_id:
        # Read the most recent eval run for this model.
        history = store.get_evaluation_history(args.model_id)
        if not history:
            raise SystemExit(
                f"No evaluation runs for model {args.model_id}"
            )
        return history[-1].run_id
    raise SystemExit("Pass --run-id or --model-id")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", help="Evaluation run UUID", default=None)
    parser.add_argument(
        "--model-id", help="Model SHA (uses its latest eval run)", default=None,
    )
    parser.add_argument(
        "--format", choices=("markdown", "json"), default="markdown",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="config.yaml (used to locate the registry DB + bet logs)",
    )
    args = parser.parse_args(argv)

    import yaml
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    store = ModelStore(
        db_path=config["paths"]["registry_db"],
        weights_dir=config["paths"]["model_weights"],
    )

    run_id = _resolve_run_id(args, store)
    bets = store.get_evaluation_bets(run_id)
    if not bets:
        logger.warning("No bets found for run_id %s", run_id)
        return 2

    metrics = compute_activation_metrics(bets, run_id=run_id)
    if args.format == "json":
        print(json.dumps(asdict(metrics), indent=2))
    else:
        print(_format_markdown(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
