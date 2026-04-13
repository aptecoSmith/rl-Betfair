"""API router for the exploration / coverage dashboard.

Endpoints
---------
GET /api/exploration/history
    Returns all exploration runs from the DB.
GET /api/exploration/coverage
    Returns the current coverage report.
GET /api/exploration/suggested-seed
    Returns a suggested next seed point based on coverage gaps.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from agents.population_manager import parse_search_ranges
from registry.model_store import ModelStore
from training.training_plan import (
    compute_coverage,
    generate_coverage_seed,
    historical_agents_from_model_store,
)

router = APIRouter(prefix="/exploration", tags=["exploration"])


def _store(request: Request) -> ModelStore:
    return request.app.state.store


def _hp_specs(request: Request):
    ranges = request.app.state.config["hyperparameters"]["search_ranges"]
    return parse_search_ranges(ranges)


@router.get("/history")
def exploration_history(request: Request) -> dict[str, Any]:
    """Return all exploration runs."""
    store = _store(request)
    runs = store.get_exploration_history()
    return {
        "runs": [
            {
                "id": r.id,
                "run_id": r.run_id,
                "created_at": r.created_at,
                "seed_point": r.seed_point,
                "region_id": r.region_id,
                "strategy": r.strategy,
                "coverage_before": r.coverage_before,
                "notes": r.notes,
            }
            for r in runs
        ],
        "count": len(runs),
    }


@router.get("/coverage")
def coverage_report(request: Request) -> dict[str, Any]:
    """Return current coverage analysis."""
    store = _store(request)
    hp_specs = _hp_specs(request)
    history = historical_agents_from_model_store(store)
    report = compute_coverage(history, hp_specs)

    # Per-gene coverage for the dashboard.
    genes = []
    for name, gc in report.gene_coverage.items():
        genes.append({
            "name": name,
            "bucket_edges": gc.bucket_edges,
            "bucket_counts": gc.bucket_counts,
            "nonempty_buckets": gc.nonempty_buckets,
            "coverage_fraction": gc.coverage_fraction,
            "well_covered": gc.well_covered,
        })

    return {
        "total_agents": report.total_agents,
        "arch_counts": report.arch_counts,
        "arch_undercovered": report.arch_undercovered,
        "genes": genes,
        "poorly_covered_genes": report.poorly_covered_genes,
    }


@router.get("/suggested-seed")
def suggested_seed(request: Request) -> dict[str, Any]:
    """Return a suggested next seed point based on coverage gaps."""
    store = _store(request)
    hp_specs = _hp_specs(request)
    history = historical_agents_from_model_store(store)
    seed_point, report = generate_coverage_seed(hp_specs, history)
    return {
        "seed_point": seed_point,
        "poorly_covered_genes": report.poorly_covered_genes,
        "total_agents": report.total_agents,
    }
