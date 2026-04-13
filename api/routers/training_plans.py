"""
api/routers/training_plans.py -- read/write endpoints for Session-4 plans.

Thin façade over :mod:`training.training_plan`.  These endpoints are the
backend the Session-8 UI will consume; the UI itself is out of scope for
Session 4.

State injection
---------------
The router pulls its dependencies from ``request.app.state``:

- ``plan_registry`` -- a :class:`training.training_plan.PlanRegistry`.
  Required.  In production it is created in ``api/main.py`` lifespan.
  In tests, mount the router on a fresh ``FastAPI`` and assign a
  registry pointed at a ``tmp_path``.
- ``config`` -- the loaded ``config.yaml`` dict.  Used by the coverage
  endpoint to read ``hyperparameters.search_ranges`` so the report
  knows which buckets to compute.
- ``store`` -- optional :class:`registry.model_store.ModelStore`.  If
  set, the coverage endpoint reads historical agents from it; if
  absent (test scenario), the endpoint falls back to whatever
  ``app.state.coverage_history`` provides (also a test seam).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from training.training_plan import (
    CoverageReport,
    HistoricalAgent,
    PlanRegistry,
    TrainingPlan,
    ValidationIssue,
    bias_sampler,
    compute_coverage,
    historical_agents_from_model_store,
    is_launchable,
    validate_plan,
)
from agents.population_manager import parse_search_ranges

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training-plans", tags=["training-plans"])


# -- Helpers ------------------------------------------------------------------


def _registry(request: Request) -> PlanRegistry:
    reg = getattr(request.app.state, "plan_registry", None)
    if reg is None:
        raise HTTPException(503, "Plan registry not configured")
    return reg


def _hp_specs(request: Request) -> list:
    config = getattr(request.app.state, "config", None) or {}
    raw = (
        config.get("hyperparameters", {})
        .get("search_ranges", {})
    )
    return parse_search_ranges(raw)


def _history(request: Request) -> list[HistoricalAgent]:
    # Test seam: ``coverage_history`` overrides the model store.
    seam = getattr(request.app.state, "coverage_history", None)
    if seam is not None:
        return list(seam)
    store = getattr(request.app.state, "store", None)
    return historical_agents_from_model_store(store)


def _issues_to_dicts(issues: list[ValidationIssue]) -> list[dict]:
    return [i.to_dict() for i in issues]


# -- Endpoints ----------------------------------------------------------------


@router.get("")
def list_plans(request: Request) -> dict[str, Any]:
    """Return all known plans (full payload, sorted by created_at desc)."""
    reg = _registry(request)
    plans = sorted(reg.list(), key=lambda p: p.created_at, reverse=True)
    return {
        "plans": [p.to_dict() for p in plans],
        "count": len(plans),
    }


@router.get("/coverage")
def coverage(request: Request) -> dict[str, Any]:
    """Coverage stats over the active history (model store or test seam)."""
    specs = _hp_specs(request)
    history = _history(request)
    report: CoverageReport = compute_coverage(history, specs)
    biased = bias_sampler(specs, history)
    return {
        "report": report.to_dict(),
        "biased_genes": [b.spec.name for b in biased if b.is_biased],
    }


@router.get("/{plan_id}")
def get_plan(plan_id: str, request: Request) -> dict[str, Any]:
    reg = _registry(request)
    try:
        plan = reg.load(plan_id)
    except KeyError:
        raise HTTPException(404, f"No such plan: {plan_id}")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {
        "plan": plan.to_dict(),
        "validation": _issues_to_dicts(validate_plan(plan)),
    }


@router.delete("/{plan_id}")
def delete_plan(plan_id: str, request: Request) -> dict[str, Any]:
    """Delete a plan by ID. Returns 404 if the plan doesn't exist."""
    reg = _registry(request)
    if not reg.delete(plan_id):
        raise HTTPException(404, f"No such plan: {plan_id}")
    return {"deleted": True, "plan_id": plan_id}


@router.post("")
def create_plan(payload: dict, request: Request) -> dict[str, Any]:
    """Create + persist a plan after validating it.

    Returns 422 with the issue list when the plan has any
    error-severity validation issues -- the plan is *not* saved in
    that case.  This endpoint never launches training; it is purely
    a "park this configuration so we can run it later" call.
    """
    reg = _registry(request)
    # Validate starting_budget if provided
    raw_budget = payload.get("starting_budget")
    if raw_budget is not None:
        try:
            budget_val = float(raw_budget)
        except (TypeError, ValueError):
            raise HTTPException(422, "starting_budget must be a number")
        if budget_val <= 0:
            raise HTTPException(422, "starting_budget must be positive")
    else:
        budget_val = None

    try:
        plan = TrainingPlan.new(
            name=str(payload.get("name", "unnamed")),
            population_size=int(payload["population_size"]),
            architectures=list(payload.get("architectures", [])),
            hp_ranges=dict(payload.get("hp_ranges", {})),
            seed=payload.get("seed"),
            arch_mix=payload.get("arch_mix"),
            min_arch_samples=int(payload.get("min_arch_samples", 5)),
            notes=str(payload.get("notes", "")),
            starting_budget=budget_val,
            exploration_strategy=str(payload.get("exploration_strategy", "random")),
            manual_seed_point=payload.get("manual_seed_point"),
            n_generations=int(payload.get("n_generations", 3)),
            n_epochs=int(payload.get("n_epochs", 3)),
            generations_per_session=(
                int(payload["generations_per_session"])
                if payload.get("generations_per_session") is not None
                else None
            ),
            auto_continue=bool(payload.get("auto_continue", False)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(422, f"Malformed plan payload: {exc}")

    issues = validate_plan(plan)
    if not is_launchable(issues):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Plan failed validation",
                "issues": _issues_to_dicts(issues),
            },
        )

    reg.save(plan)
    return {
        "plan": plan.to_dict(),
        "validation": _issues_to_dicts(issues),
    }
