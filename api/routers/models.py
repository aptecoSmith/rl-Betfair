"""Scoreboard, model detail, lineage, and genetic event endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    GarageToggleRequest,
    GarageToggleResponse,
    GeneticEvent,
    GeneticsResponse,
    LineageNode,
    LineageResponse,
    ModelDetail,
    DayMetric,
    ScoreboardEntry,
    ScoreboardResponse,
)
from registry.model_store import ModelStore
from registry.scoreboard import Scoreboard, ModelScore

router = APIRouter(prefix="/models", tags=["models"])


def _store(request: Request) -> ModelStore:
    return request.app.state.store


def _scoreboard(request: Request) -> Scoreboard:
    return request.app.state.scoreboard


def _score_to_entry(model: "ModelScore", store: ModelStore) -> ScoreboardEntry:
    rec = store.get_model(model.model_id)
    return ScoreboardEntry(
        model_id=model.model_id,
        generation=rec.generation if rec else 0,
        architecture_name=rec.architecture_name if rec else "",
        status=rec.status if rec else "active",
        composite_score=model.composite_score,
        win_rate=model.win_rate,
        sharpe=model.sharpe,
        mean_daily_pnl=model.mean_daily_pnl,
        efficiency=model.efficiency,
        test_days=model.test_days,
        profitable_days=model.profitable_days,
        garaged=rec.garaged if rec else False,
    )


@router.get("", response_model=ScoreboardResponse)
def get_scoreboard(request: Request):
    """All active models ranked by composite score, with per-day win_rate."""
    sb = _scoreboard(request)
    store = _store(request)
    ranked = sb.rank_all()
    entries = [_score_to_entry(m, store) for m in ranked]
    return ScoreboardResponse(models=entries)


@router.get("/garage", response_model=ScoreboardResponse)
def get_garage(request: Request):
    """List all garaged models with scores."""
    store = _store(request)
    sb = _scoreboard(request)
    garaged = store.list_garaged_models()
    entries = []
    for m in garaged:
        score = sb.score_model(m.model_id)
        if score:
            entries.append(_score_to_entry(score, store))
        else:
            entries.append(ScoreboardEntry(
                model_id=m.model_id, generation=m.generation,
                architecture_name=m.architecture_name, status=m.status,
                garaged=True, composite_score=m.composite_score,
                win_rate=0, sharpe=0, mean_daily_pnl=0,
                efficiency=0, test_days=0, profitable_days=0,
            ))
    return ScoreboardResponse(models=entries)


@router.put("/{model_id}/garage", response_model=GarageToggleResponse)
def toggle_garage(model_id: str, body: GarageToggleRequest, request: Request):
    """Set or clear the garaged flag on a model."""
    store = _store(request)
    rec = store.get_model(model_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Model not found")
    store.set_garaged(model_id, body.garaged)
    return GarageToggleResponse(model_id=model_id, garaged=body.garaged)


@router.get("/{model_id}", response_model=ModelDetail)
def get_model_detail(model_id: str, request: Request):
    """Model detail: hyperparams, architecture, metrics history."""
    store = _store(request)
    rec = store.get_model(model_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Model not found")

    # Get metrics history from latest evaluation run
    metrics: list[DayMetric] = []
    run = store.get_latest_evaluation_run(model_id)
    if run:
        days = store.get_evaluation_days(run.run_id)
        metrics = [
            DayMetric(
                date=d.date,
                day_pnl=d.day_pnl,
                bet_count=d.bet_count,
                winning_bets=d.winning_bets,
                bet_precision=d.bet_precision,
                pnl_per_bet=d.pnl_per_bet,
                early_picks=d.early_picks,
                profitable=d.profitable,
            )
            for d in days
        ]

    return ModelDetail(
        model_id=rec.model_id,
        generation=rec.generation,
        parent_a_id=rec.parent_a_id,
        parent_b_id=rec.parent_b_id,
        architecture_name=rec.architecture_name,
        architecture_description=rec.architecture_description,
        hyperparameters=rec.hyperparameters,
        status=rec.status,
        created_at=rec.created_at,
        last_evaluated_at=rec.last_evaluated_at,
        composite_score=rec.composite_score,
        garaged=rec.garaged,
        metrics_history=metrics,
    )


@router.get("/{model_id}/lineage", response_model=LineageResponse)
def get_lineage(model_id: str, request: Request):
    """Ancestry tree: walk parent chain up to root(s)."""
    store = _store(request)
    rec = store.get_model(model_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Model not found")

    nodes: list[LineageNode] = []
    visited: set[str] = set()
    queue = [model_id]

    while queue:
        mid = queue.pop(0)
        if mid in visited:
            continue
        visited.add(mid)

        m = store.get_model(mid)
        if not m:
            continue

        hp = m.hyperparameters if isinstance(m.hyperparameters, dict) else {}
        nodes.append(
            LineageNode(
                model_id=m.model_id,
                generation=m.generation,
                parent_a_id=m.parent_a_id,
                parent_b_id=m.parent_b_id,
                architecture_name=m.architecture_name,
                hyperparameters=hp,
                composite_score=m.composite_score,
            )
        )

        if m.parent_a_id and m.parent_a_id not in visited:
            queue.append(m.parent_a_id)
        if m.parent_b_id and m.parent_b_id not in visited:
            queue.append(m.parent_b_id)

    return LineageResponse(nodes=nodes)


@router.get("/{model_id}/genetics", response_model=GeneticsResponse)
def get_genetics(model_id: str, request: Request):
    """Genetic event log for this model's creation."""
    store = _store(request)
    rec = store.get_model(model_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Model not found")

    events = store.get_genetic_events(child_model_id=model_id)
    return GeneticsResponse(
        events=[
            GeneticEvent(
                event_id=e.event_id,
                generation=e.generation,
                event_type=e.event_type,
                child_model_id=e.child_model_id,
                parent_a_id=e.parent_a_id,
                parent_b_id=e.parent_b_id,
                hyperparameter=e.hyperparameter,
                parent_a_value=e.parent_a_value,
                parent_b_value=e.parent_b_value,
                inherited_from=e.inherited_from,
                mutation_delta=e.mutation_delta,
                final_value=e.final_value,
                selection_reason=e.selection_reason,
                human_summary=e.human_summary,
            )
            for e in events
        ]
    )
