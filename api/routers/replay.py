"""Race replay endpoints — tick-by-tick state with agent bet events overlaid."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    BetEvent,
    BetExplorerResponse,
    ExplorerBet,
    RaceSummary,
    ReplayDayResponse,
    ReplayRaceResponse,
    ReplayTick,
    TickRunner,
)
from registry.model_store import ModelStore

router = APIRouter(prefix="/replay", tags=["replay"])


def _store(request: Request) -> ModelStore:
    return request.app.state.store


def _config(request: Request) -> dict:
    return request.app.state.config


def _load_bets_for_run(store: ModelStore, run_id: str, date: str) -> list[dict]:
    """Load bet records for a specific run+date. Try Parquet first, fall back to SQLite."""
    # Try Parquet path
    bet_logs_dir = Path(store.bet_logs_dir)
    parquet_path = bet_logs_dir / run_id / f"{date}.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        return df.to_dict("records")

    # Fall back to SQLite evaluation_bets table (pre-2.6 data)
    try:
        bets = store.get_evaluation_bets(run_id)
        return [
            {
                "market_id": b.market_id,
                "tick_timestamp": b.tick_timestamp,
                "seconds_to_off": b.seconds_to_off,
                "runner_id": b.runner_id,
                "runner_name": b.runner_name,
                "action": b.action,
                "price": b.price,
                "stake": b.stake,
                "matched_size": b.matched_size,
                "outcome": b.outcome,
                "pnl": b.pnl,
            }
            for b in bets
            if b.date == date
        ]
    except Exception:
        return []


def _load_tick_data(config: dict, date: str) -> pd.DataFrame | None:
    """Load the raw Parquet tick data for a given date."""
    data_dir = Path(config["paths"]["processed_data"])
    path = data_dir / f"{date}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@router.get("/{model_id}/bets", response_model=BetExplorerResponse)
def get_model_bets(model_id: str, request: Request):
    """All bets for a model across all evaluation days."""
    store = _store(request)
    config = _config(request)

    rec = store.get_model(model_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Model not found")

    run = store.get_latest_evaluation_run(model_id)
    if not run:
        raise HTTPException(status_code=404, detail="No evaluation run found")

    bet_records = store.get_evaluation_bets(run.run_id)

    # Build market_id → {venue, market_start_time} lookup from tick data
    market_info: dict[str, dict[str, str]] = {}
    unique_dates = {b.date for b in bet_records}
    for date in unique_dates:
        ticks_df = _load_tick_data(config, date)
        if ticks_df is None:
            continue
        for market_id, group in ticks_df.groupby("market_id"):
            first = group.iloc[0]
            market_info[str(market_id)] = {
                "venue": str(first.get("venue", "")),
                "market_start_time": str(first.get("market_start_time", "")),
            }

    bets = [
        ExplorerBet(
            date=b.date,
            race_id=b.market_id,
            venue=market_info.get(b.market_id, {}).get("venue", ""),
            race_time=market_info.get(b.market_id, {}).get("market_start_time", ""),
            tick_timestamp=b.tick_timestamp,
            seconds_to_off=b.seconds_to_off,
            runner_id=b.runner_id,
            runner_name=b.runner_name,
            action=b.action,
            price=b.price,
            stake=b.stake,
            matched_size=b.matched_size,
            outcome=b.outcome,
            pnl=b.pnl,
            is_each_way=b.is_each_way,
            each_way_divisor=b.each_way_divisor,
            number_of_places=b.number_of_places,
            settlement_type=b.settlement_type,
            effective_place_odds=b.effective_place_odds,
        )
        for b in bet_records
    ]

    total_bets = len(bets)
    total_pnl = round(sum(b.pnl for b in bets), 2)
    winning = sum(1 for b in bets if b.pnl > 0)
    bet_precision = round(winning / total_bets, 4) if total_bets > 0 else 0.0
    pnl_per_bet = round(total_pnl / total_bets, 4) if total_bets > 0 else 0.0

    return BetExplorerResponse(
        model_id=model_id,
        total_bets=total_bets,
        total_pnl=total_pnl,
        bet_precision=bet_precision,
        pnl_per_bet=pnl_per_bet,
        bets=bets,
    )


@router.get("/{model_id}/{date}", response_model=ReplayDayResponse)
def get_replay_day(model_id: str, date: str, request: Request):
    """All races for a model+day with summary per race."""
    store = _store(request)
    config = _config(request)

    rec = store.get_model(model_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Model not found")

    run = store.get_latest_evaluation_run(model_id)
    if not run:
        raise HTTPException(status_code=404, detail="No evaluation run found")

    # Load bet records for this date
    bets = _load_bets_for_run(store, run.run_id, date)

    # Load tick data for race metadata
    ticks_df = _load_tick_data(config, date)
    if ticks_df is None:
        raise HTTPException(status_code=404, detail=f"No tick data for {date}")

    # Build race summaries
    races: list[RaceSummary] = []
    for market_id, group in ticks_df.groupby("market_id"):
        first = group.iloc[0]
        race_bets = [b for b in bets if b.get("market_id") == market_id]
        race_pnl = sum(b.get("pnl", 0.0) for b in race_bets)

        races.append(
            RaceSummary(
                race_id=str(market_id),
                market_name=str(first.get("market_name", "")),
                venue=str(first.get("venue", "")),
                market_start_time=str(first.get("market_start_time", "")),
                n_runners=int(first.get("number_of_active_runners", 0) or 0),
                bet_count=len(race_bets),
                race_pnl=round(race_pnl, 2),
            )
        )

    # Sort by market_start_time
    races.sort(key=lambda r: r.market_start_time)

    return ReplayDayResponse(model_id=model_id, date=date, races=races)


@router.get("/{model_id}/{date}/{race_id}", response_model=ReplayRaceResponse)
def get_replay_race(model_id: str, date: str, race_id: str, request: Request):
    """Full tick-by-tick state + agent actions for one race."""
    store = _store(request)
    config = _config(request)

    rec = store.get_model(model_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Model not found")

    run = store.get_latest_evaluation_run(model_id)
    if not run:
        raise HTTPException(status_code=404, detail="No evaluation run found")

    # Load tick data
    ticks_df = _load_tick_data(config, date)
    if ticks_df is None:
        raise HTTPException(status_code=404, detail=f"No tick data for {date}")

    # Filter to this race
    race_ticks = ticks_df[ticks_df["market_id"] == race_id].sort_values(
        "sequence_number"
    )
    if race_ticks.empty:
        raise HTTPException(status_code=404, detail=f"Race {race_id} not found")

    # Load bet records
    all_bets = _load_bets_for_run(store, run.run_id, date)
    race_bets = [b for b in all_bets if b.get("market_id") == race_id]

    # Index bets by tick_timestamp for overlay
    bets_by_ts: dict[str, list[dict]] = {}
    for b in race_bets:
        ts = str(b.get("tick_timestamp", ""))
        bets_by_ts.setdefault(ts, []).append(b)

    # Build tick-by-tick response
    first_row = race_ticks.iloc[0]
    venue = str(first_row.get("venue", ""))
    market_start_time = str(first_row.get("market_start_time", ""))
    winner = (
        int(first_row["winner_selection_id"])
        if pd.notna(first_row.get("winner_selection_id"))
        else None
    )

    replay_ticks: list[ReplayTick] = []
    for _, row in race_ticks.iterrows():
        ts_str = str(row["timestamp"])

        # Parse snap_json for runner order book state
        runners: list[TickRunner] = []
        snap = row.get("snap_json")
        if snap:
            try:
                parsed = json.loads(snap) if isinstance(snap, str) else snap
                runner_list = parsed
                if isinstance(parsed, dict):
                    runner_list = parsed.get("MarketRunners", parsed.get("runners", []))
                for r in runner_list:
                    sid = r.get("RunnerId", r.get("selection_id", 0))
                    if isinstance(sid, dict):
                        sid = sid.get("SelectionId", 0)
                    # Prices and status may be nested inside sub-objects
                    prices = r.get("Prices", r.get("prices", {})) or {}
                    defn = r.get("Definition", r.get("definition", {})) or {}
                    ltp = (
                        prices.get("LastTradedPrice")
                        or r.get("LastTradedPrice")
                        or r.get("last_traded_price")
                        or 0
                    )
                    status = (
                        defn.get("Status")
                        or r.get("Status")
                        or r.get("status")
                        or "ACTIVE"
                    )
                    total_matched = (
                        prices.get("TradedVolume")
                        or r.get("TotalMatched")
                        or r.get("total_matched")
                        or 0
                    )
                    runners.append(
                        TickRunner(
                            selection_id=int(sid),
                            status=str(status),
                            last_traded_price=float(ltp or 0),
                            total_matched=float(total_matched or 0),
                            available_to_back=_parse_prices(prices, "AvailableToBack"),
                            available_to_lay=_parse_prices(prices, "AvailableToLay"),
                        )
                    )
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

        # Overlay bet events at this tick
        tick_bets = [
            BetEvent(
                tick_timestamp=str(b.get("tick_timestamp", "")),
                seconds_to_off=float(b.get("seconds_to_off", 0)),
                runner_id=int(b.get("runner_id", 0)),
                runner_name=str(b.get("runner_name", "")),
                action=str(b.get("action", "")),
                price=float(b.get("price", 0)),
                stake=float(b.get("stake", 0)),
                matched_size=float(b.get("matched_size", 0)),
                outcome=str(b.get("outcome", "")),
                pnl=float(b.get("pnl", 0)),
            )
            for b in bets_by_ts.get(ts_str, [])
        ]

        replay_ticks.append(
            ReplayTick(
                timestamp=ts_str,
                sequence_number=int(row["sequence_number"]),
                in_play=bool(row.get("in_play", False)),
                traded_volume=float(row.get("traded_volume", 0) or 0),
                runners=runners,
                bets=tick_bets,
            )
        )

    # All bets as flat list for the side panel
    all_bet_events = [
        BetEvent(
            tick_timestamp=str(b.get("tick_timestamp", "")),
            seconds_to_off=float(b.get("seconds_to_off", 0)),
            runner_id=int(b.get("runner_id", 0)),
            runner_name=str(b.get("runner_name", "")),
            action=str(b.get("action", "")),
            price=float(b.get("price", 0)),
            stake=float(b.get("stake", 0)),
            matched_size=float(b.get("matched_size", 0)),
            outcome=str(b.get("outcome", "")),
            pnl=float(b.get("pnl", 0)),
        )
        for b in race_bets
    ]

    race_pnl = sum(b.get("pnl", 0.0) for b in race_bets)

    # Load runner names from runners Parquet
    runner_names: dict[str, str] = {}
    data_dir = Path(config["paths"]["processed_data"])
    runners_path = data_dir / f"{date}_runners.parquet"
    if runners_path.exists():
        runners_df = pd.read_parquet(runners_path)
        race_runners = runners_df[runners_df["market_id"] == race_id]
        for _, r in race_runners.iterrows():
            sid = str(int(r["selection_id"]))
            runner_names[sid] = str(r["runner_name"])

    return ReplayRaceResponse(
        model_id=model_id,
        date=date,
        race_id=race_id,
        venue=venue,
        market_start_time=market_start_time,
        winner_selection_id=winner,
        ticks=replay_ticks,
        all_bets=all_bet_events,
        race_pnl=round(race_pnl, 2),
        runner_names=runner_names,
    )


def _parse_prices(prices_obj: dict | None, key: str) -> list[dict]:
    """Extract price/size ladder from snap_json Prices object."""
    if not prices_obj:
        return []
    # Handle both camelCase (real data) and snake_case (test data)
    snake = key.replace("AvailableTo", "available_to_").lower()
    raw = prices_obj.get(key, prices_obj.get(snake, []))
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw[:3]:  # Top 3 levels only
        if isinstance(item, dict):
            result.append(
                {
                    "price": float(item.get("Price", item.get("price", 0)) or 0),
                    "size": float(item.get("Size", item.get("size", 0)) or 0),
                }
            )
    return result
