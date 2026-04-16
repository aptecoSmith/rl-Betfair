"""Unit tests for api/routers/replay.py — race replay endpoints."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.model_store import (
    ModelStore,
    EvaluationDayRecord,
    EvaluationBetRecord,
)
from api.routers import replay


# ── Helpers ──────────────────────────────────────────────────────────


def _create_store(tmp_dir: str) -> ModelStore:
    db_path = str(Path(tmp_dir) / "test.db")
    weights_dir = str(Path(tmp_dir) / "weights")
    bet_logs_dir = str(Path(tmp_dir) / "bet_logs")
    return ModelStore(db_path=db_path, weights_dir=weights_dir, bet_logs_dir=bet_logs_dir)


def _make_snap_json(runners: list[dict]) -> str:
    """Build a minimal SnapJson with runner data."""
    snap = []
    for r in runners:
        snap.append(
            {
                "RunnerId": {"SelectionId": r["sid"]},
                "Status": r.get("status", "ACTIVE"),
                "LastTradedPrice": r.get("ltp", 3.5),
                "TotalMatched": r.get("total_matched", 1000.0),
                "Prices": {
                    "AvailableToBack": [
                        {"Price": 3.5, "Size": 100.0},
                        {"Price": 3.4, "Size": 50.0},
                    ],
                    "AvailableToLay": [
                        {"Price": 3.6, "Size": 80.0},
                        {"Price": 3.7, "Size": 40.0},
                    ],
                },
            }
        )
    return json.dumps(snap)


def _create_tick_parquet(tmp_dir: str, date: str, markets: dict[str, int]) -> None:
    """Create a synthetic tick Parquet file.

    markets: {market_id: n_ticks}
    """
    rows = []
    for mid, n_ticks in markets.items():
        for i in range(n_ticks):
            rows.append(
                {
                    "market_id": mid,
                    "timestamp": f"2026-03-26T14:00:{i:02d}",
                    "sequence_number": i,
                    "snap_json": _make_snap_json([{"sid": 101}, {"sid": 102}]),
                    "winner_selection_id": 101,
                    "venue": "Newmarket",
                    "market_start_time": "2026-03-26T14:30:00",
                    "market_type": "WIN",
                    "market_name": f"14:30 Newmarket ({mid})",
                    "number_of_active_runners": 2,
                    "traded_volume": 5000.0 + i * 100,
                    "in_play": False,
                    "temperature": 12.0,
                    "precipitation": 0.0,
                    "wind_speed": 5.0,
                    "wind_direction": 180.0,
                    "humidity": 65.0,
                    "weather_code": 0,
                }
            )

    df = pd.DataFrame(rows)
    data_dir = Path(tmp_dir) / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_dir / f"{date}.parquet", index=False)


def _make_app(store: ModelStore, config: dict) -> TestClient:
    app = FastAPI()
    app.include_router(replay.router)
    app.state.store = store
    app.state.config = config
    return TestClient(app)


def _seed_model_with_bets(
    store: ModelStore, date: str, market_id: str
) -> tuple[str, str]:
    """Create model + evaluation run + Parquet bet logs. Returns (model_id, run_id)."""
    model_id = store.create_model(
        generation=0,
        architecture_name="ppo_lstm_v1",
        architecture_description="Test",
        hyperparameters={"lr": 0.001},
    )
    run_id = store.create_evaluation_run(
        model_id=model_id,
        train_cutoff_date="2026-03-25",
        test_days=[date],
    )
    store.record_evaluation_day(
        EvaluationDayRecord(
            run_id=run_id,
            date=date,
            day_pnl=25.0,
            bet_count=2,
            winning_bets=1,
            bet_precision=0.5,
            pnl_per_bet=12.5,
            early_picks=1,
            profitable=True,
        )
    )

    # Write bet logs to Parquet
    bets = [
        EvaluationBetRecord(
            run_id=run_id,
            date=date,
            market_id=market_id,
            tick_timestamp="2026-03-26T14:00:05",
            seconds_to_off=1795.0,
            runner_id=101,
            runner_name="Fast Horse",
            action="back",
            price=3.5,
            stake=10.0,
            matched_size=10.0,
            outcome="won",
            pnl=25.0,
        ),
        EvaluationBetRecord(
            run_id=run_id,
            date=date,
            market_id=market_id,
            tick_timestamp="2026-03-26T14:00:10",
            seconds_to_off=1790.0,
            runner_id=102,
            runner_name="Slow Horse",
            action="lay",
            price=5.0,
            stake=5.0,
            matched_size=5.0,
            outcome="won",
            pnl=-20.0,
        ),
    ]
    store.write_bet_logs_parquet(run_id, date, bets)

    return model_id, run_id


# ── Replay Day Tests ─────────────────────────────────────────────────


class TestReplayDay:
    def test_model_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)
            resp = client.get("/replay/nonexistent/2026-03-26")
            assert resp.status_code == 404

    def test_no_evaluation_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            mid = store.create_model(0, "ppo_lstm_v1", "Test", {"lr": 0.001})
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)
            resp = client.get(f"/replay/{mid}/2026-03-26")
            assert resp.status_code == 404

    def test_no_tick_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            mid = store.create_model(0, "ppo_lstm_v1", "Test", {"lr": 0.001})
            store.create_evaluation_run(mid, "2026-03-25", ["2026-03-26"])
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)
            resp = client.get(f"/replay/{mid}/2026-03-26")
            assert resp.status_code == 404

    def test_replay_day_returns_races(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            date = "2026-03-26"
            market_a = "1.234567890"
            market_b = "1.234567891"
            _create_tick_parquet(tmp, date, {market_a: 5, market_b: 3})
            model_id, run_id = _seed_model_with_bets(store, date, market_a)
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)

            resp = client.get(f"/replay/{model_id}/{date}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["model_id"] == model_id
            assert data["date"] == date
            assert len(data["races"]) == 2

            # Race with bets should show bet_count and pnl
            race_a = next(r for r in data["races"] if r["race_id"] == market_a)
            assert race_a["bet_count"] == 2
            assert race_a["race_pnl"] == 5.0  # 25 + (-20)

            # Race without bets
            race_b = next(r for r in data["races"] if r["race_id"] == market_b)
            assert race_b["bet_count"] == 0
            assert race_b["race_pnl"] == 0.0

    def test_replay_day_race_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            date = "2026-03-26"
            _create_tick_parquet(tmp, date, {"1.111": 3})
            model_id, _ = _seed_model_with_bets(store, date, "1.111")
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)

            resp = client.get(f"/replay/{model_id}/{date}")
            race = resp.json()["races"][0]
            assert race["venue"] == "Newmarket"
            assert "14:30" in race["market_start_time"]
            assert race["n_runners"] == 2


# ── Replay Race Tests ────────────────────────────────────────────────


class TestReplayRace:
    def test_race_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            date = "2026-03-26"
            _create_tick_parquet(tmp, date, {"1.111": 3})
            model_id, _ = _seed_model_with_bets(store, date, "1.111")
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)
            resp = client.get(f"/replay/{model_id}/{date}/1.999")
            assert resp.status_code == 404

    def test_replay_race_tick_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            date = "2026-03-26"
            market_id = "1.111"
            n_ticks = 5
            _create_tick_parquet(tmp, date, {market_id: n_ticks})
            model_id, _ = _seed_model_with_bets(store, date, market_id)
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)

            resp = client.get(f"/replay/{model_id}/{date}/{market_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["ticks"]) == n_ticks

            # Ticks are ordered by sequence_number
            seq_nums = [t["sequence_number"] for t in data["ticks"]]
            assert seq_nums == list(range(n_ticks))

    def test_replay_race_runners_parsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            date = "2026-03-26"
            market_id = "1.111"
            _create_tick_parquet(tmp, date, {market_id: 2})
            model_id, _ = _seed_model_with_bets(store, date, market_id)
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)

            resp = client.get(f"/replay/{model_id}/{date}/{market_id}")
            tick = resp.json()["ticks"][0]
            assert len(tick["runners"]) == 2
            r1 = tick["runners"][0]
            assert r1["selection_id"] == 101
            assert r1["last_traded_price"] == 3.5
            assert len(r1["available_to_back"]) == 2
            assert r1["available_to_back"][0]["price"] == 3.5
            assert r1["available_to_back"][0]["size"] == 100.0

    def test_replay_race_bets_overlaid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            date = "2026-03-26"
            market_id = "1.111"
            _create_tick_parquet(tmp, date, {market_id: 15})
            model_id, _ = _seed_model_with_bets(store, date, market_id)
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)

            resp = client.get(f"/replay/{model_id}/{date}/{market_id}")
            data = resp.json()

            # all_bets should contain both bets
            assert len(data["all_bets"]) == 2
            back_bet = next(b for b in data["all_bets"] if b["action"] == "back")
            assert back_bet["runner_name"] == "Fast Horse"
            assert back_bet["price"] == 3.5
            assert back_bet["pnl"] == 25.0

            # Check bets are overlaid on correct ticks
            ticks_with_bets = [t for t in data["ticks"] if len(t["bets"]) > 0]
            assert len(ticks_with_bets) >= 1

    def test_replay_race_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            date = "2026-03-26"
            market_id = "1.111"
            _create_tick_parquet(tmp, date, {market_id: 3})
            model_id, _ = _seed_model_with_bets(store, date, market_id)
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)

            resp = client.get(f"/replay/{model_id}/{date}/{market_id}")
            data = resp.json()
            assert data["venue"] == "Newmarket"
            assert data["winner_selection_id"] == 101
            assert data["race_pnl"] == 5.0  # 25 + (-20)

    def test_replay_race_no_bets(self):
        """A race with no bets still returns all ticks."""
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            date = "2026-03-26"
            market_id = "1.222"  # Different from the one with bets
            _create_tick_parquet(tmp, date, {market_id: 3, "1.111": 2})
            # Bets are seeded for market 1.111, not 1.222
            model_id, _ = _seed_model_with_bets(store, date, "1.111")
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)

            resp = client.get(f"/replay/{model_id}/{date}/{market_id}")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["ticks"]) == 3
            assert data["all_bets"] == []
            assert data["race_pnl"] == 0.0


# ── Bet Explorer Tests ──────────────────────────────────────────────


class TestBetExplorer:
    def test_model_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)
            resp = client.get("/replay/nonexistent/bets")
            assert resp.status_code == 404

    def test_no_evaluation_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            mid = store.create_model(0, "ppo_lstm_v1", "Test", {"lr": 0.001})
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)
            resp = client.get(f"/replay/{mid}/bets")
            assert resp.status_code == 404

    def test_returns_all_bets(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            date = "2026-03-26"
            market_id = "1.111"
            _create_tick_parquet(tmp, date, {market_id: 5})
            model_id, _ = _seed_model_with_bets(store, date, market_id)
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)

            resp = client.get(f"/replay/{model_id}/bets")
            assert resp.status_code == 200
            data = resp.json()
            assert data["model_id"] == model_id
            assert data["total_bets"] == 2
            assert len(data["bets"]) == 2

    def test_summary_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            date = "2026-03-26"
            market_id = "1.111"
            _create_tick_parquet(tmp, date, {market_id: 5})
            model_id, _ = _seed_model_with_bets(store, date, market_id)
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)

            resp = client.get(f"/replay/{model_id}/bets")
            data = resp.json()
            assert data["total_pnl"] == 5.0  # 25 + (-20)
            assert data["bet_precision"] == 0.5  # 1 winning / 2 total
            assert data["pnl_per_bet"] == 2.5  # 5 / 2

    def test_bet_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            date = "2026-03-26"
            market_id = "1.111"
            _create_tick_parquet(tmp, date, {market_id: 5})
            model_id, _ = _seed_model_with_bets(store, date, market_id)
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)

            resp = client.get(f"/replay/{model_id}/bets")
            bet = resp.json()["bets"][0]
            assert "date" in bet
            assert "race_id" in bet
            assert "runner_name" in bet
            assert "action" in bet
            assert "price" in bet
            assert "stake" in bet
            assert "pnl" in bet
            assert "outcome" in bet
            assert "seconds_to_off" in bet

    def test_empty_bets(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            mid = store.create_model(0, "ppo_lstm_v1", "Test", {"lr": 0.001})
            run_id = store.create_evaluation_run(mid, "2026-03-25", ["2026-03-26"])
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)

            resp = client.get(f"/replay/{mid}/bets")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_bets"] == 0
            assert data["bets"] == []
            assert data["bet_precision"] == 0.0
            assert data["pnl_per_bet"] == 0.0

    def test_ew_fallback_from_tick_data(self):
        """Old bet logs (no EW columns) get is_each_way from tick data."""
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            date = "2026-03-26"
            market_id = "1.222"

            # Create tick data with EACH_WAY market type
            rows = []
            for i in range(3):
                rows.append({
                    "market_id": market_id,
                    "timestamp": f"2026-03-26T14:00:{i:02d}",
                    "sequence_number": i,
                    "snap_json": _make_snap_json([{"sid": 101}]),
                    "winner_selection_id": 101,
                    "venue": "Cork",
                    "market_start_time": "2026-03-26T12:15:00",
                    "market_type": "EACH_WAY",
                    "market_name": "12:15 Cork",
                    "number_of_active_runners": 1,
                    "traded_volume": 5000.0,
                    "in_play": False,
                    "each_way_divisor": 5.0,
                    "number_of_each_way_places": 3,
                    "temperature": None, "precipitation": None,
                    "wind_speed": None, "wind_direction": None,
                    "humidity": None, "weather_code": None,
                })
            data_dir = Path(tmp) / "processed"
            data_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(data_dir / f"{date}.parquet", index=False)

            # Create model + eval run with OLD format bet logs (no EW columns)
            mid = store.create_model(0, "ppo_lstm_v1", "Test", {"lr": 0.001})
            rid = store.create_evaluation_run(mid, "2026-03-25", [date])
            run_dir = store.bet_logs_dir / rid
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{
                "run_id": rid, "date": date, "market_id": market_id,
                "tick_timestamp": "2026-03-26T14:00:01",
                "seconds_to_off": 300.0, "runner_id": 101,
                "runner_name": "Jackmoon", "action": "back",
                "price": 7.4, "stake": 11.50, "matched_size": 11.50,
                "outcome": "won", "pnl": 41.952,
                "opportunity_window_s": 60.0,
            }]).to_parquet(run_dir / f"{date}.parquet", index=False)

            config = {"paths": {"processed_data": str(data_dir)}}
            client = _make_app(store, config)

            resp = client.get(f"/replay/{mid}/bets")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_bets"] == 1
            bet = data["bets"][0]
            # EW fields should be populated from tick data fallback
            assert bet["is_each_way"] is True
            assert bet["each_way_divisor"] == 5.0
            assert bet["number_of_places"] == 3

    def test_scalping_aux_head_fields_present(self):
        """Session 04: `/bets` response includes the three aux-head fields."""
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            date = "2026-03-26"
            market_id = "1.111"
            _create_tick_parquet(tmp, date, {market_id: 5})
            model_id, _ = _seed_model_with_bets(store, date, market_id)
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)

            resp = client.get(f"/replay/{model_id}/bets")
            assert resp.status_code == 200
            bet = resp.json()["bets"][0]
            # Fields must be present (even if null) so the frontend can rely
            # on reading them directly off the response.
            assert "fill_prob_at_placement" in bet
            assert "predicted_locked_pnl_at_placement" in bet
            assert "predicted_locked_stddev_at_placement" in bet

    def test_scalping_aux_head_values_preserved(self):
        """A record with all three set round-trips through the endpoint."""
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            date = "2026-03-26"
            market_id = "1.222"
            _create_tick_parquet(tmp, date, {market_id: 3})

            mid = store.create_model(0, "ppo_lstm_v1", "Test", {"lr": 0.001})
            rid = store.create_evaluation_run(mid, "2026-03-25", [date])
            store.record_evaluation_day(
                EvaluationDayRecord(
                    run_id=rid, date=date, day_pnl=3.50, bet_count=1,
                    winning_bets=1, bet_precision=1.0, pnl_per_bet=3.50,
                    early_picks=0, profitable=True,
                )
            )
            bets = [
                EvaluationBetRecord(
                    run_id=rid, date=date, market_id=market_id,
                    tick_timestamp="2026-03-26T14:00:05",
                    seconds_to_off=1795.0, runner_id=101,
                    runner_name="Fast Horse", action="back",
                    price=3.5, stake=10.0, matched_size=10.0,
                    outcome="won", pnl=3.50,
                    fill_prob_at_placement=0.85,
                    predicted_locked_pnl_at_placement=3.50,
                    predicted_locked_stddev_at_placement=1.25,
                ),
            ]
            store.write_bet_logs_parquet(rid, date, bets)

            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)
            resp = client.get(f"/replay/{mid}/bets")
            assert resp.status_code == 200
            bet = resp.json()["bets"][0]
            assert bet["fill_prob_at_placement"] == pytest.approx(0.85)
            assert bet["predicted_locked_pnl_at_placement"] == pytest.approx(3.50)
            assert bet["predicted_locked_stddev_at_placement"] == pytest.approx(1.25)

    def test_scalping_aux_head_null_when_missing(self):
        """Bets without aux-head predictions return null for the three fields."""
        with tempfile.TemporaryDirectory() as tmp:
            store = _create_store(tmp)
            date = "2026-03-26"
            market_id = "1.111"
            _create_tick_parquet(tmp, date, {market_id: 5})
            # _seed_model_with_bets() writes bets without the new aux-head
            # fields, so they default to None.
            model_id, _ = _seed_model_with_bets(store, date, market_id)
            config = {"paths": {"processed_data": str(Path(tmp) / "processed")}}
            client = _make_app(store, config)

            resp = client.get(f"/replay/{model_id}/bets")
            bet = resp.json()["bets"][0]
            assert bet["fill_prob_at_placement"] is None
            assert bet["predicted_locked_pnl_at_placement"] is None
            assert bet["predicted_locked_stddev_at_placement"] is None
