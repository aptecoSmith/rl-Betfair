"""
training/evaluator.py -- Evaluate a trained model on test days.

Runs a model independently on each test day (fresh budget, no memory across
days), records per-day metrics and full bet logs, and writes everything to
the model registry.

Usage::

    evaluator = Evaluator(config, model_store, progress_queue=queue)
    run_id, day_records = evaluator.evaluate(
        model_id="abc123",
        policy=policy,
        test_days=days,
        train_cutoff_date="2026-03-26",
    )
"""

from __future__ import annotations

import asyncio
import logging
import time

from training.perf_log import perf_log

import numpy as np
import torch

from agents.policy_network import BasePolicy, PolicyOutput
from data.episode_builder import Day
from env.bet_manager import BetOutcome, BetSide
from env.betfair_env import BetfairEnv
import pandas as pd

from registry.model_store import (
    EvaluationBetRecord,
    EvaluationDayRecord,
    ModelStore,
)
from training.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate a trained policy on held-out test days.

    Each day is run as an independent episode (budget reset, no LSTM carry-over
    between days).  Per-day metrics and individual bet records are written to
    the model registry.

    Parameters
    ----------
    config : dict
        Project config (from config.yaml).
    model_store : ModelStore | None
        Registry for persisting evaluation results.  Pass None to skip
        persistence (useful for unit tests).
    progress_queue : asyncio.Queue | None
        If provided, progress events are published here for WebSocket.
    device : str
        PyTorch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        config: dict,
        model_store: ModelStore | None = None,
        progress_queue: asyncio.Queue | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.model_store = model_store
        self.progress_queue = progress_queue
        self.device = device

    def evaluate(
        self,
        model_id: str,
        policy: BasePolicy,
        test_days: list[Day],
        train_cutoff_date: str,
    ) -> tuple[str | None, list[EvaluationDayRecord]]:
        """Run the policy on each test day and record results.

        Parameters
        ----------
        model_id :
            Registry ID of the model being evaluated.
        policy :
            The trained policy network (already loaded with correct weights).
        test_days :
            Days to evaluate on (each is a fresh episode).
        train_cutoff_date :
            Last training day — stored in the evaluation run record.

        Returns
        -------
        (run_id, day_records)
            run_id is the evaluation run UUID (None if no model_store).
            day_records is the list of per-day metric records.
        """
        if not test_days:
            return None, []

        policy = policy.to(self.device)
        policy.eval()

        # Create evaluation run in registry
        test_dates = [d.date for d in test_days]
        run_id = None
        if self.model_store is not None:
            run_id = self.model_store.create_evaluation_run(
                model_id=model_id,
                train_cutoff_date=train_cutoff_date,
                test_days=test_dates,
            )

        tracker = ProgressTracker(
            total=len(test_days),
            label=f"Evaluating {model_id[:12]}",
        )
        tracker.reset_timer()

        day_records: list[EvaluationDayRecord] = []

        eval_start = time.perf_counter()

        for day in test_days:
            day_start = time.perf_counter()
            day_record, bet_records = self._evaluate_day(
                policy, day, run_id or "",
            )
            day_elapsed = time.perf_counter() - day_start
            logger.info(
                "Eval day %s: %.2fs | pnl=%+.2f bets=%d",
                day.date, day_elapsed, day_record.day_pnl, day_record.bet_count,
            )
            day_records.append(day_record)

            # Persist to registry
            if self.model_store is not None and run_id is not None:
                self.model_store.record_evaluation_day(day_record)
                if bet_records:
                    self.model_store.write_bet_logs_parquet(
                        run_id, day.date, bet_records,
                    )

            tracker.tick()
            self._publish_progress(tracker, day_record)

        eval_elapsed = time.perf_counter() - eval_start
        logger.info(
            "Evaluation of %s complete: %d days in %.2fs",
            model_id[:12], len(test_days), eval_elapsed,
        )

        return run_id, day_records

    def _evaluate_day(
        self,
        policy: BasePolicy,
        day: Day,
        run_id: str,
    ) -> tuple[EvaluationDayRecord, list[EvaluationBetRecord]]:
        """Run one episode (one day) in eval mode and collect metrics."""
        env = BetfairEnv(day, self.config)
        obs, info = env.reset()

        hidden_state = policy.init_hidden(batch_size=1)
        hidden_state = (
            hidden_state[0].to(self.device),
            hidden_state[1].to(self.device),
        )

        done = False
        with torch.no_grad():
            while not done:
                obs_tensor = torch.as_tensor(
                    obs, dtype=torch.float32, device=self.device,
                ).unsqueeze(0)

                out: PolicyOutput = policy(obs_tensor, hidden_state)
                hidden_state = out.hidden_state

                # Deterministic action (use mean, no sampling)
                action = out.action_mean.squeeze(0).cpu().numpy()
                action = np.clip(action, -1.0, 1.0)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

        # Extract metrics from the env's final info
        bm = env.bet_manager
        assert bm is not None

        day_pnl = bm.realised_pnl
        bet_count = bm.bet_count
        winning_bets = bm.winning_bets
        bet_precision = winning_bets / bet_count if bet_count > 0 else 0.0
        pnl_per_bet = day_pnl / bet_count if bet_count > 0 else 0.0

        # Count early picks from race records
        race_records = info.get("race_records", [])
        early_picks = sum(rr.early_picks for rr in race_records)

        day_record = EvaluationDayRecord(
            run_id=run_id,
            date=day.date,
            day_pnl=day_pnl,
            bet_count=bet_count,
            winning_bets=winning_bets,
            bet_precision=bet_precision,
            pnl_per_bet=pnl_per_bet,
            early_picks=early_picks,
            profitable=day_pnl > 0,
        )

        # Build bet records from bet manager
        bet_records: list[EvaluationBetRecord] = []
        for bet in bm.bets:
            # Find the race this bet belongs to
            race = None
            for r in day.races:
                if r.market_id == bet.market_id:
                    race = r
                    break

            # Determine seconds_to_off (approximate — from env's bet_times if available)
            seconds_to_off = 0.0
            runner_name = ""

            if race is not None:
                # Look up runner name from metadata
                meta = race.runner_metadata.get(bet.selection_id)
                if meta is not None:
                    runner_name = meta.runner_name

            bet_records.append(EvaluationBetRecord(
                run_id=run_id,
                date=day.date,
                market_id=bet.market_id,
                tick_timestamp="",  # not tracked at eval level
                seconds_to_off=seconds_to_off,
                runner_id=bet.selection_id,
                runner_name=runner_name,
                action=bet.side.value,
                price=bet.average_price,
                stake=bet.matched_stake,
                matched_size=bet.matched_stake,
                outcome=bet.outcome.value,
                pnl=bet.pnl,
            ))

        logger.info(
            "Eval %s | pnl=%.2f bets=%d winning=%d precision=%.2f",
            day.date, day_pnl, bet_count, winning_bets, bet_precision,
        )

        return day_record, bet_records

    def _publish_progress(
        self,
        tracker: ProgressTracker,
        day_record: EvaluationDayRecord,
    ) -> None:
        """Publish an evaluation progress event to the queue."""
        event = {
            "event": "progress",
            "phase": "evaluating",
            "item": tracker.to_dict(),
            "detail": (
                f"Eval day {day_record.date} | "
                f"pnl={day_record.day_pnl:+.2f} | "
                f"bets={day_record.bet_count}"
            ),
        }
        if self.progress_queue is not None:
            try:
                self.progress_queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
