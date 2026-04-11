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
from data.episode_builder import Day, Race
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


def compute_opportunity_window(
    race: Race,
    tick_index: int,
    selection_id: int,
    side: str,
    price: float,
) -> float:
    """Compute how many seconds a bet's price was available in the order book.

    Scans backward and forward from *tick_index* through *race.ticks*,
    counting consecutive ticks where price *price* (or better) was available
    for the runner with *selection_id*.

    "Better" means:
    - **Back** at price P: ``available_to_back`` has an entry with ``price >= P``
    - **Lay** at price P: ``available_to_lay`` has an entry with ``price <= P``

    Returns the total duration in seconds (from first available tick to last).
    Returns 0.0 if *tick_index* is -1 (not recorded) or the race has no ticks.
    """
    if tick_index < 0 or not race.ticks:
        return 0.0

    ticks = race.ticks

    def _price_available(tick_idx: int) -> bool:
        """Check if the price is available for this runner at the given tick."""
        tick = ticks[tick_idx]
        # Fast path: find runner by selection_id without scanning all runners
        for runner in tick.runners:
            if runner.selection_id == selection_id:
                if side == "back":
                    return any(ps.price >= price for ps in runner.available_to_back)
                else:  # lay
                    return any(ps.price <= price for ps in runner.available_to_lay)
                # Found the runner, no need to continue
                break
        return False

    # Scan backward
    first_idx = tick_index
    for i in range(tick_index - 1, -1, -1):
        if ticks[i].in_play:
            break
        if _price_available(i):
            first_idx = i
        else:
            break

    # Scan forward
    last_idx = tick_index
    for i in range(tick_index + 1, len(ticks)):
        if ticks[i].in_play:
            break
        if _price_available(i):
            last_idx = i
        else:
            break

    # Convert to seconds using tick timestamps
    duration = (ticks[last_idx].timestamp - ticks[first_idx].timestamp).total_seconds()
    return max(duration, 0.0)


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
        feature_cache: dict[str, list] | None = None,
    ) -> None:
        self.config = config
        self.model_store = model_store
        self.progress_queue = progress_queue
        self.device = device
        self.feature_cache = feature_cache

    def evaluate(
        self,
        model_id: str,
        policy: BasePolicy,
        test_days: list[Day],
        train_cutoff_date: str,
        market_type_filter: str = "BOTH",
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
                market_type_filter=market_type_filter,
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
        market_type_filter: str = "BOTH",
    ) -> tuple[EvaluationDayRecord, list[EvaluationBetRecord]]:
        """Run one episode (one day) in eval mode and collect metrics."""
        t_env_start = time.perf_counter()
        env = BetfairEnv(day, self.config, feature_cache=self.feature_cache,
                         emit_debug_features=False,
                         market_type_filter=market_type_filter)
        obs, info = env.reset()
        t_env_ready = time.perf_counter()

        hidden_state = policy.init_hidden(batch_size=1)
        hidden_state = (
            hidden_state[0].to(self.device),
            hidden_state[1].to(self.device),
        )

        # Pre-allocate reusable GPU buffer
        obs_dim = obs.shape[0]
        obs_buffer = torch.empty(
            1, obs_dim, dtype=torch.float32, device=self.device,
        )

        n_ticks = sum(len(r.ticks) for r in day.races)
        logger.info(
            "Eval starting %s: %d races, %d ticks, device=%s, "
            "policy_device=%s, obs_buffer_device=%s",
            day.date, len(day.races), n_ticks,
            self.device,
            next(policy.parameters()).device,
            obs_buffer.device,
        )
        # Also emit as progress event
        if self.progress_queue is not None:
            try:
                self.progress_queue.put_nowait({
                    "event": "progress",
                    "phase": "evaluating",
                    "detail": (
                        f"Eval starting {day.date}: {len(day.races)} races, "
                        f"{n_ticks} ticks, device={self.device}, "
                        f"policy={next(policy.parameters()).device}"
                    ),
                })
            except Exception:
                pass
        done = False
        steps = 0
        log_interval = 100  # frequent progress for debugging
        slow_step_threshold = 5.0  # log any step taking > 5 seconds
        with torch.no_grad():
            while not done:
                step_t0 = time.perf_counter()
                obs_buffer[0] = torch.as_tensor(obs, dtype=torch.float32)

                out: PolicyOutput = policy(obs_buffer, hidden_state)
                hidden_state = out.hidden_state

                # Deterministic action (use mean, no sampling)
                action = out.action_mean.squeeze(0).cpu().numpy()
                np.clip(action, -1.0, 1.0, out=action)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                step_dur = time.perf_counter() - step_t0

                if step_dur > slow_step_threshold:
                    logger.warning(
                        "SLOW STEP %d: %.1fs (race=%d tick=%d)",
                        steps, step_dur, env._race_idx, env._tick_idx,
                    )
                    if self.progress_queue is not None:
                        try:
                            self.progress_queue.put_nowait({
                                "event": "progress",
                                "phase": "evaluating",
                                "detail": f"SLOW STEP {steps}: {step_dur:.1f}s race={env._race_idx}",
                            })
                        except Exception:
                            pass

                if steps % log_interval == 0:
                    elapsed = time.perf_counter() - t_env_ready
                    rate = steps / max(elapsed, 0.001)
                    step_detail = (
                        f"Eval {day.date} step {steps}/{n_ticks} "
                        f"({elapsed:.1f}s, {int(rate)}/s) "
                        f"race={env._race_idx}/{env._total_races}"
                    )
                    logger.info(step_detail)
                    if self.progress_queue is not None:
                        try:
                            self.progress_queue.put_nowait({
                                "event": "progress",
                                "phase": "evaluating",
                                "detail": step_detail,
                            })
                        except Exception:
                            pass
        t_loop_done = time.perf_counter()

        # Extract metrics from the env.  ``bet_manager`` is recreated
        # between races, so it only contains the *last race's* bets —
        # use the full-day bet log and the env's accumulated day_pnl
        # instead.  See bugs.md B1.
        all_bets = env.all_settled_bets

        day_pnl = info.get("day_pnl", 0.0)
        bet_count = len(all_bets)
        winning_bets = sum(1 for b in all_bets if b.outcome is BetOutcome.WON)
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
            starting_budget=env.starting_budget,
        )

        t_metrics_done = time.perf_counter()

        # Build bet records from the full day's settled bets (across all
        # races, not just the last race's BetManager).
        race_by_market = {r.market_id: r for r in day.races}
        bet_records: list[EvaluationBetRecord] = []
        for bet in all_bets:
            race = race_by_market.get(bet.market_id)

            runner_name = ""
            tick_timestamp = ""
            seconds_to_off = 0.0
            opp_window = 0.0

            if race is not None:
                meta = race.runner_metadata.get(bet.selection_id)
                if meta is not None:
                    runner_name = meta.runner_name

                # Derive tick-level fields from stored tick_index
                if bet.tick_index >= 0 and bet.tick_index < len(race.ticks):
                    tick = race.ticks[bet.tick_index]
                    tick_timestamp = tick.timestamp.isoformat()
                    seconds_to_off = (
                        race.market_start_time - tick.timestamp
                    ).total_seconds()

                    opp_window = compute_opportunity_window(
                        race, bet.tick_index, bet.selection_id,
                        bet.side.value, bet.average_price,
                    )

            bet_records.append(EvaluationBetRecord(
                run_id=run_id,
                date=day.date,
                market_id=bet.market_id,
                tick_timestamp=tick_timestamp,
                seconds_to_off=seconds_to_off,
                runner_id=bet.selection_id,
                runner_name=runner_name,
                action=bet.side.value,
                price=bet.average_price,
                stake=bet.matched_stake,
                matched_size=bet.matched_stake,
                outcome=bet.outcome.value,
                pnl=bet.pnl,
                opportunity_window_s=opp_window,
                is_each_way=bet.is_each_way,
                each_way_divisor=bet.each_way_divisor,
                number_of_places=bet.number_of_places,
                settlement_type=bet.settlement_type,
                effective_place_odds=bet.effective_place_odds,
                starting_budget=env.starting_budget,
            ))

        # Compute opportunity window aggregates for the day record
        if bet_records:
            windows = [b.opportunity_window_s for b in bet_records]
            day_record.mean_opportunity_window_s = float(np.mean(windows))
            day_record.median_opportunity_window_s = float(np.median(windows))

        t_bets_done = time.perf_counter()

        step_rate = int(steps / max(t_loop_done - t_env_ready, 0.001))
        timing_detail = (
            f"Eval {day.date} done | "
            f"env_init={t_env_ready - t_env_start:.1f}s "
            f"step_loop={t_loop_done - t_env_ready:.1f}s "
            f"({steps} steps, {step_rate}/s) "
            f"bet_records={t_bets_done - t_metrics_done:.1f}s "
            f"total={t_bets_done - t_env_start:.1f}s"
        )
        logger.info(
            "Eval %s | pnl=%.2f bets=%d winning=%d precision=%.2f opp_window=%.1fs | %s",
            day.date, day_pnl, bet_count, winning_bets, bet_precision,
            day_record.mean_opportunity_window_s,
            timing_detail,
        )
        # Emit timing as a progress event so it's visible on WebSocket
        if self.progress_queue is not None:
            try:
                self.progress_queue.put_nowait({
                    "event": "progress",
                    "phase": "evaluating",
                    "detail": timing_detail,
                })
            except Exception:
                pass

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
