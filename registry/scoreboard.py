"""
registry/scoreboard.py -- Compute and rank models by composite score.

The composite score formula (from PLAN.md)::

    win_rate         = profitable_days / total_test_days
    mean_daily_pnl   = mean(day_pnl)
    sharpe           = mean(day_pnl) / std(day_pnl)
    bet_precision    = mean(winning_bets / bet_count)
    pnl_per_bet      = mean(day_pnl / bet_count)

    efficiency = (bet_precision * 0.5) + (normalised_pnl_per_bet * 0.5)

    composite_score = (win_rate       * w_win_rate)
                    + (sharpe_norm    * w_sharpe)
                    + (pnl_norm       * w_mean_daily_pnl)
                    + (efficiency     * w_efficiency)

Coefficients are configurable in config.yaml under ``reward.coefficients``.

Usage::

    board = Scoreboard(store, config)
    rankings = board.rank_all()
    board.update_scores()       # persists scores to registry
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from registry.model_store import EvaluationDayRecord, ModelRecord, ModelStore


@dataclass
class ModelScore:
    """Computed score breakdown for one model."""

    model_id: str
    win_rate: float
    mean_daily_pnl: float
    sharpe: float
    bet_precision: float
    pnl_per_bet: float
    efficiency: float
    composite_score: float
    test_days: int
    profitable_days: int
    total_early_picks: int = 0
    early_picks_per_day: float = 0.0
    mean_opportunity_window_s: float = 0.0
    mean_daily_return_pct: float | None = None
    recorded_budget: float | None = None


class Scoreboard:
    """Compute composite scores and rank all active models.

    Parameters
    ----------
    store : ModelStore
        The model registry.
    config : dict
        Project config (from config.yaml).
    """

    def __init__(self, store: ModelStore, config: dict) -> None:
        self.store = store
        coefficients = config.get("reward", {}).get("coefficients", {})
        self.w_win_rate = coefficients.get("win_rate", 0.35)
        self.w_sharpe = coefficients.get("sharpe", 0.30)
        self.w_mean_daily_pnl = coefficients.get("mean_daily_pnl", 0.15)
        self.w_efficiency = coefficients.get("efficiency", 0.20)
        self.starting_budget = config.get("training", {}).get("starting_budget", 100.0)

    def compute_score(self, days: list[EvaluationDayRecord]) -> ModelScore | None:
        """Compute the composite score from per-day evaluation records.

        Returns None if there are no evaluation days.
        """
        if not days:
            return None

        n = len(days)
        pnls = [d.day_pnl for d in days]
        profitable_days = sum(1 for d in days if d.profitable)

        win_rate = profitable_days / n

        mean_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls, ddof=0))
        sharpe = mean_pnl / std_pnl if std_pnl > 1e-8 else 0.0

        # Bet precision: mean across days of (winning_bets / bet_count)
        precisions = []
        pnl_per_bets = []
        for d in days:
            if d.bet_count > 0:
                precisions.append(d.winning_bets / d.bet_count)
                pnl_per_bets.append(d.day_pnl / d.bet_count)
            else:
                precisions.append(0.0)
                pnl_per_bets.append(0.0)

        bet_precision = float(np.mean(precisions))
        pnl_per_bet = float(np.mean(pnl_per_bets))

        # Use the recorded budget from day records (they should all be
        # the same for a given model's evaluation run). Fall back to the
        # global config budget for backward compat with old records that
        # default to 100.0.
        recorded_budget = days[0].starting_budget if days else self.starting_budget
        budget_for_norm = recorded_budget if recorded_budget > 0 else self.starting_budget

        # Normalise components to roughly [-1, 1] range
        sharpe_norm = np.clip(sharpe / 3.0, -1.0, 1.0)  # sharpe >3 is exceptional
        pnl_norm = np.clip(mean_pnl / budget_for_norm, -1.0, 1.0)

        # Normalise pnl_per_bet for efficiency calc
        pnl_per_bet_norm = np.clip(pnl_per_bet / (budget_for_norm * 0.1), -1.0, 1.0)

        # Percentage return
        mean_daily_return_pct = (mean_pnl / budget_for_norm) * 100

        efficiency = bet_precision * 0.5 + float(pnl_per_bet_norm) * 0.5

        composite = (
            win_rate * self.w_win_rate
            + float(sharpe_norm) * self.w_sharpe
            + float(pnl_norm) * self.w_mean_daily_pnl
            + efficiency * self.w_efficiency
        )

        # Early picks: informational only, not part of composite score
        total_early_picks = sum(d.early_picks for d in days)
        early_picks_per_day = total_early_picks / n

        # Opportunity window: informational only, not part of composite score
        opp_windows = [d.mean_opportunity_window_s for d in days]
        mean_opp_window = float(np.mean(opp_windows)) if opp_windows else 0.0

        return ModelScore(
            model_id="",  # filled by caller
            win_rate=win_rate,
            mean_daily_pnl=mean_pnl,
            sharpe=sharpe,
            bet_precision=bet_precision,
            pnl_per_bet=pnl_per_bet,
            efficiency=efficiency,
            composite_score=composite,
            test_days=n,
            profitable_days=profitable_days,
            total_early_picks=total_early_picks,
            early_picks_per_day=early_picks_per_day,
            mean_opportunity_window_s=mean_opp_window,
            mean_daily_return_pct=mean_daily_return_pct,
            recorded_budget=recorded_budget,
        )

    def score_model(self, model_id: str) -> ModelScore | None:
        """Compute the score for a single model from its latest evaluation."""
        run = self.store.get_latest_evaluation_run(model_id)
        if run is None:
            return None

        days = self.store.get_evaluation_days(run.run_id)
        score = self.compute_score(days)
        if score is not None:
            score.model_id = model_id
        return score

    def rank_all(self) -> list[ModelScore]:
        """Score and rank all active + garaged models.  Returns sorted by composite_score desc."""
        models = self.store.list_models(status="active")
        seen = {m.model_id for m in models}
        # Include garaged models even if discarded
        for g in self.store.list_garaged_models():
            if g.model_id not in seen:
                models.append(g)
                seen.add(g.model_id)

        scores: list[ModelScore] = []
        for m in models:
            s = self.score_model(m.model_id)
            if s is not None:
                scores.append(s)

        scores.sort(key=lambda s: s.composite_score, reverse=True)
        return scores

    def update_scores(self) -> list[ModelScore]:
        """Recompute and persist scores for all active models.

        Returns the ranked list.
        """
        rankings = self.rank_all()
        for s in rankings:
            self.store.update_composite_score(s.model_id, s.composite_score)
        return rankings

    def check_discard_candidates(self, config: dict) -> list[str]:
        """Return model IDs that meet all discard criteria.

        A model is a discard candidate only if ALL of these are true:
        - win_rate < min_win_rate
        - P&L below threshold (percentage-based if ``min_mean_return_pct``
          is set, else falls back to absolute ``min_mean_pnl``)
        - sharpe < min_sharpe
        """
        dp = config.get("discard_policy", {})
        min_wr = dp.get("min_win_rate", 0.35)
        min_pnl = dp.get("min_mean_pnl", 0.0)
        min_return_pct = dp.get("min_mean_return_pct")
        min_sharpe = dp.get("min_sharpe", -0.5)

        candidates = []
        models = self.store.list_models(status="active")
        for m in models:
            s = self.score_model(m.model_id)
            if s is None:
                continue
            # P&L check: prefer percentage threshold if configured
            if min_return_pct is not None and s.mean_daily_return_pct is not None:
                pnl_below = s.mean_daily_return_pct < min_return_pct
            else:
                pnl_below = s.mean_daily_pnl < min_pnl
            if s.win_rate < min_wr and pnl_below and s.sharpe < min_sharpe:
                candidates.append(m.model_id)

        return candidates
