"""Predictor integration — wires `betfair-predictors` champions into rl-betfair.

Public surface (Session 01 of plans/predictor-integration/):
    PredictorBundle, RaceLevelOutputs, TickLevelOutputs, PredictorLoaderError
    SegmentRouter, ConsumerHint
"""

from predictors.loader import (
    PredictorBundle,
    PredictorLoaderError,
    RaceLevelOutputs,
    TickLevelOutputs,
)
from predictors.segment_router import ConsumerHint, SegmentRouter

__all__ = [
    "PredictorBundle",
    "PredictorLoaderError",
    "RaceLevelOutputs",
    "TickLevelOutputs",
    "SegmentRouter",
    "ConsumerHint",
]
