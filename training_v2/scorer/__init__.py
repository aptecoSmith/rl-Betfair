"""training_v2/scorer — Phase 0 supervised scorer.

Standalone supervised pipeline for the Phase 0 deliverable described in
``plans/rewrite/phase-0-supervised-scorer/purpose.md``. Builds a labelled,
featurised parquet dataset of historical opening opportunities; the
trained scorer feeds Phase 1's actor as a frozen feature.

No RL is touched in this module. The label simulator reuses
``env.exchange_matcher``, ``env.scalping_math``, and ``env.bet_manager``
verbatim — re-implementing matching / sizing / force-close logic would
silently diverge from the env's runtime behaviour and miscalibrate the
scorer.
"""

from training_v2.scorer.feature_extractor import (
    FEATURE_NAMES,
    FeatureExtractor,
)
from training_v2.scorer.label_generator import (
    LabelGenerator,
    LabelOutcome,
)

__all__ = [
    "FEATURE_NAMES",
    "FeatureExtractor",
    "LabelGenerator",
    "LabelOutcome",
]
