"""PredictorBundle — loads the three `betfair-predictors` champions at startup.

Per `plans/predictor-integration/integration_contract.md` §1, the bundle
exposes:

    bundle = PredictorBundle.from_manifests(champion_manifest=..., ranker_manifest=..., direction_manifest=...)
    race_outputs = bundle.predict_race(race_card)        # cached by market_id
    tick_outputs = bundle.predict_tick(runner, ladder_window)  # per-call

The loader appends the sibling `betfair-predictors` repo to `sys.path` so
the inference factories (`scripts.predictor.models.build_model`,
`scripts.outcome_predictor.datasets.numeric_feature_matrix`) load against
the manifest's `weights_path`.

This module is the standalone Session-01 deliverable: env, trainer, and
config-flag plumbing land in Session 02. No env imports here.

Hard constraints honoured (see hard_constraints.md):
- §4 predictors frozen — loader is read-only.
- §10 loader robustness — silent fallback forbidden; raise loudly on
  missing manifest, schema mismatch, or weights mismatch.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from predictors.segment_router import ConsumerHint, SegmentRouter


class PredictorLoaderError(RuntimeError):
    """Raised when a predictor cannot be loaded.

    Per hard_constraints §10 the loader does NOT silently fall back —
    a corrupt manifest, missing weights, or schema mismatch surfaces
    as this exception so the trainer refuses to start.
    """


# Sibling-repo import path (master_todo.md "After Session 01" decision: stick with
# sys.path.insert until a third consumer makes packaging worthwhile).
_RL_REPO_ROOT = Path(__file__).resolve().parents[1]
_BETFAIR_PREDICTORS_REPO = _RL_REPO_ROOT.parent / "betfair-predictors"


def _ensure_betfair_predictors_on_path() -> Path:
    """Append the sibling repo to sys.path; return its root.

    Failure mode: the repo doesn't exist on disk → raise loudly. This is
    the only acceptable outcome per §10 — silent fallback would leave
    the integration claiming to be enabled while predictor calls would
    return sentinel zeroes.
    """
    if not _BETFAIR_PREDICTORS_REPO.exists():
        raise PredictorLoaderError(
            f"betfair-predictors sibling repo not found at "
            f"{_BETFAIR_PREDICTORS_REPO}. The predictor integration "
            f"requires the sibling repo to be present."
        )
    repo_str = str(_BETFAIR_PREDICTORS_REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return _BETFAIR_PREDICTORS_REPO


# ---------------------------------------------------------------------------
# Output dataclasses (see integration_contract.md §1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RaceLevelOutputs:
    """Per-runner, per-race outputs from champion + ranker.

    Computed once per market at race-card-known time; cached on the
    bundle keyed by market_id.
    """

    p_win: dict[int, float]
    p_placed: dict[int, float]
    ranker_score: dict[int, float]
    ranker_rank: dict[int, int]
    ranker_softmax_share: dict[int, float]
    ranker_top1_flag: dict[int, bool]
    ranker_top1_high_confidence_flag: dict[int, bool]
    segment_strong_flag: dict[int, bool]


@dataclass(frozen=True)
class TickLevelOutputs:
    """Per-runner, per-tick outputs from the direction predictor.

    Computed each tick; not cached (cheap forward over a 32x26 window).
    """

    q10_1m: float
    q50_1m: float
    q90_1m: float
    q10_3m: float
    q50_3m: float
    q90_3m: float
    q10_7m: float
    q50_7m: float
    q90_7m: float
    fire_drift: bool
    fire_shorten: bool
    fire_no_signal: bool


# ---------------------------------------------------------------------------
# Manifest reading (Session 01 lays the contract; Session 02 wires the env)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Manifest:
    """In-memory view of one production manifest.json."""

    path: Path
    payload: dict[str, Any]
    experiment_id: str
    weights_path: Path
    architecture_family: str
    architecture_kwargs: dict[str, Any]
    feature_columns_source: str | None


_REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "experiment_id",
    "weights_path",
    "architecture",
)


def _read_manifest(manifest_path: Path, repo_root: Path) -> _Manifest:
    if not manifest_path.exists():
        raise PredictorLoaderError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    for key in _REQUIRED_MANIFEST_KEYS:
        if key not in payload:
            raise PredictorLoaderError(
                f"manifest {manifest_path} missing required key {key!r}"
            )

    arch = payload["architecture"]
    if "family" not in arch or "kwargs" not in arch:
        raise PredictorLoaderError(
            f"manifest {manifest_path} architecture block missing "
            f"'family' or 'kwargs'"
        )

    weights_rel = payload["weights_path"]
    weights_path = (repo_root / weights_rel).resolve()
    if not weights_path.exists():
        raise PredictorLoaderError(
            f"weights file not found at {weights_path} "
            f"(referenced by {manifest_path})"
        )

    feature_source = (
        payload.get("input_shape", {}).get("feature_columns_source")
    )

    return _Manifest(
        path=manifest_path,
        payload=payload,
        experiment_id=str(payload["experiment_id"]),
        weights_path=weights_path,
        architecture_family=str(arch["family"]),
        architecture_kwargs=dict(arch["kwargs"]),
        feature_columns_source=feature_source,
    )


# ---------------------------------------------------------------------------
# Model payload loaders — eager at PredictorBundle.from_manifests time
# ---------------------------------------------------------------------------


_EXPECTED_CHAMPION_KEYS: tuple[str, ...] = ("win", "placed", "feature_names")
_EXPECTED_RANKER_KEYS: tuple[str, ...] = (
    "win_ranker",
    "placed_model",
    "feature_names",
)


def _load_champion(manifest: _Manifest) -> _ChampionPayload:
    """Load the GBM two-head champion (`production/race-outcome/weights.joblib`).

    The pickled payload is a plain dict. We validate the documented keys
    and refuse on missing keys (hard_constraints §10).
    """
    import joblib  # local import keeps the loader's startup graph small

    try:
        payload = joblib.load(manifest.weights_path)
    except Exception as exc:
        raise PredictorLoaderError(
            f"failed to joblib.load champion weights at {manifest.weights_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise PredictorLoaderError(
            f"champion weights at {manifest.weights_path} unpickled as "
            f"{type(payload).__name__}, expected dict"
        )
    for k in _EXPECTED_CHAMPION_KEYS:
        if k not in payload:
            raise PredictorLoaderError(
                f"champion weights missing expected key {k!r}; got {list(payload)}"
            )
    feat_variant = manifest.payload.get("training", {}).get("feature_variant")
    if feat_variant is None:
        raise PredictorLoaderError(
            f"champion manifest {manifest.path} missing training.feature_variant"
        )
    return _ChampionPayload(
        win_model=payload["win"],
        placed_model=payload["placed"],
        feature_names=tuple(payload["feature_names"]),
        feature_variant=str(feat_variant),
    )


def _load_ranker(manifest: _Manifest) -> _RankerPayload:
    """Load the lambdarank companion."""
    import joblib

    try:
        payload = joblib.load(manifest.weights_path)
    except Exception as exc:
        raise PredictorLoaderError(
            f"failed to joblib.load ranker weights at {manifest.weights_path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise PredictorLoaderError(
            f"ranker weights at {manifest.weights_path} unpickled as "
            f"{type(payload).__name__}, expected dict"
        )
    for k in _EXPECTED_RANKER_KEYS:
        if k not in payload:
            raise PredictorLoaderError(
                f"ranker weights missing expected key {k!r}; got {list(payload)}"
            )
    feat_variant = manifest.payload.get("training", {}).get("feature_variant")
    if feat_variant is None:
        raise PredictorLoaderError(
            f"ranker manifest {manifest.path} missing training.feature_variant"
        )
    return _RankerPayload(
        win_ranker=payload["win_ranker"],
        placed_model=payload["placed_model"],
        feature_names=tuple(payload["feature_names"]),
        feature_variant=str(feat_variant),
    )


def _fit_categorical_encoder(train_corpus: str, feature_variant: str) -> Any:
    """Fit the F1-categorical encoder against the predictor repo's training shards.

    The joblib payload doesn't persist the encoder state (see
    `betfair-predictors/incoming/persist_encoder_state_alongside_weights.md`).
    Fitting takes ~1-2s on the `last_12m` corpus and is done once per
    worker at bundle-construction time. Cold-start values map to
    `<UNKNOWN>` per the predictor repo's §9 contract.

    Hard_constraints §4: predictors are FROZEN — the encoder is fit
    against the SAME training corpus the model trained on, not against
    rl-betfair runtime data. This is the canonical inference-time
    encoder; once fit, it is held read-only on the bundle.
    """
    _ensure_betfair_predictors_on_path()
    from scripts.outcome_predictor.datasets import (  # type: ignore[import-not-found]
        DEFAULT_DATASET_DIR,
        fit_encoders,
        load_split,
    )

    if not DEFAULT_DATASET_DIR.exists():
        raise PredictorLoaderError(
            f"predictor training corpus not found at {DEFAULT_DATASET_DIR}; "
            f"the rl-betfair worker needs the betfair-predictors repo's "
            f"data/outcome_dataset/ shards available to fit categorical "
            f"encoders at bundle startup. (Workaround documented in "
            f"betfair-predictors/incoming/persist_encoder_state_alongside_weights.md.)"
        )
    try:
        train_df = load_split(
            split_name="train",
            feature_variant=feature_variant,
            train_corpus=train_corpus,
        )
    except Exception as exc:
        raise PredictorLoaderError(
            f"failed to load predictor training corpus for encoder fit "
            f"(variant={feature_variant!r}, corpus={train_corpus!r}): {exc}"
        ) from exc

    return fit_encoders(train_df, variant=feature_variant)


def _load_direction(manifest: _Manifest) -> _DirectionPayload:
    """Load the Conv1D price-mover champion.

    Resolves the model factory through the sibling repo's
    `scripts.predictor.models.build_model` per
    `intended_consumer.md` §"What rl-betfair needs from a release".
    """
    import torch

    # The factory module needs the sibling repo on sys.path; the bundle
    # entry-point already arranged this, but loaders may be called in
    # isolation (e.g. by tests), so re-arrange here.
    _ensure_betfair_predictors_on_path()
    from scripts.predictor.datasets import feature_columns
    from scripts.predictor.models import build_model

    feature_variant = manifest.payload.get("training", {}).get("feature_variant")
    if feature_variant is None:
        raise PredictorLoaderError(
            f"direction manifest {manifest.path} missing training.feature_variant"
        )
    try:
        feat_cols = feature_columns(feature_variant)
    except ValueError as exc:
        raise PredictorLoaderError(
            f"direction manifest references unknown feature_variant "
            f"{feature_variant!r}: {exc}"
        ) from exc

    n_features_expected = manifest.payload.get("input_shape", {}).get("n_features")
    if n_features_expected is not None and len(feat_cols) != n_features_expected:
        raise PredictorLoaderError(
            f"direction manifest n_features={n_features_expected} disagrees with "
            f"feature_columns({feature_variant!r})={len(feat_cols)}"
        )

    horizons = tuple(manifest.payload.get("training", {}).get("horizons", []))
    quantiles = tuple(manifest.payload.get("training", {}).get("quantiles", []))
    time_window = manifest.payload.get("input_shape", {}).get("time_window")
    if not horizons or not quantiles or time_window is None:
        raise PredictorLoaderError(
            f"direction manifest {manifest.path} missing horizons / quantiles "
            f"/ time_window in training/input_shape blocks"
        )

    try:
        model = build_model(
            family=manifest.architecture_family,
            n_features=len(feat_cols),
            n_horizons=len(horizons),
            n_quantiles=len(quantiles),
            arch_kwargs=manifest.architecture_kwargs,
        )
    except Exception as exc:
        raise PredictorLoaderError(
            f"direction model factory failed: {exc}"
        ) from exc

    try:
        state_dict = torch.load(
            manifest.weights_path, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise PredictorLoaderError(
            f"failed to torch.load direction weights at "
            f"{manifest.weights_path}: {exc}"
        ) from exc

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise PredictorLoaderError(
            f"direction state_dict mismatched architecture "
            f"({manifest.architecture_family} {manifest.architecture_kwargs}): {exc}"
        ) from exc
    model.eval()

    return _DirectionPayload(
        model=model,
        n_features=len(feat_cols),
        n_horizons=len(horizons),
        n_quantiles=len(quantiles),
        quantiles=tuple(float(q) for q in quantiles),
        horizons=tuple(str(h) for h in horizons),
        time_window=int(time_window),
        feature_variant=str(feature_variant),
    )


# ---------------------------------------------------------------------------
# PredictorBundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ChampionPayload:
    """In-memory unpickled champion (`production/race-outcome/weights.joblib`).

    The training-side wrapper is `scripts.outcome_predictor.models._GBMTwoHead`
    but joblib stores its `__dict__` projection: a plain dict carrying both
    LGBMClassifier heads + the feature-name list.

    The categorical `encoder_state` is NOT in the joblib payload (see
    `betfair-predictors/incoming/persist_encoder_state_alongside_weights.md`).
    The bundle lazy-fits it from the training corpus at startup; the
    fit state is held on the bundle's `categorical_encoder` field and
    threaded through to `predict_race` at call time.
    """

    win_model: Any  # LGBMClassifier; .predict_proba(X)[:, 1] -> P(win)
    placed_model: Any  # LGBMClassifier; .predict_proba(X)[:, 1] -> P(placed)
    feature_names: tuple[str, ...]
    feature_variant: str  # "F2" for the production champion


@dataclass(frozen=True)
class _RankerPayload:
    """In-memory unpickled ranker (`production/race-outcome-ranker/weights.joblib`)."""

    win_ranker: Any  # LGBMRanker; .predict(X) -> raw lambdarank scores
    placed_model: Any  # LGBMClassifier; .predict_proba(X)[:, 1] -> P(placed)
    feature_names: tuple[str, ...]
    feature_variant: str  # "F5" for the production ranker


@dataclass(frozen=True)
class _DirectionPayload:
    """In-memory loaded direction predictor (Conv1D state_dict + arch)."""

    model: Any  # torch.nn.Module set to eval mode
    n_features: int
    n_horizons: int
    n_quantiles: int
    quantiles: tuple[float, ...]
    horizons: tuple[str, ...]
    time_window: int
    feature_variant: str = "V2"  # 2026-05-22: env builder uses this to
    # build the right (32, D) window shape. V2/V3/V4 supported.


@dataclass
class PredictorBundle:
    """Container for the three production predictors + their segment routers.

    Constructed via :py:meth:`from_manifests`. Model payloads load eagerly
    at startup so any failure (missing weights, version skew, bad
    architecture) surfaces immediately per hard_constraints §10.

    `champion_encoder` / `ranker_encoder` are
    `scripts.outcome_predictor.datasets.EncoderState` instances fit lazily
    on the predictor repo's training corpus at bundle-construction time
    (the joblib payloads do not persist them — see
    `betfair-predictors/incoming/persist_encoder_state_alongside_weights.md`).
    """

    champion_manifest: _Manifest
    ranker_manifest: _Manifest
    direction_manifest: _Manifest
    champion_segments: SegmentRouter
    ranker_segments: SegmentRouter
    champion: _ChampionPayload
    ranker: _RankerPayload
    direction: _DirectionPayload
    champion_encoder: Any  # scripts.outcome_predictor.datasets.EncoderState
    ranker_encoder: Any  # scripts.outcome_predictor.datasets.EncoderState
    # Per-race cache populated by predict_race; key = market_id.
    _race_cache: dict[str, RaceLevelOutputs] = field(default_factory=dict)

    # ---- experiment-id tags for registry capture (hard_constraints §7) ----

    @property
    def champion_experiment_id(self) -> str:
        return self.champion_manifest.experiment_id

    @property
    def ranker_experiment_id(self) -> str:
        return self.ranker_manifest.experiment_id

    @property
    def direction_experiment_id(self) -> str:
        return self.direction_manifest.experiment_id

    def validate_compatibility(
        self,
        cohort_hp: dict,
    ) -> None:
        """Refuse loudly if a stored cohort row's predictor
        experiment_ids don't match this live bundle.

        Honours `hard_constraints.md §7`: two cohort runs with different
        predictor `experiment_id`s are NOT cross-comparable. Re-eval
        tooling reads the cohort row's stored ids and calls this method
        to refuse a re-eval where the operator's current bundle
        diverges from the cohort's recorded versions.

        Cohort rows that pre-date this contract (no
        `predictor_*_experiment_id` keys in `hp`) pass through cleanly —
        the legacy interpretation is "this cohort didn't use predictors".

        Cohort rows whose stored experiment_id is empty string also pass
        (a flag-off cohort that landed AFTER this contract carries empty
        strings; that's still compatible with any live bundle).
        """
        for label, attr, key in (
            ("champion",
             "champion_experiment_id",
             "predictor_champion_experiment_id"),
            ("ranker",
             "ranker_experiment_id",
             "predictor_ranker_experiment_id"),
            ("direction",
             "direction_experiment_id",
             "predictor_direction_experiment_id"),
        ):
            stored = cohort_hp.get(key)
            if stored is None or stored == "":
                continue
            live = getattr(self, attr)
            if str(stored) != str(live):
                raise PredictorLoaderError(
                    f"predictor {label} experiment_id mismatch: cohort row "
                    f"recorded {stored!r}, live bundle has {live!r}. "
                    f"hard_constraints.md §7 forbids cross-comparing cohorts "
                    f"trained against different predictor versions. "
                    f"Re-pin the live bundle (or re-train the cohort) before "
                    f"re-evaluating."
                )

    @classmethod
    def from_manifests(
        cls,
        champion_manifest: Path | str,
        ranker_manifest: Path | str,
        direction_manifest: Path | str,
    ) -> "PredictorBundle":
        repo_root = _ensure_betfair_predictors_on_path()

        champ_m = _read_manifest(Path(champion_manifest), repo_root=repo_root)
        rank_m = _read_manifest(Path(ranker_manifest), repo_root=repo_root)
        dir_m = _read_manifest(Path(direction_manifest), repo_root=repo_root)

        # Sidecar segment routers — same directory as the manifest.
        champ_segments_path = champ_m.path.parent / "segment_performance.json"
        rank_segments_path = rank_m.path.parent / "segment_performance.json"
        if not champ_segments_path.exists():
            raise PredictorLoaderError(
                f"champion segment_performance.json missing at {champ_segments_path}"
            )
        if not rank_segments_path.exists():
            raise PredictorLoaderError(
                f"ranker segment_performance.json missing at {rank_segments_path}"
            )

        champ_segments = SegmentRouter.from_path(champ_segments_path)
        rank_segments = SegmentRouter.from_path(rank_segments_path)

        champion_payload = _load_champion(champ_m)
        ranker_payload = _load_ranker(rank_m)
        direction_payload = _load_direction(dir_m)

        champ_encoder = _fit_categorical_encoder(
            train_corpus=champ_m.payload.get("training", {}).get(
                "train_corpus", "last_12m"
            ),
            feature_variant=champion_payload.feature_variant,
        )
        rank_encoder = _fit_categorical_encoder(
            train_corpus=rank_m.payload.get("training", {}).get(
                "train_corpus", "last_12m"
            ),
            feature_variant=ranker_payload.feature_variant,
        )

        return cls(
            champion_manifest=champ_m,
            ranker_manifest=rank_m,
            direction_manifest=dir_m,
            champion_segments=champ_segments,
            ranker_segments=rank_segments,
            champion=champion_payload,
            ranker=ranker_payload,
            direction=direction_payload,
            champion_encoder=champ_encoder,
            ranker_encoder=rank_encoder,
        )

    # ------------------------------------------------------------------ predict_race

    def predict_race(
        self,
        race_card: Any,
        *,
        high_confidence_threshold: float = 0.30,
    ) -> RaceLevelOutputs:
        """Per-race per-runner outputs.

        ``race_card`` is a ``pandas.DataFrame`` with one row per runner.
        Required columns: the union of the champion's F2 + ranker's F5
        column sets. The Session-02 env data layer will assemble this
        DataFrame; for Session 01 the test fixture pulls it from the
        predictor repo's val split.

        Required identifier columns: ``selection_id`` (per-runner key,
        the dict key in the returned outputs) and ``market_id`` (used
        for caching).

        Cached by the unique ``market_id`` value in the DataFrame so the
        env can call once per race-card-known event and re-read for every
        tick of the market without re-running inference.

        ``high_confidence_threshold`` is the ranker softmax-share gate
        for ``ranker_top1_high_confidence_flag`` (default ``0.30`` per
        the ranker's manifest's ``high_confidence_threshold_for_consumer``
        recommendation).
        """
        import numpy as np
        import pandas as pd

        if not isinstance(race_card, pd.DataFrame):
            raise TypeError(
                f"predict_race expects a pandas.DataFrame; got {type(race_card).__name__}"
            )

        if "selection_id" not in race_card.columns:
            raise ValueError(
                "predict_race race_card DataFrame missing 'selection_id' column"
            )

        market_ids = race_card["market_id"].unique()
        if len(market_ids) != 1:
            raise ValueError(
                f"predict_race expects one market per call; got {len(market_ids)}"
            )
        market_id = str(market_ids[0])

        cached = self._race_cache.get(market_id)
        if cached is not None:
            return cached

        selection_ids = [int(s) for s in race_card["selection_id"].tolist()]

        # Lazy import — avoid touching the predictor repo's heavy modules at
        # import time of this file.
        _ensure_betfair_predictors_on_path()
        from scripts.outcome_predictor.datasets import (  # type: ignore[import-not-found]
            apply_encoders,
            numeric_feature_matrix,
        )

        # ── Champion: P(win), P(placed) per runner ───────────────────────
        champ_df = apply_encoders(race_card, self.champion_encoder)
        X_champ, used_champ = numeric_feature_matrix(
            champ_df, variant=self.champion.feature_variant
        )
        if list(used_champ) != list(self.champion.feature_names):
            raise PredictorLoaderError(
                f"champion feature column mismatch: numeric_feature_matrix "
                f"returned {used_champ}, model expects {self.champion.feature_names}"
            )
        win_proba = self.champion.win_model.predict_proba(X_champ)[:, 1]
        placed_proba = self.champion.placed_model.predict_proba(X_champ)[:, 1]

        # ── Ranker: lambdarank score → softmax → rank → top1 flags ──────
        rank_df = apply_encoders(race_card, self.ranker_encoder)
        X_rank, used_rank = numeric_feature_matrix(
            rank_df, variant=self.ranker.feature_variant
        )
        if list(used_rank) != list(self.ranker.feature_names):
            raise PredictorLoaderError(
                f"ranker feature column mismatch: numeric_feature_matrix "
                f"returned {used_rank}, model expects {self.ranker.feature_names}"
            )
        raw_scores = self.ranker.win_ranker.predict(X_rank)
        # softmax within market
        shifted = raw_scores - float(np.max(raw_scores))
        exp_scores = np.exp(shifted)
        softmax = exp_scores / float(np.sum(exp_scores))
        # rank: 1 = top pick. argsort descending then rank.
        order = np.argsort(-raw_scores, kind="stable")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        top1_idx = int(np.argmax(raw_scores))

        # ── Champion segment_strong_flag per market ─────────────────────
        market_features = _market_features_for_segment_lookup(race_card)
        segment_hint = self.champion_segments.lookup(market_features)
        segment_strong = segment_hint == ConsumerHint.STRONG

        # Assemble per-selection_id dicts
        p_win = {sid: float(win_proba[i]) for i, sid in enumerate(selection_ids)}
        p_placed = {sid: float(placed_proba[i]) for i, sid in enumerate(selection_ids)}
        ranker_score = {sid: float(raw_scores[i]) for i, sid in enumerate(selection_ids)}
        ranker_rank = {sid: int(ranks[i]) for i, sid in enumerate(selection_ids)}
        ranker_softmax_share = {
            sid: float(softmax[i]) for i, sid in enumerate(selection_ids)
        }
        ranker_top1_flag = {
            sid: bool(i == top1_idx) for i, sid in enumerate(selection_ids)
        }
        ranker_top1_high_confidence_flag = {
            sid: bool(
                (i == top1_idx) and (softmax[i] >= high_confidence_threshold)
            )
            for i, sid in enumerate(selection_ids)
        }
        # segment_strong_flag is per-market (single value); broadcast per-runner.
        segment_strong_flag = {sid: bool(segment_strong) for sid in selection_ids}

        outputs = RaceLevelOutputs(
            p_win=p_win,
            p_placed=p_placed,
            ranker_score=ranker_score,
            ranker_rank=ranker_rank,
            ranker_softmax_share=ranker_softmax_share,
            ranker_top1_flag=ranker_top1_flag,
            ranker_top1_high_confidence_flag=ranker_top1_high_confidence_flag,
            segment_strong_flag=segment_strong_flag,
        )
        self._race_cache[market_id] = outputs
        return outputs

    # ------------------------------------------------------------------ predict_tick

    def predict_tick_batch(
        self,
        ladder_windows: Any,
        *,
        chunk_size: int = 4096,
    ) -> tuple[Any, Any]:
        """Batched per-tick direction prediction.

        ``ladder_windows`` is a ``np.ndarray`` of shape ``(N, 32, 26)``
        float32 — N (tick, runner) pairs. Returns
        ``(quantiles, fires)`` where:

        - ``quantiles`` has shape ``(N, 3, 3)`` — [horizon][quantile].
          Horizons are ``self.direction.horizons`` (1m, 3m, 7m);
          quantiles are ``self.direction.quantiles`` (q10, q50, q90).
        - ``fires`` has shape ``(N, 3)`` boolean —
          [drift, shorten, no_signal] per row, mutually exclusive +
          exhaustive (sum across the 3 = 1).

        Far cheaper than calling :meth:`predict_tick` per (tick, runner)
        — the env's ``_compute_tick_predictor_outputs`` builds all
        windows for a race upfront then forwards them through this
        batched path with chunking to bound peak memory.
        """
        import numpy as np
        import torch

        arr = np.asarray(ladder_windows, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(
                f"ladder_windows must be 3-D (N, 32, 26); got shape {arr.shape}"
            )
        n, t, f = arr.shape
        if t != self.direction.time_window or f != self.direction.n_features:
            raise ValueError(
                f"ladder_windows trailing dims {arr.shape[1:]} != expected "
                f"({self.direction.time_window}, {self.direction.n_features})"
            )
        if n == 0:
            return (
                np.zeros((0, self.direction.n_horizons, self.direction.n_quantiles), dtype=np.float32),
                np.zeros((0, 3), dtype=bool),
            )

        # Forward in chunks to bound peak GPU memory.
        out_q = np.empty(
            (n, self.direction.n_horizons, self.direction.n_quantiles),
            dtype=np.float32,
        )
        # The Conv1D model lives on CPU per the loader's eval mode; keep
        # it there so we don't fight per-call device shuttle.
        with torch.no_grad():
            for start in range(0, n, chunk_size):
                end = min(n, start + chunk_size)
                x = torch.from_numpy(arr[start:end])
                y = self.direction.model(x)  # (chunk, n_h, n_q)
                out_q[start:end] = y.numpy()

        # Derive fire flags from the 7m horizon's q10/q50/q90 — use the
        # manifest's `signal_description` thresholds.
        h_idx = {h: i for i, h in enumerate(self.direction.horizons)}
        q_idx = {q: i for i, q in enumerate(self.direction.quantiles)}
        i_7m = h_idx["7m"]
        q10_7m = out_q[:, i_7m, q_idx[0.1]]
        q50_7m = out_q[:, i_7m, q_idx[0.5]]
        q90_7m = out_q[:, i_7m, q_idx[0.9]]
        fire_drift = (q50_7m >= 5.0) & (q10_7m >= 0.0)
        fire_shorten = (q50_7m <= -5.0) & (q90_7m <= 0.0)
        fire_no_signal = ~(fire_drift | fire_shorten)
        fires = np.stack([fire_drift, fire_shorten, fire_no_signal], axis=1)
        return out_q, fires

    def predict_tick(self, ladder_window: Any) -> TickLevelOutputs:
        """Per-tick directional quantiles + fire flags.

        ``ladder_window`` is a ``(time_window, n_features)`` numpy array
        of the V2 feature variant (32 ticks × 26 features by default;
        the bundle's `direction.time_window` and `direction.n_features`
        carry the canonical sizes). Column ordering must match
        `betfair-predictors/scripts/predictor/datasets.feature_columns('V2')`.

        Output: `TickLevelOutputs` with the 9 `q*_*m` quantiles plus the
        three mutually-exclusive `fire_*` booleans derived from the 7m
        horizon per the manifest's `signal_description`:

            fire_drift     = (q50_7m >= +5) AND (q10_7m >= 0)
            fire_shorten   = (q50_7m <= -5) AND (q90_7m <= 0)
            fire_no_signal = NOT (fire_drift OR fire_shorten)

        No caching — the call is cheap (~ms per runner per tick) and
        each tick has a fresh window.

        NOTE: the q*_1m/3m/7m fields are POSITIONAL LABELS for the
        predictor's 1st/2nd/3rd declared horizons, which differ by manifest
        (V2 was 1m/3m/7m; the retrained V4 is 3m/7m/15m). This mirrors the
        env's obs mapping in ``env/betfair_env.py`` — consumers read by
        position, not by the literal name. The model forward + fire logic
        are delegated to :meth:`predict_tick_batch` so there is a single,
        horizon-agnostic source of truth.
        """
        import numpy as np

        # Validate shape against the manifest-declared sizes.
        arr = np.asarray(ladder_window, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"predict_tick ladder_window must be 2-D "
                f"(time_window, n_features); got shape {arr.shape}"
            )
        expected_t = self.direction.time_window
        expected_f = self.direction.n_features
        if arr.shape != (expected_t, expected_f):
            raise ValueError(
                f"predict_tick ladder_window shape {arr.shape} != expected "
                f"({expected_t}, {expected_f})"
            )

        # Delegate the model forward + fire derivation to predict_tick_batch
        # (it reads horizons/quantiles off the loaded payload, so it adapts
        # to any retrained bundle — no hardcoded horizon names). Add the
        # batch dim, call, drop it.
        quantiles, fires = self.predict_tick_batch(arr[np.newaxis, :, :])
        y_np = quantiles[0]          # (n_horizons, n_quantiles)
        fire_row = fires[0]          # (3,) [drift, shorten, no_signal]
        q_idx = {q: i for i, q in enumerate(self.direction.quantiles)}
        q10i = q_idx.get(0.1, 0)
        q50i = q_idx.get(0.5, 1)
        q90i = q_idx.get(0.9, 2)

        def _pos(h_pos: int, q_pos: int) -> float:
            # Positional read of the predictor's 1st/2nd/3rd horizon;
            # missing horizons zero-fill (matches the env's obs mapping).
            return float(y_np[h_pos, q_pos]) if h_pos < y_np.shape[0] else 0.0

        q10_1m, q50_1m, q90_1m = _pos(0, q10i), _pos(0, q50i), _pos(0, q90i)
        q10_3m, q50_3m, q90_3m = _pos(1, q10i), _pos(1, q50i), _pos(1, q90i)
        q10_7m, q50_7m, q90_7m = _pos(2, q10i), _pos(2, q50i), _pos(2, q90i)

        fire_drift = bool(fire_row[0])
        fire_shorten = bool(fire_row[1])
        fire_no_signal = bool(fire_row[2])

        return TickLevelOutputs(
            q10_1m=q10_1m,
            q50_1m=q50_1m,
            q90_1m=q90_1m,
            q10_3m=q10_3m,
            q50_3m=q50_3m,
            q90_3m=q90_3m,
            q10_7m=q10_7m,
            q50_7m=q50_7m,
            q90_7m=q90_7m,
            fire_drift=fire_drift,
            fire_shorten=fire_shorten,
            fire_no_signal=fire_no_signal,
        )


# ---------------------------------------------------------------------------
# Market-feature extraction for SegmentRouter lookups
# ---------------------------------------------------------------------------


def _market_features_for_segment_lookup(race_card: Any) -> dict[str, Any]:
    """Extract the market-level features the segment router indexes by.

    The champion's `segment_performance.json` carries axes
    `field_size`, `sp_band`, `distance`, `race_type`, `surface`,
    `agree_disagree_sp`, `confidence_threshold`. The first five are
    derivable from race-card columns; `agree_disagree_sp` and
    `confidence_threshold` need model output to compute and so are
    deliberately omitted (the SegmentRouter falls back to "no STRONG
    vote" on missing axes — conservative, correct).
    """
    import pandas as pd

    out: dict[str, Any] = {}
    if "field_size" in race_card.columns:
        # field_size is per-market constant; pick the first.
        out["field_size"] = int(race_card["field_size"].iloc[0])
    if "race_type" in race_card.columns:
        out["race_type"] = str(race_card["race_type"].iloc[0])
    if "surface" in race_card.columns:
        out["surface"] = str(race_card["surface"].iloc[0])
    # `distance` axis in the JSON is bucketed (e.g. "10f", "long(10-16f)") —
    # we don't have a canonical bucket mapping here. Leaving missing means
    # this axis abstains; safe per the SegmentRouter reduce rule.
    return out
