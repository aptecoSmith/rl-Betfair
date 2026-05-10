"""SegmentRouter — loads `segment_performance.json` and exposes per-market lookup.

Per `plans/predictor-integration/predictor_contracts.md` §1, §2 + the
production manifests, each champion ships a `segment_performance.json`
sidecar with per-axis buckets carrying a `consumer_hint` of
"strong" | "weak" | "insufficient_data". The loader caches the JSON at
startup; the env consults `lookup(market_features)` once per race and
flattens the result into `segment_strong_flag` per runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ConsumerHint(str, Enum):
    STRONG = "strong"
    WEAK = "weak"
    NEUTRAL = "neutral"  # observed in real manifests; not in contracts.md as of 2026-05-10
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class _Bucket:
    axis: str
    bucket_label: str
    bucket_definition: dict[str, Any]
    consumer_hint: ConsumerHint
    n_markets_total: int


# Axis names in the segment_performance.json top-level keys (e.g. "by_field_size"
# strips to axis "field_size"). The router indexes per-axis buckets by their
# `bucket_definition.value` so a market's feature value can route in O(1).
_AXIS_KEY_PREFIX = "by_"


class SegmentRouter:
    """Indexed view of one model's `segment_performance.json`.

    `lookup(market_features)` reduces across the registered axes:
    if ANY axis hint is "weak", return WEAK (caller skips); if any
    axis is "insufficient_data", return INSUFFICIENT_DATA only when
    no axis is STRONG; otherwise return STRONG. Conservative —
    matches the predictor manifests' "skip or de-weight" guidance.
    """

    def __init__(self, axes: dict[str, dict[Any, _Bucket]]):
        self._axes = axes

    @classmethod
    def from_path(cls, segment_performance_path: Path) -> "SegmentRouter":
        if not segment_performance_path.exists():
            raise FileNotFoundError(
                f"segment_performance.json not found at {segment_performance_path}"
            )
        with segment_performance_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        axes: dict[str, dict[Any, _Bucket]] = {}
        for top_key, value in payload.items():
            if not top_key.startswith(_AXIS_KEY_PREFIX):
                continue
            if not isinstance(value, list):
                continue
            axis_name = top_key[len(_AXIS_KEY_PREFIX):]
            buckets: dict[Any, _Bucket] = {}
            for entry in value:
                bdef = entry.get("bucket_definition", {})
                hint_raw = entry.get("consumer_hint", "insufficient_data")
                try:
                    hint = ConsumerHint(hint_raw)
                except ValueError as exc:
                    raise ValueError(
                        f"unknown consumer_hint {hint_raw!r} in axis {axis_name!r} "
                        f"bucket {entry.get('bucket_label')!r}"
                    ) from exc
                key = bdef.get("value")
                buckets[key] = _Bucket(
                    axis=axis_name,
                    bucket_label=entry.get("bucket_label", ""),
                    bucket_definition=bdef,
                    consumer_hint=hint,
                    n_markets_total=int(entry.get("n_markets_total", 0)),
                )
            if buckets:
                axes[axis_name] = buckets
        return cls(axes=axes)

    def lookup(self, market_features: dict[str, Any]) -> ConsumerHint:
        """Reduce per-axis hints to a single market-level hint.

        Conservative: WEAK on any axis dominates. STRONG requires at
        least one axis to be STRONG and no WEAK; INSUFFICIENT_DATA is
        the fallback when the market's value isn't in any axis index.
        """
        any_strong = False
        any_weak = False
        any_insufficient_or_unknown = False

        for axis_name, buckets in self._axes.items():
            if axis_name not in market_features:
                any_insufficient_or_unknown = True
                continue
            value = market_features[axis_name]
            bucket = buckets.get(value)
            if bucket is None:
                any_insufficient_or_unknown = True
                continue
            if bucket.consumer_hint is ConsumerHint.WEAK:
                any_weak = True
            elif bucket.consumer_hint is ConsumerHint.STRONG:
                any_strong = True
            else:
                # NEUTRAL or INSUFFICIENT_DATA: no STRONG vote.
                any_insufficient_or_unknown = True

        if any_weak:
            return ConsumerHint.WEAK
        if any_strong:
            return ConsumerHint.STRONG
        return ConsumerHint.INSUFFICIENT_DATA

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(self._axes.keys())
