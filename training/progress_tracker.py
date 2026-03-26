"""
ProgressTracker — two-level ETA tracking for long-running operations.

Usage:
    tracker = ProgressTracker(total=100, label="Extracting days")
    for item in items:
        process(item)
        tracker.tick()
        print(tracker.to_dict())   # or publish to WebSocket queue
"""

from __future__ import annotations

import time
from collections import deque
from statistics import mean


def _fmt(seconds: float | None) -> str:
    """Format a duration in seconds to a human-readable string.

    Examples:
        45      → "45s"
        125     → "2m 5s"
        3600    → "1h"
        3900    → "1h 5m"
        None    → "unknown"
    """
    if seconds is None:
        return "unknown"
    s = int(seconds)
    if s < 0:
        return "0s"
    if s < 60:
        return f"{s}s"
    if s < 3600:
        minutes, secs = divmod(s, 60)
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours, remainder = divmod(s, 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


class ProgressTracker:
    """Rolling-window ETA tracker for a fixed-size batch of work items.

    Two ETA properties are exposed:
    - ``item_eta_seconds``    — estimated time for the *next* item
    - ``process_eta_seconds`` — estimated time for *all remaining* items

    Both are derived from a rolling average of the last ``rolling_window``
    completed item durations, so early slow items do not distort later ETAs.

    Call ``tick()`` once each time an item completes.
    Call ``to_dict()`` to get a serialisable snapshot suitable for WebSocket
    broadcast or logging.
    """

    def __init__(self, total: int, label: str, rolling_window: int = 10) -> None:
        self.total = total
        self.label = label
        self.completed = 0
        self._times: deque[float] = deque(maxlen=rolling_window)
        self._last_tick: float = time.monotonic()

    def tick(self) -> None:
        """Record one completed item and update the rolling timing window."""
        now = time.monotonic()
        self._times.append(now - self._last_tick)
        self._last_tick = now
        self.completed += 1

    def reset_timer(self) -> None:
        """Reset the inter-tick timer without recording a completion.

        Call this just before starting the first real item if there was a
        significant setup delay after __init__, so the first tick duration
        reflects only the item's own processing time.
        """
        self._last_tick = time.monotonic()

    @property
    def item_eta_seconds(self) -> float | None:
        """ETA for the next single item, based on rolling average.

        Returns None if no items have completed yet.
        """
        return mean(self._times) if self._times else None

    @property
    def process_eta_seconds(self) -> float | None:
        """ETA for all remaining items.

        Returns None if no items have completed yet.
        """
        if not self._times:
            return None
        remaining = self.total - self.completed
        if remaining <= 0:
            return 0.0
        return mean(self._times) * remaining

    @property
    def pct(self) -> float:
        """Percentage complete (0.0–100.0)."""
        if self.total <= 0:
            return 0.0
        return round(self.completed / self.total * 100, 1)

    def to_dict(self) -> dict:
        """Serialisable snapshot for WebSocket broadcast or logging.

        Schema matches the WebSocket message format defined in PLAN.md::

            {
                "label":              str,
                "completed":          int,
                "total":              int,
                "pct":                float,   # 0.0–100.0
                "item_eta_s":         float | None,
                "process_eta_s":      float | None,
                "item_eta_human":     str,     # e.g. "4m 12s"
                "process_eta_human":  str,     # e.g. "1h 18m"
            }
        """
        item_eta = self.item_eta_seconds
        process_eta = self.process_eta_seconds
        return {
            "label": self.label,
            "completed": self.completed,
            "total": self.total,
            "pct": self.pct,
            "item_eta_s": item_eta,
            "process_eta_s": process_eta,
            "item_eta_human": _fmt(item_eta),
            "process_eta_human": _fmt(process_eta),
        }
