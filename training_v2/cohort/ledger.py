"""Gauntlet ledger + queues — the persistent state (Phase 3).

`plans/gauntlet-pipeline/`. The ledger is a durable record of every lineage
climbing the gauntlet: its recipe, origin, latest weights, how many tranches it
has cleared, and its fc=0 validation score at each depth. From it we DERIVE the
``needs-T(K)`` queues (a lineage that has completed K-1 tranches needs tranche
K) and the FRONTIER pool (lineages at the current deepest depth — the breeder's
input). The ledger IS the resume state: a restarted run (or a second machine)
reads it to know exactly what to run next.

Design choices:
- **One JSONL file, last-snapshot-per-lineage wins.** Each `record_tranche` /
  `add_recipe` appends a full entry snapshot; `load` replays and keeps the last
  per `lineage_id`. `compact()` rewrites the file atomically (tmp + os.replace)
  to bound growth. Append-then-compact is crash-safe: a half-written trailing
  line is skipped on load, losing at most the last update (re-runnable).
- **A lineage is the climber identity** (NOT one row per tranche). Survivors
  keep their lineage across tranches (genes fixed — recipe purity); each tranche
  completion updates the same entry. Fresh blood + mutants are NEW lineages.
- **No selection logic here.** The ledger reports queues + frontier; the breeder
  (Phase 4) decides who survives. Keeps execution/selection decoupled.

Leakage asserts (carried from `holdout-selection.md`): the day split is stored in
the ledger meta and `assert_day_split_disjoint` enforces
``validation ∩ train == ∅`` and ``final_test ∩ (train ∪ validation) == ∅``.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "LedgerEntry",
    "GauntletLedger",
    "DaySplit",
    "assert_day_split_disjoint",
]


# ── Day split + leakage guard ──────────────────────────────────────────────


@dataclass(frozen=True)
class DaySplit:
    """The gauntlet's day partition (carried in the ledger meta).

    ``tranche_days`` is the ordered list of tranches (each a list of training
    days); the gauntlet grows by APPENDING a tranche (never resizing one).
    ``validation_days`` is the FIXED fc=0 selection set; ``final_test_days`` is
    the sealed `--holdout-recent` set, never trained or selected on.
    """

    tranche_days: list[list[str]]
    validation_days: list[str]
    final_test_days: list[str] = field(default_factory=list)

    @property
    def n_tranches(self) -> int:
        return len(self.tranche_days)

    def train_days_for(self, tranche_K: int) -> list[str]:
        """Tranche K's training days (1-indexed — K=1 is the first tranche)."""
        if tranche_K < 1 or tranche_K > self.n_tranches:
            raise IndexError(
                f"tranche_K={tranche_K} out of range [1, {self.n_tranches}]")
        return list(self.tranche_days[tranche_K - 1])

    def all_train_days(self) -> list[str]:
        return [d for tr in self.tranche_days for d in tr]


def assert_day_split_disjoint(split: DaySplit) -> None:
    """Enforce the leakage invariants. Raises ``AssertionError`` on a breach.

    Same guard the lockstep held-out path applies (`holdout-selection.md`):
    validation must be disjoint from every tranche's train days, and the sealed
    final test must be disjoint from BOTH train and validation.
    """
    train = set(split.all_train_days())
    val = set(split.validation_days)
    test = set(split.final_test_days)
    leak_tv = train & val
    assert not leak_tv, f"validation ∩ train != empty (leakage): {sorted(leak_tv)}"
    leak_test = test & (train | val)
    assert not leak_test, (
        f"final_test ∩ (train ∪ validation) != empty (leakage): "
        f"{sorted(leak_test)}")


# ── Ledger entry ────────────────────────────────────────────────────────────


@dataclass
class LedgerEntry:
    """One lineage's full state in the gauntlet."""

    lineage_id: str
    genes: dict                      # CohortGenes.to_dict()
    config_hash: str
    origin: str                      # "fresh" | "mutant" | "climber"/"survivor"
    tranches_completed: int = 0
    weights_path: str = ""           # latest checkpoint (depth == completed)
    parent_model_id: str | None = None
    parent_lineage_id: str | None = None
    status: str = "active"           # "active" | "culled" | "survivor"
    # fc=0 validation outcome per depth K (string keys for JSON round-trip).
    validation_score: dict = field(default_factory=dict)   # K -> composite
    validation_locked: dict = field(default_factory=dict)  # K -> locked £
    validation_naked: dict = field(default_factory=dict)    # K -> naked £
    last_agent_id: str = ""          # the agent_id of the most recent run

    def needs_tranche(self) -> int:
        """The next tranche this lineage should run (== completed + 1)."""
        return int(self.tranches_completed) + 1


def _entry_from_dict(rec: dict) -> LedgerEntry:
    """Build a LedgerEntry from a persisted snapshot, tolerant of schema drift
    (unknown keys ignored, missing keys defaulted)."""
    known = {f for f in LedgerEntry.__dataclass_fields__}
    kw = {k: v for k, v in rec.items() if k in known}
    e = LedgerEntry(
        lineage_id=kw["lineage_id"], genes=kw.get("genes", {}),
        config_hash=kw.get("config_hash", ""), origin=kw.get("origin", "fresh"))
    for k, v in kw.items():
        setattr(e, k, v)
    # JSON object keys are strings; tranches_completed must be int.
    e.tranches_completed = int(e.tranches_completed)
    return e


# ── Ledger ───────────────────────────────────────────────────────────────────


class GauntletLedger:
    """Durable, resumable lineage ledger backed by one JSONL file."""

    META_KEY = "__meta__"
    BRED_KEY = "__bred__"

    def __init__(self, path: "str | Path", *, split: DaySplit | None = None):
        self.path = Path(path)
        self._entries: dict[str, LedgerEntry] = {}
        self.split: DaySplit | None = split
        # Depths whose per-tranche cull (breed) has already fired — the
        # cull-early ("the tick") resume marker, so a resumed run does NOT
        # re-cull a depth it already bred. Empty for the full-fair-shot path.
        self._bred_depths: set[int] = set()

    # ── persistence ──────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: "str | Path") -> "GauntletLedger":
        """Replay the JSONL; last snapshot per lineage wins. Missing file ⇒
        an empty ledger (a fresh run). A truncated trailing line is skipped."""
        led = cls(path)
        p = Path(path)
        if not p.exists():
            return led
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("ledger: skipping unparseable line in %s", p)
                    continue
                if rec.get("lineage_id") == cls.META_KEY:
                    sp = rec.get("split")
                    if sp:
                        led.split = DaySplit(
                            tranche_days=[list(t) for t in sp["tranche_days"]],
                            validation_days=list(sp["validation_days"]),
                            final_test_days=list(sp.get("final_test_days", [])))
                    continue
                if rec.get("lineage_id") == cls.BRED_KEY:
                    led._bred_depths.add(int(rec["depth"]))
                    continue
                led._entries[rec["lineage_id"]] = _entry_from_dict(rec)
        return led

    def _append(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def set_split(self, split: DaySplit, *, check: bool = True) -> None:
        if check:
            assert_day_split_disjoint(split)
        self.split = split
        self._append({"lineage_id": self.META_KEY, "split": {
            "tranche_days": split.tranche_days,
            "validation_days": split.validation_days,
            "final_test_days": split.final_test_days,
        }})

    def compact(self) -> None:
        """Rewrite the file from current in-memory state (atomic)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                if self.split is not None:
                    fh.write(json.dumps({"lineage_id": self.META_KEY, "split": {
                        "tranche_days": self.split.tranche_days,
                        "validation_days": self.split.validation_days,
                        "final_test_days": self.split.final_test_days,
                    }}) + "\n")
                for d in sorted(self._bred_depths):
                    fh.write(json.dumps(
                        {"lineage_id": self.BRED_KEY, "depth": int(d)}) + "\n")
                for e in self._entries.values():
                    fh.write(json.dumps(asdict(e), default=str) + "\n")
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ── mutation ─────────────────────────────────────────────────────────

    def add_recipe(self, genes, *, origin: str, config_hash: str,
                   lineage_id: str | None = None,
                   parent_model_id: str | None = None,
                   parent_lineage_id: str | None = None) -> LedgerEntry:
        """Register a NEW recipe at depth 0 (lands in needs-T1).

        ``genes`` may be a ``CohortGenes`` (``.to_dict()`` is called) or a plain
        dict. ``config_hash`` is the recipe identity (caller passes
        ``executor.config_hash(genes)`` to keep one definition).
        """
        gdict = genes.to_dict() if hasattr(genes, "to_dict") else dict(genes)
        lid = lineage_id or uuid.uuid4().hex
        if lid in self._entries:
            raise ValueError(f"lineage_id {lid} already in ledger")
        e = LedgerEntry(
            lineage_id=lid, genes=gdict, config_hash=config_hash,
            origin=str(origin), parent_model_id=parent_model_id,
            parent_lineage_id=parent_lineage_id)
        self._entries[lid] = e
        self._append(asdict(e))
        return e

    def record_tranche(self, lineage_id: str, tranche_K: int, *,
                       weights_path: str, composite: float, locked: float,
                       naked: float, agent_id: str = "") -> LedgerEntry:
        """Record a completed tranche for a lineage (sets completed = K)."""
        e = self._entries.get(lineage_id)
        if e is None:
            raise KeyError(f"unknown lineage_id {lineage_id}")
        if int(tranche_K) != e.tranches_completed + 1:
            raise ValueError(
                f"lineage {lineage_id} completed {e.tranches_completed}; cannot "
                f"record tranche {tranche_K} (must be {e.tranches_completed + 1})")
        e.tranches_completed = int(tranche_K)
        e.weights_path = str(weights_path)
        e.validation_score[str(tranche_K)] = float(composite)
        e.validation_locked[str(tranche_K)] = float(locked)
        e.validation_naked[str(tranche_K)] = float(naked)
        if agent_id:
            e.last_agent_id = str(agent_id)
        self._append(asdict(e))
        return e

    def set_status(self, lineage_id: str, status: str) -> LedgerEntry:
        e = self._entries[lineage_id]
        e.status = str(status)
        self._append(asdict(e))
        return e

    def mark_bred(self, depth: int) -> None:
        """Record that the per-tranche cull (breed) has fired at ``depth`` — the
        cull-early resume marker (idempotent: a resumed run skips re-culling)."""
        self._bred_depths.add(int(depth))
        self._append({"lineage_id": self.BRED_KEY, "depth": int(depth)})

    def bred_depths(self) -> set:
        """Depths whose cull has already fired (see :meth:`mark_bred`)."""
        return set(self._bred_depths)

    # ── queries (derive the queues) ──────────────────────────────────────

    def all_entries(self) -> list[LedgerEntry]:
        return list(self._entries.values())

    def get(self, lineage_id: str) -> LedgerEntry:
        return self._entries[lineage_id]

    def needs(self, tranche_K: int, *, active_only: bool = True) -> list[LedgerEntry]:
        """The needs-T(K) queue: lineages that have completed exactly K-1."""
        out = []
        for e in self._entries.values():
            if active_only and e.status != "active":
                continue
            if e.tranches_completed == int(tranche_K) - 1:
                out.append(e)
        return out

    def frontier_depth(self, *, active_only: bool = True) -> int:
        """The deepest tranches_completed across (active) lineages (0 if none)."""
        depths = [e.tranches_completed for e in self._entries.values()
                  if not active_only or e.status == "active"]
        return max(depths) if depths else 0

    def frontier(self, depth: int | None = None, *,
                 active_only: bool = True) -> list[LedgerEntry]:
        """Lineages at ``depth`` (default: the current frontier depth) — the
        same-depth pool the breeder ranks. Empty if depth==0."""
        d = self.frontier_depth(active_only=active_only) if depth is None else depth
        if d <= 0:
            return []
        return [e for e in self._entries.values()
                if e.tranches_completed == d
                and (not active_only or e.status == "active")]

    def counts_by_depth(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for e in self._entries.values():
            out[e.tranches_completed] = out.get(e.tranches_completed, 0) + 1
        return out
