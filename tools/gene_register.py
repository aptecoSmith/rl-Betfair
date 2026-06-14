"""Gene register v1 — read-only coverage map of explored gene space.

Part of ``plans/gauntlet-pipeline/`` (Phase 1). Loads every persisted
agent gene config across ALL eras in ``registry/`` (each agent's full
``CohortGenes`` is already written to ``scoreboard.jsonl`` /
``model_register.csv`` / the re-eval boards) and prints a per-gene
**coverage map**: where the GA has looked, what the held-out outcome was
there, and where the gene space is BLANK or promising-but-thin.

This is read-only — it trains nothing and writes nothing except its own
report (``--output``). It is the foundation for Phase 7's gap-targeted
fresh-blood sampler (which replaces the uniform roll in
``sample_fresh_blood_genes`` with draws from under-explored cells), and it
is immediately useful for picking the next Tock's seed bands.

Data sources (all under ``registry/`` unless ``--registry`` overrides):
  * ``**/scoreboard.jsonl``      — schema ``v2_cohort_scoreboard``. Full
    gene config in ``hyperparameters`` + IN-SAMPLE eval outcome
    (``eval_locked_pnl`` / ``eval_naked_pnl`` / ``composite_score``).
  * ``**/model_register.csv``    — per-model register with ``gene_*``
    columns (reconstructs the config when a row lacks a scoreboard
    twin) + in-sample ``locked_pnl`` / ``naked_pnl`` / ``naked_std``.
  * ``**/*reeval*.jsonl``, ``**/tt_*_fc*.jsonl`` — schema
    ``v2_cohort_reeval``. Full gene config + HELD-OUT outcome
    (``reeval_locked_pnl`` / ``reeval_naked_pnl`` / maturation).
  * ``registry/cross_era_holdout_board.jsonl`` — per-leg σ
    (``ho_sigma_leg``) keyed by 8-char model prefix, joined on for the
    deployment-critical naked-variance metric.

The unit of "a visited cell" is a distinct **gene config** (de-duped by a
canonical hash of the full gene dict), NOT a model_id — survivors carry
the same genes across tranches, so counting model_ids would over-count.
Held-out outcomes are attached to the config they were produced under.

Usage:
    python -m tools.gene_register
    python -m tools.gene_register --bins 10 --output registry/gene_register
    python -m tools.gene_register --gene open_cost --gene mature_prob_open_threshold
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("gene_register")

# ── Gene specs (ranges / choices) ─────────────────────────────────────────
# Sourced from training_v2.cohort.genes so the register stays in sync with
# the sampler. Imported defensively: if the module can't load (e.g. run from
# a stripped checkout) we fall back to observed-range binning for every gene.

try:  # pragma: no cover - import shim
    from training_v2.cohort import genes as _g

    _GENES_OK = True
except Exception as _err:  # pragma: no cover
    logger.warning("Could not import training_v2.cohort.genes (%s); "
                   "falling back to observed-range binning for all genes.",
                   _err)
    _g = None
    _GENES_OK = False


# Gene kinds:
#   "logfloat" — numeric, log-spaced bins (LRs etc.)
#   "float"    — numeric, linear bins
#   "intrange" — integer gene with a continuous [lo, hi] range (linear bins)
#   "choice"   — small discrete set (int or str); one column per value
#   "bool"     — True/False
#   "observed" — no declared spec; bin by observed min/max (flagged in report)
@dataclass(frozen=True)
class GeneSpec:
    name: str
    kind: str
    lo: float | None = None
    hi: float | None = None
    choices: tuple = ()


def _build_gene_specs() -> dict[str, GeneSpec]:
    """Construct the gene→spec map from the genes module constants."""
    specs: dict[str, GeneSpec] = {}
    if not _GENES_OK:
        return specs

    # Legacy 7.
    specs["learning_rate"] = GeneSpec("learning_rate", "logfloat",
                                      *_g.LEARNING_RATE_RANGE)
    specs["entropy_coeff"] = GeneSpec("entropy_coeff", "logfloat",
                                      *_g.ENTROPY_COEFF_RANGE)
    specs["clip_range"] = GeneSpec("clip_range", "float", *_g.CLIP_RANGE_RANGE)
    specs["gae_lambda"] = GeneSpec("gae_lambda", "float", *_g.GAE_LAMBDA_RANGE)
    specs["value_coeff"] = GeneSpec("value_coeff", "float", *_g.VALUE_COEFF_RANGE)
    specs["mini_batch_size"] = GeneSpec("mini_batch_size", "choice",
                                        choices=tuple(_g.MINI_BATCH_SIZE_CHOICES))
    specs["hidden_size"] = GeneSpec("hidden_size", "choice",
                                    choices=tuple(_g.HIDDEN_SIZE_VALID))

    log_floats = set(getattr(_g, "_LOG_UNIFORM_FLOATS", frozenset()))
    int_genes = set(getattr(_g, "_PHASE5_INT_GENES", frozenset()))
    for name, (lo, hi) in getattr(_g, "_PHASE5_RANGES", {}).items():
        if name in log_floats:
            kind = "logfloat"
        elif name in int_genes:
            kind = "intrange"
        else:
            kind = "float"
        specs[name] = GeneSpec(name, kind, float(lo), float(hi))

    # Structural / categorical genes (choices, not ranges).
    specs["architecture"] = GeneSpec("architecture", "choice",
                                     choices=tuple(_g.ARCHITECTURE_CHOICES))
    specs["transformer_depth"] = GeneSpec(
        "transformer_depth", "choice",
        choices=tuple(_g.TRANSFORMER_DEPTH_CHOICES))
    specs["transformer_heads"] = GeneSpec(
        "transformer_heads", "choice",
        choices=tuple(_g.TRANSFORMER_HEADS_CHOICES))
    specs["transformer_ctx_ticks"] = GeneSpec(
        "transformer_ctx_ticks", "choice",
        choices=tuple(_g.TRANSFORMER_CTX_TICKS_CHOICES))
    specs["transformer_ffn_mult"] = GeneSpec(
        "transformer_ffn_mult", "choice",
        choices=tuple(_g.TRANSFORMER_FFN_MULT_CHOICES))
    specs["transformer_pos_encoding"] = GeneSpec(
        "transformer_pos_encoding", "choice",
        choices=tuple(_g.TRANSFORMER_POS_ENCODING_CHOICES))

    # Bools.
    for b in ("predictor_lean_obs", "use_direction_predictor",
              "direction_gate_enabled"):
        specs[b] = GeneSpec(b, "bool")

    # Env-behaviour knobs sampled from explicit small sets — treat as choices
    # over the sampled values (observed values may include reeval overrides
    # like force_close=120, which the "observed" fallback handles via extra
    # columns when a value falls outside the declared choices).
    specs["force_close_before_off_seconds"] = GeneSpec(
        "force_close_before_off_seconds", "choice",
        choices=tuple(getattr(_g, "FORCE_CLOSE_BEFORE_OFF_SAMPLE", (0.0,))))
    specs["close_walk_ticks"] = GeneSpec(
        "close_walk_ticks", "choice",
        choices=tuple(getattr(_g, "CLOSE_WALK_TICKS_SAMPLE", (0, 5, 10))))
    specs["bc_pretrain_steps"] = GeneSpec(
        "bc_pretrain_steps", "choice",
        choices=tuple(getattr(_g, "BC_PRETRAIN_STEPS_SAMPLE", (0, 500))))
    return specs


GENE_SPECS = _build_gene_specs()


# ── Agent record ──────────────────────────────────────────────────────────


@dataclass
class AgentRecord:
    """One model's gene config + the outcomes observed for it."""

    model_id: str
    era: str
    genes: dict
    insample_locked: float = float("nan")
    insample_naked: float = float("nan")
    composite: float = float("nan")
    holdout_locked: float = float("nan")
    holdout_naked: float = float("nan")
    holdout_sigma_leg: float = float("nan")
    holdout_mat_rate: float = float("nan")
    sources: set = field(default_factory=set)


def _config_hash(genes: dict) -> str:
    """Canonical hash of a full gene config (rounded to dampen float noise).

    Two agents with the same recipe (e.g. survivor + its clone, or a
    repeated fresh-blood draw) collapse to one visited cell. Floats are
    rounded to 9 s.f. so bit-noise from JSON/CSV round-trips doesn't split a
    config; ints/strs/bools pass through.
    """
    items = []
    for k in sorted(genes):
        v = genes[k]
        if isinstance(v, float):
            v = round(v, 9)
        items.append((k, v))
    return json.dumps(items, sort_keys=True, default=str)


# ── Loaders ───────────────────────────────────────────────────────────────


def _coerce_gene_value(name: str, raw):
    """Best-effort type coercion for a gene value read from CSV/JSON."""
    if raw is None or raw == "":
        return None
    spec = GENE_SPECS.get(name)
    if spec is not None and spec.kind == "bool":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes")
    # Numeric?
    if isinstance(raw, (int, float)):
        return raw
    s = str(raw).strip()
    try:
        f = float(s)
        return int(f) if f.is_integer() and (spec is None
                                             or spec.kind != "logfloat") else f
    except ValueError:
        return s  # categorical string (architecture, pos_encoding, …)


def _load_scoreboards(registry: Path, records: dict[str, AgentRecord]) -> int:
    n = 0
    for path in sorted(registry.rglob("scoreboard.jsonl")):
        era = path.parent.name
        for line in _iter_jsonl(path):
            hp = line.get("hyperparameters")
            mid = line.get("model_id") or line.get("agent_id")
            if not hp or not mid:
                continue
            rec = records.get(mid)
            if rec is None:
                rec = AgentRecord(model_id=mid, era=era, genes=dict(hp))
                records[mid] = rec
            elif not rec.genes:
                rec.genes = dict(hp)
            rec.sources.add("scoreboard")
            # Keep the best-locked in-sample row per model (mirrors the
            # champion de-dup used elsewhere).
            locked = _f(line.get("eval_locked_pnl"))
            if math.isnan(rec.insample_locked) or (
                    not math.isnan(locked) and locked > rec.insample_locked):
                rec.insample_locked = locked
                rec.insample_naked = _f(line.get("eval_naked_pnl"))
                rec.composite = _f(line.get("composite_score"))
            n += 1
    return n


def _load_reevals(registry: Path, records: dict[str, AgentRecord]) -> int:
    """Held-out re-eval boards (schema v2_cohort_reeval)."""
    n = 0
    seen_paths: set[Path] = set()
    patterns = ("**/*reeval*.jsonl", "**/tt_*_fc*.jsonl",
                "**/reeval_heldout*.jsonl", "**/holdout_board*.jsonl")
    for pat in patterns:
        for path in sorted(registry.glob(pat)):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            era = path.parent.name
            for line in _iter_jsonl(path):
                if line.get("schema") != "v2_cohort_reeval":
                    continue
                hp = line.get("hyperparameters")
                mid = line.get("model_id") or line.get("agent_id")
                if not mid:
                    continue
                rec = records.get(mid)
                if rec is None:
                    if not hp:
                        continue
                    rec = AgentRecord(model_id=mid, era=era, genes=dict(hp))
                    records[mid] = rec
                elif not rec.genes and hp:
                    rec.genes = dict(hp)
                rec.sources.add("reeval")
                locked = _f(line.get("reeval_locked_pnl"))
                # Prefer the best held-out locked seen for this model.
                if math.isnan(rec.holdout_locked) or (
                        not math.isnan(locked)
                        and locked > rec.holdout_locked):
                    rec.holdout_locked = locked
                    rec.holdout_naked = _f(line.get("reeval_naked_pnl"))
                    rec.holdout_mat_rate = _f(
                        line.get("reeval_maturation_rate"))
                n += 1
    return n


def _load_register_csvs(registry: Path,
                        records: dict[str, AgentRecord]) -> int:
    """model_register.csv — reconstruct a config from ``gene_*`` columns for
    any model not already seen in a scoreboard/reeval, and attach the
    in-sample ``naked_std`` (per-leg σ proxy) where present."""
    n = 0
    for path in sorted(registry.rglob("model_register.csv")):
        era = path.parent.name
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    mid = row.get("model_id") or row.get("agent_id")
                    if not mid:
                        continue
                    genes = {}
                    for col, val in row.items():
                        if col and col.startswith("gene_"):
                            gv = _coerce_gene_value(col[5:], val)
                            if gv is not None:
                                genes[col[5:]] = gv
                    rec = records.get(mid)
                    if rec is None:
                        if not genes:
                            continue
                        rec = AgentRecord(model_id=mid, era=era, genes=genes)
                        records[mid] = rec
                    elif not rec.genes and genes:
                        rec.genes = genes
                    rec.sources.add("register_csv")
                    # naked_std here is the in-sample per-leg σ; only fill the
                    # held-out σ if we don't have one from the cross-era board.
                    if math.isnan(rec.holdout_sigma_leg):
                        ns = _f(row.get("naked_std"))
                        if not math.isnan(ns):
                            rec.insample_naked_std = ns  # type: ignore[attr-defined]
                    n += 1
        except Exception:
            logger.exception("Failed to read %s — skipping", path)
    return n


def _load_cross_era_sigma(registry: Path,
                          records: dict[str, AgentRecord]) -> int:
    """Join per-leg σ from the cross-era holdout board (keyed by 8-char
    model prefix) onto the deployment-critical held-out naked-variance."""
    board = registry / "cross_era_holdout_board.jsonl"
    if not board.exists():
        return 0
    by_prefix: dict[str, AgentRecord] = {}
    for rec in records.values():
        by_prefix.setdefault(rec.model_id[:8], rec)
    n = 0
    for line in _iter_jsonl(board):
        pref = str(line.get("model", ""))[:8]
        rec = by_prefix.get(pref)
        if rec is None:
            continue
        rec.holdout_sigma_leg = _f(line.get("ho_sigma_leg"))
        if math.isnan(rec.holdout_locked):
            rec.holdout_locked = _f(line.get("ho_locked"))
            rec.holdout_naked = _f(line.get("ho_naked"))
            rec.holdout_mat_rate = _f(line.get("ho_mat_rate"))
        rec.sources.add("cross_era_board")
        n += 1
    return n


def _iter_jsonl(path: Path):
    try:
        with path.open(encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    yield json.loads(ln)
                except json.JSONDecodeError:
                    continue
    except Exception:
        logger.exception("Failed to read %s — skipping", path)


def _f(v) -> float:
    try:
        if v is None or v == "":
            return float("nan")
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ── Binning + coverage ─────────────────────────────────────────────────────


@dataclass
class Bin:
    label: str
    lo: float | None = None
    hi: float | None = None
    n_configs: int = 0
    n_holdout: int = 0
    holdout_locked: list[float] = field(default_factory=list)
    holdout_sigma: list[float] = field(default_factory=list)
    insample_locked: list[float] = field(default_factory=list)


def _bins_for_spec(spec: GeneSpec, observed: list, n_bins: int) -> list[Bin]:
    if spec.kind in ("choice", "bool"):
        choices = list(spec.choices) if spec.kind == "choice" else [False, True]
        # Append any observed values outside the declared choices (e.g. a
        # reeval override) so nothing is silently dropped.
        seen = set()
        for v in observed:
            key = v
            if key not in choices and key not in seen:
                choices.append(key)
                seen.add(key)
        return [Bin(label=str(c), lo=c if isinstance(c, (int, float)) else None,
                    hi=c if isinstance(c, (int, float)) else None)
                for c in choices]

    nums = [float(v) for v in observed
            if isinstance(v, (int, float)) and not _is_nan(v)]
    lo = spec.lo
    hi = spec.hi
    if lo is None or hi is None:
        if not nums:
            return [Bin(label="(no data)")]
        lo, hi = min(nums), max(nums)
    if hi <= lo:
        hi = lo + 1e-9
    edges = _bin_edges(lo, hi, n_bins, log=(spec.kind == "logfloat"))
    bins: list[Bin] = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        bins.append(Bin(label=_fmt_range(a, b, spec.kind), lo=a, hi=b))
    return bins


def _bin_edges(lo: float, hi: float, n: int, log: bool) -> list[float]:
    if log and lo > 0 and hi > 0:
        la, lb = math.log(lo), math.log(hi)
        return [math.exp(la + (lb - la) * i / n) for i in range(n + 1)]
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def _assign_bin(spec: GeneSpec, bins: list[Bin], value) -> int | None:
    if spec.kind in ("choice", "bool"):
        target = value
        for i, b in enumerate(bins):
            if str(b.label) == str(target):
                return i
        return None
    if not isinstance(value, (int, float)) or _is_nan(value):
        return None
    v = float(value)
    for i, b in enumerate(bins):
        if b.lo is None or b.hi is None:
            continue
        # Last bin is inclusive on the high edge.
        if v >= b.lo and (v < b.hi or i == len(bins) - 1):
            return i
    # Out of declared range — clamp to the nearest end bin.
    return 0 if v < bins[0].lo else len(bins) - 1


def _is_nan(v) -> bool:
    return isinstance(v, float) and v != v


def _fmt_range(a: float, b: float, kind: str) -> str:
    def f(x):
        if abs(x) >= 1000 or (x != 0 and abs(x) < 0.001):
            return f"{x:.2e}"
        return f"{x:.4g}"
    return f"[{f(a)}, {f(b)})"


# ── Report ──────────────────────────────────────────────────────────────────


def build_coverage(records: list[AgentRecord], gene_names: list[str],
                   n_bins: int) -> dict[str, list[Bin]]:
    """For each gene, bin the DISTINCT configs and aggregate outcomes."""
    # De-dup configs; carry the best outcomes observed for each config.
    by_hash: dict[str, AgentRecord] = {}
    for rec in records:
        if not rec.genes:
            continue
        h = _config_hash(rec.genes)
        prev = by_hash.get(h)
        if prev is None:
            by_hash[h] = rec
        else:
            # Merge outcomes: prefer present (non-nan) and best held-out locked.
            if (math.isnan(prev.holdout_locked)
                    or (not math.isnan(rec.holdout_locked)
                        and rec.holdout_locked > prev.holdout_locked)):
                prev.holdout_locked = rec.holdout_locked
                prev.holdout_naked = rec.holdout_naked
                prev.holdout_mat_rate = rec.holdout_mat_rate
            if math.isnan(prev.holdout_sigma_leg) and not math.isnan(
                    rec.holdout_sigma_leg):
                prev.holdout_sigma_leg = rec.holdout_sigma_leg
            if math.isnan(prev.insample_locked) and not math.isnan(
                    rec.insample_locked):
                prev.insample_locked = rec.insample_locked
    configs = list(by_hash.values())

    coverage: dict[str, list[Bin]] = {}
    for name in gene_names:
        spec = GENE_SPECS.get(name) or GeneSpec(name, "observed")
        observed = [c.genes[name] for c in configs if name in c.genes
                    and c.genes[name] is not None]
        if not observed:
            continue
        bins = _bins_for_spec(spec, observed, n_bins)
        for c in configs:
            if name not in c.genes or c.genes[name] is None:
                continue
            idx = _assign_bin(spec, bins, c.genes[name])
            if idx is None:
                continue
            b = bins[idx]
            b.n_configs += 1
            if not math.isnan(c.holdout_locked):
                b.n_holdout += 1
                b.holdout_locked.append(c.holdout_locked)
                if not math.isnan(c.holdout_sigma_leg):
                    b.holdout_sigma.append(c.holdout_sigma_leg)
            if not math.isnan(c.insample_locked):
                b.insample_locked.append(c.insample_locked)
        coverage[name] = bins
    return coverage, len(configs)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def render_report(coverage: dict[str, list[Bin]], n_configs: int,
                  n_records: int, thin_threshold: int) -> str:
    L: list[str] = []
    A = L.append
    A("=" * 78)
    A("GENE REGISTER v1 — explored gene-space coverage map")
    A("=" * 78)
    A(f"distinct gene configs: {n_configs}   (from {n_records} model records)")
    A(f"genes mapped: {len(coverage)}   thin-cell threshold: "
      f"<{thin_threshold} configs")
    A("")
    A("Per gene: each bin shows #configs that visited it, how many have a")
    A("held-out outcome (ho), mean held-out locked, mean held-out sigma_leg,")
    A("and mean in-sample locked. BLANK = 0 configs. THIN = under threshold.")
    A("")

    blanks: list[str] = []
    thin_promising: list[str] = []

    for name in sorted(coverage):
        bins = coverage[name]
        spec = GENE_SPECS.get(name)
        kind = spec.kind if spec else "observed"
        tag = "" if spec else "  [OBSERVED-RANGE — no declared spec]"
        A(f"── {name}  ({kind}){tag}")
        A(f"   {'bin':28} {'#cfg':>5} {'ho':>4} {'ho_lck':>8} "
          f"{'ho_sig':>7} {'in_lck':>8}")
        for b in bins:
            holk = _mean(b.holdout_locked)
            hsig = _mean(b.holdout_sigma)
            ins = _mean(b.insample_locked)
            flag = ""
            if b.n_configs == 0:
                flag = "  <-- BLANK"
                blanks.append(f"{name} {b.label}")
            elif b.n_configs < thin_threshold:
                flag = "  <- thin"
                # Promising-thin: thin but a positive held-out locked seen.
                if b.n_holdout > 0 and not math.isnan(holk) and holk > 0:
                    thin_promising.append(
                        f"{name} {b.label}  (n={b.n_configs}, "
                        f"ho_lck={holk:.1f})")
            A(f"   {b.label:28} {b.n_configs:>5} {b.n_holdout:>4} "
              f"{_fmtf(holk):>8} {_fmtf(hsig):>7} {_fmtf(ins):>8}{flag}")
        A("")

    A("=" * 78)
    A("BLANK CELLS (never visited) — first candidates for gap-targeted blood")
    A("=" * 78)
    if blanks:
        for s in blanks:
            A(f"  {s}")
    else:
        A("  (none — every declared bin has at least one config)")
    A("")
    A("=" * 78)
    A("PROMISING-BUT-THIN CELLS (few configs, positive held-out locked)")
    A("=" * 78)
    if thin_promising:
        for s in thin_promising:
            A(f"  {s}")
    else:
        A("  (none surfaced — held-out coverage may be sparse)")
    A("")
    return "\n".join(L) + "\n"


def _fmtf(x: float) -> str:
    if x != x:  # nan
        return "-"
    return f"{x:.2f}"


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import sys

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(message)s")
    # Windows consoles default to cp1252; the report uses a few box-drawing
    # glyphs. Reconfigure stdout to UTF-8 so printing never crashes (the
    # written .txt is already UTF-8).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", type=Path, default=Path("registry"),
                   help="Registry root to scan (default: registry).")
    p.add_argument("--bins", type=int, default=8,
                   help="Numeric bins per gene (default 8).")
    p.add_argument("--gene", action="append", dest="genes", default=[],
                   help="Restrict the map to these gene names (repeatable). "
                        "Default: every gene seen in the data.")
    p.add_argument("--thin-threshold", type=int, default=3,
                   help="Cells with fewer configs than this are flagged thin.")
    p.add_argument("--output", type=Path, default=None,
                   help="Write the report to <output>.txt and a machine-"
                        "readable <output>.json. Default: stdout only.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    registry = args.registry
    if not registry.exists():
        raise SystemExit(f"registry not found: {registry}")

    records: dict[str, AgentRecord] = {}
    n_sb = _load_scoreboards(registry, records)
    n_re = _load_reevals(registry, records)
    n_csv = _load_register_csvs(registry, records)
    n_sig = _load_cross_era_sigma(registry, records)
    logger.warning("loaded: %d scoreboard rows, %d reeval rows, %d csv rows, "
                   "%d cross-era sigma joins -> %d distinct models",
                   n_sb, n_re, n_csv, n_sig, len(records))

    rec_list = [r for r in records.values() if r.genes]
    # Which genes to map.
    if args.genes:
        gene_names = list(args.genes)
    else:
        seen: set[str] = set()
        for r in rec_list:
            seen.update(r.genes.keys())
        gene_names = sorted(seen)

    coverage, n_configs = build_coverage(rec_list, gene_names, int(args.bins))
    report = render_report(coverage, n_configs, len(rec_list),
                           int(args.thin_threshold))
    print(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".txt").write_text(report, encoding="utf-8")
        machine = {
            "n_distinct_configs": n_configs,
            "n_model_records": len(rec_list),
            "genes": {
                name: [
                    {
                        "label": b.label,
                        "lo": b.lo, "hi": b.hi,
                        "n_configs": b.n_configs,
                        "n_holdout": b.n_holdout,
                        "mean_holdout_locked": _mean(b.holdout_locked),
                        "mean_holdout_sigma_leg": _mean(b.holdout_sigma),
                        "mean_insample_locked": _mean(b.insample_locked),
                    }
                    for b in bins
                ]
                for name, bins in coverage.items()
            },
        }
        out.with_suffix(".json").write_text(
            json.dumps(machine, indent=2, default=str), encoding="utf-8")
        logger.warning("wrote %s.txt + %s.json", out, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
