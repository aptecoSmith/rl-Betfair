"""Repeatable gene -> trading-behaviour phenotype analysis for a PBT cohort.

Answers the standing question: *which genes drive which trading
behaviours?* across every agent of a PBT cohort (all generations), so the
operator can engineer a recipe that combines strengths — e.g. one champion
matures 20 % of pairs, another actively closes 57 % with 0 % force-close,
and we want both phenotypes in one agent.

Unlike `tools/gene_scatter_analysis.py` (a quick stdout-only scoreboard
probe with hand-rolled stats), this tool:

* reads the richer flat `model_register.csv` (full per-agent gene settings
  AND per-agent outcome counts for every agent across every generation),
* derives per-agent BEHAVIOUR RATES (maturation / close / force_close /
  naked, each as a fraction of pairs_opened) plus locked_pnl and naked_sd,
* computes BOTH Pearson and Spearman correlation (with scipy p-values)
  between each VARYING gene and each behaviour, across all agents,
* ranks the top gene drivers per behaviour by |Spearman|,
* synthesises a **combined-recipe** suggestion — the gene directions /
  magnitudes that would jointly raise maturation_rate + close_rate while
  lowering force_close_rate — with explicit correlation!=causation,
  sample-size and confound (architecture) caveats,
* SAVES a timestamped Markdown report + the full gene x behaviour
  correlation CSV under the cohort dir (or --out).

Pure pandas / numpy / scipy. No torch, no env. Read-only on the cohort
(safe to run while the cohort is still TRAINING — it only READS the
metrics file and WRITES its report).

Usage
-----
    python -m tools.phenotype_analysis --cohort-dir registry/pbt_genes_v2
    python -m tools.phenotype_analysis --cohort-dir registry/pbt_genes_v2 \
        --out C:/tmp/pheno

`--out` may be a directory (report files are written inside it) or a
file-stem; default is the cohort dir. `--source` overrides the metrics
file (default: <cohort-dir>/model_register.csv, with scoreboard.jsonl as
an automatic fallback).

A gene is included only if it actually VARIED across the cohort (>=2
distinct numeric values). Pinned / constant genes are reported in a
skipped-list and excluded from correlations (a constant has no defined
correlation). The report flags low-n (n < 15) and low-variance behaviours
(e.g. force_close_rate is identically 0 when the cohort trains with
force_close_before_off_seconds pinned to 0) so the reader does not
over-interpret a degenerate column.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - scipy is a hard dep of the repo
    _scipy_stats = None

# The report files are written UTF-8; console output may include non-ASCII
# in future, so make stdout/stderr UTF-8 on consoles (e.g. Windows cp1252)
# that would otherwise raise UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - older/odd stream objects
        pass


# ── Schema ───────────────────────────────────────────────────────────

# Behaviour RATES are derived from these outcome-count columns over the
# pairs_opened denominator. (name -> numerator column in model_register).
RATE_NUMERATORS: dict[str, str] = {
    "maturation_rate": "arbs_completed",
    "close_rate": "arbs_closed",
    "force_close_rate": "arbs_force_closed",
    "naked_rate": "arbs_naked",
}
PAIRS_COL = "pairs_opened"

# Absolute (non-rate) behaviours correlated directly.
ABS_BEHAVIOURS: dict[str, str] = {
    "locked_pnl": "locked_pnl",
    "naked_sd": "naked_std",
}

# A 5th outcome bucket worth surfacing: stop-loss closes. In cohorts that
# pin force_close_before_off_seconds=0 (the standard "train with naked
# variance" setup), arbs_force_closed is identically 0 and stop_closed is
# the live bail-out channel. Reported as an extra rate when present.
EXTRA_RATE_NUMERATORS: dict[str, str] = {
    "stop_close_rate": "arbs_stop_closed",
}

# Genes whose values are categorical strings — encoded to integer codes so
# they can be correlated (Spearman on the codes is an ordinal proxy; flag
# in the report that the encoding is arbitrary for nominal categories).
KNOWN_STRING_GENES = ("gene_architecture", "gene_transformer_pos_encoding")

LOW_N_THRESHOLD = 15  # below this, correlations are caveat-flagged.


@dataclass
class CorrResult:
    gene: str
    behaviour: str
    n: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float


# ── Loading ──────────────────────────────────────────────────────────


def load_from_register(path: Path) -> pd.DataFrame:
    """Load the flat per-agent register (genes + outcome counts)."""
    df = pd.read_csv(path)
    return df


def load_from_scoreboard(path: Path) -> pd.DataFrame:
    """Fallback loader: flatten scoreboard.jsonl into the register schema.

    Each row carries a nested ``hyperparameters`` dict (genes) plus
    ``eval_*`` outcome fields. We flatten the hyperparameters to
    ``gene_<name>`` columns and rename the eval outcome fields to the
    register's column names so the rest of the pipeline is source-agnostic.
    """
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()

    eval_to_register = {
        "eval_pairs_opened": "pairs_opened",
        "eval_arbs_completed": "arbs_completed",
        "eval_arbs_closed": "arbs_closed",
        "eval_arbs_force_closed": "arbs_force_closed",
        "eval_arbs_naked": "arbs_naked",
        "eval_arbs_stop_closed": "arbs_stop_closed",
        "eval_locked_pnl": "locked_pnl",
        "eval_naked_pnl": "naked_pnl",
        "eval_day_pnl": "day_pnl",
        "eval_bet_count": "bet_count",
        "architecture_name": "arch_name",
    }
    flat: list[dict] = []
    for r in rows:
        out: dict = {}
        out["generation"] = r.get("generation")
        out["model_id"] = r.get("model_id")
        out["agent_id"] = r.get("agent_id")
        for src, dst in eval_to_register.items():
            if src in r:
                out[dst] = r[src]
        # naked_std is not always present on scoreboard rows; tolerate.
        if "eval_naked_std" in r:
            out["naked_std"] = r["eval_naked_std"]
        hp = r.get("hyperparameters", {}) or {}
        for k, v in hp.items():
            out[f"gene_{k}"] = v
        flat.append(out)
    return pd.DataFrame(flat)


def resolve_source(cohort_dir: Path, explicit: Path | None) -> tuple[pd.DataFrame, Path, str]:
    """Pick and load the metrics source. Returns (df, path, kind)."""
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"--source {explicit} not found")
        if explicit.suffix == ".jsonl":
            return load_from_scoreboard(explicit), explicit, "scoreboard.jsonl"
        return load_from_register(explicit), explicit, explicit.name

    register = cohort_dir / "model_register.csv"
    scoreboard = cohort_dir / "scoreboard.jsonl"
    if register.exists():
        df = load_from_register(register)
        if len(df) > 0:
            return df, register, "model_register.csv"
    if scoreboard.exists():
        return load_from_scoreboard(scoreboard), scoreboard, "scoreboard.jsonl"
    raise FileNotFoundError(
        f"No metrics source found in {cohort_dir} "
        f"(looked for model_register.csv and scoreboard.jsonl)."
    )


# ── Behaviour derivation ─────────────────────────────────────────────


def derive_behaviours(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Return a DataFrame of per-agent behaviour columns (rates + abs)."""
    out = pd.DataFrame(index=df.index)

    have_pairs = PAIRS_COL in df.columns
    if not have_pairs:
        warnings.append(
            f"missing '{PAIRS_COL}' column — cannot derive any *_rate "
            f"behaviour; rate analysis skipped."
        )
    denom = None
    if have_pairs:
        denom = df[PAIRS_COL].astype(float)
        # Guard against divide-by-zero: agents with 0 pairs opened get NaN
        # rates (excluded pairwise from each correlation, not zero-filled —
        # a 0-pair agent has *undefined* maturation rate, not 0).
        denom = denom.where(denom > 0, other=np.nan)

    all_rate_specs = {**RATE_NUMERATORS}
    # Append stop_close_rate only if the column exists.
    for name, col in EXTRA_RATE_NUMERATORS.items():
        if col in df.columns:
            all_rate_specs[name] = col

    for name, num_col in all_rate_specs.items():
        if not have_pairs:
            continue
        if num_col not in df.columns:
            warnings.append(
                f"missing '{num_col}' — behaviour '{name}' skipped."
            )
            continue
        out[name] = df[num_col].astype(float) / denom

    for name, col in ABS_BEHAVIOURS.items():
        if col not in df.columns:
            warnings.append(
                f"missing '{col}' — behaviour '{name}' skipped."
            )
            continue
        out[name] = df[col].astype(float)

    return out


# ── Gene selection ───────────────────────────────────────────────────


def select_varying_genes(
    df: pd.DataFrame, warnings: list[str]
) -> tuple[pd.DataFrame, list[str], list[tuple[str, object]]]:
    """Return (encoded gene frame, varying gene names, skipped [(name, val)]).

    * Numeric genes with >=2 distinct values are kept as-is.
    * String genes in KNOWN_STRING_GENES are integer-encoded (factorize)
      and kept iff they vary.
    * Constant genes (numeric or string) are collected into `skipped`.
    """
    gene_cols = [c for c in df.columns if c.startswith("gene_")]
    kept_frame = pd.DataFrame(index=df.index)
    varying: list[str] = []
    skipped: list[tuple[str, object]] = []

    for c in gene_cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            nun = s.nunique(dropna=True)
            if nun >= 2:
                kept_frame[c] = s.astype(float)
                varying.append(c)
            else:
                val = s.dropna().iloc[0] if s.notna().any() else np.nan
                skipped.append((c, val))
        else:
            # Categorical: encode to codes.
            nun = s.nunique(dropna=True)
            if nun >= 2:
                codes, uniques = pd.factorize(s)
                codes = pd.Series(codes, index=df.index).astype(float)
                codes = codes.where(codes >= 0, other=np.nan)  # -1 = NaN
                kept_frame[c] = codes
                varying.append(c)
                warnings.append(
                    f"gene '{c}' is categorical {list(uniques)} - encoded "
                    f"to integer codes; Spearman on codes is an ordinal "
                    f"proxy (arbitrary order for nominal categories)."
                )
            else:
                val = s.dropna().iloc[0] if s.notna().any() else np.nan
                skipped.append((c, val))

    return kept_frame, sorted(varying), skipped


# ── Correlation ──────────────────────────────────────────────────────


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if _scipy_stats is None:
        r = float(np.corrcoef(x, y)[0, 1])
        return r, float("nan")
    r, p = _scipy_stats.pearsonr(x, y)
    return float(r), float(p)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if _scipy_stats is None:
        # Rank then Pearson as a fallback.
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        r = float(np.corrcoef(rx, ry)[0, 1])
        return r, float("nan")
    r, p = _scipy_stats.spearmanr(x, y)
    return float(r), float(p)


def correlate(
    genes: pd.DataFrame,
    behaviours: pd.DataFrame,
    gene_names: list[str],
    behaviour_names: list[str],
) -> list[CorrResult]:
    """Compute Pearson + Spearman for each (gene, behaviour) pair.

    Each pair uses its own complete-case rows (drop NaN in either column)
    so a single 0-pair agent doesn't nuke a whole gene's row.
    """
    results: list[CorrResult] = []
    for b in behaviour_names:
        yb = behaviours[b]
        for g in gene_names:
            xg = genes[g]
            mask = xg.notna() & yb.notna()
            x = xg[mask].to_numpy(dtype=float)
            y = yb[mask].to_numpy(dtype=float)
            n = int(mask.sum())
            if n < 3 or np.std(x) == 0 or np.std(y) == 0:
                results.append(
                    CorrResult(g, b, n, float("nan"), float("nan"),
                               float("nan"), float("nan"))
                )
                continue
            pr, pp = _safe_pearson(x, y)
            sr, sp = _safe_spearman(x, y)
            results.append(CorrResult(g, b, n, pr, pp, sr, sp))
    return results


def results_to_frame(results: list[CorrResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "behaviour": r.behaviour,
                "gene": r.gene,
                "n": r.n,
                "pearson_r": r.pearson_r,
                "pearson_p": r.pearson_p,
                "spearman_r": r.spearman_r,
                "spearman_p": r.spearman_p,
                "abs_spearman": abs(r.spearman_r)
                if not np.isnan(r.spearman_r)
                else np.nan,
            }
            for r in results
        ]
    )


# ── Behaviour variance / caveat flags ────────────────────────────────


def behaviour_flags(behaviours: pd.DataFrame) -> dict[str, str]:
    """Return per-behaviour caveat strings ('' if none)."""
    flags: dict[str, str] = {}
    for b in behaviours.columns:
        col = behaviours[b].dropna()
        if len(col) == 0:
            flags[b] = "no data"
            continue
        if col.nunique() <= 1:
            flags[b] = (
                f"ZERO-VARIANCE (constant = {col.iloc[0]:.4g}) — no gene "
                f"can correlate; behaviour is pinned/inactive in this cohort"
            )
        elif float(col.std()) < 1e-9:
            flags[b] = "near-zero variance — correlations unreliable"
        else:
            flags[b] = ""
    return flags


# ── Combined-recipe synthesis ────────────────────────────────────────

# For the recipe we want genes that, by their Spearman sign, push the
# three target behaviours the right way:
#   raise maturation_rate, raise close_rate, lower force_close_rate
# (and, where present, lower stop_close_rate as a proxy bail channel).
RECIPE_TARGETS = {
    "maturation_rate": +1,   # want higher
    "close_rate": +1,        # want higher
    "force_close_rate": -1,  # want lower
    "stop_close_rate": -1,   # want lower (bail proxy)
}


def synthesise_recipe(
    corr: pd.DataFrame,
    genes: pd.DataFrame,
    behaviours: pd.DataFrame,
    *,
    sig_p: float = 0.10,
    min_abs_spear: float = 0.20,
) -> list[dict]:
    """Pick genes whose direction jointly helps the recipe targets.

    A gene scores a 'vote' per target behaviour when its Spearman sign
    (times the target's desired direction) is positive and the association
    clears |spearman|>=min_abs_spear with p<=sig_p. We aggregate the votes
    into a net recommendation per gene and a suggested direction
    (increase/decrease) + an approximate magnitude (the observed value at
    the high-behaviour end of the cohort).
    """
    present_targets = [
        b for b in RECIPE_TARGETS
        if b in behaviours.columns and behaviours[b].nunique(dropna=True) > 1
    ]
    if not present_targets:
        return []

    recs: dict[str, dict] = {}
    for gene in genes.columns:
        net = 0.0
        votes: list[str] = []
        # Per-vote signed weight of the gene-direction needed (+ = increase
        # gene to help that target, - = decrease). Conflict = both signs.
        gene_dir_weights: list[float] = []
        for b in present_targets:
            want = RECIPE_TARGETS[b]
            row = corr[(corr["gene"] == gene) & (corr["behaviour"] == b)]
            if row.empty:
                continue
            sr = float(row["spearman_r"].iloc[0])
            sp = float(row["spearman_p"].iloc[0])
            if np.isnan(sr) or abs(sr) < min_abs_spear or sp > sig_p:
                continue
            contribution = sr * want  # >0 means gene moves behaviour the wanted way
            net += contribution
            # Direction the GENE should move to HELP this target, weighted
            # by association strength: sign(want*sr) * |sr|.
            gene_dir_weights.append(want * sr)
            gene_dir = "increase" if (want * sr) > 0 else "decrease"
            votes.append(
                f"{b}{('+' if want > 0 else '-')}: {gene_dir} gene "
                f"(rho={sr:+.2f}, p={sp:.3f})"
            )
        if not votes:
            continue
        # Net required gene direction = sign of the strength-weighted sum.
        dir_sum = sum(gene_dir_weights)
        direction = "increase" if dir_sum > 0 else "decrease"
        # Conflict: the gene must move one way to help target A and the
        # opposite way to help target B (both votes non-trivial).
        n_inc = sum(1 for w in gene_dir_weights if w > 0)
        n_dec = sum(1 for w in gene_dir_weights if w < 0)
        conflicted = n_inc > 0 and n_dec > 0
        # Approx magnitude: median gene value among the top-quartile agents
        # on the single strongest target this gene helps.
        strongest_b = max(
            present_targets,
            key=lambda bb: _abs_spear(corr, gene, bb),
        )
        approx = _suggested_magnitude(
            genes[gene], behaviours[strongest_b],
            want=RECIPE_TARGETS[strongest_b],
        )
        recs[gene] = {
            "gene": gene,
            "net_vote": net,
            # Coherence = |strength-weighted net direction|. For a clean
            # single-direction gene this equals the sum of |rho|; for a
            # conflicted gene the opposing votes cancel, shrinking it. So
            # ranking by this naturally rewards strong, coherent genes and
            # demotes genes whose helping/opposing votes are balanced —
            # without hard-gating a strong gene that has a minor conflict.
            "dir_strength": abs(dir_sum),
            "direction": direction,
            "approx_value": approx,
            "n_targets_helped": len(votes),
            "conflicted": conflicted,
            "detail": "; ".join(votes),
        }

    # Rank by strength-weighted direction coherence (strong, coherent genes
    # first); a minor conflict only shrinks the score rather than dropping
    # the gene below weak single-target genes.
    ordered = sorted(
        recs.values(), key=lambda d: d["dir_strength"], reverse=True
    )
    return ordered


def _abs_spear(corr: pd.DataFrame, gene: str, behaviour: str) -> float:
    row = corr[(corr["gene"] == gene) & (corr["behaviour"] == behaviour)]
    if row.empty:
        return 0.0
    v = float(row["spearman_r"].iloc[0])
    return 0.0 if np.isnan(v) else abs(v)


def _suggested_magnitude(
    gene_vals: pd.Series, behaviour_vals: pd.Series, *, want: int
) -> float:
    """Median gene value among agents in the best quartile of behaviour.

    'Best' = highest behaviour if want>0, lowest if want<0. This is a
    descriptive 'what did the good agents actually use' figure, NOT a
    causal optimum.
    """
    mask = gene_vals.notna() & behaviour_vals.notna()
    g = gene_vals[mask]
    b = behaviour_vals[mask]
    if len(g) < 4:
        return float(g.median()) if len(g) else float("nan")
    if want > 0:
        thresh = b.quantile(0.75)
        sel = g[b >= thresh]
    else:
        thresh = b.quantile(0.25)
        sel = g[b <= thresh]
    if len(sel) == 0:
        return float(g.median())
    return float(sel.median())


# ── Report rendering ─────────────────────────────────────────────────


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "  n/a"
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"


def render_report(
    *,
    cohort_dir: Path,
    source_kind: str,
    source_path: Path,
    n_agents: int,
    gen_counts: pd.Series,
    arch_counts: pd.Series,
    behaviours: pd.DataFrame,
    bflags: dict[str, str],
    varying_genes: list[str],
    skipped_genes: list[tuple[str, object]],
    corr: pd.DataFrame,
    recipe: list[dict],
    top_k: int,
    timestamp: str,
) -> str:
    L: list[str] = []
    A = L.append
    A(f"# Phenotype analysis — {cohort_dir.name}")
    A("")
    A(f"_Generated {timestamp} by `tools/phenotype_analysis.py`._")
    A("")
    A(f"- **Source:** `{source_path}` ({source_kind})")
    A(f"- **Agents (n):** {n_agents}")
    gens = ", ".join(f"gen{int(k)}={int(v)}" for k, v in gen_counts.items())
    A(f"- **Generations:** {gens}")
    if arch_counts is not None and len(arch_counts):
        archs = ", ".join(f"{k}={int(v)}" for k, v in arch_counts.items())
        A(f"- **Architectures:** {archs}")
    A(f"- **Varying genes analysed:** {len(varying_genes)} "
      f"(of {len(varying_genes) + len(skipped_genes)} total gene columns)")
    if n_agents < LOW_N_THRESHOLD:
        A("")
        A(f"> ⚠ **LOW SAMPLE SIZE (n={n_agents} < {LOW_N_THRESHOLD}).** "
          f"All correlations below are exploratory only; expect wide "
          f"confidence intervals and unstable rankings across generations.")
    A("")
    A("> **Correlation is not causation.** PBT genes co-vary (elite agents "
      "carry whole gene vectors forward; offspring inherit blocks), and "
      "architecture is itself a gene — so a gene's apparent effect may be a "
      "proxy for the architecture or for a co-inherited gene. Treat every "
      "driver below as a hypothesis to A/B, not a proven lever.")
    A("")

    # Behaviour summary table.
    A("## Behaviour summary (per-agent rates / values)")
    A("")
    A("| behaviour | n | mean | std | min | max | caveat |")
    A("|---|---|---|---|---|---|---|")
    for b in behaviours.columns:
        col = behaviours[b].dropna()
        n = len(col)
        if n == 0:
            A(f"| {b} | 0 | — | — | — | — | no data |")
            continue
        A(f"| {b} | {n} | {col.mean():.4g} | {col.std():.4g} | "
          f"{col.min():.4g} | {col.max():.4g} | {bflags.get(b, '')} |")
    A("")

    # Per-behaviour driver tables.
    A("## Gene drivers per behaviour")
    A("")
    A(f"Top {top_k} genes by |Spearman| for each behaviour. ρ = Spearman "
      "(rank, robust to outliers/nonlinearity); r = Pearson (linear). "
      "Sign shows direction: + means the gene and the behaviour rise "
      "together.")
    A("")
    for b in behaviours.columns:
        flag = bflags.get(b, "")
        A(f"### {b}")
        if flag and "ZERO-VARIANCE" in flag:
            A("")
            A(f"_{flag}. No gene correlations are defined — skipping._")
            A("")
            continue
        sub = corr[corr["behaviour"] == b].copy()
        sub = sub.dropna(subset=["abs_spearman"])
        sub = sub.sort_values("abs_spearman", ascending=False).head(top_k)
        if sub.empty:
            A("")
            A("_No computable correlations (insufficient variance/n)._")
            A("")
            continue
        if flag:
            A("")
            A(f"_Caveat: {flag}._")
        A("")
        A("| gene | ρ (Spearman) | p(ρ) | r (Pearson) | p(r) | n | interpretation |")
        A("|---|---|---|---|---|---|---|")
        for _, row in sub.iterrows():
            g = row["gene"].replace("gene_", "")
            sr = row["spearman_r"]
            interp = _interp_line(g, b, sr, row["spearman_p"])
            A(f"| {g} | {sr:+.3f} | {fmt_p(row['spearman_p'])} | "
              f"{row['pearson_r']:+.3f} | {fmt_p(row['pearson_p'])} | "
              f"{int(row['n'])} | {interp} |")
        A("")

    # Combined recipe.
    A("## Combined-recipe suggestion")
    A("")
    A("Goal: jointly **raise maturation_rate + close_rate** while "
      "**lowering force_close_rate** (and stop_close_rate where it is the "
      "active bail channel). The genes below are those whose Spearman "
      "direction helps at least one of these targets at |ρ|≥0.20, p≤0.10. "
      "`direction` is which way to move the gene; `approx value` is the "
      "median value the *already-good* agents used (descriptive, not a "
      "causal optimum).")
    A("")
    if not recipe:
        A("_No gene cleared the recipe thresholds (|ρ|≥0.20, p≤0.10) for "
          "any target behaviour. With this n the cohort has not yet "
          "differentiated the maturation/close/force-close levers — re-run "
          "after more generations complete._")
    else:
        # Headline: the strongest recipe candidates by direction coherence.
        # Conflicted genes are kept (the operator explicitly wants the
        # close-heavy and maturation-heavy phenotypes even if a minor
        # secondary association opposes), but annotated so the trade-off is
        # visible.
        headline = recipe[:8]
        if headline:
            A("**Top picks** (strongest direction-coherence first — set "
              "these toward the stated value/direction):")
            A("")
            for rec in headline:
                g = rec["gene"].replace("gene_", "")
                av = rec["approx_value"]
                av_s = (f"~{av:.4g}" if not (isinstance(av, float)
                        and np.isnan(av)) else "n/a")
                tag = " ⚠*trade-off*" if rec["conflicted"] else ""
                A(f"- **{rec['direction']} `{g}`**{tag} (toward {av_s}) — "
                  f"helps {rec['n_targets_helped']} target(s): {rec['detail']}")
            A("")
        conflicted = [r for r in recipe if r["conflicted"]]
        if conflicted:
            A(f"_{len(conflicted)} gene(s) marked ⚠ show a **direction "
              "conflict** — they move one way to help one target and the "
              "opposite way to help another. Where the helping association "
              "is much stronger than the opposing one (e.g. "
              "`direction_gate_enabled` close_rate ρ=+0.63 vs stop_close "
              "ρ=+0.25), follow the dominant direction but expect the "
              "secondary trade-off._")
            A("")
        # Full table (capped to keep it readable).
        cap = 16
        A(f"Full candidate table (top {min(cap, len(recipe))} by direction "
          "coherence; `conflict?` = gene must move both ways across "
          "targets):")
        A("")
        A("| gene | suggested direction | approx value | targets helped | "
          "conflict? | evidence |")
        A("|---|---|---|---|---|---|")
        for rec in recipe[:cap]:
            g = rec["gene"].replace("gene_", "")
            av = rec["approx_value"]
            av_s = f"{av:.4g}" if not (isinstance(av, float) and np.isnan(av)) else "—"
            conf = "**yes**" if rec["conflicted"] else "no"
            A(f"| {g} | **{rec['direction']}** | {av_s} | "
              f"{rec['n_targets_helped']} | {conf} | "
              f"{rec['detail']} |")
    A("")
    A("### Recipe caveats")
    A("")
    A("1. **Correlation ≠ causation.** These are observational associations "
      "across a co-evolving population, not interventional effects. Confirm "
      "any gene by an A/B that pins it while holding the rest at cohort "
      "defaults.")
    A(f"2. **Sample size n={n_agents}.** "
      + ("Below the n≥15 comfort threshold — rankings may reshuffle as later "
         "generations land. " if n_agents < LOW_N_THRESHOLD else "")
      + "p-values are uncorrected for multiple comparisons (dozens of "
        "gene×behaviour tests); a p≈0.05 here is weak evidence.")
    A("3. **Architecture confound.** `architecture` (and the hidden_size / "
      "transformer_* structural genes) is itself an evolved gene. If it "
      "appears as a driver, the 'recipe' may really be 'use that "
      "architecture', and other genes may be proxies for the architecture "
      "mix. Stratify by architecture before trusting a non-structural gene.")
    A("4. **Behaviour coupling.** maturation/close/naked/force_close rates "
      "share the pairs_opened denominator and sum to ~1, so raising one "
      "rate mechanically tends to lower others. A gene that raises "
      "close_rate may lower maturation_rate as an accounting side-effect, "
      "not a real trade-off in the policy.")
    if "force_close_rate" in bflags and "ZERO-VARIANCE" in bflags.get("force_close_rate", ""):
        A("5. **force_close_rate is identically 0 in this cohort** "
          "(force_close_before_off_seconds pinned to 0 during training — the "
          "standard 'keep naked-variance signal' setup). The recipe's "
          "'lower force_close' target therefore contributed nothing here; "
          "`stop_close_rate` is the live bail channel and is used as the "
          "proxy. Re-evaluate force-close behaviour on a held-out run with "
          "force_close enabled.")
    A("")

    # Skipped genes appendix.
    A("## Appendix — pinned / constant genes (excluded)")
    A("")
    A(f"{len(skipped_genes)} gene columns showed no variation across the "
      "cohort and were excluded from correlation (a constant has no defined "
      "correlation):")
    A("")
    if skipped_genes:
        for name, val in sorted(skipped_genes):
            A(f"- `{name.replace('gene_', '')}` = {val}")
    else:
        A("_(none — every gene varied.)_")
    A("")
    return "\n".join(L)


def _interp_line(gene: str, behaviour: str, sr: float, sp: float) -> str:
    """One-line plain-English interpretation of a driver."""
    if np.isnan(sr):
        return "n/a"
    strength = (
        "strong" if abs(sr) >= 0.5 else
        "moderate" if abs(sr) >= 0.3 else
        "weak"
    )
    sig = "" if sp <= 0.05 else " (not significant)"
    direction = "higher" if sr > 0 else "lower"
    return f"{strength}: {direction} `{gene}` => {direction} {behaviour}{sig}"


# ── Main ─────────────────────────────────────────────────────────────


def run(cohort_dir: Path, out: Path | None, source: Path | None,
        top_k: int) -> int:
    warnings: list[str] = []
    try:
        df, source_path, source_kind = resolve_source(cohort_dir, source)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if df is None or len(df) == 0:
        print("ERROR: metrics source is empty (no completed agents yet).",
              file=sys.stderr)
        return 2

    n_agents = len(df)
    gen_counts = (
        df["generation"].value_counts().sort_index()
        if "generation" in df.columns else pd.Series(dtype=int)
    )
    arch_col = "arch_name" if "arch_name" in df.columns else None
    arch_counts = (
        df[arch_col].apply(_short_arch).value_counts()
        if arch_col else pd.Series(dtype=int)
    )

    behaviours = derive_behaviours(df, warnings)
    if behaviours.shape[1] == 0:
        print("ERROR: no behaviour columns could be derived "
              "(missing outcome columns). Warnings:\n  "
              + "\n  ".join(warnings), file=sys.stderr)
        return 2
    bflags = behaviour_flags(behaviours)

    genes, varying, skipped = select_varying_genes(df, warnings)
    if not varying:
        print("ERROR: no varying genes found — every gene is pinned.",
              file=sys.stderr)
        return 2

    corr_results = correlate(genes, behaviours, varying,
                             list(behaviours.columns))
    corr = results_to_frame(corr_results)

    recipe = synthesise_recipe(corr, genes, behaviours)

    timestamp_human = datetime.now().strftime("%Y-%m-%d %H:%M")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    report_md = render_report(
        cohort_dir=cohort_dir,
        source_kind=source_kind,
        source_path=source_path,
        n_agents=n_agents,
        gen_counts=gen_counts,
        arch_counts=arch_counts,
        behaviours=behaviours,
        bflags=bflags,
        varying_genes=varying,
        skipped_genes=skipped,
        corr=corr,
        recipe=recipe,
        top_k=top_k,
        timestamp=timestamp_human,
    )

    # Resolve output paths.
    md_path, csv_path = _resolve_out_paths(cohort_dir, out, stamp)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report_md, encoding="utf-8")

    # Full correlation matrix CSV (long form: one row per gene×behaviour).
    corr_out = corr.copy()
    corr_out = corr_out.sort_values(
        ["behaviour", "abs_spearman"], ascending=[True, False]
    )
    corr_out.to_csv(csv_path, index=False)

    # Console summary.
    print(f"Source: {source_path} ({source_kind})")
    print(f"Agents: {n_agents}  |  varying genes: {len(varying)}  |  "
          f"behaviours: {list(behaviours.columns)}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
    print()
    _print_console_top(corr, behaviours, top_k=min(top_k, 6))
    print()
    print(f"Report : {md_path}")
    print(f"Matrix : {csv_path}")
    return 0


def _short_arch(name: object) -> str:
    s = str(name)
    if "lstm" in s:
        # v2_discrete_ppo_lstm_h1024 -> lstm_h1024
        parts = s.split("lstm")
        return "lstm" + (parts[-1] if len(parts) > 1 else "")
    if "transformer" in s:
        return "transformer"
    return s


def _print_console_top(corr: pd.DataFrame, behaviours: pd.DataFrame,
                       top_k: int) -> None:
    for b in behaviours.columns:
        sub = corr[corr["behaviour"] == b].dropna(subset=["abs_spearman"])
        sub = sub.sort_values("abs_spearman", ascending=False).head(top_k)
        if sub.empty:
            continue
        print(f"[{b}] top drivers:")
        for _, row in sub.iterrows():
            print(f"    {row['gene'].replace('gene_',''):38s} "
                  f"spearman={row['spearman_r']:+.3f} "
                  f"(p={fmt_p(row['spearman_p'])})  "
                  f"pearson={row['pearson_r']:+.3f}")


def _resolve_out_paths(cohort_dir: Path, out: Path | None,
                       stamp: str) -> tuple[Path, Path]:
    md_name = f"phenotype_analysis_{stamp}.md"
    csv_name = f"phenotype_corr_{stamp}.csv"
    if out is None:
        base = cohort_dir
        return base / md_name, base / csv_name
    if out.suffix:  # treated as a file stem
        return (out.with_suffix(".md"),
                out.with_name(out.stem + "_corr").with_suffix(".csv"))
    # treated as a directory
    return out / md_name, out / csv_name


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Repeatable gene->behaviour phenotype analysis for a PBT "
            "cohort. Reads the per-agent metrics (model_register.csv, "
            "scoreboard.jsonl fallback), correlates every varying gene "
            "against per-agent trading-behaviour rates + locked_pnl + "
            "naked_sd (Pearson & Spearman), and saves a timestamped "
            "Markdown report + full correlation CSV."
        )
    )
    p.add_argument(
        "--cohort-dir", required=True, type=Path,
        help="Cohort output dir (e.g. registry/pbt_genes_v2).",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output dir or file-stem for the report + CSV "
             "(default: write into --cohort-dir).",
    )
    p.add_argument(
        "--source", type=Path, default=None,
        help="Override the metrics source file "
             "(default: <cohort-dir>/model_register.csv, with "
             "scoreboard.jsonl as fallback).",
    )
    p.add_argument(
        "--top-k", type=int, default=6,
        help="How many gene drivers to list per behaviour (default 6).",
    )
    args = p.parse_args()
    return run(args.cohort_dir, args.out, args.source, args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
