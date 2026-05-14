"""Produce per-agent profile cards across all behavioral dimensions.

Combines:
- phenotypes.csv from the bet-log sweep (faithful for actions, prices, pwins)
- scoreboard.jsonl from the cohort (faithful for stop-close + force-close
  which the sweep had disabled)

Output:
- registry/<TAG>/agent_profile_cards.csv — one row per agent with
  categorical labels for side phenotype, price region, predictor-edge
  band, closer activity, stop-close reliance, purity, floor tier.

Usage:

    python -m tools.build_agent_profile_cards \\
        --cohort-tag _predictor_SCALPING_layq_1778712871

Requires phenotypes.csv to exist (run tools/sweep_bet_capture.py first).
"""
import argparse
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path

_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--cohort-tag", required=True)
_args = _p.parse_args(sys.argv[1:])

cohort_dir = Path('registry') / _args.cohort_tag

# Pull n_sc + n_fc from scoreboard (sweep can't capture these accurately).
sb_metrics = {}
with open(cohort_dir / 'scoreboard.jsonl') as f:
    for line in f:
        r = json.loads(line)
        sb_metrics[r['agent_id']] = {
            'sb_pnl': r['eval_day_pnl'],
            'sb_locked': r['eval_locked_pnl'],
            'sb_naked': r['eval_naked_pnl'],
            'sb_n_sc': r['eval_arbs_stop_closed'],
            'sb_n_fc': r['eval_arbs_force_closed'],
            'sb_n_mat': r['eval_arbs_completed'],
            'sb_n_naked': r['eval_arbs_naked'],
            'sb_n_closed': r['eval_arbs_closed'],
            'sb_bets': r['eval_bet_count'],
        }

ph = pd.read_csv(cohort_dir / 'phenotypes.csv')
ph['sb_n_sc'] = ph['agent_id'].map(lambda a: sb_metrics.get(a, {}).get('sb_n_sc', 0))
ph['sb_n_fc'] = ph['agent_id'].map(lambda a: sb_metrics.get(a, {}).get('sb_n_fc', 0))
ph['sb_pnl'] = ph['agent_id'].map(lambda a: sb_metrics.get(a, {}).get('sb_pnl', 0) * 3)  # 3-day total
ph['sb_locked'] = ph['agent_id'].map(lambda a: sb_metrics.get(a, {}).get('sb_locked', 0) * 3)


# ── Category labeling ───────────────────────────────────────────────

def side_label(p):
    """Aggressive side phenotype."""
    if p >= 0.70:
        return 'BACK-FIRST'
    if p <= 0.30:
        return 'LAY-FIRST'
    return 'MIXED'


def price_label(price, side='back'):
    """Price region for the side we trade."""
    if pd.isna(price):
        return 'n/a'
    if side == 'back':
        if price < 4: return 'SHORT(<4)'
        if price < 6: return 'MID(4-6)'
        if price < 8: return 'LONG(6-8)'
        return 'V.LONG(>8)'
    else:  # lay
        if price < 5: return 'SHORT(<5)'
        if price < 10: return 'MID(5-10)'
        if price < 15: return 'LONG(10-15)'
        return 'V.LONG(>15)'


def pwin_label(p):
    """Predictor-edge band."""
    if pd.isna(p): return 'n/a'
    if p < 0.20: return 'LONGSHOT(<.20)'
    if p < 0.35: return 'MID-FAV(.20-.35)'
    if p < 0.50: return 'FAV(.35-.50)'
    return 'TOP-FAV(.50+)'


def closer_label(n_closed, n_sc):
    """Active closer or stop-close-only."""
    if n_closed >= 6:
        return 'ACTIVE-CLOSER'
    if n_closed >= 3:
        return 'LIGHT-CLOSER'
    if n_closed >= 1:
        return 'RARE-CLOSER'
    return 'NEVER-CLOSES'


def stop_close_label(n_sc):
    """Stop-loss reliance."""
    if n_sc >= 15:
        return 'HEAVY-SC'
    if n_sc >= 5:
        return 'MOD-SC'
    if n_sc >= 1:
        return 'LIGHT-SC'
    return 'NO-SC'


def purity_label(locked_share, total_pnl):
    """How much of pnl is structural vs naked-driven."""
    if total_pnl <= 0:
        return 'LOSING'
    if locked_share >= 0.75:
        return 'PURE-SCALPER'
    if locked_share >= 0.40:
        return 'BALANCED'
    return 'NAKED-DRIVEN'


def quality_label(sb_locked):
    """Locked-floor quality tier (3-day total)."""
    if sb_locked >= 350: return 'HIGH-FLOOR'    # ~£117/day+
    if sb_locked >= 250: return 'GOOD-FLOOR'    # ~£83/day+
    if sb_locked >= 150: return 'MOD-FLOOR'
    return 'LOW-FLOOR'


# ── Build cards ───────────────────────────────────────────────────

cards = []
for _, r in ph.iterrows():
    primary_price = r['avg_back_price'] if r['agg_back_pct'] >= 0.5 else r['avg_lay_price']
    primary_side = 'back' if r['agg_back_pct'] >= 0.5 else 'lay'
    primary_pwin = r['avg_back_pwin'] if r['agg_back_pct'] >= 0.5 else r['avg_lay_pwin']

    cards.append({
        'gen': int(r['gen']) if pd.notna(r['gen']) else -1,
        'agent': r['agent_short'],
        'side_pheno': side_label(r['agg_back_pct']),
        'agg_back_%': f"{r['agg_back_pct']:.0%}",
        'primary_price': price_label(primary_price, primary_side),
        'primary_pwin': pwin_label(primary_pwin),
        'closer': closer_label(r['n_closed'], r['sb_n_sc']),
        'stop_close': stop_close_label(r['sb_n_sc']),
        'purity': purity_label(r['locked_share'], r['sb_pnl']),
        'floor_tier': quality_label(r['sb_locked']),
        'sb_pnl_3d': int(r['sb_pnl']),
        'sb_locked_3d': int(r['sb_locked']),
        'n_mat': int(r['n_mat']),
        'n_naked': int(r['n_naked']),
        'n_closed': int(r['n_closed']),
        'sb_n_sc': int(r['sb_n_sc']),
        'avg_back_price': round(r['avg_back_price'], 2) if pd.notna(r['avg_back_price']) else None,
        'avg_lay_price': round(r['avg_lay_price'], 2) if pd.notna(r['avg_lay_price']) else None,
        'avg_back_pwin': round(r['avg_back_pwin'], 3) if pd.notna(r['avg_back_pwin']) else None,
    })

cards_df = pd.DataFrame(cards).sort_values(['gen', 'agent']).reset_index(drop=True)
out = cohort_dir / 'agent_profile_cards.csv'
cards_df.to_csv(out, index=False)
print(f"Wrote {out}")
print()

# ── Distribution summary ─────────────────────────────────────────

def print_dist(col, title):
    print(f"\n--- {title} ---")
    vc = cards_df[col].value_counts().sort_index() if col == 'primary_price' else cards_df[col].value_counts()
    for k, v in vc.items():
        print(f"  {k:<22} {v:>3} agents ({v / len(cards_df):.0%})")

print("=" * 80)
print("PHENOTYPE DISTRIBUTION ACROSS THE COHORT")
print("=" * 80)
print_dist('side_pheno', 'Aggressive side')
print_dist('primary_price', 'Primary price region (back or lay depending on side)')
print_dist('primary_pwin', 'Predictor-edge band (pwin of admitted runners)')
print_dist('closer', 'close_signal activity')
print_dist('stop_close', 'Stop-close reliance (per cohort scoreboard)')
print_dist('purity', 'Locked-vs-naked dominance')
print_dist('floor_tier', 'Locked-floor quality tier')

# ── Print top 5 in each phenotype cluster ──────────────────────

print()
print("=" * 80)
print("EXAMPLE AGENTS PER PHENOTYPE")
print("=" * 80)

for label, df in cards_df.groupby('side_pheno'):
    print(f"\n--- {label} (n={len(df)}) — top 5 by locked floor ---")
    top = df.sort_values('sb_locked_3d', ascending=False).head(5)
    cols = ['gen', 'agent', 'agg_back_%', 'primary_price', 'primary_pwin',
            'closer', 'purity', 'sb_pnl_3d', 'sb_locked_3d',
            'n_mat', 'n_closed', 'sb_n_sc']
    print(top[cols].to_string(index=False))
