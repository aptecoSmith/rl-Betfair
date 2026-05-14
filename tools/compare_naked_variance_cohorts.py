"""Side-by-side per-agent naked-variance comparison: layq vs raceconf.

Both restricted to a single day of in-sample data so the sample sizes
are comparable (raceconf was only swept on day 1 of in-sample).
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ── Load raceconf (CSV from the naked-pnl-only sweep, day 1 only) ──
race_csv = Path('registry/_predictor_SCALPING_raceconf_1778661062/naked_pnl_per_leg.csv')
race_df = pd.read_csv(race_csv)
# day column already in there; restrict to day 1
race_df1 = race_df[race_df['day'] == '2026-05-04'].copy()
race_df1.columns  # ['agent_id', 'gen', 'day', 'pair_id', 'side', 'price', 'stake', 'selection_id', 'pnl', 'tick_index']

# ── Load layq day 1 (from bet log parquets) ──
layq_dir = Path('registry/_predictor_SCALPING_layq_1778712871/bet_logs')
layq_rows = []
for agent_dir in layq_dir.iterdir():
    if not agent_dir.is_dir() or not agent_dir.name.startswith('adhoc_'):
        continue
    aid = agent_dir.name[len('adhoc_'):]
    parquet = agent_dir / '2026-05-04.parquet'
    if not parquet.exists():
        continue
    df = pd.read_parquet(parquet)
    df = df[df['final_outcome'] == 'naked'].copy()
    df['agent_id'] = aid
    layq_rows.append(df[['agent_id', 'pair_id', 'action', 'price', 'matched_size', 'runner_id', 'pnl']])

layq_df1 = pd.concat(layq_rows, ignore_index=True) if layq_rows else pd.DataFrame()
print(f"raceconf day-1 naked legs: {len(race_df1)} across {race_df1['agent_id'].nunique()} agents")
print(f"layq    day-1 naked legs: {len(layq_df1)} across {layq_df1['agent_id'].nunique()} agents")


def per_agent_stats(df, pnl_col='pnl', agent_col='agent_id'):
    out = []
    for aid, grp in df.groupby(agent_col):
        pnls = grp[pnl_col].values
        out.append({
            'agent_id': aid,
            'n': len(pnls),
            'mean': pnls.mean(),
            'std': pnls.std(),
            'mad': np.abs(pnls - pnls.mean()).mean(),
            'iqr': np.percentile(pnls, 75) - np.percentile(pnls, 25),
            'max_loss': pnls.min(),
            'max_gain': pnls.max(),
            'sum': pnls.sum(),
        })
    return pd.DataFrame(out)


race_per_agent = per_agent_stats(race_df1)
layq_per_agent = per_agent_stats(layq_df1)
# only keep agents with at least 5 naked legs (sigma below this is noise)
race_per_agent = race_per_agent[race_per_agent['n'] >= 5]
layq_per_agent = layq_per_agent[layq_per_agent['n'] >= 5]
print(f"\nWith n_naked >= 5: race {len(race_per_agent)} agents | layq {len(layq_per_agent)} agents")

# Add gen, locked from scoreboards
for label, cohort_dir, df in [
    ('race', 'registry/_predictor_SCALPING_raceconf_1778661062', race_per_agent),
    ('layq', 'registry/_predictor_SCALPING_layq_1778712871', layq_per_agent),
]:
    sb = {}
    with open(Path(cohort_dir) / 'scoreboard.jsonl') as f:
        for line in f:
            r = json.loads(line)
            sb[r['agent_id']] = r
    df['gen'] = df['agent_id'].map(lambda a: sb.get(a, {}).get('generation', -1))
    df['locked_per_day'] = df['agent_id'].map(lambda a: sb.get(a, {}).get('eval_locked_pnl', 0))
    df['agent_short'] = df['agent_id'].str[:12]

# Save both
race_per_agent.to_csv(Path('registry/_predictor_SCALPING_raceconf_1778661062') / 'naked_variance_day1.csv', index=False)
layq_per_agent.to_csv(Path('registry/_predictor_SCALPING_layq_1778712871') / 'naked_variance_day1.csv', index=False)

# ── Distribution comparison ──
print()
print("=" * 110)
print("DISTRIBUTION OF σ_naked_leg (per-agent stddev of naked leg pnl) — day 1 only")
print("=" * 110)
hdr = f"{'percentile':<14} {'raceconf':>12} {'layq':>12} {'delta':>10}"
print(hdr)
for p, label in [(10, 'p10'), (25, 'p25'), (50, 'median'), (75, 'p75'), (90, 'p90'), (100, 'max')]:
    if p == 100:
        rv, lv = race_per_agent['std'].max(), layq_per_agent['std'].max()
    else:
        rv = np.percentile(race_per_agent['std'], p)
        lv = np.percentile(layq_per_agent['std'], p)
    delta = lv - rv
    print(f"{label:<14} {rv:>+12.2f} {lv:>+12.2f} {delta:>+10.2f}")

print()
print(f"mean σ_naked_leg: race £{race_per_agent['std'].mean():.2f} | layq £{layq_per_agent['std'].mean():.2f}")
print(f"mean max_loss:    race £{race_per_agent['max_loss'].mean():.2f} | layq £{layq_per_agent['max_loss'].mean():.2f}")
print(f"mean max_gain:    race £{race_per_agent['max_gain'].mean():.2f} | layq £{layq_per_agent['max_gain'].mean():.2f}")
print(f"mean n_naked/agent: race {race_per_agent['n'].mean():.1f} | layq {layq_per_agent['n'].mean():.1f}")

# Compute daily naked aggregate volatility:
# daily σ ≈ sqrt(N) × σ_leg if assumed independent
race_per_agent['daily_naked_vol'] = race_per_agent['std'] * np.sqrt(race_per_agent['n'])
layq_per_agent['daily_naked_vol'] = layq_per_agent['std'] * np.sqrt(layq_per_agent['n'])

print()
print("=" * 110)
print("DAILY NAKED VOLATILITY (sqrt(N)×σ_leg) — proxy for day-to-day naked pnl swing")
print("=" * 110)
print(hdr)
for p, label in [(25, 'p25'), (50, 'median'), (75, 'p75'), (90, 'p90')]:
    rv = np.percentile(race_per_agent['daily_naked_vol'], p)
    lv = np.percentile(layq_per_agent['daily_naked_vol'], p)
    print(f"{label:<14} {rv:>+12.2f} {lv:>+12.2f} {lv-rv:>+10.2f}")

# ── Tightest scalpers in each cohort ──
print()
print("=" * 110)
print("RACECONF — top-10 TIGHT SCALPERS (lowest σ_leg, n≥10, locked≥75)")
print("=" * 110)
hdr2 = f"{'agent':<14}{'gen':>4}{'n':>5}{'σ_leg':>8}{'max_loss':>10}{'max_gain':>9}{'mean':>8}{'lk/day':>8}{'lk/σ':>7}"
print(hdr2)
cands = race_per_agent[(race_per_agent['n'] >= 10) & (race_per_agent['locked_per_day'] >= 75)].copy()
cands['lk_per_sigma'] = cands['locked_per_day'] / cands['std'].clip(lower=0.1)
cands = cands.sort_values('std').head(10)
for _, r in cands.iterrows():
    print(f"{r['agent_short']:<14}{int(r['gen']):>4}{int(r['n']):>5}{r['std']:>8.2f}{r['max_loss']:>+10.1f}{r['max_gain']:>+9.1f}{r['mean']:>+8.2f}{r['locked_per_day']:>+8.0f}{r['lk_per_sigma']:>7.1f}")

print()
print("=" * 110)
print("LAYQ — top-10 TIGHT SCALPERS (lowest σ_leg, n≥10, locked≥75)")
print("=" * 110)
print(hdr2)
cands = layq_per_agent[(layq_per_agent['n'] >= 10) & (layq_per_agent['locked_per_day'] >= 75)].copy()
cands['lk_per_sigma'] = cands['locked_per_day'] / cands['std'].clip(lower=0.1)
cands = cands.sort_values('std').head(10)
for _, r in cands.iterrows():
    print(f"{r['agent_short']:<14}{int(r['gen']):>4}{int(r['n']):>5}{r['std']:>8.2f}{r['max_loss']:>+10.1f}{r['max_gain']:>+9.1f}{r['mean']:>+8.2f}{r['locked_per_day']:>+8.0f}{r['lk_per_sigma']:>7.1f}")
