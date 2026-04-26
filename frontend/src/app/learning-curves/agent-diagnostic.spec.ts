import { describe, it, expect } from 'vitest';
import {
  bucketIntoRuns,
  diagnoseAgent,
  EpisodeRecord,
  RUN_BOUNDARY_GAP_SECONDS,
  sliceToMostRecentRun,
  summarisePopulation,
} from './agent-diagnostic';

/** Build one episode with sensible defaults; callers override only what matters. */
function ep(
  episode: number,
  day_date: string,
  overrides: Partial<EpisodeRecord> = {},
): EpisodeRecord {
  return {
    episode,
    model_id: 'test-agent',
    architecture_name: 'ppo_lstm_v1',
    day_date,
    total_reward: -100,
    total_pnl: -50,
    bet_count: 10,
    policy_loss: 0.15,
    value_loss: 5.0,
    entropy: 100,
    arbs_completed: 2,
    arbs_naked: 8,
    timestamp: '2026-04-17T12:00:00Z',
    ...overrides,
  };
}

describe('diagnoseAgent', () => {
  it('returns WARMING UP when too few episodes', () => {
    const d = diagnoseAgent([ep(1, '2026-04-06'), ep(2, '2026-04-07')]);
    expect(d.verdict).toBe('warming_up');
    expect(d.verdictLabel).toContain('WARMING UP');
    expect(d.headline).toMatch(/too early/i);
    expect(d.nEpisodes).toBe(2);
  });

  it('flags COLLAPSED when policy_loss exploded AND multiple dates are locked in', () => {
    // 15 episodes over 5 dates, 3 passes each. Ep 1 explodes, then
    // behaviour locks to a fixed per-date reward.
    const lockedRewards: Record<string, number> = {
      '2026-04-06': -323,
      '2026-04-07': -168,
      '2026-04-08': -503,
      '2026-04-09': -405,
      '2026-04-10': -571,
    };
    const dates = Object.keys(lockedRewards);
    const episodes: EpisodeRecord[] = [];
    let epNum = 1;
    for (let pass = 0; pass < 3; pass++) {
      for (const date of dates) {
        const isFirst = epNum === 1;
        episodes.push(ep(epNum, date, {
          total_reward: lockedRewards[date],
          policy_loss: isFirst ? 2.4e12 : 0.15,
        }));
        epNum += 1;
      }
    }

    const d = diagnoseAgent(episodes);
    expect(d.verdict).toBe('collapsed');
    expect(d.verdictLabel).toContain('COLLAPSED');
    expect(d.headline).toMatch(/saturation/i);
    expect(d.headline).toMatch(/locked in|bit-identical/i);
    expect(d.flags).toContain('collapsed');
  });

  it('flags UNSTABLE when a recent policy_loss spike exists but no lock-in yet', () => {
    // 10 episodes, spike on ep 9, still varied rewards per date.
    const episodes: EpisodeRecord[] = [];
    for (let i = 1; i <= 10; i++) {
      episodes.push(ep(i, `2026-04-0${((i - 1) % 5) + 1}`, {
        total_reward: -100 - i * 10,
        policy_loss: i === 9 ? 5000 : 0.2,
      }));
    }
    const d = diagnoseAgent(episodes);
    expect(d.verdict).toBe('unstable');
    expect(d.headline).toMatch(/spike/i);
    expect(d.flags).toContain('unstable');
  });

  it('flags LEARNING when arb rate trends up with no recent spikes', () => {
    // 12 episodes, arb rate climbing from 10% to 40%.
    const episodes: EpisodeRecord[] = [];
    for (let i = 1; i <= 12; i++) {
      const frac = i / 12;
      const completed = Math.round(frac * 40);
      const naked = 40 - completed + 10;
      episodes.push(ep(i, `2026-04-0${((i - 1) % 5) + 1}`, {
        arbs_completed: completed,
        arbs_naked: naked,
        policy_loss: 0.18,
      }));
    }
    const d = diagnoseAgent(episodes);
    expect(d.verdict).toBe('learning');
    // mature-prob-head (2026-04-26): new headline format cites
    // moving signals + their percentage values rather than the
    // word "rising". The arb-completion fragment is the one that
    // triggers here.
    expect(d.headline).toMatch(/arb-completion .*→.*%/);
    expect(d.captions.arbRate).toMatch(/%/);
  });

  it('flags LEARNING when force-close % is dropping (mature-prob-head 2026-04-26)', () => {
    // 12 episodes, force-close % dropping from 60% to 30%.
    // Need ≥4 forceCloseRate values for the trend to compute.
    const episodes: EpisodeRecord[] = [];
    for (let i = 1; i <= 12; i++) {
      const frac = i / 12;
      // Linear drop 60% → 30%.
      const fcPct = 0.6 - frac * 0.3;
      const total = 100;
      const force = Math.round(fcPct * total);
      episodes.push(ep(i, `2026-04-0${((i - 1) % 5) + 1}`, {
        arbs_completed: 30,
        arbs_naked: 10,
        arbs_closed: 5,
        arbs_force_closed: force,
        policy_loss: 0.18,
        total_reward: -100,
      }));
    }
    const d = diagnoseAgent(episodes);
    expect(d.verdict).toBe('learning');
    expect(d.headline).toMatch(/force-close/);
    expect(d.captions.forceCloseRate).toMatch(/falling/);
  });

  it('flags LEARNING when mature-prob accuracy is climbing', () => {
    // 12 episodes, mature accuracy climbing 50% → 80%.
    const episodes: EpisodeRecord[] = [];
    for (let i = 1; i <= 12; i++) {
      const frac = i / 12;
      const acc = 0.5 + frac * 0.3;
      episodes.push(ep(i, `2026-04-0${((i - 1) % 5) + 1}`, {
        arbs_completed: 3,
        arbs_naked: 7,
        policy_loss: 0.18,
        mature_prob_accuracy: acc,
        mature_prob_n_resolved: 50,
      }));
    }
    const d = diagnoseAgent(episodes);
    expect(d.verdict).toBe('learning');
    expect(d.headline).toMatch(/mature-acc/);
  });

  it('flags LEARNING when reward is rising', () => {
    // 12 episodes, reward rising linearly. arbs and policy_loss
    // held flat so no other channel triggers.
    const episodes: EpisodeRecord[] = [];
    for (let i = 1; i <= 12; i++) {
      episodes.push(ep(i, `2026-04-0${((i - 1) % 5) + 1}`, {
        arbs_completed: 3,
        arbs_naked: 7,
        policy_loss: 0.18,
        total_reward: -200 + i * 20,
      }));
    }
    const d = diagnoseAgent(episodes);
    expect(d.verdict).toBe('learning');
    expect(d.headline).toMatch(/reward/);
  });

  it('flags STAGNANT when stable but not improving', () => {
    // 10 episodes, flat arb rate, no spikes.
    const episodes: EpisodeRecord[] = [];
    for (let i = 1; i <= 10; i++) {
      episodes.push(ep(i, `2026-04-0${((i - 1) % 5) + 1}`, {
        arbs_completed: 3,
        arbs_naked: 7,
        policy_loss: 0.15,
        total_reward: -100,
      }));
    }
    const d = diagnoseAgent(episodes);
    expect(d.verdict).toBe('stagnant');
    expect(d.headline).toMatch(/not visibly improving/i);
  });

  it('STAGNANT headline flags selectivity_stuck when force-close stays high and flat', () => {
    // 12 episodes, force-close ~76% throughout (stuck high), no
    // assistant data, reward flat. Cohort-O / cohort-O2 shape.
    // Episodes are spread across many dates so the locked-in-date
    // rule doesn't trigger COLLAPSED. Reward must vary slightly so
    // the saturation-collapse rule doesn't trigger COLLAPSED either
    // (variance < SATURATED_COLLAPSE_REWARD_FLAT = 5).
    const episodes: EpisodeRecord[] = [];
    for (let i = 1; i <= 12; i++) {
      // Force-close fraction: ~76% with ±0.5pp jitter so the trend
      // is flat (deltaPP within ±2pp).
      const force = 76 + ((i % 2) === 0 ? 0 : 1);
      const completed = 12;
      const naked = 8;
      const closed = 4;
      // total = c + cl + n + f = 12 + 4 + 8 + (76 or 77) = 100 or 101
      // fc_rate ≈ 76/100 to 77/101 ≈ 0.760 to 0.762 — flat.
      episodes.push(ep(i, `2026-04-${String(((i - 1) % 18) + 1).padStart(2, '0')}`, {
        arbs_completed: completed,
        arbs_naked: naked,
        arbs_closed: closed,
        arbs_force_closed: force,
        policy_loss: 0.18,
        // Vary reward slightly (variance > 5 so saturation-collapse
        // skips this case but trend is still flat at the LEARNING
        // threshold of ±5%).
        total_reward: -100 + ((i % 3) - 1) * 6,
      }));
    }
    const d = diagnoseAgent(episodes);
    expect(d.verdict).toBe('stagnant');
    expect(d.flags).toContain('selectivity_stuck');
    expect(d.headline).toMatch(/force-close stuck/i);
  });

  it('flags actor_ignoring_assistant when mature-acc climbs but force-close stays stuck (cohort-F shape)', () => {
    // 12 episodes: mature accuracy 50% → 75% (assistant learning),
    // but force-close stays at ~76% (actor not using it). The
    // cohort-F failure shape.
    //
    // Verdict stays LEARNING (the policy IS learning *something* —
    // the auxiliary head). But the headline carries a warning
    // fragment about the actor ignoring the assistant, and the
    // ``actor_ignoring_assistant`` flag is set so downstream UI
    // (chart styling, sort order) can highlight it.
    const episodes: EpisodeRecord[] = [];
    for (let i = 1; i <= 12; i++) {
      const frac = i / 12;
      const acc = 0.5 + frac * 0.25;
      const force = 76 + ((i % 2) === 0 ? 0 : 1);
      episodes.push(ep(i, `2026-04-${String(((i - 1) % 18) + 1).padStart(2, '0')}`, {
        arbs_completed: 12,
        arbs_naked: 8,
        arbs_closed: 4,
        arbs_force_closed: force,
        policy_loss: 0.18,
        // Reward varies ±6 around -100 so saturation-collapse
        // (variance ≤ 5) doesn't trigger; reward TREND stays
        // within ±5% so reward channel doesn't trigger LEARNING.
        total_reward: -100 + ((i % 3) - 1) * 6,
        mature_prob_accuracy: acc,
        mature_prob_n_resolved: 50,
      }));
    }
    const d = diagnoseAgent(episodes);
    expect(d.verdict).toBe('learning');
    expect(d.headline).toMatch(/mature-acc/);
    expect(d.headline).toMatch(/actor isn't using it|cohort-F/i);
    expect(d.flags).toContain('actor_ignoring_assistant');
  });

  it('flags COLLAPSED via saturation when force-close ≥75% and reward flat for 10+ episodes', () => {
    // 12 episodes, all force-closing 78%, reward bit-identical.
    // No policy spike — the new (mature-prob-head 2026-04-26)
    // collapse trigger.
    const episodes: EpisodeRecord[] = [];
    for (let i = 1; i <= 12; i++) {
      episodes.push(ep(i, `2026-04-${String(((i - 1) % 18) + 1).padStart(2, '0')}`, {
        arbs_completed: 10,
        arbs_naked: 7,
        arbs_closed: 5,
        arbs_force_closed: 78,
        // 78 / (10+7+5+78) = 78/100 = 78%
        policy_loss: 0.18,
        total_reward: -250,
      }));
    }
    const d = diagnoseAgent(episodes);
    expect(d.verdict).toBe('collapsed');
    expect(d.flags).toContain('saturated');
    expect(d.headline).toMatch(/saturation collapse/i);
  });

  it('sorts episodes by episode number before analysing', () => {
    const shuffled = [
      ep(5, '2026-04-05', { total_reward: -50 }),
      ep(1, '2026-04-01', { total_reward: -100 }),
      ep(3, '2026-04-03', { total_reward: -75 }),
    ];
    // Should not throw and should pick first episode correctly.
    const d = diagnoseAgent(shuffled);
    expect(d.verdict).toBe('warming_up');
  });

  it('caption for reward reports direction + percent change', () => {
    const episodes: EpisodeRecord[] = [];
    for (let i = 1; i <= 12; i++) {
      episodes.push(ep(i, '2026-04-01', { total_reward: -200 + i * 10 }));
    }
    const d = diagnoseAgent(episodes);
    expect(d.captions.reward).toMatch(/trending up/i);
  });

  it('arbRate caption flags zero-completion dates', () => {
    const episodes: EpisodeRecord[] = [];
    for (let pass = 0; pass < 4; pass++) {
      // date A always zero, date B steady completions
      episodes.push(ep(pass * 2 + 1, '2026-04-07', {
        arbs_completed: 0, arbs_naked: 20,
      }));
      episodes.push(ep(pass * 2 + 2, '2026-04-08', {
        arbs_completed: 5, arbs_naked: 15,
      }));
    }
    const d = diagnoseAgent(episodes);
    expect(d.captions.arbRate).toMatch(/zero on 1 date/);
  });

  it('policyLoss caption shows peak value when spikes exist', () => {
    const episodes: EpisodeRecord[] = [];
    for (let i = 1; i <= 10; i++) {
      episodes.push(ep(i, '2026-04-01', {
        policy_loss: i === 3 ? 1e10 : 0.15,
      }));
    }
    const d = diagnoseAgent(episodes);
    expect(d.captions.policyLoss).toMatch(/spike/);
    expect(d.captions.policyLoss).toMatch(/1\.0e\+10/);
  });

  it('entropy caption reports climbing / decaying / stable', () => {
    const climbing: EpisodeRecord[] = [];
    for (let i = 1; i <= 10; i++) {
      climbing.push(ep(i, '2026-04-01', { entropy: 50 + i * 20 }));
    }
    expect(diagnoseAgent(climbing).captions.entropy).toMatch(/climbing/);

    const decaying: EpisodeRecord[] = [];
    for (let i = 1; i <= 10; i++) {
      decaying.push(ep(i, '2026-04-01', { entropy: 250 - i * 20 }));
    }
    expect(diagnoseAgent(decaying).captions.entropy).toMatch(/decaying|tightening/);

    const stable: EpisodeRecord[] = [];
    for (let i = 1; i <= 10; i++) {
      stable.push(ep(i, '2026-04-01', { entropy: 100 }));
    }
    expect(diagnoseAgent(stable).captions.entropy).toMatch(/stable/);
  });
});

describe('sliceToMostRecentRun', () => {
  function mk(episode: number, ts: number): EpisodeRecord {
    return {
      episode, day_date: '2026-04-06', total_reward: 0, total_pnl: 0,
      bet_count: 0, policy_loss: 0.1, value_loss: 0, entropy: 0,
      timestamp: ts as any,
    };
  }

  it('returns all rows when there is no large gap', () => {
    const eps = [mk(1, 1000), mk(2, 1060), mk(3, 1120)];
    expect(sliceToMostRecentRun(eps)).toHaveLength(3);
  });

  it('cuts at the most recent gap > 30 min', () => {
    const eps = [
      mk(1, 1000), mk(2, 1060),                 // old run
      mk(3, 1000 + 3600 * 2), mk(4, 1000 + 3600 * 2 + 60),  // new run
    ];
    const sliced = sliceToMostRecentRun(eps);
    expect(sliced.map(e => e.episode)).toEqual([3, 4]);
  });

  it('handles unsorted input', () => {
    const eps = [
      mk(4, 1000 + 3600 * 2 + 60),
      mk(1, 1000),
      mk(3, 1000 + 3600 * 2),
      mk(2, 1060),
    ];
    expect(sliceToMostRecentRun(eps).map(e => e.episode)).toEqual([3, 4]);
  });

  it('returns singleton when only one episode exists', () => {
    expect(sliceToMostRecentRun([mk(1, 100)]).map(e => e.episode)).toEqual([1]);
  });

  it('accepts ISO-string timestamps as a fallback', () => {
    const eps: EpisodeRecord[] = [
      { ...mk(1, 0), timestamp: '2026-04-17T10:00:00Z' as any },
      { ...mk(2, 0), timestamp: '2026-04-17T10:01:00Z' as any },
      { ...mk(3, 0), timestamp: '2026-04-17T14:00:00Z' as any },  // 4h gap
    ];
    const sliced = sliceToMostRecentRun(eps);
    expect(sliced.map(e => e.episode)).toEqual([3]);
  });
});

describe('summarisePopulation', () => {
  it('counts by verdict', () => {
    const summary = summarisePopulation([
      { verdict: 'learning' } as any,
      { verdict: 'learning' } as any,
      { verdict: 'collapsed' } as any,
      { verdict: 'unstable' } as any,
      { verdict: 'stagnant' } as any,
      { verdict: 'stagnant' } as any,
      { verdict: 'warming_up' } as any,
    ]);
    expect(summary.total).toBe(7);
    expect(summary.learning).toBe(2);
    expect(summary.collapsed).toBe(1);
    expect(summary.unstable).toBe(1);
    expect(summary.stagnant).toBe(2);
    expect(summary.warmingUp).toBe(1);
  });

  it('handles empty input', () => {
    expect(summarisePopulation([])).toEqual({
      total: 0, learning: 0, collapsed: 0, unstable: 0, stagnant: 0, warmingUp: 0,
    });
  });
});

// ── bucketIntoRuns — per-run filter source of truth -------------------
//
// A "run" is a contiguous cluster of episodes whose timestamps are
// within RUN_BOUNDARY_GAP_SECONDS (30 min) of each other. The UI's
// run-filter dropdown lets the operator pick a specific run instead
// of always seeing the latest — fixtures here pin the boundary
// behaviour so a future change to RUN_BOUNDARY_GAP_SECONDS surfaces
// explicitly rather than quietly shifting which rows bucket together.

function rowAt(modelId: string, epoch: number): EpisodeRecord {
  return ep(1, '2026-04-18', { model_id: modelId, timestamp: epoch as unknown as string });
}

describe('bucketIntoRuns', () => {
  it('returns empty for no episodes', () => {
    expect(bucketIntoRuns([])).toEqual([]);
  });

  it('puts adjacent rows in one run', () => {
    const rows = [
      rowAt('a', 1000),
      rowAt('a', 1100),
      rowAt('a', 1200),
    ];
    const runs = bucketIntoRuns(rows);
    expect(runs).toHaveLength(1);
    expect(runs[0].startTs).toBe(1000);
    expect(runs[0].endTs).toBe(1200);
    expect(runs[0].episodes).toHaveLength(3);
  });

  it('splits at the > RUN_BOUNDARY_GAP_SECONDS boundary', () => {
    const rows = [
      rowAt('a', 1000),
      rowAt('a', 2000),
      rowAt('b', 2000 + RUN_BOUNDARY_GAP_SECONDS + 1),  // new run
      rowAt('b', 3500 + RUN_BOUNDARY_GAP_SECONDS),
    ];
    const runs = bucketIntoRuns(rows);
    expect(runs).toHaveLength(2);
    // Newest first — the second run starts later.
    expect(runs[0].startTs).toBe(2000 + RUN_BOUNDARY_GAP_SECONDS + 1);
    expect(runs[1].startTs).toBe(1000);
  });

  it('keeps rows together at exactly RUN_BOUNDARY_GAP_SECONDS (inclusive)', () => {
    const rows = [
      rowAt('a', 0),
      rowAt('a', RUN_BOUNDARY_GAP_SECONDS),  // exactly on boundary — NOT a split
      rowAt('a', RUN_BOUNDARY_GAP_SECONDS + 1), // 1 s beyond also keeps
      rowAt('a', 2 * RUN_BOUNDARY_GAP_SECONDS + 2),  // gap > 30 min → split
    ];
    const runs = bucketIntoRuns(rows);
    expect(runs).toHaveLength(2);
    expect(runs[1].episodes).toHaveLength(3);
    expect(runs[0].episodes).toHaveLength(1);
  });

  it('returns runs sorted newest-first', () => {
    const rows = [
      rowAt('a', 10000),
      rowAt('b', 1000),
      rowAt('c', 50000),
    ];
    const runs = bucketIntoRuns(rows);
    expect(runs.map(r => r.startTs)).toEqual([50000, 10000, 1000]);
  });

  it('id equals startTs — stable across polls', () => {
    const rows = [rowAt('a', 1234), rowAt('a', 1235)];
    const [run] = bucketIntoRuns(rows);
    expect(run.id).toBe(1234);
  });

  it('handles unparseable timestamps without throwing', () => {
    const rows = [
      { ...rowAt('a', 1000), timestamp: '' },
      { ...rowAt('b', 2000), timestamp: 'not-a-date' },
      rowAt('c', 3000),
    ];
    const runs = bucketIntoRuns(rows);
    // At minimum: no crash, and the valid row appears somewhere.
    const allEpisodes = runs.flatMap(r => r.episodes);
    expect(allEpisodes).toHaveLength(3);
    expect(allEpisodes.some(e => e.model_id === 'c')).toBe(true);
  });

  it('a lone episode produces a one-row run', () => {
    const runs = bucketIntoRuns([rowAt('a', 500)]);
    expect(runs).toHaveLength(1);
    expect(runs[0].startTs).toBe(500);
    expect(runs[0].endTs).toBe(500);
    expect(runs[0].episodes).toHaveLength(1);
  });
});
