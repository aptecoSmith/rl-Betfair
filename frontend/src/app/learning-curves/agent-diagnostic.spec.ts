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
    expect(d.headline).toMatch(/rising/i);
    expect(d.captions.arbRate).toMatch(/%/);
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
