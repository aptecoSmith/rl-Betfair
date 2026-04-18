import { describe, it, expect } from 'vitest';
import {
  epochSecondsOf,
  formatTs,
  maxEpochSeconds,
  PANELS_PER_PAGE,
} from './learning-curves';
import { EpisodeRecord } from './agent-diagnostic';

// -- Helper: build a row with a timestamp ------------------------------

function row(
  model_id: string,
  episode: number,
  timestamp: number | string,
  overrides: Partial<EpisodeRecord> = {},
): EpisodeRecord {
  return {
    episode,
    model_id,
    architecture_name: 'ppo_lstm_v1',
    day_date: '2026-04-18',
    total_reward: 0,
    total_pnl: 0,
    bet_count: 0,
    policy_loss: 1.0,
    value_loss: 0.0,
    entropy: 50.0,
    timestamp: timestamp as unknown as string,
    ...overrides,
  };
}

describe('epochSecondsOf', () => {
  it('passes through numeric epoch seconds', () => {
    expect(epochSecondsOf(row('a', 1, 1776542333.5))).toBe(1776542333.5);
  });

  it('parses numeric-string epoch', () => {
    expect(epochSecondsOf(row('a', 1, '1776542333'))).toBe(1776542333);
  });

  it('parses ISO date strings to epoch seconds', () => {
    const iso = '2026-04-18T12:00:00Z';
    const expected = Date.parse(iso) / 1000;
    expect(epochSecondsOf(row('a', 1, iso))).toBe(expected);
  });

  it('returns 0 for unparseable / missing timestamps', () => {
    expect(epochSecondsOf(row('a', 1, ''))).toBe(0);
    expect(epochSecondsOf(row('a', 1, 'not-a-date'))).toBe(0);
  });
});

describe('maxEpochSeconds', () => {
  it('returns the max across a mixed set', () => {
    const eps = [
      row('a', 1, 100.0),
      row('a', 2, 500.0),
      row('a', 3, 300.0),
    ];
    expect(maxEpochSeconds(eps)).toBe(500.0);
  });

  it('ignores zero / unparseable rows', () => {
    const eps = [
      row('a', 1, 'junk'),
      row('a', 2, 200.0),
    ];
    expect(maxEpochSeconds(eps)).toBe(200.0);
  });

  it('returns 0 for an all-unparseable set', () => {
    const eps = [row('a', 1, ''), row('a', 2, 'junk')];
    expect(maxEpochSeconds(eps)).toBe(0);
  });
});

describe('formatTs', () => {
  it('formats local time as YYYY-MM-DD HH:MM:SS', () => {
    const d = new Date(2026, 3, 18, 14, 5, 9);  // local
    const out = formatTs(d.getTime() / 1000);
    expect(out).toBe('2026-04-18 14:05:09');
  });

  it('pads single-digit month/day/time components', () => {
    const d = new Date(2026, 0, 2, 3, 4, 5);
    expect(formatTs(d.getTime() / 1000)).toBe('2026-01-02 03:04:05');
  });
});

describe('PANELS_PER_PAGE', () => {
  it('is exactly 10 — changing this changes operator muscle memory', () => {
    // Locked at 10 per the user's "max 10 charts per page" requirement.
    // If a future plan bumps this, update the user-facing copy too.
    expect(PANELS_PER_PAGE).toBe(10);
  });
});

// -- Sort + pagination semantics at the "data shape" level -------------
//
// The component's ``allPanels`` computed sorts by lastTs desc + shortId
// tie-break, and ``pagedPanels`` slices PANELS_PER_PAGE per page. The
// integration layer (Angular signal plumbing) is thin enough that
// testing the algorithms here — on plain arrays — protects the contract
// without standing up TestBed. The algorithm: bucket by model_id, take
// max timestamp per bucket, sort desc, slice into pages of 10.

function buildPanelLite(
  eps: EpisodeRecord[],
): { modelId: string; shortId: string; lastTs: number } {
  return {
    modelId: eps[0].model_id ?? 'unknown',
    shortId: (eps[0].model_id ?? 'unknown').slice(0, 8),
    lastTs: maxEpochSeconds(eps),
  };
}

function sortPanelsNewestFirst<T extends { lastTs: number; shortId: string }>(
  panels: T[],
): T[] {
  return [...panels].sort((a, b) => {
    if (b.lastTs !== a.lastTs) return b.lastTs - a.lastTs;
    return a.shortId.localeCompare(b.shortId);
  });
}

describe('panel ordering (newest first)', () => {
  it('puts the most-recent-activity panel first', () => {
    const panels = [
      buildPanelLite([row('old-agent', 1, 100.0)]),
      buildPanelLite([row('new-agent', 1, 500.0)]),
      buildPanelLite([row('mid-agent', 1, 300.0)]),
    ];
    const sorted = sortPanelsNewestFirst(panels);
    expect(sorted.map(p => p.modelId)).toEqual([
      'new-agent', 'mid-agent', 'old-agent',
    ]);
  });

  it('ties break by shortId so order is stable across polls', () => {
    const panels = [
      buildPanelLite([row('zzzz1234', 1, 500.0)]),
      buildPanelLite([row('aaaa1234', 1, 500.0)]),
    ];
    const sorted = sortPanelsNewestFirst(panels);
    expect(sorted.map(p => p.modelId)).toEqual([
      'aaaa1234', 'zzzz1234',
    ]);
  });
});

describe('pagination slicing', () => {
  function page<T>(items: T[], pageIdx: number): T[] {
    const start = pageIdx * PANELS_PER_PAGE;
    return items.slice(start, start + PANELS_PER_PAGE);
  }

  function totalPages(count: number): number {
    return Math.max(1, Math.ceil(count / PANELS_PER_PAGE));
  }

  it('returns the first 10 on page 0', () => {
    const items = Array.from({ length: 25 }, (_, i) => i);
    expect(page(items, 0)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('returns items 10..19 on page 1', () => {
    const items = Array.from({ length: 25 }, (_, i) => i);
    expect(page(items, 1)).toEqual([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
  });

  it('returns the 5-item tail on the final partial page', () => {
    const items = Array.from({ length: 25 }, (_, i) => i);
    expect(page(items, 2)).toEqual([20, 21, 22, 23, 24]);
  });

  it('totalPages is 1 on empty (prevents "page 1 of 0" copy)', () => {
    expect(totalPages(0)).toBe(1);
  });

  it('totalPages rounds up on partial final page', () => {
    expect(totalPages(1)).toBe(1);
    expect(totalPages(10)).toBe(1);
    expect(totalPages(11)).toBe(2);
    expect(totalPages(111)).toBe(12);  // the user's actual 110+ chart case
  });
});
