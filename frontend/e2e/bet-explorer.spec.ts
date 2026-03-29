import { test, expect } from './fixtures';
import { mockScoreboard } from './mocks/scoreboard.mock';
import { mockBetExplorer } from './mocks/bet-explorer.mock';

const MODEL_ID = 'aaa11111-1111-1111-1111-111111111111';

test.describe('Bet Explorer', () => {
  test('empty state before model selection', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/models', body: mockScoreboard() },
    ]);
    await page.goto('/bets');
    await expect(page.locator('[data-testid="empty-state"]')).toBeVisible();
  });

  test('selecting model loads bets table and summary', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/models', body: mockScoreboard() },
      { url: '**/api/replay/**/bets', body: mockBetExplorer() },
    ]);
    await page.goto('/bets');

    await page.locator('[data-testid="model-select"]').selectOption({ index: 1 });

    await expect(page.locator('[data-testid="bets-table"]')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('[data-testid="summary-bar"]')).toBeVisible();

    const rows = page.locator('[data-testid="bets-table"] tbody tr[data-testid="bet-row"]');
    const count = await rows.count();
    expect(count).toBe(10);

    // Should have 11 columns (Date, Venue, Race, Runner, Action, Time to Off, Price, Stake, Matched, Outcome, P&L)
    const headers = page.locator('[data-testid="bets-table"] thead th');
    const headerCount = await headers.count();
    expect(headerCount).toBe(11);

    const headerTexts = await headers.allTextContents();
    expect(headerTexts.map(h => h.trim())).toContain('Venue');
    expect(headerTexts.map(h => h.trim())).toContain('Race');
  });

  test('race filter shows venue and time labels', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/models', body: mockScoreboard() },
      { url: '**/api/replay/**/bets', body: mockBetExplorer() },
    ]);
    await page.goto('/bets');

    await page.locator('[data-testid="model-select"]').selectOption({ index: 1 });
    await expect(page.locator('[data-testid="bets-table"]')).toBeVisible({ timeout: 5000 });

    const raceFilter = page.locator('[data-testid="filter-race"]');
    const options = raceFilter.locator('option');
    const optionTexts = await options.allTextContents();

    // Should contain venue+time labels, not raw IDs
    expect(optionTexts.some(t => t.includes('Newmarket'))).toBe(true);
    expect(optionTexts.some(t => t.includes('Ascot'))).toBe(true);
    // Should NOT contain raw race IDs
    expect(optionTexts.some(t => t.includes('race-001'))).toBe(false);
  });

  test('default sort is chronological ascending', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/models', body: mockScoreboard() },
      { url: '**/api/replay/**/bets', body: mockBetExplorer() },
    ]);
    await page.goto('/bets');

    await page.locator('[data-testid="model-select"]').selectOption({ index: 1 });
    await expect(page.locator('[data-testid="bets-table"]')).toBeVisible({ timeout: 5000 });

    // First row should be the earliest bet by tick_timestamp
    const firstRowDate = await page.locator('[data-testid="bet-row"]:first-child td:first-child').textContent();
    expect(firstRowDate?.trim()).toBe('2026-03-26');

    // Last row should be the latest bet
    const lastRowDate = await page.locator('[data-testid="bet-row"]:last-child td:first-child').textContent();
    expect(lastRowDate?.trim()).toBe('2026-03-28');
  });

  test('filters change visible rows', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/models', body: mockScoreboard() },
      { url: '**/api/replay/**/bets', body: mockBetExplorer() },
    ]);
    await page.goto('/bets');

    await page.locator('[data-testid="model-select"]').selectOption({ index: 1 });
    await expect(page.locator('[data-testid="bets-table"]')).toBeVisible({ timeout: 5000 });

    const actionFilter = page.locator('[data-testid="filter-action"]');
    await actionFilter.selectOption('back');

    const rows = page.locator('[data-testid="bets-table"] tbody tr[data-testid="bet-row"]');
    const count = await rows.count();
    expect(count).toBeLessThan(10);
    expect(count).toBeGreaterThan(0);
  });

  test('sort by P&L header toggles order', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/models', body: mockScoreboard() },
      { url: '**/api/replay/**/bets', body: mockBetExplorer() },
    ]);
    await page.goto('/bets');

    await page.locator('[data-testid="model-select"]').selectOption({ index: 1 });
    await expect(page.locator('[data-testid="bets-table"]')).toBeVisible({ timeout: 5000 });

    // Click P&L to sort by pnl desc.
    const pnlHeader = page.locator('[data-testid="sort-pnl"]');
    await pnlHeader.click();
    await page.waitForTimeout(200);

    const getCellValues = async () => {
      const cells = page.locator('[data-testid="bets-table"] tbody tr[data-testid="bet-row"] td:last-child');
      return (await cells.allTextContents()).map((s) => parseFloat(s.replace(/[£,]/g, '')));
    };

    const firstSort = await getCellValues();

    // Click again to toggle to asc
    await pnlHeader.click();
    await page.waitForTimeout(200);
    const secondSort = await getCellValues();

    // Check that the order is genuinely reversed
    expect(firstSort[0]).toBeGreaterThanOrEqual(firstSort[firstSort.length - 1]);
    expect(secondSort[0]).toBeLessThanOrEqual(secondSort[secondSort.length - 1]);
  });
});
