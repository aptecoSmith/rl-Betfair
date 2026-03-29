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

    // Default sort is by seconds_to_off desc. Click P&L to sort by pnl desc.
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
