import { test, expect } from './fixtures';
import { mockScoreboard, mockEmptyScoreboard } from './mocks/scoreboard.mock';

test.describe('Scoreboard', () => {
  test('table loads with 5 model rows', async ({ page, mockApi }) => {
    await mockApi([{ url: '**/api/models', body: mockScoreboard() }]);
    await page.goto('/scoreboard');
    const rows = page.locator('[data-testid="scoreboard-table"] tbody tr');
    await expect(rows).toHaveCount(5);
  });

  test('all 9 column headers present', async ({ page, mockApi }) => {
    await mockApi([{ url: '**/api/models', body: mockScoreboard() }]);
    await page.goto('/scoreboard');
    const headers = page.locator('[data-testid="scoreboard-table"] thead th');
    await expect(headers).toHaveCount(9);
    const text = await headers.allTextContents();
    const joined = text.join(' ').toLowerCase();
    for (const col of ['rank', 'model', 'gen', 'architecture', 'win rate', 'sharpe', 'p&l', 'efficiency', 'score']) {
      expect(joined).toContain(col);
    }
  });

  test('rows sorted by composite score descending', async ({ page, mockApi }) => {
    await mockApi([{ url: '**/api/models', body: mockScoreboard() }]);
    await page.goto('/scoreboard');
    const scoreCells = page.locator('[data-testid="scoreboard-table"] tbody tr td:last-child');
    const scores = (await scoreCells.allTextContents()).map((s) => parseFloat(s));
    for (let i = 1; i < scores.length; i++) {
      expect(scores[i]).toBeLessThanOrEqual(scores[i - 1]);
    }
  });

  test('click row navigates to model detail', async ({ page, mockApi }) => {
    await mockApi([{ url: '**/api/models', body: mockScoreboard() }]);
    await page.goto('/scoreboard');
    const firstRow = page.locator('[data-testid="scoreboard-table"] tbody tr').first();
    await firstRow.click();
    await expect(page).toHaveURL(/models\/aaa11111/);
  });

  test('empty registry shows empty state', async ({ page, mockApi }) => {
    await mockApi([{ url: '**/api/models', body: mockEmptyScoreboard() }]);
    await page.goto('/scoreboard');
    await expect(page.locator('[data-testid="empty-state"]')).toBeVisible();
  });

  test('API error shows error message', async ({ page, mockApi }) => {
    await mockApi([{ url: '**/api/models', body: { detail: 'Server error' }, status: 500 }]);
    await page.goto('/scoreboard');
    await expect(page.locator('[data-testid="error"]')).toBeVisible();
  });
});
