import { test, expect } from './fixtures';
import { mockScoreboard } from './mocks/scoreboard.mock';
import { mockBetExplorer } from './mocks/bet-explorer.mock';
import { mockModelDetail } from './mocks/model-detail.mock';
import { mockReplayDay, mockReplayRace } from './mocks/replay.mock';

const MODEL_ID = 'aaa11111-1111-1111-1111-111111111111';

test.describe('Tab Memory — selection persistence across navigations', () => {

  test('model selected on bet explorer persists to replay', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/models', body: mockScoreboard() },
      { url: '**/api/replay/**/bets', body: mockBetExplorer() },
      { url: `**/api/models/${MODEL_ID}`, body: mockModelDetail() },
      { url: `**/api/replay/${MODEL_ID}/*`, body: mockReplayDay() },
    ]);

    // Select a model on bet explorer
    await page.goto('/bets');
    await page.locator('[data-testid="model-select"]').selectOption({ index: 1 });
    await expect(page.locator('[data-testid="bets-table"]')).toBeVisible({ timeout: 5000 });

    // Navigate to replay
    await page.goto('/replay');
    await page.waitForLoadState('networkidle');

    // Model should be pre-selected (select value should match)
    const replayModelSelect = page.locator('[data-testid="model-select"]');
    await expect(replayModelSelect).toHaveValue(MODEL_ID, { timeout: 5000 });
  });

  test('bet explorer filters persist across navigation', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/models', body: mockScoreboard() },
      { url: '**/api/replay/**/bets', body: mockBetExplorer() },
    ]);

    // Select model and apply filters
    await page.goto('/bets');
    await page.locator('[data-testid="model-select"]').selectOption({ index: 1 });
    await expect(page.locator('[data-testid="bets-table"]')).toBeVisible({ timeout: 5000 });

    // Apply date filter
    const dateSelect = page.locator('[data-testid="filter-date"]');
    const dateOptions = await dateSelect.locator('option:not([value=""])').allTextContents();
    if (dateOptions.length > 0) {
      await dateSelect.selectOption({ index: 1 });
    }

    // Apply action filter
    await page.locator('[data-testid="filter-action"]').selectOption('back');

    // Navigate away to scoreboard
    await page.goto('/scoreboard');
    await page.waitForLoadState('networkidle');

    // Navigate back to bet explorer
    await page.goto('/bets');
    await page.waitForLoadState('networkidle');

    // Model should still be selected and data loaded
    const modelSelect = page.locator('[data-testid="model-select"]');
    await expect(modelSelect).toHaveValue(MODEL_ID, { timeout: 5000 });
    await expect(page.locator('[data-testid="bets-table"]')).toBeVisible({ timeout: 5000 });

    // Action filter should be restored
    const actionFilter = page.locator('[data-testid="filter-action"]');
    await expect(actionFilter).toHaveValue('back');
  });

  test('replay cascade selections persist across navigation', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/models', body: mockScoreboard() },
      { url: `**/api/models/${MODEL_ID}`, body: mockModelDetail() },
      { url: `**/api/replay/${MODEL_ID}/2026-03-26`, body: mockReplayDay() },
      { url: `**/api/replay/${MODEL_ID}/2026-03-26/race-001`, body: mockReplayRace() },
    ]);

    // Select model, date, race on replay
    await page.goto('/replay');
    await page.locator('[data-testid="model-select"]').selectOption({ index: 1 });

    // Wait for dates to load and select one
    const dateSelect = page.locator('[data-testid="date-select"]');
    await expect(dateSelect).toBeEnabled({ timeout: 5000 });
    await dateSelect.selectOption('2026-03-26');

    // Wait for races to load and select one
    const raceSelect = page.locator('[data-testid="race-select"]');
    await expect(raceSelect).toBeEnabled({ timeout: 5000 });
    await raceSelect.selectOption('race-001');

    // Wait for race data to load
    await expect(page.locator('[data-testid="race-data"]').or(page.locator('[data-testid="chart-area"]'))).toBeVisible({ timeout: 5000 });

    // Navigate away
    await page.goto('/scoreboard');
    await page.waitForLoadState('networkidle');

    // Navigate back to replay
    await page.goto('/replay');
    await page.waitForLoadState('networkidle');

    // All three selections should be restored
    await expect(page.locator('[data-testid="model-select"]')).toHaveValue(MODEL_ID, { timeout: 5000 });
    await expect(dateSelect).toHaveValue('2026-03-26', { timeout: 5000 });
    await expect(raceSelect).toHaveValue('race-001', { timeout: 5000 });
  });
});
