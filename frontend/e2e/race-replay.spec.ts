import { test, expect } from './fixtures';
import { mockScoreboard } from './mocks/scoreboard.mock';
import { mockModelDetail } from './mocks/model-detail.mock';
import { mockReplayDay, mockReplayRace } from './mocks/replay.mock';

const MODEL_ID = 'aaa11111-1111-1111-1111-111111111111';

function replayMocks() {
  return [
    { url: '**/api/models', body: mockScoreboard() },
    { url: `**/api/models/${MODEL_ID}`, body: mockModelDetail() },
    { url: '**/api/models/*/lineage', body: { nodes: [] } },
    { url: '**/api/models/*/genetics', body: { events: [] } },
    // Order matters: more specific pattern first
    { url: '**/api/replay/*/*/*', body: mockReplayRace() },
    { url: '**/api/replay/*/*', body: mockReplayDay() },
  ];
}

async function selectCascade(page: import('@playwright/test').Page) {
  // Select model
  await page.locator('[data-testid="model-select"]').selectOption({ index: 1 });
  // Wait for dates to populate from model detail metrics_history
  await page.waitForTimeout(500);

  // Select first real date
  const dateSelect = page.locator('[data-testid="date-select"]');
  const dateOptions = await dateSelect.locator('option').allTextContents();
  const dateOption = dateOptions.find(o => /\d{4}/.test(o));
  if (dateOption) {
    await dateSelect.selectOption({ label: dateOption });
  } else {
    await dateSelect.selectOption({ index: 1 });
  }

  // Wait for races to populate from getReplayDay
  await page.waitForTimeout(500);

  // Select first real race
  const raceSelect = page.locator('[data-testid="race-select"]');
  const raceOptions = await raceSelect.locator('option').allTextContents();
  const raceOption = raceOptions.find(o => o !== 'Select race…' && o.trim() !== '');
  if (raceOption) {
    await raceSelect.selectOption({ label: raceOption });
  } else {
    await raceSelect.selectOption({ index: 1 });
  }
}

test.describe('Race Replay', () => {
  test('empty state before selection', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/models', body: mockScoreboard() },
    ]);
    await page.goto('/replay');
    await expect(page.locator('[data-testid="empty-state"]')).toBeVisible();
  });

  test('select model populates date dropdown', async ({ page, mockApi }) => {
    await mockApi(replayMocks());
    await page.goto('/replay');

    await page.locator('[data-testid="model-select"]').selectOption({ index: 1 });
    await page.waitForTimeout(500);

    const dateSelect = page.locator('[data-testid="date-select"]');
    await expect(dateSelect).toBeVisible();
    await expect(dateSelect).toBeEnabled();
    const options = await dateSelect.locator('option').allTextContents();
    // Should have placeholder + at least 1 date
    expect(options.length).toBeGreaterThanOrEqual(2);
  });

  test('select date populates race dropdown', async ({ page, mockApi }) => {
    await mockApi(replayMocks());
    await page.goto('/replay');

    await page.locator('[data-testid="model-select"]').selectOption({ index: 1 });
    await page.waitForTimeout(500);

    const dateSelect = page.locator('[data-testid="date-select"]');
    const dateOptions = await dateSelect.locator('option').allTextContents();
    const dateOption = dateOptions.find(o => /\d{4}/.test(o));
    if (dateOption) {
      await dateSelect.selectOption({ label: dateOption });
    } else {
      await dateSelect.selectOption({ index: 1 });
    }

    await page.waitForTimeout(500);
    const raceSelect = page.locator('[data-testid="race-select"]');
    await expect(raceSelect).toBeEnabled({ timeout: 10_000 });
    const raceOptions = await raceSelect.locator('option').allTextContents();
    expect(raceOptions.length).toBeGreaterThanOrEqual(2);
  });

  test('select race renders chart and action log', async ({ page, mockApi }) => {
    await mockApi(replayMocks());
    await page.goto('/replay');
    await selectCascade(page);

    await expect(page.locator('[data-testid="chart-container"]')).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('[data-testid="action-log"]')).toBeVisible();
    await expect(page.locator('[data-testid="summary-bar"]')).toBeVisible();
  });

  test('playback controls toggle play/pause', async ({ page, mockApi }) => {
    await mockApi(replayMocks());
    await page.goto('/replay');
    await selectCascade(page);

    const playBtn = page.locator('[data-testid="play-btn"]');
    await expect(playBtn).toBeVisible({ timeout: 10_000 });

    const initialText = (await playBtn.textContent())?.trim();
    await playBtn.click();
    await page.waitForTimeout(300);
    const newText = (await playBtn.textContent())?.trim();
    expect(newText).not.toBe(initialText);
  });
});
