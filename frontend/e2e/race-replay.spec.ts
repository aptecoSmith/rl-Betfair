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

  test('select race renders chart container and bet panel', async ({ page, mockApi }) => {
    await mockApi(replayMocks());
    await page.goto('/replay');
    await selectCascade(page);

    await expect(page.locator('[data-testid="chart-container"]')).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('[data-testid="bet-panel"]')).toBeVisible();
    await expect(page.locator('[data-testid="summary-bar"]')).toBeVisible();
  });

  test('uPlot canvas is rendered in chart container', async ({ page, mockApi }) => {
    await mockApi(replayMocks());
    await page.goto('/replay');
    await selectCascade(page);

    // uPlot renders a canvas element inside the target div
    await expect(page.locator('[data-testid="uplot-target"] canvas')).toBeVisible({ timeout: 10_000 });
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

  test('conclusion panel shows race result', async ({ page, mockApi }) => {
    await mockApi(replayMocks());
    await page.goto('/replay');
    await selectCascade(page);

    const conclusion = page.locator('[data-testid="conclusion-panel"]');
    await expect(conclusion).toBeVisible({ timeout: 10_000 });
    // Winner from mock data
    await expect(conclusion).toContainText('Star Runner');
    await expect(conclusion).toContainText('Race Result');
  });

  test('bet panel shows bets and updates during playback', async ({ page, mockApi }) => {
    await mockApi(replayMocks());
    await page.goto('/replay');
    await selectCascade(page);

    const betPanel = page.locator('[data-testid="bet-panel"]');
    await expect(betPanel).toBeVisible({ timeout: 10_000 });

    // At initial tick (tick 0), no bets should be visible yet (bets start at tick 3)
    // Seek slider to end to show all bets
    const slider = page.locator('[data-testid="tick-slider"]');
    await slider.fill('9'); // last tick index
    await page.waitForTimeout(300);

    const betCards = page.locator('[data-testid="bet-card"]');
    const count = await betCards.count();
    expect(count).toBe(3); // 3 bets in mock data
  });

  test('clicking a bet card jumps to correct tick', async ({ page, mockApi }) => {
    await mockApi(replayMocks());
    await page.goto('/replay');
    await selectCascade(page);

    // Seek to last tick so all bets are visible
    const slider = page.locator('[data-testid="tick-slider"]');
    await slider.fill('9');
    await page.waitForTimeout(300);

    // Click first bet card
    const firstCard = page.locator('[data-testid="bet-card"]').first();
    await firstCard.click();
    await page.waitForTimeout(300);

    // Check tick counter updated (first bet is at tick 3)
    const counter = page.locator('[data-testid="tick-counter"]');
    await expect(counter).toContainText('4 / 10'); // tick index 3 → "Tick 4"
  });

  test('runner legend toggles runner visibility', async ({ page, mockApi }) => {
    await mockApi(replayMocks());
    await page.goto('/replay');
    await selectCascade(page);

    const legendItems = page.locator('[data-testid="runner-legend"] .legend-item');
    await expect(legendItems.first()).toBeVisible({ timeout: 10_000 });

    // Click first legend item to toggle
    await legendItems.first().click();
    await page.waitForTimeout(200);

    // It should get the dimmed class
    await expect(legendItems.first()).toHaveClass(/dimmed/);

    // Click again to restore
    await legendItems.first().click();
    await page.waitForTimeout(200);
    await expect(legendItems.first()).not.toHaveClass(/dimmed/);
  });
});
