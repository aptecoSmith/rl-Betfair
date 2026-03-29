import { test, expect } from './fixtures';
import { mockModelDetail, mockLineage, mockGenetics } from './mocks/model-detail.mock';
import { mockScoreboard } from './mocks/scoreboard.mock';

const MODEL_ID = 'aaa11111-1111-1111-1111-111111111111';

function setupModelDetailMocks() {
  return [
    { url: `**/api/models/${MODEL_ID}`, body: mockModelDetail() },
    { url: `**/api/models/${MODEL_ID}/lineage`, body: mockLineage() },
    { url: `**/api/models/${MODEL_ID}/genetics`, body: mockGenetics() },
  ];
}

test.describe('Model Detail', () => {
  test('all sections render', async ({ page, mockApi }) => {
    await mockApi(setupModelDetailMocks());
    await page.goto(`/models/${MODEL_ID}`);
    await expect(page.locator('[data-testid="hyperparams"]')).toBeVisible();
    await expect(page.locator('[data-testid="pnl-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="lineage-tree"]')).toBeVisible();
    await expect(page.locator('[data-testid="genetic-events"]')).toBeVisible();
    await expect(page.locator('[data-testid="metrics-summary"]')).toBeVisible();
  });

  test('hyperparameters table has key-value rows', async ({ page, mockApi }) => {
    await mockApi(setupModelDetailMocks());
    await page.goto(`/models/${MODEL_ID}`);
    const rows = page.locator('[data-testid="hyperparams"] tr, [data-testid="hyperparams"] .param-row');
    await expect(rows.first()).toBeVisible();
    const count = await rows.count();
    expect(count).toBeGreaterThanOrEqual(4);
  });

  test('back button returns to scoreboard', async ({ page, mockApi }) => {
    await mockApi([
      ...setupModelDetailMocks(),
      { url: '**/api/models', body: mockScoreboard() },
    ]);
    await page.goto(`/models/${MODEL_ID}`);
    const backBtn = page.locator('.back-btn, button:has-text("Scoreboard")');
    await expect(backBtn).toBeVisible();
    await backBtn.click();
    await expect(page).toHaveURL(/scoreboard/);
  });

  test('P&L chart renders SVG elements', async ({ page, mockApi }) => {
    await mockApi(setupModelDetailMocks());
    await page.goto(`/models/${MODEL_ID}`);
    const chart = page.locator('[data-testid="pnl-chart"]');
    await expect(chart).toBeVisible();
    const svg = chart.locator('svg');
    await expect(svg).toBeVisible();
    const bars = svg.locator('rect, line, path');
    const count = await bars.count();
    expect(count).toBeGreaterThan(0);
  });

  test('lineage tree renders nodes', async ({ page, mockApi }) => {
    await mockApi(setupModelDetailMocks());
    await page.goto(`/models/${MODEL_ID}`);
    const tree = page.locator('[data-testid="lineage-tree"]');
    await expect(tree).toBeVisible();
    const nodes = tree.locator('[data-testid^="tree-node-"]');
    const count = await nodes.count();
    expect(count).toBeGreaterThanOrEqual(2);
  });

  test('full flow: scoreboard -> detail -> back -> scoreboard', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/models', body: mockScoreboard() },
      ...setupModelDetailMocks(),
    ]);
    await page.goto('/scoreboard');
    await page.locator('[data-testid="scoreboard-table"] tbody tr').first().click();
    await expect(page).toHaveURL(/models\//);
    await expect(page.locator('[data-testid="hyperparams"]')).toBeVisible();

    const backBtn = page.locator('.back-btn, button:has-text("Scoreboard")');
    await backBtn.click();
    await expect(page).toHaveURL(/scoreboard/);
    await expect(page.locator('[data-testid="scoreboard-table"]')).toBeVisible();
  });
});
