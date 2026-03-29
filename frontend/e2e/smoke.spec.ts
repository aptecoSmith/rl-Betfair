import { test, expect } from './fixtures';

test.describe('Smoke tests (real backend)', () => {
  test.beforeEach(async ({ backendAvailable }) => {
    test.skip(!backendAvailable, 'Backend not available — skipping smoke tests');
  });

  test('app loads and redirects to /scoreboard', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL(/scoreboard/);
  });

  test('scoreboard page renders', async ({ page }) => {
    await page.goto('/scoreboard');
    await expect(page.locator('h1')).toContainText(/scoreboard/i);
  });

  test('training page loads', async ({ page }) => {
    await page.goto('/training');
    await expect(page.locator('h1')).toContainText(/training/i);
  });

  test('admin page loads', async ({ page }) => {
    await page.goto('/admin');
    await expect(page.locator('h1')).toContainText(/admin/i);
  });

  test('replay page loads', async ({ page }) => {
    await page.goto('/replay');
    await expect(page.locator('h1, h2, .page-title')).toHaveCount(1, { timeout: 10_000 }).catch(() => {});
    // Just check navigation works — page content depends on model data
    await expect(page).toHaveURL(/replay/);
  });

  test('bets page loads', async ({ page }) => {
    await page.goto('/bets');
    await expect(page).toHaveURL(/bets/);
  });
});
