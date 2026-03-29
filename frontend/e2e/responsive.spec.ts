import { test, expect } from './fixtures';
import { mockScoreboard } from './mocks/scoreboard.mock';

test.describe('Responsive layout', () => {
  test('scoreboard table fits at 1280x800 (no horizontal overflow)', async ({ page, mockApi }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await mockApi([{ url: '**/api/models', body: mockScoreboard() }]);
    await page.goto('/scoreboard');
    await expect(page.locator('[data-testid="scoreboard-table"]')).toBeVisible();

    const hasOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    expect(hasOverflow).toBe(false);
  });

  test('scoreboard renders at 768x1024 (tablet)', async ({ page, mockApi }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await mockApi([{ url: '**/api/models', body: mockScoreboard() }]);
    await page.goto('/scoreboard');
    await expect(page.locator('[data-testid="scoreboard-table"]')).toBeVisible();

    // At tablet width the table should be visible — either it fits or it's in a scrollable container
    const tableInfo = await page.evaluate(() => {
      const table = document.querySelector('[data-testid="scoreboard-table"]');
      if (!table) return { visible: false };
      const rect = table.getBoundingClientRect();
      return { visible: rect.width > 0 && rect.height > 0, width: rect.width };
    });
    expect(tableInfo.visible).toBe(true);
    expect(tableInfo.width).toBeGreaterThan(0);
  });

  test('header nav links visible at 768px width', async ({ page, mockApi }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await mockApi([{ url: '**/api/models', body: mockScoreboard() }]);
    await page.goto('/scoreboard');

    const navLinks = page.locator('nav a, .nav-links a, header a');
    const count = await navLinks.count();
    expect(count).toBeGreaterThanOrEqual(3);

    for (let i = 0; i < count; i++) {
      await expect(navLinks.nth(i)).toBeVisible();
    }
  });
});
