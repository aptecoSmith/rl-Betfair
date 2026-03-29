import { test, expect } from './fixtures';
import { mockIdleTraining, mockRunningTraining, mockTrainingInfo } from './mocks/training.mock';

test.describe('Training Monitor', () => {
  test('idle state shows start form with inputs', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/training/status', body: mockIdleTraining() },
      { url: '**/api/training/info', body: mockTrainingInfo() },
    ]);
    await page.goto('/training');
    await expect(page.locator('[data-testid="idle-state"]')).toBeVisible();
    await expect(page.locator('[data-testid="start-form"]')).toBeVisible();
    await expect(page.locator('[data-testid="input-population"]')).toBeVisible();
    await expect(page.locator('[data-testid="input-generations"]')).toBeVisible();
    await expect(page.locator('[data-testid="input-epochs"]')).toBeVisible();
  });

  test('running state shows ETA bars and stop button', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/training/status', body: mockRunningTraining() },
      { url: '**/api/training/info', body: mockTrainingInfo() },
    ]);
    await page.goto('/training');
    await expect(page.locator('[data-testid="eta-bars"]')).toBeVisible();
    await expect(page.locator('[data-testid="stop-btn"]')).toBeVisible();
    await expect(page.locator('[data-testid="stop-btn"]')).toContainText(/stop/i);
  });

  test('data summary chips show day counts', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/training/status', body: mockIdleTraining() },
      { url: '**/api/training/info', body: mockTrainingInfo() },
    ]);
    await page.goto('/training');
    await expect(page.locator('[data-testid="data-summary"]')).toBeVisible();
    await expect(page.locator('[data-testid="data-chip"]')).toContainText('3');
  });

  test('form inputs have correct min/max ranges', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/training/status', body: mockIdleTraining() },
      { url: '**/api/training/info', body: mockTrainingInfo() },
    ]);
    await page.goto('/training');
    const popInput = page.locator('[data-testid="input-population"]');
    await expect(popInput).toHaveAttribute('min', '2');
    await expect(popInput).toHaveAttribute('max', '200');

    const genInput = page.locator('[data-testid="input-generations"]');
    await expect(genInput).toHaveAttribute('min', '1');
    await expect(genInput).toHaveAttribute('max', '50');

    const epochInput = page.locator('[data-testid="input-epochs"]');
    await expect(epochInput).toHaveAttribute('min', '1');
    await expect(epochInput).toHaveAttribute('max', '20');
  });
});
