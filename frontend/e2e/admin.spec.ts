import { test, expect } from './fixtures';
import {
  mockExtractedDays,
  mockBackupDays,
  mockAdminAgents,
  mockEmptyDays,
  mockEmptyAgents,
  mockEmptyBackup,
} from './mocks/admin.mock';

function defaultAdminMocks() {
  return [
    { url: '**/api/admin/days', body: mockExtractedDays() },
    { url: '**/api/admin/backup-days', body: mockBackupDays() },
    { url: '**/api/admin/agents', body: mockAdminAgents() },
  ];
}

test.describe('Admin', () => {
  test('all 3 sections load', async ({ page, mockApi }) => {
    await mockApi(defaultAdminMocks());
    await page.goto('/admin');
    await expect(page.locator('[data-testid="days-table"]')).toBeVisible();
    await expect(page.locator('[data-testid="backup-table"]')).toBeVisible();
    await expect(page.locator('[data-testid="agents-table"]')).toBeVisible();
  });

  test('delete day: dialog appears, cancel dismisses', async ({ page, mockApi }) => {
    await mockApi(defaultAdminMocks());
    await page.goto('/admin');
    const deleteBtn = page.locator('[data-testid="days-table"] tbody tr').first().locator('button:has-text("Delete")');
    await deleteBtn.click();
    await expect(page.locator('[data-testid="delete-day-dialog"]')).toBeVisible();
    await page.locator('[data-testid="cancel-delete-day"]').click();
    await expect(page.locator('[data-testid="delete-day-dialog"]')).not.toBeVisible();
  });

  test('delete day: confirm deletes and shows success', async ({ page, mockApi }) => {
    await mockApi([
      ...defaultAdminMocks(),
      { url: '**/api/admin/days/**', body: { deleted: true, detail: 'Deleted 2026-03-26' } },
    ]);
    await page.goto('/admin');
    const deleteBtn = page.locator('[data-testid="days-table"] tbody tr').first().locator('button:has-text("Delete")');
    await deleteBtn.click();
    await page.locator('[data-testid="confirm-delete-day"]').click();
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
  });

  test('reset: wrong confirmation text keeps button disabled', async ({ page, mockApi }) => {
    await mockApi(defaultAdminMocks());
    await page.goto('/admin');
    await page.locator('button:has-text("Start Afresh")').click();
    await expect(page.locator('[data-testid="reset-dialog"]')).toBeVisible();
    await page.locator('[data-testid="reset-input"]').fill('WRONG_TEXT');
    await expect(page.locator('[data-testid="confirm-reset"]')).toBeDisabled();
  });

  test('reset: DELETE_EVERYTHING enables confirm button', async ({ page, mockApi }) => {
    await mockApi(defaultAdminMocks());
    await page.goto('/admin');
    await page.locator('button:has-text("Start Afresh")').click();
    await page.locator('[data-testid="reset-input"]').fill('DELETE_EVERYTHING');
    await expect(page.locator('[data-testid="confirm-reset"]')).toBeEnabled();
  });

  test('empty registry shows empty state messages', async ({ page, mockApi }) => {
    await mockApi([
      { url: '**/api/admin/days', body: mockEmptyDays() },
      { url: '**/api/admin/backup-days', body: mockEmptyBackup() },
      { url: '**/api/admin/agents', body: mockEmptyAgents() },
    ]);
    await page.goto('/admin');
    // Wait for loading to finish — look for "No extracted days" and "No agents" text
    await expect(page.getByText(/no extracted days/i).or(page.getByText(/no.*found/i))).toBeVisible({ timeout: 5000 });
    await expect(page.getByText(/no agents/i).or(page.getByText(/no.*registry/i))).toBeVisible({ timeout: 5000 });
  });

  test('import single day shows importing state', async ({ page, mockApi }) => {
    await mockApi([
      ...defaultAdminMocks(),
      { url: '**/api/admin/import-day', body: { success: true, date: '2026-03-28', detail: 'Imported' } },
    ]);
    await page.goto('/admin');
    const importBtn = page.locator('[data-testid="backup-table"] tbody tr').first().locator('button:has-text("Import")');
    await importBtn.click();
    await expect(
      page.locator('[data-testid="success-message"]').or(page.locator('button:has-text("Importing...")'))
    ).toBeVisible({ timeout: 5000 });
  });
});
