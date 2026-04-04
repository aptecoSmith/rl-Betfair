import { test, expect } from './fixtures';
import {
  mockExtractedDays,
  mockBackupDays,
  mockAdminAgents,
  mockEmptyDays,
  mockEmptyAgents,
  mockEmptyBackup,
  mockStreamrecorderBackups,
  mockEmptyStreamrecorderBackups,
  mockRestoreResponse,
  mockDeleteAgentResponse,
  mockImportRangeResponse,
  mockPurgeResponse,
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

  // ── Restore wizard tests ──────────────────────────────────────

  test('wizard: open shows scanning then backup list', async ({ page, mockApi }) => {
    await mockApi([
      ...defaultAdminMocks(),
      { url: '**/api/admin/streamrecorder-backups', body: mockStreamrecorderBackups() },
    ]);
    await page.goto('/admin');
    await page.locator('[data-testid="open-restore-wizard"]').click();

    // Step 1: scanning should flash briefly
    // Step 2: backup list should appear
    await expect(page.locator('[data-testid="start-restore"]')).toBeVisible({ timeout: 5_000 });
    // Verify 3 rows in the table
    await expect(page.locator('.wizard-table tbody tr')).toHaveCount(3);
  });

  test('wizard: already-extracted dates have disabled checkbox', async ({ page, mockApi }) => {
    await mockApi([
      ...defaultAdminMocks(),
      { url: '**/api/admin/streamrecorder-backups', body: mockStreamrecorderBackups() },
    ]);
    await page.goto('/admin');
    await page.locator('[data-testid="open-restore-wizard"]').click();
    await expect(page.locator('[data-testid="start-restore"]')).toBeVisible({ timeout: 5_000 });

    // 2026-04-01 is already_extracted — checkbox should be disabled
    await expect(page.locator('[data-testid="restore-checkbox-2026-04-01"]')).toBeDisabled();
    // 2026-04-02 is new — checkbox should be enabled
    await expect(page.locator('[data-testid="restore-checkbox-2026-04-02"]')).toBeEnabled();
  });

  test('wizard: select and deselect dates updates button text', async ({ page, mockApi }) => {
    await mockApi([
      ...defaultAdminMocks(),
      { url: '**/api/admin/streamrecorder-backups', body: mockStreamrecorderBackups() },
    ]);
    await page.goto('/admin');
    await page.locator('[data-testid="open-restore-wizard"]').click();
    await expect(page.locator('[data-testid="start-restore"]')).toBeVisible({ timeout: 5_000 });

    // Nothing selected — button disabled
    await expect(page.locator('[data-testid="start-restore"]')).toBeDisabled();

    // Select one date
    await page.locator('[data-testid="restore-checkbox-2026-04-02"]').check();
    await expect(page.locator('[data-testid="start-restore"]')).toBeEnabled();
    await expect(page.locator('[data-testid="start-restore"]')).toContainText('1 Date');

    // Select another
    await page.locator('[data-testid="restore-checkbox-2026-04-03"]').check();
    await expect(page.locator('[data-testid="start-restore"]')).toContainText('2 Dates');

    // Deselect All
    await page.locator('button:has-text("Deselect All")').click();
    await expect(page.locator('[data-testid="start-restore"]')).toBeDisabled();
  });

  test('wizard: Select All New selects only non-extracted', async ({ page, mockApi }) => {
    await mockApi([
      ...defaultAdminMocks(),
      { url: '**/api/admin/streamrecorder-backups', body: mockStreamrecorderBackups() },
    ]);
    await page.goto('/admin');
    await page.locator('[data-testid="open-restore-wizard"]').click();
    await expect(page.locator('[data-testid="start-restore"]')).toBeVisible({ timeout: 5_000 });

    await page.locator('button:has-text("Select All New")').click();
    // 2 new dates selected (04-02 and 04-03), not 04-01 which is already extracted
    await expect(page.locator('[data-testid="start-restore"]')).toContainText('2 Dates');
  });

  test('wizard: empty backup folder shows message', async ({ page, mockApi }) => {
    await mockApi([
      ...defaultAdminMocks(),
      { url: '**/api/admin/streamrecorder-backups', body: mockEmptyStreamrecorderBackups() },
    ]);
    await page.goto('/admin');
    await page.locator('[data-testid="open-restore-wizard"]').click();
    await expect(page.getByText(/no backup pairs/i)).toBeVisible({ timeout: 5_000 });
  });

  test('wizard: scan error shows error message', async ({ page, mockApi }) => {
    await mockApi([
      ...defaultAdminMocks(),
      { url: '**/api/admin/streamrecorder-backups', body: { detail: 'Backup folder not found' }, status: 500 },
    ]);
    await page.goto('/admin');
    await page.locator('[data-testid="open-restore-wizard"]').click();
    // The wizard should show an error (wizardError signal set on HTTP error)
    await expect(page.getByText(/failed to scan|backup folder/i)).toBeVisible({ timeout: 5_000 });
  });

  test('wizard: cancel from step 2 closes wizard', async ({ page, mockApi }) => {
    await mockApi([
      ...defaultAdminMocks(),
      { url: '**/api/admin/streamrecorder-backups', body: mockStreamrecorderBackups() },
    ]);
    await page.goto('/admin');
    await page.locator('[data-testid="open-restore-wizard"]').click();
    await expect(page.locator('[data-testid="start-restore"]')).toBeVisible({ timeout: 5_000 });

    await page.locator('button:has-text("Cancel")').click();
    // Wizard should be hidden — the open button should be re-enabled
    await expect(page.locator('[data-testid="open-restore-wizard"]')).toBeEnabled();
    await expect(page.locator('[data-testid="start-restore"]')).not.toBeVisible();
  });

  // ── Agent management tests ────────────────────────────────────

  test('delete agent: dialog appears, confirm deletes', async ({ page, mockApi }) => {
    await mockApi([
      ...defaultAdminMocks(),
      { url: '**/api/admin/agents/**', body: mockDeleteAgentResponse() },
    ]);
    await page.goto('/admin');
    const deleteBtn = page.locator('[data-testid="agents-table"] tbody tr').first().locator('button:has-text("Delete")');
    await deleteBtn.click();
    await expect(page.locator('[data-testid="delete-agent-dialog"]')).toBeVisible();
    await page.locator('[data-testid="confirm-delete-agent"]').click();
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
  });

  test('purge discarded shows success message', async ({ page, mockApi }) => {
    await mockApi([
      ...defaultAdminMocks(),
      { url: '**/api/admin/purge-discarded', body: mockPurgeResponse() },
    ]);
    await page.goto('/admin');
    await page.locator('button:has-text("Purge Discarded")').click();
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible({ timeout: 5_000 });
  });

  // ── Import range test ─────────────────────────────────────────

  test('import range shows progress state', async ({ page, mockApi }) => {
    await mockApi([
      ...defaultAdminMocks(),
      { url: '**/api/admin/import-range', body: mockImportRangeResponse() },
    ]);
    await page.goto('/admin');
    // Fill in range inputs and click Import Range
    await page.locator('[data-testid="import-range-start"]').fill('2026-03-26');
    await page.locator('[data-testid="import-range-end"]').fill('2026-03-28');
    await page.locator('[data-testid="import-range-btn"]').click();
    // Should show importing state or success
    await expect(
      page.locator('[data-testid="success-message"]').or(page.locator('button:has-text("Importing...")'))
    ).toBeVisible({ timeout: 5_000 });
  });
});
