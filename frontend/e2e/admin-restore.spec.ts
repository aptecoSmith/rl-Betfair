/**
 * Integration test for the restore wizard.
 * Requires: API on 8001, frontend on 4202, MySQL running, StreamRecorder backups present.
 * Restores 2026-04-02 from backup → MySQL → Parquet, then verifies the new day appears.
 */
import { test, expect } from './fixtures';

const TARGET_DATE = '2026-04-02';
const API = 'http://localhost:8001';

test.describe('Restore Wizard (integration)', () => {
  test.skip(({ backendAvailable }) => !backendAvailable,
    'Skipped — backend not running on localhost:8001');

  // Give the full flow up to 5 minutes
  test.setTimeout(300_000);

  test(`restore ${TARGET_DATE} produces parquet files`, async ({ page }) => {
    // ── Setup: delete target date if it already exists ──────────
    const daysResp = await fetch(`${API}/admin/days`);
    const { days } = await daysResp.json() as { days: { date: string }[] };
    if (days.some(d => d.date === TARGET_DATE)) {
      const del = await fetch(`${API}/admin/days/${TARGET_DATE}?confirm=true`, { method: 'DELETE' });
      expect(del.ok).toBe(true);
    }

    // Verify precondition: target date is NOT in extracted days
    const checkResp = await fetch(`${API}/admin/days`);
    const checkDays = await checkResp.json() as { days: { date: string }[] };
    expect(checkDays.days.map(d => d.date)).not.toContain(TARGET_DATE);

    // ── Open admin page ─────────────────────────────────────────
    await page.goto('/admin');
    await expect(page.locator('[data-testid="open-restore-wizard"]')).toBeVisible();

    // ── Open wizard (step 1 → 2) ────────────────────────────────
    await page.locator('[data-testid="open-restore-wizard"]').click();

    // Wait for backup list to load (step 2)
    await expect(page.locator('[data-testid="start-restore"]')).toBeVisible({ timeout: 30_000 });

    // ── Select target date ──────────────────────────────────────
    const checkbox = page.locator(`[data-testid="restore-checkbox-${TARGET_DATE}"]`);
    await expect(checkbox).toBeVisible({ timeout: 5_000 });
    await checkbox.check();

    // Verify the restore button shows 1 date
    await expect(page.locator('[data-testid="start-restore"]')).toContainText('1 Date');

    // ── Start restore (step 3) ──────────────────────────────────
    await page.locator('[data-testid="start-restore"]').click();

    // Wizard should move to step 3 — restoring
    await expect(page.getByText(/Restoring.*date/)).toBeVisible({ timeout: 5_000 });

    // ── Wait for completion (step 4) ────────────────────────────
    // The done text appears when the restore finishes
    const doneText = page.locator('[data-testid="restore-done-text"]');
    await expect(doneText).toBeVisible({ timeout: 300_000 });

    // ── Assert success ──────────────────────────────────────────
    await expect(doneText).toContainText('1 date(s) processed');
    await expect(doneText).not.toContainText('failed');

    // Close wizard
    await page.locator('button:has-text("Close")').click();

    // ── Verify parquet file appears in extracted days table ─────
    const finalResp = await fetch(`${API}/admin/days`);
    const finalDays = await finalResp.json() as { days: { date: string }[] };
    expect(finalDays.days.map(d => d.date)).toContain(TARGET_DATE);

    // ── Teardown: delete so the test is re-runnable ────────────
    await fetch(`${API}/admin/days/${TARGET_DATE}?confirm=true`, { method: 'DELETE' });
  });

  test('already-extracted dates show as disabled in wizard', async ({ page }) => {
    // 2026-03-31 and 2026-04-01 should already exist as parquet files
    const daysResp = await fetch(`${API}/admin/days`);
    const { days } = await daysResp.json() as { days: { date: string }[] };
    const existingDate = days[0]?.date;
    test.skip(!existingDate, 'No extracted days available to test');

    await page.goto('/admin');
    await page.locator('[data-testid="open-restore-wizard"]').click();
    await expect(page.locator('[data-testid="start-restore"]')).toBeVisible({ timeout: 30_000 });

    // The already-extracted date should have a disabled checkbox
    const checkbox = page.locator(`[data-testid="restore-checkbox-${existingDate}"]`);
    // If this date is in the backup list, it should be disabled
    const count = await checkbox.count();
    if (count > 0) {
      await expect(checkbox).toBeDisabled();
    }

    // Should show "extracted" badge for this date
    const row = page.locator(`.wizard-table tr:has([data-testid="restore-checkbox-${existingDate}"])`);
    if (await row.count() > 0) {
      await expect(row.locator('.status-active')).toBeVisible();
    }

    await page.locator('button:has-text("Cancel")').click();
  });
});
