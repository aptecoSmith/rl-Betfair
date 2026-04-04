import { test as base, type Page, type Route } from '@playwright/test';

type MockRoute = {
  url: string;
  body: unknown;
  status?: number;
};

type Fixtures = {
  backendAvailable: boolean;
  mockApi: (routes: MockRoute[]) => Promise<void>;
};

export const test = base.extend<Fixtures>({
  backendAvailable: async ({}, use) => {
    let available = false;
    try {
      const resp = await fetch('http://localhost:8001/models');
      available = resp.ok;
    } catch {
      available = false;
    }
    await use(available);
  },

  mockApi: async ({ page }, use) => {
    const setup = async (routes: MockRoute[]) => {
      // Abort WebSocket to prevent console noise
      await page.route('**/api/ws/**', (route: Route) => route.abort());

      for (const r of routes) {
        await page.route(r.url, (route: Route) =>
          route.fulfill({
            status: r.status ?? 200,
            contentType: 'application/json',
            body: JSON.stringify(r.body),
          })
        );
      }
    };
    await use(setup);
  },
});

export { expect } from '@playwright/test';
