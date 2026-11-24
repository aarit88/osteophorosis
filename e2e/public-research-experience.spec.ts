import { expect, test } from '@playwright/test';

test('public research routes render without generated clinical results', async ({ page }) => {
  const routes: Array<[string, RegExp]> = [
    ['/', /A research framework for bone imaging/],
    ['/about', /About OsteoGuard AI/],
    ['/architecture', /A layered research platform/],
    ['/workbench', /^Workbench$/],
    ['/tracks', /Research tracks/],
    ['/radiomics', /Radiomics/],
    ['/biomechanics', /Biomechanics/],
    ['/explainability', /^Explainability$/],
    ['/uncertainty', /Uncertainty and abstention/],
    ['/governance', /Governance/],
    ['/evidence', /Model Evidence/],
    ['/audit', /Research Feedback Log/],
    ['/results', /No current analysis result/],
  ];
  const syntheticCalls: string[] = [];

  await page.route(/\/api\/v1\/(radiomics\/demo|biomechanics\/geometry|explainability\/demo|uncertainty\/demo|workbench\/run-demo)/, async (route) => {
    syntheticCalls.push(route.request().url());
    await route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ error_code: 'SYNTHETIC_DEMO_NOT_AVAILABLE' }) });
  });

  for (const [path, heading] of routes) {
    await page.goto(path);
    await expect(page.getByRole('heading', { name: heading }).first()).toBeVisible();
  }

  await expect(page.getByText(/FOUR_CLASS_MODEL_NOT_READY/)).toBeVisible();
  expect(syntheticCalls).toEqual([]);
});

test('Track D remains blocked and mobile navigation exposes implemented pages', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/workbench');
  await expect(page.getByRole('heading', { name: 'Workbench' })).toBeVisible();
  await page.getByLabel('Research track').selectOption('track_d');
  await expect(page.getByText('FOUR_CLASS_MODEL_NOT_READY')).toBeVisible();
  await expect(page.getByRole('link', { name: 'Review governance' })).toHaveAttribute('href', '/governance');

  await page.getByRole('button', { name: 'Open navigation menu' }).click();
  const mobileNavigation = page.locator('#mobile-primary-navigation');
  await expect(mobileNavigation.getByRole('link', { name: 'Governance' })).toBeVisible();
  await expect(mobileNavigation.getByRole('link', { name: 'Workbench' })).toHaveAttribute('aria-current', 'page');
});

test('application shell exposes keyboard skip navigation and no result history', async ({ page }) => {
  await page.goto('/');
  const skipLink = page.getByRole('link', { name: 'Skip to main content' });
  await expect(skipLink).toHaveAttribute('href', '#main-content');
  await page.keyboard.press('Tab');
  await expect(skipLink).toBeFocused();

  await page.goto('/results');
  await expect(page.getByRole('heading', { name: 'No current analysis result' })).toBeVisible();
  await expect(page.getByText(/FOUR_CLASS_MODEL_NOT_READY/)).toBeVisible();
  await expect(page.getByText(/Backend-reported class/)).toHaveCount(0);
});
