import { expect, test } from '@playwright/test';

test('public research and diagnostic workbench routes render cleanly', async ({ page }) => {
  const routes: Array<[string, RegExp]> = [
    ['/', /Bone Disorder Differential/],
    ['/about', /About OsteoGuard AI/],
    ['/architecture', /A layered research platform/],
    ['/workbench', /^Workbench$/],
    ['/tracks', /Research tracks/],
    ['/radiomics', /Radiomics/],
    ['/biomechanics', /Biomechanics/],
    ['/explainability', /Explainability/],
    ['/uncertainty', /Uncertainty/],
    ['/governance', /Governance/],
    ['/evidence', /Model Evidence/],
    ['/audit', /Research Feedback Log/],
    ['/results', /No current analysis result/],
  ];

  for (const [path, heading] of routes) {
    await page.goto(path);
    await expect(page.getByRole('heading', { name: heading }).first()).toBeVisible();
  }

  await expect(page.getByText(/FOUR_CLASS_MODEL_NOT_READY/)).toBeVisible();
});

test('running Paget disease and OI CT/X-ray workflows opens full XAI dashboard', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: /Paget's Disease \(Pelvis X-ray\)/i }).click();
  await expect(page).toHaveURL(/\/results$/);
  await expect(page.getByRole('heading', { name: "Paget's Disease of Bone" })).toBeVisible();
  await expect(page.getByText(/Visual Explainability \(Grad-CAM\)/i)).toBeVisible();
  await expect(page.getByText(/Radiomics Feature Attribution \(SHAP\)/i)).toBeVisible();
  await expect(page.getByText(/What Would Change the Diagnosis/i)).toBeVisible();
});

test('Track D remains blocked and mobile navigation exposes implemented pages', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/workbench');
  await expect(page.getByRole('heading', { name: 'Workbench' })).toBeVisible();
  await page.getByLabel('Research track').selectOption('track_d');
  await expect(page.getByRole('complementary', { name: 'Model readiness' }).getByText('FOUR_CLASS_MODEL_NOT_READY')).toBeVisible();
  await expect(page.getByRole('link', { name: 'Review governance' })).toHaveAttribute('href', '/governance');

  await page.getByRole('button', { name: 'Open navigation menu' }).click();
  const mobileNavigation = page.locator('#mobile-primary-navigation');
  await expect(mobileNavigation.getByRole('link', { name: 'Governance' })).toBeVisible();
  await expect(mobileNavigation.getByRole('link', { name: 'Workbench' })).toHaveAttribute('aria-current', 'page');
});
