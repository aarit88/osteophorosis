import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  reporter: 'dot',
  outputDir: 'test-results/playwright',
  use: {
    ...devices['Desktop Chrome'],
    baseURL: 'http://127.0.0.1:3000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: 'npm run dev',
    url: 'http://127.0.0.1:3000/health',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: {
      NODE_ENV: 'development',
      HOST: '127.0.0.1',
      PORT: '3000',
      OSTEO_DISABLE_MOCK_FALLBACK: 'true',
      OSTEO_DISABLE_DEMO_SEED: 'true',
      OSTEO_STRICT_CLINICAL_MODE: 'true',
    },
  },
});
