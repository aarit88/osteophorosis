# Stage 48 — Accessibility, Responsive Design, and Performance

## Accessibility and motion

- The shared application shell includes a keyboard skip link, a focusable main landmark, route breadcrumbs, and route-change focus management.
- Global `:focus-visible` styling provides a visible keyboard focus indicator.
- `prefers-reduced-motion: reduce` suppresses most CSS transitions and animations.
- New research status panels use headings, labeled sections, `role="status"` for loading, and `role="alert"` for failures.
- Icons that do not convey text are hidden from assistive technology.

## Responsive behavior

The shared shell and pages use responsive Tailwind layouts and horizontally scrollable data tables where needed. The mobile menu and route layout are covered by browser tests when the Playwright browser runtime is installed.

## Performance

Pages continue to use the existing route and component architecture. This change adds no runtime UI dependency other than Playwright's development-only test runner. The production bundle is checked with the existing build command; no production chunk budget or performance benchmark is claimed.

## Known gaps

Automated DOM and browser tests do not replace manual screen-reader, keyboard, zoom, and contrast review. No accessibility conformance certification is claimed.
