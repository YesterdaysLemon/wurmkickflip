import { expect, test, type Page } from '@playwright/test'

test('keeps the terrarium and controls usable in a narrow mobile viewport', async ({ page }) => {
  const runtimeErrors = watchRuntimeErrors(page)
  await page.goto('/', { waitUntil: 'networkidle' })

  const simulation = page.getByRole('region', { name: 'Wurmkickflip terrarium simulation' })
  await expect(simulation).toBeVisible()
  await expect(simulation.locator('canvas')).toBeVisible()
  await expect(page.getByRole('status', { name: 'Neural controller status' })).toContainText('Online', {
    timeout: 15_000,
  })

  const viewport = await page.evaluate(() => ({
    innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }))
  expect(viewport.scrollWidth).toBeLessThanOrEqual(viewport.innerWidth + 1)

  const pause = page.getByRole('button', { name: 'Pause' })
  await pause.scrollIntoViewIfNeeded()
  await expect(pause).toBeVisible()
  const freeCrawl = page.getByRole('button', { name: /Free crawl/i })
  await freeCrawl.click()
  await expect(freeCrawl).toHaveAttribute('aria-pressed', 'true')
  const microscope = page.getByTestId('gait-microscope')
  await microscope.scrollIntoViewIfNeeded()
  await expect(microscope).toContainText('Evolved crawl', { timeout: 5_000 })
  const mobileGeometry = await microscope.evaluate(element => ({
    left: element.getBoundingClientRect().left,
    right: element.getBoundingClientRect().right,
    viewportWidth: innerWidth,
  }))
  expect(mobileGeometry.left).toBeGreaterThanOrEqual(0)
  expect(mobileGeometry.right).toBeLessThanOrEqual(mobileGeometry.viewportWidth + 1)
  const actionHeights = await microscope
    .locator('.gait-actions button')
    .evaluateAll(buttons => buttons.map(button => button.getBoundingClientRect().height))
  expect(actionHeights.every(height => height >= 40)).toBe(true)
  await page.getByRole('slider', { name: 'Traction scale' }).fill('0.5')
  await expect(microscope).toContainText('Traction 50%')
  await page.getByRole('button', { name: 'Clear perturbations' }).click()
  await expect(microscope).toContainText('Traction 100%')
  expect(runtimeErrors).toEqual([])
})

function watchRuntimeErrors(page: Page) {
  const errors: string[] = []
  page.on('console', message => {
    if (message.type() === 'error') errors.push(`console: ${message.text()}`)
  })
  page.on('pageerror', error => errors.push(`page: ${error.message}`))
  return errors
}
