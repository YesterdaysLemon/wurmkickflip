import { expect, test, type Page } from '@playwright/test'

test('keeps the terrarium and controls usable in a narrow mobile viewport', async ({ page }) => {
  const runtimeErrors = watchRuntimeErrors(page)
  await page.goto('/', { waitUntil: 'networkidle' })

  const simulation = page.getByRole('region', { name: 'Wurmkickflip terrarium simulation' })
  await expect(simulation).toBeVisible()
  await expect(simulation.locator('canvas')).toBeVisible()
  await expect(page.getByRole('status')).toContainText('Online', { timeout: 15_000 })

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
