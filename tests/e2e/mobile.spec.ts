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

  const forge = page.getByRole('region', { name: 'Seed Forge' })
  await forge.scrollIntoViewIfNeeded()
  await expect(forge.locator('.seed-forge__parameter')).toHaveCount(14)
  const forgeGeometry = await forge.evaluate(element => ({
    left: element.getBoundingClientRect().left,
    right: element.getBoundingClientRect().right,
    scrollWidth: element.scrollWidth,
    clientWidth: element.clientWidth,
    viewportWidth: innerWidth,
  }))
  expect(forgeGeometry.left).toBeGreaterThanOrEqual(0)
  expect(forgeGeometry.right).toBeLessThanOrEqual(forgeGeometry.viewportWidth + 1)
  expect(forgeGeometry.scrollWidth).toBeLessThanOrEqual(forgeGeometry.clientWidth + 1)
  const forgeButtonHeights = await forge
    .locator('.seed-forge__button, .seed-forge__preset, .seed-forge__lock')
    .evaluateAll(buttons => buttons.map(button => button.getBoundingClientRect().height))
  expect(forgeButtonHeights.every(height => height >= 44)).toBe(true)
  await expect(forge.getByLabel('uint32 seed')).toHaveCSS('height', '44px')

  await page.locator('.forge-trials > summary').click()
  const olympics = page.getByRole('region', { name: 'Wurm Olympics' })
  await olympics.scrollIntoViewIfNeeded()
  const olympicsGeometry = await olympics.evaluate(element => ({
    left: element.getBoundingClientRect().left,
    right: element.getBoundingClientRect().right,
    scrollWidth: element.scrollWidth,
    clientWidth: element.clientWidth,
    viewportWidth: innerWidth,
  }))
  expect(olympicsGeometry.left).toBeGreaterThanOrEqual(0)
  expect(olympicsGeometry.right).toBeLessThanOrEqual(olympicsGeometry.viewportWidth + 1)
  expect(olympicsGeometry.scrollWidth).toBeLessThanOrEqual(olympicsGeometry.clientWidth + 1)
  const olympicsRun = olympics.getByRole('button', { name: 'Run 16-seed meet' })
  expect(await olympicsRun.evaluate(button => button.getBoundingClientRect().height)).toBeGreaterThanOrEqual(
    44,
  )

  await page.setViewportSize({ width: 320, height: 640 })
  await olympics.scrollIntoViewIfNeeded()
  const narrowViewport = await page.evaluate(() => ({
    innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }))
  expect(narrowViewport.scrollWidth).toBeLessThanOrEqual(narrowViewport.innerWidth + 1)
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
