import { expect, test, type Page } from '@playwright/test'
import { readFile } from 'node:fs/promises'

test('mounts the production terrarium with WebGL and its neural brain online', async ({ page }) => {
  const runtimeErrors = watchRuntimeErrors(page)
  await page.goto('/', { waitUntil: 'networkidle' })

  await expect(page.getByRole('heading', { level: 1, name: /Wurm\s*Kickflip/i })).toBeVisible()
  const simulation = page.getByRole('region', { name: 'Wurmkickflip terrarium simulation' })
  await expect(simulation).toBeVisible()

  const canvas = simulation.locator('canvas')
  await expect(canvas).toBeVisible()
  const webgl = await canvas.evaluate(element => {
    if (!(element instanceof HTMLCanvasElement))
      throw new Error('Simulation canvas was not a canvas element.')
    const context = element.getContext('webgl2') ?? element.getContext('webgl')
    return {
      available: context !== null,
      contextLost: context?.isContextLost() ?? true,
      version: context?.getParameter(context.VERSION) ?? null,
    }
  })
  expect(webgl.available, 'the Three.js canvas should expose a WebGL context').toBe(true)
  expect(webgl.contextLost, 'the WebGL context should remain healthy').toBe(false)
  expect(webgl.version).toMatch(/WebGL/i)

  const neuralStatus = page.getByRole('status')
  await expect(neuralStatus).toContainText('Online', { timeout: 15_000 })
  await expect(neuralStatus).toContainText('Evolved recurrent gait / Neural JS')
  expect(runtimeErrors).toEqual([])
})

test('pause, reset, and assignment controls update the running simulation', async ({ page }) => {
  const runtimeErrors = watchRuntimeErrors(page)
  await page.goto('/', { waitUntil: 'networkidle' })
  await expect(page.getByRole('status')).toContainText('Online', { timeout: 15_000 })

  const freeCrawl = page.getByRole('button', { name: /Free crawl/i })
  await freeCrawl.click()
  await expect(freeCrawl).toHaveAttribute('aria-pressed', 'true')
  await expect(page.getByRole('button', { name: /Autonomous life/i })).toHaveAttribute(
    'aria-pressed',
    'false',
  )

  await page.getByRole('button', { name: 'Pause' }).click()
  const play = page.getByRole('button', { name: 'Play', exact: true })
  await expect(play).toBeVisible()
  await expect(play).toHaveAttribute('aria-pressed', 'true')

  await page.getByRole('button', { name: 'Reset' }).click()
  await expect(page.getByRole('button', { name: 'Pause' })).toBeVisible()
  await expect(freeCrawl).toHaveAttribute('aria-pressed', 'true')

  const autonomousLife = page.getByRole('button', { name: /Autonomous life/i })
  await autonomousLife.click()
  await expect(autonomousLife).toHaveAttribute('aria-pressed', 'true')
  await expect(freeCrawl).toHaveAttribute('aria-pressed', 'false')
  expect(runtimeErrors).toEqual([])
})

test('honors reduced-motion preferences', async ({ page }) => {
  const runtimeErrors = watchRuntimeErrors(page)
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('/', { waitUntil: 'networkidle' })

  expect(await page.evaluate(() => matchMedia('(prefers-reduced-motion: reduce)').matches)).toBe(true)
  await expect(page.getByText(/Reduced motion is on/i)).toBeVisible()
  const time = page.getByLabel(/Simulation time/i).locator('b')
  const play = page.getByRole('button', { name: 'Play', exact: true })
  await expect(play).toBeVisible()
  await expect(time).toHaveText('0.0s')
  await page.waitForTimeout(750)
  await expect(time).toHaveText('0.0s')

  const animationDurationMs = await page.locator('.live-dot').evaluate(element => {
    const value = getComputedStyle(element).animationDuration
    const amount = Number.parseFloat(value)
    return value.endsWith('ms') ? amount : amount * 1000
  })
  expect(animationDurationMs).toBeLessThanOrEqual(0.02)

  await play.click()
  await expect(page.getByRole('button', { name: 'Pause', exact: true })).toBeVisible()
  await expect(time).not.toHaveText('0.0s', { timeout: 3_000 })
  expect(runtimeErrors).toEqual([])
})

test('preserves valid configs and retries a failed resource in place', async ({ page }) => {
  const runtimeErrors = watchRuntimeErrors(page)
  let failTripod = true
  await page.route('**/configs/creatures/tripod-pusher.json', async route => {
    if (failTripod) {
      await route.fulfill({ status: 503, contentType: 'application/json', body: '{"error":"synthetic"}' })
    } else {
      await route.continue()
    }
  })

  await page.goto('/', { waitUntil: 'networkidle' })
  await expect(page.getByText('Some configuration resources failed', { exact: true })).toBeVisible()
  await expect(page.getByText('/configs/creatures/tripod-pusher.json')).toBeVisible()
  await page.locator('.genome-drawer summary').click()
  const environmentSelect = page.getByRole('combobox', { name: 'Environment', exact: true })
  const creatureSelect = page.getByRole('combobox', { name: 'Creature genome', exact: true })
  await expect(environmentSelect).toBeEnabled()
  await expect(creatureSelect.locator('option')).toHaveCount(3)

  failTripod = false
  await page.getByRole('button', { name: 'Retry configuration' }).click()
  await expect(page.getByText('Some configuration resources failed', { exact: true })).toBeHidden()
  await expect(creatureSelect.locator('option')).toHaveCount(4)
  expect(runtimeErrors.filter(error => !error.includes('status of 503'))).toEqual([])
})

test('captures, exports, replays, and rejects a tampered recorder-core artifact', async ({ page }) => {
  const runtimeErrors = watchRuntimeErrors(page)
  await page.goto('/', { waitUntil: 'networkidle' })
  await expect(page.getByRole('status')).toContainText('Online', { timeout: 15_000 })

  await page.locator('.replay-panel summary').click()
  await page.getByRole('button', { name: 'Start capture' }).click()
  const replayStatus = page.locator('.replay-status')
  await expect(replayStatus).toContainText('Recording')
  await page.waitForTimeout(550)
  await expect(replayStatus).toContainText(/Recording [1-9]\d* fixed-step frames/)
  await page.getByRole('button', { name: 'Finish capture' }).click()
  await expect(replayStatus).toContainText(/Captured [1-9]\d* frames/)

  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Export replay JSON' }).click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toMatch(/^browser-.+\.json$/)
  const downloadedPath = await download.path()
  expect(downloadedPath).not.toBeNull()
  const artifact = JSON.parse(await readFile(downloadedPath!, 'utf8')) as {
    frameCount: number
    source: { creatureId: string; environmentId: string; modelVersion: string }
    environmentSample: { seed: number }
    frames: Array<{ reward: number; muscleActivations: number[] }>
  }
  expect(artifact.frameCount).toBeGreaterThan(0)
  expect(artifact.frames).toHaveLength(artifact.frameCount)
  expect(artifact.frames.every(frame => frame.muscleActivations.length === 32)).toBe(true)
  expect(artifact.source.creatureId).toBeTruthy()
  expect(artifact.source.environmentId).toBeTruthy()
  expect(artifact.source.modelVersion).toBeTruthy()
  expect(Number.isFinite(artifact.environmentSample.seed)).toBe(true)

  await page.getByRole('button', { name: 'Load captured replay' }).click()
  await expect(replayStatus).toContainText('Playback 0.00')
  await expect(page.getByText('Terrarium replay', { exact: true })).toBeVisible()
  const time = page.getByLabel(/Simulation time/i).locator('b')
  await expect(time).toHaveText('0.0s')
  await page.getByRole('button', { name: 'Play replay' }).click()
  await expect(time).not.toHaveText('0.0s', { timeout: 3_000 })
  await page.getByRole('button', { name: 'Pause replay' }).click()
  const pausedTime = await time.textContent()
  await page.waitForTimeout(350)
  await expect(time).toHaveText(pausedTime!)
  await page.getByRole('button', { name: 'Restart replay' }).click()
  await expect(time).toHaveText('0.0s')
  await page.getByRole('button', { name: 'Return to live' }).click()
  await expect(page.getByText('Terrarium live', { exact: true })).toBeVisible()

  await page.getByRole('button', { name: 'Pause', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Play', exact: true })).toBeVisible()
  // Allow the final throttled live telemetry publication to settle after the
  // synchronous pause before proving the rejected import cannot reset it.
  await page.waitForTimeout(200)
  const liveTime = await time.textContent()
  artifact.frames[0].reward += 1
  await page.getByLabel('Import replay JSON').setInputFiles({
    name: 'tampered-replay.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(artifact)),
  })
  await expect(page.getByRole('alert')).toContainText('integrity.digest does not match')
  await expect(page.getByRole('alert')).toContainText('untampered recorder-core JSON')
  await expect(page.getByText('Terrarium live', { exact: true })).toBeVisible()
  await page.waitForTimeout(250)
  await expect(time).toHaveText(liveTime!)
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
