import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { copyFile, mkdir, mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

type JsonRecord = Record<string, unknown>

const root = resolve(import.meta.dirname, '..')
const trainingRoot = resolve(root, 'training')
const retainedWorkspace = process.env.WURMKICKFLIP_REPRO_WORKSPACE
const workspace = retainedWorkspace
  ? resolve(retainedWorkspace)
  : await mkdtemp(join(tmpdir(), 'wurmkickflip-published-locomotion-'))
if (retainedWorkspace) await mkdir(workspace, { recursive: false })

try {
  const stagedSeedDirectory = join(workspace, 'seeds')
  await mkdir(stagedSeedDirectory, { recursive: true })
  const stagedLegacySeed = join(stagedSeedDirectory, 'legacy-v1.json')
  await copyFile(resolve(trainingRoot, 'seeds/wurmkickflip_locomotion_warm_start_v1.json'), stagedLegacySeed)

  const baseOutput = join(workspace, 'base-model.json')
  const baseSummary = join(workspace, 'base-summary.json')
  runEvolution([
    '--seed',
    '20260720',
    '--generations',
    '40',
    '--population-size',
    '96',
    '--elite-count',
    '20',
    '--episode-steps',
    '360',
    '--model-version',
    'locomotion-articulated-contact-es-v2',
    '--warm-start',
    stagedLegacySeed,
    '--out',
    baseOutput,
    '--summary',
    baseSummary,
  ])

  const reproducedOutput = join(workspace, 'published-model.json')
  const reproducedSummary = join(workspace, 'published-summary.json')
  runEvolution([
    '--seed',
    '20260721',
    '--generations',
    '10',
    '--population-size',
    '64',
    '--elite-count',
    '12',
    '--episode-steps',
    '480',
    '--model-version',
    'locomotion-articulated-contact-es-v2.1-candidate',
    '--warm-start',
    baseOutput,
    '--out',
    reproducedOutput,
    '--summary',
    reproducedSummary,
  ])

  const base = asRecord(JSON.parse(await readFile(baseOutput, 'utf8')), 'base artifact')
  const reproduced = asRecord(JSON.parse(await readFile(reproducedOutput, 'utf8')), 'reproduced artifact')
  const tracked = asRecord(
    JSON.parse(await readFile(resolve(root, 'public/models/wurmkickflip_locomotion_policy.json'), 'utf8')),
    'tracked artifact',
  )

  verifyRecipe(base, {
    seed: 20260720,
    generations: 40,
    populationSize: 96,
    eliteCount: 20,
    episodeSteps: 360,
    modelVersion: 'locomotion-articulated-contact-es-v2',
  })
  verifyRecipe(reproduced, {
    seed: 20260721,
    generations: 10,
    populationSize: 64,
    eliteCount: 12,
    episodeSteps: 480,
    modelVersion: 'locomotion-articulated-contact-es-v2.1-candidate',
  })
  verifyStableWarmStart(base, 'base artifact')
  verifyStableWarmStart(reproduced, 'refined artifact')

  const baseBytes = await readFile(baseOutput)
  const refinedWarmStart = asRecord(
    asRecord(reproduced.training, 'reproduced.training').warmStart,
    'reproduced.training.warmStart',
  )
  const baseSha256 = createHash('sha256').update(canonicalLineEndings(baseBytes)).digest('hex')
  if (refinedWarmStart.sha256 !== baseSha256) {
    throw new Error('refined artifact warmStart.sha256 does not identify the generated base artifact')
  }

  // Older tracked candidates contained a relative path. It is non-semantic and
  // deliberately omitted from newly generated artifacts before object comparison.
  const trackedCanonical = JSON.stringify(withoutWarmStartPaths(tracked))
  const reproducedCanonical = JSON.stringify(withoutWarmStartPaths(reproduced))
  if (trackedCanonical !== reproducedCanonical) {
    const trackedWarm = asOptionalRecord(asOptionalRecord(tracked.training)?.warmStart)
    throw new Error(
      [
        'published locomotion artifact is not reproduced by the documented two-stage run',
        `tracked warm SHA: ${String(trackedWarm?.sha256 ?? 'missing')}`,
        `reproduced warm SHA: ${String(refinedWarmStart.sha256)}`,
        'If only provenance changed, publish the newly reproduced artifact after reviewing its metrics.',
      ].join('\n'),
    )
  }

  console.log(
    JSON.stringify(
      {
        modelVersion: tracked.modelVersion,
        canonicalSha256: createHash('sha256').update(trackedCanonical).digest('hex'),
        base: { seed: 20260720, generations: 40, populationSize: 96, eliteCount: 20 },
        refinement: { seed: 20260721, generations: 10, populationSize: 64, eliteCount: 12 },
        warmStartSha256: refinedWarmStart.sha256,
      },
      null,
      2,
    ),
  )
  console.log('Published locomotion evolution reproduced exactly in an isolated workspace.')
} finally {
  if (!retainedWorkspace) await rm(workspace, { recursive: true, force: true })
}

function runEvolution(arguments_: string[]) {
  const result = spawnSync(
    'uv',
    ['run', 'python', '-m', 'wurmkickflip_rl.evolve_locomotion_policy', ...arguments_],
    { cwd: trainingRoot, encoding: 'utf8', maxBuffer: 8 * 1024 * 1024 },
  )
  if (result.status !== 0) {
    throw new Error(`published locomotion evolution failed:\n${result.stderr || result.stdout}`)
  }
}

function verifyRecipe(
  artifact: JsonRecord,
  expected: {
    seed: number
    generations: number
    populationSize: number
    eliteCount: number
    episodeSteps: number
    modelVersion: string
  },
) {
  const training = asRecord(artifact.training, `${expected.modelVersion}.training`)
  for (const key of ['seed', 'generations', 'populationSize', 'eliteCount', 'episodeSteps'] as const) {
    if (training[key] !== expected[key]) {
      throw new Error(`${expected.modelVersion} training.${key} must equal ${expected[key]}`)
    }
  }
  if (artifact.modelVersion !== expected.modelVersion) {
    throw new Error(`unexpected modelVersion for stage ${expected.seed}`)
  }
}

function verifyStableWarmStart(artifact: JsonRecord, label: string) {
  const training = asRecord(artifact.training, `${label}.training`)
  const warmStart = asRecord(training.warmStart, `${label}.training.warmStart`)
  const keys = Object.keys(warmStart).sort()
  if (JSON.stringify(keys) !== JSON.stringify(['modelVersion', 'sha256'])) {
    throw new Error(`${label} warm-start metadata must contain only modelVersion and sha256`)
  }
  if (typeof warmStart.modelVersion !== 'string' || !/^[a-f\d]{64}$/u.test(String(warmStart.sha256))) {
    throw new Error(`${label} warm-start metadata is malformed`)
  }
}

function withoutWarmStartPaths(value: JsonRecord): JsonRecord {
  const clone = structuredClone(value)
  const training = asOptionalRecord(clone.training)
  const warmStart = asOptionalRecord(training?.warmStart)
  if (warmStart) delete warmStart.path
  return clone
}

function canonicalLineEndings(value: Buffer) {
  return Buffer.from(value.toString('utf8').replace(/\r\n?/gu, '\n'))
}

function asRecord(value: unknown, label: string): JsonRecord {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`)
  }
  return value as JsonRecord
}

function asOptionalRecord(value: unknown): JsonRecord | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as JsonRecord)
    : undefined
}
