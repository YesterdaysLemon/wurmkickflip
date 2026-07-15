import { createHash } from 'node:crypto'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { spawnSync } from 'node:child_process'

const root = resolve(import.meta.dirname, '..')
const temp = await mkdtemp(join(tmpdir(), 'wurmkickflip-published-locomotion-'))

try {
  const output = join(temp, 'locomotion.json')
  const summary = join(temp, 'summary.json')
  const result = spawnSync(
    'uv',
    [
      'run',
      'python',
      '-m',
      'wurmkickflip_rl.evolve_locomotion_policy',
      '--seed',
      '20260719',
      '--generations',
      '80',
      '--population-size',
      '128',
      '--elite-count',
      '18',
      '--episode-steps',
      '420',
      '--model-version',
      'locomotion-segmental-es-quality-robust-v1',
      '--warm-start',
      'seeds/wurmkickflip_locomotion_warm_start_v1.json',
      '--out',
      output,
      '--summary',
      summary,
    ],
    { cwd: resolve(root, 'training'), encoding: 'utf8', maxBuffer: 4 * 1024 * 1024 },
  )
  if (result.status !== 0) {
    throw new Error(`published locomotion evolution failed: ${result.stderr || result.stdout}`)
  }

  const tracked = JSON.parse(
    await readFile(resolve(root, 'public/models/wurmkickflip_locomotion_policy.json'), 'utf8'),
  )
  const reproduced = JSON.parse(await readFile(output, 'utf8'))
  const trackedCanonical = JSON.stringify(tracked)
  const reproducedCanonical = JSON.stringify(reproduced)
  if (trackedCanonical !== reproducedCanonical) {
    throw new Error('published locomotion artifact is not reproduced by the documented full evolution run')
  }

  console.log(
    JSON.stringify(
      {
        modelVersion: tracked.modelVersion,
        canonicalSha256: createHash('sha256').update(trackedCanonical).digest('hex'),
        seed: tracked.training.seed,
        generations: tracked.training.generations,
        populationSize: tracked.training.populationSize,
        warmStartSha256: tracked.training.warmStart?.sha256,
      },
      null,
      2,
    ),
  )
  console.log('Published locomotion evolution reproduced exactly.')
} finally {
  await rm(temp, { recursive: true, force: true })
}
