import { spawnSync } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { validateCreatureGenome } from '../src/creature/configValidation'

const root = resolve(import.meta.dirname, '..')
const outPath = resolve(root, 'training/runs/evolution_smoke/summary.json')
const bestCreaturePath = resolve(root, 'training/runs/evolution_smoke/best-creature.json')
const manifestPath = resolve(root, 'training/runs/evolution_smoke/manifest.json')

const python = spawnSync(
  'uv',
  [
    'run',
    'python',
    '-m',
    'wurmkickflip_rl.evolve',
    '--generations',
    '2',
    '--population-size',
    '4',
    '--elite-count',
    '2',
    '--eval-seeds',
    '101',
    '202',
    '--seed',
    '13',
    '--out',
    'runs/evolution_smoke/summary.json',
    '--export-creature',
    'runs/evolution_smoke/best-creature.json',
    '--export-manifest',
    'runs/evolution_smoke/manifest.json',
  ],
  {
    cwd: resolve(root, 'training'),
    encoding: 'utf8',
  },
)

if (python.status !== 0) {
  console.error(python.stdout)
  console.error(python.stderr)
  process.exit(python.status ?? 1)
}

const summary = JSON.parse(await readFile(outPath, 'utf8')) as {
  schemaVersion?: number
  kind?: string
  creatureId?: string
  environmentId?: string
  generationSummaries?: unknown[]
  best?: { fitness?: number; controller?: Record<string, unknown>; fallReasons?: Record<string, number> }
  bestCreaturePath?: string
}
const bestCreature = JSON.parse(await readFile(bestCreaturePath, 'utf8')) as unknown
const manifest = JSON.parse(await readFile(manifestPath, 'utf8')) as { kind?: string; creatures?: unknown[] }

const failures: string[] = []
if (summary.schemaVersion !== 1) failures.push('schemaVersion must equal 1.')
if (summary.kind !== 'wurmkickflip.evolutionSummary') failures.push('kind mismatch.')
if (summary.creatureId !== 'segmented-starter') failures.push('creatureId mismatch.')
if (summary.environmentId !== 'adaptive-skate-terrarium') failures.push('environmentId mismatch.')
if (!Array.isArray(summary.generationSummaries) || summary.generationSummaries.length !== 2) {
  failures.push('generationSummaries must contain 2 generations.')
}
if (typeof summary.best?.fitness !== 'number' || !Number.isFinite(summary.best.fitness)) {
  failures.push('best fitness must be finite.')
}
if (summary.best?.controller?.kind !== 'cpg') {
  failures.push('best controller kind must be cpg.')
}
if (!summary.best?.fallReasons || Object.keys(summary.best.fallReasons).length === 0) {
  failures.push('best fallReasons must include at least one reason count.')
}
if (summary.bestCreaturePath !== 'runs\\evolution_smoke\\best-creature.json' && summary.bestCreaturePath !== 'runs/evolution_smoke/best-creature.json') {
  failures.push('summary.bestCreaturePath must point to the exported creature.')
}
const bestCreatureValidation = validateCreatureGenome(bestCreature)
if (!bestCreatureValidation.ok) {
  failures.push(`exported best creature must validate: ${bestCreatureValidation.errors.slice(0, 2).join(' ')}`)
} else {
  if (!bestCreatureValidation.value.id.startsWith('segmented-starter-evolved-')) {
    failures.push('exported best creature id must include the evolved candidate id.')
  }
  if (typeof bestCreatureValidation.value.controller.parameters.evolutionFitness !== 'number') {
    failures.push('exported best creature must include evolutionFitness in controller parameters.')
  }
}
if (manifest.kind !== 'wurmkickflip.generatedCreatureManifest') {
  failures.push('generated manifest kind mismatch.')
}
if (!Array.isArray(manifest.creatures) || manifest.creatures.length !== 1) {
  failures.push('generated manifest must include one creature path.')
}

if (failures.length > 0) {
  console.error('Evolution verification failed:')
  for (const failure of failures) {
    console.error(`- ${failure}`)
  }
  process.exit(1)
}

console.log('Evolution verification passed.')
