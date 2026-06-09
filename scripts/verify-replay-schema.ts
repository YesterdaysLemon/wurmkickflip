import { spawnSync } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { REPLAY_KIND, REPLAY_SCHEMA_VERSION } from '../src/replay/types'
import { validateReplayArtifact } from '../src/replay/replayValidation'

const root = resolve(import.meta.dirname, '..')
const fixturePath = resolve(root, 'fixtures/replay-artifact.json')

const fixture = JSON.parse(await readFile(fixturePath, 'utf8')) as unknown
const validation = validateReplayArtifact(fixture)
if (!validation.ok) {
  console.error('TypeScript replay validation failed:')
  for (const error of validation.errors) {
    console.error(`- ${error}`)
  }
  process.exit(1)
}

const invalid = structuredClone(validation.value)
invalid.frameCount = invalid.frames.length + 1
const invalidValidation = validateReplayArtifact(invalid)
if (invalidValidation.ok || !invalidValidation.errors.includes('replay.frameCount must equal replay.frames length 3.')) {
  console.error('TypeScript replay validation did not reject frameCount mismatch.')
  process.exit(1)
}

const pythonConstants = await readFile(resolve(root, 'training/wurmkickflip_rl/replay_schema.py'), 'utf8')
if (!pythonConstants.includes(`REPLAY_SCHEMA_VERSION = ${REPLAY_SCHEMA_VERSION}`)) {
  console.error('Python REPLAY_SCHEMA_VERSION does not match TypeScript.')
  process.exit(1)
}
if (!pythonConstants.includes(`REPLAY_KIND = "${REPLAY_KIND}"`)) {
  console.error('Python REPLAY_KIND does not match TypeScript.')
  process.exit(1)
}

const python = spawnSync('uv', ['run', 'python', '-m', 'wurmkickflip_rl.replay_schema', '../fixtures/replay-artifact.json'], {
  cwd: resolve(root, 'training'),
  encoding: 'utf8',
})

if (python.status !== 0) {
  console.error(python.stdout)
  console.error(python.stderr)
  process.exit(python.status ?? 1)
}

console.log('Replay schema verification passed.')
