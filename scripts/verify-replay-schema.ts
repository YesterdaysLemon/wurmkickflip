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
if (validation.value.source.policyBackend !== 'neural-js') {
  console.error('Canonical replay fixture must exercise the default neural-js backend.')
  process.exit(1)
}

const invalid = structuredClone(validation.value)
invalid.frameCount = invalid.frames.length + 1
const invalidValidation = validateReplayArtifact(invalid)
if (invalidValidation.ok || !invalidValidation.errors.includes('replay.frameCount must equal replay.frames length 3.')) {
  console.error('TypeScript replay validation did not reject frameCount mismatch.')
  process.exit(1)
}

const invalidBackend = structuredClone(validation.value) as unknown as {
  source: { policyBackend: string }
}
invalidBackend.source.policyBackend = 'unregistered-backend'
const invalidBackendValidation = validateReplayArtifact(invalidBackend)
if (
  invalidBackendValidation.ok ||
  !invalidBackendValidation.errors.some((error) => error.startsWith('replay.source.policyBackend must be one of:'))
) {
  console.error('TypeScript replay validation accepted an unregistered policy backend.')
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
if (!pythonConstants.includes('"neural-js"')) {
  console.error('Python policy backend allowlist is missing neural-js.')
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

const pythonRejectsUnknownBackend = spawnSync(
  'uv',
  [
    'run',
    'python',
    '-c',
    [
      'import json',
      'from pathlib import Path',
      'from wurmkickflip_rl.replay_schema import validate_replay_artifact',
      'artifact = json.loads(Path("../fixtures/replay-artifact.json").read_text(encoding="utf-8"))',
      'artifact["source"]["policyBackend"] = "unregistered-backend"',
      'errors = validate_replay_artifact(artifact)',
      'raise SystemExit(0 if any(error.startswith("replay.source.policyBackend must be one of:") for error in errors) else 1)',
    ].join('; '),
  ],
  {
    cwd: resolve(root, 'training'),
    encoding: 'utf8',
  },
)

if (pythonRejectsUnknownBackend.status !== 0) {
  console.error('Python replay validation accepted an unregistered policy backend.')
  console.error(pythonRejectsUnknownBackend.stdout)
  console.error(pythonRejectsUnknownBackend.stderr)
  process.exit(pythonRejectsUnknownBackend.status ?? 1)
}

console.log('Replay schema verification passed.')
