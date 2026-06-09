import { spawnSync } from 'node:child_process'
import { resolve } from 'node:path'

const root = resolve(import.meta.dirname, '..')

const python = spawnSync(
  'uv',
  [
    'run',
    'python',
    '-c',
    `
import json
import numpy as np
from wurmkickflip_rl.env import WurmkickflipEnv

env = WurmkickflipEnv()
_observation, reset_info = env.reset(seed=123)
action = np.zeros(env.action_space.shape, dtype=np.float32)
last_info = reset_info
for _ in range(8):
    _observation, _reward, terminated, truncated, last_info = env.step(action)
    if terminated or truncated:
        break

print(json.dumps({
    "resetKeys": sorted(reset_info.keys()),
    "stepKeys": sorted(last_info.keys()),
    "distance": last_info["distance"],
    "survivalTime": last_info["survival_time"],
    "averageContactRatio": last_info["average_contact_ratio"],
    "energyUse": last_info["energy_use"],
    "fallReason": last_info["fall_reason"],
}))
`,
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

const result = JSON.parse(python.stdout) as {
  resetKeys: string[]
  stepKeys: string[]
  distance: number
  survivalTime: number
  averageContactRatio: number
  energyUse: number
  fallReason: string
}

const requiredKeys = ['average_contact_ratio', 'contact_ratio', 'distance', 'energy_use', 'fall_reason', 'survival_time']
const failures: string[] = []
for (const key of requiredKeys) {
  if (!result.resetKeys.includes(key)) failures.push(`reset info missing ${key}.`)
  if (!result.stepKeys.includes(key)) failures.push(`step info missing ${key}.`)
}
if (result.distance <= 0) failures.push('distance must increase after stepping.')
if (result.survivalTime <= 0) failures.push('survivalTime must increase after stepping.')
if (result.averageContactRatio < 0 || result.averageContactRatio > 1) {
  failures.push('averageContactRatio must be in [0, 1].')
}
if (result.energyUse !== 0) failures.push('zero action energyUse must stay 0.')
if (result.fallReason !== 'none') failures.push('zero-action smoke rollout should not fail immediately.')

if (failures.length > 0) {
  console.error('Environment metrics verification failed:')
  for (const failure of failures) {
    console.error(`- ${failure}`)
  }
  process.exit(1)
}

console.log('Environment metrics verification passed.')
