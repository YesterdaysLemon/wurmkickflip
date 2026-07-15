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

def rollout(creature=None, environment=None):
    env = WurmkickflipEnv(creature_config=creature, environment_config=environment)
    _observation, info = env.reset(seed=321)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0::2] = 0.55
    action[1::2] = -0.55
    for _ in range(90):
        _observation, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return float(info["distance"])

def morphology_distance_delta():
    import copy
    import json
    from pathlib import Path
    base = json.loads(Path("../public/configs/creatures/segmented-starter.json").read_text(encoding="utf-8"))
    heavy = copy.deepcopy(base)
    for part in heavy["morphology"]["bodyParts"]:
        part["size"] = [value * 1.35 for value in part["size"]]
        part["mass"] = part["mass"] * 1.8
        part["material"]["friction"] = part["material"]["friction"] * 0.55
    return abs(rollout(base) - rollout(heavy))

def environment_distance_delta():
    import copy
    import json
    from pathlib import Path
    base = json.loads(Path("../public/configs/environments/adaptive-skate-terrarium.json").read_text(encoding="utf-8"))
    slick = copy.deepcopy(base)
    slick["randomization"]["frictionScale"] = [0.2, 0.2]
    slick["randomization"]["skateboardMass"] = [2.4, 2.4]
    slick["randomization"]["wheelFriction"] = [0.25, 0.25]
    grippy = copy.deepcopy(base)
    grippy["randomization"]["frictionScale"] = [1.6, 1.6]
    grippy["randomization"]["skateboardMass"] = [0.55, 0.55]
    grippy["randomization"]["wheelFriction"] = [1.5, 1.5]
    return abs(rollout(environment=slick) - rollout(environment=grippy))

print(json.dumps({
    "resetKeys": sorted(reset_info.keys()),
    "stepKeys": sorted(last_info.keys()),
    "distance": last_info["distance"],
    "survivalTime": last_info["survival_time"],
    "averageContactRatio": last_info["average_contact_ratio"],
    "energyUse": last_info["energy_use"],
    "fallReason": last_info["fall_reason"],
    "morphologyDistanceDelta": morphology_distance_delta(),
    "environmentDistanceDelta": environment_distance_delta(),
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
  morphologyDistanceDelta: number
  environmentDistanceDelta: number
}

const requiredKeys = [
  'average_contact_ratio',
  'contact_ratio',
  'distance',
  'energy_use',
  'fall_reason',
  'survival_time',
]
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
if (result.morphologyDistanceDelta <= 0.001) {
  failures.push('morphology changes must affect Python rollout distance.')
}
if (result.environmentDistanceDelta <= 0.001) {
  failures.push('environment config changes must affect Python rollout distance.')
}

if (failures.length > 0) {
  console.error('Environment metrics verification failed:')
  for (const failure of failures) {
    console.error(`- ${failure}`)
  }
  process.exit(1)
}

console.log('Environment metrics verification passed.')
