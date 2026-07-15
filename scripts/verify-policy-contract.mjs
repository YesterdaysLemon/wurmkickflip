import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const root = resolve(import.meta.dirname, '..')
const failures = []
const read = path => readFile(resolve(root, path), 'utf8')
const assert = (condition, message) => {
  if (!condition) failures.push(message)
}
const equal = (actual, expected, message) => {
  assert(JSON.stringify(actual) === JSON.stringify(expected), message)
}
const record = (value, label) => {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    failures.push(`${label} must be an object.`)
    return {}
  }
  return value
}

const [
  contractText,
  artifactText,
  legacyText,
  typesSource,
  runtimeContractSource,
  policySource,
  dynamicsSource,
  pythonContractsSource,
  pythonPlantSource,
  pythonTrainerSource,
  policyDocs,
  locomotionDocs,
] = await Promise.all([
  read('contracts/locomotion-v2.json'),
  read('public/models/wurmkickflip_locomotion_policy.json'),
  read('training/seeds/wurmkickflip_locomotion_warm_start_v1.json'),
  read('src/policy/types.ts'),
  read('src/policy/locomotionContract.ts'),
  read('src/policy/locomotionPolicy.ts'),
  read('src/scene/wormDynamics.ts'),
  read('training/wurmkickflip_rl/contracts.py'),
  read('training/wurmkickflip_rl/articulated_locomotion.py'),
  read('training/wurmkickflip_rl/evolve_locomotion_policy.py'),
  read('docs/policy-contract.md'),
  read('training/LOCOMOTION_POLICY.md'),
])

const contract = record(JSON.parse(contractText), 'locomotion contract')
const artifact = record(JSON.parse(artifactText), 'published locomotion artifact')
const legacy = record(JSON.parse(legacyText), 'legacy locomotion seed')
const joint = record(contract.joint, 'contract.joint')
const dynamics = record(contract.dynamics, 'contract.dynamics')
const weights = record(artifact.weights, 'artifact.weights')
const plant = record(artifact.plant, 'artifact.plant')
const training = record(artifact.training, 'artifact.training')
const warmStart = record(training.warmStart, 'artifact.training.warmStart')

assert(contract.schemaVersion === 2, 'Locomotion contract schemaVersion must equal 2.')
assert(contract.kind === 'wurmkickflip.locomotionContract', 'Locomotion contract kind mismatch.')
assert(contract.plantVersion === 'articulated-contact-v2', 'Locomotion plantVersion mismatch.')
assert(contract.segmentCount === 16, 'Locomotion segmentCount must equal 16.')
assert(contract.muscleChannelCount === 32, 'Locomotion muscleChannelCount must equal 32.')
assert(
  contract.muscleChannelCount === contract.segmentCount * 2,
  'Locomotion muscle channel count must be two antagonistic channels per segment.',
)
assert(contract.timestep === 1 / 60, 'Locomotion timestep must equal 1/60.')
equal(
  contract.sensorNames,
  [
    'targetForward',
    'targetRight',
    'targetDistance',
    'forwardSpeed',
    'angularSpeed',
    'terrainFriction',
    'urgency',
    'contactLoad',
    'slipSpeed',
    'obstacleForward',
    'obstacleRight',
  ],
  'Locomotion sensor ordering mismatch.',
)
equal(
  contract.inputWeightNames,
  [
    'bias',
    'segmentPosition',
    'targetForward',
    'targetRight',
    'targetRightByPosition',
    'targetDistance',
    'forwardSpeed',
    'angularSpeed',
    'terrainFriction',
    'urgency',
    'segmentBend',
    'segmentBendVelocity',
    'previousCommand',
    'contactLoad',
    'slipSpeed',
    'obstacleForward',
    'obstacleRight',
  ],
  'Locomotion input weight ordering mismatch.',
)
equal(
  contract.recurrentWeightNames,
  ['self', 'anteriorNeighbor', 'posteriorNeighbor'],
  'Locomotion recurrent weight ordering mismatch.',
)
equal(
  contract.outputWeightNames,
  [
    'bias',
    'self',
    'anteriorNeighbor',
    'posteriorNeighbor',
    'targetRight',
    'targetRightByPosition',
    'segmentBend',
    'obstacleRight',
    'contactLoad',
  ],
  'Locomotion output weight ordering mismatch.',
)
equal(joint, { stiffness: 22, damping: 7, limit: 1.15 }, 'Locomotion joint contract mismatch.')
equal(
  dynamics,
  {
    spacing: 0.102,
    actuatorStiffness: 78,
    actuatorDamping: 8.5,
    shapeBendScale: 0.19,
    constraintIterations: 6,
    constraintCompliance: 0.035,
    longitudinalFriction: 1.15,
    lateralFriction: 17.5,
    freeVelocityDrag: 0.42,
    maximumSpeed: 2.4,
    maximumVerticalSpeed: 2.4,
    baseRadius: 0.085,
    baseGroundClearance: 0.105,
  },
  'Locomotion articulated dynamics contract mismatch.',
)
assert(
  !contract.sensorNames.some(name => /time|clock|phase|cycle/i.test(name)),
  'Locomotion contract must remain clock-free.',
)

assert(artifact.schemaVersion === 2, 'Published locomotion schemaVersion must equal 2.')
assert(artifact.kind === 'wurmkickflip.locomotionPolicy', 'Published locomotion artifact kind mismatch.')
assert(artifact.architecture === 'segmental-recurrent-tanh', 'Published locomotion architecture mismatch.')
assert(artifact.segmentCount === contract.segmentCount, 'Published locomotion segmentCount mismatch.')
assert(artifact.actionSize === contract.muscleChannelCount, 'Published locomotion actionSize mismatch.')
equal(artifact.sensorNames, contract.sensorNames, 'Published locomotion sensor names mismatch.')
equal(weights.inputNames, contract.inputWeightNames, 'Published locomotion input names mismatch.')
equal(weights.recurrentNames, contract.recurrentWeightNames, 'Published recurrent names mismatch.')
equal(weights.outputNames, contract.outputWeightNames, 'Published output names mismatch.')
assert(
  Array.isArray(artifact.initialState) && artifact.initialState.length === 16,
  'Initial state length mismatch.',
)
assert(Array.isArray(weights.input) && weights.input.length === 17, 'Input weight length mismatch.')
assert(
  Array.isArray(weights.recurrent) && weights.recurrent.length === 3,
  'Recurrent weight length mismatch.',
)
assert(Array.isArray(weights.output) && weights.output.length === 9, 'Output weight length mismatch.')
assert(plant.version === contract.plantVersion, 'Published plant version mismatch.')
assert(plant.timestep === contract.timestep, 'Published plant timestep mismatch.')
assert(plant.jointStiffness === joint.stiffness, 'Published joint stiffness mismatch.')
assert(plant.jointDamping === joint.damping, 'Published joint damping mismatch.')
assert(plant.jointLimit === joint.limit, 'Published joint limit mismatch.')
for (const [name, expected] of Object.entries(dynamics)) {
  const actual = name === 'maximumVerticalSpeed' && plant[name] === undefined ? expected : plant[name]
  assert(actual === expected, `Published plant.${name} mismatch.`)
}
assert(training.algorithm === 'elitist-mutation-evolution', 'Published evolution algorithm mismatch.')
assert(
  training.objectiveVersion === 'articulated-contact-obstacle-recovery-v2',
  'Published evolution objective mismatch.',
)
assert(training.actuatorPrecision === 'float32-plant-command', 'Published actuator precision mismatch.')
assert(training.scenarioCount === 12, 'Published scenario count mismatch.')
equal(
  Object.keys(warmStart).sort(),
  ['modelVersion', 'sha256'],
  'Published warm-start metadata must not contain a filesystem path.',
)
assert(/^[a-f\d]{64}$/.test(String(warmStart.sha256)), 'Published warm-start SHA-256 is malformed.')
equal(
  training.domainRandomization,
  ['obstacles', 'spatial-friction', 'body-scale', 'target-switches', 'contact-loss'],
  'Published domain randomization list mismatch.',
)

assert(legacy.schemaVersion === 1, 'Tracked legacy warm start must remain schema v1.')
equal(
  legacy.sensorNames,
  contract.sensorNames.slice(0, 7),
  'Legacy sensor names must be the v2 global-sensor prefix.',
)
const legacyWeights = record(legacy.weights, 'legacy.weights')
equal(
  legacyWeights.inputNames,
  contract.inputWeightNames.slice(0, 13),
  'Legacy input names must be the v2 input prefix.',
)
equal(
  legacyWeights.outputNames,
  contract.outputWeightNames.slice(0, 7),
  'Legacy output names must be the v2 output prefix.',
)

assert(
  runtimeContractSource.includes("import contractJson from '../../contracts/locomotion-v2.json'"),
  'TypeScript locomotion runtime must import the canonical JSON contract.',
)
assert(
  typesSource.includes('LOCOMOTION_CONTRACT.segmentCount') &&
    typesSource.includes('LOCOMOTION_CONTRACT.muscleChannelCount') &&
    typesSource.includes('LOCOMOTION_CONTRACT.timestep'),
  'TypeScript policy dimensions and timestep must derive from the locomotion contract.',
)
assert(
  policySource.includes('LOCOMOTION_CONTRACT.sensorNames') &&
    policySource.includes('LOCOMOTION_CONTRACT.inputWeightNames') &&
    policySource.includes('LOCOMOTION_CONTRACT.outputWeightNames'),
  'TypeScript locomotion name vectors must derive from the canonical contract.',
)
assert(
  dynamicsSource.includes("import { LOCOMOTION_CONTRACT } from '../policy/locomotionContract'"),
  'Browser articulated dynamics must import the canonical contract.',
)
assert(dynamicsSource.includes('stepArticulatedWorm'), 'Browser dynamics must expose the articulated plant.')
assert(
  pythonContractsSource.includes('contracts" / "locomotion-v2.json'),
  'Python mounted constants must derive dimensions and timestep from the canonical contract.',
)
assert(
  pythonPlantSource.includes('CONTRACT_PATH = ROOT / "contracts" / "locomotion-v2.json"'),
  'Python articulated plant must load the canonical contract.',
)
assert(
  pythonTrainerSource.includes('SENSOR_NAMES = tuple(CONTRACT["sensorNames"])') &&
    pythonTrainerSource.includes('INPUT_WEIGHT_NAMES = tuple(CONTRACT["inputWeightNames"])'),
  'Python evolution name vectors must derive from the canonical contract.',
)
assert(
  pythonTrainerSource.includes('articulated_plant_step(') &&
    !/wave_gain|maximum_forward_acceleration|inverse_traction_drag|front_turn_base/i.test(
      pythonTrainerSource,
    ),
  'Python evolution must use the articulated plant without removed scalar fields.',
)
const warmLoader = pythonTrainerSource.slice(
  pythonTrainerSource.indexOf('def load_warm_start'),
  pythonTrainerSource.indexOf('\ndef evolve'),
)
assert(!/["']path["']\s*:/.test(warmLoader), 'Warm-start metadata must not include filesystem paths.')

for (const [name, text] of [
  ['docs/policy-contract.md', policyDocs],
  ['training/LOCOMOTION_POLICY.md', locomotionDocs],
]) {
  assert(text.includes('16'), `${name} must document the 16-segment locomotion lattice.`)
  assert(text.includes('32'), `${name} must document the 32-channel action contract.`)
}

if (failures.length > 0) {
  console.error('Policy contract verification failed:')
  for (const failure of failures) console.error(`- ${failure}`)
  process.exit(1)
}

console.log('Policy contracts verified from the canonical articulated-contact-v2 JSON contract.')
