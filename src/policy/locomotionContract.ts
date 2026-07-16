import contractJson from '../../contracts/locomotion-v2.json'

type LocomotionContract = {
  schemaVersion: 2
  kind: 'wurmkickflip.locomotionContract'
  plantVersion: string
  segmentCount: number
  muscleChannelCount: number
  timestep: number
  postIntegrationRelaxationPasses: number
  sensorNames: string[]
  inputWeightNames: string[]
  recurrentWeightNames: string[]
  outputWeightNames: string[]
  joint: {
    stiffness: number
    damping: number
    limit: number
  }
  dynamics: {
    spacing: number
    actuatorStiffness: number
    actuatorDamping: number
    shapeBendScale: number
    constraintIterations: number
    constraintCompliance: number
    longitudinalFriction: number
    lateralFriction: number
    freeVelocityDrag: number
    maximumSpeed: number
    maximumVerticalSpeed: number
    baseRadius: number
    baseGroundClearance: number
  }
}

export const LOCOMOTION_CONTRACT = contractJson as LocomotionContract

if (
  LOCOMOTION_CONTRACT.segmentCount !== 16 ||
  LOCOMOTION_CONTRACT.muscleChannelCount !== LOCOMOTION_CONTRACT.segmentCount * 2 ||
  !Number.isInteger(LOCOMOTION_CONTRACT.postIntegrationRelaxationPasses) ||
  LOCOMOTION_CONTRACT.postIntegrationRelaxationPasses < 0
) {
  throw new Error(
    'locomotion contract must describe 16 antagonistic actuator pairs / 32 muscle channels and a non-negative relaxation pass count',
  )
}
