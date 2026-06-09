import {
  ACTION_SIZE,
  MUSCLE_COUNT,
  POLICY_TIMESTEP,
  type PolicyAction,
  type PolicyObservation,
} from './types'

export class ScriptedMusclePolicy {
  private elapsed = 0

  run(_observation: PolicyObservation): PolicyAction {
    this.elapsed += POLICY_TIMESTEP
    const action = new Float32Array(ACTION_SIZE)
    const phaseVelocity = 5.4
    const waveLength = 0.78

    for (let muscle = 0; muscle < MUSCLE_COUNT; muscle += 2) {
      const segment = muscle / 2
      const phase = this.elapsed * phaseVelocity - segment * waveLength
      const drive = Math.sin(phase)
      action[muscle] = drive
      action[muscle + 1] = -drive
    }

    return action
  }

  reset() {
    this.elapsed = 0
  }
}
