import { validateRuntimeProfile, type ConfigValidationResult } from './configValidation'
import type { CreatureGenome, CreatureRuntimeAdapter, RuntimeProfile } from './types'

export const FIXED_ARTICULATED_RUNTIME_PROFILE = {
  schemaVersion: 1,
  kind: 'wurmkickflip.runtimeProfile',
  id: 'fixed-wurm-articulated-v1',
  name: 'Fixed 16-segment articulated worm',
  plantVersion: 'articulated-contact-v2',
  segmentCount: 16,
  actionSize: 32,
  actuatorLayout: 'antagonistic-pairs',
  genomeProjection: 'appearance-only',
  policy: {
    kind: 'segmental-recurrent-json',
    artifactPath: '/models/wurmkickflip_locomotion_policy.json',
  },
} as const satisfies RuntimeProfile

const fixedCapability = {
  plantVersion: 'articulated-contact-v2',
  segmentCount: 16,
  actionSize: 32,
  policyKind: 'segmental-recurrent-json',
  artifactPath: '/models/wurmkickflip_locomotion_policy.json',
} as const

/**
 * Binds a source genome to the only articulated runtime currently implemented.
 * Morphology is projected onto the visual lattice; the genome's declared
 * controller remains provenance/design metadata and is never executed here.
 */
export function createCreatureRuntimeAdapter(
  creature: CreatureGenome,
  profileValue: unknown,
): ConfigValidationResult<CreatureRuntimeAdapter> {
  const validation = validateRuntimeProfile(profileValue)
  if (!validation.ok) return validation
  const profile = validation.value
  const errors: string[] = []

  if (profile.plantVersion !== fixedCapability.plantVersion) {
    errors.push(
      `runtimeProfile.plantVersion ${profile.plantVersion} is incompatible; expected ${fixedCapability.plantVersion}.`,
    )
  }
  if (profile.segmentCount !== fixedCapability.segmentCount) {
    errors.push(
      `runtimeProfile.segmentCount ${profile.segmentCount} is incompatible; expected ${fixedCapability.segmentCount}.`,
    )
  }
  if (profile.actionSize !== fixedCapability.actionSize) {
    errors.push(
      `runtimeProfile.actionSize ${profile.actionSize} is incompatible; expected ${fixedCapability.actionSize}.`,
    )
  }
  if (profile.policy.kind !== fixedCapability.policyKind) {
    errors.push(
      `runtimeProfile.policy.kind ${profile.policy.kind} is incompatible; expected ${fixedCapability.policyKind}.`,
    )
  }
  if (profile.policy.artifactPath !== fixedCapability.artifactPath) {
    errors.push(
      `runtimeProfile.policy.artifactPath ${profile.policy.artifactPath} is incompatible; expected ${fixedCapability.artifactPath}.`,
    )
  }
  if (errors.length > 0) return { ok: false, errors }

  return {
    ok: true,
    value: {
      profile,
      creatureId: creature.id,
      compatibility: 'fixed-articulated-runtime',
      actuatorContract: {
        segmentCount: 16,
        antagonisticPairCount: 16,
        muscleChannelCount: 32,
        channelsPerPair: 2,
        channelOrder: ['dorsal', 'ventral'],
      },
      appearanceProjection: {
        kind: 'appearance-only',
        declaredControllerKind: creature.controller.kind,
        declaredControllerActionSize: creature.controller.actionSize,
        declaredControllerExecuted: false,
      },
    },
  }
}
