import type { WurmAnatomy } from '../creature/anatomy'
import type { EnvironmentConfig } from '../creature/types'
import {
  DomainActionPipeline,
  handoffDomainActionOwner,
  perturbLocomotionInputs,
  perturbPolicyObservation,
  type DomainActionOwner,
} from '../environment/domainRuntime'
import type { DomainSample } from '../environment/seedForge'
import type { LocomotionSensors } from '../policy/locomotionPolicy'
import { makeInitialAction, snapshotToObservation } from '../policy/simulationAdapter'
import { POLICY_TIMESTEP, type PolicyAction, type PolicyObservation } from '../policy/types'
import { replayRecorderFrameForScene, type ReplayRecorderFrame } from '../replay'
import {
  advanceStunt,
  createStuntState,
  gaitControllerOwnsBody,
  locomotionSensorsFor,
  makeTerrariumDecor,
  smoothAction,
  toSnapshot,
  type LocomotionState,
  type ShowcaseMode,
  type StuntState,
  type TerrariumDecor,
} from './terrariumSimulation'
import { createTerrainField, type TerrainField } from './terrainField'

export type EpisodeLocomotionController = {
  run(
    sensors: LocomotionSensors,
    segmentBends: ArrayLike<number>,
    segmentBendVelocities: ArrayLike<number>,
  ): PolicyAction
}

export type EpisodeMountedController = {
  run(observation: PolicyObservation): PolicyAction
}

export type TerrariumEpisodeOptions = {
  environmentConfig: EnvironmentConfig | null
  domainSample?: DomainSample | null
  anatomy: WurmAnatomy
  locomotionController: EpisodeLocomotionController
  mountedController: EpisodeMountedController
  field?: TerrainField
  state?: StuntState
  decor?: TerrariumDecor
}

export type TerrariumEpisodeStepOptions = {
  showcaseMode?: ShowcaseMode
  gaitTractionScale?: number
}

export type TerrariumEpisodeStep = {
  step: number
  previousLocomotionState: LocomotionState
  locomotionState: LocomotionState
  locomotionOwnedBefore: boolean
  locomotionOwnedAfter: boolean
  previousActionOwner: DomainActionOwner
  nextActionOwner: DomainActionOwner
  actionOwnerChanged: boolean
  replayFrame: ReplayRecorderFrame
}

type EpisodeRuntimeDomain = Pick<
  DomainSample,
  'seed' | 'actuatorStrength' | 'actuatorLatencyMs' | 'sensorNoise'
>

/**
 * The canonical fixed-step episode engine shared by the live exhibit and
 * headless Forge Trials.
 *
 * Rendering owns elapsed-frame accumulation only. Every policy observation,
 * controller handoff, forged-domain perturbation, plant step, and recorder
 * frame is resolved here on the exact 60 Hz policy grid.
 */
export class TerrariumEpisode {
  readonly field: TerrainField
  readonly state: StuntState
  readonly decor: TerrariumDecor
  readonly environmentConfig: EnvironmentConfig | null
  readonly domainSample: DomainSample | null
  readonly anatomy: WurmAnatomy

  private readonly locomotionController: EpisodeLocomotionController
  private readonly mountedController: EpisodeMountedController
  private readonly runtimeDomain: EpisodeRuntimeDomain
  private readonly actionPipeline: DomainActionPipeline
  private readonly mountedTarget = makeInitialAction()
  private readonly appliedAction = makeInitialAction()
  private simulationStep = 0

  constructor(options: TerrariumEpisodeOptions) {
    this.environmentConfig = options.environmentConfig
    this.domainSample = options.domainSample ?? null
    this.anatomy = options.anatomy
    this.locomotionController = options.locomotionController
    this.mountedController = options.mountedController
    this.runtimeDomain = this.domainSample ?? nominalRuntimeDomain(options.environmentConfig)
    this.actionPipeline = new DomainActionPipeline(this.runtimeDomain)
    this.field = options.field ?? createTerrainField(options.environmentConfig)
    this.state =
      options.state ??
      createStuntState(this.field, options.environmentConfig, options.anatomy, this.domainSample)
    this.decor =
      options.decor ??
      makeTerrariumDecor(
        options.environmentConfig?.seed ?? 1337,
        this.field,
        this.state.resources,
        options.environmentConfig?.terrain.obstacleDensity ?? 0.08,
      )
  }

  get stepIndex() {
    return this.simulationStep
  }

  step(options: TerrariumEpisodeStepOptions = {}): TerrariumEpisodeStep {
    const showcaseMode = options.showcaseMode ?? 'kickflip'
    const gaitTractionScale = options.gaitTractionScale ?? 1
    const previousLocomotionState = this.state.locomotionState
    const locomotionOwnedBefore = gaitControllerOwnsBody(this.state)

    if (previousLocomotionState === 'riding') {
      this.mountedTarget.set(
        this.mountedController.run(
          perturbPolicyObservation(
            snapshotToObservation(toSnapshot(this.state)),
            this.runtimeDomain,
            this.simulationStep,
          ),
        ),
      )
      smoothAction(this.appliedAction, this.mountedTarget, POLICY_TIMESTEP)
    } else if (locomotionOwnedBefore) {
      const noisyInputs = perturbLocomotionInputs(
        locomotionSensorsFor(this.state, this.field, gaitTractionScale),
        this.state.locomotionPlant.joints,
        this.state.locomotionPlant.jointVelocities,
        this.runtimeDomain,
        this.simulationStep,
      )
      this.appliedAction.set(
        this.locomotionController.run(
          noisyInputs.sensors,
          noisyInputs.segmentBends,
          noisyInputs.segmentBendVelocities,
        ),
      )
    } else {
      this.appliedAction.fill(0)
      this.actionPipeline.reset()
    }

    const previousActionOwner = actionOwnerFor(previousLocomotionState, locomotionOwnedBefore)
    const domainAction = this.actionPipeline.step(this.appliedAction, previousActionOwner)
    advanceStunt(
      this.state,
      domainAction,
      POLICY_TIMESTEP,
      Math.abs(this.environmentConfig?.world.gravity[1] ?? -9.81),
      showcaseMode,
      this.field,
      this.environmentConfig,
      this.decor.obstacles,
      this.anatomy,
      gaitTractionScale,
    )

    const locomotionOwnedAfter = gaitControllerOwnsBody(this.state)
    const nextActionOwner = actionOwnerFor(this.state.locomotionState, locomotionOwnedAfter)
    const actionOwnerChanged = handoffDomainActionOwner(
      previousActionOwner,
      nextActionOwner,
      this.actionPipeline,
      this.appliedAction,
      this.mountedTarget,
    )
    const replayFrame = replayRecorderFrameForScene(
      this.state,
      this.environmentConfig?.skateboard.discoveryRadius ?? 1.35,
    )
    const step = this.simulationStep
    this.simulationStep += 1

    return {
      step,
      previousLocomotionState,
      locomotionState: this.state.locomotionState,
      locomotionOwnedBefore,
      locomotionOwnedAfter,
      previousActionOwner,
      nextActionOwner,
      actionOwnerChanged,
      replayFrame,
    }
  }

  resetTransport() {
    this.actionPipeline.reset()
    this.mountedTarget.fill(0)
    this.appliedAction.fill(0)
    this.simulationStep = 0
  }
}

function actionOwnerFor(state: LocomotionState, locomotionOwnsBody: boolean): DomainActionOwner {
  if (state === 'riding') return 'mounted'
  return locomotionOwnsBody ? 'locomotion' : 'authored'
}

function nominalRuntimeDomain(environmentConfig: EnvironmentConfig | null): EpisodeRuntimeDomain {
  return {
    seed: environmentConfig?.seed ?? 1337,
    actuatorStrength: 1,
    actuatorLatencyMs: 0,
    sensorNoise: 0,
  }
}
