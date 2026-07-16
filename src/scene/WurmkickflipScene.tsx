import { OrbitControls, PerspectiveCamera, RoundedBox } from '@react-three/drei'
import { Canvas, useFrame } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import type { Group, Mesh, MeshStandardMaterial, Object3D } from 'three'
import { BufferAttribute, BufferGeometry, Color, MathUtils, Quaternion, Vector3 } from 'three'
import { deriveWurmAnatomy, type GenomeAppendage, type WurmAnatomy } from '../creature/anatomy'
import type { CreatureGenome, EnvironmentConfig, Vec3 } from '../creature/types'
import { LocomotionPolicyRunner } from '../policy/locomotionRunner'
import { PolicyRunner } from '../policy/policyRunner'
import { makeInitialAction, snapshotToObservation } from '../policy/simulationAdapter'
import {
  POLICY_TIMESTEP,
  SEGMENT_COUNT,
  type ActiveGaitExperiment,
  type GaitExperimentCommand,
  type GaitTelemetry,
  type PolicyAction,
  type PolicyStatus,
  type ViewerMetrics,
} from '../policy/types'
import {
  projectReplaySample,
  replayRecorderFrameForScene,
  type ReplayPlaybackSample,
  type ReplayRecorderFrame,
} from '../replay'
import { readNeedsMetrics, readResourceContents, type TerrariumResource } from './terrariumNeeds'
import { createTerrainField, TERRAIN_GRID_RESOLUTION, type TerrainField } from './terrainField'
import {
  advanceStunt,
  applyGaitLateralShove,
  boardGroundY,
  createStuntState,
  dampAngle,
  decodeAction,
  gaitControllerOwnsBody,
  gaitExperimentLifecycleNotice,
  interactionSampleFor,
  lerpAngle,
  locomotionSensorsFor,
  makeTerrariumDecor,
  scriptedGaitMusclesAnteriorToPosterior,
  smoothAction,
  smoothStep,
  stuntNameFor,
  toSnapshot,
  type ShowcaseMode,
  type StuntState,
  type TerrariumDecor,
} from './terrariumSimulation'

export * from './terrariumSimulation'

type SceneProps = {
  policyRunner: PolicyRunner
  running: boolean
  resetNonce: number
  interactionNonce?: number
  showcaseMode?: ShowcaseMode
  creature: CreatureGenome | null
  environmentConfig: EnvironmentConfig | null
  onMetrics: (metrics: ViewerMetrics) => void
  onGaitTelemetry?: (gait: GaitTelemetry) => void
  onPolicyStatus: (status: PolicyStatus) => void
  onReplayFrame?: (frame: ReplayRecorderFrame) => void
  replaySample?: ReplayPlaybackSample | null
  gaitExperiment?: GaitExperimentCommand | null
  gaitTractionScale?: number
}

const up = new Vector3(0, 1, 0)
const connectorQuaternion = new Quaternion()
const connectorDelta = new Vector3()
const connectorStart = new Vector3()
const connectorEnd = new Vector3()
const renderTarget = new Vector3()
const EXPERIMENT_ENDED_NOTICE = 'Experiment ended because authored motion took control.'

export function WurmkickflipScene({
  policyRunner,
  running,
  resetNonce,
  interactionNonce = 0,
  showcaseMode = 'kickflip',
  creature,
  environmentConfig,
  onMetrics,
  onGaitTelemetry,
  onPolicyStatus,
  onReplayFrame,
  replaySample = null,
  gaitExperiment = null,
  gaitTractionScale = 1,
}: SceneProps) {
  const locomotionRunner = useMemo(() => new LocomotionPolicyRunner(), [])
  useEffect(() => {
    let mounted = true
    void Promise.all([policyRunner.load(), locomotionRunner.load()]).then(
      ([stuntStatus, locomotionStatus]) => {
        if (!mounted) return
        onPolicyStatus({
          backend: locomotionStatus.loaded ? 'neural-js' : 'unavailable',
          message:
            `${locomotionStatus.message} Detached segment motion is the evolved controller driving the causal plant. ` +
            `Neural obstacle sensing and segment actuation drive traversal; collision projection is plant physics. ` +
            `Skateboard approach and mounting are neural, contact-driven motion. ` +
            `Feeding, dismounting, and the aerial kickflip are authored environment choreography. ` +
            `Mounted stunt rig: ${stuntStatus.modelVersion}.`,
          modelVersion: locomotionStatus.loaded
            ? `${locomotionStatus.modelVersion} + scripted-kickflip`
            : `${locomotionStatus.modelVersion} + mounted-${stuntStatus.modelVersion}`,
        })
      },
    )
    return () => {
      mounted = false
    }
  }, [locomotionRunner, onPolicyStatus, policyRunner])

  const sceneKey = `${creature?.id ?? 'wurm'}-${environmentConfig?.id ?? 'terrarium'}-${resetNonce}`
  const arenaSpan = Math.max(
    environmentConfig?.world.size[0] ?? 11.5,
    environmentConfig?.world.size[2] ?? 11.5,
  )
  const cameraDistance = arenaSpan * 0.69

  return (
    <Canvas dpr={[1, 1.5]} shadows="percentage">
      <PerspectiveCamera makeDefault position={[cameraDistance, arenaSpan * 0.52, cameraDistance]} fov={45} />
      <color attach="background" args={['#dce9df']} />
      <fog attach="fog" args={['#dce9df', arenaSpan * 0.85, arenaSpan * 1.75]} />
      <hemisphereLight color="#fffaf0" groundColor="#4e7660" intensity={1.25} />
      <directionalLight
        castShadow
        color="#fff4d6"
        intensity={2.55}
        position={[-arenaSpan * 0.38, arenaSpan * 0.78, arenaSpan * 0.42]}
        shadow-camera-far={arenaSpan * 1.7}
        shadow-camera-left={-arenaSpan * 0.62}
        shadow-camera-right={arenaSpan * 0.62}
        shadow-camera-top={arenaSpan * 0.62}
        shadow-camera-bottom={-arenaSpan * 0.18}
        shadow-mapSize={[1536, 1536]}
      />
      <TerrariumWorld
        creature={creature}
        environmentConfig={environmentConfig}
        gaitExperiment={gaitExperiment}
        gaitTractionScale={gaitTractionScale}
        interactionNonce={interactionNonce}
        key={sceneKey}
        onMetrics={onMetrics}
        onGaitTelemetry={onGaitTelemetry}
        onReplayFrame={onReplayFrame}
        policyRunner={policyRunner}
        replaySample={replaySample}
        locomotionRunner={locomotionRunner}
        running={running}
        showcaseMode={showcaseMode}
      />
      <SceneIntegrityProbe />
      <OrbitControls
        enableDamping
        enablePan={false}
        maxDistance={arenaSpan * 1.55}
        maxPolarAngle={Math.PI / 2.08}
        minDistance={4.2}
        minPolarAngle={0.42}
        target={[0, 0.6, 0]}
      />
    </Canvas>
  )
}

type TerrariumWorldProps = {
  policyRunner: PolicyRunner
  locomotionRunner: LocomotionPolicyRunner
  running: boolean
  interactionNonce: number
  showcaseMode: ShowcaseMode
  creature: CreatureGenome | null
  environmentConfig: EnvironmentConfig | null
  onMetrics: (metrics: ViewerMetrics) => void
  onGaitTelemetry?: (gait: GaitTelemetry) => void
  onReplayFrame?: (frame: ReplayRecorderFrame) => void
  replaySample: ReplayPlaybackSample | null
  gaitExperiment: GaitExperimentCommand | null
  gaitTractionScale: number
}

function TerrariumWorld({
  policyRunner,
  locomotionRunner,
  running,
  interactionNonce,
  showcaseMode,
  creature,
  environmentConfig,
  gaitExperiment,
  gaitTractionScale,
  onMetrics,
  onGaitTelemetry,
  onReplayFrame,
  replaySample,
}: TerrariumWorldProps) {
  const terrainField = useMemo(() => createTerrainField(environmentConfig), [environmentConfig])
  const anatomy = useMemo(() => deriveWurmAnatomy(creature), [creature])
  const state = useRef(createStuntState(terrainField, environmentConfig, anatomy))
  const latestAction = useRef<PolicyAction>(makeInitialAction())
  const appliedAction = useRef<PolicyAction>(makeInitialAction())
  const inferencePending = useRef(false)
  const physicsAccumulator = useRef(0)
  const inferenceAccumulator = useRef(POLICY_TIMESTEP)
  const metricsAccumulator = useRef(0)
  const lastInteractionNonce = useRef(interactionNonce)
  // A remount can race a command emitted from the previous scene. Start
  // unacknowledged so the new body either applies it or reports why it could not.
  const lastGaitExperimentSequence = useRef(-1)
  const bodyExperiment = useRef<ActiveGaitExperiment | null>(null)
  const experimentNotice = useRef<string | null>(null)
  const boardRef = useRef<Group>(null)
  const wheelRefs = useRef<Array<Mesh | null>>([])
  const segmentRefs = useRef<Array<Group | null>>([])
  const segmentMaterialRefs = useRef<Array<MeshStandardMaterial | null>>([])
  const connectorRefs = useRef<Array<Mesh | null>>([])
  const burstRefs = useRef<Array<Mesh | null>>([])
  const mouthRef = useRef<Mesh>(null)

  useEffect(() => {
    locomotionRunner.reset()
    bodyExperiment.current = null
    experimentNotice.current = null
  }, [locomotionRunner])

  useEffect(() => {
    let cancelled = false
    void locomotionRunner.load().then(() => {
      if (cancelled) return
      onGaitTelemetry?.(
        gaitTelemetryFor(
          state.current,
          locomotionRunner,
          terrainField,
          gaitTractionScale,
          bodyExperiment.current,
          experimentNotice.current,
        ),
      )
    })
    return () => {
      cancelled = true
    }
  }, [gaitTractionScale, locomotionRunner, onGaitTelemetry, terrainField])

  const gravity = Math.abs(environmentConfig?.world.gravity[1] ?? -9.81)
  const palette = useMemo(() => makeWurmPalette(creature), [creature])
  const environmentSeed = environmentConfig?.seed ?? 1337
  const obstacleDensity = environmentConfig?.terrain.obstacleDensity ?? 0.08
  const terrain = useMemo(
    () => makeTerrariumDecor(environmentSeed, terrainField, state.current.resources, obstacleDensity),
    [environmentSeed, obstacleDensity, terrainField],
  )

  useEffect(() => {
    if (lastInteractionNonce.current !== interactionNonce) {
      state.current.poke = 1
      lastInteractionNonce.current = interactionNonce
    }
  }, [interactionNonce])

  useEffect(() => {
    if (!gaitExperiment || lastGaitExperimentSequence.current === gaitExperiment.sequence) return
    lastGaitExperimentSequence.current = gaitExperiment.sequence
    if (gaitExperiment.kind === 'clear') {
      locomotionRunner.clearPerturbation()
      bodyExperiment.current = null
      experimentNotice.current = null
      onGaitTelemetry?.(
        gaitTelemetryFor(state.current, locomotionRunner, terrainField, gaitTractionScale, null, null),
      )
      return
    }
    if (gaitExperiment.kind === 'lateral-shove') {
      locomotionRunner.clearPerturbation()
      const accepted = applyGaitLateralShove(state.current, gaitExperiment.impulse)
      bodyExperiment.current = accepted
        ? {
            kind: 'lateral-shove',
            segment: null,
            remainingSeconds: gaitExperiment.durationSeconds,
          }
        : null
      experimentNotice.current = accepted ? null : gaitExperimentLifecycleNotice(state.current)
      onGaitTelemetry?.(
        gaitTelemetryFor(
          state.current,
          locomotionRunner,
          terrainField,
          gaitTractionScale,
          bodyExperiment.current,
          experimentNotice.current,
        ),
      )
      return
    }
    if (!gaitControllerOwnsBody(state.current)) {
      locomotionRunner.clearPerturbation()
      bodyExperiment.current = null
      experimentNotice.current = gaitExperimentLifecycleNotice(state.current)
      onGaitTelemetry?.(
        gaitTelemetryFor(
          state.current,
          locomotionRunner,
          terrainField,
          gaitTractionScale,
          null,
          experimentNotice.current,
        ),
      )
      return
    }
    bodyExperiment.current = null
    experimentNotice.current = null
    locomotionRunner.applyPerturbation(
      gaitExperiment.kind,
      gaitExperiment.kind === 'numb-neuron' ? gaitExperiment.segment : null,
      gaitExperiment.durationSeconds,
    )
    onGaitTelemetry?.(
      gaitTelemetryFor(
        state.current,
        locomotionRunner,
        terrainField,
        gaitTractionScale,
        bodyExperiment.current,
        experimentNotice.current,
      ),
    )
  }, [gaitExperiment, gaitTractionScale, locomotionRunner, onGaitTelemetry, terrainField])

  useFrame((_, rawDelta) => {
    const frameDelta = Math.min(rawDelta, 0.08)
    if (replaySample) {
      const current = state.current
      const replay = projectReplaySample(replaySample)
      current.time = replay.time
      current.boardX = replay.board.x
      current.boardY = replay.board.y
      current.boardZ = replay.board.z
      current.boardVx = replay.board.vx
      current.boardVy = replay.board.vy
      current.boardVz = replay.board.vz
      current.boardRoll = replay.board.roll
      current.boardYaw = replay.board.yaw
      current.boardPitch = replay.board.pitch
      current.boardHeading = replay.board.heading
      current.boardSpeed = replay.board.speed
      current.wheelSpin = replay.board.wheelSpin
      current.wormX = replay.creatureRoot.x
      current.wormY = replay.creatureRoot.y
      current.wormZ = replay.creatureRoot.z
      current.wormVx = replay.creatureRoot.vx
      current.wormVz = replay.creatureRoot.vz
      current.wormHeading = replay.creatureRoot.heading
      current.contactRatio = replay.contactRatio
      current.reward = replay.reward
      current.previousAction = replay.action
      current.segments = replay.segments
      // Schema v1 stores root + action channels, not lifecycle state or every
      // segment pose. The replay adapter reconstructs a detached body so those
      // recorded values own the visible result.
      current.locomotionState = 'crawling'
      current.mountBlend = 0
      current.grounded = replay.contactRatio >= 0.5
      current.phase = 'free crawl'
      physicsAccumulator.current = 0
      inferenceAccumulator.current = 0
    } else if (running) {
      physicsAccumulator.current += frameDelta
      inferenceAccumulator.current += frameDelta
      metricsAccumulator.current += frameDelta

      let steps = 0
      while (physicsAccumulator.current >= POLICY_TIMESTEP && steps < 5) {
        const wasRiding = state.current.locomotionState === 'riding'
        const locomotionOwnsBody = gaitControllerOwnsBody(state.current)
        if (!locomotionOwnsBody) {
          locomotionRunner.clearPerturbation()
          bodyExperiment.current = null
        }
        if (state.current.locomotionState === 'riding') {
          smoothAction(appliedAction.current, latestAction.current, POLICY_TIMESTEP)
        } else if (locomotionOwnsBody) {
          appliedAction.current.set(
            locomotionRunner.run(
              locomotionSensorsFor(state.current, terrainField, gaitTractionScale),
              state.current.locomotionPlant.joints,
              state.current.locomotionPlant.jointVelocities,
            ),
          )
        } else {
          appliedAction.current.fill(0)
        }
        advanceStunt(
          state.current,
          appliedAction.current,
          POLICY_TIMESTEP,
          gravity,
          showcaseMode,
          terrainField,
          environmentConfig,
          terrain.obstacles,
          anatomy,
          gaitTractionScale,
        )
        const controllerStillOwnsBody = gaitControllerOwnsBody(state.current)
        if (!controllerStillOwnsBody) {
          if (locomotionOwnsBody && (bodyExperiment.current || locomotionRunner.hasActivePerturbation())) {
            experimentNotice.current = EXPERIMENT_ENDED_NOTICE
          }
          locomotionRunner.clearPerturbation()
          bodyExperiment.current = null
        } else if (bodyExperiment.current) {
          bodyExperiment.current.remainingSeconds -= POLICY_TIMESTEP
          if (bodyExperiment.current.remainingSeconds <= 0) bodyExperiment.current = null
        }
        onReplayFrame?.(
          replayRecorderFrameForScene(state.current, environmentConfig?.skateboard.discoveryRadius ?? 1.35),
        )
        if (wasRiding && state.current.locomotionState !== 'riding') {
          locomotionRunner.reset()
          appliedAction.current.fill(0)
        }
        physicsAccumulator.current -= POLICY_TIMESTEP
        steps += 1
      }

      if (
        state.current.locomotionState === 'riding' &&
        inferenceAccumulator.current >= POLICY_TIMESTEP &&
        !inferencePending.current
      ) {
        inferenceAccumulator.current %= POLICY_TIMESTEP
        inferencePending.current = true
        const observation = snapshotToObservation(toSnapshot(state.current))
        void policyRunner
          .run(observation)
          .then(action => {
            latestAction.current = action
          })
          .catch(() => {
            // Keep the last known-safe action; PolicyRunner reports backend failures in its status.
          })
          .finally(() => {
            inferencePending.current = false
          })
      }

      if (metricsAccumulator.current >= 0.1) {
        metricsAccumulator.current %= 0.1
        const status = policyRunner.getStatus()
        const current = state.current
        const needsMetrics = readNeedsMetrics(current.needs, current.resources, [
          current.wormX,
          current.wormY,
          current.wormZ,
        ])
        const interactionResource = current.feedingResourceId
          ? current.resources.find(resource => resource.id === current.feedingResourceId)
          : null
        const displayedNeedTarget = interactionResource?.id ?? needsMetrics.targetResourceId
        const displayedNeedDistance = interactionResource
          ? Math.hypot(
              interactionResource.position[0] - current.wormX,
              interactionResource.position[2] - current.wormZ,
            )
          : needsMetrics.targetDistance
        const rootVx = current.mountBlend > 0.5 ? current.boardVx : current.wormVx
        const rootVz = current.mountBlend > 0.5 ? current.boardVz : current.wormVz
        const bodySpeed = current.segments.reduce(
          (maximum, segment) =>
            Math.max(maximum, Math.hypot(segment.vx - rootVx, segment.vy, segment.vz - rootVz)),
          0,
        )
        const gait = gaitTelemetryFor(
          current,
          locomotionRunner,
          terrainField,
          gaitTractionScale,
          bodyExperiment.current,
          experimentNotice.current,
        )
        onMetrics({
          time: current.time,
          reward: current.reward,
          distance: current.distance,
          contactRatio: current.contactRatio,
          backend: status.backend,
          message: status.message,
          running,
          muscleActivity: decodeAction(current.previousAction).bends,
          phase: current.phase,
          flipProgress: current.flipProgress,
          flipsLanded: current.flipsLanded,
          airtime: current.grounded ? current.lastAirtime : current.currentAirtime,
          height:
            current.locomotionState === 'riding'
              ? Math.max(0, current.boardY - boardGroundY(terrainField, current.boardX, current.boardZ))
              : Math.max(0, current.wormY - terrainField.sample(current.wormX, current.wormZ).height),
          speed:
            current.locomotionState === 'riding'
              ? Math.hypot(current.boardVx, current.boardVz)
              : Math.hypot(current.wormVx, current.wormVz),
          landingQuality: current.landingQuality,
          attempt: current.attempt,
          stuntName: stuntNameFor(current, showcaseMode),
          bodySpeed,
          mounted: current.mountBlend > 0.92,
          distanceToBoard: current.distanceToBoard,
          terrainFriction: current.terrainFriction,
          crawlDistance: current.wormDistance,
          hunger: needsMetrics.hunger,
          thirst: needsMetrics.thirst,
          wellbeing: needsMetrics.wellbeing,
          activeNeed: interactionResource?.need ?? needsMetrics.activeNeed,
          needTarget: displayedNeedTarget,
          needTargetDistance: displayedNeedDistance,
          gait,
        })
      }
    }

    renderStunt(
      state.current,
      state.current.previousAction,
      frameDelta,
      showcaseMode,
      anatomy,
      boardRef.current,
      wheelRefs.current,
      segmentRefs.current,
      segmentMaterialRefs.current,
      connectorRefs.current,
      burstRefs.current,
      mouthRef.current,
    )
  })

  return (
    <group>
      <Terrarium terrain={terrain} environmentConfig={environmentConfig} field={terrainField} />
      <TerrariumResources resources={state.current.resources} stateRef={state} />
      <BoardVisual boardRef={boardRef} environmentConfig={environmentConfig} wheelRefs={wheelRefs} />
      <WurmVisual
        anatomy={anatomy}
        connectorRefs={connectorRefs}
        materialRefs={segmentMaterialRefs}
        palette={palette}
        segmentRefs={segmentRefs}
        mouthRef={mouthRef}
      />
      <LandingBurst burstRefs={burstRefs} palette={palette} />
    </group>
  )
}

function gaitTelemetryFor(
  state: StuntState,
  runner: LocomotionPolicyRunner,
  field: TerrainField,
  tractionScale: number,
  bodyExperiment: ActiveGaitExperiment | null,
  experimentNotice: string | null,
): GaitTelemetry {
  const controllerActive = gaitControllerOwnsBody(state)
  const appliedNeuralAction = state.previousActionApplication === 'neural'
  const lifecycleHandoff = state.previousActionApplication === 'lifecycle-handoff'
  const runnerStatus = runner.getStatus()
  const policy = runner.getTelemetry()
  const forwardX = Math.cos(state.wormHeading)
  const forwardZ = Math.sin(state.wormHeading)
  const rightX = -forwardZ
  const rightZ = forwardX
  const sensors = locomotionSensorsFor(state, field, tractionScale)
  const appliedMuscles = lifecycleHandoff
    ? Array.from({ length: SEGMENT_COUNT }, () => 0)
    : appliedNeuralAction
      ? decodeAction(state.previousAction).bends
      : scriptedGaitMusclesAnteriorToPosterior(state.previousAction)
  const activeExperiment = policy.activePerturbation ?? bodyExperiment
  return {
    controllerActive: controllerActive && runnerStatus.loaded,
    source: lifecycleHandoff
      ? 'lifecycle-handoff'
      : appliedNeuralAction
        ? runnerStatus.loaded
          ? 'live-neural'
          : 'unavailable'
        : 'scripted-stunt',
    segmentOrder: 'anterior-to-posterior',
    tractionScale: MathUtils.clamp(tractionScale, 0, 1.5),
    tractionAvailable: true,
    bodyForwardSpeed: state.wormVx * forwardX + state.wormVz * forwardZ,
    bodyLateralSpeed: state.wormVx * rightX + state.wormVz * rightZ,
    targetAlignment: sensors.targetForward,
    targetAlignmentAvailable: controllerActive && appliedNeuralAction && runnerStatus.loaded,
    activeExperiment: activeExperiment ? { ...activeExperiment } : null,
    experimentNotice,
    segments: Array.from({ length: SEGMENT_COUNT }, (_, segment) => ({
      segment,
      neuralActivation: policy.hidden[segment] ?? 0,
      neuralDrive: policy.drives[segment] ?? 0,
      muscleCommand: appliedMuscles[segment] ?? 0,
      requestedMuscleCommand: appliedNeuralAction
        ? (policy.requestedCommands[segment] ?? 0)
        : (appliedMuscles[segment] ?? 0),
      jointBend: state.locomotionPlant.joints[segment] ?? 0,
      jointVelocity: state.locomotionPlant.jointVelocities[segment] ?? 0,
      afferentJointBend: policy.sensedBends[segment] ?? 0,
      afferentJointVelocity: policy.sensedBendVelocities[segment] ?? 0,
      afferentContactLoad: policy.sensedContactLoads[segment] ?? 0,
      afferentSlipSpeed: policy.sensedSlipSpeeds[segment] ?? 0,
      afferentObstacleForward: policy.sensedObstacleForward[segment] ?? 0,
      afferentObstacleRight: policy.sensedObstacleRight[segment] ?? 0,
      contactLoad: state.locomotionPlant.contactLoads[segment] ?? 0,
      slipSpeed: state.locomotionPlant.slipSpeeds[segment] ?? 0,
      obstacleForward: state.locomotionPlant.obstacleForward[segment] ?? 0,
      obstacleRight: state.locomotionPlant.obstacleRight[segment] ?? 0,
    })),
  }
}

function renderStunt(
  state: StuntState,
  action: PolicyAction,
  delta: number,
  mode: ShowcaseMode,
  anatomy: WurmAnatomy,
  board: Group | null,
  wheels: Array<Mesh | null>,
  segments: Array<Group | null>,
  materials: Array<MeshStandardMaterial | null>,
  connectors: Array<Mesh | null>,
  bursts: Array<Mesh | null>,
  mouth: Mesh | null,
) {
  const renderAlpha = 1 - Math.exp(-22 * delta)
  if (board) {
    renderTarget.set(state.boardX, state.boardY, state.boardZ)
    board.position.lerp(renderTarget, renderAlpha)
    board.rotation.x = MathUtils.damp(board.rotation.x, state.boardRoll, 24, delta)
    board.rotation.y = dampAngle(board.rotation.y, state.boardYaw, 20, delta)
    board.rotation.z = MathUtils.damp(board.rotation.z, state.boardPitch, 20, delta)
  }
  for (const wheel of wheels) {
    if (wheel) wheel.rotation.y = state.wheelSpin
  }

  const decoded = decodeAction(action)
  const mount = smoothStep(state.mountBlend)
  const anchorX = MathUtils.lerp(state.wormX, state.boardX, mount)
  const anchorZ = MathUtils.lerp(state.wormZ, state.boardZ, mount)
  const heading = lerpAngle(state.wormHeading, state.boardHeading, mount)
  const forwardX = Math.cos(heading)
  const forwardZ = Math.sin(heading)
  const rightX = -forwardZ
  const rightZ = forwardX
  const plantOwnsBody = gaitControllerOwnsBody(state)
  const interaction =
    state.locomotionState === 'dismounting' || state.locomotionState === 'feeding'
      ? interactionSampleFor(state)
      : null
  segments.forEach((segmentGroup, index) => {
    if (!segmentGroup) return
    const pose = state.segments[index]
    const dx = pose.x - anchorX
    const dz = pose.z - anchorZ
    const axial = dx * forwardX + dz * forwardZ
    const lateral = dx * rightX + dz * rightZ
    renderTarget.set(
      anchorX +
        forwardX * axial * (plantOwnsBody ? 1 : anatomy.visualLengthScale) +
        rightX * lateral * (plantOwnsBody ? 1 : anatomy.motionWidthScale),
      pose.y + (plantOwnsBody ? 0 : (anatomy.verticalScale - 1) * 0.025),
      anchorZ +
        forwardZ * axial * (plantOwnsBody ? 1 : anatomy.visualLengthScale) +
        rightZ * lateral * (plantOwnsBody ? 1 : anatomy.motionWidthScale),
    )
    segmentGroup.position.lerp(renderTarget, renderAlpha)
    segmentGroup.rotation.x = MathUtils.damp(segmentGroup.rotation.x, 0, 20, delta)
    segmentGroup.rotation.y = dampAngle(segmentGroup.rotation.y, -pose.yaw, 18, delta)
    segmentGroup.rotation.z = MathUtils.damp(segmentGroup.rotation.z, Math.PI / 2 + pose.pitch, 18, delta)
    const muscleIndex = plantOwnsBody ? SEGMENT_COUNT - 1 - index : index
    const squeeze = 1 - Math.abs(decoded.contractions[muscleIndex] ?? 0) * 0.12
    const taper = 0.74 + Math.sin((index / (SEGMENT_COUNT - 1)) * Math.PI) * 0.28
    const interactionScale = interaction?.segments[index]
    const radialScale = interactionScale?.radialScale ?? 1
    const axialScale = interactionScale?.axialScale ?? 1
    segmentGroup.scale.x = MathUtils.damp(
      segmentGroup.scale.x,
      taper * squeeze * radialScale * anatomy.thicknessScale,
      16,
      delta,
    )
    segmentGroup.scale.y = MathUtils.damp(
      segmentGroup.scale.y,
      taper * (2 - squeeze) * axialScale * anatomy.axialScale,
      16,
      delta,
    )
    segmentGroup.scale.z = MathUtils.damp(
      segmentGroup.scale.z,
      taper * radialScale * anatomy.thicknessScale,
      16,
      delta,
    )
    const material = materials[index]
    if (material) {
      material.emissiveIntensity = 0.06 + Math.abs(decoded.bends[muscleIndex] ?? 0) * 0.28
    }
  })

  if (mouth) {
    const mouthOpen = interaction?.contact.mouthOpen ?? 0
    mouth.scale.x = MathUtils.damp(mouth.scale.x, 0.52, 18, delta)
    mouth.scale.y = MathUtils.damp(mouth.scale.y, 0.16 + mouthOpen * 0.84, 18, delta)
    mouth.scale.z = MathUtils.damp(mouth.scale.z, 0.72 + mouthOpen * 0.18, 18, delta)
  }

  connectors.forEach((connector, index) => {
    if (!connector) return
    const startGroup = segments[index]
    const endGroup = segments[index + 1]
    if (!startGroup || !endGroup) return
    connectorStart.copy(startGroup.position)
    connectorEnd.copy(endGroup.position)
    connectorDelta.copy(connectorEnd).sub(connectorStart)
    const length = connectorDelta.length()
    renderTarget.copy(connectorStart).add(connectorEnd).multiplyScalar(0.5)
    connector.position.lerp(renderTarget, renderAlpha)
    connector.scale.x = MathUtils.damp(connector.scale.x, anatomy.connectorScale, 18, delta)
    connector.scale.y = MathUtils.damp(connector.scale.y, length, 18, delta)
    connector.scale.z = MathUtils.damp(connector.scale.z, anatomy.connectorScale, 18, delta)
    connectorQuaternion.setFromUnitVectors(up, connectorDelta.normalize())
    connector.quaternion.slerp(connectorQuaternion, renderAlpha)
  })

  bursts.forEach((burst, index) => {
    if (!burst) return
    const strength = state.landingFlash
    const angle = (index / bursts.length) * Math.PI * 2
    const radius = (1 - strength) * (0.28 + (index % 3) * 0.16)
    burst.visible = strength > 0.02 && mode === 'kickflip'
    burst.position.set(
      state.boardX + Math.cos(angle) * radius,
      state.boardY + 0.08 + Math.sin(angle * 2) * 0.08 + strength * 0.18,
      state.boardZ + Math.sin(angle) * radius * 0.65,
    )
    const scale = Math.max(0.001, strength * (0.5 + (index % 2) * 0.35))
    burst.scale.setScalar(scale)
    burst.rotation.x += delta * 4
  })
}

function TerrariumResources({
  resources,
  stateRef,
}: {
  resources: readonly TerrariumResource[]
  stateRef: React.RefObject<StuntState>
}) {
  return (
    <group>
      {resources
        .filter(resource => resource.presentation === 'bowl')
        .map(resource => (
          <ResourceBowl key={resource.id} resource={resource} stateRef={stateRef} />
        ))}
    </group>
  )
}

function ResourceBowl({
  resource,
  stateRef,
}: {
  resource: TerrariumResource
  stateRef: React.RefObject<StuntState>
}) {
  const { radius, height, vesselColor, contentsColor } = resource.appearance
  const food = resource.id === 'food-bowl'
  const contentsRef = useRef<Group>(null)
  const contentsSurfaceRef = useRef<Mesh>(null)
  const rippleRefs = useRef<Array<Mesh | null>>([])
  useFrame((_, delta) => {
    const state = stateRef.current
    const active = state?.locomotionState === 'feeding' && state.feedingResourceId === resource.id
    const interaction = active && state ? interactionSampleFor(state) : null
    const headContact = interaction?.contact.headWeight ?? 0
    const mouth = interaction?.contact.mouthOpen ?? 0
    const swallow = interaction?.contact.swallow ?? 0
    const contents = state ? readResourceContents(state.needs, resource) : null
    const fillRatio = contents?.fillRatio ?? 0
    const radialFill = Math.sqrt(fillRatio)
    if (contentsRef.current) {
      contentsRef.current.visible = fillRatio > 0.001
      contentsRef.current.position.y = MathUtils.damp(
        contentsRef.current.position.y,
        -(1 - fillRatio) * height * 0.28 + mouth * 0.024,
        14,
        delta,
      )
      contentsRef.current.rotation.y = dampAngle(
        contentsRef.current.rotation.y,
        food ? (interaction?.root.yaw ?? 0) : 0,
        12,
        delta,
      )
      const contentsScale = 1 + headContact * (food ? 0.035 : 0.018)
      const targetScale = Math.max(0.001, radialFill * contentsScale)
      contentsRef.current.scale.setScalar(MathUtils.damp(contentsRef.current.scale.x, targetScale, 12, delta))
    }
    if (contentsSurfaceRef.current && contents) {
      contentsSurfaceRef.current.visible = fillRatio > 0.001
      contentsSurfaceRef.current.position.y = MathUtils.damp(
        contentsSurfaceRef.current.position.y,
        contents.surfaceY - resource.groundHeight,
        12,
        delta,
      )
      const surfaceScale = Math.max(0.001, radialFill)
      contentsSurfaceRef.current.scale.x = MathUtils.damp(
        contentsSurfaceRef.current.scale.x,
        surfaceScale,
        12,
        delta,
      )
      contentsSurfaceRef.current.scale.z = MathUtils.damp(
        contentsSurfaceRef.current.scale.z,
        surfaceScale,
        12,
        delta,
      )
    }
    rippleRefs.current.forEach((ripple, index) => {
      if (!ripple) return
      ripple.visible = !food && fillRatio > 0.001 && headContact > 0.025
      const rippleScale = 0.25 + headContact * (0.85 + index * 0.4) + swallow * 0.28
      ripple.scale.setScalar(MathUtils.damp(ripple.scale.x, rippleScale, 16, delta))
      ripple.position.y =
        (contents?.surfaceY ?? resource.groundHeight) - resource.groundHeight + index * 0.008
    })
  })
  return (
    <group position={[resource.position[0], resource.groundHeight, resource.position[2]]}>
      <mesh position={[0, 0.018, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[radius * 1.38, 0.025, 7, 32]} />
        <meshStandardMaterial
          color={contentsColor}
          emissive={contentsColor}
          emissiveIntensity={0.42}
          opacity={0.62}
          transparent
        />
      </mesh>
      <mesh castShadow position={[0, height * 0.48, 0]}>
        <cylinderGeometry args={[radius, radius * 0.78, height, 28]} />
        <meshStandardMaterial color={vesselColor} roughness={0.68} />
      </mesh>
      <mesh position={[0, height * 0.92, 0]} ref={contentsSurfaceRef}>
        <cylinderGeometry args={[radius * 0.76, radius * 0.76, 0.035, 28]} />
        <meshStandardMaterial
          color={contentsColor}
          emissive={contentsColor}
          emissiveIntensity={food ? 0.05 : 0.14}
          metalness={food ? 0 : 0.12}
          opacity={food ? 1 : 0.82}
          roughness={food ? 0.82 : 0.18}
          transparent={!food}
        />
      </mesh>
      <mesh position={[0, height * 0.92, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[radius * 0.83, radius * 0.075, 8, 28]} />
        <meshStandardMaterial color={vesselColor} roughness={0.58} />
      </mesh>
      <mesh castShadow position={[radius * 0.92, 0.24, 0]}>
        <cylinderGeometry args={[0.018, 0.025, 0.38, 8]} />
        <meshStandardMaterial color={vesselColor} roughness={0.72} />
      </mesh>
      <mesh castShadow position={[radius * 0.92, 0.46, 0]} rotation={[0, 0, Math.PI / 4]}>
        <boxGeometry args={[0.14, 0.14, 0.055]} />
        <meshStandardMaterial
          color={contentsColor}
          emissive={contentsColor}
          emissiveIntensity={0.25}
          roughness={0.4}
        />
      </mesh>
      <group ref={contentsRef}>
        {food
          ? Array.from({ length: 9 }, (_, index) => {
              const angle = index * 2.399
              const pelletRadius = radius * (0.18 + (index % 3) * 0.13)
              return (
                <mesh
                  castShadow
                  key={index}
                  position={[Math.cos(angle) * pelletRadius, height * 1.02, Math.sin(angle) * pelletRadius]}
                  scale={0.72 + (index % 2) * 0.2}
                >
                  <dodecahedronGeometry args={[0.055, 0]} />
                  <meshStandardMaterial color={index % 2 === 0 ? '#d59a43' : '#b87531'} roughness={0.9} />
                </mesh>
              )
            })
          : null}
      </group>
      {!food
        ? [0, 1].map(index => (
            <mesh
              key={`ripple-${index}`}
              position={[0, height * 0.91, 0]}
              ref={value => {
                rippleRefs.current[index] = value
              }}
              rotation={[Math.PI / 2, 0, 0]}
              visible={false}
            >
              <torusGeometry args={[radius * 0.46, 0.012, 6, 30]} />
              <meshStandardMaterial
                color="#b9edff"
                emissive="#59b9dc"
                emissiveIntensity={0.55}
                transparent
                opacity={0.68}
              />
            </mesh>
          ))
        : null}
    </group>
  )
}

function BoardVisual({
  boardRef,
  wheelRefs,
  environmentConfig,
}: {
  boardRef: React.RefObject<Group | null>
  wheelRefs: React.MutableRefObject<Array<Mesh | null>>
  environmentConfig: EnvironmentConfig | null
}) {
  const configured = environmentConfig?.skateboard.deckSize ?? [1.9, 0.12, 0.58]
  const deck: Vec3 = [
    MathUtils.clamp(configured[0], 1.65, 2.1),
    MathUtils.clamp(configured[1], 0.09, 0.15),
    MathUtils.clamp(configured[2], 0.52, 0.68),
  ]
  const wheelRadius = MathUtils.clamp(environmentConfig?.skateboard.wheelRadius ?? 0.12, 0.1, 0.15)
  const wheelX = deck[0] * 0.34
  const wheelZ = deck[2] * 0.68
  const wheelPositions = [
    [-wheelX, -0.15, -wheelZ],
    [-wheelX, -0.15, wheelZ],
    [wheelX, -0.15, -wheelZ],
    [wheelX, -0.15, wheelZ],
  ] as Vec3[]

  return (
    <group ref={boardRef}>
      <RoundedBox args={deck} castShadow radius={0.08} smoothness={4}>
        <meshStandardMaterial color="#d8523f" metalness={0.08} roughness={0.38} />
      </RoundedBox>
      <RoundedBox
        args={[deck[0] * 0.86, deck[1] + 0.012, deck[2] * 0.92]}
        position={[0, 0.014, 0]}
        radius={0.055}
        smoothness={3}
      >
        <meshStandardMaterial color="#28352f" roughness={0.82} />
      </RoundedBox>
      {[-wheelX, wheelX].map(x => (
        <group key={`truck-${x}`} position={[x, -0.085, 0]}>
          <mesh castShadow>
            <boxGeometry args={[0.18, 0.06, deck[2] * 1.12]} />
            <meshStandardMaterial color="#d8d8ce" metalness={0.72} roughness={0.24} />
          </mesh>
          <mesh position={[0, -0.035, 0]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.055, 0.07, 0.12, 12]} />
            <meshStandardMaterial color="#867c6f" metalness={0.6} roughness={0.32} />
          </mesh>
        </group>
      ))}
      {wheelPositions.map((position, index) => (
        <mesh
          castShadow
          key={`wheel-${index}`}
          position={position}
          ref={value => {
            wheelRefs.current[index] = value
          }}
          rotation={[Math.PI / 2, 0, 0]}
        >
          <cylinderGeometry args={[wheelRadius, wheelRadius, 0.1, 20]} />
          <meshStandardMaterial color={index % 2 === 0 ? '#f2c964' : '#f19d67'} roughness={0.58} />
        </mesh>
      ))}
    </group>
  )
}

/**
 * Runtime scene-graph invariant exposed on the canvas for browser regression
 * tests. Keeping this probe outside the keyed world lets its high-water marks
 * catch even a one-frame stale/new rig overlap during resets.
 */
function SceneIntegrityProbe() {
  const highWater = useRef({
    roots: 0,
    segments: 0,
    connectors: 0,
    faces: 0,
    duplicateNames: 0,
    shadowCasters: 0,
  })
  useFrame(({ gl, scene }) => {
    let roots = 0
    let segments = 0
    let connectors = 0
    let faces = 0
    let duplicateNames = 0
    let shadowCasters = 0
    let rootUuid = ''
    const seenNames = new Set<string>()
    const wormRoots: Object3D[] = []

    scene.traverse(object => {
      const { name } = object
      if (name === 'worm-root') {
        roots += 1
        rootUuid = rootUuid || object.uuid
        wormRoots.push(object)
      } else if (name.startsWith('worm-segment-')) {
        segments += 1
      } else if (name.startsWith('worm-connector-')) {
        connectors += 1
      } else if (name === 'worm-face') {
        faces += 1
      } else {
        return
      }
      if (seenNames.has(name)) duplicateNames += 1
      seenNames.add(name)
    })
    for (const root of wormRoots) {
      root.traverse(object => {
        if (object.castShadow) shadowCasters += 1
      })
    }

    const maximum = highWater.current
    maximum.roots = Math.max(maximum.roots, roots)
    maximum.segments = Math.max(maximum.segments, segments)
    maximum.connectors = Math.max(maximum.connectors, connectors)
    maximum.faces = Math.max(maximum.faces, faces)
    maximum.duplicateNames = Math.max(maximum.duplicateNames, duplicateNames)
    maximum.shadowCasters = Math.max(maximum.shadowCasters, shadowCasters)

    const dataset = gl.domElement.dataset
    writeDataset(dataset, 'wormRoots', roots)
    writeDataset(dataset, 'wormSegments', segments)
    writeDataset(dataset, 'wormConnectors', connectors)
    writeDataset(dataset, 'wormFaces', faces)
    writeDataset(dataset, 'wormDuplicateNames', duplicateNames)
    writeDataset(dataset, 'wormShadowCasters', shadowCasters)
    writeDataset(dataset, 'wormMaxRoots', maximum.roots)
    writeDataset(dataset, 'wormMaxSegments', maximum.segments)
    writeDataset(dataset, 'wormMaxConnectors', maximum.connectors)
    writeDataset(dataset, 'wormMaxFaces', maximum.faces)
    writeDataset(dataset, 'wormMaxDuplicateNames', maximum.duplicateNames)
    writeDataset(dataset, 'wormMaxShadowCasters', maximum.shadowCasters)
    if (dataset.wormRootUuid !== rootUuid) dataset.wormRootUuid = rootUuid
  })
  return null
}

function writeDataset(dataset: DOMStringMap, key: string, value: number) {
  const serialized = String(value)
  if (dataset[key] !== serialized) dataset[key] = serialized
}

function WurmVisual({
  anatomy,
  palette,
  segmentRefs,
  materialRefs,
  connectorRefs,
  mouthRef,
}: {
  anatomy: WurmAnatomy
  palette: Color[]
  segmentRefs: React.MutableRefObject<Array<Group | null>>
  materialRefs: React.MutableRefObject<Array<MeshStandardMaterial | null>>
  connectorRefs: React.MutableRefObject<Array<Mesh | null>>
  mouthRef: React.RefObject<Mesh | null>
}) {
  return (
    <group name="worm-root">
      {Array.from({ length: SEGMENT_COUNT }, (_, index) => (
        <group
          key={`wurm-segment-${index}`}
          name={`worm-segment-${String(index + 1).padStart(2, '0')}`}
          ref={value => {
            segmentRefs.current[index] = value
          }}
        >
          <mesh receiveShadow>
            <WurmSegmentGeometry shape={anatomy.segmentShape} />
            <meshStandardMaterial
              color={palette[index]}
              emissive={palette[index]}
              emissiveIntensity={0.08}
              ref={value => {
                materialRefs.current[index] = value
              }}
              roughness={0.62}
            />
          </mesh>
          {anatomy.appendages
            .filter(appendage => appendage.anchorIndex === index)
            .map(appendage => (
              <GenomeAppendageVisual appendage={appendage} key={appendage.id} />
            ))}
          {index === SEGMENT_COUNT - 1 ? <WurmFace mouthRef={mouthRef} /> : null}
        </group>
      ))}
      {Array.from({ length: SEGMENT_COUNT - 1 }, (_, index) => (
        <mesh
          key={`wurm-connector-${index}`}
          name={`worm-connector-${String(index + 1).padStart(2, '0')}`}
          ref={value => {
            connectorRefs.current[index] = value
          }}
        >
          <cylinderGeometry args={[0.062, 0.062, 1, 12]} />
          <meshStandardMaterial color={palette[Math.min(index + 1, palette.length - 1)]} roughness={0.7} />
        </mesh>
      ))}
    </group>
  )
}

function WurmSegmentGeometry({ shape }: { shape: WurmAnatomy['segmentShape'] }) {
  if (shape === 'sphere') return <sphereGeometry args={[0.082, 16, 12]} />
  if (shape === 'box') return <boxGeometry args={[0.15, 0.11, 0.14]} />
  if (shape === 'cylinder') return <cylinderGeometry args={[0.085, 0.085, 0.13, 14]} />
  return <capsuleGeometry args={[0.085, 0.085, 5, 14]} />
}

function GenomeAppendageVisual({ appendage }: { appendage: GenomeAppendage }) {
  const lateral = appendage.side !== 0
  const side = appendage.side === 0 ? 1 : appendage.side
  return (
    <group
      position={lateral ? [0, 0, side * 0.13] : [0.12, 0, 0]}
      rotation={lateral ? [0, 0, side * 0.08] : [0, Math.PI / 2, 0]}
      scale={[appendage.thicknessScale, appendage.thicknessScale, appendage.lengthScale]}
    >
      <mesh
        rotation={
          appendage.shape === 'cylinder' || appendage.shape === 'capsule' ? [Math.PI / 2, 0, 0] : [0, 0, 0]
        }
      >
        <GenomeAppendageGeometry shape={appendage.shape} />
        <meshStandardMaterial color={appendage.color} roughness={0.68} />
      </mesh>
      <mesh position={[0, 0, side * 0.16]} scale={[1.12, 0.72, 0.42]}>
        <sphereGeometry args={[0.075, 10, 8]} />
        <meshStandardMaterial color={appendage.color} roughness={0.78} />
      </mesh>
    </group>
  )
}

function GenomeAppendageGeometry({ shape }: { shape: GenomeAppendage['shape'] }) {
  if (shape === 'box') return <boxGeometry args={[0.11, 0.1, 0.34]} />
  if (shape === 'sphere') return <sphereGeometry args={[0.11, 12, 10]} />
  if (shape === 'cylinder') return <cylinderGeometry args={[0.055, 0.07, 0.34, 12]} />
  return <capsuleGeometry args={[0.055, 0.24, 4, 10]} />
}

function WurmFace({ mouthRef }: { mouthRef: React.RefObject<Mesh | null> }) {
  return (
    <group name="worm-face" rotation={[0, 0, -Math.PI / 2]}>
      {[-0.055, 0.055].map(z => (
        <group key={z} position={[0.078, 0.055, z]}>
          <mesh>
            <sphereGeometry args={[0.032, 12, 12]} />
            <meshStandardMaterial color="#fffaf0" roughness={0.5} />
          </mesh>
          <mesh position={[0.026, 0.004, 0]}>
            <sphereGeometry args={[0.014, 10, 10]} />
            <meshStandardMaterial color="#17231e" roughness={0.35} />
          </mesh>
        </group>
      ))}
      <mesh position={[0.092, -0.018, 0]} ref={mouthRef} scale={[0.52, 0.16, 0.72]}>
        <sphereGeometry args={[0.035, 12, 9]} />
        <meshStandardMaterial color="#6f302b" roughness={0.64} />
      </mesh>
    </group>
  )
}

function Terrarium({
  terrain,
  environmentConfig,
  field,
}: {
  terrain: TerrariumDecor
  environmentConfig: EnvironmentConfig | null
  field: TerrainField
}) {
  const wallHeight = Math.max(2.35, environmentConfig?.world.size[1] ?? 2.8)
  const supportDepth = 0.42
  const supportTop = field.minimumHeight - 0.05
  const wallBottom = supportTop - 0.03
  const wallSpan = wallHeight - wallBottom
  const wallCenter = wallBottom + wallSpan * 0.5
  return (
    <group>
      <mesh receiveShadow position={[0, supportTop - supportDepth * 0.5, 0]}>
        <boxGeometry args={[field.width + 0.42, supportDepth, field.depth + 0.42]} />
        <meshStandardMaterial color="#7da17e" roughness={0.96} />
      </mesh>
      <TerrainGround field={field} roughness={environmentConfig?.terrain.roughness ?? 0.18} />
      {terrain.stones.map((stone, index) => (
        <mesh
          castShadow
          key={stone.id}
          position={stone.position}
          rotation={[0.2, index * 0.7, 0.1]}
          scale={stone.scale}
        >
          <dodecahedronGeometry args={[0.16, 0]} />
          <meshStandardMaterial color={stone.color} roughness={0.92} />
        </mesh>
      ))}
      {terrain.sprouts.map(sprout => (
        <group
          key={sprout.id}
          position={sprout.position}
          rotation={[0, sprout.rotation, 0]}
          scale={sprout.scale}
        >
          <mesh castShadow position={[0, 0.34, 0]}>
            <cylinderGeometry args={[0.05, 0.075, 0.68, 9]} />
            <meshStandardMaterial color="#5f4a32" roughness={0.94} />
          </mesh>
          <mesh castShadow position={[0.09, 0.64, 0]} rotation={[0, 0, -0.48]}>
            <sphereGeometry args={[0.19, 14, 10]} />
            <meshStandardMaterial color="#4f865b" roughness={0.82} />
          </mesh>
          <mesh castShadow position={[-0.11, 0.55, 0.035]} rotation={[0, 0, 0.58]}>
            <sphereGeometry args={[0.16, 13, 9]} />
            <meshStandardMaterial color="#6d9f65" roughness={0.84} />
          </mesh>
          <mesh castShadow position={[0, 0.8, -0.03]}>
            <sphereGeometry args={[0.17, 13, 9]} />
            <meshStandardMaterial color="#5b965f" roughness={0.8} />
          </mesh>
          <mesh position={[0, 0.035, 0]} rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.095, 0.022, 6, 16]} />
            <meshStandardMaterial color="#58442f" roughness={0.98} />
          </mesh>
        </group>
      ))}
      <GlassWall
        position={[0, wallCenter, -field.depth * 0.5 - 0.08]}
        scale={[field.width + 0.42, wallSpan, 0.045]}
      />
      <GlassWall
        position={[0, wallCenter, field.depth * 0.5 + 0.08]}
        scale={[field.width + 0.42, wallSpan, 0.045]}
      />
      <GlassWall
        position={[-field.width * 0.5 - 0.08, wallCenter, 0]}
        scale={[0.045, wallSpan, field.depth + 0.42]}
      />
      <GlassWall
        position={[field.width * 0.5 + 0.08, wallCenter, 0]}
        scale={[0.045, wallSpan, field.depth + 0.42]}
      />
      <TerrariumRim bottom={supportTop} depth={field.depth} height={wallHeight} width={field.width} />
    </group>
  )
}

function TerrainGround({ field, roughness }: { field: TerrainField; roughness: number }) {
  const geometry = useMemo(() => makeTerrainGeometry(field), [field])
  useEffect(() => () => geometry.dispose(), [geometry])
  return (
    <mesh geometry={geometry} receiveShadow>
      <meshStandardMaterial roughness={MathUtils.clamp(0.78 + roughness * 0.28, 0.78, 0.98)} vertexColors />
    </mesh>
  )
}

function makeTerrainGeometry(field: TerrainField) {
  const resolution = TERRAIN_GRID_RESOLUTION
  const row = resolution + 1
  const positions: number[] = []
  const colors: number[] = []
  const indices: number[] = []
  const color = new Color()
  const sand = new Color('#c8ad78')
  const moss = new Color('#6f9c69')
  const clay = new Color('#b87555')

  for (let zIndex = 0; zIndex <= resolution; zIndex += 1) {
    const z = (zIndex / resolution - 0.5) * field.depth
    for (let xIndex = 0; xIndex <= resolution; xIndex += 1) {
      const x = (xIndex / resolution - 0.5) * field.width
      const sample = field.sample(x, z)
      const renderedHeight = field.heightAtGridVertex(xIndex, zIndex)
      positions.push(x, renderedHeight, z)
      color.copy(sample.surface === 'moss' ? moss : sample.surface === 'clay' ? clay : sand)
      const heightTint = MathUtils.clamp((sample.height - 0.04) * 0.16, -0.05, 0.08)
      color.offsetHSL(0, 0, heightTint)
      colors.push(color.r, color.g, color.b)
    }
  }

  for (let zIndex = 0; zIndex < resolution; zIndex += 1) {
    for (let xIndex = 0; xIndex < resolution; xIndex += 1) {
      const topLeft = zIndex * row + xIndex
      const topRight = topLeft + 1
      const bottomLeft = (zIndex + 1) * row + xIndex
      const bottomRight = bottomLeft + 1
      indices.push(topLeft, bottomLeft, topRight, topRight, bottomLeft, bottomRight)
    }
  }

  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3))
  geometry.setAttribute('color', new BufferAttribute(new Float32Array(colors), 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  return geometry
}

function GlassWall({ position, scale }: { position: Vec3; scale: Vec3 }) {
  return (
    <mesh position={position} scale={scale}>
      <boxGeometry args={[1, 1, 1]} />
      <meshPhysicalMaterial color="#c9edf0" opacity={0.11} roughness={0.05} thickness={0.12} transparent />
    </mesh>
  )
}

function TerrariumRim({
  width,
  depth,
  height,
  bottom,
}: {
  width: number
  depth: number
  height: number
  bottom: number
}) {
  const x = width * 0.5 + 0.1
  const z = depth * 0.5 + 0.1
  const longBars: Array<{ position: Vec3; scale: Vec3 }> = [
    { position: [0, bottom, -z], scale: [width + 0.42, 0.09, 0.09] },
    { position: [0, bottom, z], scale: [width + 0.42, 0.09, 0.09] },
    { position: [0, height, -z], scale: [width + 0.42, 0.075, 0.075] },
    { position: [0, height, z], scale: [width + 0.42, 0.075, 0.075] },
    { position: [-x, bottom, 0], scale: [0.09, 0.09, depth + 0.42] },
    { position: [x, bottom, 0], scale: [0.09, 0.09, depth + 0.42] },
  ]
  return (
    <group>
      {longBars.map((bar, index) => (
        <mesh key={index} position={bar.position}>
          <boxGeometry args={bar.scale} />
          <meshStandardMaterial color="#315947" metalness={0.18} roughness={0.52} />
        </mesh>
      ))}
    </group>
  )
}

function LandingBurst({
  burstRefs,
  palette,
}: {
  burstRefs: React.MutableRefObject<Array<Mesh | null>>
  palette: Color[]
}) {
  return (
    <group>
      {Array.from({ length: 14 }, (_, index) => (
        <mesh
          key={`burst-${index}`}
          ref={value => {
            burstRefs.current[index] = value
          }}
          visible={false}
        >
          <tetrahedronGeometry args={[0.065, 0]} />
          <meshStandardMaterial
            color={palette[(index * 3) % palette.length]}
            emissive={palette[(index * 3) % palette.length]}
            emissiveIntensity={0.45}
          />
        </mesh>
      ))}
    </group>
  )
}

function makeWurmPalette(creature: CreatureGenome | null): Color[] {
  const configured = creature?.morphology.bodyParts.map(part => part.visual.color) ?? []
  const start = new Color(configured[0] ?? '#ef765f')
  const middle = new Color(configured[Math.floor(configured.length / 2)] ?? '#f6c882')
  const end = new Color(configured.at(-1) ?? '#df5948')
  return Array.from({ length: SEGMENT_COUNT }, (_, index) => {
    const t = index / (SEGMENT_COUNT - 1)
    return t < 0.5 ? start.clone().lerp(middle, t * 2) : middle.clone().lerp(end, (t - 0.5) * 2)
  })
}
