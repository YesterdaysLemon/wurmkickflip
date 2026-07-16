import {
  Activity,
  Apple,
  Brain,
  Cpu,
  Gauge,
  Hand,
  HeartPulse,
  Mountain,
  Pause,
  Play,
  RotateCcw,
  Sparkles,
  Timer,
  Trophy,
  Droplets,
  Download,
  FileJson,
  Upload,
  Zap,
} from 'lucide-react'
import type { CSSProperties, ChangeEvent, ReactNode } from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { deriveWurmAnatomy } from './creature/anatomy'
import { createCreatureRuntimeAdapter, FIXED_ARTICULATED_RUNTIME_PROFILE } from './creature/runtimeProfile'
import { useLabConfigs } from './creature/useLabConfigs'
import { PolicyRunner } from './policy/policyRunner'
import {
  POLICY_TIMESTEP,
  SEGMENT_COUNT,
  type GaitExperimentCommand,
  type GaitTelemetry,
  type PolicyBackend,
  type PolicyStatus,
  type ViewerMetrics,
} from './policy/types'
import {
  downloadReplayArtifact,
  LiveReplayCapture,
  replayEnvironmentSampleFor,
  replayPairDrive,
  ReplayPlayer,
  replayPlayerFromJson,
  type RecordedReplayArtifact,
  type ReplayArtifact,
  type ReplayPlaybackSample,
  type ReplayPolicyBackend,
  type ReplayRecorderFrame,
} from './replay'
import { WurmkickflipScene } from './scene/WurmkickflipScene'

type ShowcaseMode = 'kickflip' | 'freestyle'
type CaptureState = 'idle' | 'recording' | 'captured'

type ActiveReplay = {
  artifact: ReplayArtifact
  player: ReplayPlayer
}

type StuntMetrics = ViewerMetrics & {
  phase?: string
  flipProgress?: number
  flipsLanded?: number
  airtime?: number
  height?: number
  speed?: number
  landingQuality?: number
  attempt?: number
  stuntName?: string
}

const initialStatus: PolicyStatus = {
  backend: 'loading',
  message: 'Policy runtime is starting.',
  modelVersion: 'scripted-v0',
}

const initialMetrics: StuntMetrics = {
  time: 0,
  reward: 0,
  distance: 0,
  contactRatio: 0,
  backend: 'loading',
  message: 'Policy runtime is starting.',
  running: true,
  muscleActivity: new Array(16).fill(0),
  phase: 'ready',
  flipProgress: 0,
  flipsLanded: 0,
  airtime: 0,
  height: 0,
  speed: 0,
  landingQuality: 0,
  attempt: 1,
  stuntName: 'Scripted kickflip',
  bodySpeed: 0,
  mounted: true,
  distanceToBoard: 0,
  terrainFriction: 0.9,
  crawlDistance: 0,
  hunger: 0.38,
  thirst: 0.34,
  wellbeing: 0.42,
  activeNeed: null,
  needTarget: null,
  needTargetDistance: 0,
  gait: emptyGaitTelemetry('awaiting-sample'),
}

export function App() {
  const policyRunner = useMemo(() => new PolicyRunner(), [])
  const labConfigs = useLabConfigs()
  const [reducedMotion, setReducedMotion] = useState(readReducedMotionPreference)
  const [selectedCreatureId, setSelectedCreatureId] = useState<string | null>(null)
  const [selectedEnvironmentId, setSelectedEnvironmentId] = useState<string | null>(null)
  const [running, setRunning] = useState(() => !readReducedMotionPreference())
  const [resetNonce, setResetNonce] = useState(0)
  const [interactionNonce, setInteractionNonce] = useState(0)
  const [showcaseMode, setShowcaseMode] = useState<ShowcaseMode>('kickflip')
  const [policyStatus, setPolicyStatus] = useState<PolicyStatus>(initialStatus)
  const [metrics, setMetrics] = useState<StuntMetrics>(initialMetrics)
  const [selectedGaitSegment, setSelectedGaitSegment] = useState(7)
  const [gaitTractionScale, setGaitTractionScale] = useState(1)
  const gaitExperimentSequence = useRef(0)
  const [gaitExperiment, setGaitExperiment] = useState<GaitExperimentCommand | null>(null)
  const captureRef = useRef<LiveReplayCapture | null>(null)
  const captureSequenceRef = useRef(0)
  const [captureState, setCaptureState] = useState<CaptureState>('idle')
  const [captureFrameCount, setCaptureFrameCount] = useState(0)
  const [capturedReplay, setCapturedReplay] = useState<RecordedReplayArtifact | null>(null)
  const [activeReplay, setActiveReplay] = useState<ActiveReplay | null>(null)
  const [replaySample, setReplaySample] = useState<ReplayPlaybackSample | null>(null)
  const [replayCursorSeconds, setReplayCursorSeconds] = useState(0)
  const [replayPlaying, setReplayPlaying] = useState(false)
  const [replayError, setReplayError] = useState<string | null>(null)

  const selectedCreature = useMemo(
    () => labConfigs.creatures.find(creature => creature.id === selectedCreatureId) ?? labConfigs.creature,
    [labConfigs.creature, labConfigs.creatures, selectedCreatureId],
  )
  const selectedEnvironment = useMemo(
    () =>
      labConfigs.environments.find(environment => environment.id === selectedEnvironmentId) ??
      labConfigs.environment,
    [labConfigs.environment, labConfigs.environments, selectedEnvironmentId],
  )
  const selectedAnatomy = useMemo(() => deriveWurmAnatomy(selectedCreature), [selectedCreature])
  const selectedRuntime = useMemo(
    () =>
      selectedCreature
        ? createCreatureRuntimeAdapter(selectedCreature, FIXED_ARTICULATED_RUNTIME_PROFILE)
        : null,
    [selectedCreature],
  )
  const runtimeAdapter = selectedRuntime?.ok ? selectedRuntime.value : null
  const configFailures = labConfigs.resources.filter(resource => resource.state === 'error')

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return
    const query = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = (event: MediaQueryListEvent) => {
      setReducedMotion(event.matches)
      if (event.matches) {
        setRunning(false)
        setReplayPlaying(false)
      }
    }
    query.addEventListener('change', update)
    return () => query.removeEventListener('change', update)
  }, [])

  useEffect(() => {
    if (!activeReplay || !replayPlaying) return
    let animationFrame = 0
    let previousTime: number | null = null
    const tick = (time: number) => {
      if (previousTime !== null) {
        const deltaSeconds = Math.min(0.08, Math.max(0, (time - previousTime) / 1000))
        const sample = activeReplay.player.advance(deltaSeconds)
        setReplaySample(sample)
        setReplayCursorSeconds(activeReplay.player.currentTime)
        if (activeReplay.player.ended) {
          setReplayPlaying(false)
          return
        }
      }
      previousTime = time
      animationFrame = requestAnimationFrame(tick)
    }
    animationFrame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(animationFrame)
  }, [activeReplay, replayPlaying])

  const handleReplayFrame = useCallback((frame: ReplayRecorderFrame) => {
    const capture = captureRef.current
    if (!capture) return
    try {
      const index = capture.record(frame)
      if (index === 0 || (index + 1) % 6 === 0) setCaptureFrameCount(index + 1)
    } catch (error) {
      captureRef.current = null
      setCaptureState('idle')
      setReplayError(replayErrorMessage(error, 'Live capture stopped.'))
    }
  }, [])

  const handleGaitTelemetry = useCallback((gait: GaitTelemetry) => {
    setMetrics(current => ({ ...current, gait }))
  }, [])

  const isReplay = activeReplay !== null && replaySample !== null
  const activeRunning = isReplay ? replayPlaying : running
  const replayProgress = activeReplay
    ? Math.min(1, replayCursorSeconds / Math.max(activeReplay.player.durationSeconds, POLICY_TIMESTEP))
    : 0
  const displayedMetrics: StuntMetrics = isReplay
    ? {
        ...metrics,
        time: replayCursorSeconds,
        reward: replaySample.reward,
        distance: activeReplay.artifact.taskMetrics.rollingDistance * replayProgress,
        contactRatio: replaySample.contactRatio,
        backend: policyStatus.backend,
        message: policyStatus.message,
        running: replayPlaying,
        muscleActivity: replayPairDrive(replaySample),
        phase: 'replay playback',
        flipProgress: Math.min(1, Math.abs(replaySample.board.rotation[0]) / (Math.PI * 2)),
        airtime: 0,
        height: replaySample.creatureRoot.position[1],
        speed: Math.abs(replaySample.rollingVelocity),
        stuntName: 'Recorder-core replay',
        bodySpeed: Math.hypot(
          replaySample.creatureRoot.velocity[0],
          replaySample.creatureRoot.velocity[1],
          replaySample.creatureRoot.velocity[2],
        ),
        mounted: replaySample.contactRatio >= 0.5,
        crawlDistance: activeReplay.artifact.taskMetrics.rollingDistance * replayProgress,
        activeNeed: null,
        needTarget: null,
        needTargetDistance: 0,
        gait: replayGaitTelemetry(replaySample),
      }
    : {
        ...metrics,
        backend: policyStatus.backend,
        message: policyStatus.message,
        running,
      }
  const flipProgress = clamp01(displayedMetrics.flipProgress)
  const flipPercent = Math.round(flipProgress * 100)
  const flipDegrees = Math.round(flipProgress * 360)
  const landingPercent = toPercent(displayedMetrics.landingQuality)
  const attempt = Math.max(1, Math.floor(finite(displayedMetrics.attempt, 1)))
  const phase = displayedMetrics.phase || 'ready'
  const stuntName =
    displayedMetrics.stuntName || (showcaseMode === 'kickflip' ? 'Kickflip' : 'Free terrarium crawl')
  const neural = getNeuralStatus(policyStatus.backend)
  const sceneInteractionProps = { interactionNonce, showcaseMode }
  const needTargetLabel = formatNeedTarget(displayedMetrics.needTarget)

  const clearReplayMode = () => {
    setActiveReplay(null)
    setReplaySample(null)
    setReplayCursorSeconds(0)
    setReplayPlaying(false)
  }

  const restartSimulation = (mode: ShowcaseMode = showcaseMode) => {
    captureRef.current = null
    setCaptureState(capturedReplay ? 'captured' : 'idle')
    clearReplayMode()
    setRunning(!reducedMotion)
    setMetrics({
      ...initialMetrics,
      backend: policyStatus.backend,
      message: policyStatus.message,
      stuntName: mode === 'kickflip' ? 'Scripted kickflip' : 'Evolved free crawl',
    })
    gaitExperimentSequence.current += 1
    setGaitExperiment({ sequence: gaitExperimentSequence.current, kind: 'clear' })
    setGaitTractionScale(1)
    policyRunner.reset()
    setResetNonce(value => value + 1)
  }

  const activateReplay = (player: ReplayPlayer) => {
    const artifact = player.artifact()
    const sample = player.reset()
    captureRef.current = null
    setCaptureState(capturedReplay ? 'captured' : 'idle')
    setActiveReplay({ artifact, player })
    setReplaySample(sample)
    setReplayCursorSeconds(0)
    setReplayPlaying(false)
    setRunning(false)
    setReplayError(null)
  }

  const startCapture = () => {
    if (!selectedCreature || !selectedEnvironment || !runtimeAdapter) {
      setReplayError(
        'Capture needs a runtime-compatible creature and a valid environment. Retry configuration first.',
      )
      return
    }
    if (isReplay) restartSimulation()
    captureSequenceRef.current += 1
    try {
      captureRef.current = new LiveReplayCapture({
        replayId: `browser-${selectedCreature.id}-${String(captureSequenceRef.current).padStart(3, '0')}`,
        createdAt: new Date().toISOString(),
        source: {
          creatureId: selectedCreature.id,
          environmentId: selectedEnvironment.id,
          policyBackend: replayBackendFor(policyStatus.backend),
          modelVersion: policyStatus.modelVersion,
        },
        timestep: POLICY_TIMESTEP,
        environmentSample: replayEnvironmentSampleFor(selectedEnvironment),
      })
      setCaptureState('recording')
      setCaptureFrameCount(0)
      setReplayError(null)
    } catch (error) {
      captureRef.current = null
      setReplayError(replayErrorMessage(error, 'Capture could not start.'))
    }
  }

  const finishCapture = () => {
    const capture = captureRef.current
    if (!capture) {
      setReplayError('No capture is active. Choose Start capture first.')
      return
    }
    try {
      const artifact = capture.finalize()
      captureRef.current = null
      setCapturedReplay(artifact)
      setCaptureFrameCount(artifact.frameCount)
      setCaptureState('captured')
      setReplayError(null)
    } catch (error) {
      setCaptureFrameCount(capture.frameCount)
      setReplayError(replayErrorMessage(error, 'Capture could not be finalized.'))
    }
  }

  const loadCapturedReplay = () => {
    if (!capturedReplay) {
      setReplayError('Capture and finish at least one live frame before loading playback.')
      return
    }
    try {
      activateReplay(new ReplayPlayer(capturedReplay))
    } catch (error) {
      setReplayError(replayErrorMessage(error, 'Captured replay was rejected.'))
    }
  }

  const importReplay = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    try {
      const player = replayPlayerFromJson(await file.text())
      // Activate only after parsing, schema validation, and integrity checks
      // all succeed. A rejected file cannot replace live or replay state.
      activateReplay(player)
    } catch (error) {
      setReplayError(
        `${replayErrorMessage(error, 'Replay rejected.')} Select an untampered recorder-core JSON file or capture a new replay.`,
      )
    }
  }

  const exportReplay = () => {
    const artifact = activeReplay?.artifact ?? capturedReplay
    if (!artifact) {
      setReplayError('There is no finalized replay to export. Finish a live capture first.')
      return
    }
    try {
      downloadReplayArtifact(artifact, artifact.replayId)
      setReplayError(null)
    } catch (error) {
      setReplayError(replayErrorMessage(error, 'Replay export failed.'))
    }
  }

  const toggleActiveRunning = () => {
    if (!isReplay || !activeReplay) {
      setRunning(value => !value)
      return
    }
    if (!replayPlaying && activeReplay.player.ended) {
      const sample = activeReplay.player.reset()
      setReplaySample(sample)
      setReplayCursorSeconds(0)
    }
    setReplayPlaying(value => !value)
  }

  const restartReplay = () => {
    if (!activeReplay) return
    setReplaySample(activeReplay.player.reset())
    setReplayCursorSeconds(0)
    setReplayPlaying(false)
  }

  const returnToLive = () => restartSimulation()
  const resetActive = () => (isReplay ? restartReplay() : restartSimulation())

  const chooseMode = (mode: ShowcaseMode) => {
    setShowcaseMode(mode)
    restartSimulation(mode)
  }

  const runGaitExperiment = (kind: 'numb-neuron' | 'reverse-sensors' | 'lateral-shove') => {
    if (isReplay || !displayedMetrics.gait.controllerActive) return
    gaitExperimentSequence.current += 1
    const sequence = gaitExperimentSequence.current
    if (kind === 'numb-neuron') {
      setGaitExperiment({ sequence, kind, segment: selectedGaitSegment, durationSeconds: 3 })
    } else if (kind === 'reverse-sensors') {
      setGaitExperiment({ sequence, kind, durationSeconds: 3 })
    } else {
      setGaitExperiment({ sequence, kind, impulse: 1.1, durationSeconds: 3 })
    }
    if (!reducedMotion) setRunning(true)
  }

  const clearGaitExperiments = () => {
    gaitExperimentSequence.current += 1
    setGaitExperiment({ sequence: gaitExperimentSequence.current, kind: 'clear' })
    setGaitTractionScale(1)
  }

  const replayStatus = isReplay
    ? `Playback ${fixed(replayCursorSeconds, 2)} / ${fixed(activeReplay.player.durationSeconds, 2)} s`
    : captureState === 'recording'
      ? `Recording ${captureFrameCount} fixed-step frames`
      : capturedReplay
        ? `Captured ${capturedReplay.frameCount} frames · ${fixed(capturedReplay.durationSeconds, 2)} s`
        : 'Ready to capture fixed-step live frames'

  return (
    <main className="app-shell">
      <section className="sim-stage" aria-label="Wurmkickflip terrarium simulation">
        <WurmkickflipScene
          {...sceneInteractionProps}
          creature={selectedCreature}
          environmentConfig={selectedEnvironment}
          gaitExperiment={gaitExperiment}
          gaitTractionScale={gaitTractionScale}
          onMetrics={setMetrics}
          onGaitTelemetry={handleGaitTelemetry}
          onPolicyStatus={setPolicyStatus}
          onReplayFrame={captureState === 'recording' ? handleReplayFrame : undefined}
          policyRunner={policyRunner}
          replaySample={isReplay ? replaySample : null}
          resetNonce={resetNonce}
          running={running}
        />

        <div className="scene-corner-tag" aria-hidden="true">
          <span className={`live-dot${isReplay ? ' is-replay' : ''}`} />
          <span>{isReplay ? 'Terrarium replay' : 'Terrarium live'}</span>
          <b>
            {isReplay ? `${fixed(replayCursorSeconds, 1)}s` : `Attempt ${String(attempt).padStart(2, '0')}`}
          </b>
        </div>
        <div className="scene-counter" aria-hidden="true">
          <span>Clean landings</span>
          <strong>{Math.max(0, Math.floor(finite(displayedMetrics.flipsLanded)))}</strong>
        </div>
        <p className="scene-hint" aria-hidden="true">
          Drag to orbit · Scroll to zoom
        </p>
      </section>

      <aside className="control-panel" aria-label="Wurm stunt controls and telemetry">
        <header className="panel-header">
          <div className="brand-lockup">
            <p className="eyebrow">Neural terrarium / unit 01</p>
            <h1>
              Wurm
              <br />
              Kickflip
            </h1>
            <p className="brand-note">Tiny brain. Impossible sport.</p>
          </div>

          <div
            className={`neural-status neural-status--${neural.tone}`}
            role="status"
            aria-label="Neural controller status"
            aria-live="polite"
          >
            <span className="neural-status__icon">
              <Brain size={19} aria-hidden="true" />
            </span>
            <span>
              <small>Neural control</small>
              <strong>{neural.label}</strong>
              <em>{formatBackend(policyStatus.backend)}</em>
            </span>
          </div>
        </header>

        <section className="mode-section" aria-labelledby="showcase-heading">
          <div className="section-kicker">
            <Sparkles size={15} aria-hidden="true" />
            <h2 id="showcase-heading">Choose the assignment</h2>
          </div>
          <div className="mode-switch">
            <button
              className={`mode-button mode-button--kickflip${showcaseMode === 'kickflip' ? ' is-active' : ''}`}
              type="button"
              aria-pressed={showcaseMode === 'kickflip'}
              onClick={() => chooseMode('kickflip')}
            >
              <Zap size={20} aria-hidden="true" />
              <span>
                <strong>Autonomous life</strong>
                <small>Needs + neural boarding</small>
              </span>
            </button>
            <button
              className={`mode-button${showcaseMode === 'freestyle' ? ' is-active' : ''}`}
              type="button"
              aria-pressed={showcaseMode === 'freestyle'}
              onClick={() => chooseMode('freestyle')}
            >
              <Activity size={20} aria-hidden="true" />
              <span>
                <strong>Free crawl</strong>
                <small>Evolved locomotion lab</small>
              </span>
            </button>
          </div>
        </section>

        <section className={`stunt-card phase-${slugify(phase)}`} aria-labelledby="stunt-name">
          <div className="stunt-copy">
            <div className="attempt-row">
              <span>Attempt {String(attempt).padStart(2, '0')}</span>
              <b aria-live="polite">{formatPhase(phase)}</b>
            </div>
            <h2 id="stunt-name">{stuntName}</h2>
            <p>{phaseMessage(phase, showcaseMode)}</p>

            <div
              className="landed-count"
              aria-label={`${Math.max(0, Math.floor(finite(displayedMetrics.flipsLanded)))} kickflips landed`}
            >
              <Trophy size={18} aria-hidden="true" />
              <span>landed</span>
              <strong>{Math.max(0, Math.floor(finite(displayedMetrics.flipsLanded)))}</strong>
            </div>
          </div>

          <div
            className="flip-dial"
            role="progressbar"
            aria-label="Kickflip rotation progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={flipPercent}
            style={{ '--flip-progress': `${flipPercent * 3.6}deg` } as CSSProperties}
          >
            <div>
              <strong>{flipDegrees}°</strong>
              <span>board roll</span>
            </div>
          </div>
        </section>

        <dl className="telemetry-grid" aria-label="Live stunt telemetry">
          <Telemetry
            icon={<Timer size={16} />}
            label="Airtime"
            value={`${fixed(displayedMetrics.airtime, 2)} s`}
          />
          <Telemetry
            icon={<Mountain size={16} />}
            label="Height"
            value={`${fixed(displayedMetrics.height, 2)} m`}
          />
          <Telemetry
            icon={<Gauge size={16} />}
            label="Speed"
            value={`${fixed(displayedMetrics.speed, 1)} m/s`}
          />
          <Telemetry
            icon={<Sparkles size={16} />}
            label="Landing"
            value={`${landingPercent}%`}
            meter={landingPercent}
          />
        </dl>

        <section className="needs-panel" aria-labelledby="needs-heading">
          <div className="section-kicker">
            <HeartPulse size={15} aria-hidden="true" />
            <h2 id="needs-heading">Homeostasis</h2>
            <span>{needTargetLabel}</span>
          </div>
          <div className="needs-grid" aria-label="Live worm needs">
            <NeedMeter
              active={displayedMetrics.activeNeed === 'hunger'}
              icon={<Apple size={15} />}
              label="Hunger"
              urgency={displayedMetrics.hunger}
            />
            <NeedMeter
              active={displayedMetrics.activeNeed === 'thirst'}
              icon={<Droplets size={15} />}
              label="Thirst"
              urgency={displayedMetrics.thirst}
            />
            <NeedMeter
              active={displayedMetrics.activeNeed === 'wellbeing'}
              icon={<HeartPulse size={15} />}
              label="Well-being"
              urgency={displayedMetrics.wellbeing}
            />
          </div>
          <p className="needs-target">
            Selected goal <strong>{needTargetLabel}</strong>
            {displayedMetrics.needTarget ? ` · ${fixed(displayedMetrics.needTargetDistance, 1)} m away` : ''}
          </p>
        </section>

        <div className="utility-actions">
          <button
            className="poke-button"
            type="button"
            disabled={isReplay}
            onClick={() => {
              if (!reducedMotion) setRunning(true)
              setInteractionNonce(value => value + 1)
            }}
          >
            <Hand size={18} aria-hidden="true" />
            Poke wurm
          </button>
          <button type="button" aria-pressed={!activeRunning} onClick={toggleActiveRunning}>
            {activeRunning ? <Pause size={17} aria-hidden="true" /> : <Play size={17} aria-hidden="true" />}
            {activeRunning ? 'Pause' : 'Play'}
          </button>
          <button type="button" onClick={resetActive}>
            <RotateCcw size={17} aria-hidden="true" />
            {isReplay ? 'Restart' : 'Reset'}
          </button>
        </div>
        <details className="replay-panel">
          <summary>
            <FileJson size={14} aria-hidden="true" />
            <span>Replay lab</span>
            <b>{isReplay ? 'Playback' : captureState}</b>
          </summary>
          <div className="replay-panel__body">
            <p className="replay-status" role="status" aria-live="polite">
              {replayStatus}
            </p>
            {activeReplay ? (
              <p className="replay-source">
                <b>{activeReplay.artifact.source.creatureId}</b> in{' '}
                <b>{activeReplay.artifact.source.environmentId}</b> ·{' '}
                {activeReplay.artifact.source.modelVersion}
              </p>
            ) : null}
            {replayError ? (
              <div className="replay-error" role="alert">
                {replayError}
              </div>
            ) : null}
            <div className="replay-actions">
              {captureState === 'recording' ? (
                <button type="button" onClick={finishCapture}>
                  <Pause size={14} aria-hidden="true" /> Finish capture
                </button>
              ) : !isReplay ? (
                <button type="button" onClick={startCapture}>
                  <Activity size={14} aria-hidden="true" /> Start capture
                </button>
              ) : null}
              {capturedReplay && !isReplay ? (
                <button type="button" onClick={loadCapturedReplay}>
                  <Play size={14} aria-hidden="true" /> Load captured replay
                </button>
              ) : null}
              {isReplay ? (
                <>
                  <button type="button" onClick={toggleActiveRunning}>
                    {replayPlaying ? (
                      <Pause size={14} aria-hidden="true" />
                    ) : (
                      <Play size={14} aria-hidden="true" />
                    )}
                    {replayPlaying ? 'Pause replay' : 'Play replay'}
                  </button>
                  <button type="button" onClick={restartReplay}>
                    <RotateCcw size={14} aria-hidden="true" /> Restart replay
                  </button>
                  <button type="button" onClick={returnToLive}>
                    Return to live
                  </button>
                </>
              ) : null}
              {activeReplay || capturedReplay ? (
                <button type="button" onClick={exportReplay}>
                  <Download size={14} aria-hidden="true" /> Export replay JSON
                </button>
              ) : null}
              <label className="replay-import">
                <Upload size={14} aria-hidden="true" />
                Import replay JSON
                <input type="file" accept="application/json,.json" onChange={importReplay} />
              </label>
            </div>
            <p className="replay-limitation">
              Board and root poses, contact, replay time, and all 32 muscle channels come from the verified
              artifact. Schema v1 omits segment poses and needs state, so the visible body is reconstructed
              from root + activations while live bowls remain context. It also omits lifecycle muscle ordering
              and traction, so the gait microscope leaves those fields unassigned.
            </p>
          </div>
        </details>
        {reducedMotion ? (
          <p className="reduced-motion-note" aria-live="polite">
            Reduced motion is on. Live simulation and loaded replays start paused; choose Play whenever you
            want them to run.
          </p>
        ) : null}

        <section className="terrarium-controls" aria-labelledby="terrarium-heading">
          <div className="section-kicker">
            <Mountain size={15} aria-hidden="true" />
            <h2 id="terrarium-heading">Terrarium</h2>
          </div>
          {labConfigs.status !== 'ready' ? (
            <div
              className={`config-alert config-alert--${labConfigs.status}`}
              role={labConfigs.status === 'error' ? 'alert' : 'status'}
              aria-live="polite"
            >
              <strong>
                {labConfigs.status === 'loading'
                  ? 'Loading configuration resources'
                  : labConfigs.status === 'partial'
                    ? 'Some configuration resources failed'
                    : 'Configuration needs attention'}
              </strong>
              <p>{labConfigs.message}</p>
              {configFailures.length > 0 ? (
                <ul aria-label="Configuration resource errors">
                  {configFailures.slice(0, 4).map(resource => (
                    <li key={`${resource.kind}-${resource.path}`}>
                      <code>{resource.path}</code>{' '}
                      <span>{resource.errors[0] ?? 'Unknown resource error.'}</span>
                    </li>
                  ))}
                </ul>
              ) : null}
              <button type="button" disabled={labConfigs.status === 'loading'} onClick={labConfigs.retry}>
                <RotateCcw size={15} aria-hidden="true" />
                {labConfigs.status === 'loading' ? 'Loading…' : 'Retry configuration'}
              </button>
            </div>
          ) : null}
          <label className="select-control">
            <span>Environment</span>
            <select
              value={selectedEnvironment?.id ?? ''}
              disabled={labConfigs.environments.length === 0}
              onChange={event => {
                setSelectedEnvironmentId(event.target.value)
                restartSimulation()
              }}
            >
              {labConfigs.environments.map(environment => (
                <option key={environment.id} value={environment.id}>
                  {environment.name}
                </option>
              ))}
            </select>
          </label>
          {selectedEnvironment ? (
            <dl className="environment-facts" aria-label="Selected environment parameters">
              <div>
                <dt>Terrain</dt>
                <dd>
                  {selectedEnvironment.terrain.kind} / seed {selectedEnvironment.seed}
                </dd>
              </div>
              <div>
                <dt>Gravity</dt>
                <dd>{Math.abs(selectedEnvironment.world.gravity[1]).toFixed(1)} m/s²</dd>
              </div>
              <div>
                <dt>Arena</dt>
                <dd>
                  {selectedEnvironment.world.size[0].toFixed(1)} ×{' '}
                  {selectedEnvironment.world.size[2].toFixed(1)} m
                </dd>
              </div>
              <div>
                <dt>Grip</dt>
                <dd>
                  {selectedEnvironment.terrain.baseFriction.toFixed(2)} ground /{' '}
                  {selectedEnvironment.skateboard.wheelFriction.toFixed(2)} wheel
                </dd>
              </div>
            </dl>
          ) : null}

          <details className="genome-drawer">
            <summary>
              <span>Wurm body</span>
              <strong>{selectedCreature?.name ?? labConfigs.message}</strong>
            </summary>
            <label className="select-control select-control--inside">
              <span>Creature genome</span>
              <select
                value={selectedCreature?.id ?? ''}
                disabled={labConfigs.creatures.length === 0}
                onChange={event => {
                  setSelectedCreatureId(event.target.value)
                  restartSimulation()
                }}
              >
                {labConfigs.creatures.map(creature => (
                  <option key={creature.id} value={creature.id}>
                    {creature.name}
                  </option>
                ))}
              </select>
            </label>
            <dl className="genome-facts" aria-label="Selected genome anatomy">
              <div>
                <dt>Form</dt>
                <dd>{selectedAnatomy.silhouette}</dd>
              </div>
              <div>
                <dt>Source body</dt>
                <dd>
                  {selectedAnatomy.bodyPartCount} parts / {selectedAnatomy.appendages.length} limbs
                </dd>
              </div>
              <div>
                <dt>Mass</dt>
                <dd>{selectedAnatomy.totalMass.toFixed(2)} kg</dd>
              </div>
              <div>
                <dt>Stance</dt>
                <dd>{selectedAnatomy.stanceWidth.toFixed(2)} m</dd>
              </div>
            </dl>
            {runtimeAdapter ? (
              <>
                <dl className="genome-facts" aria-label="Selected runtime compatibility">
                  <div>
                    <dt>Compatibility</dt>
                    <dd>Compatible · {runtimeAdapter.profile.id}</dd>
                  </div>
                  <div>
                    <dt>Plant</dt>
                    <dd>{runtimeAdapter.profile.plantVersion}</dd>
                  </div>
                  <div>
                    <dt>Control</dt>
                    <dd>16 antagonistic pairs / 32 muscle channels</dd>
                  </div>
                  <div>
                    <dt>Policy</dt>
                    <dd>{runtimeAdapter.profile.policy.kind}</dd>
                  </div>
                </dl>
                <p className="genome-contract">
                  Appearance-only projection onto the fixed 16-segment articulated runtime. The genome's
                  declared {runtimeAdapter.appearanceProjection.declaredControllerKind} controller (
                  {runtimeAdapter.appearanceProjection.declaredControllerActionSize} actions) is metadata and
                  is not executed.
                </p>
              </>
            ) : selectedRuntime && !selectedRuntime.ok ? (
              <div className="genome-contract" role="alert">
                Runtime incompatible: {selectedRuntime.errors.join(' ')}
              </div>
            ) : (
              <p className="genome-contract">Choose a valid creature to check fixed-runtime compatibility.</p>
            )}
          </details>
        </section>

        <GaitMicroscope
          disabled={isReplay}
          gait={displayedMetrics.gait}
          onClear={clearGaitExperiments}
          onExperiment={runGaitExperiment}
          onSelectedSegmentChange={setSelectedGaitSegment}
          onTractionScaleChange={setGaitTractionScale}
          selectedSegment={selectedGaitSegment}
          tractionScale={gaitTractionScale}
        />

        <div className="lab-tape" aria-label="Rollout diagnostics">
          <span>
            RWD <b>{fixed(displayedMetrics.reward, 1)}</b>
          </span>
          <span>
            {displayedMetrics.mounted ? 'RIDE' : 'CRAWL'}{' '}
            <b>
              {fixed(
                displayedMetrics.mounted ? displayedMetrics.distance : displayedMetrics.crawlDistance,
                1,
              )}
              m
            </b>
          </span>
          <span>
            GRIP <b>{toPercent(displayedMetrics.contactRatio)}%</b>
          </span>
          <span>
            BODY <b>{fixed(displayedMetrics.bodySpeed, 1)}</b>
          </span>
          <span aria-label={`Simulation time ${fixed(displayedMetrics.time, 1)} seconds`}>
            T <b>{fixed(displayedMetrics.time, 1)}s</b>
          </span>
        </div>

        <details className="policy-note">
          <summary>
            <Cpu size={14} aria-hidden="true" /> Brain note · {policyStatus.modelVersion}
          </summary>
          <p>{displayedMetrics.message}</p>
          <code>
            <span>
              LEARNED · goal sensors + proprioception → 16 antagonistic pairs / 32 muscle channels @ 60 Hz
            </span>
            <span>DERIVED · mean-free segment forces, terrain + skateboard support, swept prop contacts</span>
            <span>SCRIPTED · feed/dismount choreography + aerial kickflip</span>
          </code>
        </details>
      </aside>
    </main>
  )
}

type GaitMicroscopeProps = {
  gait: GaitTelemetry
  selectedSegment: number
  tractionScale: number
  disabled: boolean
  onSelectedSegmentChange: (segment: number) => void
  onTractionScaleChange: (scale: number) => void
  onExperiment: (kind: 'numb-neuron' | 'reverse-sensors' | 'lateral-shove') => void
  onClear: () => void
}

const GAIT_ROWS = [
  {
    key: 'neural',
    label: 'Neuron',
    maximum: 1,
    signed: true,
    liveOnly: true,
    read: (segment: GaitTelemetry['segments'][number]) => segment.neuralActivation,
  },
  {
    key: 'muscle',
    label: 'Muscle',
    maximum: 1,
    signed: true,
    liveOnly: false,
    read: (segment: GaitTelemetry['segments'][number]) => segment.muscleCommand,
  },
  {
    key: 'bend',
    label: 'Bend',
    maximum: 1.2,
    signed: true,
    liveOnly: true,
    read: (segment: GaitTelemetry['segments'][number]) => segment.jointBend,
  },
  {
    key: 'support',
    label: 'Support',
    maximum: 1,
    signed: false,
    liveOnly: true,
    read: (segment: GaitTelemetry['segments'][number]) => segment.contactLoad,
  },
  {
    key: 'slip',
    label: 'Slip',
    maximum: 2,
    signed: false,
    liveOnly: true,
    read: (segment: GaitTelemetry['segments'][number]) => segment.slipSpeed,
  },
] as const

function GaitMicroscope({
  gait,
  selectedSegment,
  tractionScale,
  disabled,
  onSelectedSegmentChange,
  onTractionScaleChange,
  onExperiment,
  onClear,
}: GaitMicroscopeProps) {
  const selected = gait.segments[selectedSegment] ?? emptyGaitSegment(selectedSegment)
  const liveNeural = gait.source === 'live-neural'
  const experimentDisabled = disabled || !gait.controllerActive
  const activeKind = gait.activeExperiment?.kind ?? null
  const sourceLabel =
    gait.source === 'live-neural'
      ? 'Evolved crawl'
      : gait.source === 'scripted-stunt'
        ? 'Scripted pose'
        : gait.source === 'lifecycle-handoff'
          ? 'Control handoff'
          : gait.source === 'replay-muscles'
            ? 'Replay v1 · order unknown'
            : gait.source === 'awaiting-sample'
              ? 'Awaiting sample'
              : 'Brain offline'
  const activeDescription = gait.activeExperiment
    ? `${formatExperimentKind(gait.activeExperiment.kind)}${
        gait.activeExperiment.segment === null
          ? ''
          : ` on S${String(gait.activeExperiment.segment + 1).padStart(2, '0')}`
      }`
    : gait.experimentNotice
      ? gait.experimentNotice
      : tractionScale !== 1 && gait.tractionAvailable
        ? `Traction scaled to ${Math.round(tractionScale * 100)}%`
        : liveNeural
          ? 'Baseline body and wiring'
          : gait.source === 'replay-muscles'
            ? 'Replay v1 did not record anatomical muscle ordering, traction, hidden state, or local afferents.'
            : gait.source === 'scripted-stunt'
              ? 'Choose Free crawl to hand the body to the evolved segment controller.'
              : gait.source === 'lifecycle-handoff'
                ? 'One fixed step is transferring the body between authored motion and the evolved controller.'
                : gait.source === 'awaiting-sample'
                  ? 'Waiting for the first detached body sample.'
                  : 'The evolved crawl artifact is unavailable.'

  return (
    <section className="gait-microscope" data-testid="gait-microscope" aria-labelledby="gait-heading">
      <div className="section-kicker">
        <Brain size={15} aria-hidden="true" />
        <h2 id="gait-heading">Gait microscope</h2>
        <span>{sourceLabel}</span>
      </div>

      <div className="gait-direction" aria-label="Body-axis gait measurements">
        <span>
          Forward <b>{formatSigned(gait.bodyForwardSpeed)} m/s</b>
        </span>
        <span>
          Sideways <b>{formatSigned(gait.bodyLateralSpeed)} m/s</b>
        </span>
        <span>
          Facing goal <b>{gait.targetAlignmentAvailable ? formatSigned(gait.targetAlignment) : '—'}</b>
        </span>
      </div>

      <div className="gait-axis" aria-hidden="true">
        <span>Head · S01</span>
        <i />
        <span>S16 · Tail</span>
      </div>
      <div className="gait-matrix" role="table" aria-label="Live gait microscope">
        <div role="rowgroup">
          {GAIT_ROWS.map(row => (
            <div className={`gait-row gait-row--${row.key}`} role="row" key={row.key}>
              <span className="gait-row-label" role="rowheader">
                {row.label}
              </span>
              {gait.segments.map(segment => {
                const value = row.read(segment)
                const available = row.liveOnly
                  ? liveNeural
                  : gait.source !== 'unavailable' &&
                    gait.source !== 'awaiting-sample' &&
                    gait.segmentOrder === 'anterior-to-posterior'
                const level = available ? clamp01(Math.abs(value) / row.maximum) : 0
                const label = available
                  ? `${row.label}, segment ${segment.segment + 1}: ${value.toFixed(3)}`
                  : `${row.label}, segment ${segment.segment + 1}: unavailable`
                return (
                  <span
                    aria-label={label}
                    className={`gait-cell${value < 0 && row.signed ? ' is-negative' : ''}${
                      segment.segment === selectedSegment ? ' is-selected' : ''
                    }${available ? '' : ' is-unavailable'}`}
                    data-testid={`gait-cell-${row.key}-${String(segment.segment + 1).padStart(2, '0')}`}
                    data-value={available ? value.toFixed(6) : ''}
                    key={segment.segment}
                    role="cell"
                  >
                    <i style={{ '--gait-level': level } as CSSProperties} />
                  </span>
                )
              })}
            </div>
          ))}
        </div>
      </div>

      <p className="gait-probe">
        {liveNeural ? (
          <>
            <b>S{String(selectedSegment + 1).padStart(2, '0')}</b> drive {formatSigned(selected.neuralDrive)}
            {' · '}neuron {formatSigned(selected.neuralActivation)} {' · '}command{' '}
            {formatSigned(selected.requestedMuscleCommand)} → {formatSigned(selected.muscleCommand)}
            {' · '}afferent/body support {fixed(selected.afferentContactLoad, 2)} →{' '}
            {fixed(selected.contactLoad, 2)}
            {' · '}obstacle {formatSigned(selected.obstacleForward)}/{formatSigned(selected.obstacleRight)}
          </>
        ) : (
          'Neural drive, bend, support, and slip appear only when the evolved crawl controller owns the body.'
        )}
      </p>

      <div className="gait-sliders">
        <label>
          <span>Probe S{String(selectedSegment + 1).padStart(2, '0')}</span>
          <input
            aria-label="Perturbation segment"
            type="range"
            min={0}
            max={SEGMENT_COUNT - 1}
            step={1}
            value={selectedSegment}
            onChange={event => onSelectedSegmentChange(Number(event.target.value))}
          />
          <small>head → tail</small>
        </label>
        <label>
          <span>
            {!gait.tractionAvailable ? 'Traction unrecorded' : `Traction ${Math.round(tractionScale * 100)}%`}
          </span>
          <input
            aria-label="Traction scale"
            disabled={disabled}
            type="range"
            min={0}
            max={1.5}
            step={0.05}
            value={gait.tractionAvailable ? tractionScale : 1}
            onChange={event => onTractionScaleChange(Number(event.target.value))}
          />
          <small>slick → grippy</small>
        </label>
      </div>

      <div className="gait-actions">
        <button
          type="button"
          aria-pressed={activeKind === 'numb-neuron'}
          disabled={experimentDisabled}
          onClick={() => onExperiment('numb-neuron')}
        >
          Numb selected segment
        </button>
        <button
          type="button"
          aria-pressed={activeKind === 'reverse-sensors'}
          disabled={experimentDisabled}
          onClick={() => onExperiment('reverse-sensors')}
        >
          Reverse sensory wiring
        </button>
        <button
          type="button"
          aria-pressed={activeKind === 'lateral-shove'}
          disabled={experimentDisabled}
          onClick={() => onExperiment('lateral-shove')}
        >
          Shove worm sideways
        </button>
        <button type="button" disabled={disabled} onClick={onClear}>
          Clear perturbations
        </button>
      </div>
      <p className="gait-experiment-status">
        <span aria-live="polite">{activeDescription}</span>
        {gait.activeExperiment ? (
          <span aria-hidden="true"> · {fixed(gait.activeExperiment.remainingSeconds, 1)} s</span>
        ) : null}
      </p>
    </section>
  )
}

type TelemetryProps = {
  icon: ReactNode
  label: string
  value: string
  meter?: number
}

type NeedMeterProps = {
  active: boolean
  icon: ReactNode
  label: string
  urgency: number
}

function NeedMeter({ active, icon, label, urgency }: NeedMeterProps) {
  const level = clamp01(urgency)
  const percent = toPercent(level)
  return (
    <div className={`need-meter${active ? ' is-active' : ''}`}>
      <span>
        {icon}
        <b>{label}</b>
      </span>
      <strong>{percent}%</strong>
      <i aria-hidden="true">
        <b style={{ width: `${percent}%` }} />
      </i>
      <span
        className="sr-only"
        role="meter"
        aria-label={`${label} urgency`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
      />
    </div>
  )
}

function Telemetry({ icon, label, value, meter }: TelemetryProps) {
  return (
    <div className="telemetry-card">
      <dt>
        {icon}
        {label}
      </dt>
      <dd>{value}</dd>
      {meter === undefined ? null : (
        <span className="telemetry-meter" aria-hidden="true">
          <i style={{ width: `${meter}%` }} />
        </span>
      )}
    </div>
  )
}

function getNeuralStatus(backend: PolicyBackend) {
  if (backend === 'loading') return { label: 'Waking', tone: 'loading' }
  if (backend === 'unavailable') return { label: 'Offline', tone: 'fallback' }
  if (backend === 'neural-js') return { label: 'Online', tone: 'online' }
  return { label: 'Fallback', tone: 'fallback' }
}

function emptyGaitSegment(segment: number): GaitTelemetry['segments'][number] {
  return {
    segment,
    neuralActivation: 0,
    neuralDrive: 0,
    muscleCommand: 0,
    requestedMuscleCommand: 0,
    jointBend: 0,
    jointVelocity: 0,
    afferentJointBend: 0,
    afferentJointVelocity: 0,
    afferentContactLoad: 0,
    afferentSlipSpeed: 0,
    afferentObstacleForward: 0,
    afferentObstacleRight: 0,
    contactLoad: 0,
    slipSpeed: 0,
    obstacleForward: 0,
    obstacleRight: 0,
  }
}

function emptyGaitTelemetry(source: GaitTelemetry['source']): GaitTelemetry {
  return {
    controllerActive: false,
    source,
    segmentOrder: 'anterior-to-posterior',
    tractionScale: 1,
    tractionAvailable: source !== 'replay-muscles',
    bodyForwardSpeed: 0,
    bodyLateralSpeed: 0,
    targetAlignment: 0,
    targetAlignmentAvailable: false,
    activeExperiment: null,
    experimentNotice: null,
    segments: Array.from({ length: SEGMENT_COUNT }, (_, segment) => emptyGaitSegment(segment)),
  }
}

function replayGaitTelemetry(sample: ReplayPlaybackSample): GaitTelemetry {
  const gait = emptyGaitTelemetry('replay-muscles')
  const heading = sample.creatureRoot.rotation[1]
  const forwardX = Math.cos(heading)
  const forwardZ = Math.sin(heading)
  const rightX = -forwardZ
  const rightZ = forwardX
  gait.segmentOrder = 'unavailable'
  gait.bodyForwardSpeed =
    sample.creatureRoot.velocity[0] * forwardX + sample.creatureRoot.velocity[2] * forwardZ
  gait.bodyLateralSpeed = sample.creatureRoot.velocity[0] * rightX + sample.creatureRoot.velocity[2] * rightZ
  return gait
}

function formatExperimentKind(kind: NonNullable<GaitTelemetry['activeExperiment']>['kind']) {
  if (kind === 'numb-neuron') return 'Segment numbed'
  if (kind === 'reverse-sensors') return 'Sensory wiring reversed'
  return 'Side shove recovery'
}

function formatSigned(value: number | undefined) {
  const safe = finite(value)
  return `${safe >= 0 ? '+' : ''}${safe.toFixed(2)}`
}

function formatBackend(backend: PolicyBackend) {
  if (backend === 'neural-js') return 'Evolved recurrent gait / Neural JS'
  if (backend === 'scripted') return 'Scripted cortex'
  if (backend === 'unavailable') return 'Locomotion brain unavailable'
  return 'Neural cortex warming'
}

function formatPhase(phase: string) {
  return phase.replace(/[-_]/g, ' ').trim().toUpperCase() || 'READY'
}

function phaseMessage(phase: string, mode: ShowcaseMode) {
  const normalized = slugify(phase)
  if (normalized.includes('dismount')) return 'Board break. Tiny feet on terrain.'
  if (normalized.includes('finding-board')) return 'Sniffing out the ride.'
  if (normalized.includes('seeking-food')) return 'The little brain has selected lunch.'
  if (normalized.includes('seeking-water')) return 'Hydration waypoint acquired.'
  if (normalized.includes('eating')) return 'Pellet contact. Hunger falling.'
  if (normalized.includes('drinking')) return 'Tiny sips, serious science.'
  if (normalized.includes('mounting')) return 'Climb aboard, long athlete.'
  if (normalized.includes('crawl')) return 'Independent worm transportation.'
  if (normalized.includes('landed')) return 'Four wheels down. Maximum glory.'
  if (normalized.includes('landing')) return 'Find the bolts, tiny champion.'
  if (normalized.includes('flip') || normalized.includes('air')) return 'No ground. Only courage.'
  if (normalized.includes('pop') || normalized.includes('launch')) return 'Tail down. Wurm up.'
  if (normalized.includes('flop') || normalized.includes('tumble'))
    return 'That absolutely counts as research.'
  if (normalized.includes('coil') || normalized.includes('setup')) return 'Coiling the little athlete.'
  return mode === 'kickflip'
    ? 'Brain linked. Needs online. Believe in wurm.'
    : 'Clock-free gait under observation.'
}

function formatNeedTarget(target: ViewerMetrics['needTarget']) {
  if (target === 'food-bowl') return 'Food bowl'
  if (target === 'water-bowl') return 'Water bowl'
  if (target === 'skateboard') return 'Skateboard'
  return 'Choosing'
}

function replayBackendFor(backend: PolicyBackend): ReplayPolicyBackend {
  if (backend === 'neural-js' || backend === 'scripted') return backend
  return 'scripted'
}

function replayErrorMessage(error: unknown, prefix: string) {
  return `${prefix} ${error instanceof Error ? error.message : 'Unknown replay error.'}`
}

function slugify(value: string) {
  return (
    value
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '') || 'ready'
  )
}

function finite(value: number | undefined, fallback = 0) {
  return Number.isFinite(value) ? (value as number) : fallback
}

function clamp01(value: number | undefined) {
  return Math.max(0, Math.min(1, finite(value)))
}

function toPercent(value: number | undefined) {
  const safe = finite(value)
  return Math.round(Math.max(0, Math.min(100, safe > 1 ? safe : safe * 100)))
}

function fixed(value: number | undefined, digits: number) {
  return finite(value).toFixed(digits)
}

function readReducedMotionPreference() {
  return (
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  )
}
