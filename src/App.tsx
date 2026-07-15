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
  Zap,
} from 'lucide-react'
import type { CSSProperties, ReactNode } from 'react'
import { useMemo, useState } from 'react'
import { deriveWurmAnatomy } from './creature/anatomy'
import { useLabConfigs } from './creature/useLabConfigs'
import { PolicyRunner } from './policy/policyRunner'
import type { PolicyBackend, PolicyStatus, ViewerMetrics } from './policy/types'
import { WurmkickflipScene } from './scene/WurmkickflipScene'

type ShowcaseMode = 'kickflip' | 'freestyle'

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
}

export function App() {
  const policyRunner = useMemo(() => new PolicyRunner(), [])
  const labConfigs = useLabConfigs()
  const [selectedCreatureId, setSelectedCreatureId] = useState<string | null>(null)
  const [selectedEnvironmentId, setSelectedEnvironmentId] = useState<string | null>(null)
  const [running, setRunning] = useState(true)
  const [resetNonce, setResetNonce] = useState(0)
  const [interactionNonce, setInteractionNonce] = useState(0)
  const [showcaseMode, setShowcaseMode] = useState<ShowcaseMode>('kickflip')
  const [policyStatus, setPolicyStatus] = useState<PolicyStatus>(initialStatus)
  const [metrics, setMetrics] = useState<StuntMetrics>(initialMetrics)

  const selectedCreature = useMemo(
    () =>
      labConfigs.creatures.find((creature) => creature.id === selectedCreatureId) ??
      labConfigs.creature,
    [labConfigs.creature, labConfigs.creatures, selectedCreatureId],
  )
  const selectedEnvironment = useMemo(
    () =>
      labConfigs.environments.find((environment) => environment.id === selectedEnvironmentId) ??
      labConfigs.environment,
    [labConfigs.environment, labConfigs.environments, selectedEnvironmentId],
  )
  const selectedAnatomy = useMemo(() => deriveWurmAnatomy(selectedCreature), [selectedCreature])

  const displayedMetrics: StuntMetrics = {
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
  const stuntName = displayedMetrics.stuntName || (showcaseMode === 'kickflip' ? 'Kickflip' : 'Free terrarium crawl')
  const neural = getNeuralStatus(policyStatus.backend)
  const sceneInteractionProps = { interactionNonce, showcaseMode }
  const needTargetLabel = formatNeedTarget(displayedMetrics.needTarget)

  const restartSimulation = (mode: ShowcaseMode = showcaseMode) => {
    setRunning(true)
    setMetrics({
      ...initialMetrics,
      backend: policyStatus.backend,
      message: policyStatus.message,
      stuntName: mode === 'kickflip' ? 'Scripted kickflip' : 'Evolved free crawl',
    })
    policyRunner.reset()
    setResetNonce((value) => value + 1)
  }

  const resetSimulation = () => restartSimulation()

  const chooseMode = (mode: ShowcaseMode) => {
    setShowcaseMode(mode)
    restartSimulation(mode)
  }

  return (
    <main className="app-shell">
      <section className="sim-stage" aria-label="Wurmkickflip terrarium simulation">
        <WurmkickflipScene
          {...sceneInteractionProps}
          creature={selectedCreature}
          environmentConfig={selectedEnvironment}
          onMetrics={setMetrics}
          onPolicyStatus={setPolicyStatus}
          policyRunner={policyRunner}
          resetNonce={resetNonce}
          running={running}
        />

        <div className="scene-corner-tag" aria-hidden="true">
          <span className="live-dot" />
          <span>Terrarium live</span>
          <b>Attempt {String(attempt).padStart(2, '0')}</b>
        </div>
        <div className="scene-counter" aria-hidden="true">
          <span>Clean landings</span>
          <strong>{Math.max(0, Math.floor(finite(displayedMetrics.flipsLanded)))}</strong>
        </div>
        <p className="scene-hint" aria-hidden="true">Drag to orbit · Scroll to zoom</p>
      </section>

      <aside className="control-panel" aria-label="Wurm stunt controls and telemetry">
        <header className="panel-header">
          <div className="brand-lockup">
            <p className="eyebrow">Neural terrarium / unit 01</p>
            <h1>Wurm<br />Kickflip</h1>
            <p className="brand-note">Tiny brain. Impossible sport.</p>
          </div>

          <div className={`neural-status neural-status--${neural.tone}`} role="status" aria-live="polite">
            <span className="neural-status__icon"><Brain size={19} aria-hidden="true" /></span>
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
              <span><strong>Autonomous life</strong><small>Needs + scripted kickflip</small></span>
            </button>
            <button
              className={`mode-button${showcaseMode === 'freestyle' ? ' is-active' : ''}`}
              type="button"
              aria-pressed={showcaseMode === 'freestyle'}
              onClick={() => chooseMode('freestyle')}
            >
              <Activity size={20} aria-hidden="true" />
              <span><strong>Free crawl</strong><small>Evolved locomotion lab</small></span>
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

            <div className="landed-count" aria-label={`${Math.max(0, Math.floor(finite(displayedMetrics.flipsLanded)))} kickflips landed`}>
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
          <Telemetry icon={<Timer size={16} />} label="Airtime" value={`${fixed(displayedMetrics.airtime, 2)} s`} />
          <Telemetry icon={<Mountain size={16} />} label="Height" value={`${fixed(displayedMetrics.height, 2)} m`} />
          <Telemetry icon={<Gauge size={16} />} label="Speed" value={`${fixed(displayedMetrics.speed, 1)} m/s`} />
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
            onClick={() => {
              setRunning(true)
              setInteractionNonce((value) => value + 1)
            }}
          >
            <Hand size={18} aria-hidden="true" />
            Poke wurm
          </button>
          <button type="button" aria-pressed={!running} onClick={() => setRunning((value) => !value)}>
            {running ? <Pause size={17} aria-hidden="true" /> : <Play size={17} aria-hidden="true" />}
            {running ? 'Pause' : 'Resume'}
          </button>
          <button type="button" onClick={resetSimulation}>
            <RotateCcw size={17} aria-hidden="true" />
            Reset
          </button>
        </div>

        <section className="terrarium-controls" aria-labelledby="terrarium-heading">
          <div className="section-kicker">
            <Mountain size={15} aria-hidden="true" />
            <h2 id="terrarium-heading">Terrarium</h2>
          </div>
          <label className="select-control">
            <span>Environment</span>
            <select
              value={selectedEnvironment?.id ?? ''}
              disabled={labConfigs.environments.length === 0}
              onChange={(event) => {
                setSelectedEnvironmentId(event.target.value)
                restartSimulation()
              }}
            >
              {labConfigs.environments.map((environment) => (
                <option key={environment.id} value={environment.id}>{environment.name}</option>
              ))}
            </select>
          </label>
          {selectedEnvironment ? (
            <dl className="environment-facts" aria-label="Selected environment parameters">
              <div><dt>Terrain</dt><dd>{selectedEnvironment.terrain.kind} / seed {selectedEnvironment.seed}</dd></div>
              <div><dt>Gravity</dt><dd>{Math.abs(selectedEnvironment.world.gravity[1]).toFixed(1)} m/s²</dd></div>
              <div><dt>Arena</dt><dd>{selectedEnvironment.world.size[0].toFixed(1)} × {selectedEnvironment.world.size[2].toFixed(1)} m</dd></div>
              <div><dt>Grip</dt><dd>{selectedEnvironment.terrain.baseFriction.toFixed(2)} ground / {selectedEnvironment.skateboard.wheelFriction.toFixed(2)} wheel</dd></div>
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
                onChange={(event) => {
                  setSelectedCreatureId(event.target.value)
                  restartSimulation()
                }}
              >
                {labConfigs.creatures.map((creature) => (
                  <option key={creature.id} value={creature.id}>{creature.name}</option>
                ))}
              </select>
            </label>
            <dl className="genome-facts" aria-label="Selected genome anatomy">
              <div><dt>Form</dt><dd>{selectedAnatomy.silhouette}</dd></div>
              <div><dt>Source body</dt><dd>{selectedAnatomy.bodyPartCount} parts / {selectedAnatomy.appendages.length} limbs</dd></div>
              <div><dt>Mass</dt><dd>{selectedAnatomy.totalMass.toFixed(2)} kg</dd></div>
              <div><dt>Stance</dt><dd>{selectedAnatomy.stanceWidth.toFixed(2)} m</dd></div>
            </dl>
            <p className="genome-contract">Projected onto the policy's fixed 16-node muscle lattice.</p>
          </details>
        </section>

        <section className="muscle-readout" aria-labelledby="muscle-heading">
          <div className="section-kicker">
            <Activity size={15} aria-hidden="true" />
            <h2 id="muscle-heading">Muscle bus</h2>
            <span>16 channels</span>
          </div>
          <div className="muscle-grid" aria-label="Live segment muscle activation">
            {displayedMetrics.muscleActivity.map((value, index) => {
              const level = Math.max(0.06, Math.min(1, Math.abs(value)))
              return (
                <span
                  className={`muscle-cell${value < 0 ? ' is-ventral' : ''}`}
                  key={index}
                  title={`Segment ${index + 1}: ${value.toFixed(2)}`}
                >
                  <i style={{ '--muscle-level': level } as CSSProperties} />
                  <b className="sr-only">Segment {index + 1}: {value.toFixed(2)}</b>
                </span>
              )
            })}
          </div>
        </section>

        <div className="lab-tape" aria-label="Rollout diagnostics">
          <span>RWD <b>{fixed(displayedMetrics.reward, 1)}</b></span>
          <span>
            {displayedMetrics.mounted ? 'RIDE' : 'CRAWL'}{' '}
            <b>{fixed(displayedMetrics.mounted ? displayedMetrics.distance : displayedMetrics.crawlDistance, 1)}m</b>
          </span>
          <span>GRIP <b>{toPercent(displayedMetrics.contactRatio)}%</b></span>
          <span>BODY <b>{fixed(displayedMetrics.bodySpeed, 1)}</b></span>
          <span>T <b>{fixed(displayedMetrics.time, 1)}s</b></span>
        </div>

        <details className="policy-note">
          <summary><Cpu size={14} aria-hidden="true" /> Brain note · {policyStatus.modelVersion}</summary>
          <p>{displayedMetrics.message}</p>
          <code>
            <span>LEARNED · goal sensors + proprioception → 32 segment muscles @ 60 Hz</span>
            <span>DERIVED · joint work, terrain grip, swept tree/rock contacts</span>
            <span>SCRIPTED · mount/feed choreography + aerial kickflip</span>
          </code>
        </details>
      </aside>
    </main>
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
      <span>{icon}<b>{label}</b></span>
      <strong>{percent}%</strong>
      <i aria-hidden="true"><b style={{ width: `${percent}%` }} /></i>
      <span className="sr-only" role="meter" aria-label={`${label} urgency`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent} />
    </div>
  )
}

function Telemetry({ icon, label, value, meter }: TelemetryProps) {
  return (
    <div className="telemetry-card">
      <dt>{icon}{label}</dt>
      <dd>{value}</dd>
      {meter === undefined ? null : (
        <span className="telemetry-meter" aria-hidden="true"><i style={{ width: `${meter}%` }} /></span>
      )}
    </div>
  )
}

function getNeuralStatus(backend: PolicyBackend) {
  if (backend === 'loading') return { label: 'Waking', tone: 'loading' }
  if (backend === 'unavailable') return { label: 'Offline', tone: 'fallback' }
  if (backend === 'neural-js' || backend === 'onnx-webgpu' || backend === 'onnx-wasm') {
    return { label: 'Online', tone: 'online' }
  }
  return { label: 'Fallback', tone: 'fallback' }
}

function formatBackend(backend: PolicyBackend) {
  if (backend === 'neural-js') return 'Evolved recurrent gait / Neural JS'
  if (backend === 'onnx-webgpu') return 'ONNX / WebGPU'
  if (backend === 'onnx-wasm') return 'ONNX / WASM'
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
  if (normalized.includes('flop') || normalized.includes('tumble')) return 'That absolutely counts as research.'
  if (normalized.includes('coil') || normalized.includes('setup')) return 'Coiling the little athlete.'
  return mode === 'kickflip' ? 'Brain linked. Needs online. Believe in wurm.' : 'Clock-free gait under observation.'
}

function formatNeedTarget(target: ViewerMetrics['needTarget']) {
  if (target === 'food-bowl') return 'Food bowl'
  if (target === 'water-bowl') return 'Water bowl'
  if (target === 'skateboard') return 'Skateboard'
  return 'Choosing'
}

function slugify(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'ready'
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
