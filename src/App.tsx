import { Brain, Cpu, Dna, Gauge, Goal, Mountain, Pause, Play, RotateCcw } from 'lucide-react'
import type { ReactNode } from 'react'
import { useMemo, useState } from 'react'
import { useLabConfigs } from './creature/useLabConfigs'
import { PolicyRunner } from './policy/policyRunner'
import type { PolicyStatus, ViewerMetrics } from './policy/types'
import { WurmkickflipScene } from './scene/WurmkickflipScene'

const initialStatus: PolicyStatus = {
  backend: 'loading',
  message: 'Policy runtime is starting.',
  modelVersion: 'scripted-v0',
}

const initialMetrics: ViewerMetrics = {
  time: 0,
  reward: 0,
  distance: 0,
  contactRatio: 0,
  backend: 'loading',
  message: 'Policy runtime is starting.',
  running: true,
  muscleActivity: new Array(16).fill(0),
}

export function App() {
  const policyRunner = useMemo(() => new PolicyRunner(), [])
  const labConfigs = useLabConfigs()
  const [selectedCreatureId, setSelectedCreatureId] = useState<string | null>(null)
  const [selectedEnvironmentId, setSelectedEnvironmentId] = useState<string | null>(null)
  const [running, setRunning] = useState(true)
  const [resetNonce, setResetNonce] = useState(0)
  const [policyStatus, setPolicyStatus] = useState<PolicyStatus>(initialStatus)
  const [metrics, setMetrics] = useState<ViewerMetrics>(initialMetrics)

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

  const displayedMetrics = {
    ...metrics,
    backend: policyStatus.backend,
    message: policyStatus.message,
    running,
  }

  return (
    <main className="app-shell">
      <section className="sim-stage" aria-label="Wurmkickflip terrarium simulation">
        <WurmkickflipScene
          creature={selectedCreature}
          environmentConfig={selectedEnvironment}
          onMetrics={setMetrics}
          onPolicyStatus={setPolicyStatus}
          policyRunner={policyRunner}
          resetNonce={resetNonce}
          running={running}
        />
      </section>

      <aside className="control-panel" aria-label="Training viewer">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Evolutionary skate lab</p>
            <h1>Wurmkickflip</h1>
          </div>
          <div className={`backend-pill backend-${displayedMetrics.backend}`}>
            <Cpu size={16} aria-hidden="true" />
            <span>{displayedMetrics.backend}</span>
          </div>
        </div>

        <div className="actions">
          <button type="button" onClick={() => setRunning((value) => !value)}>
            {running ? <Pause size={18} aria-hidden="true" /> : <Play size={18} aria-hidden="true" />}
            {running ? 'Pause' : 'Run'}
          </button>
          <button
            type="button"
            onClick={() => {
              policyRunner.reset()
              setResetNonce((value) => value + 1)
            }}
          >
            <RotateCcw size={18} aria-hidden="true" />
            Reset
          </button>
        </div>

        <section className="readout">
          <div className="selector-grid">
            <label>
              <span>Creature</span>
              <select
                value={selectedCreature?.id ?? ''}
                onChange={(event) => {
                  setSelectedCreatureId(event.target.value)
                  policyRunner.reset()
                  setResetNonce((value) => value + 1)
                }}
              >
                {labConfigs.creatures.map((creature) => (
                  <option key={creature.id} value={creature.id}>
                    {creature.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>Environment</span>
              <select
                value={selectedEnvironment?.id ?? ''}
                onChange={(event) => {
                  setSelectedEnvironmentId(event.target.value)
                  policyRunner.reset()
                  setResetNonce((value) => value + 1)
                }}
              >
                {labConfigs.environments.map((environment) => (
                  <option key={environment.id} value={environment.id}>
                    {environment.name}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </section>

        <div className="metric-grid">
          <Metric icon={<Gauge size={18} />} label="Reward" value={displayedMetrics.reward.toFixed(2)} />
          <Metric label="Distance" value={`${displayedMetrics.distance.toFixed(2)} m`} />
          <Metric label="Contact" value={`${Math.round(displayedMetrics.contactRatio * 100)}%`} />
          <Metric label="Rollout" value={`${displayedMetrics.time.toFixed(1)} s`} />
        </div>

        <section className="readout">
          <div className="readout-title">
            <Brain size={18} aria-hidden="true" />
            <h2>Muscle activity</h2>
          </div>
          <div className="muscle-grid" aria-label="Segment muscle activity">
            {displayedMetrics.muscleActivity.map((value, index) => (
              <div className="muscle-cell" key={index}>
                <span>{index + 1}</span>
                <div>
                  <i style={{ transform: `scaleY(${Math.max(0.08, Math.abs(value))})` }} />
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="readout">
          <div className="readout-title">
            <Dna size={18} aria-hidden="true" />
            <h2>Creature genome</h2>
          </div>
          <p>{selectedCreature?.name ?? labConfigs.message}</p>
          {selectedCreature ? (
            <div className="compact-grid">
              <span>Parts</span>
              <strong>{selectedCreature.morphology.bodyParts.length}</strong>
              <span>Joints</span>
              <strong>{selectedCreature.morphology.joints.length}</strong>
              <span>Controller</span>
              <strong>{selectedCreature.controller.kind}</strong>
            </div>
          ) : null}
        </section>

        <section className="readout">
          <div className="readout-title">
            <Mountain size={18} aria-hidden="true" />
            <h2>Dynamic environment</h2>
          </div>
          <p>{selectedEnvironment?.name ?? labConfigs.message}</p>
          {selectedEnvironment ? (
            <div className="compact-grid">
              <span>Task</span>
              <strong>{selectedEnvironment.task.kind}</strong>
              <span>Gravity</span>
              <strong>{rangeLabel(selectedEnvironment.randomization.gravityScale)}</strong>
              <span>Friction</span>
              <strong>{rangeLabel(selectedEnvironment.randomization.frictionScale)}</strong>
              <span>Terrain</span>
              <strong>{selectedEnvironment.terrain.kind}</strong>
            </div>
          ) : null}
        </section>

        <section className="readout">
          <div className="readout-title">
            <Goal size={18} aria-hidden="true" />
            <h2>Skateboard task</h2>
          </div>
          {selectedEnvironment ? (
            <div className="compact-grid">
              <span>Spawn X</span>
              <strong>{rangeLabel(selectedEnvironment.randomization.skateboardSpawnX)} m</strong>
              <span>Spawn Z</span>
              <strong>{rangeLabel(selectedEnvironment.randomization.skateboardSpawnZ)} m</strong>
              <span>Mass</span>
              <strong>{rangeLabel(selectedEnvironment.randomization.skateboardMass)} kg</strong>
              <span>Wheel friction</span>
              <strong>{rangeLabel(selectedEnvironment.randomization.wheelFriction)}</strong>
            </div>
          ) : (
            <p>{labConfigs.message}</p>
          )}
        </section>

        <section className="readout">
          <h2>Policy status</h2>
          <p>{displayedMetrics.message}</p>
          <p>
            Contract: 118 observation floats to 32 dorsal/ventral muscle activations at 60 Hz.
          </p>
        </section>
      </aside>
    </main>
  )
}

function rangeLabel(range: [number, number]) {
  return `${range[0]}-${range[1]}`
}

type MetricProps = {
  icon?: ReactNode
  label: string
  value: string
}

function Metric({ icon, label, value }: MetricProps) {
  return (
    <div className="metric-card">
      <span>
        {icon}
        {label}
      </span>
      <strong>{value}</strong>
    </div>
  )
}
