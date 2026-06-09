import { Brain, Cpu, Gauge, Pause, Play, RotateCcw } from 'lucide-react'
import type { ReactNode } from 'react'
import { useMemo, useState } from 'react'
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
  const [running, setRunning] = useState(true)
  const [resetNonce, setResetNonce] = useState(0)
  const [policyStatus, setPolicyStatus] = useState<PolicyStatus>(initialStatus)
  const [metrics, setMetrics] = useState<ViewerMetrics>(initialMetrics)

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
            <p className="eyebrow">OpenWorm-inspired RL prototype</p>
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
