import { AlertTriangle, FlaskConical, Play, Trophy, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { CreatureGenome, EnvironmentConfig } from '../creature/types'
import type { RecordedReplayArtifact } from '../replay'
import type { ForgeTrialHeat, ForgeTrialsProgress, ForgeTrialsReport } from './forgeTrials'
import './ForgeTrialsPanel.css'

type ForgeTrialsPanelProps = {
  environment: EnvironmentConfig
  creature: CreatureGenome | null
  baseSeed: number
  disabled?: boolean
  onLoadReplay: (artifact: RecordedReplayArtifact) => void
  onRunStart?: () => void
}

type RunState = 'idle' | 'loading' | 'running' | 'complete' | 'error' | 'cancelled'

const trialCountOptions = [8, 16, 24, 32] as const

export function ForgeTrialsPanel({
  environment,
  creature,
  baseSeed,
  disabled = false,
  onLoadReplay,
  onRunStart,
}: ForgeTrialsPanelProps) {
  const [trialCount, setTrialCount] = useState(16)
  const [runState, setRunState] = useState<RunState>('idle')
  const [progress, setProgress] = useState<ForgeTrialsProgress | null>(null)
  const [report, setReport] = useState<ForgeTrialsReport | null>(null)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(
    () => () => {
      abortRef.current?.abort()
    },
    [],
  )

  const running = runState === 'loading' || runState === 'running'
  const progressValue = progress ? progress.completedLanes / progress.totalLanes : 0
  const statusMessage = statusFor(runState, progress, report)

  const runMeet = async () => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setRunState('loading')
    setProgress(null)
    setError(null)
    onRunStart?.()

    try {
      const trials = await import('./forgeTrials')
      const models = await trials.loadForgeTrialModels(controller.signal)
      if (controller.signal.aborted) return
      setRunState('running')
      const nextReport = await trials.runForgeTrials({
        environment,
        creature,
        models,
        baseSeed,
        trialCount,
        signal: controller.signal,
        onProgress: setProgress,
      })
      if (controller.signal.aborted) return
      setReport(nextReport)
      setRunState('complete')
    } catch (caught) {
      if (
        controller.signal.aborted ||
        (caught instanceof Error && caught.name === 'ForgeTrialsAbortedError')
      ) {
        setRunState('cancelled')
      } else {
        setRunState('error')
        setError(caught instanceof Error ? caught.message : 'Forge Trials failed unexpectedly.')
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null
    }
  }

  const cancelMeet = () => {
    abortRef.current?.abort()
    setRunState('cancelled')
  }

  return (
    <details className="forge-trials">
      <summary>
        <Trophy size={15} aria-hidden="true" />
        <span>Wurm Olympics</span>
        <b>{report ? `${report.summary.neuralMounts}/${report.summary.heatCount} mounts` : 'Forge Trials'}</b>
      </summary>
      <section className="forge-trials__body" aria-label="Wurm Olympics">
        <div className="forge-trials__intro">
          <span className="forge-trials__eyebrow">
            <FlaskConical size={14} aria-hidden="true" />
            Held-out deterministic meet
          </span>
          <h3>Can this worm travel?</h3>
          <p>
            Run the same forged worlds with the recurrent controller, zero muscle commands, and a post-warmup
            frozen controller. A causal win means only the changing neural lane reaches a stable ride. Seeds
            exclude the active attempt; historical training-set exclusion is not claimed.
          </p>
        </div>

        <div className="forge-trials__setup">
          <label>
            <span>Meet size</span>
            <select
              aria-label="Forge Trial meet size"
              disabled={disabled || running}
              value={trialCount}
              onChange={event => setTrialCount(Number(event.target.value))}
            >
              {trialCountOptions.map(count => (
                <option key={count} value={count}>
                  {count} seeds / {count * 3} lanes
                </option>
              ))}
            </select>
          </label>
          <div className="forge-trials__seed">
            <span>After live seed</span>
            <b>#{baseSeed}</b>
          </div>
        </div>

        <div className="forge-trials__actions">
          {running ? (
            <button className="forge-trials__cancel" type="button" onClick={cancelMeet}>
              <X size={15} aria-hidden="true" />
              Cancel meet
            </button>
          ) : (
            <button
              className="forge-trials__run"
              type="button"
              disabled={disabled}
              onClick={() => void runMeet()}
            >
              <Play size={15} aria-hidden="true" />
              Run {trialCount}-seed meet
            </button>
          )}
          <p className="forge-trials__status" role="status" aria-live="polite">
            {statusMessage}
          </p>
        </div>

        {running ? (
          <div
            className="forge-trials__progress"
            role="progressbar"
            aria-label="Forge Trials progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(progressValue * 100)}
          >
            <span style={{ width: `${Math.max(2, progressValue * 100)}%` }} />
          </div>
        ) : null}

        {error ? (
          <div className="forge-trials__error" role="alert">
            <AlertTriangle size={15} aria-hidden="true" />
            <span>{error}</span>
          </div>
        ) : null}

        {report ? (
          <>
            <ForgeTrialsSummary report={report} />
            <div className="forge-trials__heats" aria-label="Forge Trial heat results">
              {report.heats.map(heat => (
                <ForgeTrialHeatCard
                  disabled={disabled}
                  heat={heat}
                  key={heat.seed}
                  onLoadReplay={onLoadReplay}
                />
              ))}
            </div>
            <p className="forge-trials__honesty">
              These are deterministic compact-plant evaluations, not real-world transfer evidence. Replay
              schema v1 preserves the domain, timing, root/board poses, contact, and all 32 muscle channels;
              individual segment poses are reconstructed for viewing.
            </p>
          </>
        ) : null}
      </section>
    </details>
  )
}

function ForgeTrialsSummary({ report }: { report: ForgeTrialsReport }) {
  const { summary } = report
  return (
    <div className="forge-trials__summary" aria-label="Forge Trials summary">
      <div>
        <span>Neural mounts</span>
        <strong>
          {summary.neuralMounts}/{summary.heatCount}
        </strong>
      </div>
      <div>
        <span>Causal wins</span>
        <strong>{summary.causalWins}</strong>
      </div>
      <div>
        <span>Control mounts</span>
        <strong>{summary.zeroMounts + summary.frozenMounts}</strong>
      </div>
      <div>
        <span>Median ride</span>
        <strong>{summary.medianRideTime === null ? '—' : `${summary.medianRideTime.toFixed(2)}s`}</strong>
      </div>
    </div>
  )
}

function ForgeTrialHeatCard({
  heat,
  disabled,
  onLoadReplay,
}: {
  heat: ForgeTrialHeat
  disabled: boolean
  onLoadReplay: (artifact: RecordedReplayArtifact) => void
}) {
  const replay = heat.full.replay
  return (
    <article className={`forge-trials__heat is-${heat.verdict}`}>
      <header>
        <div>
          <span>Heat {String(heat.index + 1).padStart(2, '0')}</span>
          <strong>#{heat.seed}</strong>
        </div>
        <b>{verdictLabel(heat.verdict)}</b>
      </header>
      <div className="forge-trials__lanes">
        <LaneResult label="Neural" mounted={heat.full.mounted} rideTime={heat.full.rideTime} />
        <LaneResult label="Zero" mounted={heat.zero.mounted} rideTime={heat.zero.rideTime} />
        <LaneResult label="Frozen" mounted={heat.frozen.mounted} rideTime={heat.frozen.rideTime} />
      </div>
      <dl>
        <div>
          <dt>Closest</dt>
          <dd>{heat.full.minimumDistanceToBoard.toFixed(2)} m</dd>
        </div>
        <div>
          <dt>Contact</dt>
          <dd>{Math.round(heat.full.maximumContactRatio * 100)}%</dd>
        </div>
        <div>
          <dt>Latency</dt>
          <dd>{Math.round(heat.sample.actuatorLatencyMs)} ms</dd>
        </div>
        <div>
          <dt>Grip</dt>
          <dd>{heat.sample.wheelFriction.toFixed(2)}</dd>
        </div>
      </dl>
      <footer>
        <code>{heat.fingerprint}</code>
        <button
          type="button"
          disabled={disabled || replay === null}
          onClick={() => {
            if (replay) onLoadReplay(replay)
          }}
        >
          <Play size={13} aria-hidden="true" />
          Inspect replay
        </button>
      </footer>
    </article>
  )
}

function LaneResult({
  label,
  mounted,
  rideTime,
}: {
  label: string
  mounted: boolean
  rideTime: number | null
}) {
  return (
    <div className={mounted ? 'did-mount' : 'did-miss'}>
      <span>{label}</span>
      <b>{mounted && rideTime !== null ? `${rideTime.toFixed(2)}s` : 'no ride'}</b>
    </div>
  )
}

function verdictLabel(verdict: ForgeTrialHeat['verdict']) {
  if (verdict === 'causal-win') return 'Causal win'
  if (verdict === 'contested') return 'Control mounted'
  return 'Neural miss'
}

function statusFor(state: RunState, progress: ForgeTrialsProgress | null, report: ForgeTrialsReport | null) {
  if (state === 'loading') return 'Loading the tracked controllers…'
  if (state === 'running' && progress) {
    return `Lane ${progress.completedLanes}/${progress.totalLanes} · heat ${progress.heatIndex + 1} · ${progress.intervention}`
  }
  if (state === 'complete' && report) {
    return `${report.summary.causalWins} clean causal wins · ${report.summary.misses} neural misses · ${report.summary.contested} contested`
  }
  if (state === 'cancelled') return 'Meet cancelled. Completed results were discarded.'
  if (state === 'error') return 'Meet stopped before results were published.'
  return 'Eight simulated seconds per lane. The current live seed is never included.'
}
