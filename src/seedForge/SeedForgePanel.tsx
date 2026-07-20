import { Dices, LockKeyhole, LockKeyholeOpen, RotateCcw, SlidersHorizontal, Sparkles } from 'lucide-react'
import { type FormEvent, useId } from 'react'
import type { EnvironmentConfig } from '../creature/types'
import {
  DOMAIN_PARAMETER_KEYS,
  SEED_FORGE_PARAMETER_SPECS,
  SEED_FORGE_PRESETS,
  type DomainLocks,
  type DomainParameterKey,
  type DomainSample,
  type SeedForgePresetId,
} from '../environment/seedForge'
import './SeedForgePanel.css'

const UINT32_MAX = 0xffff_ffff

type ParameterGroupId = 'world' | 'wurm' | 'board'

const PARAMETER_GROUPS: ReadonlyArray<{
  id: ParameterGroupId
  label: string
  note: string
}> = [
  { id: 'world', label: 'World + ground', note: '6 channels' },
  { id: 'wurm', label: 'Wurm signals', note: '4 channels' },
  { id: 'board', label: 'Skateboard', note: '4 channels' },
]

export type SeedForgePanelProps = {
  sample: DomainSample
  draftSeed: string
  ranges: EnvironmentConfig['randomization']
  locks: DomainLocks
  activePreset?: SeedForgePresetId | null
  disabled?: boolean
  onDraftSeedChange: (draftSeed: string) => void
  onApplySeed: (seed: number) => void
  onReroll: () => void
  onResetNominal: () => void
  onPreset: (presetId: SeedForgePresetId) => void
  onLockChange: (key: DomainParameterKey, locked: boolean) => void
}

export function SeedForgePanel({
  sample,
  draftSeed,
  ranges,
  locks,
  activePreset = null,
  disabled = false,
  onDraftSeedChange,
  onApplySeed,
  onReroll,
  onResetNominal,
  onPreset,
  onLockChange,
}: SeedForgePanelProps) {
  const headingId = useId()
  const seedInputId = useId()
  const seedHelpId = useId()
  const parsedSeed = parseUint32(draftSeed)
  const seedIsValid = parsedSeed !== null

  const applyDraftSeed = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (parsedSeed !== null && !disabled) onApplySeed(parsedSeed)
  }

  return (
    <section className="seed-forge" aria-labelledby={headingId}>
      <header className="seed-forge__header">
        <span className="seed-forge__mark" aria-hidden="true">
          <Dices size={18} strokeWidth={2.35} />
        </span>
        <div className="seed-forge__title">
          <p className="seed-forge__eyebrow">Deterministic domain laboratory</p>
          <h2 id={headingId}>Seed Forge</h2>
          <p>
            {disabled
              ? 'Inspecting the immutable domain captured with this replay.'
              : 'Recast the terrarium, then replay the exact same challenge.'}
          </p>
        </div>
        <div className="seed-forge__sample-badge">
          <span>{disabled ? 'Replay sample' : 'Live sample'}</span>
          <output aria-label={`Current domain seed ${sample.seed}`} aria-live="polite">
            #{sample.seed}
          </output>
        </div>
      </header>

      {disabled ? (
        <p className="seed-forge__mode-note" role="status">
          Replay provenance is locked. Return to live simulation to forge another domain.
        </p>
      ) : null}

      <form className="seed-forge__seed-form" noValidate onSubmit={applyDraftSeed}>
        <label className="seed-forge__seed-field" htmlFor={seedInputId}>
          <span>uint32 seed</span>
          <input
            className="seed-forge__seed-input"
            id={seedInputId}
            type="text"
            inputMode="numeric"
            autoComplete="off"
            spellCheck={false}
            pattern="[0-9]*"
            maxLength={10}
            value={draftSeed}
            disabled={disabled}
            aria-describedby={seedHelpId}
            aria-errormessage={seedIsValid ? undefined : seedHelpId}
            aria-invalid={!seedIsValid}
            onChange={event => onDraftSeedChange(event.target.value)}
          />
        </label>
        <button
          className="seed-forge__button seed-forge__button--apply"
          type="submit"
          disabled={disabled || !seedIsValid}
        >
          Apply seed
        </button>
        <button
          className="seed-forge__button"
          type="button"
          disabled={disabled}
          title="Choose the next deterministic seed and resample unlocked values"
          onClick={onReroll}
        >
          <Dices size={13} aria-hidden="true" />
          Reroll
        </button>
        <p
          className="seed-forge__seed-help"
          id={seedHelpId}
          data-invalid={!seedIsValid}
          role={seedIsValid ? undefined : 'alert'}
          aria-live="polite"
          aria-atomic="true"
        >
          {seedIsValid
            ? 'Any whole number from 0 to 4,294,967,295 can be replayed exactly.'
            : 'Enter a whole number from 0 to 4,294,967,295.'}
        </p>
      </form>

      <section className="seed-forge__section" aria-labelledby={`${headingId}-presets`}>
        <div className="seed-forge__section-heading">
          <Sparkles size={13} aria-hidden="true" />
          <h3 id={`${headingId}-presets`}>Challenge presets</h3>
          <span>Hand-tuned domains</span>
        </div>
        <div className="seed-forge__presets">
          {SEED_FORGE_PRESETS.map(preset => (
            <button
              className="seed-forge__preset"
              key={preset.id}
              type="button"
              data-preset={preset.id}
              disabled={disabled}
              aria-pressed={activePreset === preset.id}
              onClick={() => onPreset(preset.id)}
            >
              <strong>{preset.label}</strong>
              <small>{preset.description}</small>
            </button>
          ))}
        </div>
      </section>

      <section className="seed-forge__section" aria-labelledby={`${headingId}-parameters`}>
        <div className="seed-forge__section-heading">
          <SlidersHorizontal size={13} aria-hidden="true" />
          <h3 id={`${headingId}-parameters`}>Forged parameters</h3>
          <span>{DOMAIN_PARAMETER_KEYS.length} sampled channels</span>
        </div>
        <div className="seed-forge__groups">
          {PARAMETER_GROUPS.map(group => (
            <section
              className="seed-forge__group"
              key={group.id}
              aria-labelledby={`${headingId}-${group.id}`}
            >
              <div className="seed-forge__group-heading">
                <h4 id={`${headingId}-${group.id}`}>{group.label}</h4>
                <span>{group.note}</span>
              </div>
              <div className="seed-forge__parameters">
                {DOMAIN_PARAMETER_KEYS.filter(key => parameterGroupFor(key) === group.id).map(key => {
                  const spec = parameterSpecFor(key)
                  const range = ranges[key]
                  const locked = locks[key]
                  const value = sample[key]
                  const displayValue = formatValue(value, spec.decimals, spec.unit, spec.displayScale)
                  const displayRange = formatRange(range, spec.decimals, spec.unit, spec.displayScale)

                  return (
                    <div className="seed-forge__parameter" key={key} data-locked={locked}>
                      <span className="seed-forge__parameter-label" title={spec.label}>
                        {spec.shortLabel}
                      </span>
                      <output aria-label={`${spec.label}: ${displayValue}`}>{displayValue}</output>
                      <span className="seed-forge__range">range {displayRange}</span>
                      <button
                        className="seed-forge__lock"
                        type="button"
                        disabled={disabled}
                        aria-label={`${spec.label} lock. Current value ${displayValue}.`}
                        aria-pressed={locked}
                        title={`${locked ? 'Unlock' : 'Lock'} ${spec.label}`}
                        onClick={() => onLockChange(key, !locked)}
                      >
                        {locked ? (
                          <LockKeyhole size={12} aria-hidden="true" />
                        ) : (
                          <LockKeyholeOpen size={12} aria-hidden="true" />
                        )}
                      </button>
                    </div>
                  )
                })}
              </div>
            </section>
          ))}
        </div>
      </section>

      <footer className="seed-forge__footer">
        <p className="seed-forge__lock-note">
          <LockKeyhole size={11} aria-hidden="true" />
          <span>
            <strong>Locks persist across rerolls.</strong> Unlock a value to let the next seed recast it.
          </span>
        </p>
        <button
          className="seed-forge__button seed-forge__button--reset"
          type="button"
          disabled={disabled}
          onClick={onResetNominal}
        >
          <RotateCcw size={12} aria-hidden="true" />
          Restore authored baseline
        </button>
      </footer>
    </section>
  )
}

function parseUint32(draftSeed: string) {
  const normalized = draftSeed.trim()
  if (!/^\d+$/.test(normalized)) return null
  const parsed = Number(normalized)
  if (!Number.isSafeInteger(parsed) || parsed < 0 || parsed > UINT32_MAX) return null
  return parsed
}

function parameterSpecFor(key: DomainParameterKey) {
  const spec = SEED_FORGE_PARAMETER_SPECS.find(candidate => candidate.key === key)
  if (!spec) throw new Error(`Seed Forge display metadata is missing for ${key}.`)
  return spec
}

function parameterGroupFor(key: DomainParameterKey): ParameterGroupId {
  switch (key) {
    case 'gravityScale':
    case 'frictionScale':
    case 'dragScale':
    case 'slopeDegrees':
    case 'roughness':
    case 'obstacleDensity':
      return 'world'
    case 'actuatorStrength':
    case 'actuatorLatencyMs':
    case 'sensorNoise':
    case 'spawnYawDegrees':
      return 'wurm'
    case 'skateboardSpawnX':
    case 'skateboardSpawnZ':
    case 'skateboardMass':
    case 'wheelFriction':
      return 'board'
  }
}

function formatValue(value: number, decimals: number, unit: string, displayScale = 1) {
  return `${formatNumber(value * displayScale, decimals)}${formatUnit(unit)}`
}

function formatRange(range: readonly [number, number], decimals: number, unit: string, displayScale = 1) {
  return `${formatNumber(range[0] * displayScale, decimals)}–${formatNumber(
    range[1] * displayScale,
    decimals,
  )}${formatUnit(unit)}`
}

function formatUnit(unit: string) {
  const normalized = unit.trim()
  return normalized ? ` ${normalized}` : ''
}

function formatNumber(value: number, decimals: number) {
  const safeValue = Object.is(value, -0) ? 0 : value
  return safeValue.toFixed(decimals)
}
