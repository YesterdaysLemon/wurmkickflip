import { REPLAY_INTEGRITY_ALGORITHM, type ReplayIntegrity } from './types'

const FNV_OFFSET_BASIS_64 = 0xcbf29ce484222325n
const FNV_PRIME_64 = 0x100000001b3n
const UINT64_MASK = 0xffffffffffffffffn

/**
 * Returns the deterministic checksum used by recorder-core artifacts. The
 * checksum covers the complete JSON-compatible payload except the top-level
 * integrity object itself, including unknown extension fields.
 *
 * FNV-1a is an accidental-corruption and tamper-detection checksum, not a
 * signature. Replays from an untrusted party still require an authenticated
 * transport or a separately signed manifest.
 */
export function computeReplayDigest(value: unknown): string {
  const canonical = canonicalReplayPayload(value)
  let hash = FNV_OFFSET_BASIS_64
  for (const byte of new TextEncoder().encode(canonical)) {
    hash ^= BigInt(byte)
    hash = (hash * FNV_PRIME_64) & UINT64_MASK
  }
  return hash.toString(16).padStart(16, '0')
}

export function replayIntegrityFor(value: unknown): ReplayIntegrity {
  return {
    algorithm: REPLAY_INTEGRITY_ALGORITHM,
    digest: computeReplayDigest(value),
  }
}

export function replayIntegrityMatches(value: unknown, integrity: ReplayIntegrity): boolean {
  return integrity.algorithm === REPLAY_INTEGRITY_ALGORITHM && computeReplayDigest(value) === integrity.digest
}

function canonicalReplayPayload(value: unknown): string {
  if (!isRecord(value)) {
    throw new Error('Replay integrity requires a top-level object.')
  }
  const payload: Record<string, unknown> = {}
  for (const [key, item] of Object.entries(value)) {
    if (key !== 'integrity') payload[key] = item
  }
  return canonicalJson(payload, new Set<object>())
}

function canonicalJson(value: unknown, ancestors: Set<object>): string {
  if (value === null) return 'null'
  if (typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value)
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error('Replay integrity cannot encode non-finite numbers.')
    return JSON.stringify(Object.is(value, -0) ? 0 : value)
  }
  if (Array.isArray(value)) {
    guardCycle(value, ancestors)
    const encoded = `[${value.map(item => canonicalJson(item, ancestors)).join(',')}]`
    ancestors.delete(value)
    return encoded
  }
  if (isRecord(value)) {
    guardCycle(value, ancestors)
    const encoded = `{${Object.keys(value)
      .sort()
      .map(key => `${JSON.stringify(key)}:${canonicalJson(value[key], ancestors)}`)
      .join(',')}}`
    ancestors.delete(value)
    return encoded
  }
  throw new Error(`Replay integrity cannot encode ${typeof value} values.`)
}

function guardCycle(value: object, ancestors: Set<object>) {
  if (ancestors.has(value)) throw new Error('Replay integrity cannot encode cyclic values.')
  ancestors.add(value)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
