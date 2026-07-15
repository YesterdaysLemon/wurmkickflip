import { ReplayPlayer } from './replayPlayer'
import type { ReplayArtifact } from './types'

/** Validates recorder-core integrity before returning stable, human-readable JSON. */
export function serializeReplayArtifact(value: unknown): string {
  const player = new ReplayPlayer(value)
  return `${JSON.stringify(sortJsonKeys(player.artifact()), null, 2)}\n`
}

/** Parse and fully validate imported JSON without mutating any active player. */
export function replayPlayerFromJson(text: string): ReplayPlayer {
  let value: unknown
  try {
    value = JSON.parse(text)
  } catch (error) {
    const detail = error instanceof Error ? error.message : 'The file is not valid JSON.'
    throw new Error(`Replay JSON could not be parsed: ${detail}`, { cause: error })
  }
  return new ReplayPlayer(value)
}

/**
 * Starts a browser download and always revokes the temporary object URL. The
 * click is dispatched synchronously before cleanup, so repeated exports do not
 * retain blobs for the lifetime of the app.
 */
export function downloadReplayArtifact(artifact: ReplayArtifact, filename: string) {
  const contents = serializeReplayArtifact(artifact)
  const objectUrl = URL.createObjectURL(new Blob([contents], { type: 'application/json' }))
  const link = document.createElement('a')
  link.download = safeReplayFilename(filename)
  link.href = objectUrl
  link.hidden = true
  document.body.append(link)
  try {
    link.click()
  } finally {
    link.remove()
    URL.revokeObjectURL(objectUrl)
  }
}

function sortJsonKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortJsonKeys)
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, sortJsonKeys(item)]),
    )
  }
  return value
}

function safeReplayFilename(value: string) {
  const stem = value
    .replace(/\.json$/i, '')
    .replace(/[^a-z0-9_-]+/gi, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase()
  return `${stem || 'wurmkickflip-replay'}.json`
}
