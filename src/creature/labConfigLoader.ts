import {
  validateCreatureGenome,
  validateEnvironmentConfig,
  type ConfigValidationResult,
} from './configValidation'
import type { CreatureGenome, EnvironmentConfig } from './types'

export const DEFAULT_LAB_CONFIG_PATHS = {
  creatures: [
    '/configs/creatures/segmented-starter.json',
    '/configs/creatures/tripod-pusher.json',
    '/configs/creatures/boxfish-scrambler.json',
  ],
  environments: [
    '/configs/environments/adaptive-skate-terrarium.json',
    '/configs/environments/ripple-yard.json',
    '/configs/environments/tilt-basin.json',
  ],
  generatedCreatureManifest: '/configs/evolved/manifest.json',
} as const

export type LabConfigPaths = {
  creatures: readonly string[]
  environments: readonly string[]
  generatedCreatureManifest: string
}

export type LabConfigResourceKind = 'creature' | 'environment' | 'generated-creature-manifest'
export type LabConfigResourceState = 'loaded' | 'error' | 'optional-missing'

export type LabConfigResourceStatus = {
  path: string
  kind: LabConfigResourceKind
  required: boolean
  state: LabConfigResourceState
  errors: string[]
}

export type LabConfigFetchResult =
  { ok: true; value: unknown } | { ok: false; status: number | null; error: string }

export type LabConfigFetcher = (path: string) => Promise<LabConfigFetchResult>

export type LabConfigLoadResult = {
  creatures: CreatureGenome[]
  environments: EnvironmentConfig[]
  resources: LabConfigResourceStatus[]
  status: 'ready' | 'partial' | 'error'
  message: string
}

type LoadedResource<T> = {
  resource: LabConfigResourceStatus
  value: T | null
}

export async function loadLabConfigResources(
  fetcher: LabConfigFetcher,
  paths: LabConfigPaths = DEFAULT_LAB_CONFIG_PATHS,
): Promise<LabConfigLoadResult> {
  const initial = await Promise.all([
    ...paths.creatures.map(path => loadValidated(path, 'creature', true, fetcher, validateCreatureGenome)),
    ...paths.environments.map(path =>
      loadValidated(path, 'environment', true, fetcher, validateEnvironmentConfig),
    ),
    loadGeneratedManifest(paths.generatedCreatureManifest, fetcher),
  ])

  const builtInCreatureCount = paths.creatures.length
  const environmentStart = builtInCreatureCount
  const manifestIndex = builtInCreatureCount + paths.environments.length
  const builtInCreatures = initial.slice(0, builtInCreatureCount) as LoadedResource<CreatureGenome>[]
  const environments = initial.slice(environmentStart, manifestIndex) as LoadedResource<EnvironmentConfig>[]
  const manifest = initial[manifestIndex] as LoadedResource<string[]>
  const generatedCreatures = manifest.value
    ? await Promise.all(
        manifest.value.map(path => loadValidated(path, 'creature', false, fetcher, validateCreatureGenome)),
      )
    : []

  const uniqueCreatures = keepUniqueIds([...builtInCreatures, ...generatedCreatures], 'creature')
  const uniqueEnvironments = keepUniqueIds(environments, 'environment')
  const resources = [
    ...uniqueCreatures.map(entry => entry.resource),
    ...uniqueEnvironments.map(entry => entry.resource),
    manifest.resource,
  ]
  const creatureValues = uniqueCreatures.flatMap(entry => (entry.value ? [entry.value] : []))
  const environmentValues = uniqueEnvironments.flatMap(entry => (entry.value ? [entry.value] : []))
  const failures = resources.filter(resource => resource.state === 'error')
  const hasRunnablePair = creatureValues.length > 0 && environmentValues.length > 0
  const status: LabConfigLoadResult['status'] = !hasRunnablePair
    ? 'error'
    : failures.length > 0
      ? 'partial'
      : 'ready'
  const loadedMessage = `Loaded ${creatureValues.length} creature${creatureValues.length === 1 ? '' : 's'} and ${environmentValues.length} environment${environmentValues.length === 1 ? '' : 's'}.`
  const message =
    status === 'ready'
      ? loadedMessage
      : status === 'partial'
        ? `${loadedMessage} ${failures.length} resource${failures.length === 1 ? '' : 's'} failed; valid selections remain available.`
        : `${loadedMessage} A valid creature and environment are required; review the resource errors and retry.`

  return {
    creatures: creatureValues,
    environments: environmentValues,
    resources,
    status,
    message,
  }
}

export async function browserLabConfigFetcher(path: string): Promise<LabConfigFetchResult> {
  try {
    const response = await fetch(path)
    if (!response.ok) {
      return { ok: false, status: response.status, error: `Request returned HTTP ${response.status}.` }
    }
    try {
      return { ok: true, value: (await response.json()) as unknown }
    } catch (error) {
      return {
        ok: false,
        status: response.status,
        error:
          error instanceof Error
            ? `Response was not valid JSON: ${error.message}`
            : 'Response was not valid JSON.',
      }
    }
  } catch (error) {
    return {
      ok: false,
      status: null,
      error: error instanceof Error ? `Network request failed: ${error.message}` : 'Network request failed.',
    }
  }
}

async function loadValidated<T>(
  path: string,
  kind: Exclude<LabConfigResourceKind, 'generated-creature-manifest'>,
  required: boolean,
  fetcher: LabConfigFetcher,
  validator: (value: unknown) => ConfigValidationResult<T>,
): Promise<LoadedResource<T>> {
  const fetched = await safelyFetch(fetcher, path)
  if (!fetched.ok) {
    return failedResource(path, kind, required, fetchError(fetched))
  }
  const validation = validator(fetched.value)
  if (!validation.ok) return failedResource(path, kind, required, validation.errors)
  return {
    resource: { path, kind, required, state: 'loaded', errors: [] },
    value: validation.value,
  }
}

async function loadGeneratedManifest(
  path: string,
  fetcher: LabConfigFetcher,
): Promise<LoadedResource<string[]>> {
  const fetched = await safelyFetch(fetcher, path)
  if (!fetched.ok) {
    if (fetched.status === 404) {
      return {
        resource: {
          path,
          kind: 'generated-creature-manifest',
          required: false,
          state: 'optional-missing',
          errors: [],
        },
        value: null,
      }
    }
    return failedResource(path, 'generated-creature-manifest', false, fetchError(fetched))
  }

  const validation = validateGeneratedManifest(fetched.value)
  if (!validation.ok) {
    return failedResource(path, 'generated-creature-manifest', false, validation.errors)
  }
  return {
    resource: {
      path,
      kind: 'generated-creature-manifest',
      required: false,
      state: 'loaded',
      errors: [],
    },
    value: validation.value,
  }
}

function validateGeneratedManifest(value: unknown): ConfigValidationResult<string[]> {
  const errors: string[] = []
  if (!isRecord(value)) return { ok: false, errors: ['generated manifest must be an object.'] }
  if (value.kind !== 'wurmkickflip.generatedCreatureManifest') {
    errors.push('generated manifest kind must equal wurmkickflip.generatedCreatureManifest.')
  }
  if (!Array.isArray(value.creatures)) {
    errors.push('generated manifest creatures must be an array.')
    return { ok: false, errors }
  }
  const paths: string[] = []
  const seen = new Set<string>()
  for (let index = 0; index < value.creatures.length; index += 1) {
    const path = value.creatures[index]
    if (
      typeof path !== 'string' ||
      !path.startsWith('/configs/evolved/') ||
      !path.endsWith('.json') ||
      path.includes('..') ||
      path.includes('\\')
    ) {
      errors.push(`generated manifest creatures[${index}] must be a safe /configs/evolved/*.json path.`)
      continue
    }
    if (seen.has(path)) {
      errors.push(`generated manifest creature path ${path} must be unique.`)
      continue
    }
    seen.add(path)
    paths.push(path)
  }
  return errors.length > 0 ? { ok: false, errors } : { ok: true, value: paths }
}

function keepUniqueIds<T extends { id: string }>(
  entries: LoadedResource<T>[],
  label: 'creature' | 'environment',
): LoadedResource<T>[] {
  const seen = new Map<string, string>()
  for (const entry of entries) {
    if (!entry.value) continue
    const firstPath = seen.get(entry.value.id)
    if (firstPath) {
      entry.resource.state = 'error'
      entry.resource.errors.push(`${label} id ${entry.value.id} duplicates ${firstPath}.`)
      entry.value = null
    } else {
      seen.set(entry.value.id, entry.resource.path)
    }
  }
  return entries
}

async function safelyFetch(fetcher: LabConfigFetcher, path: string): Promise<LabConfigFetchResult> {
  try {
    return await fetcher(path)
  } catch (error) {
    return {
      ok: false,
      status: null,
      error: error instanceof Error ? `Fetcher rejected: ${error.message}` : 'Fetcher rejected.',
    }
  }
}

function failedResource<T>(
  path: string,
  kind: LabConfigResourceKind,
  required: boolean,
  errors: string | string[],
): LoadedResource<T> {
  return {
    resource: {
      path,
      kind,
      required,
      state: 'error',
      errors: Array.isArray(errors) ? [...errors] : [errors],
    },
    value: null,
  }
}

function fetchError(result: Extract<LabConfigFetchResult, { ok: false }>) {
  return `${result.error}${result.status === null ? '' : ` (HTTP ${result.status})`}`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
