import { useEffect, useState } from 'react'
import { validateCreatureGenome, validateEnvironmentConfig } from './configValidation'
import type { CreatureGenome, EnvironmentConfig } from './types'

const creaturePaths = [
  '/configs/creatures/segmented-starter.json',
  '/configs/creatures/tripod-pusher.json',
  '/configs/creatures/boxfish-scrambler.json',
]
const generatedCreatureManifestPath = '/configs/evolved/manifest.json'

const environmentPaths = [
  '/configs/environments/adaptive-skate-terrarium.json',
  '/configs/environments/ripple-yard.json',
  '/configs/environments/tilt-basin.json',
]

export type LabConfigs = {
  creature: CreatureGenome | null
  creatures: CreatureGenome[]
  environment: EnvironmentConfig | null
  environments: EnvironmentConfig[]
  status: 'loading' | 'ready' | 'error'
  message: string
}

export function useLabConfigs(): LabConfigs {
  const [configs, setConfigs] = useState<LabConfigs>({
    creature: null,
    creatures: [],
    environment: null,
    environments: [],
    status: 'loading',
    message: 'Loading lab configs.',
  })

  useEffect(() => {
    let mounted = true

    Promise.all([
      loadCreatureConfigs(),
      Promise.all(environmentPaths.map(fetchJson)),
    ])
      .then(([creatureValues, environmentValues]) => {
        const creatures: CreatureGenome[] = []
        const environments: EnvironmentConfig[] = []
        const errors: string[] = []

        for (const creature of creatureValues) {
          const validation = validateCreatureGenome(creature)
          if (validation.ok) {
            creatures.push(validation.value)
          } else {
            errors.push(...validation.errors)
          }
        }

        for (const environment of environmentValues) {
          const validation = validateEnvironmentConfig(environment)
          if (validation.ok) {
            environments.push(validation.value)
          } else {
            errors.push(...validation.errors)
          }
        }

        if (errors.length > 0) {
          throw new Error(`Invalid lab config: ${errors.slice(0, 3).join(' ')}`)
        }

        if (mounted) {
          setConfigs({
            creature: creatures[0] ?? null,
            creatures,
            environment: environments[0] ?? null,
            environments,
            status: 'ready',
            message: `Loaded ${creatures.length} creatures and ${environments.length} environments.`,
          })
        }
      })
      .catch((error) => {
        if (mounted) {
          setConfigs({
            creature: null,
            creatures: [],
            environment: null,
            environments: [],
            status: 'error',
            message: error instanceof Error ? error.message : 'Failed to load lab configs.',
          })
        }
      })

    return () => {
      mounted = false
    }
  }, [])

  return configs
}

async function loadCreatureConfigs(): Promise<unknown[]> {
  const builtIns = await Promise.all(creaturePaths.map(fetchJson))
  const generatedPaths = await fetchGeneratedCreaturePaths()
  const generated = await Promise.all(generatedPaths.map(fetchJson))
  return [...builtIns, ...generated]
}

async function fetchGeneratedCreaturePaths(): Promise<string[]> {
  const manifest = await fetchOptionalJson(generatedCreatureManifestPath)
  if (!isRecord(manifest) || manifest.kind !== 'wurmkickflip.generatedCreatureManifest') {
    return []
  }
  if (!Array.isArray(manifest.creatures)) {
    return []
  }
  return manifest.creatures.filter((path): path is string => typeof path === 'string' && path.startsWith('/configs/'))
}

async function fetchJson(path: string): Promise<unknown> {
  const response = await fetch(path)
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`)
  }
  return response.json() as Promise<unknown>
}

async function fetchOptionalJson(path: string): Promise<unknown | null> {
  const response = await fetch(path)
  if (response.status === 404) {
    return null
  }
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`)
  }
  return response.json() as Promise<unknown>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
