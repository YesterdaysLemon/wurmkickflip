import { useEffect, useState } from 'react'
import { validateCreatureGenome, validateEnvironmentConfig } from './configValidation'
import type { CreatureGenome, EnvironmentConfig } from './types'

export type LabConfigs = {
  creature: CreatureGenome | null
  environment: EnvironmentConfig | null
  status: 'loading' | 'ready' | 'error'
  message: string
}

export function useLabConfigs(): LabConfigs {
  const [configs, setConfigs] = useState<LabConfigs>({
    creature: null,
    environment: null,
    status: 'loading',
    message: 'Loading lab configs.',
  })

  useEffect(() => {
    let mounted = true

    Promise.all([fetchJson('/configs/creatures/segmented-starter.json'), fetchJson('/configs/environments/adaptive-skate-terrarium.json')])
      .then(([creature, environment]) => {
        const creatureValidation = validateCreatureGenome(creature)
        const environmentValidation = validateEnvironmentConfig(environment)
        if (!creatureValidation.ok || !environmentValidation.ok) {
          const errors = [
            ...(creatureValidation.ok ? [] : creatureValidation.errors),
            ...(environmentValidation.ok ? [] : environmentValidation.errors),
          ]
          throw new Error(`Invalid lab config: ${errors.slice(0, 3).join(' ')}`)
        }

        if (mounted) {
          setConfigs({
            creature: creatureValidation.value,
            environment: environmentValidation.value,
            status: 'ready',
            message: `Loaded ${creatureValidation.value.name} in ${environmentValidation.value.name}.`,
          })
        }
      })
      .catch((error) => {
        if (mounted) {
          setConfigs({
            creature: null,
            environment: null,
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

async function fetchJson(path: string): Promise<unknown> {
  const response = await fetch(path)
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`)
  }
  return response.json() as Promise<unknown>
}
