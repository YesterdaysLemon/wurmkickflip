import { useEffect, useState } from 'react'
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

    Promise.all([
      fetch('/configs/creatures/segmented-starter.json').then((response) => response.json() as Promise<CreatureGenome>),
      fetch('/configs/environments/adaptive-skate-terrarium.json').then(
        (response) => response.json() as Promise<EnvironmentConfig>,
      ),
    ])
      .then(([creature, environment]) => {
        if (mounted) {
          setConfigs({
            creature,
            environment,
            status: 'ready',
            message: `Loaded ${creature.name} in ${environment.name}.`,
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
