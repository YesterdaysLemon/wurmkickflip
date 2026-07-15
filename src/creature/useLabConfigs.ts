import { useCallback, useEffect, useState } from 'react'
import {
  browserLabConfigFetcher,
  loadLabConfigResources,
  type LabConfigLoadResult,
  type LabConfigResourceStatus,
} from './labConfigLoader'
import type { CreatureGenome, EnvironmentConfig } from './types'

export type LabConfigs = {
  creature: CreatureGenome | null
  creatures: CreatureGenome[]
  environment: EnvironmentConfig | null
  environments: EnvironmentConfig[]
  resources: LabConfigResourceStatus[]
  status: 'loading' | LabConfigLoadResult['status']
  message: string
  retry: () => void
}

type LabConfigState = Omit<LabConfigs, 'retry'>

const initialState: LabConfigState = {
  creature: null,
  creatures: [],
  environment: null,
  environments: [],
  resources: [],
  status: 'loading',
  message: 'Loading creature and environment resources.',
}

export function useLabConfigs(): LabConfigs {
  const [configs, setConfigs] = useState<LabConfigState>(initialState)
  const [loadAttempt, setLoadAttempt] = useState(0)
  const retry = useCallback(() => {
    setConfigs(current => ({
      ...current,
      status: 'loading',
      message: 'Retrying configuration resources; current valid selections remain available.',
    }))
    setLoadAttempt(attempt => attempt + 1)
  }, [])

  useEffect(() => {
    let mounted = true

    void loadLabConfigResources(browserLabConfigFetcher)
      .then(result => {
        if (!mounted) return
        setConfigs({
          creature: result.creatures[0] ?? null,
          creatures: result.creatures,
          environment: result.environments[0] ?? null,
          environments: result.environments,
          resources: result.resources,
          status: result.status,
          message: result.message,
        })
      })
      .catch(error => {
        if (!mounted) return
        setConfigs(current => ({
          ...current,
          status: 'error',
          message:
            error instanceof Error
              ? `Configuration loader failed unexpectedly: ${error.message}. Retry to recover.`
              : 'Configuration loader failed unexpectedly. Retry to recover.',
        }))
      })

    return () => {
      mounted = false
    }
  }, [loadAttempt])

  return { ...configs, retry }
}
