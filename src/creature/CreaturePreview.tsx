import type { CreatureGenome } from './types'

type CreaturePreviewProps = {
  genome: CreatureGenome | null
  offset?: [number, number, number]
}

export function CreaturePreview({ genome, offset = [0, 0, -0.95] }: CreaturePreviewProps) {
  if (!genome) {
    return null
  }

  return (
    <group position={offset}>
      {genome.morphology.bodyParts.map(part => (
        <mesh castShadow key={part.id} position={part.position} rotation={part.rotation} scale={part.size}>
          <CreatureGeometry shape={part.shape} />
          <meshStandardMaterial color={part.visual.color} roughness={0.72} />
        </mesh>
      ))}
      {genome.morphology.joints.map(joint => (
        <mesh castShadow key={joint.id} position={joint.anchor}>
          <sphereGeometry args={[0.035, 12, 12]} />
          <meshStandardMaterial color="#25332d" roughness={0.5} />
        </mesh>
      ))}
    </group>
  )
}

function CreatureGeometry({ shape }: { shape: CreatureGenome['morphology']['bodyParts'][number]['shape'] }) {
  if (shape === 'box') {
    return <boxGeometry args={[1, 1, 1]} />
  }
  if (shape === 'sphere') {
    return <sphereGeometry args={[0.5, 18, 18]} />
  }
  if (shape === 'cylinder') {
    return <cylinderGeometry args={[0.5, 0.5, 1, 18]} />
  }
  return <capsuleGeometry args={[0.5, 1, 6, 14]} />
}
