# Asset Roadmap

Current visible assets are intentionally primitive and generated in code/config:

- Creature bodies: colored boxes, spheres, cylinders, and capsules from creature genome JSON.
- Future creature connectors: articulated spherical/hinge constraints with simple joint markers.
- Skateboard: procedural deck, trucks, and wheels.
- Terrain: procedural tile field, ramps/slopes, obstacle blocks, and terrarium walls.

Useful authored assets later:

- Skateboard deck mesh with trucks and wheel assemblies.
- Material set for rubber, wood, metal, glass, and terrarium substrate.
- Contact/rolling markers for replay inspection.
- Creature part decals or labels for evolved morphology debugging.
- Lightweight sound cues for impact/contact if the viewer grows into a richer sim.

Until these exist, use code-native primitives so physics and generated morphology stay inspectable.
