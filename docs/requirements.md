# Requirements

## Product Goal

Wurmkickflip is a browser-based training viewer and simulation prototype for an OpenWorm-inspired worm controller learning to ride a skateboard inside a small terrarium.

The project should stay honest about fidelity: this is not a literal full OpenWorm biophysical simulation in the browser. It is a practical control-learning prototype that borrows the C. elegans theme, segment/muscle structure, and future OpenWorm/VirtualWorm visual references.

## MVP Behavior

- Render a nonblank 3D terrarium scene in a modern browser.
- Show a worm-like articulated visual body on a skateboard with trucks and wheels.
- Drive the worm with dorsal/ventral segment muscle activations.
- Run today without a trained policy by using the scripted muscle-wave fallback.
- Load `public/models/wurmkickflip_policy.onnx` when present and compatible with `public/models/wurmkickflip_policy.meta.json`.
- Prefer ONNX Runtime WebGPU when available, and degrade to WASM or scripted control when unavailable.
- Show a training viewer with reward, distance, contact ratio, rollout time, policy backend, status message, and muscle activity bars.

## Success Criteria

The v1 learned-policy target is:

- The worm stays on the skateboard for at least 20 seconds.
- The board moves at least 8 deck lengths forward in the terrarium replay.
- Browser inference runs without changing the observation/action contract.
- Policy fallback behavior remains usable when no ONNX model is present.

## Non-Goals

- Do not implement a biologically faithful whole-organism OpenWorm simulator in the browser for the first production path.
- Do not make OpenWorm/c302/NEURON tooling a browser runtime dependency.
- Do not prioritize a polished toy over the training viewer and policy integration until the policy contract is validated.
- Do not replace the TypeScript contract with ad hoc JSON or untyped arrays.

## Runtime Requirements

- Web app: Vite, React, TypeScript, Three.js, React Three Fiber, Drei, Rapier, ONNX Runtime Web.
- Browser: Chrome or Edge recommended for WebGPU; fallback must keep the app usable elsewhere.
- Training: Python 3.11 managed with `uv`, Gymnasium, Stable Baselines3, PyTorch, ONNX.
- Model artifacts:
  - `public/models/wurmkickflip_policy.onnx`
  - `public/models/wurmkickflip_policy.meta.json`

## Acceptance Checks

- `npm run build` completes.
- The local app loads at the dev server URL and shows a visible canvas plus training viewer.
- With no ONNX model present, backend status becomes `scripted` with a clear missing-model message.
- With an ONNX model present, metadata shape validation passes before inference starts.
- TypeScript and Python constants remain aligned:
  - `SEGMENT_COUNT = 16`
  - `OBSERVATION_SIZE = 118`
  - `ACTION_SIZE = 32`
  - timestep = `1 / 60`
