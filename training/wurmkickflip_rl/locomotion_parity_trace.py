"""Emit a browser-precision locomotion trace from the Python evolution runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np

from . import evolve_locomotion_policy as evolution_runtime
from .articulated_locomotion import articulated_plant_step, initial_body
from .evolve_locomotion_policy import (
    POLICY_TIMESTEP,
    SEGMENT_COUNT,
    runtime_controller_sensors,
)


Mode = Literal["full", "zero", "frozen", "shuffled"]

CHECKPOINT_STEPS = {0, 1, 2, 23, 24, 59, 119, 239, 419}
FROZEN_CAPTURE_STEP = 24
SEGMENT_SHUFFLE = np.array(
    [0, 9, 2, 13, 4, 15, 6, 11, 8, 1, 10, 3, 12, 5, 14, 7],
    dtype=np.int64,
)


def neural_transition_fixtures(
    artifact: dict[str, object],
    input_weights: np.ndarray,
    recurrent_weights: np.ndarray,
    output_weights: np.ndarray,
) -> list[dict[str, object]]:
    """Emit identical-state controller transitions for the TypeScript oracle."""
    index = np.arange(SEGMENT_COUNT, dtype=np.float64)
    fixture_inputs = [
        {
            "name": "asymmetric-loaded-state",
            "targetForward": 0.37,
            "targetRight": -0.42,
            "targetDistance": 1.1,
            "forwardSpeed": -0.18,
            "angularSpeed": 0.27,
            "terrainFriction": 0.63,
            "urgency": 0.81,
            "joints": (index - 7.5) * 0.03125,
            "jointVelocities": (7.5 - index) * 0.021,
            "previousCommands": ((index % 5.0) - 2.0) * 0.125,
            "contactLoads": 0.22 + (index % 4.0) * 0.21,
            "slipSpeeds": (index % 3.0) * 0.17,
            "obstacleForward": np.where(index % 2.0 == 0.0, 0.31, -0.27),
            "obstacleRight": (index - 7.5) * 0.073,
        },
        {
            "name": "sensor-clamp-boundaries",
            "targetForward": 1.7,
            "targetRight": -1.6,
            "targetDistance": 2.4,
            "forwardSpeed": -2.8,
            "angularSpeed": 3.8,
            "terrainFriction": 1.7,
            "urgency": -0.4,
            "joints": np.where(index % 2.0 == 0.0, -0.44, 0.38),
            "jointVelocities": ((index % 6.0) - 2.5) * 0.19,
            "previousCommands": (index - 7.5) * 0.09,
            "contactLoads": np.where(index % 2.0 == 0.0, -0.3, 1.4),
            "slipSpeeds": np.where(index % 3.0 == 0.0, -0.2, 2.6),
            "obstacleForward": np.where(index % 2.0 == 0.0, 1.4, -1.3),
            "obstacleRight": (index - 7.5) * 0.24,
        },
    ]
    fixtures: list[dict[str, object]] = []
    initial_state = np.asarray(artifact["initialState"], dtype=np.float64)[None, None, :]
    for fixture in fixture_inputs:

        def scalar(name: str) -> np.ndarray:
            return np.array([[fixture[name]]], dtype=np.float64)

        def local(name: str) -> np.ndarray:
            return np.asarray(fixture[name], dtype=np.float64)[None, None, :]

        controller_sensors = runtime_controller_sensors(
            scalar("targetForward"),
            scalar("targetRight"),
            scalar("targetDistance"),
            scalar("forwardSpeed"),
            scalar("angularSpeed"),
            scalar("terrainFriction"),
            scalar("urgency"),
            local("contactLoads"),
            local("slipSpeeds"),
            local("obstacleForward"),
            local("obstacleRight"),
        )
        hidden, commands = evolution_runtime.segmental_network_step(
            initial_state.copy(),
            local("previousCommands"),
            local("joints"),
            local("jointVelocities"),
            *controller_sensors,
            input_weights,
            recurrent_weights,
            output_weights,
        )
        fixtures.append(
            {
                "name": fixture["name"],
                "input": {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in fixture.items()
                    if key != "name"
                },
                "output": {
                    "hidden": hidden[0, 0].tolist(),
                    "commands": commands[0, 0].tolist(),
                    "plantCommands": commands[0, 0].astype(np.float32).astype(np.float64).tolist(),
                },
            }
        )
    return fixtures


def isolated_plant_transition() -> dict[str, object]:
    """Emit one asymmetric browser-contract plant transition."""
    index = np.arange(SEGMENT_COUNT, dtype=np.float64)
    body_scale = np.ones(1, dtype=np.float64)
    body_position, body_velocity = initial_body(1, 1, body_scale)
    body_position[0, 0, :, 1] = ((index - 7.5) / 7.5) ** 3 * 0.028 + ((index % 3.0) - 1.0) * 0.0017
    # Exceed the shared COM speed cap while keeping asymmetric relative motion,
    # so this fixture catches both limiter and controller-to-edge mapping drift.
    body_velocity[0, 0, :, 0] = 3.15 + ((index % 4.0) - 1.5) * 0.08
    body_velocity[0, 0, :, 1] = 0.72 + (index - 7.5) * 0.012
    joint = (((index * 7.0) % 17.0) - 8.0)[None, None, :] * 0.047
    joint_velocity = (((index * 5.0) % 13.0) - 6.0)[None, None, :] * 0.038
    command = np.tanh((index - 6.2) * 0.19 + (index % 3.0 - 1.0) * 0.23)[None, None, :]
    plant_command = command.astype(np.float32).astype(np.float64)
    input_state = {
        "friction": 0.73,
        "joints": joint[0, 0].copy().tolist(),
        "jointVelocities": joint_velocity[0, 0].copy().tolist(),
        "plantCommands": plant_command[0, 0].tolist(),
        "bodyPositions": body_position[0, 0].copy().tolist(),
        "bodyVelocities": body_velocity[0, 0].copy().tolist(),
    }
    (
        _step_distance,
        root,
        forward,
        speed,
        angular_speed,
        _contact_load,
        _slip_speed,
        _obstacle_forward,
        _obstacle_right,
    ) = articulated_plant_step(
        plant_command,
        joint,
        joint_velocity,
        body_position,
        body_velocity,
        np.array([[input_state["friction"]]], dtype=np.float64),
        np.zeros((1, 1), dtype=np.float64),
        body_scale,
        np.zeros((1, 2), dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        POLICY_TIMESTEP,
    )
    return {
        "input": input_state,
        "output": {
            "root": root[0, 0].tolist(),
            "heading": float(np.arctan2(forward[0, 0, 1], forward[0, 0, 0])),
            "forwardSpeed": float(speed[0, 0]),
            "angularSpeed": float(angular_speed[0, 0]),
            "joints": joint[0, 0].tolist(),
            "jointVelocities": joint_velocity[0, 0].tolist(),
            "bodyPositions": body_position[0, 0].tolist(),
            "bodyVelocities": body_velocity[0, 0].tolist(),
        },
    }


def intervene_command(
    neural_command: np.ndarray,
    mode: Mode,
    step: int,
    frozen_command: np.ndarray,
) -> np.ndarray:
    """Apply the verifier intervention without lowering controller precision."""
    if mode == "zero":
        return np.zeros_like(neural_command)
    if mode == "frozen":
        if step == FROZEN_CAPTURE_STEP:
            frozen_command[:] = neural_command
        return frozen_command.copy() if step >= FROZEN_CAPTURE_STEP else neural_command.copy()
    if mode == "shuffled":
        return neural_command[:, :, SEGMENT_SHUFFLE].copy()
    return neural_command.copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the evolved controller through Python for browser parity checks."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("full", "zero", "frozen", "shuffled"),
        default="full",
        help="Actuator intervention to apply before recurrent feedback and the plant.",
    )
    args = parser.parse_args()
    mode: Mode = args.mode

    artifact = json.loads(args.model.read_text(encoding="utf-8"))
    weights = artifact["weights"]
    segment_positions = np.asarray(artifact["segmentPositions"], dtype=np.float64)
    if segment_positions.shape != (SEGMENT_COUNT,):
        raise ValueError(f"model must contain {SEGMENT_COUNT} segment positions")
    # Replay the artifact's serialized values explicitly so this trace audits
    # the selected model even if a future trainer changes its default anatomy.
    evolution_runtime.SEGMENT_POSITIONS = segment_positions
    steps = int(artifact["training"]["episodeSteps"])
    target = np.array([[[3.2, 3.2]]], dtype=np.float64)
    friction = np.array([[0.8]], dtype=np.float64)
    urgency = np.array([[0.9]], dtype=np.float64)

    hidden = np.asarray(artifact["initialState"], dtype=np.float64)[None, None, :].copy()
    previous_command = np.zeros((1, 1, SEGMENT_COUNT), dtype=np.float64)
    frozen_command = np.zeros_like(previous_command)
    joint = np.zeros_like(previous_command)
    joint_velocity = np.zeros_like(previous_command)
    body_scale = np.ones(1, dtype=np.float64)
    body_position, body_velocity = initial_body(1, 1, body_scale)
    position = np.mean(body_position, axis=2)
    forward = np.zeros_like(position)
    forward[:, :, 0] = 1.0
    speed = np.zeros((1, 1), dtype=np.float64)
    angular_speed = np.zeros_like(speed)
    contact_load = np.ones_like(joint)
    slip_speed = np.zeros_like(joint)
    obstacle_forward = np.zeros_like(joint)
    obstacle_right = np.zeros_like(joint)
    input_weights = np.asarray(weights["input"], dtype=np.float64)[None, :]
    recurrent_weights = np.asarray(weights["recurrent"], dtype=np.float64)[None, :]
    output_weights = np.asarray(weights["output"], dtype=np.float64)[None, :]
    neural_transitions = neural_transition_fixtures(
        artifact,
        input_weights,
        recurrent_weights,
        output_weights,
    )
    plant_transition = isolated_plant_transition()
    checkpoints: list[dict[str, object]] = []

    for step in range(steps):
        relative = target - position
        distance = np.maximum(np.linalg.norm(relative, axis=2), 1.0e-9)
        right = np.stack((-forward[:, :, 1], forward[:, :, 0]), axis=2)
        target_forward = np.sum(relative * forward, axis=2) / distance
        target_right = np.sum(relative * right, axis=2) / distance
        controller_sensors = runtime_controller_sensors(
            target_forward,
            target_right,
            distance / 5.0,
            speed,
            angular_speed,
            friction,
            urgency,
            contact_load,
            slip_speed,
            obstacle_forward,
            obstacle_right,
        )
        hidden, neural_command = evolution_runtime.segmental_network_step(
            hidden,
            previous_command,
            joint,
            joint_velocity,
            *controller_sensors,
            input_weights,
            recurrent_weights,
            output_weights,
        )

        # Intervention feedback remains Float64 in both runtimes. The browser's
        # PolicyAction separately rounds to Float32 before the articulated plant
        # reads it, so reproduce that precision boundary explicitly here.
        intervened_command = intervene_command(
            neural_command,
            mode,
            step,
            frozen_command,
        )
        applied_command = intervened_command.astype(np.float32).astype(np.float64)
        (
            _step_distance,
            position,
            forward,
            speed,
            angular_speed,
            contact_load,
            slip_speed,
            obstacle_forward,
            obstacle_right,
        ) = articulated_plant_step(
            applied_command,
            joint,
            joint_velocity,
            body_position,
            body_velocity,
            friction,
            np.zeros_like(friction),
            body_scale,
            np.zeros((1, 2), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            POLICY_TIMESTEP,
        )
        previous_command = intervened_command.copy()

        if step in CHECKPOINT_STEPS or step == steps - 1:
            checkpoints.append(
                {
                    "step": step,
                    "root": position[0, 0].tolist(),
                    "heading": float(np.arctan2(forward[0, 0, 1], forward[0, 0, 0])),
                    "forwardSpeed": float(speed[0, 0]),
                    "angularSpeed": float(angular_speed[0, 0]),
                    "joints": joint[0, 0].tolist(),
                    "jointVelocities": joint_velocity[0, 0].tolist(),
                    "commands": intervened_command[0, 0].tolist(),
                    "plantCommands": applied_command[0, 0].tolist(),
                }
            )

    print(
        json.dumps(
            {
                "mode": mode,
                "steps": steps,
                "neuralTransitions": neural_transitions,
                "isolatedPlantTransition": plant_transition,
                "checkpoints": checkpoints,
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
