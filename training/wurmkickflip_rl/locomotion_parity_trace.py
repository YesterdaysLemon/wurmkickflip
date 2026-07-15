"""Emit a browser-precision locomotion trace from the Python evolution runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .articulated_locomotion import articulated_plant_step, initial_body
from .evolve_locomotion_policy import (
    POLICY_TIMESTEP,
    SEGMENT_COUNT,
    segmental_network_step,
)


CHECKPOINT_STEPS = {0, 1, 2, 23, 59, 119, 239, 419}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the evolved controller through Python for browser parity checks."
    )
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    artifact = json.loads(args.model.read_text(encoding="utf-8"))
    weights = artifact["weights"]
    steps = int(artifact["training"]["episodeSteps"])
    target = np.array([[[3.2, 3.2]]], dtype=np.float64)
    friction = np.array([[0.8]], dtype=np.float64)
    urgency = np.array([[0.9]], dtype=np.float64)

    hidden = np.asarray(artifact["initialState"], dtype=np.float64)[None, None, :].copy()
    previous_command = np.zeros((1, 1, SEGMENT_COUNT), dtype=np.float64)
    joint = np.zeros_like(previous_command)
    joint_velocity = np.zeros_like(previous_command)
    body_scale = np.ones(1, dtype=np.float64)
    body_position, body_velocity = initial_body(1, 1, body_scale)
    position = np.mean(body_position, axis=2)
    forward = np.zeros_like(position)
    forward[:, :, 0] = 1.0
    speed = np.zeros((1, 1), dtype=np.float64)
    angular_speed = np.zeros_like(speed)
    contact_load = np.full_like(joint, 0.8)
    slip_speed = np.zeros_like(joint)
    obstacle_forward = np.zeros_like(joint)
    obstacle_right = np.zeros_like(joint)
    input_weights = np.asarray(weights["input"], dtype=np.float64)[None, :]
    recurrent_weights = np.asarray(weights["recurrent"], dtype=np.float64)[None, :]
    output_weights = np.asarray(weights["output"], dtype=np.float64)[None, :]
    checkpoints: list[dict[str, object]] = []

    for step in range(steps):
        relative = target - position
        distance = np.maximum(np.linalg.norm(relative, axis=2), 1.0e-9)
        right = np.stack((-forward[:, :, 1], forward[:, :, 0]), axis=2)
        target_forward = np.sum(relative * forward, axis=2) / distance
        target_right = np.sum(relative * right, axis=2) / distance
        hidden, neural_command = segmental_network_step(
            hidden,
            previous_command,
            joint,
            joint_velocity,
            target_forward,
            target_right,
            np.minimum(distance / 5.0, 1.5),
            speed,
            angular_speed,
            friction,
            urgency,
            contact_load,
            slip_speed,
            obstacle_forward,
            obstacle_right,
            input_weights,
            recurrent_weights,
            output_weights,
        )

        # The JS policy keeps recurrent state in Float64, but its PolicyAction is
        # Float32 before the plant reads it. Preserve that exact quantization split.
        applied_command = neural_command.astype(np.float32).astype(np.float64)
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
        previous_command = neural_command.copy()

        if step in CHECKPOINT_STEPS or step == steps - 1:
            checkpoints.append(
                {
                    "step": step,
                    "position": position[0, 0].tolist(),
                    "heading": float(np.arctan2(forward[0, 0, 1], forward[0, 0, 0])),
                    "forwardSpeed": float(speed[0, 0]),
                    "angularSpeed": float(angular_speed[0, 0]),
                    "joints": joint[0, 0].tolist(),
                    "jointVelocities": joint_velocity[0, 0].tolist(),
                    "commands": applied_command[0, 0].tolist(),
                }
            )

    print(json.dumps({"steps": steps, "checkpoints": checkpoints}, separators=(",", ":")))


if __name__ == "__main__":
    main()
