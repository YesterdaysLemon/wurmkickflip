from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPLAY_SCHEMA_VERSION = 1
REPLAY_KIND = "wurmkickflip.replay"
POLICY_BACKENDS = {"scripted", "neural-js", "onnx-webgpu", "onnx-wasm", "training"}
FALL_REASONS = {"none", "lost_contact", "fell_below_height", "out_of_bounds", "timeout"}


def validate_replay_artifact(value: Any) -> list[str]:
    errors: list[str] = []
    root = as_record(value, "replay", errors)
    source = as_record(root.get("source"), "replay.source", errors) if root else {}
    environment = as_record(root.get("environmentSample"), "replay.environmentSample", errors) if root else {}
    metrics = as_record(root.get("taskMetrics"), "replay.taskMetrics", errors) if root else {}
    frames = as_array(root.get("frames"), "replay.frames", errors) if root else []

    if root:
        expect_literal(root.get("schemaVersion"), REPLAY_SCHEMA_VERSION, "replay.schemaVersion", errors)
        expect_literal(root.get("kind"), REPLAY_KIND, "replay.kind", errors)
        non_empty_string(root.get("replayId"), "replay.replayId", errors)
        non_empty_string(root.get("createdAt"), "replay.createdAt", errors)
        positive_number(root.get("timestep"), "replay.timestep", errors)
        non_negative_number(root.get("durationSeconds"), "replay.durationSeconds", errors)
        frame_count = non_negative_integer(root.get("frameCount"), "replay.frameCount", errors)
        if frame_count != len(frames):
            errors.append(f"replay.frameCount must equal replay.frames length {len(frames)}.")

    if source:
        non_empty_string(source.get("creatureId"), "replay.source.creatureId", errors)
        non_empty_string(source.get("environmentId"), "replay.source.environmentId", errors)
        enum_value(source.get("policyBackend"), POLICY_BACKENDS, "replay.source.policyBackend", errors)
        non_empty_string(source.get("modelVersion"), "replay.source.modelVersion", errors)

    if environment:
        finite_number(environment.get("seed"), "replay.environmentSample.seed", errors)
        finite_number(environment.get("gravityScale"), "replay.environmentSample.gravityScale", errors)
        finite_number(environment.get("frictionScale"), "replay.environmentSample.frictionScale", errors)
        finite_number(environment.get("dragScale"), "replay.environmentSample.dragScale", errors)
        finite_number(environment.get("slopeDegrees"), "replay.environmentSample.slopeDegrees", errors)
        non_negative_number(environment.get("roughness"), "replay.environmentSample.roughness", errors)
        non_negative_number(environment.get("obstacleDensity"), "replay.environmentSample.obstacleDensity", errors)
        tuple_numbers(environment.get("skateboardSpawn"), 2, "replay.environmentSample.skateboardSpawn", errors)
        positive_number(environment.get("skateboardMass"), "replay.environmentSample.skateboardMass", errors)
        non_negative_number(environment.get("wheelFriction"), "replay.environmentSample.wheelFriction", errors)

    if metrics:
        nullable_non_negative_number(metrics.get("skateboardDiscoveredAt"), "replay.taskMetrics.skateboardDiscoveredAt", errors)
        nullable_non_negative_number(metrics.get("firstContactAt"), "replay.taskMetrics.firstContactAt", errors)
        non_negative_number(metrics.get("mountTime"), "replay.taskMetrics.mountTime", errors)
        non_negative_number(metrics.get("rollingDistance"), "replay.taskMetrics.rollingDistance", errors)
        non_negative_number(metrics.get("maxRollingVelocity"), "replay.taskMetrics.maxRollingVelocity", errors)
        unit_number(metrics.get("averageContactRatio"), "replay.taskMetrics.averageContactRatio", errors)
        non_negative_number(metrics.get("energy"), "replay.taskMetrics.energy", errors)
        non_negative_number(metrics.get("survivalTime"), "replay.taskMetrics.survivalTime", errors)
        enum_value(metrics.get("fallReason"), FALL_REASONS, "replay.taskMetrics.fallReason", errors)

    previous_time = 0.0
    for index, frame in enumerate(frames):
        record = as_record(frame, f"replay.frames[{index}]", errors)
        if not record:
            continue
        time = non_negative_number(record.get("time"), f"replay.frames[{index}].time", errors)
        if index > 0 and time < previous_time:
            errors.append(f"replay.frames[{index}].time must be greater than or equal to the previous frame time.")
        previous_time = time
        validate_pose(record.get("board"), f"replay.frames[{index}].board", errors)
        validate_pose(record.get("creatureRoot"), f"replay.frames[{index}].creatureRoot", errors)
        unit_number(record.get("contactRatio"), f"replay.frames[{index}].contactRatio", errors)
        boolean_value(record.get("skateboardDiscovered"), f"replay.frames[{index}].skateboardDiscovered", errors)
        finite_number(record.get("rollingVelocity"), f"replay.frames[{index}].rollingVelocity", errors)
        finite_number(record.get("reward"), f"replay.frames[{index}].reward", errors)

    return errors


def validate_pose(value: Any, path: str, errors: list[str]) -> None:
    record = as_record(value, path, errors)
    if not record:
        return
    tuple_numbers(record.get("position"), 3, f"{path}.position", errors)
    tuple_numbers(record.get("velocity"), 3, f"{path}.velocity", errors)
    tuple_numbers(record.get("rotation"), 3, f"{path}.rotation", errors)


def as_record(value: Any, path: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object.")
        return {}
    return value


def as_array(value: Any, path: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list):
        errors.append(f"{path} must be an array.")
        return []
    return value


def expect_literal(value: Any, expected: Any, path: str, errors: list[str]) -> None:
    if value != expected:
        errors.append(f"{path} must equal {expected}.")


def non_empty_string(value: Any, path: str, errors: list[str]) -> str:
    if not isinstance(value, str) or len(value) == 0:
        errors.append(f"{path} must be a non-empty string.")
        return ""
    return value


def boolean_value(value: Any, path: str, errors: list[str]) -> bool:
    if not isinstance(value, bool):
        errors.append(f"{path} must be a boolean.")
        return False
    return value


def finite_number(value: Any, path: str, errors: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not float("-inf") < float(value) < float("inf"):
        errors.append(f"{path} must be a finite number.")
        return 0.0
    return float(value)


def non_negative_integer(value: Any, path: str, errors: list[str]) -> int:
    number = finite_number(value, path, errors)
    if int(number) != number or number < 0:
        errors.append(f"{path} must be a non-negative integer.")
    return int(number)


def positive_number(value: Any, path: str, errors: list[str]) -> float:
    number = finite_number(value, path, errors)
    if number <= 0:
        errors.append(f"{path} must be greater than 0.")
    return number


def non_negative_number(value: Any, path: str, errors: list[str]) -> float:
    number = finite_number(value, path, errors)
    if number < 0:
        errors.append(f"{path} must be at least 0.")
    return number


def nullable_non_negative_number(value: Any, path: str, errors: list[str]) -> float | None:
    if value is None:
        return None
    return non_negative_number(value, path, errors)


def unit_number(value: Any, path: str, errors: list[str]) -> float:
    number = finite_number(value, path, errors)
    if number < 0 or number > 1:
        errors.append(f"{path} must be between 0 and 1.")
    return number


def tuple_numbers(value: Any, length: int, path: str, errors: list[str]) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        errors.append(f"{path} must be an array of {length} finite numbers.")
        return [0.0] * length
    return [finite_number(item, f"{path}[{index}]", errors) for index, item in enumerate(value)]


def enum_value(value: Any, allowed: set[str], path: str, errors: list[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        errors.append(f"{path} must be one of: {', '.join(sorted(allowed))}.")
        return sorted(allowed)[0]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    args = parser.parse_args()

    errors = validate_replay_artifact(json.loads(args.replay.read_text(encoding="utf-8")))
    if errors:
        print("Replay validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Replay validation passed.")


if __name__ == "__main__":
    main()
