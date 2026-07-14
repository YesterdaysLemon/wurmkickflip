from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    ACTION_SIZE,
    IGNORED_OBSERVATION_INDICES,
    OBSERVATION_HEADER_SIZE,
    OBSERVATION_SIZE,
    SEGMENT_COUNT,
    SEGMENT_OBSERVATION_SIZE,
    TEACHER_FEATURE_INDICES,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "public/models/wurmkickflip_stunt_policy.json"
EXPECTED_KIND = "wurmkickflip.stuntPolicy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the distilled stunt-policy JSON artifact.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    artifact = json.loads(args.model.read_text(encoding="utf-8"))
    parameters = validate_artifact(artifact)
    metrics = validate_inference(parameters)
    print(json.dumps({"model": str(args.model), **metrics}, indent=2, sort_keys=True))
    print("Stunt policy validation passed.")


def validate_artifact(artifact: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(artifact, dict):
        raise SystemExit("artifact must be a JSON object")
    expect(artifact.get("schemaVersion") == 1, "schemaVersion must equal 1")
    expect(artifact.get("kind") == EXPECTED_KIND, f"kind must equal {EXPECTED_KIND}")
    expect(isinstance(artifact.get("modelVersion"), str) and artifact["modelVersion"], "modelVersion must be non-empty")
    expect(artifact.get("inputSize") == OBSERVATION_SIZE == 174, "inputSize must equal 174")
    expect(artifact.get("outputSize") == ACTION_SIZE == 32, "outputSize must equal 32")
    expect(artifact.get("activation") == "tanh", "activation must equal tanh")
    hidden_size = artifact.get("hiddenSize")
    expect(isinstance(hidden_size, int) and hidden_size > 0, "hiddenSize must be a positive integer")

    hidden_weight = matrix(artifact.get("hiddenWeights"), hidden_size, OBSERVATION_SIZE, "hiddenWeights")
    hidden_bias = vector(artifact.get("hiddenBias"), hidden_size, "hiddenBias")
    output_weight = matrix(artifact.get("outputWeights"), ACTION_SIZE, hidden_size, "outputWeights")
    output_bias = vector(artifact.get("outputBias"), ACTION_SIZE, "outputBias")

    training = artifact.get("training")
    expect(isinstance(training, dict), "training must be an object")
    expect(isinstance(training.get("seed"), int), "training.seed must be an integer")
    expect(isinstance(training.get("samples"), int) and training["samples"] > 0, "training.samples must be positive")
    expect(isinstance(training.get("epochs"), int) and training["epochs"] > 0, "training.epochs must be positive")
    finite_number(training.get("validationMse"), "training.validationMse", lower=0.0)
    agreement = finite_number(training.get("teacherAgreement"), "training.teacherAgreement", lower=0.0)
    expect(agreement <= 1.0, "training.teacherAgreement must be at most 1")
    feature_indices = training.get("teacherFeatureIndices")
    expect(isinstance(feature_indices, list), "training.teacherFeatureIndices must be an array")
    expect(
        all(type(index) is int for index in feature_indices),
        "training.teacherFeatureIndices must contain only integers",
    )
    expect(
        tuple(feature_indices) == TEACHER_FEATURE_INDICES,
        "training.teacherFeatureIndices does not match the documented teacher mask",
    )

    ignored_weight = hidden_weight[:, IGNORED_OBSERVATION_INDICES]
    ignored_nonzero = int(np.count_nonzero(ignored_weight))
    expect(ignored_nonzero == 0, f"ignored observation columns contain {ignored_nonzero} nonzero weights")
    return hidden_weight, hidden_bias, output_weight, output_bias


def validate_inference(
    parameters: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, float]:
    hidden_weight, hidden_bias, output_weight, output_bias = parameters
    times = np.concatenate(
        (
            np.linspace(0.3, 1.9, 15),
            np.linspace(2.27, 2.57, 12),
            np.linspace(2.69, 2.82, 12),
            np.linspace(2.98, 3.48, 12),
            np.linspace(3.84, 4.12, 10),
            np.linspace(4.55, 6.9, 18),
        )
    )
    observations = canonical_observations(times)
    actions = infer(observations, hidden_weight, hidden_bias, output_weight, output_bias)
    expect(actions.shape == (len(times), ACTION_SIZE), f"inference shape must be ({len(times)}, {ACTION_SIZE})")
    expect(bool(np.isfinite(actions).all()), "inference produced a non-finite action")
    expect(float(np.max(np.abs(actions))) <= 1.000001, "tanh inference escaped [-1, 1]")
    ignored_feature_delta = validate_ignored_feature_invariance(
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
    )

    bend = (actions[:, 0::2] - actions[:, 1::2]) * 0.5
    co_contraction = (actions[:, 0::2] + actions[:, 1::2]) * 0.5
    side_weight = np.where(np.arange(SEGMENT_COUNT) < SEGMENT_COUNT / 2, -1.0, 1.0)
    kick = np.mean(bend * side_weight[None, :], axis=1)

    coil = (times >= 2.27) & (times <= 2.57)
    release = (times >= 2.69) & (times <= 2.82)
    air = (times >= 2.98) & (times <= 3.48)
    travel = (times < 2.0) | (times > 4.4)
    coil_co = float(np.mean(co_contraction[coil]))
    release_co = float(np.mean(co_contraction[release]))
    release_kick = float(np.mean(kick[release]))
    air_co = float(np.mean(co_contraction[air]))
    travel_bend_rms = float(np.sqrt(np.mean(bend[travel] ** 2)))

    expect(coil_co > 0.42, f"coil co-contraction too weak: {coil_co:.4f}")
    expect(release_co < -0.34, f"release co-contraction must be negative: {release_co:.4f}")
    expect(release_kick > 0.15, f"release kick signal too weak: {release_kick:.4f}")
    expect(air_co > 0.10, f"air tuck co-contraction too weak: {air_co:.4f}")
    expect(travel_bend_rms > 0.15, f"traveling bend wave too weak: {travel_bend_rms:.4f}")

    feedback = canonical_observations(np.array([3.12, 3.12]))
    feedback[0, 8] -= 0.8
    feedback[1, 8] += 0.8
    feedback_actions = infer(feedback, hidden_weight, hidden_bias, output_weight, output_bias)
    feedback_bend = (feedback_actions[:, 0::2] - feedback_actions[:, 1::2]) * 0.5
    feedback_kick = np.mean(feedback_bend * side_weight[None, :], axis=1)
    roll_feedback_delta = float(feedback_kick[0] - feedback_kick[1])
    expect(roll_feedback_delta > 0.01, f"roll-state kick feedback too weak: {roll_feedback_delta:.4f}")

    return {
        "airCoContraction": round(air_co, 6),
        "coilCoContraction": round(coil_co, 6),
        "ignoredFeatureMaxDelta": ignored_feature_delta,
        "maxAbsAction": round(float(np.max(np.abs(actions))), 6),
        "releaseCoContraction": round(release_co, 6),
        "releaseKickSignal": round(release_kick, 6),
        "rollFeedbackDelta": round(roll_feedback_delta, 6),
        "travelBendRms": round(travel_bend_rms, 6),
    }


def canonical_observations(times: np.ndarray) -> np.ndarray:
    values = np.asarray(times, dtype=np.float64)
    observations = np.zeros((len(values), OBSERVATION_SIZE), dtype=np.float64)
    observations[:, 0] = values
    observations[:, 2] = 0.42
    observations[:, 4] = 0.8
    observations[:, 7] = -0.04
    observations[:, 10] = np.where(
        values < 2.65,
        0.84,
        np.where(values < 2.9, 0.24, np.where(values < 3.75, 0.06, 0.72)),
    )
    observations[:, 11] = 1.0
    for segment, x in enumerate(np.linspace(-0.76, 0.76, SEGMENT_COUNT)):
        start = OBSERVATION_HEADER_SIZE + segment * SEGMENT_OBSERVATION_SIZE
        observations[:, start] = x
        observations[:, start + 1] = 0.19
    return observations


def validate_ignored_feature_invariance(
    hidden_weight: np.ndarray,
    hidden_bias: np.ndarray,
    output_weight: np.ndarray,
    output_bias: np.ndarray,
) -> float:
    baseline = canonical_observations(np.array([0.8, 2.43, 2.75, 3.18, 4.02, 6.1]))
    perturbed = baseline.copy()
    rows = np.arange(len(baseline), dtype=np.float64)[:, None] + 1.0
    columns = np.asarray(IGNORED_OBSERVATION_INDICES, dtype=np.float64)[None, :] + 1.0
    # Exercise ignored channels well beyond the browser's normal position and
    # velocity envelope. An exact zero-column export must be bit-stable here.
    perturbed[:, IGNORED_OBSERVATION_INDICES] = np.sin(rows * columns * 0.731) * 500.0

    expected = infer(baseline, hidden_weight, hidden_bias, output_weight, output_bias)
    actual = infer(perturbed, hidden_weight, hidden_bias, output_weight, output_bias)
    maximum = float(np.max(np.abs(actual - expected)))
    expect(maximum <= 1e-12, f"ignored-feature perturbations changed actions by {maximum}")
    return round(maximum, 12)


def infer(
    observations: np.ndarray,
    hidden_weight: np.ndarray,
    hidden_bias: np.ndarray,
    output_weight: np.ndarray,
    output_bias: np.ndarray,
) -> np.ndarray:
    hidden = np.tanh(observations @ hidden_weight.T + hidden_bias)
    return np.tanh(hidden @ output_weight.T + output_bias)


def matrix(value: Any, rows: int, columns: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"{label} must be a numeric matrix: {error}") from error
    expect(result.shape == (rows, columns), f"{label} must have shape ({rows}, {columns}), found {result.shape}")
    expect(bool(np.isfinite(result).all()), f"{label} must contain only finite numbers")
    return result


def vector(value: Any, length: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"{label} must be a numeric vector: {error}") from error
    expect(result.shape == (length,), f"{label} must have shape ({length},), found {result.shape}")
    expect(bool(np.isfinite(result).all()), f"{label} must contain only finite numbers")
    return result


def finite_number(value: Any, label: str, *, lower: float | None = None) -> float:
    expect(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    result = float(value)
    expect(math.isfinite(result), f"{label} must be finite")
    if lower is not None:
        expect(result >= lower, f"{label} must be at least {lower}")
    return result


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


if __name__ == "__main__":
    main()
