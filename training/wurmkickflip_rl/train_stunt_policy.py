from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .contracts import (
    ACTION_SIZE,
    IGNORED_OBSERVATION_INDICES,
    OBSERVATION_HEADER_SIZE,
    OBSERVATION_SIZE,
    PREVIOUS_ACTION_OFFSET,
    SEGMENT_COUNT,
    SEGMENT_OBSERVATION_SIZE,
    TEACHER_FEATURE_INDICES,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "public/models/wurmkickflip_stunt_policy.json"
MODEL_KIND = "wurmkickflip.stuntPolicy"
CYCLE_SECONDS = 7.2
TRAVEL_WAVE_HZ = 2.0 / CYCLE_SECONDS
PHASE_TRAVEL = 0
PHASE_COIL = 1
PHASE_RELEASE = 2
PHASE_AIR = 3
PHASE_LANDING = 4
PHASE_NAMES = ("travel", "coil", "release", "air", "landing")


@dataclass(frozen=True)
class Dataset:
    observations: np.ndarray
    actions: np.ndarray
    phases: np.ndarray


class NormalizedStuntPolicy(nn.Module):
    """One-hidden-layer policy whose input normalization is folded into export weights."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray, hidden_size: int) -> None:
        super().__init__()
        self.register_buffer("input_mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("input_scale", torch.from_numpy(scale.astype(np.float32)))
        feature_mask = torch.zeros(OBSERVATION_SIZE, dtype=torch.float32)
        feature_mask[list(TEACHER_FEATURE_INDICES)] = 1.0
        self.register_buffer("input_mask", feature_mask)
        self.hidden = nn.Linear(OBSERVATION_SIZE, hidden_size)
        self.output = nn.Linear(hidden_size, ACTION_SIZE)
        with torch.no_grad():
            self.hidden.weight[:, self.input_mask == 0] = 0.0

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = ((observation - self.input_mean) / self.input_scale) * self.input_mask
        return torch.tanh(self.output(torch.tanh(self.hidden(normalized))))

    def raw_input_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return weights for direct inference on unnormalized browser observations."""
        with torch.no_grad():
            scale = self.input_scale
            mean = self.input_mean
            masked_weight = self.hidden.weight * self.input_mask.unsqueeze(0)
            hidden_weight = masked_weight / scale.unsqueeze(0)
            hidden_bias = self.hidden.bias - torch.sum(masked_weight * (mean / scale).unsqueeze(0), dim=1)
            return (
                hidden_weight.cpu().numpy(),
                hidden_bias.cpu().numpy(),
                self.output.weight.cpu().numpy(),
                self.output.bias.cpu().numpy(),
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill a deterministic kickflip/flop expert into a compact browser JSON MLP."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-version", default="stunt-distilled-v2")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--samples", type=int, default=16_000)
    parser.add_argument("--validation-samples", type=int, default=4_000)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=360)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    args = parser.parse_args()

    validate_args(args)
    assert_contract()
    seed_everything(args.seed)

    train = make_dataset(args.seed, args.samples)
    validation = make_dataset(args.seed + 1, args.validation_samples)
    input_mean = train.observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    input_scale = train.observations.std(axis=0, dtype=np.float64).astype(np.float32)
    input_scale = np.maximum(input_scale, 0.05).astype(np.float32)

    model = NormalizedStuntPolicy(input_mean, input_scale, args.hidden_size)
    train_model(
        model,
        train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    metrics = validation_metrics(model, validation)
    raw_parameters = model.raw_input_parameters()
    assert_ignored_features_are_zero(raw_parameters[0])
    assert_folded_inference_matches(model, validation.observations[:128], raw_parameters)
    artifact = make_artifact(args, metrics, raw_parameters)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, separators=(",", ":")) + "\n", encoding="utf-8")

    printable = {
        "artifact": str(args.out),
        "modelVersion": args.model_version,
        "seed": args.seed,
        "samples": args.samples,
        "validationSamples": args.validation_samples,
        "epochs": args.epochs,
        "hiddenSize": args.hidden_size,
        **metrics,
    }
    print(json.dumps(printable, indent=2, sort_keys=True))


def validate_args(args: argparse.Namespace) -> None:
    if args.samples < 1_000:
        raise SystemExit("--samples must be at least 1000")
    if args.validation_samples < 500:
        raise SystemExit("--validation-samples must be at least 500")
    if args.hidden_size <= 0:
        raise SystemExit("--hidden-size must be positive")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")


def assert_contract() -> None:
    expected = OBSERVATION_HEADER_SIZE + SEGMENT_COUNT * SEGMENT_OBSERVATION_SIZE + ACTION_SIZE
    if OBSERVATION_SIZE != expected:
        raise RuntimeError(f"stunt policy requires the complete {expected}-value observation, found {OBSERVATION_SIZE}")
    if PREVIOUS_ACTION_OFFSET != 142:
        raise RuntimeError(f"unexpected previous-action offset {PREVIOUS_ACTION_OFFSET}")
    if len(TEACHER_FEATURE_INDICES) != 37 or len(set(TEACHER_FEATURE_INDICES)) != 37:
        raise RuntimeError("teacher feature mask must contain 37 unique inputs")
    if set(TEACHER_FEATURE_INDICES) & set(IGNORED_OBSERVATION_INDICES):
        raise RuntimeError("teacher and ignored feature masks must not overlap")
    if len(TEACHER_FEATURE_INDICES) + len(IGNORED_OBSERVATION_INDICES) != OBSERVATION_SIZE:
        raise RuntimeError("teacher and ignored feature masks must cover the observation contract")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def make_dataset(seed: int, sample_count: int) -> Dataset:
    rng = np.random.default_rng(seed)
    phases = rng.choice(
        np.arange(len(PHASE_NAMES), dtype=np.int64),
        size=sample_count,
        p=np.array([0.32, 0.17, 0.18, 0.19, 0.14], dtype=np.float64),
    )
    times = sample_phase_times(rng, phases)
    observations = sample_observations(rng, times, phases)

    # A loop begins with a zero previous action. Keep these columns zero and their
    # learned first-layer weights zero so phase semantics cannot leak through an
    # autoregressive shortcut and remain stable after a browser reset.
    observations[:, PREVIOUS_ACTION_OFFSET:] = 0.0
    actions = expert_actions(observations)
    return Dataset(observations=observations, actions=actions, phases=phases)


def sample_phase_times(rng: np.random.Generator, phases: np.ndarray) -> np.ndarray:
    times = np.empty(len(phases), dtype=np.float32)
    for phase in range(len(PHASE_NAMES)):
        mask = phases == phase
        count = int(mask.sum())
        if count == 0:
            continue
        if phase == PHASE_TRAVEL:
            early = rng.random(count) < 0.46
            sampled = np.empty(count, dtype=np.float64)
            sampled[early] = rng.uniform(0.0, 2.16, int(early.sum()))
            sampled[~early] = rng.uniform(4.22, CYCLE_SECONDS, int((~early).sum()))
        elif phase == PHASE_COIL:
            sampled = rng.uniform(2.17, 2.65, count)
        elif phase == PHASE_RELEASE:
            sampled = rng.uniform(2.645, 2.875, count)
        elif phase == PHASE_AIR:
            sampled = rng.uniform(2.82, 3.78, count)
        else:
            sampled = rng.uniform(3.68, 4.34, count)
        times[mask] = sampled.astype(np.float32)
    return times


def sample_observations(
    rng: np.random.Generator,
    times: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    count = len(times)
    observation = np.zeros((count, OBSERVATION_SIZE), dtype=np.float32)
    observation[:, 0] = times

    gates = phase_gates(times)
    air_amount = np.clip(gates["release"] * 0.35 + gates["air"], 0.0, 1.0)
    air_progress = np.clip((times - 2.68) / 1.08, 0.0, 1.0)
    desired_roll = 2.0 * np.pi * smoothstep(air_progress)

    observation[:, 1] = rng.uniform(-3.4, 3.4, count)
    observation[:, 2] = (
        0.42
        + air_amount * (0.14 + 0.34 * np.sin(np.pi * air_progress))
        + rng.normal(0.0, 0.035, count)
    )
    observation[:, 3] = rng.normal(0.0, 0.28, count)
    observation[:, 4] = rng.uniform(0.0, 2.8, count)
    observation[:, 5] = (
        air_amount * (1.25 * np.cos(np.pi * air_progress))
        + rng.normal(0.0, 0.28, count)
    )
    observation[:, 6] = rng.normal(0.0, 0.32, count)
    observation[:, 7] = rng.normal(-0.04, 0.20, count)
    observation[:, 8] = wrap_angle(desired_roll + rng.normal(0.0, 0.58, count))
    roll_perturbation = rng.random(count) < 0.28
    observation[roll_perturbation, 8] = rng.uniform(-np.pi, np.pi, int(roll_perturbation.sum()))
    observation[:, 9] = rng.normal(0.0, 0.24, count)

    base_contact = np.where(
        phases == PHASE_COIL,
        0.88,
        np.where(
            phases == PHASE_RELEASE,
            0.28,
            np.where(phases == PHASE_AIR, 0.08, np.where(phases == PHASE_LANDING, 0.62, 0.78)),
        ),
    )
    observation[:, 10] = np.clip(base_contact + rng.normal(0.0, 0.16, count), 0.0, 1.0)
    target = np.column_stack(
        (
            np.ones(count),
            rng.normal(0.0, 0.08, count),
            rng.normal(0.0, 0.08, count),
        )
    )
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    observation[:, 11:14] = target.astype(np.float32)

    segment_x = np.linspace(-0.76, 0.76, SEGMENT_COUNT, dtype=np.float32)
    segment_indexes = np.arange(SEGMENT_COUNT, dtype=np.float32)
    wave_phase = times[:, None] * (2.0 * np.pi * TRAVEL_WAVE_HZ) - segment_indexes[None, :] * 0.72
    for segment in range(SEGMENT_COUNT):
        start = OBSERVATION_HEADER_SIZE + segment * SEGMENT_OBSERVATION_SIZE
        bend_hint = np.sin(wave_phase[:, segment])
        observation[:, start] = segment_x[segment] + rng.normal(0.0, 0.045, count)
        observation[:, start + 1] = 0.19 + gates["coil"] * 0.05 + rng.normal(0.0, 0.04, count)
        observation[:, start + 2] = bend_hint * 0.14 + rng.normal(0.0, 0.055, count)
        observation[:, start + 3] = observation[:, 4] + rng.normal(0.0, 0.30, count)
        observation[:, start + 4] = observation[:, 5] + rng.normal(0.0, 0.30, count)
        observation[:, start + 5] = bend_hint * 0.45 + rng.normal(0.0, 0.22, count)
        observation[:, start + 6] = bend_hint * 0.34 + rng.normal(0.0, 0.16, count)
        observation[:, start + 7] = bend_hint * 0.28 + observation[:, 9] + rng.normal(0.0, 0.14, count)

    # The expert only consumes a small, explicit state subset. Hold unrelated
    # kinematics at representative values so finite data cannot create spurious
    # correlations, while retaining roll/contact/target and segment pose feedback.
    observation[:, 1] = 0.0
    observation[:, 2] = 0.42
    observation[:, 3] = 0.0
    observation[:, 4] = 0.8
    observation[:, 5:7] = 0.0
    observation[:, 7] = -0.04
    for segment in range(SEGMENT_COUNT):
        start = OBSERVATION_HEADER_SIZE + segment * SEGMENT_OBSERVATION_SIZE
        observation[:, start] = segment_x[segment]
        observation[:, start + 1] = 0.19
        observation[:, start + 2 : start + 6] = 0.0

    # Reset-like states are deliberately common in the curriculum. They stop the
    # network from relying on synthetic velocity/wave correlations that are absent
    # during cold browser inference while preserving board-roll feedback examples.
    reset_like = rng.random(count) < 0.60
    observation[reset_like, 8:10] = 0.0
    observation[reset_like, 10] = base_contact[reset_like]
    observation[reset_like, 11] = 1.0
    observation[reset_like, 12:14] = 0.0
    observation[reset_like, 9] = 0.0
    for segment in range(SEGMENT_COUNT):
        start = OBSERVATION_HEADER_SIZE + segment * SEGMENT_OBSERVATION_SIZE
        observation[reset_like, start] = segment_x[segment]
        observation[reset_like, start + 1] = 0.19
        observation[reset_like, start + 2 : start + SEGMENT_OBSERVATION_SIZE] = 0.0

    return observation


def phase_gates(times: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(times, dtype=np.float64)
    return {
        "coil": smooth_window(values, 2.18, 2.645, 0.014),
        "release": smooth_window(values, 2.645, 2.87, 0.014),
        "air": smooth_window(values, 2.84, 3.76, 0.040),
        "landing": smooth_window(values, 3.72, 4.24, 0.045),
        "early_air": smooth_window(values, 2.83, 3.28, 0.045),
    }


def expert_actions(observations: np.ndarray) -> np.ndarray:
    """State-aware teacher for travel, coil, kick release, tuck, landing, and flop phases."""
    values = np.asarray(observations, dtype=np.float64)
    times = np.mod(values[:, 0], CYCLE_SECONDS)
    gates = phase_gates(times)
    coil = gates["coil"]
    release = gates["release"]
    air = gates["air"]
    landing = gates["landing"]
    early_air = gates["early_air"]

    segment_indexes = np.arange(SEGMENT_COUNT, dtype=np.float64)
    side_weight = np.where(segment_indexes < SEGMENT_COUNT / 2, -1.0, 1.0)
    target_x = np.clip(values[:, 11], -1.0, 1.0)
    wave = np.sin(
        times[:, None] * (2.0 * np.pi * TRAVEL_WAVE_HZ)
        - segment_indexes[None, :] * 0.72
        + values[:, 9:10] * 0.18
    )
    stunt_suppression = np.clip(coil + release + air, 0.0, 1.0)
    wave_amplitude = 0.54 * (1.0 - stunt_suppression) + 0.19 * landing

    progress = np.clip((times - 2.68) / 1.08, 0.0, 1.0)
    desired_roll = 2.0 * np.pi * smoothstep(progress)
    roll_error = wrap_angle(desired_roll - values[:, 8])
    kick = 0.40 * release + 0.25 * early_air + 0.095 * np.tanh(roll_error) * np.clip(release + air, 0.0, 1.0)

    segment_pitch = np.column_stack(
        [values[:, OBSERVATION_HEADER_SIZE + segment * SEGMENT_OBSERVATION_SIZE + 6] for segment in range(SEGMENT_COUNT)]
    )
    segment_yaw = np.column_stack(
        [values[:, OBSERVATION_HEADER_SIZE + segment * SEGMENT_OBSERVATION_SIZE + 7] for segment in range(SEGMENT_COUNT)]
    )
    bend = wave_amplitude[:, None] * wave * target_x[:, None]
    bend += kick[:, None] * side_weight[None, :]
    bend -= 0.10 * coil[:, None] * side_weight[None, :]
    bend -= 0.065 * segment_yaw
    bend -= 0.035 * values[:, 8:9] * side_weight[None, :] * (1.0 - release[:, None] * 0.7)
    bend = np.clip(bend, -0.72, 0.72)

    co_contraction = 0.035 + 0.75 * coil - 0.78 * release + 0.29 * air + 0.44 * landing
    contact_adjustment = (values[:, 10] - 0.5) * (0.055 * coil + 0.035 * landing)
    segment_co = co_contraction[:, None] + contact_adjustment[:, None] - 0.025 * np.abs(segment_pitch)

    action = np.empty((len(values), ACTION_SIZE), dtype=np.float64)
    action[:, 0::2] = segment_co + bend
    action[:, 1::2] = segment_co - bend
    action = np.clip(action, -0.98, 0.98)
    return np.clip(action, -0.98, 0.98).astype(np.float32)


def train_model(
    model: NormalizedStuntPolicy,
    dataset: Dataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    observations = torch.from_numpy(dataset.observations)
    actions = torch.from_numpy(dataset.actions)
    phases = torch.from_numpy(dataset.phases)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.08)
    generator = torch.Generator().manual_seed(seed + 37)

    for epoch in range(epochs):
        eligible = curriculum_indexes(phases, epoch, epochs)
        order = eligible[torch.randperm(len(eligible), generator=generator)]
        model.train()
        for start in range(0, len(order), batch_size):
            indexes = order[start : start + batch_size]
            prediction = model(observations[indexes])
            loss = torch.mean((prediction - actions[indexes]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        scheduler.step()


def curriculum_indexes(phases: torch.Tensor, epoch: int, epochs: int) -> torch.Tensor:
    progress = (epoch + 1) / epochs
    if progress <= 0.08:
        allowed = phases == PHASE_TRAVEL
    elif progress <= 0.18:
        allowed = (phases == PHASE_TRAVEL) | (phases == PHASE_COIL)
    elif progress <= 0.32:
        allowed = phases <= PHASE_RELEASE
    elif progress <= 0.50:
        allowed = phases <= PHASE_AIR
    else:
        allowed = torch.ones_like(phases, dtype=torch.bool)
    return torch.nonzero(allowed, as_tuple=False).squeeze(1)


def validation_metrics(model: NormalizedStuntPolicy, dataset: Dataset) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(dataset.observations)).cpu().numpy()
    error = prediction - dataset.actions
    phase_mse = {
        PHASE_NAMES[phase]: round(float(np.mean(error[dataset.phases == phase] ** 2)), 8)
        for phase in range(len(PHASE_NAMES))
    }
    semantics = action_semantics(prediction, dataset.phases)
    teacher_semantics = action_semantics(dataset.actions, dataset.phases)
    return {
        "validationMse": round(float(np.mean(error**2)), 8),
        "validationMae": round(float(np.mean(np.abs(error))), 8),
        "teacherAgreement": round(float(np.mean(np.abs(error) <= 0.12)), 6),
        "phaseMse": phase_mse,
        "semantics": semantics,
        "teacherSemantics": teacher_semantics,
    }


def action_semantics(actions: np.ndarray, phases: np.ndarray) -> dict[str, float]:
    bend = (actions[:, 0::2] - actions[:, 1::2]) * 0.5
    co_contraction = (actions[:, 0::2] + actions[:, 1::2]) * 0.5
    side_weight = np.where(np.arange(SEGMENT_COUNT) < SEGMENT_COUNT / 2, -1.0, 1.0)
    kick = np.mean(bend * side_weight[None, :], axis=1)

    def phase_mean(values: np.ndarray, phase: int) -> float:
        return round(float(np.mean(values[phases == phase])), 6)

    return {
        "coilCoContraction": phase_mean(np.mean(co_contraction, axis=1), PHASE_COIL),
        "releaseCoContraction": phase_mean(np.mean(co_contraction, axis=1), PHASE_RELEASE),
        "releaseKickSignal": phase_mean(kick, PHASE_RELEASE),
        "airKickSignal": phase_mean(kick, PHASE_AIR),
        "travelBendRms": round(float(np.sqrt(np.mean(bend[phases == PHASE_TRAVEL] ** 2))), 6),
    }


def assert_folded_inference_matches(
    model: NormalizedStuntPolicy,
    observations: np.ndarray,
    parameters: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    hidden_weight, hidden_bias, output_weight, output_bias = parameters
    hidden = np.tanh(observations @ hidden_weight.T + hidden_bias)
    folded = np.tanh(hidden @ output_weight.T + output_bias)
    model.eval()
    with torch.no_grad():
        expected = model(torch.from_numpy(observations)).cpu().numpy()
    difference = float(np.max(np.abs(folded - expected)))
    if not math.isfinite(difference) or difference > 2e-5:
        raise RuntimeError(f"folded raw-input network mismatch: max abs difference {difference}")


def assert_ignored_features_are_zero(hidden_weight: np.ndarray) -> None:
    ignored_weight = hidden_weight[:, IGNORED_OBSERVATION_INDICES]
    nonzero = int(np.count_nonzero(ignored_weight))
    if nonzero:
        maximum = float(np.max(np.abs(ignored_weight)))
        raise RuntimeError(f"ignored observation columns contain {nonzero} nonzero weights (max {maximum})")


def make_artifact(
    args: argparse.Namespace,
    metrics: dict[str, Any],
    parameters: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    hidden_weight, hidden_bias, output_weight, output_bias = parameters
    return {
        "schemaVersion": 1,
        "kind": MODEL_KIND,
        "modelVersion": args.model_version,
        "inputSize": OBSERVATION_SIZE,
        "hiddenSize": args.hidden_size,
        "outputSize": ACTION_SIZE,
        "activation": "tanh",
        "hiddenWeights": rounded_list(hidden_weight),
        "hiddenBias": rounded_list(hidden_bias),
        "outputWeights": rounded_list(output_weight),
        "outputBias": rounded_list(output_bias),
        "training": {
            "seed": args.seed,
            "samples": args.samples,
            "epochs": args.epochs,
            "validationMse": metrics["validationMse"],
            "teacherAgreement": metrics["teacherAgreement"],
            "teacherFeatureIndices": list(TEACHER_FEATURE_INDICES),
        },
    }


def rounded_list(values: np.ndarray) -> list[Any]:
    return np.round(np.asarray(values, dtype=np.float64), decimals=8).tolist()


def smooth_window(values: np.ndarray, start: float, end: float, edge: float) -> np.ndarray:
    return sigmoid((values - start) / edge) * sigmoid((end - values) / edge)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def wrap_angle(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


if __name__ == "__main__":
    main()
