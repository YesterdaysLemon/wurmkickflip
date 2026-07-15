from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ACTION_SIZE, POLICY_TIMESTEP, SEGMENT_COUNT


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "public/models/wurmkickflip_locomotion_policy.json"
DEFAULT_SUMMARY = ROOT / "training/runs/locomotion_evolution/latest-summary.json"
OBJECTIVE_VERSION = "risk-sensitive-bottom-two-v1"

JOINT_STIFFNESS = 22.0
JOINT_DAMPING = 7.0
JOINT_LIMIT = 1.15
WAVE_GAIN = -1.35
MAXIMUM_FORWARD_ACCELERATION = 2.4
LINEAR_DRAG = 0.75
INVERSE_TRACTION_DRAG = 0.18
TRACTION_MINIMUM = 0.0
TRACTION_MAXIMUM = 1.2
INVERSE_TRACTION_FLOOR = 0.12
MINIMUM_FORWARD_SPEED = -1.45
MAXIMUM_FORWARD_SPEED = 1.45
FRONT_TURN_BASE = 1.15
FRONT_TURN_SPEED_GAIN = 1.65
ANGULAR_DRAG = 2.8
MINIMUM_ANGULAR_SPEED = -3.0
MAXIMUM_ANGULAR_SPEED = 3.0

SENSOR_NAMES = (
    "targetForward",
    "targetRight",
    "targetDistance",
    "forwardSpeed",
    "angularSpeed",
    "terrainFriction",
    "urgency",
)
INPUT_WEIGHT_NAMES = (
    "bias",
    "segmentPosition",
    "targetForward",
    "targetRight",
    "targetRightByPosition",
    "targetDistance",
    "forwardSpeed",
    "angularSpeed",
    "terrainFriction",
    "urgency",
    "segmentBend",
    "segmentBendVelocity",
    "previousCommand",
)
RECURRENT_WEIGHT_NAMES = ("self", "anteriorNeighbor", "posteriorNeighbor")
OUTPUT_WEIGHT_NAMES = (
    "bias",
    "self",
    "anteriorNeighbor",
    "posteriorNeighbor",
    "targetRight",
    "targetRightByPosition",
    "segmentBend",
)

INITIAL_STATE_SIZE = SEGMENT_COUNT
INPUT_WEIGHT_SIZE = len(INPUT_WEIGHT_NAMES)
RECURRENT_WEIGHT_SIZE = len(RECURRENT_WEIGHT_NAMES)
OUTPUT_WEIGHT_SIZE = len(OUTPUT_WEIGHT_NAMES)
GENOME_SIZE = INITIAL_STATE_SIZE + INPUT_WEIGHT_SIZE + RECURRENT_WEIGHT_SIZE + OUTPUT_WEIGHT_SIZE

INITIAL_SLICE = slice(0, INITIAL_STATE_SIZE)
INPUT_SLICE = slice(INITIAL_SLICE.stop, INITIAL_SLICE.stop + INPUT_WEIGHT_SIZE)
RECURRENT_SLICE = slice(INPUT_SLICE.stop, INPUT_SLICE.stop + RECURRENT_WEIGHT_SIZE)
OUTPUT_SLICE = slice(RECURRENT_SLICE.stop, RECURRENT_SLICE.stop + OUTPUT_WEIGHT_SIZE)

SEGMENT_POSITIONS = np.linspace(-1.0, 1.0, SEGMENT_COUNT, dtype=np.float64)


@dataclass(frozen=True)
class Scenario:
    target_x: float
    target_z: float
    friction: float
    urgency: float


@dataclass(frozen=True)
class RolloutMetrics:
    fitness: float
    progress: float
    path_efficiency: float
    final_distance: float
    mean_speed: float
    actuator_temporal_std: float
    segment_diversity: float
    mean_energy: float
    reached_ratio: float
    displacement: float
    heading_alignment: float


SCENARIOS = (
    Scenario(4.2, 0.0, 0.95, 0.72),
    Scenario(3.2, 3.2, 0.58, 0.88),
    Scenario(3.2, -3.2, 1.12, 0.64),
    Scenario(0.5, 4.3, 0.82, 1.0),
    Scenario(0.5, -4.3, 0.42, 0.92),
    Scenario(-2.2, 3.5, 1.05, 0.78),
    Scenario(-2.2, -3.5, 0.67, 0.84),
    Scenario(4.8, 1.2, 0.33, 1.0),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evolve a clock-free, segment-local recurrent neural controller. "
            "The plant moves only through antagonistic segment actuator work."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--generations", type=int, default=140)
    parser.add_argument("--population-size", type=int, default=160)
    parser.add_argument("--elite-count", type=int, default=20)
    parser.add_argument("--episode-steps", type=int, default=480)
    parser.add_argument("--model-version", default="locomotion-segmental-es-v1")
    parser.add_argument(
        "--warm-start",
        type=Path,
        help="Browser locomotion artifact used to seed one third of the population.",
    )
    args = parser.parse_args()

    validate_arguments(args)
    warm_start, warm_start_metadata = load_warm_start(args.warm_start)
    best, history = evolve(
        seed=args.seed,
        generations=args.generations,
        population_size=args.population_size,
        elite_count=args.elite_count,
        episode_steps=args.episode_steps,
        warm_start=warm_start,
    )
    evaluation = evaluate_genome(best, args.episode_steps, SCENARIOS)
    ablations = evaluate_ablations(best, args.episode_steps)
    trace_hash = deterministic_trace_hash(best, args.episode_steps)
    artifact = make_artifact(
        best,
        model_version=args.model_version,
        seed=args.seed,
        generations=args.generations,
        population_size=args.population_size,
        elite_count=args.elite_count,
        episode_steps=args.episode_steps,
        metrics=evaluation,
        ablations=ablations,
        trace_hash=trace_hash,
        warm_start_metadata=warm_start_metadata,
    )
    summary = {
        "schemaVersion": 1,
        "kind": "wurmkickflip.locomotionEvolutionSummary",
        "modelVersion": args.model_version,
        "seed": args.seed,
        "generations": args.generations,
        "populationSize": args.population_size,
        "eliteCount": args.elite_count,
        "episodeSteps": args.episode_steps,
        "genomeSize": GENOME_SIZE,
        "objectiveVersion": OBJECTIVE_VERSION,
        "actuatorPrecision": "float32-plant-command",
        **({"warmStart": warm_start_metadata} if warm_start_metadata else {}),
        "metrics": metrics_json(evaluation),
        "ablations": ablations,
        "deterministicTraceHash": trace_hash,
        "history": history,
    }

    write_json(args.out, artifact, compact=True)
    write_json(args.summary, summary, compact=False)
    print(f"wrote {args.out}")
    print(f"wrote {args.summary}")
    print(
        "fitness={fitness:.6f} progress={progress:.6f} efficiency={efficiency:.6f} "
        "temporalStd={temporal:.6f} diversity={diversity:.6f}".format(
            fitness=evaluation.fitness,
            progress=evaluation.progress,
            efficiency=evaluation.path_efficiency,
            temporal=evaluation.actuator_temporal_std,
            diversity=evaluation.segment_diversity,
        )
    )
    print(
        "ablations full={full:.6f} zero={zero:.6f} frozen={frozen:.6f} "
        "shuffled={shuffled:.6f} noFrictionDisplacement={no_friction:.9f}".format(
            full=ablations["fullProgress"],
            zero=ablations["zeroProgress"],
            frozen=ablations["frozenProgress"],
            shuffled=ablations["shuffledProgress"],
            no_friction=ablations["noFrictionDisplacement"],
        )
    )


def validate_arguments(args: argparse.Namespace) -> None:
    if args.generations <= 0:
        raise SystemExit("--generations must be greater than zero")
    if args.population_size < 4:
        raise SystemExit("--population-size must be at least four")
    if not 1 <= args.elite_count < args.population_size:
        raise SystemExit("--elite-count must be positive and smaller than the population")
    if args.episode_steps < 60:
        raise SystemExit("--episode-steps must be at least 60")
    if args.warm_start is not None and not args.warm_start.is_file():
        raise SystemExit(f"--warm-start does not exist: {args.warm_start}")


def load_warm_start(path: Path | None) -> tuple[np.ndarray | None, dict[str, str] | None]:
    if path is None:
        return None, None
    payload = path.read_bytes()
    canonical_payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    artifact = json.loads(payload)
    try:
        genome = np.asarray(
            artifact["initialState"]
            + artifact["weights"]["input"]
            + artifact["weights"]["recurrent"]
            + artifact["weights"]["output"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"--warm-start is not a locomotion artifact: {error}") from error
    if genome.shape != (GENOME_SIZE,) or not np.all(np.isfinite(genome)):
        raise SystemExit(f"--warm-start must contain {GENOME_SIZE} finite genome values")
    return bound_genome(genome), {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(canonical_payload).hexdigest(),
        "modelVersion": str(artifact.get("modelVersion", "unknown")),
    }


def evolve(
    *,
    seed: int,
    generations: int,
    population_size: int,
    elite_count: int,
    episode_steps: int,
    warm_start: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    rng = np.random.default_rng(seed)
    population = initial_population(population_size, rng)
    if warm_start is not None:
        population[0] = warm_start
        warm_count = min(population_size, max(12, population_size // 3))
        for warm_index in range(1, warm_count):
            scale = 0.06 + 0.24 * (warm_index / max(population_size // 3 - 1, 1))
            mutation = rng.normal(0.0, scale, GENOME_SIZE)
            mutation[INITIAL_SLICE] *= 0.5
            population[warm_index] = bound_genome(warm_start + mutation)
    best = population[0].copy()
    best_fitness = -np.inf
    history: list[dict[str, float | int]] = []

    for generation in range(generations):
        fitness = evaluate_population(population, episode_steps, SCENARIOS)
        order = np.argsort(fitness)[::-1]
        population = population[order]
        fitness = fitness[order]
        if float(fitness[0]) > best_fitness:
            best = population[0].copy()
            best_fitness = float(fitness[0])

        history.append(
            {
                "generation": generation,
                "bestFitness": round(best_fitness, 8),
                "generationBestFitness": round(float(fitness[0]), 8),
                "meanFitness": round(float(np.mean(fitness)), 8),
            }
        )

        elites = population[:elite_count].copy()
        progress = generation / max(generations - 1, 1)
        mutation_scale = 0.36 * (1.0 - progress) + 0.055
        children = [best.copy(), *[elite.copy() for elite in elites[: max(1, elite_count // 3)]]]
        while len(children) < population_size:
            rank = int(min(elite_count - 1, rng.exponential(max(1.0, elite_count * 0.22))))
            parent = elites[rank]
            scale = mutation_scale * float(rng.choice([0.45, 0.8, 1.0, 1.6], p=[0.18, 0.34, 0.34, 0.14]))
            mutation = rng.normal(0.0, scale, GENOME_SIZE)
            mutation[INITIAL_SLICE] *= 0.65
            child = parent + mutation
            if rng.random() < 0.12:
                donor = elites[int(rng.integers(0, elite_count))]
                mask = rng.random(GENOME_SIZE) < 0.14
                child[mask] = donor[mask] + mutation[mask] * 0.35
            children.append(bound_genome(child))
        population = np.stack(children[:population_size])

    final_fitness = evaluate_population(population, episode_steps, SCENARIOS)
    final_best = population[int(np.argmax(final_fitness))]
    if float(np.max(final_fitness)) > best_fitness:
        best = final_best.copy()
    return bound_genome(best), history


def initial_population(population_size: int, rng: np.random.Generator) -> np.ndarray:
    population = rng.normal(0.0, 0.82, (population_size, GENOME_SIZE))
    population[:, INITIAL_SLICE] = rng.uniform(-0.72, 0.72, (population_size, SEGMENT_COUNT))
    population[:, RECURRENT_SLICE] *= 1.5
    population[:, OUTPUT_SLICE] *= 1.2
    return np.stack([bound_genome(genome) for genome in population])


def bound_genome(genome: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(genome, dtype=np.float64), -4.0, 4.0)
    bounded[INITIAL_SLICE] = np.clip(bounded[INITIAL_SLICE], -0.98, 0.98)
    return bounded


def evaluate_population(
    population: np.ndarray,
    episode_steps: int,
    scenarios: tuple[Scenario, ...],
) -> np.ndarray:
    candidate_count = population.shape[0]
    scenario_count = len(scenarios)
    initial_state = population[:, INITIAL_SLICE]
    input_weights = population[:, INPUT_SLICE]
    recurrent_weights = population[:, RECURRENT_SLICE]
    output_weights = population[:, OUTPUT_SLICE]

    target = np.array([[scenario.target_x, scenario.target_z] for scenario in scenarios], dtype=np.float64)
    friction = np.array([scenario.friction for scenario in scenarios], dtype=np.float64)[None, :]
    urgency = np.array([scenario.urgency for scenario in scenarios], dtype=np.float64)[None, :]
    initial_distance = np.linalg.norm(target, axis=1)[None, :]

    position = np.zeros((candidate_count, scenario_count, 2), dtype=np.float64)
    forward = np.zeros_like(position)
    forward[:, :, 0] = 1.0
    speed = np.zeros((candidate_count, scenario_count), dtype=np.float64)
    angular_speed = np.zeros_like(speed)
    joint = np.zeros((candidate_count, scenario_count, SEGMENT_COUNT), dtype=np.float64)
    joint_velocity = np.zeros_like(joint)
    command = np.zeros_like(joint)
    hidden = np.broadcast_to(initial_state[:, None, :], joint.shape).copy()

    path_length = np.zeros_like(speed)
    energy_sum = np.zeros_like(speed)
    command_sum = np.zeros_like(joint)
    command_square_sum = np.zeros_like(joint)

    for _step in range(episode_steps):
        relative = target[None, :, :] - position
        distance = np.linalg.norm(relative, axis=2)
        inverse_distance = 1.0 / np.maximum(distance, 1.0e-9)
        right_x = -forward[:, :, 1]
        right_z = forward[:, :, 0]
        target_forward = (relative[:, :, 0] * forward[:, :, 0] + relative[:, :, 1] * forward[:, :, 1]) * inverse_distance
        target_right = (relative[:, :, 0] * right_x + relative[:, :, 1] * right_z) * inverse_distance

        hidden = segmental_network_step(
            hidden,
            command,
            joint,
            joint_velocity,
            target_forward,
            target_right,
            np.minimum(distance / 5.0, 1.5),
            speed,
            angular_speed,
            friction,
            urgency,
            input_weights,
            recurrent_weights,
            output_weights,
        )[0]
        command = segmental_network_output(hidden, joint, target_right, output_weights)

        step_distance = actuator_plant_step(
            command.astype(np.float32).astype(np.float64),
            joint,
            joint_velocity,
            speed,
            angular_speed,
            forward,
            position,
            friction,
            POLICY_TIMESTEP,
        )
        path_length += step_distance
        energy_sum += np.mean(np.abs(command), axis=2)
        command_sum += command
        command_square_sum += command * command

    final_relative = target[None, :, :] - position
    final_distance = np.linalg.norm(final_relative, axis=2)
    final_direction = final_relative / np.maximum(final_distance[:, :, None], 1.0e-9)
    heading_alignment = np.sum(final_direction * forward, axis=2)
    progress = initial_distance - final_distance
    mean_progress = np.mean(progress, axis=1)
    worst_progress = np.min(progress, axis=1)
    efficiency = np.mean(np.maximum(progress, 0.0) / np.maximum(path_length, 0.15), axis=1)
    reached = np.mean(final_distance < 0.72, axis=1)
    mean_speed = path_length.mean(axis=1) / (episode_steps * POLICY_TIMESTEP)
    energy = energy_sum.mean(axis=1) / episode_steps
    mean_command = command_sum / episode_steps
    variance = np.maximum(command_square_sum / episode_steps - mean_command * mean_command, 0.0)
    temporal_std = np.mean(np.sqrt(variance), axis=(1, 2))
    segment_diversity = np.mean(np.std(mean_command, axis=2), axis=1)
    spin_penalty = np.mean(np.maximum(np.abs(angular_speed) - 2.4, 0.0), axis=1)
    covered_progress = np.mean(np.clip(progress, -1.0, 1.6), axis=1)
    bottom_two_progress = np.mean(np.partition(progress, 1, axis=1)[:, :2], axis=1)
    approach_ratio = np.mean(final_distance < 2.4, axis=1)
    close_ratio = np.mean(final_distance < 1.35, axis=1)

    return (
        mean_progress * 3.8
        + worst_progress * 8.5
        + bottom_two_progress * 5.0
        + covered_progress * 1.8
        + efficiency * 1.8
        + approach_ratio * 3.5
        + close_ratio * 5.5
        + reached * 13.0
        + np.minimum(mean_speed, 1.0) * 0.55
        + np.mean(heading_alignment, axis=1) * 0.8
        + np.min(heading_alignment, axis=1) * 0.45
        + temporal_std * 0.2
        + segment_diversity * 0.25
        - energy * 0.22
        - spin_penalty * 0.8
    )


def segmental_network_step(
    hidden: np.ndarray,
    previous_command: np.ndarray,
    joint: np.ndarray,
    joint_velocity: np.ndarray,
    target_forward: np.ndarray,
    target_right: np.ndarray,
    target_distance: np.ndarray,
    speed: np.ndarray,
    angular_speed: np.ndarray,
    friction: np.ndarray,
    urgency: np.ndarray,
    input_weights: np.ndarray,
    recurrent_weights: np.ndarray,
    output_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    anterior = np.pad(hidden[:, :, :-1], ((0, 0), (0, 0), (1, 0)))
    posterior = np.pad(hidden[:, :, 1:], ((0, 0), (0, 0), (0, 1)))
    iw = input_weights[:, None, :]
    rw = recurrent_weights[:, None, :]
    drive = (
        iw[:, :, 0, None]
        + iw[:, :, 1, None] * SEGMENT_POSITIONS
        + iw[:, :, 2, None] * target_forward[:, :, None]
        + iw[:, :, 3, None] * target_right[:, :, None]
        + iw[:, :, 4, None] * target_right[:, :, None] * SEGMENT_POSITIONS
        + iw[:, :, 5, None] * target_distance[:, :, None]
        + iw[:, :, 6, None] * speed[:, :, None]
        + iw[:, :, 7, None] * angular_speed[:, :, None]
        + iw[:, :, 8, None] * friction[:, :, None]
        + iw[:, :, 9, None] * urgency[:, :, None]
        + iw[:, :, 10, None] * joint
        + iw[:, :, 11, None] * joint_velocity
        + iw[:, :, 12, None] * previous_command
        + rw[:, :, 0, None] * hidden
        + rw[:, :, 1, None] * anterior
        + rw[:, :, 2, None] * posterior
    )
    next_hidden = np.tanh(drive)
    return next_hidden, segmental_network_output(next_hidden, joint, target_right, output_weights)


def segmental_network_output(
    hidden: np.ndarray,
    joint: np.ndarray,
    target_right: np.ndarray,
    output_weights: np.ndarray,
) -> np.ndarray:
    anterior = np.pad(hidden[:, :, :-1], ((0, 0), (0, 0), (1, 0)))
    posterior = np.pad(hidden[:, :, 1:], ((0, 0), (0, 0), (0, 1)))
    ow = output_weights[:, None, :]
    return np.tanh(
        ow[:, :, 0, None]
        + ow[:, :, 1, None] * hidden
        + ow[:, :, 2, None] * anterior
        + ow[:, :, 3, None] * posterior
        + ow[:, :, 4, None] * target_right[:, :, None]
        + ow[:, :, 5, None] * target_right[:, :, None] * SEGMENT_POSITIONS
        + ow[:, :, 6, None] * joint
    )


def actuator_plant_step(
    command: np.ndarray,
    joint: np.ndarray,
    joint_velocity: np.ndarray,
    speed: np.ndarray,
    angular_speed: np.ndarray,
    forward: np.ndarray,
    position: np.ndarray,
    friction: np.ndarray,
    timestep: float,
) -> np.ndarray:
    joint_velocity += (
        (command - joint) * JOINT_STIFFNESS - joint_velocity * JOINT_DAMPING
    ) * timestep
    joint[:] = np.clip(joint + joint_velocity * timestep, -JOINT_LIMIT, JOINT_LIMIT)

    wave_work = np.mean(
        (joint[:, :, 1:] - joint[:, :, :-1])
        * 0.5
        * (joint_velocity[:, :, 1:] + joint_velocity[:, :, :-1]),
        axis=2,
    )
    traction = np.clip(friction, TRACTION_MINIMUM, TRACTION_MAXIMUM)
    forward_acceleration = np.clip(
        WAVE_GAIN * wave_work,
        -MAXIMUM_FORWARD_ACCELERATION,
        MAXIMUM_FORWARD_ACCELERATION,
    ) * traction
    speed += (
        forward_acceleration
        - (
            LINEAR_DRAG
            + INVERSE_TRACTION_DRAG / np.maximum(traction, INVERSE_TRACTION_FLOOR)
        )
        * speed
    ) * timestep
    speed[:] = np.clip(speed, MINIMUM_FORWARD_SPEED, MAXIMUM_FORWARD_SPEED)

    front_weights = np.linspace(1.0, 0.0, SEGMENT_COUNT, dtype=np.float64)
    front_bias = np.mean(joint * front_weights, axis=2)
    angular_acceleration = front_bias * (
        FRONT_TURN_BASE + FRONT_TURN_SPEED_GAIN * np.abs(speed)
    )
    angular_speed += (angular_acceleration - ANGULAR_DRAG * angular_speed) * timestep
    angular_speed[:] = np.clip(
        angular_speed, MINIMUM_ANGULAR_SPEED, MAXIMUM_ANGULAR_SPEED
    )

    right_x = -forward[:, :, 1]
    right_z = forward[:, :, 0]
    forward[:, :, 0] += right_x * angular_speed * timestep
    forward[:, :, 1] += right_z * angular_speed * timestep
    forward_length = np.maximum(np.linalg.norm(forward, axis=2), 1.0e-9)
    forward[:, :, 0] /= forward_length
    forward[:, :, 1] /= forward_length

    delta_x = forward[:, :, 0] * speed * timestep
    delta_z = forward[:, :, 1] * speed * timestep
    position[:, :, 0] += delta_x
    position[:, :, 1] += delta_z
    return np.hypot(delta_x, delta_z)


def evaluate_genome(
    genome: np.ndarray,
    episode_steps: int,
    scenarios: tuple[Scenario, ...],
    *,
    mode: str = "full",
) -> RolloutMetrics:
    scenario_count = len(scenarios)
    hidden = np.broadcast_to(genome[INITIAL_SLICE][None, :], (scenario_count, SEGMENT_COUNT)).copy()
    input_weights = genome[INPUT_SLICE][None, :]
    recurrent_weights = genome[RECURRENT_SLICE][None, :]
    output_weights = genome[OUTPUT_SLICE][None, :]
    target = np.array([[scenario.target_x, scenario.target_z] for scenario in scenarios], dtype=np.float64)
    friction = np.array([scenario.friction for scenario in scenarios], dtype=np.float64)
    urgency = np.array([scenario.urgency for scenario in scenarios], dtype=np.float64)
    initial_distance = np.linalg.norm(target, axis=1)
    position = np.zeros((scenario_count, 2), dtype=np.float64)
    forward = np.zeros_like(position)
    forward[:, 0] = 1.0
    speed = np.zeros(scenario_count, dtype=np.float64)
    angular_speed = np.zeros_like(speed)
    joint = np.zeros((scenario_count, SEGMENT_COUNT), dtype=np.float64)
    joint_velocity = np.zeros_like(joint)
    command = np.zeros_like(joint)
    path_length = np.zeros_like(speed)
    energy_sum = 0.0
    commands: list[np.ndarray] = []
    frozen_command: np.ndarray | None = None
    shuffle = np.array([0, 9, 2, 13, 4, 15, 6, 11, 8, 1, 10, 3, 12, 5, 14, 7], dtype=np.int64)

    for step in range(episode_steps):
        relative = target - position
        distance = np.linalg.norm(relative, axis=1)
        inverse_distance = 1.0 / np.maximum(distance, 1.0e-9)
        right = np.column_stack((-forward[:, 1], forward[:, 0]))
        target_forward = np.sum(relative * forward, axis=1) * inverse_distance
        target_right = np.sum(relative * right, axis=1) * inverse_distance
        hidden_batch, action_batch = segmental_network_step(
            hidden[None, :, :],
            command[None, :, :],
            joint[None, :, :],
            joint_velocity[None, :, :],
            target_forward[None, :],
            target_right[None, :],
            np.minimum(distance[None, :] / 5.0, 1.5),
            speed[None, :],
            angular_speed[None, :],
            friction[None, :],
            urgency[None, :],
            input_weights,
            recurrent_weights,
            output_weights,
        )
        hidden = hidden_batch[0]
        neural_command = action_batch[0]
        if mode == "zero":
            command = np.zeros_like(neural_command)
        elif mode == "frozen":
            if frozen_command is None and step == 24:
                frozen_command = neural_command.copy()
            command = neural_command if frozen_command is None else frozen_command
        elif mode == "shuffled":
            command = neural_command[:, shuffle]
        else:
            command = neural_command

        step_distance = actuator_plant_step(
            command.astype(np.float32).astype(np.float64)[None, :, :],
            joint[None, :, :],
            joint_velocity[None, :, :],
            speed[None, :],
            angular_speed[None, :],
            forward[None, :, :],
            position[None, :, :],
            friction[None, :],
            POLICY_TIMESTEP,
        )[0]
        path_length += step_distance
        energy_sum += float(np.mean(np.abs(command)))
        commands.append(command.copy())

    final_relative = target - position
    final_distance = np.linalg.norm(final_relative, axis=1)
    final_direction = final_relative / np.maximum(final_distance[:, None], 1.0e-9)
    heading_alignment = float(np.mean(np.sum(final_direction * forward, axis=1)))
    progress = initial_distance - final_distance
    command_trace = np.stack(commands)
    temporal_std = float(np.mean(np.std(command_trace, axis=0)))
    segment_diversity = float(np.mean(np.std(np.mean(command_trace, axis=0), axis=1)))
    efficiency = float(np.mean(np.maximum(progress, 0.0) / np.maximum(path_length, 0.15)))
    reached_ratio = float(np.mean(final_distance < 0.72))
    mean_speed = float(np.mean(path_length) / (episode_steps * POLICY_TIMESTEP))
    mean_energy = energy_sum / episode_steps
    fitness = float(evaluate_population(genome[None, :], episode_steps, scenarios)[0]) if mode == "full" else float(np.mean(progress))
    return RolloutMetrics(
        fitness=fitness,
        progress=float(np.mean(progress)),
        path_efficiency=efficiency,
        final_distance=float(np.mean(final_distance)),
        mean_speed=mean_speed,
        actuator_temporal_std=temporal_std,
        segment_diversity=segment_diversity,
        mean_energy=mean_energy,
        reached_ratio=reached_ratio,
        displacement=float(np.mean(np.linalg.norm(position, axis=1))),
        heading_alignment=heading_alignment,
    )


def evaluate_ablations(genome: np.ndarray, episode_steps: int) -> dict[str, float]:
    full = evaluate_genome(genome, episode_steps, SCENARIOS)
    zero = evaluate_genome(genome, episode_steps, SCENARIOS, mode="zero")
    frozen = evaluate_genome(genome, episode_steps, SCENARIOS, mode="frozen")
    shuffled = evaluate_genome(genome, episode_steps, SCENARIOS, mode="shuffled")
    no_friction_scenarios = tuple(
        Scenario(scenario.target_x, scenario.target_z, 0.0, scenario.urgency) for scenario in SCENARIOS
    )
    no_friction = evaluate_genome(genome, episode_steps, no_friction_scenarios)
    return {
        "fullProgress": round(full.progress, 8),
        "zeroProgress": round(zero.progress, 8),
        "frozenProgress": round(frozen.progress, 8),
        "shuffledProgress": round(shuffled.progress, 8),
        "noFrictionDisplacement": round(no_friction.displacement, 10),
    }


def deterministic_trace_hash(genome: np.ndarray, episode_steps: int) -> str:
    metrics = evaluate_genome(genome, episode_steps, SCENARIOS)
    payload = json.dumps(metrics_json(metrics), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_artifact(
    genome: np.ndarray,
    *,
    model_version: str,
    seed: int,
    generations: int,
    population_size: int,
    elite_count: int,
    episode_steps: int,
    metrics: RolloutMetrics,
    ablations: dict[str, float],
    trace_hash: str,
    warm_start_metadata: dict[str, str] | None,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "kind": "wurmkickflip.locomotionPolicy",
        "modelVersion": model_version,
        "architecture": "segmental-recurrent-tanh",
        "segmentCount": SEGMENT_COUNT,
        "actionSize": ACTION_SIZE,
        "sensorNames": list(SENSOR_NAMES),
        "segmentPositions": rounded_vector(SEGMENT_POSITIONS),
        "initialState": rounded_vector(genome[INITIAL_SLICE]),
        "weights": {
            "inputNames": list(INPUT_WEIGHT_NAMES),
            "input": rounded_vector(genome[INPUT_SLICE]),
            "recurrentNames": list(RECURRENT_WEIGHT_NAMES),
            "recurrent": rounded_vector(genome[RECURRENT_SLICE]),
            "outputNames": list(OUTPUT_WEIGHT_NAMES),
            "output": rounded_vector(genome[OUTPUT_SLICE]),
        },
        "plant": {
            "timestep": POLICY_TIMESTEP,
            "jointStiffness": JOINT_STIFFNESS,
            "jointDamping": JOINT_DAMPING,
            "jointLimit": JOINT_LIMIT,
            "waveGain": WAVE_GAIN,
            "maximumForwardAcceleration": MAXIMUM_FORWARD_ACCELERATION,
            "linearDrag": LINEAR_DRAG,
            "inverseTractionDrag": INVERSE_TRACTION_DRAG,
            "tractionMinimum": TRACTION_MINIMUM,
            "tractionMaximum": TRACTION_MAXIMUM,
            "inverseTractionFloor": INVERSE_TRACTION_FLOOR,
            "minimumForwardSpeed": MINIMUM_FORWARD_SPEED,
            "maximumForwardSpeed": MAXIMUM_FORWARD_SPEED,
            "frontTurnBase": FRONT_TURN_BASE,
            "frontTurnSpeedGain": FRONT_TURN_SPEED_GAIN,
            "angularDrag": ANGULAR_DRAG,
            "minimumAngularSpeed": MINIMUM_ANGULAR_SPEED,
            "maximumAngularSpeed": MAXIMUM_ANGULAR_SPEED,
        },
        "training": {
            "algorithm": "elitist-mutation-evolution",
            "seed": seed,
            "generations": generations,
            "populationSize": population_size,
            "eliteCount": elite_count,
            "episodeSteps": episode_steps,
            "scenarioCount": len(SCENARIOS),
            "objectiveVersion": OBJECTIVE_VERSION,
            "actuatorPrecision": "float32-plant-command",
            **({"warmStart": warm_start_metadata} if warm_start_metadata else {}),
            **metrics_json(metrics),
            "ablations": ablations,
            "deterministicTraceHash": trace_hash,
        },
    }


def metrics_json(metrics: RolloutMetrics) -> dict[str, float]:
    return {
        "fitness": round(metrics.fitness, 8),
        "meanTargetProgress": round(metrics.progress, 8),
        "pathEfficiency": round(metrics.path_efficiency, 8),
        "meanFinalDistance": round(metrics.final_distance, 8),
        "meanSpeed": round(metrics.mean_speed, 8),
        "actuatorTemporalStd": round(metrics.actuator_temporal_std, 8),
        "segmentDiversity": round(metrics.segment_diversity, 8),
        "meanEnergy": round(metrics.mean_energy, 8),
        "reachedRatio": round(metrics.reached_ratio, 8),
        "meanDisplacement": round(metrics.displacement, 8),
        "meanHeadingAlignment": round(metrics.heading_alignment, 8),
    }


def rounded_vector(values: np.ndarray) -> list[float]:
    return [round(float(value), 8) for value in values]


def write_json(path: Path, value: dict[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(value, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(value, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
