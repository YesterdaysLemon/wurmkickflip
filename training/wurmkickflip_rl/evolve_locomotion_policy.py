from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ACTION_SIZE, POLICY_TIMESTEP, SEGMENT_COUNT
from .articulated_locomotion import (
    CONTRACT,
    DYNAMICS,
    JOINT_DAMPING,
    JOINT_LIMIT,
    JOINT_STIFFNESS,
    articulated_plant_step,
    initial_body,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "public/models/wurmkickflip_locomotion_policy.json"
DEFAULT_SUMMARY = ROOT / "training/runs/locomotion_evolution/latest-summary.json"
OBJECTIVE_VERSION = "articulated-contact-obstacle-recovery-v2"

SENSOR_NAMES = tuple(CONTRACT["sensorNames"])
INPUT_WEIGHT_NAMES = tuple(CONTRACT["inputWeightNames"])
RECURRENT_WEIGHT_NAMES = tuple(CONTRACT["recurrentWeightNames"])
OUTPUT_WEIGHT_NAMES = tuple(CONTRACT["outputWeightNames"])

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
SEGMENT_SHUFFLE = np.array(
    [0, 9, 2, 13, 4, 15, 6, 11, 8, 1, 10, 3, 12, 5, 14, 7],
    dtype=np.int64,
)


@dataclass(frozen=True)
class Scenario:
    target_x: float
    target_z: float
    friction: float
    urgency: float
    roughness: float = 0.0
    body_scale: float = 1.0
    obstacle_x: float = 0.0
    obstacle_z: float = 0.0
    obstacle_radius: float = 0.0


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
    Scenario(4.2, 0.0, 0.95, 0.72, 0.08, 1.0),
    Scenario(3.2, 3.2, 0.58, 0.88, 0.22, 0.9),
    Scenario(3.2, -3.2, 1.12, 0.64, 0.12, 1.1),
    Scenario(0.5, 4.3, 0.82, 1.0, 0.3, 1.0, 0.15, 1.65, 0.42),
    Scenario(0.5, -4.3, 0.42, 0.92, 0.35, 0.86, 0.1, -1.55, 0.46),
    Scenario(-2.2, 3.5, 1.05, 0.78, 0.18, 1.12, -0.8, 1.25, 0.38),
    Scenario(-2.2, -3.5, 0.67, 0.84, 0.26, 0.94, -0.75, -1.3, 0.4),
    Scenario(4.8, 1.2, 0.33, 1.0, 0.4, 1.06, 1.35, 0.35, 0.5),
    Scenario(4.5, -1.8, 0.76, 0.9, 0.2, 1.0, 1.4, -0.55, 0.52),
    Scenario(2.3, 4.1, 0.9, 0.82, 0.28, 0.88, 0.7, 1.45, 0.44),
    Scenario(2.3, -4.1, 0.52, 0.96, 0.32, 1.1, 0.65, -1.5, 0.44),
    Scenario(-3.8, 1.8, 1.0, 0.74, 0.16, 1.04, -1.2, 0.55, 0.48),
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
    parser.add_argument("--model-version", default="locomotion-articulated-contact-es-v2")
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
        "schemaVersion": 2,
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
        input_values = list(artifact["weights"]["input"])
        output_values = list(artifact["weights"]["output"])
        if artifact.get("schemaVersion") == 1:
            input_values.extend([0.0] * (INPUT_WEIGHT_SIZE - len(input_values)))
            output_values.extend([0.0] * (OUTPUT_WEIGHT_SIZE - len(output_values)))
        genome = np.asarray(
            artifact["initialState"]
            + input_values
            + artifact["weights"]["recurrent"]
            + output_values,
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"--warm-start is not a locomotion artifact: {error}") from error
    if genome.shape != (GENOME_SIZE,) or not np.all(np.isfinite(genome)):
        raise SystemExit(f"--warm-start must contain {GENOME_SIZE} finite genome values")
    return bound_genome(genome), {
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
        fitness, causal_gap = evaluate_selection_fitness(population, episode_steps)
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
                "meanCausalGap": round(float(np.mean(causal_gap)), 8),
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

    final_fitness = evaluate_selection_fitness(population, episode_steps)[0]
    final_best = population[int(np.argmax(final_fitness))]
    if float(np.max(final_fitness)) > best_fitness:
        best = final_best.copy()
    return bound_genome(best), history


def evaluate_selection_fitness(
    population: np.ndarray,
    episode_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    full = evaluate_population(population, episode_steps, SCENARIOS)
    causal_steps = max(90, episode_steps // 2)
    causal_scenarios = SCENARIOS[3::3]
    causal_full = evaluate_population(population, causal_steps, causal_scenarios)
    causal_shuffled = evaluate_population(
        population,
        causal_steps,
        causal_scenarios,
        intervention="shuffled",
    )
    causal_gap = causal_full - causal_shuffled
    return full + np.clip(causal_gap, -4.0, 6.0) * 2.2, causal_gap


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
    *,
    intervention: str = "full",
) -> np.ndarray:
    candidate_count = population.shape[0]
    scenario_count = len(scenarios)
    initial_state = population[:, INITIAL_SLICE]
    input_weights = population[:, INPUT_SLICE]
    recurrent_weights = population[:, RECURRENT_SLICE]
    output_weights = population[:, OUTPUT_SLICE]

    target = np.array(
        [[scenario.target_x, scenario.target_z] for scenario in scenarios],
        dtype=np.float64,
    )
    friction = np.array([scenario.friction for scenario in scenarios], dtype=np.float64)[None, :]
    urgency = np.array([scenario.urgency for scenario in scenarios], dtype=np.float64)[None, :]
    roughness = np.array([scenario.roughness for scenario in scenarios], dtype=np.float64)[None, :]
    body_scale = np.array([scenario.body_scale for scenario in scenarios], dtype=np.float64)
    obstacle_center = np.array(
        [[scenario.obstacle_x, scenario.obstacle_z] for scenario in scenarios],
        dtype=np.float64,
    )
    obstacle_radius = np.array([scenario.obstacle_radius for scenario in scenarios], dtype=np.float64)

    body_position, body_velocity = initial_body(candidate_count, scenario_count, body_scale)
    position = np.mean(body_position, axis=2)
    forward = np.zeros_like(position)
    forward[:, :, 0] = 1.0
    speed = np.zeros((candidate_count, scenario_count), dtype=np.float64)
    angular_speed = np.zeros_like(speed)
    joint = np.zeros((candidate_count, scenario_count, SEGMENT_COUNT), dtype=np.float64)
    joint_velocity = np.zeros_like(joint)
    command = np.zeros_like(joint)
    hidden = np.broadcast_to(initial_state[:, None, :], joint.shape).copy()
    contact_load = np.broadcast_to(np.clip(friction[:, :, None], 0.0, 1.0), joint.shape).copy()
    slip_speed = np.zeros_like(joint)
    obstacle_forward = np.zeros_like(joint)
    obstacle_right = np.zeros_like(joint)

    path_length = np.zeros_like(speed)
    progress = np.zeros_like(speed)
    energy_sum = np.zeros_like(speed)
    slip_sum = np.zeros_like(speed)
    obstacle_contact_steps = np.zeros_like(speed)
    alignment_sum = np.zeros_like(speed)
    near_target_steps = np.zeros_like(speed)
    command_sum = np.zeros_like(joint)
    command_square_sum = np.zeros_like(joint)

    for step in range(episode_steps):
        if step == episode_steps // 2:
            switch = obstacle_radius > 0
            target[switch, 1] *= -0.72
            target[switch, 0] += np.sign(target[switch, 1] + 1.0e-9) * 0.35
        relative = target[None, :, :] - position
        distance = np.linalg.norm(relative, axis=2)
        inverse_distance = 1.0 / np.maximum(distance, 1.0e-9)
        right_x = -forward[:, :, 1]
        right_z = forward[:, :, 0]
        target_forward = (relative[:, :, 0] * forward[:, :, 0] + relative[:, :, 1] * forward[:, :, 1]) * inverse_distance
        target_right = (relative[:, :, 0] * right_x + relative[:, :, 1] * right_z) * inverse_distance
        alignment_sum += target_forward
        near_target_steps += distance < 1.15

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
            contact_load,
            slip_speed,
            obstacle_forward,
            obstacle_right,
            input_weights,
            recurrent_weights,
            output_weights,
        )[0]
        command = segmental_network_output(
            hidden,
            joint,
            target_right,
            obstacle_right,
            contact_load,
            output_weights,
        )
        if intervention == "shuffled":
            command = command[:, :, SEGMENT_SHUFFLE]

        (
            step_distance,
            position,
            forward,
            speed,
            angular_speed,
            contact_load,
            slip_speed,
            obstacle_forward,
            obstacle_right,
        ) = articulated_plant_step(
            command.astype(np.float32).astype(np.float64),
            joint,
            joint_velocity,
            body_position,
            body_velocity,
            friction,
            roughness,
            body_scale,
            obstacle_center,
            obstacle_radius,
            POLICY_TIMESTEP,
        )
        next_distance = np.linalg.norm(target[None, :, :] - position, axis=2)
        progress += distance - next_distance
        path_length += step_distance
        energy_sum += np.mean(np.abs(command), axis=2)
        slip_sum += np.mean(slip_speed, axis=2)
        obstacle_contact_steps += np.any(
            np.abs(obstacle_forward) + np.abs(obstacle_right) > 1.0e-8,
            axis=2,
        )
        command_sum += command
        command_square_sum += command * command

    final_relative = target[None, :, :] - position
    final_distance = np.linalg.norm(final_relative, axis=2)
    final_direction = final_relative / np.maximum(final_distance[:, :, None], 1.0e-9)
    heading_alignment = np.sum(final_direction * forward, axis=2)
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
    slip_penalty = slip_sum.mean(axis=1) / episode_steps
    blocked_ratio = obstacle_contact_steps.mean(axis=1) / episode_steps
    path_alignment = alignment_sum.mean(axis=1) / episode_steps
    worst_path_alignment = np.min(alignment_sum / episode_steps, axis=1)
    near_target_ratio = near_target_steps.mean(axis=1) / episode_steps
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
        + path_alignment * 3.2
        + worst_path_alignment * 1.4
        + near_target_ratio * 5.0
        + temporal_std * 0.2
        + segment_diversity * 0.25
        - energy * 0.22
        - spin_penalty * 0.8
        - slip_penalty * 0.42
        - blocked_ratio * 0.35
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
    contact_load: np.ndarray,
    slip_speed: np.ndarray,
    obstacle_forward: np.ndarray,
    obstacle_right: np.ndarray,
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
        + iw[:, :, 13, None] * contact_load
        + iw[:, :, 14, None] * slip_speed
        + iw[:, :, 15, None] * obstacle_forward
        + iw[:, :, 16, None] * obstacle_right
        + rw[:, :, 0, None] * hidden
        + rw[:, :, 1, None] * anterior
        + rw[:, :, 2, None] * posterior
    )
    next_hidden = np.tanh(drive)
    return next_hidden, segmental_network_output(
        next_hidden,
        joint,
        target_right,
        obstacle_right,
        contact_load,
        output_weights,
    )


def segmental_network_output(
    hidden: np.ndarray,
    joint: np.ndarray,
    target_right: np.ndarray,
    obstacle_right: np.ndarray,
    contact_load: np.ndarray,
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
        + ow[:, :, 7, None] * obstacle_right
        + ow[:, :, 8, None] * contact_load
    )


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
    roughness = np.array([scenario.roughness for scenario in scenarios], dtype=np.float64)
    body_scale = np.array([scenario.body_scale for scenario in scenarios], dtype=np.float64)
    obstacle_center = np.array(
        [[scenario.obstacle_x, scenario.obstacle_z] for scenario in scenarios],
        dtype=np.float64,
    )
    obstacle_radius = np.array([scenario.obstacle_radius for scenario in scenarios], dtype=np.float64)
    body_position, body_velocity = initial_body(1, scenario_count, body_scale)
    position = np.mean(body_position[0], axis=1)
    forward = np.zeros_like(position)
    forward[:, 0] = 1.0
    speed = np.zeros(scenario_count, dtype=np.float64)
    angular_speed = np.zeros_like(speed)
    joint = np.zeros((scenario_count, SEGMENT_COUNT), dtype=np.float64)
    joint_velocity = np.zeros_like(joint)
    command = np.zeros_like(joint)
    contact_load = np.broadcast_to(np.clip(friction[:, None], 0.0, 1.0), joint.shape).copy()
    slip_speed = np.zeros_like(joint)
    obstacle_forward = np.zeros_like(joint)
    obstacle_right = np.zeros_like(joint)
    path_length = np.zeros_like(speed)
    progress = np.zeros_like(speed)
    energy_sum = 0.0
    commands: list[np.ndarray] = []
    frozen_command: np.ndarray | None = None

    for step in range(episode_steps):
        if step == episode_steps // 2:
            switch = obstacle_radius > 0
            target[switch, 1] *= -0.72
            target[switch, 0] += np.sign(target[switch, 1] + 1.0e-9) * 0.35
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
            contact_load[None, :, :],
            slip_speed[None, :, :],
            obstacle_forward[None, :, :],
            obstacle_right[None, :, :],
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
            command = neural_command[:, SEGMENT_SHUFFLE]
        else:
            command = neural_command

        (
            step_distance,
            position_batch,
            forward_batch,
            speed_batch,
            angular_speed_batch,
            contact_batch,
            slip_batch,
            obstacle_forward_batch,
            obstacle_right_batch,
        ) = articulated_plant_step(
            command.astype(np.float32).astype(np.float64)[None, :, :],
            joint[None, :, :],
            joint_velocity[None, :, :],
            body_position,
            body_velocity,
            friction[None, :],
            roughness[None, :],
            body_scale,
            obstacle_center,
            obstacle_radius,
            POLICY_TIMESTEP,
        )
        position = position_batch[0]
        forward = forward_batch[0]
        speed = speed_batch[0]
        angular_speed = angular_speed_batch[0]
        contact_load = contact_batch[0]
        slip_speed = slip_batch[0]
        obstacle_forward = obstacle_forward_batch[0]
        obstacle_right = obstacle_right_batch[0]
        progress += distance - np.linalg.norm(target - position, axis=1)
        step_distance = step_distance[0]
        path_length += step_distance
        energy_sum += float(np.mean(np.abs(command)))
        commands.append(command.copy())

    final_relative = target - position
    final_distance = np.linalg.norm(final_relative, axis=1)
    final_direction = final_relative / np.maximum(final_distance[:, None], 1.0e-9)
    heading_alignment = float(np.mean(np.sum(final_direction * forward, axis=1)))
    command_trace = np.stack(commands)
    temporal_std = float(np.mean(np.std(command_trace, axis=0)))
    segment_diversity = float(np.mean(np.std(np.mean(command_trace, axis=0), axis=1)))
    efficiency = float(np.mean(np.maximum(progress, 0.0) / np.maximum(path_length, 0.15)))
    reached_ratio = float(np.mean(final_distance < 0.72))
    mean_speed = float(np.mean(path_length) / (episode_steps * POLICY_TIMESTEP))
    mean_energy = energy_sum / episode_steps
    fitness = (
        float(evaluate_population(genome[None, :], episode_steps, scenarios)[0])
        if mode == "full"
        else float(np.mean(progress))
    )
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
        Scenario(
            scenario.target_x,
            scenario.target_z,
            0.0,
            scenario.urgency,
            scenario.roughness,
            scenario.body_scale,
            scenario.obstacle_x,
            scenario.obstacle_z,
            0.0,
        )
        for scenario in SCENARIOS
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
        "schemaVersion": 2,
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
            "version": CONTRACT["plantVersion"],
            "timestep": POLICY_TIMESTEP,
            "jointStiffness": JOINT_STIFFNESS,
            "jointDamping": JOINT_DAMPING,
            "jointLimit": JOINT_LIMIT,
            **DYNAMICS,
        },
        "training": {
            "algorithm": "elitist-mutation-evolution",
            "seed": seed,
            "generations": generations,
            "populationSize": population_size,
            "eliteCount": elite_count,
            "episodeSteps": episode_steps,
            "scenarioCount": len(SCENARIOS),
            "domainRandomization": [
                "obstacles",
                "spatial-friction",
                "body-scale",
                "target-switches",
                "contact-loss",
            ],
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
