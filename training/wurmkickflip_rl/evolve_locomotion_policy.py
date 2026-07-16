from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
DEFAULT_ARTIFACT = ROOT / "training/runs/locomotion_evolution/candidate-v2.json"
DEFAULT_SUMMARY = ROOT / "training/runs/locomotion_evolution/latest-summary.json"
DEFAULT_V3_ARTIFACT = ROOT / "training/runs/locomotion_head_leading_v3/candidate.json"
DEFAULT_V3_SUMMARY = ROOT / "training/runs/locomotion_head_leading_v3/summary.json"
OBJECTIVE_V2_VERSION = "articulated-contact-obstacle-recovery-v2"
OBJECTIVE_V3_VERSION = "articulated-head-leading-transient-recovery-v3"
# Kept as the v2 value for callers that reproduce the legacy objective.
OBJECTIVE_VERSION = OBJECTIVE_V2_VERSION
ObjectiveMode = Literal["v2", "v3"]
CombinedRecoveryMode = Literal["bounded-480", "recentered-900"]
PerturbationDomain = Literal[
    "nominal",
    "slick-window",
    "body-shove",
    "combined-recovery",
]

SLICK_FRICTION = 0.035
SHOVE_LATERAL_SPEED = 1.35
SHOVE_BACKWARD_SPEED = 0.28
COMBINED_SHOVE_LATERAL_SPEED = 1.1
COMBINED_TRACTION_SCALE = 1.0 / 15.0
COMBINED_NUMB_SEGMENT = 7
V3_COMBINED_RECOVERY_STEPS = 900
DEFAULT_COMBINED_RECOVERY_MODE: CombinedRecoveryMode = "bounded-480"
RECOVERY_SPEED_FLOOR = 0.04
BOUNDED_COMBINED_CONTROL_PROGRESS_FLOOR = 4.36
V3_CAUSAL_ORDERING_GAP_FLOOR = 0.5
V3_CAUSAL_PROGRESS_GAP_FLOOR = 0.05
COMBINED_PERTURBED_PROGRESS_FLOOR_RATIO = 0.85
COMBINED_PROGRESS_RETENTION_FLOOR = 0.88
COMBINED_LATE_SPEED_RETENTION_FLOOR = 0.83
COMBINED_CONTROL_LATE_SPEED_FLOOR = 0.12
NOMINAL_SIGNED_HEAD_FLOOR = 0.72
NOMINAL_LATERAL_FRACTION_CEILING = 0.40
NOMINAL_TARGET_ALIGNMENT_FLOOR = 0.62
RECOVERED_SIGNED_HEAD_FLOOR = 0.62
RECOVERED_LATERAL_FRACTION_CEILING = 0.48
RECOVERED_TARGET_ALIGNMENT_FLOOR = 0.52
LOCAL_CAUSAL_INITIAL_FLOOR = 0.12
LOCAL_CAUSAL_RADIUS_TWO_FLOOR = 2.0e-3
LOCAL_CAUSAL_RADIUS_THREE_FLOOR = 2.0e-3
LOCAL_CAUSAL_SETTLED_CEILING = 5.0e-6
ABLATION_FULL_PROGRESS_FLOOR = 0.45
ABLATION_ZERO_PROGRESS_CEILING = 1.0e-9
ABLATION_FROZEN_PROGRESS_GAP_FLOOR = 0.30
ABLATION_SHUFFLED_PROGRESS_GAP_FLOOR = 0.20
ABLATION_NO_FRICTION_DISPLACEMENT_CEILING = 1.0e-9
SELECTION_GATE_GUARD_BAND = 1.0e-3
LOCAL_CAUSAL_RADIUS_GUARD_BAND = 1.0e-4
LOCAL_CAUSAL_SETTLED_GUARD_BAND = 1.0e-6
ABLATION_PROGRESS_GUARD_BAND = 1.0e-2
ABLATION_CONSERVATION_GUARD_CEILING = 1.0e-10
ARTIFACT_GENOME_DECIMALS = 8
V3_SELECTION_FITNESS_DECIMALS = 4
V3_GATE_DIAGNOSTIC_DECIMALS = 8
HOST_DIAGNOSTIC_DECIMALS = 8
PUBLISHED_V3_WARM_START_SHA256 = "0c46518c424528337f3a2ad5a96912b7812dda473688d468556023be095551ba"
PUBLISHED_V3_MODEL_VERSION = "locomotion-articulated-head-leading-es-v3"
V3_RECIPE_SOURCE_PATHS = (
    "training/wurmkickflip_rl/evolve_locomotion_policy.py",
    "training/wurmkickflip_rl/articulated_locomotion.py",
    "training/wurmkickflip_rl/contracts.py",
    "contracts/locomotion-v2.json",
    "training/uv.lock",
    "training/pyproject.toml",
)
RUNTIME_SENSOR_CLAMPS: dict[str, tuple[float, float]] = {
    "targetForward": (-1.0, 1.0),
    "targetRight": (-1.0, 1.0),
    "targetDistance": (0.0, 1.5),
    "forwardSpeed": (-2.0, 2.0),
    "angularSpeed": (-3.0, 3.0),
    "terrainFriction": (0.0, 1.2),
    "urgency": (0.0, 1.0),
    "contactLoad": (0.0, 1.0),
    "slipSpeed": (0.0, 2.0),
    "obstacleForward": (-1.0, 1.0),
    "obstacleRight": (-1.0, 1.0),
}

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

# Controller arithmetic consumes the exact decimal values shipped in the JSON
# artifact. Keeping the analytic linspace here would make training start from a
# subtly different controller (~2e-8 at interior segments) than the browser.
SEGMENT_POSITIONS = np.asarray(
    [
        round(float(position), ARTIFACT_GENOME_DECIMALS)
        for position in np.linspace(-1.0, 1.0, SEGMENT_COUNT, dtype=np.float64)
    ],
    dtype=np.float64,
)
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


@dataclass(frozen=True)
class PerturbationSchedule:
    domain: PerturbationDomain
    pre_start: int
    perturb_start: int
    perturb_end: int
    post_end: int


@dataclass(frozen=True)
class V3PopulationRollout:
    selection_score: np.ndarray
    mean_progress: np.ndarray
    late_progress_speed: np.ndarray
    worst_progress: np.ndarray
    path_efficiency: np.ndarray
    mean_final_distance: np.ndarray
    mean_speed: np.ndarray
    actuator_temporal_std: np.ndarray
    segment_diversity: np.ndarray
    mean_energy: np.ndarray
    hidden_saturation_ratio: np.ndarray
    mean_hidden_tanh_derivative: np.ndarray
    reached_ratio: np.ndarray
    mean_displacement: np.ndarray
    mean_heading_alignment: np.ndarray
    signed_head_leading_fraction: np.ndarray
    positive_head_leading_fraction: np.ndarray
    lateral_motion_fraction: np.ndarray
    mean_head_leading_speed: np.ndarray
    target_facing_head_leading_speed: np.ndarray
    moving_target_facing_alignment: np.ndarray
    late_target_facing_alignment: np.ndarray
    mean_pre_recovery_speed: np.ndarray
    mean_post_recovery_speed: np.ndarray
    mean_recovery_ratio: np.ndarray
    worst_recovery_ratio: np.ndarray
    mean_recovery_time_seconds: np.ndarray


@dataclass(frozen=True)
class PerturbationRecoveryMetrics:
    pre_target_facing_head_speed: float
    post_target_facing_head_speed: float
    recovery_ratio: float
    worst_recovery_ratio: float
    recovery_time_seconds: float
    target_progress: float
    signed_head_leading_fraction: float


@dataclass(frozen=True)
class CombinedRecoveryMetrics:
    control_target_progress: float
    perturbed_target_progress: float
    progress_retention: float
    control_late_progress_speed: float
    perturbed_late_progress_speed: float
    late_speed_retention: float
    perturbed_signed_head_leading_fraction: float
    perturbed_lateral_motion_fraction: float
    perturbed_moving_target_facing_alignment: float


@dataclass(frozen=True)
class CanonicalCombinedRecovery:
    control_target_progress: np.ndarray
    perturbed_target_progress: np.ndarray
    progress_retention: np.ndarray
    control_late_progress_speed: np.ndarray
    perturbed_late_progress_speed: np.ndarray
    late_speed_retention: np.ndarray
    perturbed_signed_head_leading_fraction: np.ndarray
    perturbed_lateral_motion_fraction: np.ndarray
    perturbed_moving_target_facing_alignment: np.ndarray


@dataclass(frozen=True)
class RolloutMetricsV3:
    fitness: float
    progress: float
    path_efficiency: float
    final_distance: float
    mean_speed: float
    actuator_temporal_std: float
    segment_diversity: float
    mean_energy: float
    hidden_saturation_ratio: float
    mean_hidden_tanh_derivative: float
    reached_ratio: float
    displacement: float
    heading_alignment: float
    signed_head_leading_fraction: float
    positive_head_leading_fraction: float
    lateral_motion_fraction: float
    mean_head_leading_speed: float
    target_facing_head_leading_speed: float
    moving_target_facing_alignment: float
    late_target_facing_alignment: float
    causal_gap: float
    causal_progress_gap: float
    local_causal_initial_delta: float
    local_causal_radius_two_delta: float
    local_causal_radius_three_delta: float
    local_causal_settled_delta: float
    slick_window: PerturbationRecoveryMetrics
    body_shove: PerturbationRecoveryMetrics
    combined_recovery: CombinedRecoveryMetrics


@dataclass(frozen=True)
class V3SelectionDiagnostics:
    signed_head_leading_fraction: np.ndarray
    lateral_motion_fraction: np.ndarray
    target_facing_alignment: np.ndarray
    slick_recovery_ratio: np.ndarray
    shove_recovery_ratio: np.ndarray
    combined_progress_retention: np.ndarray
    combined_late_speed_retention: np.ndarray
    combined_control_progress: np.ndarray
    causal_progress_gap: np.ndarray
    local_causal_initial_delta: np.ndarray
    local_causal_radius_two_delta: np.ndarray
    local_causal_radius_three_delta: np.ndarray
    local_causal_settled_delta: np.ndarray
    all_selection_gates_feasible: np.ndarray
    all_selection_guard_bands_feasible: np.ndarray
    selection_gate_count: np.ndarray
    selection_guard_band_count: np.ndarray


@dataclass(frozen=True)
class LocalCausalResponse:
    initial_local_delta: np.ndarray
    radius_two_delta: np.ndarray
    radius_three_delta: np.ndarray
    settled_delta: np.ndarray


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

# These fixed subsets keep v3's evaluation cost comparable to v2 while exposing
# every candidate to left/right turns, obstacles, low/high friction, and both
# deterministic transient domains. Nothing about the subset enters the policy.
V3_ROBUSTNESS_SCENARIOS = tuple(SCENARIOS[index] for index in (1, 4, 7, 10))
V3_CAUSAL_SCENARIOS = tuple(SCENARIOS[index] for index in (3, 6, 9, 11))
V3_COMBINED_RECOVERY_SCENARIOS = (Scenario(20.0, 0.0, 0.9, 0.85),)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evolve a clock-free, segment-local recurrent neural controller. "
            "The plant moves only through antagonistic segment actuator work."
        )
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--generations", type=int, default=140)
    parser.add_argument("--population-size", type=int, default=160)
    parser.add_argument("--elite-count", type=int, default=20)
    parser.add_argument("--episode-steps", type=int, default=480)
    parser.add_argument("--model-version")
    parser.add_argument(
        "--objective-version",
        choices=("v2", "v3"),
        default="v2",
        help=(
            "v2 runs the legacy objective into training/runs; v3 selects for "
            "head-leading travel and deterministic transient recovery."
        ),
    )
    parser.add_argument(
        "--combined-recovery-mode",
        choices=("bounded-480", "recentered-900"),
        default=DEFAULT_COMBINED_RECOVERY_MODE,
        help=(
            "V3 paired-domain selector. bounded-480 is the browser-certified "
            "training recipe; recentered-900 is diagnostic shaping only."
        ),
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        help="Browser locomotion artifact used to seed one third of the population.",
    )
    args = parser.parse_args()

    objective_mode: ObjectiveMode = args.objective_version
    combined_recovery_mode: CombinedRecoveryMode = args.combined_recovery_mode
    if objective_mode == "v3" and combined_recovery_mode == "recentered-900":
        print(
            "WARNING: recentered-900 is an experimental diagnostic selector; "
            "its results are not publication-authoritative and may fail the browser gait audit."
        )
    if args.out is None:
        args.out = DEFAULT_ARTIFACT if objective_mode == "v2" else DEFAULT_V3_ARTIFACT
    if args.summary is None:
        args.summary = DEFAULT_SUMMARY if objective_mode == "v2" else DEFAULT_V3_SUMMARY
    if args.model_version is None:
        args.model_version = (
            "locomotion-articulated-contact-es-v2"
            if objective_mode == "v2"
            else "locomotion-head-leading-es-v3-candidate"
        )

    validate_arguments(args)
    warm_start, warm_start_metadata = load_warm_start(args.warm_start)
    best, history = evolve(
        seed=args.seed,
        generations=args.generations,
        population_size=args.population_size,
        elite_count=args.elite_count,
        episode_steps=args.episode_steps,
        warm_start=warm_start,
        objective_mode=objective_mode,
        combined_recovery_mode=combined_recovery_mode,
    )
    if objective_mode == "v3":
        artifact_genome, evaluation_v3, ablations = finalize_v3_artifact_evaluation(
            best,
            args.episode_steps,
            combined_recovery_mode=combined_recovery_mode,
        )
        genome_recipe_hash = canonical_genome_recipe_hash(
            artifact_genome,
            model_version=args.model_version,
            seed=args.seed,
            generations=args.generations,
            population_size=args.population_size,
            elite_count=args.elite_count,
            episode_steps=args.episode_steps,
            warm_start_metadata=warm_start_metadata,
            combined_recovery_mode=combined_recovery_mode,
        )
        summary = make_summary_v3(
            artifact_genome,
            model_version=args.model_version,
            seed=args.seed,
            generations=args.generations,
            population_size=args.population_size,
            elite_count=args.elite_count,
            episode_steps=args.episode_steps,
            metrics=evaluation_v3,
            ablations=ablations,
            genome_recipe_hash=genome_recipe_hash,
            warm_start_metadata=warm_start_metadata,
            history=history,
            combined_recovery_mode=combined_recovery_mode,
        )
        try:
            artifact = make_artifact_v3(
                artifact_genome,
                model_version=args.model_version,
                seed=args.seed,
                generations=args.generations,
                population_size=args.population_size,
                elite_count=args.elite_count,
                episode_steps=args.episode_steps,
                metrics=evaluation_v3,
                ablations=ablations,
                genome_recipe_hash=genome_recipe_hash,
                warm_start_metadata=warm_start_metadata,
                combined_recovery_mode=combined_recovery_mode,
            )
        except SystemExit:
            write_json(args.summary, summary, compact=False)
            print(f"wrote diagnostic summary {args.summary}")
            print_v3_gate_failures(summary)
            raise
    else:
        artifact_genome = best
        ablations = evaluate_ablations(artifact_genome, args.episode_steps)
        evaluation = evaluate_genome(artifact_genome, args.episode_steps, SCENARIOS)
        trace_hash = deterministic_trace_hash(artifact_genome, args.episode_steps)
        artifact = make_artifact(
            artifact_genome,
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
    if objective_mode == "v3":
        print(
            "fitness={fitness:.6f} progress={progress:.6f} signedHead={head:.6f} "
            "lateral={lateral:.6f} targetFacing={facing:.6f}".format(
                fitness=evaluation_v3.fitness,
                progress=evaluation_v3.progress,
                head=evaluation_v3.signed_head_leading_fraction,
                lateral=evaluation_v3.lateral_motion_fraction,
                facing=evaluation_v3.moving_target_facing_alignment,
            )
        )
        print(
            "recovery slick={slick:.6f}/{slick_time:.3f}s "
            "shove={shove:.6f}/{shove_time:.3f}s combined={combined:.6f}/{late:.6f} "
            "causalGap={causal:.6f}".format(
                slick=evaluation_v3.slick_window.recovery_ratio,
                slick_time=evaluation_v3.slick_window.recovery_time_seconds,
                shove=evaluation_v3.body_shove.recovery_ratio,
                shove_time=evaluation_v3.body_shove.recovery_time_seconds,
                combined=evaluation_v3.combined_recovery.progress_retention,
                late=evaluation_v3.combined_recovery.late_speed_retention,
                causal=evaluation_v3.causal_gap,
            )
        )
    else:
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
    validate_model_version(args.model_version)


def print_v3_gate_failures(training: dict[str, Any]) -> None:
    for result_name in (
        "selectionGateResults",
        "selectionGateGuardBandResults",
        "ablationGateResults",
        "ablationGateGuardBandResults",
    ):
        failed = [name for name, passed in training[result_name].items() if not passed]
        if failed:
            print(f"failed {result_name}: {', '.join(failed)}")


def validate_model_version(model_version: object) -> str:
    if not isinstance(model_version, str) or not model_version:
        raise SystemExit("--model-version must be a non-empty string")
    if model_version != model_version.strip():
        raise SystemExit("--model-version must not have leading or trailing whitespace")
    return model_version


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
            artifact["initialState"] + input_values + artifact["weights"]["recurrent"] + output_values,
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
    objective_mode: ObjectiveMode = "v2",
    combined_recovery_mode: CombinedRecoveryMode = DEFAULT_COMBINED_RECOVERY_MODE,
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
    best_selection_feasible = False
    best_guarded_feasible = False
    best_selection_gate_count = -1
    best_guard_band_count = -1
    history: list[dict[str, float | int]] = []

    for generation in range(generations):
        diagnostics: V3SelectionDiagnostics | None = None
        if objective_mode == "v3":
            fitness, causal_gap, diagnostics = evaluate_selection_fitness_v3(
                population,
                episode_steps,
                combined_recovery_mode=combined_recovery_mode,
            )
            fitness, order = v3_selection_order(
                fitness,
                diagnostics.all_selection_gates_feasible,
                diagnostics.all_selection_guard_bands_feasible,
                diagnostics.selection_gate_count,
                diagnostics.selection_guard_band_count,
            )
        else:
            fitness, causal_gap = evaluate_selection_fitness(population, episode_steps)
            order = np.argsort(fitness)[::-1]
        population = population[order]
        fitness = fitness[order]
        causal_gap = causal_gap[order]
        if diagnostics is not None:
            diagnostics = reorder_v3_diagnostics(diagnostics, order)
        generation_best_selection_feasible = (
            bool(diagnostics.all_selection_gates_feasible[0]) if diagnostics is not None else False
        )
        generation_best_guarded_feasible = (
            bool(diagnostics.all_selection_guard_bands_feasible[0])
            if diagnostics is not None
            else False
        )
        generation_best_selection_gate_count = (
            int(diagnostics.selection_gate_count[0]) if diagnostics is not None else -1
        )
        generation_best_guard_band_count = (
            int(diagnostics.selection_guard_band_count[0]) if diagnostics is not None else -1
        )
        better = (
            v3_candidate_is_better(
                float(fitness[0]),
                generation_best_selection_feasible,
                generation_best_guarded_feasible,
                generation_best_selection_gate_count,
                generation_best_guard_band_count,
                best_fitness,
                best_selection_feasible,
                best_guarded_feasible,
                best_selection_gate_count,
                best_guard_band_count,
            )
            if objective_mode == "v3"
            else float(fitness[0]) > best_fitness
        )
        if better:
            best = population[0].copy()
            best_fitness = float(fitness[0])
            best_selection_feasible = generation_best_selection_feasible
            best_guarded_feasible = generation_best_guarded_feasible
            best_selection_gate_count = generation_best_selection_gate_count
            best_guard_band_count = generation_best_guard_band_count

        history_decimals = V3_SELECTION_FITNESS_DECIMALS if objective_mode == "v3" else 8
        history_entry: dict[str, float | int] = {
            "generation": generation,
            "bestFitness": round(best_fitness, history_decimals),
            "generationBestFitness": round(float(fitness[0]), history_decimals),
            "meanFitness": round(float(np.mean(fitness)), history_decimals),
            "meanCausalGap": round(float(np.mean(causal_gap)), history_decimals),
        }
        if diagnostics is not None:
            elite_cutoff = min(elite_count - 1, fitness.size - 1)
            history_entry.update(
                {
                    "generationFeasibleCount": int(
                        np.count_nonzero(diagnostics.all_selection_gates_feasible)
                    ),
                    "generationGuardedFeasibleCount": int(
                        np.count_nonzero(diagnostics.all_selection_guard_bands_feasible)
                    ),
                    "generationBestFeasible": int(generation_best_selection_feasible),
                    "generationBestGuardedFeasible": int(generation_best_guarded_feasible),
                    "generationBestSelectionGateCount": int(diagnostics.selection_gate_count[0]),
                    "generationBestGuardBandCount": int(diagnostics.selection_guard_band_count[0]),
                    "eliteCutoffFitness": round(float(fitness[elite_cutoff]), history_decimals),
                    "bestToEliteCutoffFitnessMargin": round(
                        float(fitness[0] - fitness[elite_cutoff]),
                        history_decimals,
                    ),
                    "generationBestSignedHeadLeadingFraction": round(
                        float(diagnostics.signed_head_leading_fraction[0]),
                        history_decimals,
                    ),
                    "generationBestLateralMotionFraction": round(
                        float(diagnostics.lateral_motion_fraction[0]),
                        history_decimals,
                    ),
                    "generationBestTargetFacingAlignment": round(
                        float(diagnostics.target_facing_alignment[0]),
                        history_decimals,
                    ),
                    "generationBestSlickRecoveryRatio": round(
                        float(diagnostics.slick_recovery_ratio[0]),
                        history_decimals,
                    ),
                    "generationBestShoveRecoveryRatio": round(
                        float(diagnostics.shove_recovery_ratio[0]),
                        history_decimals,
                    ),
                    "generationBestCombinedProgressRetention": round(
                        float(diagnostics.combined_progress_retention[0]),
                        history_decimals,
                    ),
                    "generationBestCombinedLateSpeedRetention": round(
                        float(diagnostics.combined_late_speed_retention[0]),
                        history_decimals,
                    ),
                    "generationBestCombinedControlProgress": round(
                        float(diagnostics.combined_control_progress[0]),
                        V3_GATE_DIAGNOSTIC_DECIMALS,
                    ),
                    "generationBestCausalProgressGap": round(
                        float(diagnostics.causal_progress_gap[0]),
                        V3_GATE_DIAGNOSTIC_DECIMALS,
                    ),
                    "generationBestLocalCausalInitialDelta": round(
                        float(diagnostics.local_causal_initial_delta[0]),
                        history_decimals,
                    ),
                    "generationBestLocalCausalRadiusThreeDelta": round(
                        float(diagnostics.local_causal_radius_three_delta[0]),
                        history_decimals,
                    ),
                    "generationBestLocalCausalRadiusTwoDelta": round(
                        float(diagnostics.local_causal_radius_two_delta[0]),
                        history_decimals,
                    ),
                    "generationBestLocalCausalSettledDelta": round(
                        float(diagnostics.local_causal_settled_delta[0]),
                        history_decimals,
                    ),
                }
            )
        history.append(history_entry)

        elites = population[:elite_count].copy()
        progress = generation / max(generations - 1, 1)
        mutation_scale = 0.36 * (1.0 - progress) + 0.055
        children = [best.copy(), *[elite.copy() for elite in elites[: max(1, elite_count // 3)]]]
        if objective_mode == "v3":
            repair_budget = min(40, population_size - len(children))
            children.extend(
                v3_near_feasible_children(
                    best,
                    rng,
                    generation=generation,
                    generations=generations,
                    maximum_children=repair_budget,
                )
            )
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

    if objective_mode == "v3":
        final_raw_fitness, _final_causal_gap, final_diagnostics = evaluate_selection_fitness_v3(
            population,
            episode_steps,
            combined_recovery_mode=combined_recovery_mode,
        )
        final_fitness, final_order = v3_selection_order(
            final_raw_fitness,
            final_diagnostics.all_selection_gates_feasible,
            final_diagnostics.all_selection_guard_bands_feasible,
            final_diagnostics.selection_gate_count,
            final_diagnostics.selection_guard_band_count,
        )
        final_index = int(final_order[0])
        final_best = population[final_index]
        final_selection_feasible = bool(final_diagnostics.all_selection_gates_feasible[final_index])
        final_guarded_feasible = bool(
            final_diagnostics.all_selection_guard_bands_feasible[final_index]
        )
        if v3_candidate_is_better(
            float(final_fitness[final_index]),
            final_selection_feasible,
            final_guarded_feasible,
            int(final_diagnostics.selection_gate_count[final_index]),
            int(final_diagnostics.selection_guard_band_count[final_index]),
            best_fitness,
            best_selection_feasible,
            best_guarded_feasible,
            best_selection_gate_count,
            best_guard_band_count,
        ):
            best = final_best.copy()
    else:
        final_fitness = evaluate_selection_fitness(population, episode_steps)[0]
        final_best = population[int(np.argmax(final_fitness))]
        if float(np.max(final_fitness)) > best_fitness:
            best = final_best.copy()
    return bound_genome(best), history


def v3_near_feasible_children(
    parent: np.ndarray,
    rng: np.random.Generator,
    *,
    generation: int,
    generations: int,
    maximum_children: int,
) -> list[np.ndarray]:
    """Exploit a nearly feasible lineage without increasing recipe population size."""
    if maximum_children <= 0:
        return []
    progress = generation / max(generations - 1, 1)
    micro_scale = 0.045 * (1.0 - progress) + 0.008
    coordinate_scale = 0.055 * (1.0 - progress) + 0.010
    micro_count = maximum_children // 2
    children: list[np.ndarray] = []
    for _ in range(micro_count):
        mutation = rng.normal(0.0, micro_scale, GENOME_SIZE)
        mutation[INITIAL_SLICE] *= 0.5
        children.append(bound_genome(parent + mutation))
    coordinate_count = maximum_children - micro_count
    for probe_index in range(coordinate_count):
        coordinate = (generation * 11 + probe_index // 2) % GENOME_SIZE
        direction = -1.0 if probe_index % 2 == 0 else 1.0
        child = parent.copy()
        child[coordinate] += direction * coordinate_scale
        children.append(bound_genome(child))
    return children


def reorder_v3_diagnostics(
    diagnostics: V3SelectionDiagnostics,
    order: np.ndarray,
) -> V3SelectionDiagnostics:
    return V3SelectionDiagnostics(
        signed_head_leading_fraction=diagnostics.signed_head_leading_fraction[order],
        lateral_motion_fraction=diagnostics.lateral_motion_fraction[order],
        target_facing_alignment=diagnostics.target_facing_alignment[order],
        slick_recovery_ratio=diagnostics.slick_recovery_ratio[order],
        shove_recovery_ratio=diagnostics.shove_recovery_ratio[order],
        combined_progress_retention=diagnostics.combined_progress_retention[order],
        combined_late_speed_retention=diagnostics.combined_late_speed_retention[order],
        combined_control_progress=diagnostics.combined_control_progress[order],
        causal_progress_gap=diagnostics.causal_progress_gap[order],
        local_causal_initial_delta=diagnostics.local_causal_initial_delta[order],
        local_causal_radius_two_delta=diagnostics.local_causal_radius_two_delta[order],
        local_causal_radius_three_delta=diagnostics.local_causal_radius_three_delta[order],
        local_causal_settled_delta=diagnostics.local_causal_settled_delta[order],
        all_selection_gates_feasible=diagnostics.all_selection_gates_feasible[order],
        all_selection_guard_bands_feasible=diagnostics.all_selection_guard_bands_feasible[order],
        selection_gate_count=diagnostics.selection_gate_count[order],
        selection_guard_band_count=diagnostics.selection_guard_band_count[order],
    )


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


def canonical_v3_selection_fitness(fitness: np.ndarray) -> np.ndarray:
    """Canonicalize V3 ranking values before stable, deterministic ordering."""
    return np.round(np.asarray(fitness, dtype=np.float64), V3_SELECTION_FITNESS_DECIMALS)


def v3_selection_order(
    fitness: np.ndarray,
    selection_feasible: np.ndarray,
    guarded_feasible: np.ndarray,
    selection_gate_count: np.ndarray,
    guard_band_count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank guard/gate tiers and counts before canonical fitness, preserving ties."""
    canonical_fitness = canonical_v3_selection_fitness(fitness)
    selection_flags = np.asarray(selection_feasible, dtype=np.bool_)
    guarded_flags = np.asarray(guarded_feasible, dtype=np.bool_)
    selection_counts = np.asarray(selection_gate_count, dtype=np.int16)
    guard_counts = np.asarray(guard_band_count, dtype=np.int16)
    score_order = np.argsort(-canonical_fitness, kind="stable")
    selection_count_order = score_order[
        np.argsort(-selection_counts[score_order], kind="stable")
    ]
    guard_count_order = selection_count_order[
        np.argsort(-guard_counts[selection_count_order], kind="stable")
    ]
    selection_order = guard_count_order[
        np.argsort(-selection_flags[guard_count_order].astype(np.int8), kind="stable")
    ]
    guarded_order = selection_order[
        np.argsort(-guarded_flags[selection_order].astype(np.int8), kind="stable")
    ]
    return canonical_fitness, guarded_order


def v3_candidate_is_better(
    candidate_fitness: float,
    candidate_selection_feasible: bool,
    candidate_guarded_feasible: bool,
    candidate_selection_gate_count: int,
    candidate_guard_band_count: int,
    retained_fitness: float,
    retained_selection_feasible: bool,
    retained_guarded_feasible: bool,
    retained_selection_gate_count: int,
    retained_guard_band_count: int,
) -> bool:
    candidate_key = (
        int(candidate_guarded_feasible),
        int(candidate_selection_feasible),
        candidate_guard_band_count,
        candidate_selection_gate_count,
        round(candidate_fitness, V3_SELECTION_FITNESS_DECIMALS),
    )
    retained_key = (
        int(retained_guarded_feasible),
        int(retained_selection_feasible),
        retained_guard_band_count,
        retained_selection_gate_count,
        round(retained_fitness, V3_SELECTION_FITNESS_DECIMALS),
    )
    return candidate_key > retained_key


def canonical_v3_diagnostic(values: np.ndarray) -> np.ndarray:
    """Canonicalize diagnostics before any V3 feasibility threshold comparison."""
    return np.round(np.asarray(values, dtype=np.float64), V3_GATE_DIAGNOSTIC_DECIMALS)


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
    contact_load = np.ones_like(joint)
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
        target_forward = (
            relative[:, :, 0] * forward[:, :, 0] + relative[:, :, 1] * forward[:, :, 1]
        ) * inverse_distance
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


def perturbation_schedule(
    episode_steps: int,
    domain: PerturbationDomain,
) -> PerturbationSchedule:
    """Return deterministic measurement and perturbation windows.

    The slick window and its recovery measurement finish before the existing
    mid-episode target switch. The shove happens after that switch, so recovery
    requires the controller to preserve or regain its newly chosen heading.
    These step indices belong only to the environment evaluator; no clock or
    phase value is exposed to the recurrent network.
    """
    if domain == "slick-window":
        pre_start = int(episode_steps * 0.12)
        perturb_start = max(pre_start + 1, int(episode_steps * 0.24))
        perturb_end = max(perturb_start + 1, int(episode_steps * 0.38))
        post_end = max(perturb_end + 1, int(episode_steps * 0.50))
    elif domain == "body-shove":
        pre_start = int(episode_steps * 0.50)
        perturb_start = max(pre_start + 1, int(episode_steps * 0.62))
        perturb_end = min(episode_steps - 1, perturb_start + 1)
        post_end = max(perturb_end + 1, int(episode_steps * 0.82))
    elif domain == "combined-recovery":
        pre_start = int(episode_steps * 0.16)
        perturb_start = max(pre_start + 1, int(episode_steps * (240.0 / 900.0)))
        perturb_end = max(perturb_start + 1, int(episode_steps * (330.0 / 900.0)))
        post_end = max(perturb_end + 1, int(episode_steps * 0.50))
    else:
        pre_start = perturb_start = perturb_end = post_end = 0
    return PerturbationSchedule(
        domain=domain,
        pre_start=pre_start,
        perturb_start=perturb_start,
        perturb_end=perturb_end,
        post_end=min(post_end, episode_steps),
    )


def evaluate_population_v3_rollout(
    population: np.ndarray,
    episode_steps: int,
    scenarios: tuple[Scenario, ...],
    *,
    perturbation: PerturbationDomain = "nominal",
    intervention: str = "full",
    apply_perturbation: bool = True,
    recenter_unbounded: bool = False,
) -> V3PopulationRollout:
    """Vectorized articulated rollout with head-axis and recovery accounting.

    Positive head-leading motion is COM displacement along the plant's
    tail-to-head body axis (`forward`), i.e. toward the rendered head. Lateral
    displacement is measured orthogonally. Turning is not penalized: alignment
    samples immediately after deterministic target switches receive a short
    grace period, then the new target direction is scored normally.
    """
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
    base_friction = np.array(
        [scenario.friction for scenario in scenarios],
        dtype=np.float64,
    )[None, :]
    urgency = np.array([scenario.urgency for scenario in scenarios], dtype=np.float64)[None, :]
    roughness = np.array([scenario.roughness for scenario in scenarios], dtype=np.float64)[None, :]
    body_scale = np.array([scenario.body_scale for scenario in scenarios], dtype=np.float64)
    obstacle_center = np.array(
        [[scenario.obstacle_x, scenario.obstacle_z] for scenario in scenarios],
        dtype=np.float64,
    )
    obstacle_radius = np.array([scenario.obstacle_radius for scenario in scenarios], dtype=np.float64)
    switch_scenarios = obstacle_radius > 0.0

    if recenter_unbounded and np.any(obstacle_radius > 0.0):
        raise ValueError("recenter_unbounded requires obstacle-free scenarios")
    body_position, body_velocity = initial_body(candidate_count, scenario_count, body_scale)
    plant_root = np.mean(body_position, axis=2)
    position = plant_root.copy()
    forward = np.zeros_like(position)
    forward[:, :, 0] = 1.0
    speed = np.zeros((candidate_count, scenario_count), dtype=np.float64)
    angular_speed = np.zeros_like(speed)
    joint = np.zeros((candidate_count, scenario_count, SEGMENT_COUNT), dtype=np.float64)
    joint_velocity = np.zeros_like(joint)
    command = np.zeros_like(joint)
    hidden = np.broadcast_to(initial_state[:, None, :], joint.shape).copy()
    # Browser plants start fully supported; the first plant step then supplies
    # measured friction/contact values for the next neural tick.
    contact_load = np.ones_like(joint)
    slip_speed = np.zeros_like(joint)
    obstacle_forward = np.zeros_like(joint)
    obstacle_right = np.zeros_like(joint)

    path_length = np.zeros_like(speed)
    progress = np.zeros_like(speed)
    energy_sum = np.zeros_like(speed)
    slip_sum = np.zeros_like(speed)
    obstacle_contact_steps = np.zeros_like(speed)
    alignment_sum = np.zeros_like(speed)
    alignment_samples = np.zeros(scenario_count, dtype=np.float64)
    moving_alignment_sum = np.zeros_like(speed)
    moving_alignment_distance = np.zeros_like(speed)
    late_alignment_sum = np.zeros_like(speed)
    late_alignment_samples = 0
    near_target_steps = np.zeros_like(speed)
    command_sum = np.zeros_like(joint)
    command_square_sum = np.zeros_like(joint)
    signed_head_distance = np.zeros_like(speed)
    late_progress = np.zeros_like(speed)
    hidden_saturation_sum = np.zeros_like(speed)
    hidden_derivative_sum = np.zeros_like(speed)
    positive_head_distance = np.zeros_like(speed)
    lateral_distance = np.zeros_like(speed)
    target_facing_head_distance = np.zeros_like(speed)

    schedule = perturbation_schedule(episode_steps, perturbation)
    pre_recovery_distance = np.zeros_like(speed)
    post_recovery_distance = np.zeros_like(speed)
    recovery_ema = np.zeros_like(speed)
    recovered = np.zeros_like(speed, dtype=np.bool_)
    recovery_time = np.zeros_like(speed)
    switch_step = episode_steps // 2
    turn_grace_steps = max(6, int(episode_steps * 0.05))
    late_start = int(episode_steps * 0.80)
    late_progress_start = int(episode_steps * 0.50)
    late_progress_end = max(late_progress_start + 1, int(episode_steps * 0.70))

    for step in range(episode_steps):
        if step == switch_step:
            target[switch_scenarios, 1] *= -0.72
            target[switch_scenarios, 0] += np.sign(target[switch_scenarios, 1] + 1.0e-9) * 0.35

        if apply_perturbation and perturbation == "body-shove" and step == schedule.perturb_start:
            root_right = np.stack((-forward[:, :, 1], forward[:, :, 0]), axis=2)
            shove_sign = np.where(np.arange(scenario_count) % 2 == 0, 1.0, -1.0)
            impulse = root_right * (SHOVE_LATERAL_SPEED * shove_sign[None, :, None]) - forward * SHOVE_BACKWARD_SPEED
            body_velocity += impulse[:, :, None, :]

        if apply_perturbation and perturbation == "combined-recovery" and step == schedule.perturb_start:
            root_right = np.stack((-forward[:, :, 1], forward[:, :, 0]), axis=2)
            body_velocity += (root_right * COMBINED_SHOVE_LATERAL_SPEED)[:, :, None, :]

        numb_active = (
            apply_perturbation
            and perturbation == "combined-recovery"
            and schedule.perturb_start <= step < schedule.perturb_end
        )
        if numb_active:
            hidden[:, :, COMBINED_NUMB_SEGMENT] = 0.0
            command[:, :, COMBINED_NUMB_SEGMENT] = 0.0

        slick_active = (
            apply_perturbation
            and perturbation == "slick-window"
            and schedule.perturb_start <= step < schedule.perturb_end
        )
        combined_slick_active = (
            apply_perturbation
            and perturbation == "combined-recovery"
            and schedule.perturb_start <= step < schedule.perturb_end
        )
        if combined_slick_active:
            step_friction = base_friction * COMBINED_TRACTION_SCALE
        elif slick_active:
            step_friction = np.minimum(base_friction, SLICK_FRICTION)
        else:
            step_friction = base_friction

        relative = target[None, :, :] - position
        distance = np.linalg.norm(relative, axis=2)
        inverse_distance = 1.0 / np.maximum(distance, 1.0e-9)
        right_x = -forward[:, :, 1]
        right_z = forward[:, :, 0]
        target_forward = (
            relative[:, :, 0] * forward[:, :, 0] + relative[:, :, 1] * forward[:, :, 1]
        ) * inverse_distance
        target_right = (relative[:, :, 0] * right_x + relative[:, :, 1] * right_z) * inverse_distance
        near_target_steps += distance < 1.15

        controller_sensors = runtime_controller_sensors(
            target_forward,
            target_right,
            distance / 5.0,
            speed,
            angular_speed,
            step_friction,
            urgency,
            contact_load,
            slip_speed,
            obstacle_forward,
            obstacle_right,
        )
        hidden = segmental_network_step(
            hidden,
            command,
            joint,
            joint_velocity,
            *controller_sensors,
            input_weights,
            recurrent_weights,
            output_weights,
        )[0]
        if numb_active:
            hidden[:, :, COMBINED_NUMB_SEGMENT] = 0.0
        hidden_saturation_sum += np.mean(np.abs(hidden) > 0.98, axis=2)
        hidden_derivative_sum += np.mean(1.0 - hidden * hidden, axis=2)
        command = segmental_network_output(
            hidden,
            joint,
            controller_sensors[1],
            controller_sensors[10],
            controller_sensors[7],
            output_weights,
        )
        if numb_active:
            command[:, :, COMBINED_NUMB_SEGMENT] = 0.0
        if intervention == "shuffled":
            command = command[:, :, SEGMENT_SHUFFLE]

        previous_position = position.copy()
        previous_plant_root = plant_root.copy()
        (
            step_distance,
            plant_root,
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
            step_friction,
            roughness,
            body_scale,
            obstacle_center,
            obstacle_radius,
            POLICY_TIMESTEP,
        )

        plant_delta = plant_root - previous_plant_root
        if recenter_unbounded:
            position = previous_position + plant_delta
            body_position -= plant_root[:, :, None, :]
            plant_root = np.zeros_like(plant_root)
        else:
            position = plant_root

        next_relative = target[None, :, :] - position
        next_distance = np.linalg.norm(next_relative, axis=2)
        next_direction = next_relative / np.maximum(next_distance[:, :, None], 1.0e-9)
        next_alignment = np.sum(next_direction * forward, axis=2)
        root_delta = position - previous_position
        root_right = np.stack((-forward[:, :, 1], forward[:, :, 0]), axis=2)
        head_step = np.sum(root_delta * forward, axis=2)
        lateral_step = np.abs(np.sum(root_delta * root_right, axis=2))
        positive_head_step = np.maximum(head_step, 0.0)
        target_head_step = positive_head_step * np.maximum(next_alignment, 0.0)

        step_progress = distance - next_distance
        progress += step_progress
        if late_progress_start <= step < late_progress_end:
            late_progress += step_progress
        path_length += step_distance
        signed_head_distance += head_step
        positive_head_distance += positive_head_step
        lateral_distance += lateral_step
        target_facing_head_distance += target_head_step
        energy_sum += np.mean(np.abs(command), axis=2)
        slip_sum += np.mean(slip_speed, axis=2)
        obstacle_contact_steps += np.any(
            np.abs(obstacle_forward) + np.abs(obstacle_right) > 1.0e-8,
            axis=2,
        )
        command_sum += command
        command_square_sum += command * command

        alignment_valid = ~switch_scenarios | (step < switch_step) | (step >= switch_step + turn_grace_steps)
        alignment_sum += next_alignment * alignment_valid[None, :]
        alignment_samples += alignment_valid
        moving_alignment_sum += next_alignment * step_distance * alignment_valid[None, :]
        moving_alignment_distance += step_distance * alignment_valid[None, :]
        if step >= late_start:
            late_alignment_sum += next_alignment
            late_alignment_samples += 1

        if perturbation != "nominal":
            if schedule.pre_start <= step < schedule.perturb_start:
                pre_recovery_distance += target_head_step
            if schedule.perturb_end <= step < schedule.post_end:
                post_recovery_distance += target_head_step
            if step >= schedule.perturb_end:
                instant_target_head_speed = target_head_step / POLICY_TIMESTEP
                recovery_ema = recovery_ema * 0.82 + instant_target_head_speed * 0.18
                pre_duration = max(
                    (schedule.perturb_start - schedule.pre_start) * POLICY_TIMESTEP,
                    POLICY_TIMESTEP,
                )
                recovery_threshold = np.maximum(
                    pre_recovery_distance / pre_duration * 0.55,
                    RECOVERY_SPEED_FLOOR,
                )
                newly_recovered = ~recovered & (recovery_ema >= recovery_threshold) & (next_alignment >= 0.1)
                recovery_time[newly_recovered] = (step - schedule.perturb_end + 1) * POLICY_TIMESTEP
                recovered |= newly_recovered

    final_relative = target[None, :, :] - position
    final_distance = np.linalg.norm(final_relative, axis=2)
    final_direction = final_relative / np.maximum(final_distance[:, :, None], 1.0e-9)
    heading_alignment = np.sum(final_direction * forward, axis=2)
    mean_progress = np.mean(progress, axis=1)
    late_progress_speed = np.mean(late_progress, axis=1) / max(
        (late_progress_end - late_progress_start) * POLICY_TIMESTEP,
        POLICY_TIMESTEP,
    )
    worst_progress = np.min(progress, axis=1)
    bottom_count = min(2, scenario_count)
    bottom_progress = np.mean(
        np.partition(progress, bottom_count - 1, axis=1)[:, :bottom_count],
        axis=1,
    )
    covered_progress = np.mean(np.clip(progress, -1.0, 1.6), axis=1)
    efficiency = np.mean(
        np.maximum(progress, 0.0) / np.maximum(path_length, 0.15),
        axis=1,
    )
    reached = np.mean(final_distance < 0.72, axis=1)
    approach_ratio = np.mean(final_distance < 2.4, axis=1)
    close_ratio = np.mean(final_distance < 1.35, axis=1)
    mean_speed = path_length.mean(axis=1) / (episode_steps * POLICY_TIMESTEP)
    energy = energy_sum.mean(axis=1) / episode_steps
    mean_command = command_sum / episode_steps
    variance = np.maximum(
        command_square_sum / episode_steps - mean_command * mean_command,
        0.0,
    )
    temporal_std = np.mean(np.sqrt(variance), axis=(1, 2))
    segment_diversity = np.mean(np.std(mean_command, axis=2), axis=1)
    hidden_saturation = hidden_saturation_sum.mean(axis=1) / episode_steps
    hidden_derivative = hidden_derivative_sum.mean(axis=1) / episode_steps
    spin_penalty = np.mean(np.maximum(np.abs(angular_speed) - 2.8, 0.0), axis=1)
    slip_penalty = slip_sum.mean(axis=1) / episode_steps
    blocked_ratio = obstacle_contact_steps.mean(axis=1) / episode_steps
    path_alignment_by_scenario = alignment_sum / np.maximum(
        alignment_samples[None, :],
        1.0,
    )
    path_alignment = np.mean(path_alignment_by_scenario, axis=1)
    worst_path_alignment = np.min(path_alignment_by_scenario, axis=1)
    moving_alignment = np.sum(moving_alignment_sum, axis=1) / np.maximum(
        np.sum(moving_alignment_distance, axis=1),
        1.0e-9,
    )
    late_alignment = np.mean(late_alignment_sum, axis=1) / max(
        late_alignment_samples,
        1,
    )
    near_target_ratio = near_target_steps.mean(axis=1) / episode_steps
    total_path = np.sum(path_length, axis=1)
    signed_head_fraction = np.sum(signed_head_distance, axis=1) / np.maximum(
        total_path,
        1.0e-9,
    )
    positive_head_fraction = np.sum(positive_head_distance, axis=1) / np.maximum(
        total_path,
        1.0e-9,
    )
    lateral_fraction = np.sum(lateral_distance, axis=1) / np.maximum(total_path, 1.0e-9)
    head_speed = np.mean(signed_head_distance, axis=1) / (episode_steps * POLICY_TIMESTEP)
    target_head_speed = np.mean(target_facing_head_distance, axis=1) / (episode_steps * POLICY_TIMESTEP)

    goal_score = (
        mean_progress * 2.8
        + worst_progress * 5.5
        + bottom_progress * 3.0
        + covered_progress * 1.2
        + efficiency * 1.0
        + approach_ratio * 2.0
        + close_ratio * 3.0
        + reached * 8.0
        + near_target_ratio * 2.0
        + path_alignment * 1.2
        + worst_path_alignment * 0.5
    )
    head_score = head_leading_objective_component(
        signed_head_fraction,
        positive_head_fraction,
        lateral_fraction,
        moving_alignment,
        target_head_speed,
        late_alignment,
    )
    quality_score = (
        np.minimum(mean_speed, 1.0) * 0.35
        + temporal_std * 0.15
        + segment_diversity * 0.18
        + hidden_derivative * 0.45
        - energy * 0.18
        - hidden_saturation * 2.0
        - spin_penalty * 0.35
        - slip_penalty * 0.25
        - blocked_ratio * 0.25
    )

    if perturbation == "nominal":
        pre_speed = np.zeros_like(mean_progress)
        post_speed = np.zeros_like(mean_progress)
        recovery_ratio_by_scenario = np.zeros_like(progress)
        mean_recovery_time = np.zeros_like(mean_progress)
    else:
        pre_duration = max(
            (schedule.perturb_start - schedule.pre_start) * POLICY_TIMESTEP,
            POLICY_TIMESTEP,
        )
        post_duration = max(
            (schedule.post_end - schedule.perturb_end) * POLICY_TIMESTEP,
            POLICY_TIMESTEP,
        )
        pre_speed_by_scenario = pre_recovery_distance / pre_duration
        post_speed_by_scenario = post_recovery_distance / post_duration
        recovery_ratio_by_scenario = post_speed_by_scenario / np.maximum(
            pre_speed_by_scenario,
            RECOVERY_SPEED_FLOOR,
        )
        pre_speed = np.mean(pre_speed_by_scenario, axis=1)
        post_speed = np.mean(post_speed_by_scenario, axis=1)
        recovery_horizon = max(
            (episode_steps - schedule.perturb_end) * POLICY_TIMESTEP,
            POLICY_TIMESTEP,
        )
        final_recovery_time = np.where(recovered, recovery_time, recovery_horizon)
        mean_recovery_time = np.mean(final_recovery_time, axis=1)

    return V3PopulationRollout(
        selection_score=goal_score + head_score + quality_score,
        mean_progress=mean_progress,
        late_progress_speed=late_progress_speed,
        worst_progress=worst_progress,
        path_efficiency=efficiency,
        mean_final_distance=np.mean(final_distance, axis=1),
        mean_speed=mean_speed,
        actuator_temporal_std=temporal_std,
        segment_diversity=segment_diversity,
        mean_energy=energy,
        hidden_saturation_ratio=hidden_saturation,
        mean_hidden_tanh_derivative=hidden_derivative,
        reached_ratio=reached,
        mean_displacement=np.mean(np.linalg.norm(position, axis=2), axis=1),
        mean_heading_alignment=np.mean(heading_alignment, axis=1),
        signed_head_leading_fraction=signed_head_fraction,
        positive_head_leading_fraction=positive_head_fraction,
        lateral_motion_fraction=lateral_fraction,
        mean_head_leading_speed=head_speed,
        target_facing_head_leading_speed=target_head_speed,
        moving_target_facing_alignment=moving_alignment,
        late_target_facing_alignment=late_alignment,
        mean_pre_recovery_speed=pre_speed,
        mean_post_recovery_speed=post_speed,
        mean_recovery_ratio=np.mean(recovery_ratio_by_scenario, axis=1),
        worst_recovery_ratio=np.min(recovery_ratio_by_scenario, axis=1),
        mean_recovery_time_seconds=mean_recovery_time,
    )


def head_leading_objective_component(
    signed_head_leading_fraction: np.ndarray,
    positive_head_leading_fraction: np.ndarray,
    lateral_motion_fraction: np.ndarray,
    moving_target_facing_alignment: np.ndarray,
    target_facing_head_leading_speed: np.ndarray,
    late_target_facing_alignment: np.ndarray,
) -> np.ndarray:
    """Score body-axis locomotion without preferring a fixed world heading."""
    return (
        signed_head_leading_fraction * 11.0
        + positive_head_leading_fraction * 3.5
        - lateral_motion_fraction * 9.5
        + moving_target_facing_alignment * 5.0
        + target_facing_head_leading_speed * 7.0
        + late_target_facing_alignment * 2.0
    )


def perturbation_recovery_score(rollout: V3PopulationRollout) -> np.ndarray:
    """Reward absolute post-event locomotion as well as relative recovery."""
    return (
        np.clip(rollout.mean_recovery_ratio, 0.0, 1.25) * 2.2
        + np.clip(rollout.worst_recovery_ratio, 0.0, 1.0) * 1.4
        + rollout.mean_post_recovery_speed * 7.0
        + rollout.signed_head_leading_fraction * 2.0
        + rollout.moving_target_facing_alignment * 1.5
        + rollout.mean_progress * 0.35
        - rollout.mean_recovery_time_seconds * 0.22
    )


def head_leading_feasibility_score(
    nominal: V3PopulationRollout,
    combined_perturbed: V3PopulationRollout,
) -> np.ndarray:
    """Use margins correlated with, but stricter than, browser promotion gates."""
    nominal_signed = canonical_v3_diagnostic(nominal.signed_head_leading_fraction)
    nominal_lateral = canonical_v3_diagnostic(nominal.lateral_motion_fraction)
    nominal_alignment = canonical_v3_diagnostic(nominal.moving_target_facing_alignment)
    recovered_signed = canonical_v3_diagnostic(combined_perturbed.signed_head_leading_fraction)
    recovered_lateral = canonical_v3_diagnostic(combined_perturbed.lateral_motion_fraction)
    recovered_alignment = canonical_v3_diagnostic(combined_perturbed.moving_target_facing_alignment)
    nominal_violation = (
        np.maximum(NOMINAL_SIGNED_HEAD_FLOOR - nominal_signed, 0.0) / NOMINAL_SIGNED_HEAD_FLOOR
        + np.maximum(nominal_lateral - NOMINAL_LATERAL_FRACTION_CEILING, 0.0)
        / NOMINAL_LATERAL_FRACTION_CEILING
        + np.maximum(NOMINAL_TARGET_ALIGNMENT_FLOOR - nominal_alignment, 0.0)
        / NOMINAL_TARGET_ALIGNMENT_FLOOR
    )
    recovered_violation = (
        np.maximum(RECOVERED_SIGNED_HEAD_FLOOR - recovered_signed, 0.0) / RECOVERED_SIGNED_HEAD_FLOOR
        + np.maximum(recovered_lateral - RECOVERED_LATERAL_FRACTION_CEILING, 0.0)
        / RECOVERED_LATERAL_FRACTION_CEILING
        + np.maximum(
            RECOVERED_TARGET_ALIGNMENT_FLOOR - recovered_alignment,
            0.0,
        )
        / RECOVERED_TARGET_ALIGNMENT_FLOOR
    )
    feasible = (nominal_violation == 0.0) & (recovered_violation == 0.0)
    return -(nominal_violation + recovered_violation) * 60.0 + feasible.astype(np.float64) * 20.0


def evaluate_local_causal_response(population: np.ndarray) -> LocalCausalResponse:
    """Match the browser microscope's one-tick local contact/obstacle pulse."""
    candidate_count = population.shape[0]
    scalar_shape = (candidate_count, 1)
    segment_shape = (candidate_count, 1, SEGMENT_COUNT)
    initial_hidden = population[:, INITIAL_SLICE, None].transpose(0, 2, 1)
    baseline_hidden = initial_hidden.copy()
    pulsed_hidden = initial_hidden.copy()
    baseline_command = np.zeros(segment_shape, dtype=np.float64)
    pulsed_command = np.zeros_like(baseline_command)
    joint = np.zeros(segment_shape, dtype=np.float64)
    joint_velocity = np.zeros_like(joint)
    zero_local = np.zeros_like(joint)
    input_weights = population[:, INPUT_SLICE]
    recurrent_weights = population[:, RECURRENT_SLICE]
    output_weights = population[:, OUTPUT_SLICE]
    target_forward = np.full(scalar_shape, 0.7, dtype=np.float64)
    target_right = np.full(scalar_shape, -0.25, dtype=np.float64)
    target_distance = np.full(scalar_shape, 0.8, dtype=np.float64)
    zero_scalar = np.zeros(scalar_shape, dtype=np.float64)
    friction = np.full(scalar_shape, 0.9, dtype=np.float64)
    urgency = np.full(scalar_shape, 0.8, dtype=np.float64)
    deltas: list[np.ndarray] = []
    pulse_segment = 7

    for tick in range(12):
        baseline_contact = np.ones(segment_shape, dtype=np.float64)
        pulsed_contact = baseline_contact.copy()
        pulsed_slip = np.zeros_like(joint)
        pulsed_obstacle_forward = np.zeros_like(joint)
        pulsed_obstacle_right = np.zeros_like(joint)
        if tick == 0:
            pulsed_contact[:, :, pulse_segment] = 0.0
            pulsed_slip[:, :, pulse_segment] = 1.7
            pulsed_obstacle_forward[:, :, pulse_segment] = 0.8
            pulsed_obstacle_right[:, :, pulse_segment] = 0.8

        baseline_hidden, baseline_command = segmental_network_step(
            baseline_hidden,
            baseline_command,
            joint,
            joint_velocity,
            *runtime_controller_sensors(
                target_forward,
                target_right,
                target_distance,
                zero_scalar,
                zero_scalar,
                friction,
                urgency,
                baseline_contact,
                zero_local,
                zero_local,
                zero_local,
            ),
            input_weights,
            recurrent_weights,
            output_weights,
        )
        pulsed_hidden, pulsed_command = segmental_network_step(
            pulsed_hidden,
            pulsed_command,
            joint,
            joint_velocity,
            *runtime_controller_sensors(
                target_forward,
                target_right,
                target_distance,
                zero_scalar,
                zero_scalar,
                friction,
                urgency,
                pulsed_contact,
                pulsed_slip,
                pulsed_obstacle_forward,
                pulsed_obstacle_right,
            ),
            input_weights,
            recurrent_weights,
            output_weights,
        )
        # Browser actions are returned as Float32 while recurrent command state
        # remains Float64 for the next tick.
        deltas.append(
            np.abs(pulsed_command[:, 0, :].astype(np.float32) - baseline_command[:, 0, :].astype(np.float32)).astype(
                np.float64
            )
        )

    return LocalCausalResponse(
        initial_local_delta=np.max(deltas[0][:, pulse_segment - 1 : pulse_segment + 2], axis=1),
        radius_two_delta=np.maximum(
            deltas[1][:, pulse_segment - 2],
            deltas[1][:, pulse_segment + 2],
        ),
        radius_three_delta=np.maximum(
            deltas[2][:, pulse_segment - 3],
            deltas[2][:, pulse_segment + 3],
        ),
        settled_delta=np.max(deltas[9], axis=1),
    )


def local_causal_response_score(response: LocalCausalResponse) -> np.ndarray:
    # The margin targets are intentionally stricter than the browser gate. A
    # normalized hinge supplies a useful gradient when the population is wholly
    # infeasible, while the large scale makes feasibility dominate gait quality.
    initial_delta = canonical_v3_diagnostic(response.initial_local_delta)
    radius_two_delta = canonical_v3_diagnostic(response.radius_two_delta)
    radius_three_delta = canonical_v3_diagnostic(response.radius_three_delta)
    settled_delta = canonical_v3_diagnostic(response.settled_delta)
    initial_deficit = np.maximum(
        (LOCAL_CAUSAL_INITIAL_FLOOR - initial_delta) / LOCAL_CAUSAL_INITIAL_FLOOR,
        0.0,
    )
    radius_two_deficit = np.maximum(
        (LOCAL_CAUSAL_RADIUS_TWO_FLOOR - radius_two_delta) / LOCAL_CAUSAL_RADIUS_TWO_FLOOR,
        0.0,
    )
    radius_three_deficit = np.maximum(
        (LOCAL_CAUSAL_RADIUS_THREE_FLOOR - radius_three_delta) / LOCAL_CAUSAL_RADIUS_THREE_FLOOR,
        0.0,
    )
    settle_excess = np.maximum(
        np.log10(np.maximum(settled_delta, LOCAL_CAUSAL_SETTLED_CEILING) / LOCAL_CAUSAL_SETTLED_CEILING),
        0.0,
    )
    all_gates = (
        (initial_delta > 0.1)
        & (radius_two_delta > 1.0e-3)
        & (radius_three_delta > 1.0e-3)
        & (settled_delta < 1.0e-5)
    )
    normalized_violation = (
        initial_deficit + radius_two_deficit + radius_three_deficit + np.clip(settle_excess, 0.0, 4.0)
    )
    return -normalized_violation * 200.0 + all_gates.astype(np.float64) * 25.0


def canonical_combined_recovery(
    control: V3PopulationRollout,
    perturbed: V3PopulationRollout,
) -> CanonicalCombinedRecovery:
    control_progress = canonical_v3_diagnostic(control.mean_progress)
    perturbed_progress = canonical_v3_diagnostic(perturbed.mean_progress)
    control_late_speed = canonical_v3_diagnostic(control.late_progress_speed)
    perturbed_late_speed = canonical_v3_diagnostic(perturbed.late_progress_speed)
    return CanonicalCombinedRecovery(
        control_target_progress=control_progress,
        perturbed_target_progress=perturbed_progress,
        progress_retention=canonical_v3_diagnostic(
            perturbed_progress / np.maximum(control_progress, 0.1)
        ),
        control_late_progress_speed=control_late_speed,
        perturbed_late_progress_speed=perturbed_late_speed,
        late_speed_retention=canonical_v3_diagnostic(
            perturbed_late_speed / np.maximum(control_late_speed, RECOVERY_SPEED_FLOOR)
        ),
        perturbed_signed_head_leading_fraction=canonical_v3_diagnostic(
            perturbed.signed_head_leading_fraction
        ),
        perturbed_lateral_motion_fraction=canonical_v3_diagnostic(
            perturbed.lateral_motion_fraction
        ),
        perturbed_moving_target_facing_alignment=canonical_v3_diagnostic(
            perturbed.moving_target_facing_alignment
        ),
    )


def selection_margin_results(
    nominal: V3PopulationRollout,
    combined: CanonicalCombinedRecovery,
    causal_gap: np.ndarray,
    causal_progress_gap: np.ndarray,
    local_causal: LocalCausalResponse,
    combined_episode_steps: int,
    *,
    guarded: bool = False,
) -> dict[str, np.ndarray]:
    control_floor = combined_control_progress_floor(combined_episode_steps)
    perturbed_floor = control_floor * COMBINED_PERTURBED_PROGRESS_FLOOR_RATIO
    scalar_guard = SELECTION_GATE_GUARD_BAND if guarded else 0.0
    radius_guard = LOCAL_CAUSAL_RADIUS_GUARD_BAND if guarded else 0.0
    settled_guard = LOCAL_CAUSAL_SETTLED_GUARD_BAND if guarded else 0.0
    return {
        "nominalSignedHeadLeading": canonical_v3_diagnostic(nominal.signed_head_leading_fraction)
        >= NOMINAL_SIGNED_HEAD_FLOOR + scalar_guard,
        "nominalLateralMotion": canonical_v3_diagnostic(nominal.lateral_motion_fraction)
        <= NOMINAL_LATERAL_FRACTION_CEILING - scalar_guard,
        "nominalTargetAlignment": canonical_v3_diagnostic(nominal.moving_target_facing_alignment)
        >= NOMINAL_TARGET_ALIGNMENT_FLOOR + scalar_guard,
        "combinedControlProgress": combined.control_target_progress
        >= control_floor + scalar_guard,
        "combinedPerturbedProgress": combined.perturbed_target_progress
        >= perturbed_floor + scalar_guard,
        "combinedProgressRetention": combined.progress_retention
        >= COMBINED_PROGRESS_RETENTION_FLOOR + scalar_guard,
        "combinedLateSpeedRetention": combined.late_speed_retention
        >= COMBINED_LATE_SPEED_RETENTION_FLOOR + scalar_guard,
        "combinedControlLateSpeed": combined.control_late_progress_speed
        >= COMBINED_CONTROL_LATE_SPEED_FLOOR + scalar_guard,
        "recoveredSignedHeadLeading": combined.perturbed_signed_head_leading_fraction
        >= RECOVERED_SIGNED_HEAD_FLOOR + scalar_guard,
        "recoveredLateralMotion": combined.perturbed_lateral_motion_fraction
        <= RECOVERED_LATERAL_FRACTION_CEILING - scalar_guard,
        "recoveredTargetAlignment": combined.perturbed_moving_target_facing_alignment
        >= RECOVERED_TARGET_ALIGNMENT_FLOOR + scalar_guard,
        "causalOrderingGap": canonical_v3_diagnostic(causal_gap)
        >= V3_CAUSAL_ORDERING_GAP_FLOOR + scalar_guard,
        "causalProgressGap": canonical_v3_diagnostic(causal_progress_gap)
        >= V3_CAUSAL_PROGRESS_GAP_FLOOR + scalar_guard,
        "localCausalInitial": canonical_v3_diagnostic(local_causal.initial_local_delta)
        >= LOCAL_CAUSAL_INITIAL_FLOOR + scalar_guard,
        "localCausalRadiusTwo": canonical_v3_diagnostic(local_causal.radius_two_delta)
        >= LOCAL_CAUSAL_RADIUS_TWO_FLOOR + radius_guard,
        "localCausalRadiusThree": canonical_v3_diagnostic(local_causal.radius_three_delta)
        >= LOCAL_CAUSAL_RADIUS_THREE_FLOOR + radius_guard,
        "localCausalSettled": canonical_v3_diagnostic(local_causal.settled_delta)
        <= LOCAL_CAUSAL_SETTLED_CEILING - settled_guard,
    }


def all_selection_margins_feasible(results: dict[str, np.ndarray]) -> np.ndarray:
    return np.logical_and.reduce(tuple(results.values()))


def combine_v3_selection_score(
    nominal: V3PopulationRollout,
    slick: V3PopulationRollout,
    shove: V3PopulationRollout,
    combined_control: V3PopulationRollout,
    combined_perturbed: V3PopulationRollout,
    causal_gap: np.ndarray,
    causal_progress_gap: np.ndarray,
    local_causal_response: LocalCausalResponse,
    combined_episode_steps: int,
) -> np.ndarray:
    slick_score = perturbation_recovery_score(slick)
    shove_score = perturbation_recovery_score(shove)
    robustness_score = (slick_score + shove_score) * 0.5 + np.minimum(slick_score, shove_score) * 0.35
    combined = canonical_combined_recovery(combined_control, combined_perturbed)
    control_progress_floor = combined_control_progress_floor(combined_episode_steps)
    perturbed_progress_floor = control_progress_floor * COMBINED_PERTURBED_PROGRESS_FLOOR_RATIO
    control_progress_violation = (
        np.maximum(control_progress_floor - combined.control_target_progress, 0.0)
        / max(control_progress_floor, 0.1)
    )
    combined_violation = (
        np.maximum(
            perturbed_progress_floor - combined.perturbed_target_progress,
            0.0,
        )
        / max(perturbed_progress_floor, 0.1)
        + np.maximum(COMBINED_PROGRESS_RETENTION_FLOOR - combined.progress_retention, 0.0)
        / COMBINED_PROGRESS_RETENTION_FLOOR
        + np.maximum(COMBINED_LATE_SPEED_RETENTION_FLOOR - combined.late_speed_retention, 0.0)
        / COMBINED_LATE_SPEED_RETENTION_FLOOR
        + np.maximum(
            COMBINED_CONTROL_LATE_SPEED_FLOOR - combined.control_late_progress_speed,
            0.0,
        )
        / COMBINED_CONTROL_LATE_SPEED_FLOOR
    )
    combined_feasible = (
        (combined.control_target_progress >= control_progress_floor)
        & (combined.perturbed_target_progress >= perturbed_progress_floor)
        & (combined.progress_retention >= COMBINED_PROGRESS_RETENTION_FLOOR)
        & (combined.late_speed_retention >= COMBINED_LATE_SPEED_RETENTION_FLOOR)
        & (combined.control_late_progress_speed >= COMBINED_CONTROL_LATE_SPEED_FLOOR)
    )
    combined_score = (
        -control_progress_violation * 1200.0
        - combined_violation * 120.0
        + combined_feasible.astype(np.float64) * 20.0
        + np.where(
            combined.control_target_progress >= control_progress_floor,
            np.clip(combined.progress_retention, 0.0, 1.1) * 2.0
            + np.clip(combined.late_speed_retention, 0.0, 1.1) * 2.0,
            0.0,
        )
        + combined.perturbed_signed_head_leading_fraction
    )
    causal_score = causal_ordering_feasibility_score(causal_gap)
    causal_progress_score = causal_progress_feasibility_score(causal_progress_gap)
    return (
        nominal.selection_score
        + head_leading_feasibility_score(nominal, combined_perturbed)
        + robustness_score
        + combined_score
        + causal_score
        + causal_progress_score
        + local_causal_response_score(local_causal_response)
    )


def combined_control_progress_floor(combined_episode_steps: int) -> float:
    scaled_progress_floor = 8.5 * combined_episode_steps / V3_COMBINED_RECOVERY_STEPS
    if combined_episode_steps < V3_COMBINED_RECOVERY_STEPS:
        return min(BOUNDED_COMBINED_CONTROL_PROGRESS_FLOOR, scaled_progress_floor)
    return scaled_progress_floor


def causal_ordering_feasibility_score(causal_gap: np.ndarray) -> np.ndarray:
    causal_gap = canonical_v3_diagnostic(causal_gap)
    violation = np.maximum(V3_CAUSAL_ORDERING_GAP_FLOOR - causal_gap, 0.0) / V3_CAUSAL_ORDERING_GAP_FLOOR
    feasible = causal_gap >= V3_CAUSAL_ORDERING_GAP_FLOOR
    return (
        -violation * 120.0
        + feasible.astype(np.float64) * 20.0
        + np.clip(causal_gap, -4.0, 5.0) * 0.65
    )


def causal_progress_feasibility_score(causal_progress_gap: np.ndarray) -> np.ndarray:
    causal_progress_gap = canonical_v3_diagnostic(causal_progress_gap)
    violation = np.maximum(
        V3_CAUSAL_PROGRESS_GAP_FLOOR - causal_progress_gap,
        0.0,
    ) / V3_CAUSAL_PROGRESS_GAP_FLOOR
    feasible = causal_progress_gap >= V3_CAUSAL_PROGRESS_GAP_FLOOR
    return (
        -violation * 160.0
        + feasible.astype(np.float64) * 18.0
        + np.clip(causal_progress_gap, -1.0, 1.0) * 4.0
    )


def combined_recovery_rollout_settings(
    episode_steps: int,
    mode: CombinedRecoveryMode,
) -> tuple[int, bool]:
    if mode == "recentered-900":
        return V3_COMBINED_RECOVERY_STEPS, True
    return episode_steps, False


def evaluate_selection_fitness_v3(
    population: np.ndarray,
    episode_steps: int,
    *,
    combined_recovery_mode: CombinedRecoveryMode = DEFAULT_COMBINED_RECOVERY_MODE,
) -> tuple[np.ndarray, np.ndarray, V3SelectionDiagnostics]:
    nominal = evaluate_population_v3_rollout(population, episode_steps, SCENARIOS)
    slick = evaluate_population_v3_rollout(
        population,
        episode_steps,
        V3_ROBUSTNESS_SCENARIOS,
        perturbation="slick-window",
    )
    shove = evaluate_population_v3_rollout(
        population,
        episode_steps,
        V3_ROBUSTNESS_SCENARIOS,
        perturbation="body-shove",
    )
    combined_steps, recenter_combined = combined_recovery_rollout_settings(
        episode_steps,
        combined_recovery_mode,
    )
    combined_control = evaluate_population_v3_rollout(
        population,
        combined_steps,
        V3_COMBINED_RECOVERY_SCENARIOS,
        perturbation="combined-recovery",
        apply_perturbation=False,
        recenter_unbounded=recenter_combined,
    )
    combined_perturbed = evaluate_population_v3_rollout(
        population,
        combined_steps,
        V3_COMBINED_RECOVERY_SCENARIOS,
        perturbation="combined-recovery",
        recenter_unbounded=recenter_combined,
    )
    causal_steps = max(90, episode_steps // 2)
    causal_full = evaluate_population_v3_rollout(
        population,
        causal_steps,
        V3_CAUSAL_SCENARIOS,
    )
    causal_shuffled = evaluate_population_v3_rollout(
        population,
        causal_steps,
        V3_CAUSAL_SCENARIOS,
        intervention="shuffled",
    )
    causal_gap = canonical_v3_diagnostic(causal_full.selection_score - causal_shuffled.selection_score)
    causal_progress_gap = canonical_v3_diagnostic(
        causal_full.mean_progress - causal_shuffled.mean_progress
    )
    local_causal = evaluate_local_causal_response(population)
    fitness = combine_v3_selection_score(
        nominal,
        slick,
        shove,
        combined_control,
        combined_perturbed,
        causal_gap,
        causal_progress_gap,
        local_causal,
        combined_steps,
    )
    combined = canonical_combined_recovery(combined_control, combined_perturbed)
    gate_results = selection_margin_results(
        nominal,
        combined,
        causal_gap,
        causal_progress_gap,
        local_causal,
        combined_steps,
    )
    guard_results = selection_margin_results(
        nominal,
        combined,
        causal_gap,
        causal_progress_gap,
        local_causal,
        combined_steps,
        guarded=True,
    )
    diagnostics = V3SelectionDiagnostics(
        signed_head_leading_fraction=nominal.signed_head_leading_fraction,
        lateral_motion_fraction=nominal.lateral_motion_fraction,
        target_facing_alignment=nominal.moving_target_facing_alignment,
        slick_recovery_ratio=slick.mean_recovery_ratio,
        shove_recovery_ratio=shove.mean_recovery_ratio,
        combined_progress_retention=combined.progress_retention,
        combined_late_speed_retention=combined.late_speed_retention,
        combined_control_progress=combined.control_target_progress,
        causal_progress_gap=causal_progress_gap,
        local_causal_initial_delta=local_causal.initial_local_delta,
        local_causal_radius_two_delta=local_causal.radius_two_delta,
        local_causal_radius_three_delta=local_causal.radius_three_delta,
        local_causal_settled_delta=local_causal.settled_delta,
        all_selection_gates_feasible=all_selection_margins_feasible(gate_results),
        all_selection_guard_bands_feasible=all_selection_margins_feasible(guard_results),
        selection_gate_count=np.sum(np.stack(tuple(gate_results.values())), axis=0),
        selection_guard_band_count=np.sum(np.stack(tuple(guard_results.values())), axis=0),
    )
    return fitness, causal_gap, diagnostics


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


def runtime_controller_sensors(
    target_forward: np.ndarray,
    target_right: np.ndarray,
    target_distance: np.ndarray,
    forward_speed: np.ndarray,
    angular_speed: np.ndarray,
    terrain_friction: np.ndarray,
    urgency: np.ndarray,
    contact_load: np.ndarray,
    slip_speed: np.ndarray,
    obstacle_forward: np.ndarray,
    obstacle_right: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return non-mutating sensor views matching the browser policy contract."""

    def sanitized(values: np.ndarray, name: str, fallback: float = 0.0) -> np.ndarray:
        minimum, maximum = RUNTIME_SENSOR_CLAMPS[name]
        finite = np.where(np.isfinite(values), values, fallback)
        return np.clip(finite, minimum, maximum)

    return (
        sanitized(target_forward, "targetForward"),
        sanitized(target_right, "targetRight"),
        sanitized(target_distance, "targetDistance"),
        sanitized(forward_speed, "forwardSpeed"),
        sanitized(angular_speed, "angularSpeed"),
        sanitized(terrain_friction, "terrainFriction"),
        sanitized(urgency, "urgency"),
        sanitized(contact_load, "contactLoad", 1.0),
        sanitized(slip_speed, "slipSpeed"),
        sanitized(obstacle_forward, "obstacleForward"),
        sanitized(obstacle_right, "obstacleRight"),
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
    runtime_sensor_contract: bool = False,
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
    contact_load = np.ones_like(joint)
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
        if runtime_sensor_contract:
            controller_sensors = runtime_controller_sensors(
                target_forward[None, :],
                target_right[None, :],
                distance[None, :] / 5.0,
                speed[None, :],
                angular_speed[None, :],
                friction[None, :],
                urgency[None, :],
                contact_load[None, :, :],
                slip_speed[None, :, :],
                obstacle_forward[None, :, :],
                obstacle_right[None, :, :],
            )
        else:
            controller_sensors = (
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
            )
        hidden_batch, action_batch = segmental_network_step(
            hidden[None, :, :],
            command[None, :, :],
            joint[None, :, :],
            joint_velocity[None, :, :],
            *controller_sensors,
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


def evaluate_genome_v3(
    genome: np.ndarray,
    episode_steps: int,
    *,
    combined_recovery_mode: CombinedRecoveryMode = DEFAULT_COMBINED_RECOVERY_MODE,
) -> RolloutMetricsV3:
    population = genome[None, :]
    nominal = evaluate_population_v3_rollout(population, episode_steps, SCENARIOS)
    slick = evaluate_population_v3_rollout(
        population,
        episode_steps,
        V3_ROBUSTNESS_SCENARIOS,
        perturbation="slick-window",
    )
    shove = evaluate_population_v3_rollout(
        population,
        episode_steps,
        V3_ROBUSTNESS_SCENARIOS,
        perturbation="body-shove",
    )
    combined_steps, recenter_combined = combined_recovery_rollout_settings(
        episode_steps,
        combined_recovery_mode,
    )
    combined_control = evaluate_population_v3_rollout(
        population,
        combined_steps,
        V3_COMBINED_RECOVERY_SCENARIOS,
        perturbation="combined-recovery",
        apply_perturbation=False,
        recenter_unbounded=recenter_combined,
    )
    combined_perturbed = evaluate_population_v3_rollout(
        population,
        combined_steps,
        V3_COMBINED_RECOVERY_SCENARIOS,
        perturbation="combined-recovery",
        recenter_unbounded=recenter_combined,
    )
    causal_steps = max(90, episode_steps // 2)
    causal_full = evaluate_population_v3_rollout(
        population,
        causal_steps,
        V3_CAUSAL_SCENARIOS,
    )
    causal_shuffled = evaluate_population_v3_rollout(
        population,
        causal_steps,
        V3_CAUSAL_SCENARIOS,
        intervention="shuffled",
    )
    causal_gap = canonical_v3_diagnostic(causal_full.selection_score - causal_shuffled.selection_score)
    causal_progress_gap = canonical_v3_diagnostic(
        causal_full.mean_progress - causal_shuffled.mean_progress
    )
    local_causal = evaluate_local_causal_response(population)
    fitness = combine_v3_selection_score(
        nominal,
        slick,
        shove,
        combined_control,
        combined_perturbed,
        causal_gap,
        causal_progress_gap,
        local_causal,
        combined_steps,
    )
    return RolloutMetricsV3(
        fitness=float(fitness[0]),
        progress=float(nominal.mean_progress[0]),
        path_efficiency=float(nominal.path_efficiency[0]),
        final_distance=float(nominal.mean_final_distance[0]),
        mean_speed=float(nominal.mean_speed[0]),
        actuator_temporal_std=float(nominal.actuator_temporal_std[0]),
        segment_diversity=float(nominal.segment_diversity[0]),
        mean_energy=float(nominal.mean_energy[0]),
        hidden_saturation_ratio=float(nominal.hidden_saturation_ratio[0]),
        mean_hidden_tanh_derivative=float(nominal.mean_hidden_tanh_derivative[0]),
        reached_ratio=float(nominal.reached_ratio[0]),
        displacement=float(nominal.mean_displacement[0]),
        heading_alignment=float(nominal.mean_heading_alignment[0]),
        signed_head_leading_fraction=float(nominal.signed_head_leading_fraction[0]),
        positive_head_leading_fraction=float(nominal.positive_head_leading_fraction[0]),
        lateral_motion_fraction=float(nominal.lateral_motion_fraction[0]),
        mean_head_leading_speed=float(nominal.mean_head_leading_speed[0]),
        target_facing_head_leading_speed=float(nominal.target_facing_head_leading_speed[0]),
        moving_target_facing_alignment=float(nominal.moving_target_facing_alignment[0]),
        late_target_facing_alignment=float(nominal.late_target_facing_alignment[0]),
        causal_gap=float(causal_gap[0]),
        causal_progress_gap=float(causal_progress_gap[0]),
        local_causal_initial_delta=float(local_causal.initial_local_delta[0]),
        local_causal_radius_two_delta=float(local_causal.radius_two_delta[0]),
        local_causal_radius_three_delta=float(local_causal.radius_three_delta[0]),
        local_causal_settled_delta=float(local_causal.settled_delta[0]),
        slick_window=recovery_metrics_from_rollout(slick),
        body_shove=recovery_metrics_from_rollout(shove),
        combined_recovery=combined_recovery_metrics(
            combined_control,
            combined_perturbed,
        ),
    )


def recovery_metrics_from_rollout(
    rollout: V3PopulationRollout,
) -> PerturbationRecoveryMetrics:
    return PerturbationRecoveryMetrics(
        pre_target_facing_head_speed=float(rollout.mean_pre_recovery_speed[0]),
        post_target_facing_head_speed=float(rollout.mean_post_recovery_speed[0]),
        recovery_ratio=float(rollout.mean_recovery_ratio[0]),
        worst_recovery_ratio=float(rollout.worst_recovery_ratio[0]),
        recovery_time_seconds=float(rollout.mean_recovery_time_seconds[0]),
        target_progress=float(rollout.mean_progress[0]),
        signed_head_leading_fraction=float(rollout.signed_head_leading_fraction[0]),
    )


def combined_recovery_metrics(
    control: V3PopulationRollout,
    perturbed: V3PopulationRollout,
) -> CombinedRecoveryMetrics:
    combined = canonical_combined_recovery(control, perturbed)
    return CombinedRecoveryMetrics(
        control_target_progress=float(combined.control_target_progress[0]),
        perturbed_target_progress=float(combined.perturbed_target_progress[0]),
        progress_retention=float(combined.progress_retention[0]),
        control_late_progress_speed=float(combined.control_late_progress_speed[0]),
        perturbed_late_progress_speed=float(combined.perturbed_late_progress_speed[0]),
        late_speed_retention=float(combined.late_speed_retention[0]),
        perturbed_signed_head_leading_fraction=float(combined.perturbed_signed_head_leading_fraction[0]),
        perturbed_lateral_motion_fraction=float(combined.perturbed_lateral_motion_fraction[0]),
        perturbed_moving_target_facing_alignment=float(
            combined.perturbed_moving_target_facing_alignment[0]
        ),
    )


def evaluate_ablations(
    genome: np.ndarray,
    episode_steps: int,
    *,
    runtime_sensor_contract: bool = False,
    raw: bool = False,
) -> dict[str, float]:
    full = evaluate_genome(
        genome,
        episode_steps,
        SCENARIOS,
        runtime_sensor_contract=runtime_sensor_contract,
    )
    zero = evaluate_genome(
        genome,
        episode_steps,
        SCENARIOS,
        mode="zero",
        runtime_sensor_contract=runtime_sensor_contract,
    )
    frozen = evaluate_genome(
        genome,
        episode_steps,
        SCENARIOS,
        mode="frozen",
        runtime_sensor_contract=runtime_sensor_contract,
    )
    shuffled = evaluate_genome(
        genome,
        episode_steps,
        SCENARIOS,
        mode="shuffled",
        runtime_sensor_contract=runtime_sensor_contract,
    )
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
    no_friction = evaluate_genome(
        genome,
        episode_steps,
        no_friction_scenarios,
        runtime_sensor_contract=runtime_sensor_contract,
    )
    measurements = {
        "fullProgress": full.progress,
        "zeroProgress": zero.progress,
        "frozenProgress": frozen.progress,
        "shuffledProgress": shuffled.progress,
        "noFrictionDisplacement": no_friction.displacement,
    }
    if raw:
        return measurements
    return {
        "fullProgress": round(measurements["fullProgress"], 8),
        "zeroProgress": round(measurements["zeroProgress"], 8),
        "frozenProgress": round(measurements["frozenProgress"], 8),
        "shuffledProgress": round(measurements["shuffledProgress"], 8),
        "noFrictionDisplacement": round(measurements["noFrictionDisplacement"], 10),
    }


def finalize_v3_artifact_evaluation(
    genome: np.ndarray,
    episode_steps: int,
    *,
    combined_recovery_mode: CombinedRecoveryMode = DEFAULT_COMBINED_RECOVERY_MODE,
) -> tuple[np.ndarray, RolloutMetricsV3, dict[str, float]]:
    """Evaluate final V3 metadata from exactly the genome persisted to JSON."""
    artifact_genome = artifact_precision_genome(genome)
    metrics = evaluate_genome_v3(
        artifact_genome,
        episode_steps,
        combined_recovery_mode=combined_recovery_mode,
    )
    ablations = evaluate_ablations(
        artifact_genome,
        episode_steps,
        runtime_sensor_contract=True,
        raw=True,
    )
    return artifact_genome, metrics, ablations


def deterministic_trace_hash(genome: np.ndarray, episode_steps: int) -> str:
    metrics = evaluate_genome(genome, episode_steps, SCENARIOS)
    payload = json.dumps(metrics_json(metrics), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_trace_hash_v3(metrics: RolloutMetricsV3) -> str:
    payload = json.dumps(
        {
            "metrics": metrics_v3_json(metrics),
            "perturbationRecovery": perturbation_recovery_json(metrics),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
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


def make_artifact_v3(
    genome: np.ndarray,
    *,
    model_version: str,
    seed: int,
    generations: int,
    population_size: int,
    elite_count: int,
    episode_steps: int,
    metrics: RolloutMetricsV3,
    ablations: dict[str, float],
    genome_recipe_hash: str,
    warm_start_metadata: dict[str, str] | None,
    combined_recovery_mode: CombinedRecoveryMode,
) -> dict[str, Any]:
    validate_artifact_precision_genome(genome)
    validate_v3_recipe_identity(
        genome,
        model_version=model_version,
        seed=seed,
        generations=generations,
        population_size=population_size,
        elite_count=elite_count,
        episode_steps=episode_steps,
        warm_start_metadata=warm_start_metadata,
        combined_recovery_mode=combined_recovery_mode,
        genome_recipe_hash=genome_recipe_hash,
    )
    training = v3_training_metadata(
        seed=seed,
        generations=generations,
        population_size=population_size,
        elite_count=elite_count,
        episode_steps=episode_steps,
        metrics=metrics,
        ablations=ablations,
        genome_recipe_hash=genome_recipe_hash,
        warm_start_metadata=warm_start_metadata,
        combined_recovery_mode=combined_recovery_mode,
    )
    publication_ready = bool(training["allPublicationGatesPassed"])
    if combined_recovery_mode == "bounded-480" and not publication_ready:
        raise SystemExit(
            "refusing to emit a bounded-480 V3 artifact whose serialized genome "
            "does not satisfy every advertised selection, ablation, and guard-band gate"
        )
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
        "training": training,
    }


def make_summary_v3(
    genome: np.ndarray,
    *,
    model_version: str,
    seed: int,
    generations: int,
    population_size: int,
    elite_count: int,
    episode_steps: int,
    metrics: RolloutMetricsV3,
    ablations: dict[str, float],
    genome_recipe_hash: str,
    warm_start_metadata: dict[str, str] | None,
    history: list[dict[str, float | int]],
    combined_recovery_mode: CombinedRecoveryMode,
) -> dict[str, Any]:
    validate_artifact_precision_genome(genome)
    validate_v3_recipe_identity(
        genome,
        model_version=model_version,
        seed=seed,
        generations=generations,
        population_size=population_size,
        elite_count=elite_count,
        episode_steps=episode_steps,
        warm_start_metadata=warm_start_metadata,
        combined_recovery_mode=combined_recovery_mode,
        genome_recipe_hash=genome_recipe_hash,
    )
    training = v3_training_metadata(
        seed=seed,
        generations=generations,
        population_size=population_size,
        elite_count=elite_count,
        episode_steps=episode_steps,
        metrics=metrics,
        ablations=ablations,
        genome_recipe_hash=genome_recipe_hash,
        warm_start_metadata=warm_start_metadata,
        combined_recovery_mode=combined_recovery_mode,
    )
    return {
        "schemaVersion": 3,
        "kind": "wurmkickflip.locomotionEvolutionSummary",
        "modelVersion": model_version,
        "genomeSize": GENOME_SIZE,
        "serializedGenome": rounded_vector(genome),
        **training,
        "hostDiagnostics": {
            "platform": {
                "system": platform.system(),
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "metrics": metrics_v3_json(metrics),
            "perturbationRecovery": perturbation_recovery_json(metrics),
            "ablations": canonical_v3_ablations(ablations),
            "deterministicTraceHash": deterministic_trace_hash_v3(metrics),
        },
        "history": history,
    }


def selection_margins_metadata(combined_episode_steps: int) -> dict[str, Any]:
    return {
        "guardBands": {
            "scalarThresholdDistance": SELECTION_GATE_GUARD_BAND,
            "localRadiusThresholdDistance": LOCAL_CAUSAL_RADIUS_GUARD_BAND,
            "localSettledThresholdDistance": LOCAL_CAUSAL_SETTLED_GUARD_BAND,
        },
        "nominal": {
            "signedHeadLeadingFractionAtLeast": NOMINAL_SIGNED_HEAD_FLOOR,
            "lateralMotionFractionAtMost": NOMINAL_LATERAL_FRACTION_CEILING,
            "movingTargetFacingAlignmentAtLeast": NOMINAL_TARGET_ALIGNMENT_FLOOR,
        },
        "combinedRecovery": {
            "controlTargetProgressAtLeast": combined_control_progress_floor(combined_episode_steps),
            "perturbedProgressFloorRatio": COMBINED_PERTURBED_PROGRESS_FLOOR_RATIO,
            "progressRetentionAtLeast": COMBINED_PROGRESS_RETENTION_FLOOR,
            "lateSpeedRetentionAtLeast": COMBINED_LATE_SPEED_RETENTION_FLOOR,
            "controlLateProgressSpeedAtLeast": COMBINED_CONTROL_LATE_SPEED_FLOOR,
            "signedHeadLeadingFractionAtLeast": RECOVERED_SIGNED_HEAD_FLOOR,
            "lateralMotionFractionAtMost": RECOVERED_LATERAL_FRACTION_CEILING,
            "movingTargetFacingAlignmentAtLeast": RECOVERED_TARGET_ALIGNMENT_FLOOR,
        },
        "causal": {
            "orderingGapAtLeast": V3_CAUSAL_ORDERING_GAP_FLOOR,
            "progressGapAtLeast": V3_CAUSAL_PROGRESS_GAP_FLOOR,
            "localInitialDeltaAtLeast": LOCAL_CAUSAL_INITIAL_FLOOR,
            "localRadiusTwoDeltaAtLeast": LOCAL_CAUSAL_RADIUS_TWO_FLOOR,
            "localRadiusThreeDeltaAtLeast": LOCAL_CAUSAL_RADIUS_THREE_FLOOR,
            "localSettledDeltaAtMost": LOCAL_CAUSAL_SETTLED_CEILING,
        },
    }


def selection_gate_results(
    metrics: RolloutMetricsV3,
    combined_episode_steps: int,
    *,
    guarded: bool = False,
) -> dict[str, bool]:
    def scalar(value: float) -> float:
        return float(canonical_v3_diagnostic(np.asarray([value], dtype=np.float64))[0])

    combined = metrics.combined_recovery
    control_floor = combined_control_progress_floor(combined_episode_steps)
    perturbed_floor = control_floor * COMBINED_PERTURBED_PROGRESS_FLOOR_RATIO
    scalar_guard = SELECTION_GATE_GUARD_BAND if guarded else 0.0
    radius_guard = LOCAL_CAUSAL_RADIUS_GUARD_BAND if guarded else 0.0
    settled_guard = LOCAL_CAUSAL_SETTLED_GUARD_BAND if guarded else 0.0
    return {
        "nominalSignedHeadLeading": scalar(metrics.signed_head_leading_fraction)
        >= NOMINAL_SIGNED_HEAD_FLOOR + scalar_guard,
        "nominalLateralMotion": scalar(metrics.lateral_motion_fraction)
        <= NOMINAL_LATERAL_FRACTION_CEILING - scalar_guard,
        "nominalTargetAlignment": scalar(metrics.moving_target_facing_alignment)
        >= NOMINAL_TARGET_ALIGNMENT_FLOOR + scalar_guard,
        "combinedControlProgress": scalar(combined.control_target_progress) >= control_floor + scalar_guard,
        "combinedPerturbedProgress": scalar(combined.perturbed_target_progress)
        >= perturbed_floor + scalar_guard,
        "combinedProgressRetention": scalar(combined.progress_retention)
        >= COMBINED_PROGRESS_RETENTION_FLOOR + scalar_guard,
        "combinedLateSpeedRetention": scalar(combined.late_speed_retention)
        >= COMBINED_LATE_SPEED_RETENTION_FLOOR + scalar_guard,
        "combinedControlLateSpeed": scalar(combined.control_late_progress_speed)
        >= COMBINED_CONTROL_LATE_SPEED_FLOOR + scalar_guard,
        "recoveredSignedHeadLeading": scalar(combined.perturbed_signed_head_leading_fraction)
        >= RECOVERED_SIGNED_HEAD_FLOOR + scalar_guard,
        "recoveredLateralMotion": scalar(combined.perturbed_lateral_motion_fraction)
        <= RECOVERED_LATERAL_FRACTION_CEILING - scalar_guard,
        "recoveredTargetAlignment": scalar(combined.perturbed_moving_target_facing_alignment)
        >= RECOVERED_TARGET_ALIGNMENT_FLOOR + scalar_guard,
        "causalOrderingGap": scalar(metrics.causal_gap)
        >= V3_CAUSAL_ORDERING_GAP_FLOOR + scalar_guard,
        "causalProgressGap": scalar(metrics.causal_progress_gap)
        >= V3_CAUSAL_PROGRESS_GAP_FLOOR + scalar_guard,
        "localCausalInitial": scalar(metrics.local_causal_initial_delta)
        >= LOCAL_CAUSAL_INITIAL_FLOOR + scalar_guard,
        "localCausalRadiusTwo": scalar(metrics.local_causal_radius_two_delta)
        >= LOCAL_CAUSAL_RADIUS_TWO_FLOOR + radius_guard,
        "localCausalRadiusThree": scalar(metrics.local_causal_radius_three_delta)
        >= LOCAL_CAUSAL_RADIUS_THREE_FLOOR + radius_guard,
        "localCausalSettled": scalar(metrics.local_causal_settled_delta)
        <= LOCAL_CAUSAL_SETTLED_CEILING - settled_guard,
    }


def ablation_protocol_metadata() -> dict[str, Any]:
    return {
        "runtimeSensorContract": True,
        "hostMagnitudesStoredInSummaryOnly": True,
        "guardBands": {
            "progressThresholdDistance": ABLATION_PROGRESS_GUARD_BAND,
            "conservationAtMost": ABLATION_CONSERVATION_GUARD_CEILING,
        },
        "thresholds": {
            "fullProgressAtLeast": ABLATION_FULL_PROGRESS_FLOOR,
            "zeroProgressAtMost": ABLATION_ZERO_PROGRESS_CEILING,
            "fullMinusFrozenProgressAtLeast": ABLATION_FROZEN_PROGRESS_GAP_FLOOR,
            "fullMinusShuffledProgressAtLeast": ABLATION_SHUFFLED_PROGRESS_GAP_FLOOR,
            "noFrictionDisplacementAtMost": ABLATION_NO_FRICTION_DISPLACEMENT_CEILING,
        },
    }


def ablation_gate_results(
    ablations: dict[str, float],
    *,
    guarded: bool = False,
) -> dict[str, bool]:
    progress_guard = ABLATION_PROGRESS_GUARD_BAND if guarded else 0.0
    conservation_ceiling = (
        ABLATION_CONSERVATION_GUARD_CEILING
        if guarded
        else ABLATION_NO_FRICTION_DISPLACEMENT_CEILING
    )
    return {
        "fullProgress": ablations["fullProgress"]
        >= ABLATION_FULL_PROGRESS_FLOOR + progress_guard,
        "zeroActionStationary": ablations["zeroProgress"] <= conservation_ceiling,
        "fullBeatsFrozen": (
            ablations["fullProgress"] - ablations["frozenProgress"]
            >= ABLATION_FROZEN_PROGRESS_GAP_FLOOR + progress_guard
        ),
        "fullBeatsShuffled": (
            ablations["fullProgress"] - ablations["shuffledProgress"]
            >= ABLATION_SHUFFLED_PROGRESS_GAP_FLOOR + progress_guard
        ),
        "noFrictionConservesCom": (
            ablations["noFrictionDisplacement"] <= conservation_ceiling
        ),
    }


def canonical_source_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def canonical_recipe_source_sha256() -> dict[str, str]:
    """Hash every source, contract, and locked environment input to V3 evolution."""
    return {
        relative_path: canonical_source_sha256(ROOT / relative_path)
        for relative_path in V3_RECIPE_SOURCE_PATHS
    }


def canonical_genome_recipe_hash(
    genome: np.ndarray,
    *,
    model_version: str,
    seed: int,
    generations: int,
    population_size: int,
    elite_count: int,
    episode_steps: int,
    warm_start_metadata: dict[str, str] | None,
    combined_recovery_mode: CombinedRecoveryMode,
) -> str:
    validate_model_version(model_version)
    combined_steps, _recenter = combined_recovery_rollout_settings(
        episode_steps,
        combined_recovery_mode,
    )
    payload = {
        "identitySchema": "wurmkickflip.canonicalGenomeRecipe.v2",
        "genome": rounded_vector(genome),
        "genomeDecimals": ARTIFACT_GENOME_DECIMALS,
        "modelVersion": model_version,
        "sourceSha256": canonical_recipe_source_sha256(),
        "recipe": {
            "algorithm": "elitist-mutation-evolution",
            "objectiveVersion": OBJECTIVE_V3_VERSION,
            "seed": seed,
            "generations": generations,
            "populationSize": population_size,
            "eliteCount": elite_count,
            "episodeSteps": episode_steps,
            "combinedRecoveryMode": combined_recovery_mode,
            "warmStart": warm_start_metadata,
            "genomeDecimals": ARTIFACT_GENOME_DECIMALS,
            "selectionFitnessDecimals": V3_SELECTION_FITNESS_DECIMALS,
            "gateDiagnosticDecimals": V3_GATE_DIAGNOSTIC_DECIMALS,
            "controllerSensorClamps": RUNTIME_SENSOR_CLAMPS,
            "selectionMargins": selection_margins_metadata(combined_steps),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def published_v3_recipe_inputs_match(
    *,
    seed: int,
    generations: int,
    population_size: int,
    elite_count: int,
    episode_steps: int,
    warm_start_metadata: dict[str, str] | None,
    combined_recovery_mode: CombinedRecoveryMode,
) -> bool:
    return (
        seed == 20260737
        and generations == 1
        and population_size == 4
        and elite_count == 1
        and episode_steps == 480
        and combined_recovery_mode == "bounded-480"
        and warm_start_metadata is not None
        and warm_start_metadata.get("sha256") == PUBLISHED_V3_WARM_START_SHA256
    )


def validate_v3_recipe_identity(
    genome: np.ndarray,
    *,
    model_version: str,
    seed: int,
    generations: int,
    population_size: int,
    elite_count: int,
    episode_steps: int,
    warm_start_metadata: dict[str, str] | None,
    combined_recovery_mode: CombinedRecoveryMode,
    genome_recipe_hash: str,
) -> None:
    validate_model_version(model_version)
    published_inputs = published_v3_recipe_inputs_match(
        seed=seed,
        generations=generations,
        population_size=population_size,
        elite_count=elite_count,
        episode_steps=episode_steps,
        warm_start_metadata=warm_start_metadata,
        combined_recovery_mode=combined_recovery_mode,
    )
    claims_published_model = model_version == PUBLISHED_V3_MODEL_VERSION
    if published_inputs and not claims_published_model:
        raise SystemExit(
            "the published V3 recipe requires --model-version "
            f"{PUBLISHED_V3_MODEL_VERSION!r}"
        )
    if claims_published_model and not published_inputs:
        raise SystemExit(
            f"model version {PUBLISHED_V3_MODEL_VERSION!r} is reserved for the exact "
            "published V3 recipe"
        )
    expected_hash = canonical_genome_recipe_hash(
        genome,
        model_version=model_version,
        seed=seed,
        generations=generations,
        population_size=population_size,
        elite_count=elite_count,
        episode_steps=episode_steps,
        warm_start_metadata=warm_start_metadata,
        combined_recovery_mode=combined_recovery_mode,
    )
    if genome_recipe_hash != expected_hash:
        raise SystemExit(
            "canonicalGenomeRecipeHash does not match the serialized genome, "
            "model version, recipe, and locked source inputs"
        )


def v3_training_metadata(
    *,
    seed: int,
    generations: int,
    population_size: int,
    elite_count: int,
    episode_steps: int,
    metrics: RolloutMetricsV3,
    ablations: dict[str, float],
    genome_recipe_hash: str,
    warm_start_metadata: dict[str, str] | None,
    combined_recovery_mode: CombinedRecoveryMode,
) -> dict[str, Any]:
    slick = perturbation_schedule(episode_steps, "slick-window")
    shove = perturbation_schedule(episode_steps, "body-shove")
    combined_steps, recenter_combined = combined_recovery_rollout_settings(
        episode_steps,
        combined_recovery_mode,
    )
    combined = perturbation_schedule(combined_steps, "combined-recovery")
    selection_results = selection_gate_results(metrics, combined_steps)
    selection_guard_results = selection_gate_results(metrics, combined_steps, guarded=True)
    ablation_results = ablation_gate_results(ablations)
    ablation_guard_results = ablation_gate_results(ablations, guarded=True)
    all_selection_gates_passed = all(selection_results.values())
    all_selection_guard_bands_passed = all(selection_guard_results.values())
    all_ablation_gates_passed = all(ablation_results.values())
    all_ablation_guard_bands_passed = all(ablation_guard_results.values())
    return {
        "algorithm": "elitist-mutation-evolution",
        "seed": seed,
        "generations": generations,
        "populationSize": population_size,
        "eliteCount": elite_count,
        "episodeSteps": episode_steps,
        "scenarioCount": len(SCENARIOS),
        "robustnessScenarioCount": len(V3_ROBUSTNESS_SCENARIOS),
        "combinedRecoveryScenarioCount": len(V3_COMBINED_RECOVERY_SCENARIOS),
        "combinedRecoveryMode": combined_recovery_mode,
        "combinedRecoveryEpisodeSteps": combined_steps,
        "causalScenarioCount": len(V3_CAUSAL_SCENARIOS),
        "domainRandomization": [
            "obstacles",
            "spatial-friction",
            "body-scale",
            "target-switches",
            "contact-loss",
            "transient-slick-window",
            "external-body-shove",
            "paired-shove-numb-low-traction-proxy",
        ],
        "objectiveVersion": OBJECTIVE_V3_VERSION,
        "actuatorPrecision": "float32-plant-command",
        "genomeDecimals": ARTIFACT_GENOME_DECIMALS,
        "selectionFitnessDecimals": V3_SELECTION_FITNESS_DECIMALS,
        "gateDiagnosticDecimals": V3_GATE_DIAGNOSTIC_DECIMALS,
        "recipeSourceSha256": canonical_recipe_source_sha256(),
        "controllerExposedTimingSignals": [],
        "controllerSensorClamps": {
            name: [minimum, maximum]
            for name, (minimum, maximum) in RUNTIME_SENSOR_CLAMPS.items()
        },
        "selectionMargins": selection_margins_metadata(combined_steps),
        "selectionGateResults": selection_results,
        "allSelectionGatesPassed": all_selection_gates_passed,
        "allSelectionMarginsFeasible": all_selection_gates_passed,
        "selectionGateGuardBandResults": selection_guard_results,
        "allSelectionGuardBandsPassed": all_selection_guard_bands_passed,
        "headLeadingMeasurement": {
            "axis": "tail-to-head segment COM axis; positive toward rendered head",
            "signedFraction": "sum(COM displacement dot body axis) / total COM path length",
            "lateralFraction": "sum(abs(COM displacement dot body right)) / total COM path length",
            "turnHandling": (
                "all world headings are valid; target-facing alignment has a short grace "
                "window after deterministic target switches"
            ),
            "selectionMargins": {
                "signedHeadLeadingFraction": NOMINAL_SIGNED_HEAD_FLOOR,
                "maximumLateralMotionFraction": NOMINAL_LATERAL_FRACTION_CEILING,
                "movingTargetFacingAlignment": NOMINAL_TARGET_ALIGNMENT_FLOOR,
                "combinedRecoveredSignedHeadLeadingFraction": RECOVERED_SIGNED_HEAD_FLOOR,
                "combinedRecoveredMaximumLateralMotionFraction": RECOVERED_LATERAL_FRACTION_CEILING,
                "combinedRecoveredTargetFacingAlignment": RECOVERED_TARGET_ALIGNMENT_FLOOR,
            },
        },
        "perturbationSchedules": {
            "slickWindow": {
                "preSteps": [slick.pre_start, slick.perturb_start],
                "activeSteps": [slick.perturb_start, slick.perturb_end],
                "postSteps": [slick.perturb_end, slick.post_end],
                "friction": SLICK_FRICTION,
            },
            "bodyShove": {
                "preSteps": [shove.pre_start, shove.perturb_start],
                "atStep": shove.perturb_start,
                "postSteps": [shove.perturb_end, shove.post_end],
                "alternatingLateralSpeed": SHOVE_LATERAL_SPEED,
                "backwardSpeed": SHOVE_BACKWARD_SPEED,
            },
            "combinedRecoveryProxy": {
                "controlTarget": [20.0, 0.0],
                "activeSteps": [combined.perturb_start, combined.perturb_end],
                "lateSpeedSteps": [
                    int(combined_steps * 0.50),
                    int(combined_steps * 0.70),
                ],
                "lateralShoveSpeed": COMBINED_SHOVE_LATERAL_SPEED,
                "numbSegment": COMBINED_NUMB_SEGMENT,
                "tractionScale": COMBINED_TRACTION_SCALE,
                **({"pythonArenaLimitedProxy": True} if not recenter_combined else {"unboundedFrameRecentering": True}),
                "plantContractModified": False,
            },
        },
        "recoveryMeasurement": {
            "speed": "positive head-axis speed multiplied by positive target-facing alignment",
            "ratio": (f"post-window mean speed / max(pre-window mean speed, {RECOVERY_SPEED_FLOOR:.2f} m/s)"),
            "time": (
                "first post-event step whose 0.18-EMA speed reaches max(55% of "
                "pre-window speed, 0.04 m/s) while target alignment is at least 0.1"
            ),
        },
        "localCausalProbe": {
            "pulseSegment": 7,
            "pulseTicks": 1,
            "browserGates": {
                "initialLocalDeltaGreaterThan": 0.1,
                "radiusTwoDeltaGreaterThan": 1.0e-3,
                "radiusThreeDeltaGreaterThan": 1.0e-3,
                "settledDeltaLessThan": 1.0e-5,
            },
            "selectionMargins": {
                "initialLocalDelta": LOCAL_CAUSAL_INITIAL_FLOOR,
                "radiusTwoDelta": LOCAL_CAUSAL_RADIUS_TWO_FLOOR,
                "radiusThreeDelta": LOCAL_CAUSAL_RADIUS_THREE_FLOOR,
                "settledDelta": LOCAL_CAUSAL_SETTLED_CEILING,
            },
            "actionPrecision": "float32-returned-action-float64-recurrent-command",
        },
        "ablationProtocol": ablation_protocol_metadata(),
        "ablationGateResults": ablation_results,
        "allAblationGatesPassed": all_ablation_gates_passed,
        "ablationGateGuardBandResults": ablation_guard_results,
        "allAblationGuardBandsPassed": all_ablation_guard_bands_passed,
        "allGateGuardBandsPassed": all_selection_guard_bands_passed
        and all_ablation_guard_bands_passed,
        "allPublicationGatesPassed": all_selection_gates_passed
        and all_ablation_gates_passed
        and all_selection_guard_bands_passed
        and all_ablation_guard_bands_passed,
        "canonicalGenomeRecipeHash": genome_recipe_hash,
        "pairedHostReproduction": {
            "requiredForPublication": True,
            "compared": ["serializedGenome", "artifactBytes"],
            "trainerAttestation": "none",
        },
        **({"warmStart": warm_start_metadata} if warm_start_metadata else {}),
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


def metrics_v3_json(metrics: RolloutMetricsV3) -> dict[str, float]:
    return {
        "fitness": round(metrics.fitness, HOST_DIAGNOSTIC_DECIMALS),
        "meanTargetProgress": round(metrics.progress, HOST_DIAGNOSTIC_DECIMALS),
        "pathEfficiency": round(metrics.path_efficiency, HOST_DIAGNOSTIC_DECIMALS),
        "meanFinalDistance": round(metrics.final_distance, HOST_DIAGNOSTIC_DECIMALS),
        "meanSpeed": round(metrics.mean_speed, HOST_DIAGNOSTIC_DECIMALS),
        "actuatorTemporalStd": round(metrics.actuator_temporal_std, HOST_DIAGNOSTIC_DECIMALS),
        "segmentDiversity": round(metrics.segment_diversity, HOST_DIAGNOSTIC_DECIMALS),
        "meanEnergy": round(metrics.mean_energy, HOST_DIAGNOSTIC_DECIMALS),
        "hiddenSaturationRatio": round(metrics.hidden_saturation_ratio, HOST_DIAGNOSTIC_DECIMALS),
        "meanHiddenTanhDerivative": round(metrics.mean_hidden_tanh_derivative, HOST_DIAGNOSTIC_DECIMALS),
        "reachedRatio": round(metrics.reached_ratio, HOST_DIAGNOSTIC_DECIMALS),
        "meanDisplacement": round(metrics.displacement, HOST_DIAGNOSTIC_DECIMALS),
        "meanHeadingAlignment": round(metrics.heading_alignment, HOST_DIAGNOSTIC_DECIMALS),
        "signedHeadLeadingFraction": round(metrics.signed_head_leading_fraction, HOST_DIAGNOSTIC_DECIMALS),
        "positiveHeadLeadingFraction": round(metrics.positive_head_leading_fraction, HOST_DIAGNOSTIC_DECIMALS),
        "lateralMotionFraction": round(metrics.lateral_motion_fraction, HOST_DIAGNOSTIC_DECIMALS),
        "meanHeadLeadingSpeed": round(metrics.mean_head_leading_speed, HOST_DIAGNOSTIC_DECIMALS),
        "targetFacingHeadLeadingSpeed": round(
            metrics.target_facing_head_leading_speed,
            HOST_DIAGNOSTIC_DECIMALS,
        ),
        "movingTargetFacingAlignment": round(
            metrics.moving_target_facing_alignment,
            HOST_DIAGNOSTIC_DECIMALS,
        ),
        "lateTargetFacingAlignment": round(metrics.late_target_facing_alignment, HOST_DIAGNOSTIC_DECIMALS),
        "causalOrderingGap": round(metrics.causal_gap, HOST_DIAGNOSTIC_DECIMALS),
        "causalProgressGap": round(metrics.causal_progress_gap, HOST_DIAGNOSTIC_DECIMALS),
        "localCausalInitialDelta": round(metrics.local_causal_initial_delta, HOST_DIAGNOSTIC_DECIMALS),
        "localCausalRadiusTwoDelta": round(metrics.local_causal_radius_two_delta, HOST_DIAGNOSTIC_DECIMALS),
        "localCausalRadiusThreeDelta": round(metrics.local_causal_radius_three_delta, HOST_DIAGNOSTIC_DECIMALS),
        "localCausalSettledDelta": round(metrics.local_causal_settled_delta, HOST_DIAGNOSTIC_DECIMALS),
    }


def perturbation_recovery_json(metrics: RolloutMetricsV3) -> dict[str, dict[str, float]]:
    return {
        "slickWindow": recovery_metrics_json(metrics.slick_window),
        "bodyShove": recovery_metrics_json(metrics.body_shove),
        "combinedRecoveryProxy": combined_recovery_json(metrics.combined_recovery),
    }


def recovery_metrics_json(
    metrics: PerturbationRecoveryMetrics,
) -> dict[str, float]:
    return {
        "preTargetFacingHeadSpeed": round(metrics.pre_target_facing_head_speed, HOST_DIAGNOSTIC_DECIMALS),
        "postTargetFacingHeadSpeed": round(metrics.post_target_facing_head_speed, HOST_DIAGNOSTIC_DECIMALS),
        "recoveryRatio": round(metrics.recovery_ratio, HOST_DIAGNOSTIC_DECIMALS),
        "worstScenarioRecoveryRatio": round(metrics.worst_recovery_ratio, HOST_DIAGNOSTIC_DECIMALS),
        "meanRecoveryTimeSeconds": round(metrics.recovery_time_seconds, HOST_DIAGNOSTIC_DECIMALS),
        "meanTargetProgress": round(metrics.target_progress, HOST_DIAGNOSTIC_DECIMALS),
        "signedHeadLeadingFraction": round(metrics.signed_head_leading_fraction, HOST_DIAGNOSTIC_DECIMALS),
    }


def combined_recovery_json(metrics: CombinedRecoveryMetrics) -> dict[str, float]:
    return {
        "controlTargetProgress": round(metrics.control_target_progress, HOST_DIAGNOSTIC_DECIMALS),
        "perturbedTargetProgress": round(metrics.perturbed_target_progress, HOST_DIAGNOSTIC_DECIMALS),
        "progressRetention": round(metrics.progress_retention, HOST_DIAGNOSTIC_DECIMALS),
        "controlLateProgressSpeed": round(metrics.control_late_progress_speed, HOST_DIAGNOSTIC_DECIMALS),
        "perturbedLateProgressSpeed": round(metrics.perturbed_late_progress_speed, HOST_DIAGNOSTIC_DECIMALS),
        "lateSpeedRetention": round(metrics.late_speed_retention, HOST_DIAGNOSTIC_DECIMALS),
        "perturbedSignedHeadLeadingFraction": round(
            metrics.perturbed_signed_head_leading_fraction,
            HOST_DIAGNOSTIC_DECIMALS,
        ),
        "perturbedLateralMotionFraction": round(
            metrics.perturbed_lateral_motion_fraction,
            HOST_DIAGNOSTIC_DECIMALS,
        ),
        "perturbedMovingTargetFacingAlignment": round(
            metrics.perturbed_moving_target_facing_alignment,
            HOST_DIAGNOSTIC_DECIMALS,
        ),
    }


def canonical_v3_ablations(ablations: dict[str, float]) -> dict[str, float]:
    return {
        name: round(value, HOST_DIAGNOSTIC_DECIMALS)
        for name, value in ablations.items()
    }


def rounded_vector(values: np.ndarray) -> list[float]:
    return [round(float(value), ARTIFACT_GENOME_DECIMALS) for value in values]


def artifact_precision_genome(genome: np.ndarray) -> np.ndarray:
    """Return the exact Float64 values recovered from the compact JSON artifact."""
    return np.asarray(rounded_vector(genome), dtype=np.float64)


def validate_artifact_precision_genome(genome: np.ndarray) -> None:
    if not np.array_equal(genome, artifact_precision_genome(genome)):
        raise SystemExit(
            "V3 artifact and summary construction require the exact eight-decimal "
            "serialized genome"
        )


def write_json(path: Path, value: dict[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(value, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(value, indent=2) + "\n"
    # Write bytes directly so artifact hashes do not depend on the host OS's
    # newline translation (Path.write_text emits CRLF on Windows).
    path.write_bytes(text.encode("utf-8"))


if __name__ == "__main__":
    main()
