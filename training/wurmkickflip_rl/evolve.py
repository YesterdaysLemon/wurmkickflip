from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ACTION_SIZE, POLICY_TIMESTEP, SEGMENT_COUNT
from .env import WurmkickflipEnv


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CREATURE = ROOT / "public/configs/creatures/segmented-starter.json"
DEFAULT_ENVIRONMENT = ROOT / "public/configs/environments/adaptive-skate-terrarium.json"


@dataclass(frozen=True)
class ControllerCandidate:
    candidate_id: str
    wave_amplitude: float
    wave_frequency: float
    phase_offset: float
    forward_bias: float
    part_scale: float
    mass_scale: float
    friction_scale: float
    body_spread: float
    joint_stiffness_scale: float
    joint_damping_scale: float
    motor_strength_scale: float


@dataclass(frozen=True)
class CandidateScore:
    candidate: ControllerCandidate
    fitness: float
    reward: float
    distance: float
    contact_ratio: float
    energy: float
    survival_time: float
    fall_reasons: dict[str, int]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creature", type=Path, default=DEFAULT_CREATURE)
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--out", type=Path, default=Path("runs/evolution/summary.json"))
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--elite-count", type=int, default=3)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[1337, 2027, 4099])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--export-creature", type=Path, default=None)
    parser.add_argument("--export-manifest", type=Path, default=None)
    args = parser.parse_args()

    if args.generations <= 0:
        raise SystemExit("--generations must be greater than 0")
    if args.population_size <= 1:
        raise SystemExit("--population-size must be greater than 1")
    if not 0 < args.elite_count < args.population_size:
        raise SystemExit("--elite-count must be greater than 0 and less than --population-size")

    creature = load_json(args.creature)
    environment = load_json(args.environment)
    rng = np.random.default_rng(args.seed)

    population = initial_population(creature, args.population_size, rng)
    summaries: list[dict[str, Any]] = []
    best_score: CandidateScore | None = None

    for generation in range(args.generations):
        scores = [
            evaluate_candidate(candidate, args.eval_seeds, creature, environment)
            for candidate in population
        ]
        scores.sort(key=lambda score: score.fitness, reverse=True)
        best_score = scores[0] if best_score is None or scores[0].fitness > best_score.fitness else best_score
        summaries.append(generation_summary(generation, scores))

        elites = [score.candidate for score in scores[: args.elite_count]]
        population = next_population(elites, args.population_size, generation + 1, rng, creature)

    if best_score is None:
        raise SystemExit("No candidate scores were produced.")

    output = {
        "schemaVersion": 1,
        "kind": "wurmkickflip.evolutionSummary",
        "creatureId": string_value(creature, "id"),
        "environmentId": string_value(environment, "id"),
        "seed": args.seed,
        "evalSeeds": args.eval_seeds,
        "generations": args.generations,
        "populationSize": args.population_size,
        "eliteCount": args.elite_count,
        "episodeSeconds": episode_seconds(environment),
        "best": score_to_json(best_score),
        "generationSummaries": summaries,
    }
    if args.export_creature is not None:
        output["bestCreaturePath"] = str(args.export_creature)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.export_creature is not None:
        evolved_creature = evolved_creature_config(creature, best_score, environment, args.seed, args.eval_seeds)
        write_json(args.export_creature, evolved_creature)
        print(f"wrote {args.export_creature}")
    if args.export_manifest is not None:
        if args.export_creature is None:
            raise SystemExit("--export-manifest requires --export-creature")
        write_json(args.export_manifest, creature_manifest(args.export_creature))
        print(f"wrote {args.export_manifest}")

    print(f"wrote {args.out}")
    print(f"best fitness {best_score.fitness:.4f} from {best_score.candidate.candidate_id}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def initial_population(creature: dict[str, Any], population_size: int, rng: np.random.Generator) -> list[ControllerCandidate]:
    parameters = creature.get("controller", {}).get("parameters", {})
    base = ControllerCandidate(
        candidate_id="g0-c0",
        wave_amplitude=float(parameters.get("waveAmplitude", 0.55)),
        wave_frequency=float(parameters.get("waveFrequency", 1.8)),
        phase_offset=float(parameters.get("phaseOffset", 0.72)),
        forward_bias=float(parameters.get("forwardBias", 0.2)),
        part_scale=1.0,
        mass_scale=1.0,
        friction_scale=1.0,
        body_spread=1.0,
        joint_stiffness_scale=1.0,
        joint_damping_scale=1.0,
        motor_strength_scale=1.0,
    )
    candidates = [base]
    for index in range(1, population_size):
        candidates.append(mutate_candidate(base, f"g0-c{index}", rng, creature))
    return candidates


def next_population(
    elites: list[ControllerCandidate],
    population_size: int,
    generation: int,
    rng: np.random.Generator,
    creature: dict[str, Any],
) -> list[ControllerCandidate]:
    candidates = [
        ControllerCandidate(
            candidate_id=f"g{generation}-c{index}",
            wave_amplitude=elite.wave_amplitude,
            wave_frequency=elite.wave_frequency,
            phase_offset=elite.phase_offset,
            forward_bias=elite.forward_bias,
            part_scale=elite.part_scale,
            mass_scale=elite.mass_scale,
            friction_scale=elite.friction_scale,
            body_spread=elite.body_spread,
            joint_stiffness_scale=elite.joint_stiffness_scale,
            joint_damping_scale=elite.joint_damping_scale,
            motor_strength_scale=elite.motor_strength_scale,
        )
        for index, elite in enumerate(elites)
    ]

    while len(candidates) < population_size:
        parent = elites[int(rng.integers(0, len(elites)))]
        candidates.append(mutate_candidate(parent, f"g{generation}-c{len(candidates)}", rng, creature))

    return candidates


def mutate_candidate(
    parent: ControllerCandidate,
    candidate_id: str,
    rng: np.random.Generator,
    creature: dict[str, Any],
) -> ControllerCandidate:
    jitter = float(creature.get("mutation", {}).get("controllerJitter", 0.25))
    part_range = number_range(creature, "partScaleRange", [0.75, 1.35])
    mass_range = number_range(creature, "massScaleRange", [0.65, 1.5])
    joint_jitter = float(creature.get("mutation", {}).get("jointLimitJitter", 0.18))
    return ControllerCandidate(
        candidate_id=candidate_id,
        wave_amplitude=clamp(parent.wave_amplitude + float(rng.normal(0, jitter * 0.35)), 0.05, 1.0),
        wave_frequency=clamp(parent.wave_frequency + float(rng.normal(0, jitter * 1.2)), 0.25, 5.0),
        phase_offset=clamp(parent.phase_offset + float(rng.normal(0, jitter * 0.8)), 0.05, 2.6),
        forward_bias=clamp(parent.forward_bias + float(rng.normal(0, jitter * 0.5)), -0.6, 0.8),
        part_scale=clamp(parent.part_scale + float(rng.normal(0, joint_jitter * 0.45)), part_range[0], part_range[1]),
        mass_scale=clamp(parent.mass_scale + float(rng.normal(0, joint_jitter * 0.6)), mass_range[0], mass_range[1]),
        friction_scale=clamp(parent.friction_scale + float(rng.normal(0, joint_jitter * 0.45)), 0.55, 1.55),
        body_spread=clamp(parent.body_spread + float(rng.normal(0, joint_jitter * 0.55)), 0.65, 1.55),
        joint_stiffness_scale=clamp(parent.joint_stiffness_scale + float(rng.normal(0, joint_jitter * 0.7)), 0.45, 1.85),
        joint_damping_scale=clamp(parent.joint_damping_scale + float(rng.normal(0, joint_jitter * 0.65)), 0.45, 1.9),
        motor_strength_scale=clamp(parent.motor_strength_scale + float(rng.normal(0, joint_jitter * 0.8)), 0.45, 2.1),
    )


def evaluate_candidate(
    candidate: ControllerCandidate,
    seeds: list[int],
    creature: dict[str, Any],
    environment: dict[str, Any],
) -> CandidateScore:
    candidate_creature = candidate_creature_config(creature, candidate)
    scores = [run_episode(candidate, seed, candidate_creature, environment) for seed in seeds]
    return CandidateScore(
        candidate=candidate,
        fitness=float(np.mean([score["fitness"] for score in scores])),
        reward=float(np.mean([score["reward"] for score in scores])),
        distance=float(np.mean([score["distance"] for score in scores])),
        contact_ratio=float(np.mean([score["contact_ratio"] for score in scores])),
        energy=float(np.mean([score["energy"] for score in scores])),
        survival_time=float(np.mean([score["survival_time"] for score in scores])),
        fall_reasons=count_fall_reasons([str(score["fall_reason"]) for score in scores]),
    )


def run_episode(
    candidate: ControllerCandidate,
    seed: int,
    creature: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, float | str]:
    env = WurmkickflipEnv(creature_config=creature, environment_config=environment)
    observation, _info = env.reset(seed=seed)
    total_reward = 0.0
    steps = min(int(episode_seconds(environment) / POLICY_TIMESTEP), 1200)
    info: dict[str, float | str] = {}

    for step in range(steps):
        action = cpg_action(candidate, step)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            steps = step + 1
            break

    distance = float(info.get("distance", max(0.0, float(env.board[0] + 1.0))))
    contact_ratio = float(info.get("average_contact_ratio", 0.0))
    energy = float(info.get("energy_use", 0.0))
    survival_time = float(info.get("survival_time", steps * POLICY_TIMESTEP))
    weights = environment.get("rewardWeights", {})
    fitness = (
        distance * float(weights.get("forwardProgress", 1.0))
        + contact_ratio * float(weights.get("skateboardContact", 1.8))
        + survival_time * float(weights.get("survival", 0.3))
        + max(0.0, float(env.board[3])) * float(weights.get("skateboardVelocity", 2.0))
        - energy * float(weights.get("energy", 0.08))
    )

    _ = observation
    return {
        "fitness": fitness,
        "reward": total_reward,
        "distance": distance,
        "contact_ratio": contact_ratio,
        "energy": energy,
        "survival_time": survival_time,
        "fall_reason": str(info.get("fall_reason", "none")),
    }


def cpg_action(candidate: ControllerCandidate, step: int) -> np.ndarray:
    action = np.zeros(ACTION_SIZE, dtype=np.float32)
    time = step * POLICY_TIMESTEP
    for segment in range(SEGMENT_COUNT):
        phase = time * candidate.wave_frequency * np.pi * 2.0 - segment * candidate.phase_offset
        drive = clamp(np.sin(phase) * candidate.wave_amplitude + candidate.forward_bias, -1.0, 1.0)
        action[segment * 2] = drive
        action[segment * 2 + 1] = -drive
    return action


def generation_summary(generation: int, scores: list[CandidateScore]) -> dict[str, Any]:
    best = scores[0]
    return {
        "generation": generation,
        "bestFitness": round(best.fitness, 6),
        "meanFitness": round(float(np.mean([score.fitness for score in scores])), 6),
        "bestCandidate": score_to_json(best),
    }


def score_to_json(score: CandidateScore) -> dict[str, Any]:
    return {
        "candidateId": score.candidate.candidate_id,
        "fitness": round(score.fitness, 6),
        "reward": round(score.reward, 6),
        "distance": round(score.distance, 6),
        "contactRatio": round(score.contact_ratio, 6),
        "energy": round(score.energy, 6),
        "survivalTime": round(score.survival_time, 6),
        "fallReasons": score.fall_reasons,
        "controller": {
            "kind": "cpg",
            "waveAmplitude": round(score.candidate.wave_amplitude, 6),
            "waveFrequency": round(score.candidate.wave_frequency, 6),
            "phaseOffset": round(score.candidate.phase_offset, 6),
            "forwardBias": round(score.candidate.forward_bias, 6),
        },
        "morphology": {
            "partScale": round(score.candidate.part_scale, 6),
            "massScale": round(score.candidate.mass_scale, 6),
            "frictionScale": round(score.candidate.friction_scale, 6),
            "bodySpread": round(score.candidate.body_spread, 6),
            "jointStiffnessScale": round(score.candidate.joint_stiffness_scale, 6),
            "jointDampingScale": round(score.candidate.joint_damping_scale, 6),
            "motorStrengthScale": round(score.candidate.motor_strength_scale, 6),
        },
    }


def evolved_creature_config(
    creature: dict[str, Any],
    best_score: CandidateScore,
    environment: dict[str, Any],
    seed: int,
    eval_seeds: list[int],
) -> dict[str, Any]:
    evolved = candidate_creature_config(creature, best_score.candidate)
    base_id = string_value(creature, "id") or "creature"
    candidate_id = best_score.candidate.candidate_id
    evolved["id"] = f"{base_id}-evolved-{candidate_id}"
    evolved["name"] = f"{string_value(creature, 'name') or base_id} Evolved {candidate_id}"
    evolved["description"] = (
        "Controller and morphology scales evolved by the Python genetic algorithm from "
        f"{base_id} against {string_value(environment, 'id') or 'environment'} "
        f"with seed {seed} and eval seeds {', '.join(str(item) for item in eval_seeds)}. "
        "The current surrogate models morphology through body size, mass, friction, spread, joint stiffness, damping, and motor-strength effects."
    )
    evolved["controller"] = {
        **dict(evolved.get("controller", {})),
        "kind": "cpg",
        "actionSize": int(evolved.get("controller", {}).get("actionSize", ACTION_SIZE)),
        "parameters": {
            **dict(evolved.get("controller", {}).get("parameters", {})),
            "waveAmplitude": round(best_score.candidate.wave_amplitude, 6),
            "waveFrequency": round(best_score.candidate.wave_frequency, 6),
            "phaseOffset": round(best_score.candidate.phase_offset, 6),
            "forwardBias": round(best_score.candidate.forward_bias, 6),
            "evolutionFitness": round(best_score.fitness, 6),
            "evolutionDistance": round(best_score.distance, 6),
            "evolutionContactRatio": round(best_score.contact_ratio, 6),
        },
    }
    return evolved


def candidate_creature_config(creature: dict[str, Any], candidate: ControllerCandidate) -> dict[str, Any]:
    evolved = copy.deepcopy(creature)
    morphology = evolved.get("morphology", {})
    body_parts = morphology.get("bodyParts", [])
    joints = morphology.get("joints", [])
    if isinstance(body_parts, list):
        root = first_position(body_parts)
        for part in body_parts:
            if not isinstance(part, dict):
                continue
            part["size"] = scale_vec3(part.get("size"), candidate.part_scale)
            part["mass"] = round(max(0.02, float(part.get("mass", 0.15)) * candidate.mass_scale), 6)
            part["position"] = spread_position(part.get("position"), root, candidate.body_spread)
            material = dict(part.get("material", {}))
            material["friction"] = round(clamp(float(material.get("friction", 0.9)) * candidate.friction_scale, 0.05, 2.5), 6)
            part["material"] = material

    if isinstance(joints, list):
        root = first_anchor(joints)
        for joint in joints:
            if not isinstance(joint, dict):
                continue
            joint["anchor"] = spread_position(joint.get("anchor"), root, candidate.body_spread)
            joint["stiffness"] = round(max(0.05, float(joint.get("stiffness", 12.0)) * candidate.joint_stiffness_scale), 6)
            joint["damping"] = round(max(0.05, float(joint.get("damping", 1.8)) * candidate.joint_damping_scale), 6)
            joint["motorStrength"] = round(max(0.05, float(joint.get("motorStrength", 4.0)) * candidate.motor_strength_scale), 6)

    evolved["controller"] = {
        **dict(evolved.get("controller", {})),
        "kind": "cpg",
        "actionSize": int(evolved.get("controller", {}).get("actionSize", ACTION_SIZE)),
        "parameters": {
            **dict(evolved.get("controller", {}).get("parameters", {})),
            "waveAmplitude": round(candidate.wave_amplitude, 6),
            "waveFrequency": round(candidate.wave_frequency, 6),
            "phaseOffset": round(candidate.phase_offset, 6),
            "forwardBias": round(candidate.forward_bias, 6),
            "partScale": round(candidate.part_scale, 6),
            "massScale": round(candidate.mass_scale, 6),
            "frictionScale": round(candidate.friction_scale, 6),
            "bodySpread": round(candidate.body_spread, 6),
            "jointStiffnessScale": round(candidate.joint_stiffness_scale, 6),
            "jointDampingScale": round(candidate.joint_damping_scale, 6),
            "motorStrengthScale": round(candidate.motor_strength_scale, 6),
        },
    }
    return evolved


def creature_manifest(export_creature: Path) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "kind": "wurmkickflip.generatedCreatureManifest",
        "creatures": [browser_public_path(export_creature)],
    }


def browser_public_path(path: Path) -> str:
    parts = path.as_posix().split("/")
    if "public" in parts:
        public_index = parts.index("public")
        return "/" + "/".join(parts[public_index + 1 :])
    return path.as_posix()


def number_range(creature: dict[str, Any], key: str, fallback: list[float]) -> tuple[float, float]:
    value = creature.get("mutation", {}).get(key, fallback)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        value = fallback
    lower = float(value[0])
    upper = float(value[1])
    return (min(lower, upper), max(lower, upper))


def scale_vec3(value: Any, scale: float) -> list[float]:
    vector = vec3(value, [0.2, 0.2, 0.2])
    return [round(max(0.025, item * scale), 6) for item in vector]


def spread_position(value: Any, root: list[float], spread: float) -> list[float]:
    position = vec3(value, root)
    return [round(root[index] + (position[index] - root[index]) * spread, 6) for index in range(3)]


def first_position(body_parts: list[Any]) -> list[float]:
    for part in body_parts:
        if isinstance(part, dict):
            return vec3(part.get("position"), [0.0, 0.0, 0.0])
    return [0.0, 0.0, 0.0]


def first_anchor(joints: list[Any]) -> list[float]:
    for joint in joints:
        if isinstance(joint, dict):
            return vec3(joint.get("anchor"), [0.0, 0.0, 0.0])
    return [0.0, 0.0, 0.0]


def vec3(value: Any, fallback: list[float]) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return fallback
    return [float(value[0]), float(value[1]), float(value[2])]


def episode_seconds(environment: dict[str, Any]) -> float:
    return float(environment.get("task", {}).get("durationSeconds", 20.0))


def string_value(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    return item if isinstance(item, str) else ""


def count_fall_reasons(reasons: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reason in reasons:
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


if __name__ == "__main__":
    main()
