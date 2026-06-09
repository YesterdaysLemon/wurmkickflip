from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class CandidateScore:
    candidate: ControllerCandidate
    fitness: float
    reward: float
    distance: float
    contact_ratio: float
    energy: float
    survival_time: float


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
            evaluate_candidate(candidate, args.eval_seeds, environment)
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"best fitness {best_score.fitness:.4f} from {best_score.candidate.candidate_id}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def initial_population(creature: dict[str, Any], population_size: int, rng: np.random.Generator) -> list[ControllerCandidate]:
    parameters = creature.get("controller", {}).get("parameters", {})
    base = ControllerCandidate(
        candidate_id="g0-c0",
        wave_amplitude=float(parameters.get("waveAmplitude", 0.55)),
        wave_frequency=float(parameters.get("waveFrequency", 1.8)),
        phase_offset=float(parameters.get("phaseOffset", 0.72)),
        forward_bias=float(parameters.get("forwardBias", 0.2)),
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
    return ControllerCandidate(
        candidate_id=candidate_id,
        wave_amplitude=clamp(parent.wave_amplitude + float(rng.normal(0, jitter * 0.35)), 0.05, 1.0),
        wave_frequency=clamp(parent.wave_frequency + float(rng.normal(0, jitter * 1.2)), 0.25, 5.0),
        phase_offset=clamp(parent.phase_offset + float(rng.normal(0, jitter * 0.8)), 0.05, 2.6),
        forward_bias=clamp(parent.forward_bias + float(rng.normal(0, jitter * 0.5)), -0.6, 0.8),
    )


def evaluate_candidate(
    candidate: ControllerCandidate,
    seeds: list[int],
    environment: dict[str, Any],
) -> CandidateScore:
    scores = [run_episode(candidate, seed, environment) for seed in seeds]
    return CandidateScore(
        candidate=candidate,
        fitness=float(np.mean([score["fitness"] for score in scores])),
        reward=float(np.mean([score["reward"] for score in scores])),
        distance=float(np.mean([score["distance"] for score in scores])),
        contact_ratio=float(np.mean([score["contact_ratio"] for score in scores])),
        energy=float(np.mean([score["energy"] for score in scores])),
        survival_time=float(np.mean([score["survival_time"] for score in scores])),
    )


def run_episode(candidate: ControllerCandidate, seed: int, environment: dict[str, Any]) -> dict[str, float]:
    env = WurmkickflipEnv()
    observation, _info = env.reset(seed=seed)
    total_reward = 0.0
    total_energy = 0.0
    total_contact = 0.0
    steps = min(int(episode_seconds(environment) / POLICY_TIMESTEP), 1200)

    for step in range(steps):
        action = cpg_action(candidate, step)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_energy += float(np.mean(np.abs(action)))
        total_contact += float(info.get("contact_ratio", 0.0))
        if terminated or truncated:
            steps = step + 1
            break

    distance = max(0.0, float(env.board[0] + 1.0))
    contact_ratio = total_contact / max(steps, 1)
    energy = total_energy / max(steps, 1)
    survival_time = steps * POLICY_TIMESTEP
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
        "controller": {
            "kind": "cpg",
            "waveAmplitude": round(score.candidate.wave_amplitude, 6),
            "waveFrequency": round(score.candidate.wave_frequency, 6),
            "phaseOffset": round(score.candidate.phase_offset, 6),
            "forwardBias": round(score.candidate.forward_bias, 6),
        },
    }


def episode_seconds(environment: dict[str, Any]) -> float:
    return float(environment.get("task", {}).get("durationSeconds", 20.0))


def string_value(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    return item if isinstance(item, str) else ""


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


if __name__ == "__main__":
    main()
