from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .contracts import ACTION_SIZE, OBSERVATION_SIZE, POLICY_TIMESTEP, SEGMENT_COUNT

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CREATURE = ROOT / "public/configs/creatures/segmented-starter.json"
DEFAULT_ENVIRONMENT = ROOT / "public/configs/environments/adaptive-skate-terrarium.json"


@dataclass
class Domain:
    friction: float
    board_mass: float
    wheel_stiffness: float
    start_roll: float
    gravity_scale: float
    drag: float
    slope_push: float
    roughness: float
    obstacle_density: float
    actuator_strength: float
    sensor_noise: float


@dataclass(frozen=True)
class MorphologyProfile:
    part_count: int
    total_mass: float
    length: float
    width: float
    height: float
    contact_span_x: float
    contact_span_z: float
    average_friction: float
    flexibility: float
    damping: float
    motor_strength: float
    propulsion_efficiency: float
    stability: float
    ride_height: float


class WurmkickflipEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        creature_config: dict[str, Any] | None = None,
        environment_config: dict[str, Any] | None = None,
        enable_sensor_noise: bool = False,
    ) -> None:
        super().__init__()
        self.creature_config = creature_config if creature_config is not None else load_json(DEFAULT_CREATURE)
        self.environment_config = environment_config if environment_config is not None else load_json(DEFAULT_ENVIRONMENT)
        self.enable_sensor_noise = enable_sensor_noise
        self.morphology = morphology_profile(self.creature_config)
        self.task_seconds = float(self.environment_config.get("task", {}).get("durationSeconds", 20.0))
        self.failure_height = float(self.environment_config.get("task", {}).get("failureHeight", -0.35))
        self.reward_weights = dict(self.environment_config.get("rewardWeights", {}))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ACTION_SIZE,),
            dtype=np.float32,
        )
        self.rng = np.random.default_rng()
        self.previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.reset()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        randomization = self.environment_config.get("randomization", {})
        terrain = self.environment_config.get("terrain", {})
        skateboard = self.environment_config.get("skateboard", {})
        spawn = vec3(skateboard.get("spawnPosition"), [1.1, 0.42, 0.0])
        spawn_x = sample_range(self.rng, randomization, "skateboardSpawnX", spawn[0])
        spawn_z = sample_range(self.rng, randomization, "skateboardSpawnZ", spawn[2])
        skateboard_mass = sample_range(self.rng, randomization, "skateboardMass", float(skateboard.get("mass", 1.1)))
        wheel_friction = sample_range(self.rng, randomization, "wheelFriction", float(skateboard.get("wheelFriction", 0.82)))
        gravity_scale = sample_range(self.rng, randomization, "gravityScale", 1.0)
        drag_scale = sample_range(self.rng, randomization, "dragScale", 1.0)
        slope_degrees = sample_range(self.rng, randomization, "slopeDegrees", float(terrain.get("slopeDegrees", 0.0)))
        roughness = sample_range(self.rng, randomization, "roughness", float(terrain.get("roughness", 0.18)))
        obstacle_density = sample_range(
            self.rng,
            randomization,
            "obstacleDensity",
            float(terrain.get("obstacleDensity", 0.08)),
        )

        self.domain = Domain(
            friction=max(0.05, sample_range(self.rng, randomization, "frictionScale", 1.0) * self.morphology.average_friction),
            board_mass=max(0.1, skateboard_mass),
            wheel_stiffness=max(0.05, wheel_friction * float(self.rng.uniform(0.85, 1.15))),
            start_roll=float(self.rng.uniform(-0.08, 0.08)),
            gravity_scale=max(0.05, gravity_scale),
            drag=max(0.0, float(self.environment_config.get("world", {}).get("airDrag", 0.025)) * drag_scale),
            slope_push=float(np.sin(np.deg2rad(slope_degrees))) * 0.42,
            roughness=max(0.0, roughness),
            obstacle_density=max(0.0, obstacle_density),
            actuator_strength=max(0.05, sample_range(self.rng, randomization, "actuatorStrength", 1.0)),
            sensor_noise=max(0.0, sample_range(self.rng, randomization, "sensorNoise", 0.0)) if self.enable_sensor_noise else 0.0,
        )
        self.time = 0.0
        self.step_count = 0
        self.start_x = spawn_x
        self.total_contact_ratio = 0.0
        self.total_energy = 0.0
        self.fall_reason = "none"
        self.board = np.array(
            [spawn_x, spawn[1], spawn_z, 0.0, 0.0, 0.0, -0.03, self.domain.start_roll, 0.0],
            dtype=np.float32,
        )
        self.segments = np.zeros((SEGMENT_COUNT, 8), dtype=np.float32)
        self.previous_action.fill(0.0)
        self._update_segments(self.previous_action)
        return self._observe(), self._episode_info(0.0)

    def step(self, action: np.ndarray):
        action = np.clip(action.astype(np.float32), -1.0, 1.0)
        prior_x = float(self.board[0])
        prior_z = float(self.board[2])
        self.time += POLICY_TIMESTEP

        bends = np.clip((action[0::2] - action[1::2]) * 0.5, -1.0, 1.0)
        propulsion = float(np.mean(np.abs(bends)))
        lean = float(np.mean(bends * (np.arange(SEGMENT_COUNT) / SEGMENT_COUNT - 0.5)))
        contact_ratio = self._contact_ratio()
        morphology_drive = self.morphology.propulsion_efficiency * self.domain.actuator_strength
        contact_drive = 0.3 + contact_ratio * 0.85
        drag = self.domain.drag + self.domain.roughness * 0.018 + self.domain.obstacle_density * 0.028
        mass_load = max(0.2, self.domain.board_mass + self.morphology.total_mass * 0.28)

        target_velocity = (
            (0.18 + propulsion * 1.08 * morphology_drive * contact_drive) * self.domain.wheel_stiffness
            + self.domain.slope_push
        ) / mass_load
        self.board[3] += (target_velocity - self.board[3]) * 0.035 * self.domain.friction
        self.board[3] *= max(0.0, 1.0 - drag)
        lateral_target = np.sin(self.time * 0.9) * self.domain.roughness * 0.04 + lean * (0.04 + self.morphology.flexibility * 0.045)
        self.board[5] += (lateral_target - self.board[5]) * 0.045
        self.board[0] += self.board[3] * POLICY_TIMESTEP
        self.board[2] = np.clip(self.board[2] + self.board[5] * POLICY_TIMESTEP, -1.15, 1.15)
        self.board[1] = max(self.failure_height - 0.02, float(self.board[1]) + np.sin(self.time * 1.8) * 0.0004)
        self.board[6] += ((-0.04 + propulsion * 0.045 * self.domain.gravity_scale) - self.board[6]) * 0.04
        roll_noise = np.sin(self.time * 2.1) * (0.018 + self.domain.roughness * 0.025)
        self.board[7] += ((lean * (0.15 + self.morphology.height * 0.04) + roll_noise) - self.board[7]) * 0.05
        self.board[8] += (self.board[5] * 0.18 - self.board[8]) * 0.04

        self._update_segments(action)
        contact_ratio = self._contact_ratio()
        progress = float(self.board[0] - prior_x)
        energy = float(np.mean(np.abs(action)))
        balance = abs(float(self.board[7])) + abs(float(self.board[6])) * 0.5
        reward = (
            progress * 18.0 * float(self.reward_weights.get("forwardProgress", 1.0))
            + contact_ratio * 0.08 * float(self.reward_weights.get("skateboardContact", 1.8))
            - balance * 0.04 * float(self.reward_weights.get("stability", 0.45))
            - energy * 0.012 * float(self.reward_weights.get("energy", 0.08))
        )

        self.step_count += 1
        self.total_contact_ratio += contact_ratio
        self.total_energy += energy
        self.fall_reason = "none"

        terminated = False
        truncated = False
        if contact_ratio < 0.35:
            terminated = True
            self.fall_reason = "lost_contact"
        elif abs(float(self.board[2])) > 1.25:
            terminated = True
            self.fall_reason = "out_of_bounds"
        elif float(self.board[1]) < self.failure_height:
            terminated = True
            self.fall_reason = "fell"
        elif self.time >= self.task_seconds:
            truncated = True
            self.fall_reason = "timeout"

        if terminated:
            reward -= 2.0
        if self.board[0] >= 14.2:
            reward += 10.0
            truncated = True
            self.fall_reason = "none"

        self.previous_action = action
        self.board[3] = (self.board[0] - prior_x) / POLICY_TIMESTEP
        self.board[5] = (self.board[2] - prior_z) / POLICY_TIMESTEP
        return self._observe(), reward, terminated, truncated, self._episode_info(contact_ratio)

    def _update_segments(self, action: np.ndarray) -> None:
        spacing = self.morphology.length / max(SEGMENT_COUNT - 1, 1)
        center_offset = ((SEGMENT_COUNT - 1) * spacing) / 2
        for index in range(SEGMENT_COUNT):
            dorsal = float(action[index * 2])
            ventral = float(action[index * 2 + 1])
            bend = float(np.clip((dorsal - ventral) * 0.5, -1.0, 1.0))
            old = self.segments[index].copy()
            x = float(self.board[0]) - center_offset + index * spacing
            z = (
                float(self.board[2])
                + bend * self.morphology.width * (0.22 + self.morphology.flexibility * 0.28)
                + np.sin(self.time * 3.2 - index * 0.35) * (0.012 + self.domain.roughness * 0.045)
            )
            y = float(self.board[1]) + self.morphology.ride_height + abs(bend) * self.morphology.height * 0.04
            self.segments[index] = np.array(
                [
                    x,
                    y,
                    z,
                    (x - old[0]) / POLICY_TIMESTEP,
                    (y - old[1]) / POLICY_TIMESTEP,
                    (z - old[2]) / POLICY_TIMESTEP,
                    bend * 0.36,
                    float(self.board[8]) + bend * 0.18,
                ],
                dtype=np.float32,
            )

    def _contact_ratio(self) -> float:
        contacts = 0
        deck = vec3(self.environment_config.get("skateboard", {}).get("deckSize"), [1.9, 0.12, 0.58])
        contact_x = max(deck[0] * 0.52, self.morphology.contact_span_x * 0.5)
        contact_z = max(deck[2] * 0.62, self.morphology.contact_span_z * 0.42)
        for segment in self.segments:
            if abs(float(segment[0] - self.board[0])) < contact_x and abs(float(segment[2] - self.board[2])) < contact_z:
                contacts += 1
        return contacts / SEGMENT_COUNT

    def _episode_info(self, contact_ratio: float) -> dict[str, float | str]:
        step_count = max(self.step_count, 1)
        return {
            "contact_ratio": contact_ratio,
            "average_contact_ratio": self.total_contact_ratio / step_count,
            "energy_use": self.total_energy / step_count,
            "distance": max(0.0, float(self.board[0] - self.start_x)),
            "survival_time": self.time,
            "fall_reason": self.fall_reason,
        }

    def _observe(self) -> np.ndarray:
        observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        values: list[float] = [
            self.time,
            *self.board.tolist(),
            self._contact_ratio(),
            1.0,
            0.0,
            0.0,
        ]
        for segment in self.segments:
            values.extend(
                [
                    float(segment[0] - self.board[0]),
                    float(segment[1] - self.board[1]),
                    float(segment[2] - self.board[2]),
                    *segment[3:].tolist(),
                ]
            )
        values.extend(self.previous_action.tolist())
        observation[: min(len(values), OBSERVATION_SIZE)] = values[:OBSERVATION_SIZE]
        if self.domain.sensor_noise > 0:
            observation += self.rng.normal(0.0, self.domain.sensor_noise, OBSERVATION_SIZE).astype(np.float32)
        return observation


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def morphology_profile(creature: dict[str, Any]) -> MorphologyProfile:
    morphology = creature.get("morphology", {})
    parts = [part for part in morphology.get("bodyParts", []) if isinstance(part, dict)]
    joints = [joint for joint in morphology.get("joints", []) if isinstance(joint, dict)]
    if not parts:
        return MorphologyProfile(1, 1.0, 1.0, 0.5, 0.25, 1.0, 0.5, 0.9, 0.5, 1.0, 4.0, 1.0, 1.0, 0.18)

    positions = np.array([vec3(part.get("position"), [0.0, 0.0, 0.0]) for part in parts], dtype=np.float32)
    sizes = np.array([vec3(part.get("size"), [0.2, 0.2, 0.2]) for part in parts], dtype=np.float32)
    masses = np.array([max(0.02, float(part.get("mass", 0.15))) for part in parts], dtype=np.float32)
    frictions = np.array(
        [max(0.05, float(part.get("material", {}).get("friction", 0.9))) for part in parts],
        dtype=np.float32,
    )
    stiffness = np.array([max(0.01, float(joint.get("stiffness", 12.0))) for joint in joints], dtype=np.float32)
    damping = np.array([max(0.01, float(joint.get("damping", 1.8))) for joint in joints], dtype=np.float32)
    motors = np.array([max(0.01, float(joint.get("motorStrength", 4.0))) for joint in joints], dtype=np.float32)
    limits = np.array([abs(float(joint.get("limits", [-0.65, 0.65])[1]) - float(joint.get("limits", [-0.65, 0.65])[0])) for joint in joints], dtype=np.float32)

    min_corner = np.min(positions - sizes * 0.5, axis=0)
    max_corner = np.max(positions + sizes * 0.5, axis=0)
    extent = np.maximum(max_corner - min_corner, np.array([0.2, 0.2, 0.2], dtype=np.float32))
    total_mass = float(np.sum(masses))
    average_friction = float(np.average(frictions, weights=masses))
    average_stiffness = float(np.mean(stiffness)) if len(stiffness) else 12.0
    average_damping = float(np.mean(damping)) if len(damping) else 1.8
    average_motor = float(np.mean(motors)) if len(motors) else 4.0
    average_limit = float(np.mean(limits)) if len(limits) else 1.2
    flexibility = clamp((average_limit / 1.6) * (14.0 / (average_stiffness + 6.0)), 0.15, 1.8)
    propulsion_efficiency = clamp((average_motor / 4.0) * average_friction * (0.85 + flexibility * 0.18), 0.15, 2.4)
    stability = clamp((extent[2] + 0.35) / (extent[1] + 0.25) * (average_damping / 2.0), 0.2, 2.6)
    ride_height = clamp(float(np.mean(positions[:, 1]) - 0.42), 0.12, 0.42)

    return MorphologyProfile(
        part_count=len(parts),
        total_mass=max(0.05, total_mass),
        length=float(max(extent[0], 0.4)),
        width=float(max(extent[2], 0.25)),
        height=float(max(extent[1], 0.16)),
        contact_span_x=float(max(extent[0], 0.35)),
        contact_span_z=float(max(extent[2], 0.25)),
        average_friction=average_friction,
        flexibility=flexibility,
        damping=average_damping,
        motor_strength=average_motor,
        propulsion_efficiency=propulsion_efficiency,
        stability=stability,
        ride_height=ride_height,
    )


def sample_range(rng: np.random.Generator, randomization: dict[str, Any], key: str, default: float) -> float:
    value = randomization.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return default
    lower = float(value[0])
    upper = float(value[1])
    if lower == upper:
        return lower
    return float(rng.uniform(min(lower, upper), max(lower, upper)))


def vec3(value: Any, default: list[float]) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return default
    return [float(value[0]), float(value[1]), float(value[2])]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
