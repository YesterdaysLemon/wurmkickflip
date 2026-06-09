from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .contracts import ACTION_SIZE, OBSERVATION_SIZE, POLICY_TIMESTEP, SEGMENT_COUNT


@dataclass
class Domain:
    friction: float
    board_mass: float
    wheel_stiffness: float
    start_roll: float


class WurmkickflipEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
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

        self.domain = Domain(
            friction=float(self.rng.uniform(0.55, 1.15)),
            board_mass=float(self.rng.uniform(0.8, 1.4)),
            wheel_stiffness=float(self.rng.uniform(0.7, 1.3)),
            start_roll=float(self.rng.uniform(-0.08, 0.08)),
        )
        self.time = 0.0
        self.step_count = 0
        self.start_x = -1.0
        self.total_contact_ratio = 0.0
        self.total_energy = 0.0
        self.fall_reason = "none"
        self.board = np.array([-1.0, 0.42, 0.0, 0.0, 0.0, 0.0, -0.03, self.domain.start_roll, 0.0], dtype=np.float32)
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

        target_velocity = (0.34 + propulsion * 0.98) * self.domain.wheel_stiffness / self.domain.board_mass
        self.board[3] += (target_velocity - self.board[3]) * 0.035 * self.domain.friction
        self.board[5] += ((np.sin(self.time * 0.9) * 0.05 + lean * 0.06) - self.board[5]) * 0.045
        self.board[0] += self.board[3] * POLICY_TIMESTEP
        self.board[2] = np.clip(self.board[2] + self.board[5] * POLICY_TIMESTEP, -1.15, 1.15)
        self.board[1] = 0.42 + np.sin(self.time * 1.8) * 0.012
        self.board[6] += ((-0.04 + propulsion * 0.045) - self.board[6]) * 0.04
        self.board[7] += ((lean * 0.18 + np.sin(self.time * 2.1) * 0.025) - self.board[7]) * 0.05
        self.board[8] += (self.board[5] * 0.18 - self.board[8]) * 0.04

        self._update_segments(action)
        contact_ratio = self._contact_ratio()
        progress = float(self.board[0] - prior_x)
        energy = float(np.mean(np.abs(action)))
        balance = abs(float(self.board[7])) + abs(float(self.board[6])) * 0.5
        reward = progress * 18.0 + contact_ratio * 0.08 - balance * 0.04 - energy * 0.012

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
        elif self.time >= 20.0:
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
        spacing = 0.105
        center_offset = ((SEGMENT_COUNT - 1) * spacing) / 2
        for index in range(SEGMENT_COUNT):
            dorsal = float(action[index * 2])
            ventral = float(action[index * 2 + 1])
            bend = float(np.clip((dorsal - ventral) * 0.5, -1.0, 1.0))
            old = self.segments[index].copy()
            x = float(self.board[0]) - center_offset + index * spacing
            z = float(self.board[2]) + bend * 0.18 + np.sin(self.time * 3.2 - index * 0.35) * 0.025
            y = float(self.board[1]) + 0.18 + abs(bend) * 0.035
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
        for segment in self.segments:
            if abs(float(segment[0] - self.board[0])) < 0.98 and abs(float(segment[2] - self.board[2])) < 0.42:
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
        return observation
