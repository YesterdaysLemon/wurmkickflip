from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .env import WurmkickflipEnv


def observation_from_fixture(path: Path) -> np.ndarray:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    env = WurmkickflipEnv()

    board = fixture["board"]
    env.time = float(fixture["time"])
    env.board = np.array(
        [
            board["x"],
            board["y"],
            board["z"],
            board["vx"],
            board["vy"],
            board["vz"],
            board["pitch"],
            board["roll"],
            board["yaw"],
        ],
        dtype=np.float32,
    )
    env.segments = np.array(
        [
            [
                segment["x"],
                segment["y"],
                segment["z"],
                segment["vx"],
                segment["vy"],
                segment["vz"],
                segment["pitch"],
                segment["yaw"],
            ]
            for segment in fixture["segments"]
        ],
        dtype=np.float32,
    )
    env.previous_action = np.array(fixture["previousAction"], dtype=np.float32)
    return env._observe()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()

    observation = observation_from_fixture(args.fixture)
    print(json.dumps(observation.astype(float).tolist()))


if __name__ == "__main__":
    main()
