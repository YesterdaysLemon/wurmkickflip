from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from .env import WurmkickflipEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--out", type=Path, default=Path("runs/ppo_wurmkickflip.zip"))
    args = parser.parse_args()

    env = WurmkickflipEnv()
    check_env(env, warn=True)
    model = PPO(
        "MlpPolicy",
        env,
        batch_size=2048,
        learning_rate=3e-4,
        n_steps=2048,
        gamma=0.995,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
