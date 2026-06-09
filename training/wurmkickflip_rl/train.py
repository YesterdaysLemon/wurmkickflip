from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from .env import DEFAULT_CREATURE, DEFAULT_ENVIRONMENT, WurmkickflipEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--out", type=Path, default=Path("runs/ppo_wurmkickflip.zip"))
    parser.add_argument("--creature", type=Path, default=DEFAULT_CREATURE)
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--progress-bar", action="store_true")
    args = parser.parse_args()

    creature = json.loads(args.creature.read_text(encoding="utf-8"))
    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    env = WurmkickflipEnv(creature_config=creature, environment_config=environment)
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
    model.learn(total_timesteps=args.timesteps, progress_bar=args.progress_bar)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
