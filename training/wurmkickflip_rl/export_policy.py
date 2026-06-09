from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch as th
from stable_baselines3 import PPO

from .contracts import ACTION_SIZE, OBSERVATION_SIZE, POLICY_TIMESTEP


class OnnxablePolicy(th.nn.Module):
    def __init__(self, policy: th.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> th.Tensor:
        actions, _values, _log_prob = self.policy(observation, deterministic=True)
        return th.clamp(actions, -1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=Path("../public/models/wurmkickflip_policy.onnx"))
    parser.add_argument("--meta-out", type=Path, default=Path("../public/models/wurmkickflip_policy.meta.json"))
    parser.add_argument("--version", type=str, default="ppo-bootstrap-v1")
    args = parser.parse_args()

    model = PPO.load(args.checkpoint, device="cpu")
    wrapper = OnnxablePolicy(model.policy)
    wrapper.eval()

    dummy = th.zeros((1, OBSERVATION_SIZE), dtype=th.float32)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    th.onnx.export(
        wrapper,
        dummy,
        args.model_out,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )

    meta = {
        "modelVersion": args.version,
        "modelPath": "/models/wurmkickflip_policy.onnx",
        "observationSize": OBSERVATION_SIZE,
        "actionSize": ACTION_SIZE,
        "timestep": POLICY_TIMESTEP,
        "trainingReward": None,
        "observationMean": [],
        "observationStd": [],
    }
    args.meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"exported {args.model_out}")
    print(f"wrote {args.meta_out}")


if __name__ == "__main__":
    main()
