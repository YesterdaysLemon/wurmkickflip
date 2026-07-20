from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch as th
from stable_baselines3 import PPO

from .contracts import ACTION_SIZE, OBSERVATION_SIZE, POLICY_TIMESTEP
from .legacy_onnx_paths import DEFAULT_META_OUT, DEFAULT_MODEL_OUT


class OnnxablePolicy(th.nn.Module):
    def __init__(self, policy: th.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor) -> th.Tensor:
        features = self.policy.extract_features(observation)
        if self.policy.share_features_extractor:
            latent_pi, _latent_vf = self.policy.mlp_extractor(features)
        else:
            pi_features, _vf_features = features
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
        actions = self.policy.action_net(latent_pi)
        return th.clamp(actions, -1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--meta-out", type=Path, default=DEFAULT_META_OUT)
    parser.add_argument("--version", type=str, default="ppo-bootstrap-v1")
    args = parser.parse_args()

    model_out = args.model_out.expanduser().resolve()
    meta_out = args.meta_out.expanduser().resolve()

    model = PPO.load(args.checkpoint, device="cpu")
    wrapper = OnnxablePolicy(model.policy)
    wrapper.eval()

    dummy = th.zeros((1, OBSERVATION_SIZE), dtype=th.float32)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    th.onnx.export(
        wrapper,
        dummy,
        model_out,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    meta = {
        "modelVersion": args.version,
        "modelPath": Path(os.path.relpath(model_out, start=meta_out.parent)).as_posix(),
        "observationSize": OBSERVATION_SIZE,
        "actionSize": ACTION_SIZE,
        "timestep": POLICY_TIMESTEP,
        "trainingReward": None,
        "observationMean": [],
        "observationStd": [],
    }
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"exported {model_out}")
    print(f"wrote {meta_out}")


if __name__ == "__main__":
    main()
