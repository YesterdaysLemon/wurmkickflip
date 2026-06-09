from __future__ import annotations

import argparse
from pathlib import Path

import onnx

from .contracts import ACTION_SIZE, OBSERVATION_SIZE


def _last_dim(value_info: onnx.ValueInfoProto) -> int | None:
    shape = value_info.type.tensor_type.shape
    if not shape.dim:
        return None
    last = shape.dim[-1]
    return last.dim_value or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("../public/models/wurmkickflip_policy.onnx"))
    args = parser.parse_args()

    model = onnx.load(args.model)
    onnx.checker.check_model(model)

    graph = model.graph
    if len(graph.input) != 1:
        raise SystemExit(f"expected 1 input, found {len(graph.input)}")
    if len(graph.output) != 1:
        raise SystemExit(f"expected 1 output, found {len(graph.output)}")

    input_last_dim = _last_dim(graph.input[0])
    output_last_dim = _last_dim(graph.output[0])
    if input_last_dim != OBSERVATION_SIZE:
        raise SystemExit(f"expected input last dim {OBSERVATION_SIZE}, found {input_last_dim}")
    if output_last_dim != ACTION_SIZE:
        raise SystemExit(f"expected output last dim {ACTION_SIZE}, found {output_last_dim}")

    print("ONNX policy validation passed.")


if __name__ == "__main__":
    main()
