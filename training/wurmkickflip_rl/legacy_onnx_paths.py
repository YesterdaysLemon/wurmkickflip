from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_DIR = REPOSITORY_ROOT / "training" / "runs" / "ppo_onnx"
DEFAULT_MODEL_OUT = DEFAULT_EXPORT_DIR / "wurmkickflip_policy.onnx"
DEFAULT_META_OUT = DEFAULT_EXPORT_DIR / "wurmkickflip_policy.meta.json"
