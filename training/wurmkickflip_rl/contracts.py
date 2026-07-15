from __future__ import annotations

import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_LOCOMOTION_CONTRACT = json.loads(
    (_ROOT / "contracts" / "locomotion-v2.json").read_text(encoding="utf-8")
)

SEGMENT_COUNT = int(_LOCOMOTION_CONTRACT["segmentCount"])
ACTION_SIZE = int(_LOCOMOTION_CONTRACT["muscleChannelCount"])
OBSERVATION_SIZE = 174
POLICY_TIMESTEP = float(_LOCOMOTION_CONTRACT["timestep"])

OBSERVATION_HEADER_SIZE = 14
SEGMENT_OBSERVATION_SIZE = 8
PREVIOUS_ACTION_OFFSET = OBSERVATION_HEADER_SIZE + SEGMENT_COUNT * SEGMENT_OBSERVATION_SIZE

# The distilled expert intentionally responds only to phase, board attitude/contact,
# forward intent, and segment attitude. Position and velocity channels are present in
# the shared 174-value contract, but this teacher never consumes them. Keeping this
# mask explicit prevents untrained first-layer weights from turning those channels
# into a high-gain feedback loop during browser rollouts.
TEACHER_FEATURE_INDICES = (
    0,  # cycle time
    8,  # board roll
    9,  # board yaw
    10,  # contact ratio
    11,  # target x
) + tuple(
    OBSERVATION_HEADER_SIZE + segment * SEGMENT_OBSERVATION_SIZE + field
    for segment in range(SEGMENT_COUNT)
    for field in (6, 7)  # segment pitch and yaw
)

TEACHER_FEATURE_INDEX_SET = frozenset(TEACHER_FEATURE_INDICES)
IGNORED_OBSERVATION_INDICES = tuple(
    index for index in range(OBSERVATION_SIZE) if index not in TEACHER_FEATURE_INDEX_SET
)
