from __future__ import annotations

SEGMENT_COUNT = 16
ACTION_SIZE = SEGMENT_COUNT * 2
OBSERVATION_SIZE = 174
POLICY_TIMESTEP = 1.0 / 60.0

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
