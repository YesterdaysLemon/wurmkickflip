from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "locomotion-v2.json"
CONTRACT: dict[str, Any] = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

SEGMENT_COUNT = int(CONTRACT["segmentCount"])
TIMESTEP = float(CONTRACT["timestep"])
JOINT_STIFFNESS = float(CONTRACT["joint"]["stiffness"])
JOINT_DAMPING = float(CONTRACT["joint"]["damping"])
JOINT_LIMIT = float(CONTRACT["joint"]["limit"])
POST_INTEGRATION_RELAXATION_PASSES = int(
    CONTRACT["postIntegrationRelaxationPasses"]
)

DYNAMICS: dict[str, float] = {
    key: float(value) for key, value in CONTRACT["dynamics"].items()
}
SPACING = DYNAMICS["spacing"]
ACTUATOR_STIFFNESS = DYNAMICS["actuatorStiffness"]
ACTUATOR_DAMPING = DYNAMICS["actuatorDamping"]
SHAPE_BEND_SCALE = DYNAMICS["shapeBendScale"]
CONSTRAINT_ITERATIONS = int(DYNAMICS["constraintIterations"])
CONSTRAINT_STIFFNESS = 1.0 - DYNAMICS["constraintCompliance"]
LONGITUDINAL_FRICTION = DYNAMICS["longitudinalFriction"]
LATERAL_FRICTION = DYNAMICS["lateralFriction"]
FREE_VELOCITY_DRAG = DYNAMICS["freeVelocityDrag"]
MAXIMUM_SPEED = DYNAMICS["maximumSpeed"]
BASE_RADIUS = DYNAMICS["baseRadius"]

SEGMENT_TAPER = 0.74 + np.sin(
    np.linspace(0.0, np.pi, SEGMENT_COUNT, dtype=np.float64)
) * 0.28


def initial_body(
    candidate_count: int,
    scenario_count: int,
    body_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a centered tail-to-head chain and zero velocity batch."""
    positions = np.zeros(
        (candidate_count, scenario_count, SEGMENT_COUNT, 2), dtype=np.float64
    )
    axial = (
        np.arange(SEGMENT_COUNT, dtype=np.float64)
        - (SEGMENT_COUNT - 1) * 0.5
    )
    positions[:, :, :, 0] = axial[None, None, :] * SPACING * body_scale[None, :, None]
    return positions, np.zeros_like(positions)


def articulated_plant_step(
    command: np.ndarray,
    joint: np.ndarray,
    joint_velocity: np.ndarray,
    body_position: np.ndarray,
    body_velocity: np.ndarray,
    base_friction: np.ndarray,
    roughness: np.ndarray,
    body_scale: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_radius: np.ndarray,
    timestep: float = TIMESTEP,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Vectorized counterpart of the browser's mean-free articulated plant.

    Shape accelerations and constraints conserve center of mass. Only ground
    friction, obstacle projection, and arena boundaries can create net impulse.
    Returned local contact arrays use the controller's anterior-to-posterior order.
    """
    dt = min(max(float(timestep), 0.0), TIMESTEP)
    old_position = body_position.copy()
    old_root = np.mean(body_position, axis=2)
    old_heading = body_heading(body_position)

    joint_velocity += (
        (command - joint) * JOINT_STIFFNESS - joint_velocity * JOINT_DAMPING
    ) * dt
    joint[:] = np.clip(joint + joint_velocity * dt, -JOINT_LIMIT, JOINT_LIMIT)

    spacing = SPACING * body_scale[None, :, None]
    edge_bend = controller_joints_to_scene_edges(joint)
    edge_yaw = old_heading[:, :, None] + np.cumsum(
        edge_bend * SHAPE_BEND_SCALE, axis=2
    )
    edge = np.stack((np.cos(edge_yaw), np.sin(edge_yaw)), axis=3) * spacing[:, :, :, None]
    local_shape = np.concatenate(
        (
            np.zeros((*edge.shape[:2], 1, 2), dtype=np.float64),
            np.cumsum(edge, axis=2),
        ),
        axis=2,
    )
    local_shape -= np.mean(local_shape, axis=2, keepdims=True)
    target = old_root[:, :, None, :] + local_shape
    mean_velocity = np.mean(body_velocity, axis=2, keepdims=True)
    acceleration = (
        (target - body_position) * ACTUATOR_STIFFNESS
        - (body_velocity - mean_velocity) * ACTUATOR_DAMPING
    )
    acceleration -= np.mean(acceleration, axis=2, keepdims=True)
    body_velocity += acceleration * dt

    tangent = body_tangent(body_position, old_heading)
    right = np.stack((-tangent[:, :, :, 1], tangent[:, :, :, 0]), axis=3)
    forward_velocity = np.sum(body_velocity * tangent, axis=3)
    lateral_velocity = np.sum(body_velocity * right, axis=3)

    spatial_friction = base_friction[:, :, None] * (
        1.0
        + roughness[:, :, None]
        * 0.24
        * np.sin(body_position[:, :, :, 0] * 1.71 + body_position[:, :, :, 1] * 1.19)
    )
    spatial_friction = np.clip(spatial_friction, 0.0, 1.2)
    scene_joint_speed = np.abs(joint_velocity[:, :, ::-1])
    contact_scene = np.clip(
        spatial_friction * np.clip(1.0 - scene_joint_speed / 3.25, 0.08, 1.0),
        0.0,
        1.0,
    )
    longitudinal_damping = 1.0 - np.exp(
        -LONGITUDINAL_FRICTION * spatial_friction * dt
    )
    lateral_damping = 1.0 - np.exp(-LATERAL_FRICTION * contact_scene * dt)
    damped_forward = forward_velocity * (1.0 - longitudinal_damping)
    damped_lateral = lateral_velocity * (1.0 - lateral_damping)
    body_velocity[:] = (
        tangent * damped_forward[:, :, :, None]
        + right * damped_lateral[:, :, :, None]
    )
    body_velocity *= np.exp(-FREE_VELOCITY_DRAG * dt)
    limit_velocity(body_velocity)
    body_position += body_velocity * dt
    satisfy_distance_constraints(body_position, spacing)

    obstacle_normal_scene = np.zeros_like(body_position)
    for relaxation_pass in range(POST_INTEGRATION_RELAXATION_PASSES + 1):
        if relaxation_pass > 0:
            satisfy_distance_constraints(body_position, spacing)
        projected_normal = project_obstacles(
            body_position,
            body_velocity,
            obstacle_center,
            obstacle_radius,
            body_scale,
        )
        projected_normal += project_bounds(body_position, body_velocity, body_scale)
        projected_contact = np.linalg.norm(projected_normal, axis=3) > 0.0
        obstacle_normal_scene[:] = np.where(
            projected_contact[:, :, :, None],
            projected_normal,
            obstacle_normal_scene,
        )

    body_velocity[:] = (body_position - old_position) / max(dt, 1.0e-9)
    limit_velocity(body_velocity)
    root = np.mean(body_position, axis=2)
    mean_velocity = np.mean(body_velocity, axis=2)
    heading = body_heading(body_position)
    forward = np.stack((np.cos(heading), np.sin(heading)), axis=2)
    speed = np.sum(mean_velocity * forward, axis=2)
    angular_speed = np.clip(wrap_angle(heading - old_heading) / max(dt, 1.0e-9), -3.0, 3.0)
    step_distance = np.linalg.norm(root - old_root, axis=2)

    obstacle_forward_scene = np.sum(obstacle_normal_scene * forward[:, :, None, :], axis=3)
    root_right = np.stack((-forward[:, :, 1], forward[:, :, 0]), axis=2)
    obstacle_right_scene = np.sum(obstacle_normal_scene * root_right[:, :, None, :], axis=3)
    return (
        step_distance,
        root,
        forward,
        speed,
        angular_speed,
        contact_scene[:, :, ::-1],
        np.abs(damped_lateral[:, :, ::-1]),
        obstacle_forward_scene[:, :, ::-1],
        obstacle_right_scene[:, :, ::-1],
    )


def controller_joints_to_scene_edges(joint: np.ndarray) -> np.ndarray:
    """Map anterior-to-posterior controller joints to tail-to-head scene edges.

    The TypeScript scene builds edge 1 from plant joint 15 through edge 15 from
    plant joint 1. Joint 0 is the anterior/head terminal actuator and has no edge.
    """
    return joint[..., :0:-1]


def satisfy_distance_constraints(body_position: np.ndarray, spacing: np.ndarray) -> None:
    for _iteration in range(CONSTRAINT_ITERATIONS):
        delta = body_position[:, :, 1:, :] - body_position[:, :, :-1, :]
        distance = np.maximum(np.linalg.norm(delta, axis=3), 1.0e-9)
        correction_scale = (
            (distance - spacing) / distance * 0.5 * CONSTRAINT_STIFFNESS
        )
        correction = delta * correction_scale[:, :, :, None]
        accumulated = np.zeros_like(body_position)
        accumulated[:, :, :-1, :] += correction
        accumulated[:, :, 1:, :] -= correction
        body_position += accumulated


def project_obstacles(
    body_position: np.ndarray,
    body_velocity: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_radius: np.ndarray,
    body_scale: np.ndarray,
) -> np.ndarray:
    relative = body_position - obstacle_center[None, :, None, :]
    distance = np.linalg.norm(relative, axis=3)
    safe_distance = np.maximum(distance, 1.0e-9)
    normal = relative / safe_distance[:, :, :, None]
    radii = (
        obstacle_radius[None, :, None]
        + BASE_RADIUS * SEGMENT_TAPER[None, None, :] * body_scale[None, :, None]
    )
    active = (obstacle_radius[None, :, None] > 0.0) & (distance < radii)
    penetration = np.where(active, radii - distance + 1.0e-4, 0.0)
    body_position += normal * penetration[:, :, :, None]
    normal_velocity = np.sum(body_velocity * normal, axis=3)
    blocked = np.where(active, np.minimum(normal_velocity, 0.0), 0.0)
    body_velocity -= normal * blocked[:, :, :, None]
    return normal * active[:, :, :, None]


def project_bounds(
    body_position: np.ndarray,
    body_velocity: np.ndarray,
    body_scale: np.ndarray,
) -> np.ndarray:
    half_extent = 4.68
    radii = BASE_RADIUS * SEGMENT_TAPER[None, None, :] * body_scale[None, :, None]
    normals = np.zeros_like(body_position)
    minimum = -half_extent + radii
    maximum = half_extent - radii
    for axis in range(2):
        below = body_position[:, :, :, axis] < minimum
        above = body_position[:, :, :, axis] > maximum
        body_position[:, :, :, axis] = np.clip(
            body_position[:, :, :, axis], minimum, maximum
        )
        normals[:, :, :, axis] = below.astype(np.float64) - above.astype(np.float64)
        outward_velocity = body_velocity[:, :, :, axis] * normals[:, :, :, axis]
        body_velocity[:, :, :, axis] -= np.minimum(outward_velocity, 0.0) * normals[:, :, :, axis]
    return normals


def body_heading(body_position: np.ndarray) -> np.ndarray:
    axis = body_position[:, :, -1, :] - body_position[:, :, 0, :]
    return np.arctan2(axis[:, :, 1], axis[:, :, 0])


def body_tangent(body_position: np.ndarray, fallback_heading: np.ndarray) -> np.ndarray:
    before = np.concatenate((body_position[:, :, :1, :], body_position[:, :, :-1, :]), axis=2)
    after = np.concatenate((body_position[:, :, 1:, :], body_position[:, :, -1:, :]), axis=2)
    delta = after - before
    length = np.linalg.norm(delta, axis=3)
    fallback = np.stack((np.cos(fallback_heading), np.sin(fallback_heading)), axis=2)
    safe = np.maximum(length, 1.0e-9)
    tangent = delta / safe[:, :, :, None]
    return np.where((length > 1.0e-9)[:, :, :, None], tangent, fallback[:, :, None, :])


def limit_velocity(body_velocity: np.ndarray) -> None:
    mean_velocity = np.mean(body_velocity, axis=2, keepdims=True)
    mean_speed = np.linalg.norm(mean_velocity, axis=3, keepdims=True)
    mean_scale = np.minimum(1.0, MAXIMUM_SPEED / np.maximum(mean_speed, 1.0e-9))
    limited_mean = mean_velocity * mean_scale
    # Match TypeScript exactly: relative velocities are measured around the
    # already-clipped mean, not the pre-limit mean.
    relative = body_velocity - limited_mean
    relative_speed = np.linalg.norm(relative, axis=3)
    relative_scale = np.minimum(
        1.0, MAXIMUM_SPEED / np.maximum(relative_speed, 1.0e-9)
    )
    relative *= relative_scale[:, :, :, None]
    relative -= np.mean(relative, axis=2, keepdims=True)
    body_velocity[:] = limited_mean + relative


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))
