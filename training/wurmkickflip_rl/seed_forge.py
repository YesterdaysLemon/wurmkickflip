from __future__ import annotations

import copy
import json
import math
import sys
from collections.abc import Mapping
from typing import Any

DOMAIN_SAMPLE_SCHEMA_VERSION = 1
DOMAIN_SAMPLE_KIND = "wurmkickflip.domainSample"
DOMAIN_PARAMETER_KEYS = (
    "gravityScale",
    "frictionScale",
    "dragScale",
    "slopeDegrees",
    "roughness",
    "obstacleDensity",
    "actuatorStrength",
    "actuatorLatencyMs",
    "sensorNoise",
    "spawnYawDegrees",
    "skateboardSpawnX",
    "skateboardSpawnZ",
    "skateboardMass",
    "wheelFriction",
)
DOMAIN_CONSTRAINTS = {
    "gravityScale": (0.0, None, False),
    "frictionScale": (0.0, None, False),
    "dragScale": (0.0, None, False),
    "roughness": (0.0, None, False),
    "obstacleDensity": (0.0, 1.0, False),
    "actuatorStrength": (0.0, None, True),
    "actuatorLatencyMs": (0.0, None, False),
    "sensorNoise": (0.0, 1.0, False),
    "spawnYawDegrees": (-180.0, 180.0, False),
    "skateboardMass": (0.0, None, True),
    "wheelFriction": (0.0, None, False),
}
UINT32_MASK = 0xFFFFFFFF
UINT32_SIZE = 0x100000000
MAX_SAFE_INTEGER = 0x1FFFFFFFFFFFFF
DOMAIN_QUANTIZATION_SCALE = 1_000_000


def normalize_domain_seed(value: int | float) -> int:
    if not _is_finite_number(value):
        raise ValueError("domain seeds must be finite before uint32 normalization")
    truncated = math.trunc(value)
    if abs(truncated) > MAX_SAFE_INTEGER:
        raise ValueError("truncated domain seeds must be safe integers before uint32 normalization")
    return truncated & UINT32_MASK


def next_domain_seed(seed: int | float) -> int:
    return (normalize_domain_seed(seed) + 0x9E3779B9) & UINT32_MASK


def create_domain_locks(locked: bool = False) -> dict[str, bool]:
    return {key: locked for key in DOMAIN_PARAMETER_KEYS}


def sample_domain(
    environment: Mapping[str, Any],
    seed: int | float,
    *,
    locks: Mapping[str, bool] | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _assert_valid_domain_environment(environment)
    normalized_seed = normalize_domain_seed(seed)
    random = _mulberry32(normalized_seed)
    values: dict[str, float] = {}
    for key in DOMAIN_PARAMETER_KEYS:
        minimum, maximum = _domain_range_for(environment, key)
        previous_value = previous.get(key) if previous is not None else None
        sampled = _quantize_to_range(_lerp(minimum, maximum, next(random)), minimum, maximum)
        if locks is not None and locks.get(key, False) and _is_finite_number(previous_value):
            values[key] = _quantize_to_range(float(previous_value), minimum, maximum)
        else:
            values[key] = sampled
    return _domain_sample(normalized_seed, values)


def create_nominal_domain_sample(environment: Mapping[str, Any], seed: int | float | None = None) -> dict[str, Any]:
    _assert_valid_domain_environment(environment)
    return _make_domain_sample(
        environment,
        environment["seed"] if seed is None else seed,
        _nominal_domain_values(environment),
    )


def create_preset_domain_sample(
    environment: Mapping[str, Any], preset_id: str, seed: int | float | None = None
) -> dict[str, Any]:
    nominal = create_nominal_domain_sample(environment, seed)
    if preset_id == "nominal":
        return nominal
    if preset_id not in {"ice-rink", "moon-dirt", "cinderblock-deck"}:
        raise ValueError(f"unknown Seed Forge preset: {preset_id}")

    values = {key: float(nominal[key]) for key in DOMAIN_PARAMETER_KEYS}

    def at(key: str, amount: float) -> float:
        minimum, maximum = _domain_range_for(environment, key)
        return _lerp(minimum, maximum, _clamp(amount, 0.0, 1.0))

    if preset_id == "ice-rink":
        values.update(
            {
                "frictionScale": at("frictionScale", 0.0),
                "dragScale": at("dragScale", 0.15),
                "slopeDegrees": _clamp(0.0, *_domain_range_for(environment, "slopeDegrees")),
                "roughness": at("roughness", 0.0),
                "obstacleDensity": at("obstacleDensity", 0.0),
                "actuatorStrength": at("actuatorStrength", 0.42),
                "actuatorLatencyMs": at("actuatorLatencyMs", 0.0),
                "sensorNoise": at("sensorNoise", 0.0),
                "skateboardMass": at("skateboardMass", 0.18),
                "wheelFriction": at("wheelFriction", 0.0),
            }
        )
    elif preset_id == "moon-dirt":
        values.update(
            {
                "gravityScale": at("gravityScale", 0.0),
                "frictionScale": at("frictionScale", 0.45),
                "dragScale": at("dragScale", 0.0),
                "slopeDegrees": at("slopeDegrees", 0.58),
                "roughness": at("roughness", 0.88),
                "obstacleDensity": at("obstacleDensity", 0.52),
                "actuatorStrength": at("actuatorStrength", 0.68),
                "actuatorLatencyMs": at("actuatorLatencyMs", 0.16),
                "sensorNoise": at("sensorNoise", 0.28),
                "spawnYawDegrees": at("spawnYawDegrees", 0.32),
                "skateboardMass": at("skateboardMass", 0.0),
                "wheelFriction": at("wheelFriction", 0.55),
            }
        )
    else:
        values = {key: at(key, 1.0) for key in DOMAIN_PARAMETER_KEYS}

    return _make_domain_sample(
        environment,
        environment["seed"] if seed is None else seed,
        values,
    )


def materialize_domain_environment(environment: Mapping[str, Any], sample: Mapping[str, Any]) -> dict[str, Any]:
    errors = validate_domain_sample(sample, environment)
    if errors:
        raise ValueError(f"invalid domain sample: {' '.join(errors)}")
    materialized = copy.deepcopy(dict(environment))
    world = _mapping(materialized["world"])
    terrain = _mapping(materialized["terrain"])
    skateboard = _mapping(materialized["skateboard"])
    gravity = _number_list(world["gravity"])
    spawn = _number_list(skateboard["spawnPosition"])
    materialized["seed"] = int(sample["seed"])
    world["gravity"] = [_quantize(value * float(sample["gravityScale"])) for value in gravity]
    world["airDrag"] = _quantize(float(world["airDrag"]) * float(sample["dragScale"]))
    terrain["baseFriction"] = _quantize(float(terrain["baseFriction"]) * float(sample["frictionScale"]))
    terrain["slopeDegrees"] = float(sample["slopeDegrees"])
    terrain["roughness"] = float(sample["roughness"])
    terrain["obstacleDensity"] = float(sample["obstacleDensity"])
    skateboard["spawnPosition"] = [
        float(sample["skateboardSpawnX"]),
        spawn[1],
        float(sample["skateboardSpawnZ"]),
    ]
    skateboard["mass"] = float(sample["skateboardMass"])
    skateboard["wheelFriction"] = float(sample["wheelFriction"])
    return materialized


def validate_domain_sample(sample: Mapping[str, Any], environment: Mapping[str, Any]) -> list[str]:
    errors = validate_domain_environment(environment)
    if errors:
        return errors
    if not isinstance(sample, Mapping):
        return ["sample must be an object."]
    if sample.get("schemaVersion") != DOMAIN_SAMPLE_SCHEMA_VERSION:
        errors.append(f"schemaVersion must be {DOMAIN_SAMPLE_SCHEMA_VERSION}.")
    if sample.get("kind") != DOMAIN_SAMPLE_KIND:
        errors.append(f"kind must be {DOMAIN_SAMPLE_KIND}.")
    seed = sample.get("seed")
    if not _is_uint32_integer(seed):
        errors.append("seed must be an unsigned 32-bit integer.")
    for key in DOMAIN_PARAMETER_KEYS:
        value = sample.get(key)
        minimum, maximum = _domain_range_for(environment, key)
        if not _is_finite_number(value):
            errors.append(f"{key} must be finite.")
        else:
            numeric_value = float(value)
            if not _is_quantized_domain_number(numeric_value):
                errors.append(f"{key} must use at most 6 decimal places within the exact quantization range.")
            if numeric_value < minimum or numeric_value > maximum:
                errors.append(f"{key} must be between {minimum:g} and {maximum:g}.")
                continue
            constraint = DOMAIN_CONSTRAINTS.get(key)
            if constraint is None:
                continue
            constraint_minimum, constraint_maximum, exclusive_minimum = constraint
            if (exclusive_minimum and numeric_value <= constraint_minimum) or (
                not exclusive_minimum and numeric_value < constraint_minimum
            ):
                qualifier = "greater than" if exclusive_minimum else "at least"
                errors.append(f"{key} must be {qualifier} {constraint_minimum:g}.")
            if constraint_maximum is not None and numeric_value > constraint_maximum:
                errors.append(f"{key} must be at most {constraint_maximum:g}.")
    return errors


def validate_domain_environment(environment: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(environment, Mapping):
        return ["environment must be an object."]

    seed = environment.get("seed")
    if not _is_safe_integer(seed):
        errors.append("environment.seed must be a safe integer before uint32 normalization.")

    randomization_value = environment.get("randomization")
    if not isinstance(randomization_value, Mapping):
        errors.append("environment.randomization must be an object.")
        randomization: Mapping[str, Any] = {}
    else:
        randomization = randomization_value

    valid_ranges: dict[str, tuple[float, float]] = {}
    for key in DOMAIN_PARAMETER_KEYS:
        path = f"environment.randomization.{key}"
        value = randomization.get(key)
        if not isinstance(value, list) or len(value) != 2 or not all(_is_finite_number(item) for item in value):
            errors.append(f"{path} must be an array of 2 finite numbers.")
            continue
        minimum, maximum = float(value[0]), float(value[1])
        valid_ranges[key] = (minimum, maximum)
        for index, endpoint in enumerate(value):
            if not _is_quantized_domain_number(endpoint):
                errors.append(f"{path}[{index}] must use at most 6 decimal places within the exact quantization range.")
        if minimum > maximum:
            errors.append(f"{path} minimum must be less than or equal to maximum.")
        constraint = DOMAIN_CONSTRAINTS.get(key)
        if constraint is None:
            continue
        constraint_minimum, constraint_maximum, exclusive_minimum = constraint
        if (exclusive_minimum and minimum <= constraint_minimum) or (
            not exclusive_minimum and minimum < constraint_minimum
        ):
            qualifier = "greater than" if exclusive_minimum else "at least"
            errors.append(f"{path} minimum must be {qualifier} {constraint_minimum:g}.")
        if constraint_maximum is not None and maximum > constraint_maximum:
            errors.append(f"{path} maximum must be at most {constraint_maximum:g}.")

    nominal, nominal_errors = _collect_nominal_domain_values(environment)
    errors.extend(nominal_errors)
    for key in DOMAIN_PARAMETER_KEYS:
        value = nominal.get(key)
        domain_range = valid_ranges.get(key)
        if value is None or domain_range is None:
            continue
        if not _is_quantized_domain_number(value):
            errors.append(
                f"environment authored nominal value for {key} must use at most 6 decimal places "
                "within the exact quantization range."
            )
        minimum, maximum = domain_range
        if value < minimum or value > maximum:
            errors.append(f"environment.randomization.{key} must contain its authored nominal value {value:g}.")
    return errors


def _make_domain_sample(
    environment: Mapping[str, Any],
    seed: int | float,
    values: Mapping[str, float],
) -> dict[str, Any]:
    clamped = {}
    for key in DOMAIN_PARAMETER_KEYS:
        minimum, maximum = _domain_range_for(environment, key)
        clamped[key] = _quantize_to_range(float(values[key]), minimum, maximum)
    return _domain_sample(normalize_domain_seed(seed), clamped)


def _nominal_domain_values(environment: Mapping[str, Any]) -> dict[str, float]:
    values, errors = _collect_nominal_domain_values(environment)
    if errors:
        raise TypeError("invalid authored nominal values: " + " ".join(errors))
    return values


def _collect_nominal_domain_values(environment: Mapping[str, Any]) -> tuple[dict[str, float], list[str]]:
    values = {
        "gravityScale": 1.0,
        "frictionScale": 1.0,
        "dragScale": 1.0,
        "actuatorStrength": 1.0,
        "actuatorLatencyMs": 0.0,
        "sensorNoise": 0.0,
        "spawnYawDegrees": 0.0,
    }
    errors: list[str] = []

    terrain_value = environment.get("terrain")
    if not isinstance(terrain_value, Mapping):
        errors.append("environment.terrain must be an object.")
    else:
        for key in ("slopeDegrees", "roughness", "obstacleDensity"):
            authored_value = terrain_value.get(key)
            if not _is_finite_number(authored_value):
                errors.append(f"environment authored nominal value for {key} must be finite.")
            else:
                values[key] = float(authored_value)

    skateboard_value = environment.get("skateboard")
    if not isinstance(skateboard_value, Mapping):
        errors.append("environment.skateboard must be an object.")
    else:
        spawn_position = skateboard_value.get("spawnPosition")
        if (
            not isinstance(spawn_position, list)
            or len(spawn_position) != 3
            or not all(_is_finite_number(item) for item in spawn_position)
        ):
            errors.append("environment.skateboard.spawnPosition must be an array of 3 finite numbers.")
        else:
            values["skateboardSpawnX"] = float(spawn_position[0])
            values["skateboardSpawnZ"] = float(spawn_position[2])
        for field, key in (("mass", "skateboardMass"), ("wheelFriction", "wheelFriction")):
            authored_value = skateboard_value.get(field)
            if not _is_finite_number(authored_value):
                errors.append(f"environment authored nominal value for {key} must be finite.")
            else:
                values[key] = float(authored_value)

    return values, errors


def _domain_sample(seed: int, values: Mapping[str, float]) -> dict[str, Any]:
    return {
        "schemaVersion": DOMAIN_SAMPLE_SCHEMA_VERSION,
        "kind": DOMAIN_SAMPLE_KIND,
        "seed": seed,
        **values,
    }


def _domain_range_for(environment: Mapping[str, Any], key: str) -> tuple[float, float]:
    randomization = _mapping(environment["randomization"])
    values = _number_list(randomization[key])
    return values[0], values[1]


def _mulberry32(seed: int):
    state = seed & UINT32_MASK
    while True:
        state = (state + 0x6D2B79F5) & UINT32_MASK
        value = state
        value = _imul(value ^ (value >> 15), value | 1)
        value ^= (value + _imul(value ^ (value >> 7), value | 61)) & UINT32_MASK
        value &= UINT32_MASK
        yield ((value ^ (value >> 14)) & UINT32_MASK) / UINT32_SIZE


def _imul(left: int, right: int) -> int:
    return ((left & UINT32_MASK) * (right & UINT32_MASK)) & UINT32_MASK


def _quantize(value: float) -> float:
    return math.floor(value * DOMAIN_QUANTIZATION_SCALE + 0.5) / DOMAIN_QUANTIZATION_SCALE


def _lerp(minimum: float, maximum: float, amount: float) -> float:
    if amount <= 0.0:
        return minimum
    if amount >= 1.0:
        return maximum
    return minimum * (1.0 - amount) + maximum * amount


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _quantize_to_range(value: float, minimum: float, maximum: float) -> float:
    return _clamp(_quantize(value), minimum, maximum)


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_safe_integer(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, int):
        return abs(value) <= MAX_SAFE_INTEGER
    return math.isfinite(value) and value.is_integer() and abs(value) <= MAX_SAFE_INTEGER


def _is_uint32_integer(value: Any) -> bool:
    return _is_safe_integer(value) and 0 <= value <= UINT32_MASK


def _is_quantized_domain_number(value: Any) -> bool:
    if not _is_finite_number(value):
        return False
    numeric_value = float(value)
    scaled = numeric_value * DOMAIN_QUANTIZATION_SCALE
    return math.isfinite(scaled) and abs(scaled) <= MAX_SAFE_INTEGER and _quantize(numeric_value) == numeric_value


def _assert_valid_domain_environment(environment: Mapping[str, Any]) -> None:
    errors = validate_domain_environment(environment)
    if errors:
        raise ValueError(f"invalid domain environment: {' '.join(errors)}")


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("expected an object")
    return value


def _number_list(value: Any) -> list[float]:
    if not isinstance(value, list) or not all(_is_finite_number(item) for item in value):
        raise TypeError("expected a numeric array")
    return [float(item) for item in value]


def _main() -> None:
    payload = json.load(sys.stdin)
    environment = _mapping(payload["environment"])
    seeds = _number_list(payload["seeds"])
    presets = payload.get("presets", [])
    output = {
        "samples": [sample_domain(environment, seed) for seed in seeds],
        "presets": [
            create_preset_domain_sample(environment, str(preset_id), environment["seed"]) for preset_id in presets
        ],
    }
    json.dump(output, sys.stdout, separators=(",", ":"))


if __name__ == "__main__":
    _main()
