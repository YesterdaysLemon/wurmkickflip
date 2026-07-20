from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from wurmkickflip_rl.seed_forge import (
    DOMAIN_PARAMETER_KEYS,
    create_domain_locks,
    create_nominal_domain_sample,
    create_preset_domain_sample,
    materialize_domain_environment,
    next_domain_seed,
    normalize_domain_seed,
    sample_domain,
    validate_domain_environment,
    validate_domain_sample,
)

ENVIRONMENT_PATH = (
    Path(__file__).resolve().parents[2] / "public" / "configs" / "environments" / "adaptive-skate-terrarium.json"
)


@pytest.fixture
def environment() -> dict[str, object]:
    return json.loads(ENVIRONMENT_PATH.read_text(encoding="utf-8"))


def test_sampling_is_reproducible_and_stays_in_declared_ranges(environment: dict[str, object]) -> None:
    first = sample_domain(environment, 1337)
    second = sample_domain(environment, 1337)

    assert first == second
    assert first["seed"] == 1337
    randomization = environment["randomization"]
    assert isinstance(randomization, dict)
    for key in DOMAIN_PARAMETER_KEYS:
        minimum, maximum = randomization[key]
        assert minimum <= first[key] <= maximum


def test_locks_preserve_values_without_shifting_unlocked_channels(environment: dict[str, object]) -> None:
    previous = sample_domain(environment, 1337)
    seed = next_domain_seed(previous["seed"])
    locks = create_domain_locks()
    locks["gravityScale"] = True
    locks["sensorNoise"] = True
    locks["skateboardMass"] = True

    locked = sample_domain(environment, seed, locks=locks, previous=previous)
    unlocked = sample_domain(environment, seed)

    for key in DOMAIN_PARAMETER_KEYS:
        assert locked[key] == (previous[key] if locks[key] else unlocked[key])


def test_nominal_and_presets_materialize_without_mutating_source(environment: dict[str, object]) -> None:
    source_snapshot = json.dumps(environment, sort_keys=True)
    nominal = create_nominal_domain_sample(environment)
    materialized = materialize_domain_environment(environment, nominal)

    assert json.dumps(environment, sort_keys=True) == source_snapshot
    assert materialized is not environment
    assert materialized["world"] is not environment["world"]
    assert materialized["terrain"] is not environment["terrain"]
    assert materialized["skateboard"] is not environment["skateboard"]
    assert validate_domain_sample(nominal, environment) == []

    for preset_id in ("nominal", "ice-rink", "moon-dirt", "cinderblock-deck"):
        preset = create_preset_domain_sample(environment, preset_id)
        assert validate_domain_sample(preset, environment) == []


def test_unsafe_ranges_are_rejected_before_sampling(environment: dict[str, object]) -> None:
    randomization = environment["randomization"]
    assert isinstance(randomization, dict)
    randomization["actuatorStrength"] = [-2, -1]

    assert "environment.randomization.actuatorStrength minimum must be greater than 0." in validate_domain_environment(
        environment
    )
    with pytest.raises(ValueError, match="invalid domain environment"):
        sample_domain(environment, 7)


def test_non_finite_authored_nominal_is_rejected(environment: dict[str, object]) -> None:
    terrain = environment["terrain"]
    assert isinstance(terrain, dict)
    terrain["slopeDegrees"] = float("nan")

    assert "environment authored nominal value for slopeDegrees must be finite." in validate_domain_environment(
        environment
    )


def test_quantization_contract_is_closed_and_overflow_safe(environment: dict[str, object]) -> None:
    off_lattice = copy.deepcopy(environment)
    terrain = off_lattice["terrain"]
    randomization = off_lattice["randomization"]
    assert isinstance(terrain, dict)
    assert isinstance(randomization, dict)
    terrain["roughness"] = 0.12345649
    randomization["roughness"] = [0.12345649, 0.12345649]

    assert any("must use at most 6 decimal places" in error for error in validate_domain_environment(off_lattice))
    with pytest.raises(ValueError, match="invalid domain environment"):
        sample_domain(off_lattice, 7)

    overflow_prone = copy.deepcopy(environment)
    overflow_randomization = overflow_prone["randomization"]
    assert isinstance(overflow_randomization, dict)
    overflow_randomization["skateboardSpawnX"] = [
        -float("1.7976931348623157e308"),
        float("1.7976931348623157e308"),
    ]

    assert any("within the exact quantization range" in error for error in validate_domain_environment(overflow_prone))
    with pytest.raises(ValueError, match="invalid domain environment"):
        sample_domain(overflow_prone, 7)

    sample = sample_domain(environment, 7)
    sample["roughness"] = 0.12345649
    assert any("must use at most 6 decimal places" in error for error in validate_domain_sample(sample, environment))


def test_seed_normalization_matches_javascript_number_semantics(environment: dict[str, object]) -> None:
    assert normalize_domain_seed(-1) == 0xFFFFFFFF
    assert normalize_domain_seed(0x100000001) == 1
    assert normalize_domain_seed(1.9) == 1
    with pytest.raises(ValueError, match="safe integers"):
        normalize_domain_seed(0x20000000000000)
    with pytest.raises(ValueError, match="finite"):
        normalize_domain_seed(float("inf"))

    sample = sample_domain(environment, 1)
    sample["seed"] = 1.0
    assert validate_domain_sample(sample, environment) == []

    environment["seed"] = 0x20000000000000
    assert "environment.seed must be a safe integer before uint32 normalization." in validate_domain_environment(
        environment
    )


def test_environment_validation_is_total_and_rejects_numeric_strings(environment: dict[str, object]) -> None:
    assert validate_domain_environment({})

    missing_terrain = copy.deepcopy(environment)
    del missing_terrain["terrain"]
    assert "environment.terrain must be an object." in validate_domain_environment(missing_terrain)

    string_nominal = copy.deepcopy(environment)
    terrain = string_nominal["terrain"]
    assert isinstance(terrain, dict)
    terrain["slopeDegrees"] = "0"
    assert "environment authored nominal value for slopeDegrees must be finite." in validate_domain_environment(
        string_nominal
    )
