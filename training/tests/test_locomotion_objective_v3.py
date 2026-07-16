from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import wurmkickflip_rl.evolve_locomotion_policy as locomotion_policy
from wurmkickflip_rl.articulated_locomotion import (
    MAXIMUM_SPEED,
    POST_INTEGRATION_RELAXATION_PASSES,
    controller_joints_to_scene_edges,
    limit_velocity,
)

from wurmkickflip_rl.evolve_locomotion_policy import (
    ABLATION_FULL_PROGRESS_FLOOR,
    ARTIFACT_GENOME_DECIMALS,
    BOUNDED_COMBINED_CONTROL_PROGRESS_FLOOR,
    DEFAULT_ARTIFACT,
    DEFAULT_SUMMARY,
    DEFAULT_V3_ARTIFACT,
    DEFAULT_V3_SUMMARY,
    HOST_DIAGNOSTIC_DECIMALS,
    NOMINAL_SIGNED_HEAD_FLOOR,
    OBJECTIVE_V2_VERSION,
    OBJECTIVE_V3_VERSION,
    PUBLISHED_V3_MODEL_VERSION,
    PUBLISHED_V3_WARM_START_SHA256,
    RUNTIME_SENSOR_CLAMPS,
    SCENARIOS,
    SENSOR_NAMES,
    V3_COMBINED_RECOVERY_SCENARIOS,
    V3_COMBINED_RECOVERY_STEPS,
    V3_GATE_DIAGNOSTIC_DECIMALS,
    V3_RECIPE_SOURCE_PATHS,
    V3_SELECTION_FITNESS_DECIMALS,
    ablation_gate_results,
    bound_genome,
    canonical_genome_recipe_hash,
    canonical_recipe_source_sha256,
    canonical_source_sha256,
    canonical_v3_ablations,
    canonical_v3_diagnostic,
    canonical_v3_selection_fitness,
    combined_recovery_rollout_settings,
    combined_control_progress_floor,
    causal_ordering_feasibility_score,
    deterministic_trace_hash_v3,
    evaluate_ablations,
    evaluate_genome,
    evaluate_genome_v3,
    evaluate_local_causal_response,
    evaluate_population,
    evaluate_population_v3_rollout,
    evaluate_selection_fitness_v3,
    finalize_v3_artifact_evaluation,
    head_leading_objective_component,
    load_warm_start,
    make_artifact_v3,
    make_summary_v3,
    metrics_v3_json,
    perturbation_schedule,
    perturbation_recovery_json,
    runtime_controller_sensors,
    selection_gate_results,
    v3_selection_order,
    validate_model_version,
    v3_near_feasible_children,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_MODEL = ROOT / "public/models/wurmkickflip_locomotion_policy.json"
RETAINED_V3_SEED = (
    ROOT
    / "training/seeds/wurmkickflip_locomotion_terrarium_causal_candidate_warm_start_v3.json"
)


def published_genome() -> np.ndarray:
    genome, _metadata = load_warm_start(PUBLISHED_MODEL)
    assert genome is not None
    return genome


def retained_seed_genome() -> np.ndarray:
    genome, _metadata = load_warm_start(RETAINED_V3_SEED)
    assert genome is not None
    return genome


def test_json_artifact_writes_are_exact_utf8_lf_bytes(tmp_path: Path) -> None:
    compact_path = tmp_path / "compact.json"
    pretty_path = tmp_path / "pretty.json"
    payload = {"label": "wurm", "values": [1.0, 2.5]}

    write_json(compact_path, payload, compact=True)
    write_json(pretty_path, payload, compact=False)

    assert compact_path.read_bytes() == b'{"label":"wurm","values":[1.0,2.5]}\n'
    assert b"\r\n" not in pretty_path.read_bytes()
    assert pretty_path.read_bytes().endswith(b"\n")


def test_controller_joint_to_scene_edge_mapping_matches_typescript() -> None:
    controller_joints = np.arange(16, dtype=np.float64)[None, None, :]
    scene_edges = controller_joints_to_scene_edges(controller_joints)

    assert scene_edges.shape == (1, 1, 15)
    assert np.array_equal(scene_edges[0, 0], np.arange(15, 0, -1, dtype=np.float64))
    assert 0.0 not in scene_edges


def test_post_integration_relaxation_count_comes_from_contract() -> None:
    assert POST_INTEGRATION_RELAXATION_PASSES == 3


def test_controller_segment_positions_are_exact_artifact_decimals() -> None:
    expected = np.array(
        [
            -1.0,
            -0.86666667,
            -0.73333333,
            -0.6,
            -0.46666667,
            -0.33333333,
            -0.2,
            -0.06666667,
            0.06666667,
            0.2,
            0.33333333,
            0.46666667,
            0.6,
            0.73333333,
            0.86666667,
            1.0,
        ],
        dtype=np.float64,
    )
    analytic = np.linspace(-1.0, 1.0, expected.size, dtype=np.float64)

    assert np.array_equal(locomotion_policy.SEGMENT_POSITIONS, expected)
    assert not np.array_equal(locomotion_policy.SEGMENT_POSITIONS, analytic)


def test_velocity_limit_matches_typescript_clipped_mean_semantics() -> None:
    velocities = np.array(
        [[[[4.2, 0.4], [3.9, 3.3], [4.4, -2.1], [3.7, 1.2]]]],
        dtype=np.float64,
    )
    expected = velocities.copy()
    mean = np.mean(expected, axis=2, keepdims=True)
    mean_speed = np.linalg.norm(mean, axis=3, keepdims=True)
    limited_mean = mean * np.minimum(1.0, MAXIMUM_SPEED / np.maximum(mean_speed, 1.0e-9))
    relative = expected - limited_mean
    relative_speed = np.linalg.norm(relative, axis=3, keepdims=True)
    relative *= np.minimum(
        1.0,
        MAXIMUM_SPEED / np.maximum(relative_speed, 1.0e-9),
    )
    relative -= np.mean(relative, axis=2, keepdims=True)
    expected = limited_mean + relative

    limit_velocity(velocities)

    assert np.allclose(velocities, expected, atol=1e-14, rtol=0.0)


def test_all_rollout_paths_start_with_browser_support_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    genome = retained_seed_genome()
    first_contact: list[np.ndarray] = []
    original_step = locomotion_policy.segmental_network_step

    def capture_first_step(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        if not first_contact:
            first_contact.append(np.asarray(args[11], dtype=np.float64).copy())
        return original_step(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(locomotion_policy, "segmental_network_step", capture_first_step)
    low_friction = (SCENARIOS[7], SCENARIOS[4])

    evaluate_population_v3_rollout(genome[None, :], 60, low_friction)
    assert np.array_equal(first_contact[0], np.ones_like(first_contact[0]))

    first_contact.clear()
    evaluate_population(genome[None, :], 60, low_friction)
    assert np.array_equal(first_contact[0], np.ones_like(first_contact[0]))

    first_contact.clear()
    evaluate_genome(
        genome,
        60,
        low_friction,
        runtime_sensor_contract=True,
    )
    assert np.array_equal(first_contact[0], np.ones_like(first_contact[0]))


def test_v3_artifact_uses_exact_genome_and_keeps_host_magnitudes_summary_only() -> None:
    full_precision = published_genome() + 4.9e-9
    serialized, metrics, ablations = finalize_v3_artifact_evaluation(
        full_precision,
        60,
        combined_recovery_mode="recentered-900",
    )
    model_version = "metadata-round-trip-test"
    genome_recipe_hash = canonical_genome_recipe_hash(
        serialized,
        model_version=model_version,
        seed=7,
        generations=1,
        population_size=4,
        elite_count=1,
        episode_steps=60,
        warm_start_metadata=None,
        combined_recovery_mode="recentered-900",
    )
    artifact = make_artifact_v3(
        serialized,
        model_version=model_version,
        seed=7,
        generations=1,
        population_size=4,
        elite_count=1,
        episode_steps=60,
        metrics=metrics,
        ablations=ablations,
        genome_recipe_hash=genome_recipe_hash,
        warm_start_metadata=None,
        combined_recovery_mode="recentered-900",
    )
    summary = make_summary_v3(
        serialized,
        model_version=model_version,
        seed=7,
        generations=1,
        population_size=4,
        elite_count=1,
        episode_steps=60,
        metrics=metrics,
        ablations=ablations,
        genome_recipe_hash=genome_recipe_hash,
        warm_start_metadata=None,
        history=[],
        combined_recovery_mode="recentered-900",
    )
    with pytest.raises(SystemExit, match="exact eight-decimal"):
        make_summary_v3(
            full_precision,
            model_version=model_version,
            seed=7,
            generations=1,
            population_size=4,
            elite_count=1,
            episode_steps=60,
            metrics=metrics,
            ablations=ablations,
            genome_recipe_hash=genome_recipe_hash,
            warm_start_metadata=None,
            history=[],
            combined_recovery_mode="recentered-900",
        )
    reloaded_artifact = json.loads(json.dumps(artifact))
    reloaded = np.asarray(
        reloaded_artifact["initialState"]
        + reloaded_artifact["weights"]["input"]
        + reloaded_artifact["weights"]["recurrent"]
        + reloaded_artifact["weights"]["output"],
        dtype=np.float64,
    )

    assert not np.array_equal(full_precision, serialized)
    assert np.array_equal(serialized, reloaded)

    full_precision_metrics = evaluate_genome_v3(
        full_precision,
        60,
        combined_recovery_mode="recentered-900",
    )
    reloaded_metrics = evaluate_genome_v3(
        reloaded,
        60,
        combined_recovery_mode="recentered-900",
    )
    assert metrics == reloaded_metrics
    assert full_precision_metrics != reloaded_metrics
    # Host diagnostics are allowed to differ; they do not participate in the
    # policy artifact bytes.

    training = reloaded_artifact["training"]
    host_metric_keys = set(metrics_v3_json(reloaded_metrics))
    assert host_metric_keys.isdisjoint(training)
    assert {
        "perturbationRecovery",
        "ablations",
        "deterministicTraceHash",
        "hostDiagnostics",
        "platform",
    }.isdisjoint(training)
    assert summary["hostDiagnostics"] == {
        "platform": {
            "system": locomotion_policy.platform.system(),
            "python": locomotion_policy.platform.python_version(),
            "numpy": np.__version__,
        },
        "metrics": metrics_v3_json(reloaded_metrics),
        "perturbationRecovery": perturbation_recovery_json(reloaded_metrics),
        "ablations": canonical_v3_ablations(ablations),
        "deterministicTraceHash": deterministic_trace_hash_v3(reloaded_metrics),
    }
    assert training["controllerSensorClamps"] == {
        name: list(bounds) for name, bounds in RUNTIME_SENSOR_CLAMPS.items()
    }
    assert training["genomeDecimals"] == ARTIFACT_GENOME_DECIMALS
    assert training["selectionFitnessDecimals"] == V3_SELECTION_FITNESS_DECIMALS
    assert training["gateDiagnosticDecimals"] == V3_GATE_DIAGNOSTIC_DECIMALS
    assert HOST_DIAGNOSTIC_DECIMALS >= V3_GATE_DIAGNOSTIC_DECIMALS
    assert training["recipeSourceSha256"] == canonical_recipe_source_sha256()
    assert set(training["recipeSourceSha256"]) == set(V3_RECIPE_SOURCE_PATHS)
    assert training["canonicalGenomeRecipeHash"] == genome_recipe_hash
    assert training["pairedHostReproduction"] == {
        "requiredForPublication": True,
        "compared": ["serializedGenome", "artifactBytes"],
        "trainerAttestation": "none",
    }
    for result_name in (
        "selectionGateResults",
        "selectionGateGuardBandResults",
        "ablationGateResults",
        "ablationGateGuardBandResults",
    ):
        assert training[result_name]
        assert all(type(value) is bool for value in training[result_name].values())
    for aggregate_name in (
        "allSelectionGatesPassed",
        "allSelectionMarginsFeasible",
        "allSelectionGuardBandsPassed",
        "allAblationGatesPassed",
        "allAblationGuardBandsPassed",
        "allGateGuardBandsPassed",
        "allPublicationGatesPassed",
    ):
        assert type(training[aggregate_name]) is bool
    assert artifact["modelVersion"] == summary["modelVersion"] == model_version
    assert summary["serializedGenome"] == serialized.tolist()


def test_v3_network_receives_browser_sanitized_inputs_without_mutating_plant_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_fixture = (
        np.array([[-1.4, np.nan]]),
        np.array([[1.4, np.inf]]),
        np.array([[-0.1, 2.0]]),
        np.array([[-2.4, 2.4]]),
        np.array([[-4.0, 4.0]]),
        np.array([[-0.2, 1.5]]),
        np.array([[-0.2, 1.2]]),
        np.array([[np.nan, -0.2]]),
        np.array([[-0.4, 3.73455]]),
        np.array([[-1.5, 1.5]]),
        np.array([[np.nan, -1.5]]),
    )
    raw_snapshot = tuple(values.copy() for values in raw_fixture)
    sanitized = runtime_controller_sensors(*raw_fixture)
    expected = (
        np.array([[-1.0, 0.0]]),
        np.array([[1.0, 0.0]]),
        np.array([[0.0, 1.5]]),
        np.array([[-2.0, 2.0]]),
        np.array([[-3.0, 3.0]]),
        np.array([[0.0, 1.2]]),
        np.array([[0.0, 1.0]]),
        np.array([[1.0, 0.0]]),
        np.array([[0.0, 2.0]]),
        np.array([[-1.0, 1.0]]),
        np.array([[0.0, -1.0]]),
    )
    assert all(np.array_equal(actual, wanted) for actual, wanted in zip(sanitized, expected, strict=True))
    assert all(
        np.array_equal(actual, snapshot, equal_nan=True)
        for actual, snapshot in zip(raw_fixture, raw_snapshot, strict=True)
    )

    received_maxima: dict[str, list[float]] = {name: [] for name in RUNTIME_SENSOR_CLAMPS}
    original_step = locomotion_policy.segmental_network_step

    def capture_network_step(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        for index, (name, (minimum, maximum)) in enumerate(RUNTIME_SENSOR_CLAMPS.items()):
            sensor = np.asarray(args[4 + index], dtype=np.float64)
            assert np.all(np.isfinite(sensor))
            assert np.all(sensor >= minimum)
            assert np.all(sensor <= maximum)
            received_maxima[name].append(float(np.max(sensor)))
        return original_step(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(locomotion_policy, "segmental_network_step", capture_network_step)
    evaluate_population_v3_rollout(
        published_genome()[None, :],
        120,
        SCENARIOS,
    )

    assert max(received_maxima["slipSpeed"]) == 2.0


def test_v3_has_explicit_version_and_no_controller_clock() -> None:
    assert OBJECTIVE_V2_VERSION == "articulated-contact-obstacle-recovery-v2"
    assert OBJECTIVE_V3_VERSION == "articulated-head-leading-transient-recovery-v3"
    assert all("time" not in name.lower() and "phase" not in name.lower() for name in SENSOR_NAMES)
    public_root = (ROOT / "public").resolve()
    for default_path in (
        DEFAULT_ARTIFACT,
        DEFAULT_SUMMARY,
        DEFAULT_V3_ARTIFACT,
        DEFAULT_V3_SUMMARY,
    ):
        resolved = default_path.resolve()
        assert public_root not in resolved.parents
        assert (ROOT / "training" / "runs").resolve() in resolved.parents


def test_perturbation_windows_are_deterministic_and_turn_aware() -> None:
    slick = perturbation_schedule(480, "slick-window")
    shove = perturbation_schedule(480, "body-shove")
    combined = perturbation_schedule(480, "combined-recovery")
    diagnostic = perturbation_schedule(V3_COMBINED_RECOVERY_STEPS, "combined-recovery")

    assert slick == perturbation_schedule(480, "slick-window")
    assert slick.pre_start < slick.perturb_start < slick.perturb_end <= 480 // 2
    assert 480 // 2 <= shove.pre_start < shove.perturb_start < shove.post_end
    assert V3_COMBINED_RECOVERY_STEPS == 900
    assert (combined.perturb_start, combined.perturb_end) == (128, 176)
    assert (diagnostic.perturb_start, diagnostic.perturb_end) == (240, 330)
    assert combined_recovery_rollout_settings(480, "bounded-480") == (480, False)
    assert combined_recovery_rollout_settings(480, "recentered-900") == (900, True)
    assert combined_control_progress_floor(480) == BOUNDED_COMBINED_CONTROL_PROGRESS_FLOOR
    assert combined_control_progress_floor(900) == 8.5

    causal_scores = causal_ordering_feasibility_score(np.array([0.49, 0.5, 2.99, -0.332]))
    assert causal_scores[1] > causal_scores[0] + 20.0
    assert causal_scores[2] > causal_scores[3] + 100.0

    boundary_diagnostics = canonical_v3_diagnostic(np.array([0.499999984, 0.500000016]))
    assert np.array_equal(boundary_diagnostics, np.array([0.49999998, 0.50000002]))

    raw_fitness = np.array([1.00004, 1.000049, 1.00014])
    canonical_fitness = canonical_v3_selection_fitness(raw_fitness)
    assert np.array_equal(canonical_fitness, np.array([1.0, 1.0, 1.0001]))
    assert np.array_equal(np.argsort(-canonical_fitness, kind="stable"), np.array([2, 0, 1]))

    tier_fitness, tier_order = v3_selection_order(
        np.array([100.0, 1.0, 50.0, 1.00004]),
        np.array([False, True, True, True]),
        np.array([False, False, True, True]),
        np.array([14, 17, 17, 17]),
        np.array([14, 16, 17, 17]),
    )
    assert np.array_equal(tier_fitness, np.array([100.0, 1.0, 50.0, 1.0]))
    assert np.array_equal(tier_order, np.array([2, 3, 1, 0]))


def test_population_and_artifact_gate_quantization_have_boundary_parity() -> None:
    metrics = evaluate_genome_v3(published_genome(), 60)
    for raw_value in (NOMINAL_SIGNED_HEAD_FLOOR - 1.6e-8, NOMINAL_SIGNED_HEAD_FLOOR - 0.4e-8):
        boundary_metrics = replace(metrics, signed_head_leading_fraction=raw_value)
        canonical = float(canonical_v3_diagnostic(np.array([raw_value]))[0])
        assert selection_gate_results(boundary_metrics, 60)["nominalSignedHeadLeading"] == (
            canonical >= NOMINAL_SIGNED_HEAD_FLOOR
        )


def test_ablation_gates_use_raw_values_and_bad_artifacts_are_refused() -> None:
    just_below = float(np.nextafter(ABLATION_FULL_PROGRESS_FLOOR, -np.inf))
    raw_boundary = {
        "fullProgress": just_below,
        "zeroProgress": 0.0,
        "frozenProgress": 0.0,
        "shuffledProgress": 0.0,
        "noFrictionDisplacement": 0.0,
    }
    assert round(just_below, HOST_DIAGNOSTIC_DECIMALS) == ABLATION_FULL_PROGRESS_FLOOR
    assert not ablation_gate_results(raw_boundary)["fullProgress"]

    serialized, metrics, ablations = finalize_v3_artifact_evaluation(
        published_genome(),
        60,
    )
    bad_ablations = {**ablations, "fullProgress": 0.0}
    model_version = "refusal-test"
    recipe_hash = canonical_genome_recipe_hash(
        serialized,
        model_version=model_version,
        seed=7,
        generations=1,
        population_size=4,
        elite_count=1,
        episode_steps=60,
        warm_start_metadata=None,
        combined_recovery_mode="bounded-480",
    )
    with pytest.raises(SystemExit, match="refusing to emit"):
        make_artifact_v3(
            serialized,
            model_version=model_version,
            seed=7,
            generations=1,
            population_size=4,
            elite_count=1,
            episode_steps=60,
            metrics=metrics,
            ablations=bad_ablations,
            genome_recipe_hash=recipe_hash,
            warm_start_metadata=None,
            combined_recovery_mode="bounded-480",
        )


def test_recipe_identity_covers_locked_inputs_and_model_version() -> None:
    genome = retained_seed_genome()
    assert canonical_source_sha256(RETAINED_V3_SEED) == PUBLISHED_V3_WARM_START_SHA256
    source_hashes = canonical_recipe_source_sha256()
    assert tuple(source_hashes) == V3_RECIPE_SOURCE_PATHS
    assert all(len(value) == 64 for value in source_hashes.values())

    base_kwargs = {
        "seed": 7,
        "generations": 1,
        "population_size": 4,
        "elite_count": 1,
        "episode_steps": 60,
        "warm_start_metadata": None,
        "combined_recovery_mode": "recentered-900",
    }
    first_hash = canonical_genome_recipe_hash(genome, model_version="model-a", **base_kwargs)
    second_hash = canonical_genome_recipe_hash(genome, model_version="model-b", **base_kwargs)
    assert first_hash != second_hash
    assert validate_model_version("model-a") == "model-a"
    with pytest.raises(SystemExit, match="whitespace"):
        validate_model_version(" model-a")

    metrics = evaluate_genome_v3(genome, 60)
    ablations = evaluate_ablations(genome, 60, runtime_sensor_contract=True, raw=True)
    with pytest.raises(SystemExit, match="canonicalGenomeRecipeHash"):
        make_artifact_v3(
            genome,
            model_version="model-a",
            metrics=metrics,
            ablations=ablations,
            genome_recipe_hash=second_hash,
            **base_kwargs,
        )

    reserved_hash = canonical_genome_recipe_hash(
        genome,
        model_version=PUBLISHED_V3_MODEL_VERSION,
        **base_kwargs,
    )
    with pytest.raises(SystemExit, match="reserved"):
        make_artifact_v3(
            genome,
            model_version=PUBLISHED_V3_MODEL_VERSION,
            metrics=metrics,
            ablations=ablations,
            genome_recipe_hash=reserved_hash,
            **base_kwargs,
        )

    published_kwargs = {
        "seed": 20260737,
        "generations": 1,
        "population_size": 4,
        "elite_count": 1,
        "episode_steps": 480,
        "warm_start_metadata": {
            "sha256": PUBLISHED_V3_WARM_START_SHA256,
            "modelVersion": "retained-seed",
        },
        "combined_recovery_mode": "bounded-480",
    }
    wrong_model_hash = canonical_genome_recipe_hash(
        genome,
        model_version="wrong-published-name",
        **published_kwargs,
    )
    with pytest.raises(SystemExit, match=PUBLISHED_V3_MODEL_VERSION):
        make_artifact_v3(
            genome,
            model_version="wrong-published-name",
            metrics=metrics,
            ablations=ablations,
            genome_recipe_hash=wrong_model_hash,
            **published_kwargs,
        )


def test_head_leading_component_rejects_equal_speed_sideways_motion() -> None:
    head_first = head_leading_objective_component(
        np.array([0.82]),
        np.array([0.86]),
        np.array([0.28]),
        np.array([0.72]),
        np.array([0.48]),
        np.array([0.68]),
    )
    sideways = head_leading_objective_component(
        np.array([0.02]),
        np.array([0.37]),
        np.array([0.68]),
        np.array([0.05]),
        np.array([0.14]),
        np.array([0.0]),
    )

    assert float(head_first[0] - sideways[0]) > 15.0


def test_local_causal_probe_matches_browser_gate_fixture() -> None:
    response = evaluate_local_causal_response(retained_seed_genome()[None, :])

    assert np.isclose(response.initial_local_delta[0], 1.08458877, atol=5e-8)
    assert np.isclose(response.radius_two_delta[0], 0.72848171, atol=5e-8)
    assert np.isclose(response.radius_three_delta[0], 0.05616283, atol=5e-8)
    assert response.settled_delta[0] == 0.0


def test_v3_selection_is_batch_independent_and_deterministic() -> None:
    published = published_genome()
    mutation = np.linspace(-0.015, 0.015, published.size, dtype=np.float64)
    population = np.stack((published, bound_genome(published + mutation)))

    batch_fitness, batch_gap, batch_diagnostics = evaluate_selection_fitness_v3(population, 60)
    repeated_fitness, repeated_gap, repeated_diagnostics = evaluate_selection_fitness_v3(population, 60)
    assert np.array_equal(batch_fitness, repeated_fitness)
    assert np.array_equal(batch_gap, repeated_gap)
    assert np.array_equal(
        batch_diagnostics.local_causal_radius_two_delta,
        repeated_diagnostics.local_causal_radius_two_delta,
    )

    for index, genome in enumerate(population):
        single_fitness, single_gap, single_diagnostics = evaluate_selection_fitness_v3(
            genome[None, :],
            60,
        )
        assert np.isclose(batch_fitness[index], single_fitness[0], atol=1e-12)
        assert np.isclose(batch_gap[index], single_gap[0], atol=1e-12)
        assert np.isclose(
            batch_diagnostics.signed_head_leading_fraction[index],
            single_diagnostics.signed_head_leading_fraction[0],
            atol=1e-12,
        )


def test_v3_near_feasible_repair_population_is_bounded_and_deterministic() -> None:
    parent = retained_seed_genome()
    first = v3_near_feasible_children(
        parent,
        np.random.default_rng(71),
        generation=7,
        generations=10,
        maximum_children=40,
    )
    second = v3_near_feasible_children(
        parent,
        np.random.default_rng(71),
        generation=7,
        generations=10,
        maximum_children=40,
    )

    assert len(first) == len(second) == 40
    assert all(child.shape == parent.shape for child in first)
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    assert any(not np.array_equal(child, parent) for child in first)


def test_combined_proxy_is_paired_and_transient() -> None:
    population = published_genome()[None, :]
    control = evaluate_population_v3_rollout(
        population,
        120,
        V3_COMBINED_RECOVERY_SCENARIOS,
        perturbation="combined-recovery",
        apply_perturbation=False,
    )
    perturbed = evaluate_population_v3_rollout(
        population,
        120,
        V3_COMBINED_RECOVERY_SCENARIOS,
        perturbation="combined-recovery",
    )
    nominal = evaluate_population_v3_rollout(population, 120, SCENARIOS[:1])

    assert control.mean_progress[0] > 0.2
    assert np.isfinite(perturbed.mean_progress[0])
    assert not np.isclose(control.mean_progress[0], perturbed.mean_progress[0])
    assert nominal.hidden_saturation_ratio[0] > 0.0
    assert 0.0 < nominal.mean_hidden_tanh_derivative[0] < 1.0
    evaluate_population,
