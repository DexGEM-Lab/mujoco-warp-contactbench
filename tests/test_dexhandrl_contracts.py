import dataclasses
import json

import pytest

from sim.dexhandrl.contracts import (
    DexHandMaterialConfig,
    DexHandPhysicsConfig,
    DexHandServoConfig,
    format_friction,
    legacy_physics_config,
    parse_friction,
    resolve_physics_config,
)


def test_legacy_port_v1_defaults_and_json_record() -> None:
    physics = DexHandPhysicsConfig()
    assert physics.profile_name == "legacy_port_v1"
    assert physics.control_dt == 0.01 and physics.substeps == 10 and physics.physics_dt == 0.001
    assert (physics.integrator, physics.solver, physics.iterations, physics.gravity) == (
        "Euler", "Newton", 80, (0.0, 0.0, -9.81)
    )
    assert (physics.model_njmax, physics.model_nconmax) == (2048, 8192)
    assert (physics.runtime_naconmax, physics.runtime_njmax) == (8192, 8192)
    assert physics.servo == DexHandServoConfig()
    assert physics.material == DexHandMaterialConfig()
    assert physics.hand_self_collision_mode == "disable_within_finger"
    assert json.loads(json.dumps(physics.record()))["provenance"]["object_asset"] == "asset_authoritative"
    with pytest.raises(dataclasses.FrozenInstanceError):
        physics.substeps = 2  # type: ignore[misc]


@pytest.mark.parametrize("factory", [
    lambda: DexHandPhysicsConfig(control_dt=0),
    lambda: DexHandPhysicsConfig(substeps=False),
    lambda: DexHandPhysicsConfig(integrator="RK4"),
    lambda: DexHandServoConfig(finger_kp=float("nan")),
    lambda: DexHandMaterialConfig(object_friction=(1, -1, 0)),
])
def test_invalid_contract_values_fail_closed(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_friction_parser_formatter_and_legacy_resolver() -> None:
    assert parse_friction("3 2 2") == (3.0, 2.0, 2.0)
    assert format_friction((0.5, 0.01, 0.001)) == "0.5 0.01 0.001"
    assert legacy_physics_config(dt=0.02, finger_kp=17, hand_friction="3 2 2") == DexHandPhysicsConfig(
        control_dt=0.02,
        servo=DexHandServoConfig(finger_kp=17),
        material=DexHandMaterialConfig(hand_friction=(3, 2, 2)),
    )
    with pytest.raises(ValueError):
        parse_friction("1 2")


def test_physics_resolver_rejects_mixed_sources_and_maps_runtime_aliases() -> None:
    custom = DexHandPhysicsConfig(runtime_naconmax=64, runtime_njmax=128)
    assert resolve_physics_config(custom) is custom
    assert resolve_physics_config(finger_kp=17, naconmax=64, njmax=128) == DexHandPhysicsConfig(
        servo=DexHandServoConfig(finger_kp=17), runtime_naconmax=64, runtime_njmax=128
    )
    with pytest.raises(ValueError, match="legacy physical"):
        resolve_physics_config(custom, finger_kp=17)
