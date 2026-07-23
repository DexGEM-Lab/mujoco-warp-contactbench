"""Immutable, dependency-light DexHandRL MuJoCo physics contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

PROFILE_NAME = "legacy_port_v1"
PROVENANCE_LABELS = frozenset(
    {"asset_authoritative", "task_authoritative", "trajectory_authoritative", "legacy_port_v1"}
)


def _validate_provenance(provenance: dict[str, str]) -> dict[str, str]:
    invalid = {field: label for field, label in provenance.items() if label not in PROVENANCE_LABELS}
    if invalid:
        raise ValueError(f"unsupported DexHand provenance label(s): {invalid}")
    return provenance


def parse_friction(value: str | tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    """Parse the MuJoCo sliding/torsional/rolling friction triple."""
    if isinstance(value, str):
        values = tuple(float(part) for part in value.split())
    else:
        values = tuple(float(part) for part in value)
    if len(values) != 3 or not all(math.isfinite(part) and part >= 0 for part in values):
        raise ValueError(f"friction must contain three finite non-negative values, got {value!r}")
    return values  # type: ignore[return-value]


def format_friction(value: str | tuple[float, float, float] | list[float]) -> str:
    # Preserve frozen legacy spellings; custom values retain 17 significant digits.
    legacy = {0.0: "0", 0.001: "0.001", 0.01: "0.01", 0.5: "0.5", 3.0: "3", 2.0: "2"}
    return " ".join(legacy.get(part, f"{part:.17g}") for part in parse_friction(value))


def _finite_positive(name: str, value: float, *, allow_zero: bool = False) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or (
        float(value) < 0 if allow_zero else float(value) <= 0
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number")


def _positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class DexHandServoConfig:
    base_pos_kp: float = 5000
    base_pos_kv: float = 500
    base_pos_force: float = 500
    base_rot_kp: float = 10000
    base_rot_kv: float = 125
    base_rot_force: float = 100
    finger_kp: float = 15
    finger_kv: float = 0
    finger_force: float = 50

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _finite_positive(name, value, allow_zero=name.endswith("_kv"))


@dataclass(frozen=True)
class DexHandMaterialConfig:
    hand_friction: tuple[float, float, float] = (0.5, 0.01, 0.001)
    object_friction: tuple[float, float, float] = (0.5, 0.01, 0.001)
    ground_friction: tuple[float, float, float] = (0.5, 0.5, 0.0)
    object_solref: tuple[float, float] = (0.006, 1.0)
    object_solimp: tuple[float, float, float, float, float] = (0.95, 0.995, 0.0001, 0.5, 2.0)

    def __post_init__(self) -> None:
        for name in ("hand_friction", "object_friction", "ground_friction"):
            object.__setattr__(self, name, parse_friction(getattr(self, name)))
        for name, count in (("object_solref", 2), ("object_solimp", 5)):
            values = tuple(float(v) for v in getattr(self, name))
            if len(values) != count or not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError(f"{name} must contain {count} finite non-negative values")
            object.__setattr__(self, name, values)
        if self.object_solref[0] <= 0:
            raise ValueError("object_solref time constant must be a finite positive number")


@dataclass(frozen=True)
class DexHandPhysicsConfig:
    profile_name: str = PROFILE_NAME
    control_dt: float = 0.01
    substeps: int = 10
    integrator: str = "Euler"
    solver: str = "Newton"
    iterations: int = 80
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    model_njmax: int = 2048
    model_nconmax: int = 8192
    # Stage 2 runtime allocation limits.  They are intentionally not serialized
    # into, or validated against, the compiled MuJoCo model capacities.
    runtime_naconmax: int = 8192
    runtime_njmax: int = 8192
    servo: DexHandServoConfig = DexHandServoConfig()
    material: DexHandMaterialConfig = DexHandMaterialConfig()
    hand_self_collision_mode: str = "disable_within_finger"

    def __post_init__(self) -> None:
        if self.profile_name != PROFILE_NAME:
            raise ValueError(f"unsupported DexHand physics profile: {self.profile_name!r}")
        _finite_positive("control_dt", self.control_dt)
        _positive_int("substeps", self.substeps)
        _positive_int("iterations", self.iterations)
        for name in ("model_njmax", "model_nconmax", "runtime_naconmax", "runtime_njmax"):
            _positive_int(name, getattr(self, name))
        gravity = tuple(float(v) for v in self.gravity)
        if len(gravity) != 3 or not all(math.isfinite(v) for v in gravity):
            raise ValueError("gravity must contain three finite values")
        object.__setattr__(self, "gravity", gravity)
        if self.integrator != "Euler" or self.solver != "Newton":
            raise ValueError("legacy_port_v1 requires Euler integration and Newton solver")
        if self.hand_self_collision_mode not in {"none", "disable_within_finger", "all_disabled"}:
            raise ValueError("invalid hand self-collision mode")
        if not isinstance(self.servo, DexHandServoConfig) or not isinstance(self.material, DexHandMaterialConfig):
            raise TypeError("servo and material must be DexHand configuration records")

    @property
    def physics_dt(self) -> float:
        return self.control_dt / self.substeps

    def record(self) -> dict[str, Any]:
        """JSON-serializable contract and explicit parameter-provenance labels."""
        record = asdict(self)
        record["physics_dt"] = self.physics_dt
        record["provenance"] = _validate_provenance({
            "profile": "legacy_port_v1",
            "servo": "legacy_port_v1",
            "material": "legacy_port_v1",
            "object_asset": "asset_authoritative",
            "trajectory": "trajectory_authoritative",
            "runtime_capacity": "task_authoritative",
        })
        return record


# Compatibility names kept at API boundaries while production code consumes
# DexHandPhysicsConfig exclusively. Values are intentionally normalized here so
# scene, environment, and launcher cannot grow divergent resolution rules.
LEGACY_PHYSICS_FIELDS = frozenset({
    "dt", "control_dt", "substeps",
    "base_pos_kp", "base_pos_kv", "base_pos_force",
    "base_rot_kp", "base_rot_kv", "base_rot_force",
    "finger_kp", "finger_kv", "finger_force",
    "hand_friction", "object_friction", "ground_friction",
    "hand_self_collision_mode", "collision", "naconmax", "njmax",
})


def legacy_physics_config(**overrides: Any) -> DexHandPhysicsConfig:
    """Resolve historical scene/runtime keyword overrides into the immutable profile."""
    overrides = dict(overrides)
    base = DexHandPhysicsConfig()
    aliases = {
        "dt": "control_dt",
        "collision": "hand_self_collision_mode",
        "naconmax": "runtime_naconmax",
        "njmax": "runtime_njmax",
    }
    for old, new in aliases.items():
        if old in overrides:
            if new in overrides:
                raise TypeError(f"cannot supply both {old!r} and {new!r}")
            overrides[new] = overrides.pop(old)
    servo_fields = set(asdict(base.servo))
    material_fields = set(asdict(base.material))
    base_fields = set(asdict(base)) - {"servo", "material"}
    unknown = set(overrides) - servo_fields - material_fields - base_fields
    if unknown:
        raise TypeError(f"unknown legacy physics keyword(s): {sorted(unknown)}")
    servo = DexHandServoConfig(**{**asdict(base.servo), **{k: overrides.pop(k) for k in tuple(overrides) if k in servo_fields}})
    material = DexHandMaterialConfig(**{**asdict(base.material), **{k: overrides.pop(k) for k in tuple(overrides) if k in material_fields}})
    return DexHandPhysicsConfig(**{**asdict(base), **overrides, "servo": servo, "material": material})


def resolve_physics_config(
    physics: DexHandPhysicsConfig | None = None, **legacy_overrides: Any
) -> DexHandPhysicsConfig:
    """Resolve one canonical profile or optional legacy physical overrides.

    Callers pass only explicitly supplied legacy values (``None`` means absent).
    Supplying a canonical profile and a legacy value is rejected rather than
    silently giving one source precedence over another.
    """
    unknown = set(legacy_overrides) - LEGACY_PHYSICS_FIELDS
    if unknown:
        raise TypeError(f"unknown legacy physics keyword(s): {sorted(unknown)}")
    supplied = {name: value for name, value in legacy_overrides.items() if value is not None}
    if physics is not None:
        if not isinstance(physics, DexHandPhysicsConfig):
            raise TypeError("physics must be a DexHandPhysicsConfig")
        if supplied:
            raise ValueError(
                "physics cannot be combined with legacy physical override(s): "
                f"{sorted(supplied)}"
            )
        return physics
    return legacy_physics_config(**supplied)


def physics_legacy_aliases(physics: DexHandPhysicsConfig) -> dict[str, Any]:
    """Compatibility attribute values derived from a canonical physics profile."""
    return {
        "dt": physics.control_dt,
        "control_dt": physics.control_dt,
        "substeps": physics.substeps,
        **asdict(physics.servo),
        "hand_friction": format_friction(physics.material.hand_friction),
        "object_friction": format_friction(physics.material.object_friction),
        "ground_friction": format_friction(physics.material.ground_friction),
        "hand_self_collision_mode": physics.hand_self_collision_mode,
        "collision": physics.hand_self_collision_mode,
        "naconmax": physics.runtime_naconmax,
        "njmax": physics.runtime_njmax,
    }
