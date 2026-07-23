"""Strict process-local CCD scratch reuse for the bundled MJX-Warp implementation.

MJX calls Warp through JAX FFI, where ``convex_narrowphase`` otherwise allocates
its EPA and multicontact buffers during graph execution.  This module intercepts
only that function's module-local ``wp`` binding; it never changes Warp globally.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import threading
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class CcdWorkspaceSpec:
    """The static allocation contract of bundled ``convex_narrowphase``."""

    device_ordinal: int
    naccdmax: int
    epa_iterations: int
    nmaxpolygon: int
    nmaxmeshdeg: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.device_ordinal, int)
            or isinstance(self.device_ordinal, bool)
            or self.device_ordinal < 0
        ):
            raise ValueError("device_ordinal must be a non-negative integer")
        for name in ("naccdmax", "epa_iterations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("nmaxpolygon", "nmaxmeshdeg"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass
class PersistentCcdWorkspace:
    spec: CcdWorkspaceSpec
    buffers: tuple[tuple[tuple[int, ...], Any, Any], ...]
    nccd_shape: tuple[int, ...]
    nccd_dtype: Any
    nccd: Any


class _WorkspaceWarpProxy:
    """A temporary exact-request allocator used only while narrowphase executes."""

    def __init__(self, workspace: PersistentCcdWorkspace, delegate: Any) -> None:
        self._workspace = workspace
        self._delegate = delegate
        self._next_empty = 0
        self._zeros_seen = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def empty(self, *args: Any, **kwargs: Any) -> Any:
        shape, dtype = _allocation_request("empty", args, kwargs)
        if self._next_empty >= len(self._workspace.buffers):
            raise RuntimeError("persistent CCD workspace exhausted by an unexpected wp.empty request")
        expected_shape, expected_dtype, buffer = self._workspace.buffers[self._next_empty]
        self._next_empty += 1
        if shape != expected_shape or dtype != expected_dtype:
            raise RuntimeError(
                "persistent CCD workspace request mismatch: "
                f"expected shape={expected_shape}, dtype={expected_dtype!r}; "
                f"received shape={shape}, dtype={dtype!r}"
            )
        return buffer

    def zeros(self, *args: Any, **kwargs: Any) -> Any:
        shape, dtype = _allocation_request("zeros", args, kwargs)
        if self._zeros_seen:
            raise RuntimeError("persistent CCD workspace received a duplicate wp.zeros request")
        if shape != self._workspace.nccd_shape or dtype != self._workspace.nccd_dtype:
            raise RuntimeError(
                "persistent CCD workspace nccd request mismatch: "
                f"expected shape={self._workspace.nccd_shape}, dtype={self._workspace.nccd_dtype!r}; "
                f"received shape={shape}, dtype={dtype!r}"
            )
        self._zeros_seen = True
        zero = getattr(self._workspace.nccd, "zero_", None)
        if zero is None:
            raise RuntimeError("persistent CCD workspace nccd buffer does not support zero_()")
        zero()
        return self._workspace.nccd

    def assert_complete(self) -> None:
        if not self._zeros_seen or self._next_empty != len(self._workspace.buffers):
            raise RuntimeError(
                "persistent CCD workspace allocation sequence was incomplete: "
                f"zeros={self._zeros_seen}, empty={self._next_empty}/{len(self._workspace.buffers)}"
            )


def _allocation_request(kind: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[int, ...], Any]:
    if len(args) > 2:
        raise RuntimeError(f"persistent CCD workspace does not accept positional {kind} arguments beyond shape and dtype")
    values = dict(kwargs)
    shape = values.pop("shape", args[0] if args else None)
    dtype = values.pop("dtype", args[1] if len(args) > 1 else None)
    if values or shape is None or dtype is None:
        raise RuntimeError(f"persistent CCD workspace received unsupported wp.{kind} request")
    if isinstance(shape, int):
        shape = (shape,)
    try:
        normalized_shape = tuple(int(dim) for dim in shape)
    except TypeError as exc:
        raise RuntimeError(f"persistent CCD workspace received invalid wp.{kind} shape") from exc
    return normalized_shape, dtype


_install_lock = threading.Lock()
_installation: tuple[ModuleType, ModuleType | None, CcdWorkspaceSpec, PersistentCcdWorkspace, Any] | None = None
_active = False


def _workspace_layout(collision: ModuleType, spec: CcdWorkspaceSpec) -> tuple[tuple[tuple[int, ...], Any], ...]:
    wp = collision.wp
    face_size = 6 + collision.MJ_MAX_EPAFACES * spec.epa_iterations
    vertex_size = 10 + 2 * spec.epa_iterations
    return (
        ((spec.naccdmax, vertex_size), wp.vec3),
        ((spec.naccdmax, vertex_size), int),
        ((spec.naccdmax, face_size), int),
        ((spec.naccdmax, face_size), wp.vec3),
        ((spec.naccdmax, face_size), float),
        ((spec.naccdmax, collision.MJ_MAX_EPAHORIZON), int),
        ((spec.naccdmax, 2 * spec.nmaxpolygon), wp.vec3),
        ((spec.naccdmax, 2 * spec.nmaxpolygon), wp.vec3),
        ((spec.naccdmax, spec.nmaxpolygon), wp.vec3),
        ((spec.naccdmax, spec.nmaxpolygon), float),
        ((spec.naccdmax, spec.nmaxmeshdeg), int),
        ((spec.naccdmax, spec.nmaxmeshdeg), int),
        ((spec.naccdmax, spec.nmaxmeshdeg), wp.vec3),
        ((spec.naccdmax, spec.nmaxmeshdeg), wp.vec3),
        ((spec.naccdmax, spec.nmaxmeshdeg), wp.vec3),
        ((spec.naccdmax, spec.nmaxpolygon), wp.vec3),
        ((spec.naccdmax, spec.nmaxpolygon), wp.vec3),
    )


def install_persistent_ccd_workspace(
    *,
    device_ordinal: int,
    naccdmax: int,
    epa_iterations: int,
    nmaxpolygon: int,
    nmaxmeshdeg: int,
    collision_module: ModuleType | None = None,
    driver_module: ModuleType | None = None,
) -> PersistentCcdWorkspace:
    """Install one exact persistent workspace for bundled Warp convex narrowphase.

    Reinstalling the same process-local contract returns the installed workspace.
    A different module or static shape is rejected before it can alias buffers.
    """

    global _installation
    spec = CcdWorkspaceSpec(
        device_ordinal=device_ordinal,
        naccdmax=naccdmax,
        epa_iterations=epa_iterations,
        nmaxpolygon=nmaxpolygon,
        nmaxmeshdeg=nmaxmeshdeg,
    )
    if collision_module is None:
        from mujoco.mjx.third_party.mujoco_warp._src import collision_convex as collision_module
        from mujoco.mjx.third_party.mujoco_warp._src import collision_driver as driver_module

    with _install_lock:
        if _installation is not None:
            installed_module, installed_driver, installed_spec, workspace, _ = _installation
            if (
                installed_module is not collision_module
                or installed_driver is not driver_module
                or installed_spec != spec
            ):
                raise RuntimeError(
                    "persistent CCD workspace is already installed with an incompatible module or static shape"
                )
            return workspace

        wp = collision_module.wp
        device = f"cuda:{spec.device_ordinal}"
        layout = _workspace_layout(collision_module, spec)
        buffers = tuple(
            (shape, dtype, wp.empty(shape=shape, dtype=dtype, device=device))
            for shape, dtype in layout
        )
        nccd_shape = (len(collision_module.GeomType) * (len(collision_module.GeomType) + 1) // 2,)
        nccd = wp.empty(shape=nccd_shape, dtype=int, device=device)
        workspace = PersistentCcdWorkspace(spec, buffers, nccd_shape, int, nccd)
        original = collision_module.convex_narrowphase

        @wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            global _active
            with _install_lock:
                if _active:
                    raise RuntimeError("persistent CCD workspace does not permit concurrent convex_narrowphase calls")
                _active = True
            proxy = _WorkspaceWarpProxy(workspace, wp)
            previous_wp = collision_module.wp
            collision_module.wp = proxy
            try:
                result = original(*args, **kwargs)
                proxy.assert_complete()
                return result
            finally:
                collision_module.wp = previous_wp
                with _install_lock:
                    _active = False

        if driver_module is not None:
            bound = getattr(driver_module, "convex_narrowphase", None)
            if bound is not original:
                raise RuntimeError(
                    "persistent CCD workspace found an unexpected collision_driver convex_narrowphase binding"
                )
            driver_module.convex_narrowphase = wrapped
        collision_module.convex_narrowphase = wrapped
        _installation = (collision_module, driver_module, spec, workspace, original)
        return workspace


def warp_device_ordinal(device: Any) -> int:
    """Extract the JAX CUDA ordinal used by the corresponding Warp allocations."""

    ordinal = getattr(device, "id", None)
    if isinstance(ordinal, int) and not isinstance(ordinal, bool) and ordinal >= 0:
        return ordinal
    raise RuntimeError(f"cannot derive a non-negative CUDA device ordinal from {device!r}")
