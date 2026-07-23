from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace

import pytest


class _FakeArray:
    def __init__(self, shape: tuple[int, ...], dtype: object) -> None:
        self.shape = shape
        self.dtype = dtype
        self.zero_calls = 0

    def zero_(self) -> None:
        self.zero_calls += 1


class _FakeWarp:
    vec3 = object()

    def __init__(self) -> None:
        self.allocations: list[tuple[tuple[int, ...], object, str]] = []

    def empty(self, *, shape: tuple[int, ...], dtype: object, device: str) -> _FakeArray:
        self.allocations.append((tuple(shape), dtype, device))
        return _FakeArray(tuple(shape), dtype)


def _fresh_workspace_module():
    import sim.manorl.mjx_warp_workspace as workspace

    return importlib.reload(workspace)


def _collision(*, malformed: bool = False) -> ModuleType:
    module = ModuleType("fake_collision_convex")
    module.wp = _FakeWarp()
    module.MJ_MAX_EPAFACES = 2
    module.MJ_MAX_EPAHORIZON = 7
    module.GeomType = (0, 1, 2)

    def narrowphase() -> None:
        wp = module.wp
        wp.zeros(len(module.GeomType) * (len(module.GeomType) + 1) // 2, dtype=int)
        expected = (
            ((5, 18), wp.vec3),
            ((5, 18), int),
            ((5, 14), int),
            ((5, 14), wp.vec3),
            ((5, 14), float),
            ((5, 7), int),
            ((5, 22), wp.vec3),
            ((5, 22), wp.vec3),
            ((5, 11), wp.vec3),
            ((5, 11), float),
            ((5, 13), int),
            ((5, 13), int),
            ((5, 13), wp.vec3),
            ((5, 13), wp.vec3),
            ((5, 13), wp.vec3),
            ((5, 11), wp.vec3),
            ((5, 11), wp.vec3),
        )
        for index, (shape, dtype) in enumerate(expected):
            if malformed and index == 3:
                shape = (5, 15)
            wp.empty(shape=shape, dtype=dtype)

    module.convex_narrowphase = narrowphase
    return module


def _install(workspace, collision):
    return workspace.install_persistent_ccd_workspace(
        device_ordinal=2,
        naccdmax=5,
        epa_iterations=4,
        nmaxpolygon=11,
        nmaxmeshdeg=13,
        collision_module=collision,
    )


def test_workspace_reuses_exact_allocation_sequence_and_zeros_nccd() -> None:
    workspace = _fresh_workspace_module()
    collision = _collision()
    installed = _install(workspace, collision)
    original_wp = collision.wp

    collision.convex_narrowphase()
    collision.convex_narrowphase()

    assert collision.wp is original_wp
    assert len(original_wp.allocations) == 18  # 17 empty buffers plus persistent nccd.
    assert all(device == "cuda:2" for _, _, device in original_wp.allocations)
    assert installed.nccd.zero_calls == 2
    assert _install(workspace, collision) is installed


def test_workspace_rejects_shape_mismatch_and_restores_module_wp() -> None:
    workspace = _fresh_workspace_module()
    collision = _collision(malformed=True)
    _install(workspace, collision)
    original_wp = collision.wp

    with pytest.raises(RuntimeError, match="request mismatch"):
        collision.convex_narrowphase()

    assert collision.wp is original_wp


def test_workspace_rejects_duplicate_and_incompatible_installations() -> None:
    workspace = _fresh_workspace_module()
    collision = _collision()
    _install(workspace, collision)

    with pytest.raises(RuntimeError, match="incompatible"):
        workspace.install_persistent_ccd_workspace(
            device_ordinal=2,
            naccdmax=6,
            epa_iterations=4,
            nmaxpolygon=11,
            nmaxmeshdeg=13,
            collision_module=collision,
        )

    def duplicate_zeros() -> None:
        collision.wp.zeros(6, dtype=int)
        collision.wp.zeros(6, dtype=int)

    collision.convex_narrowphase = duplicate_zeros
    # The installed wrapper owns the original function, so exercise the proxy
    # directly to distinguish duplicate allocation from a changed callable.
    proxy = workspace._WorkspaceWarpProxy(workspace._installation[2], collision.wp)
    proxy.zeros(6, dtype=int)
    with pytest.raises(RuntimeError, match="duplicate"):
        proxy.zeros(6, dtype=int)


def test_workspace_rejects_concurrent_reentry() -> None:
    workspace = _fresh_workspace_module()
    collision = _collision()
    _install(workspace, collision)
    original = collision.convex_narrowphase

    def reentrant() -> None:
        original()

    # Directly set the guard to model another execution already in flight;
    # the public wrapper must fail rather than share mutable buffers.
    workspace._active = True
    try:
        with pytest.raises(RuntimeError, match="concurrent"):
            original()
    finally:
        workspace._active = False


def test_warp_device_ordinal_requires_jax_device_id() -> None:
    workspace = _fresh_workspace_module()
    assert workspace.warp_device_ordinal(SimpleNamespace(id=0)) == 0
    assert workspace.warp_device_ordinal(SimpleNamespace(id=3)) == 3
    assert workspace.CcdWorkspaceSpec(0, 8, 16, 1, 1).device_ordinal == 0
    with pytest.raises(RuntimeError, match="cannot derive"):
        workspace.warp_device_ordinal(SimpleNamespace(id="3"))
