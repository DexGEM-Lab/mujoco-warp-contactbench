from __future__ import annotations

import numpy as np
import pytest

from sim.dexhandrl.contact import collect_contact_summary, decode_contact_force_world


def test_shared_contact_decoder_matches_pyramidal_force_contract() -> None:
    force_world, normal_force = decode_contact_force_world(
        frame=np.eye(3, dtype=np.float32),
        friction=np.asarray([0.5, 0.25, 0.0, 0.0, 0.0], dtype=np.float32),
        condim=3,
        efc_addresses=np.asarray([0, 1, 2, 3], dtype=np.int32),
        efc_force=np.asarray([2.0, 1.0, 4.0, 2.0], dtype=np.float32),
        pyramidal=True,
    )

    assert normal_force == 9.0
    np.testing.assert_allclose(force_world, [9.0, 0.5, 0.5])


def test_shared_contact_summary_uses_signed_body_and_object_forces() -> None:
    summary = collect_contact_summary(
        active_contacts=1,
        contact_geom=np.asarray([[0, 1]], dtype=np.int32),
        contact_frame=np.eye(3, dtype=np.float32)[None],
        contact_friction=np.asarray([[0.5, 0.25, 0.0, 0.0, 0.0]], dtype=np.float32),
        contact_dim=np.asarray([3], dtype=np.int32),
        contact_efc_address=np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        efc_force=np.asarray([2.0, 1.0, 4.0, 2.0], dtype=np.float32),
        pyramidal=True,
        geom_body_names={0: "RH1_3", 1: "cube1"},
        geom_names={0: "finger_geom", 1: "cube_geom"},
        contact_body_names=("RH1_3",),
        object_name="cube1",
    )

    np.testing.assert_allclose(summary["contact_force_vectors"][0], [-9.0, -0.5, -0.5])
    np.testing.assert_allclose(summary["object_force_vector"], [9.0, 0.5, 0.5])
    assert summary["active_contact_count"] == 1
    assert summary["hand_object_pairs"][0]["force_magnitude_N"] > 9.0


@pytest.mark.parametrize("cone", ("pyramidal", "elliptic"))
@pytest.mark.parametrize("condim", (1, 3))
def test_native_sphere_plane_contact_matches_mj_contact_force(cone: str, condim: int) -> None:
    """Exercise native positive contacts against MuJoCo's authoritative decoder."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <option timestep="0.001" gravity="0 0 -9.81" cone="{cone}"/>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1" condim="{condim}"/>
            <body name="ball" pos="0 0 0.04">
              <freejoint/>
              <geom name="sphere" type="sphere" size="0.05" mass="1" condim="{condim}"
                    friction="1 0.005 0.0001"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    for _ in range(20):
        mujoco.mj_step(model, data)
    assert data.ncon > 0

    contact = data.contact[0]
    wrench = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, 0, wrench)
    expected_world = wrench[:3] @ np.asarray(contact.frame).reshape(3, 3)
    assert np.linalg.norm(expected_world) > 0.0
    addresses = (
        np.full(6, -1, dtype=np.int32)
        if contact.efc_address < 0
        else np.arange(contact.efc_address, contact.efc_address + 6, dtype=np.int32)
    )
    force_world, normal_force = decode_contact_force_world(
        frame=np.asarray(contact.frame),
        friction=np.asarray(contact.friction),
        condim=int(contact.dim),
        efc_addresses=addresses,
        efc_force=np.asarray(data.efc_force),
        pyramidal=int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL),
    )
    np.testing.assert_allclose(force_world, expected_world, rtol=2e-5, atol=2e-5)
    assert normal_force == pytest.approx(wrench[0], rel=2e-5, abs=2e-5)

    sphere_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    summary = collect_contact_summary(
        active_contacts=1,
        contact_geom=np.asarray([[contact.geom1, contact.geom2]], dtype=np.int32),
        contact_frame=np.asarray(contact.frame, dtype=np.float32).reshape(1, 9),
        contact_friction=np.asarray(contact.friction, dtype=np.float32).reshape(1, 5),
        contact_dim=np.asarray([contact.dim], dtype=np.int32),
        contact_efc_address=addresses.reshape(1, 6),
        efc_force=np.asarray(data.efc_force),
        pyramidal=int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL),
        geom_body_names={sphere_geom: "ball", floor_geom: "floor"},
        geom_names={sphere_geom: "sphere", floor_geom: "floor"},
        contact_body_names=("ball",),
        object_name="floor",
    )
    ball_force = -expected_world if contact.geom1 == sphere_geom else expected_world
    np.testing.assert_allclose(summary["contact_force_vectors"][0], ball_force, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(summary["object_force_vector"], -ball_force, rtol=2e-5, atol=2e-5)
    assert summary["hand_object_pairs"][0]["force_normal_N"] == pytest.approx(abs(wrench[0]))
