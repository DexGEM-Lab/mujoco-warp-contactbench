from __future__ import annotations

import numpy as np

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
