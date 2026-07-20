from sim.dexhandrl.obs_layout import ACTION_DIM_021PRO, OBS_COMPONENT_SLICES, OBS_DIM_021PRO, POINT_CLOUD_DIM


def test_obs_layout_contract() -> None:
    assert ACTION_DIM_021PRO == 22
    assert OBS_DIM_021PRO == 461
    assert POINT_CLOUD_DIM == 192
    assert OBS_COMPONENT_SLICES["action_type_onehot"] == (0, 50)
    assert OBS_COMPONENT_SLICES["object_point_cloud_hand_frame"] == (269, 461)
