from sim.dexhandrl.scene import _hand_self_collision_excludes_xml, _patch_pd_gains


def test_hand_collision_excludes_match_isaac_semantic_groups() -> None:
    xml = _hand_self_collision_excludes_xml()

    assert xml.count("<exclude ") == 40
    assert 'body1="RH0_0" body2="RH0_1"' in xml
    assert 'body1="RFH1" body2="RH0_0"' in xml
    assert 'body1="RH0_0" body2="RH1_0"' not in xml


def test_hand_collision_excludes_support_all_disabled_and_none() -> None:
    all_disabled = _hand_self_collision_excludes_xml("all_disabled")

    assert all_disabled.count("<exclude ") == 210
    assert 'body1="RH3_1" body2="RH4_2"' in all_disabled
    assert _hand_self_collision_excludes_xml("none") == ""


def test_pd_gain_patch_writes_kv_and_force_limits() -> None:
    xml = """
    <actuator>
      <position name="act_ARTx" joint="ARTx" kp="1" kv="2" forcerange="-3 3" />
      <position name="act_ARRx" joint="ARRx" kp="1" kv="2" forcerange="-3 3" />
      <position name="act_r_f_jiont_1_1" joint="r_f_jiont_1_1" kp="1" />
    </actuator>
    """

    patched = _patch_pd_gains(
        xml,
        base_pos_kp=10,
        base_pos_kv=11,
        base_pos_force=12,
        base_rot_kp=20,
        base_rot_kv=21,
        base_rot_force=22,
        finger_kp=30,
        finger_kv=31,
        finger_force=32,
    )

    assert 'name="act_ARTx" joint="ARTx" kp="10" kv="11" forcerange="-12 12"' in patched
    assert 'name="act_ARRx" joint="ARRx" kp="20" kv="21" forcerange="-22 22"' in patched
    assert 'name="act_r_f_jiont_1_1" joint="r_f_jiont_1_1" kp="30" kv="31" forcelimited="true" forcerange="-32 32"' in patched
