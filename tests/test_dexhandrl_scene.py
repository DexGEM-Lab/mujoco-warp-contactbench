from pathlib import Path

import pytest

from sim.dexhandrl.scene import _patch_hand_xml


_HAND_XML = '''<mujoco>
  <compiler meshdir="old" />
  <default><geom friction="3 2 2" solref="0.01 0.9" /></default>
  <size njmax="1" nconmax="1" />
</mujoco>'''


def test_hand_friction_patch_changes_only_canonical_hand_default(tmp_path: Path) -> None:
    patched = _patch_hand_xml(_HAND_XML, tmp_path, dt=0.01, substeps=10, hand_friction="3 2 2")

    assert 'friction="3 2 2" solref=' in patched
    assert patched.count('friction="3 2 2"') == 1


@pytest.mark.parametrize("xml", [
    _HAND_XML.replace('friction="3 2 2"', 'friction="0.5 0.01 0.001"'),
    _HAND_XML.replace('</default>', '<geom friction="3 2 2" solref="0.01 0.9" /></default>'),
])
def test_hand_friction_patch_fails_closed_without_one_canonical_default(xml: str, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="exactly one canonical hand default"):
        _patch_hand_xml(xml, tmp_path, dt=0.01, substeps=10, hand_friction="3 2 2")
