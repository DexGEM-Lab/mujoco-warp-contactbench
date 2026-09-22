import json
from pathlib import Path
import sys

import pytest

from tools.replay_capacity_guard import OVERFLOW_EXIT_CODE, run_guarded


def receipt(log):
    return json.loads(Path(str(log) + '.guard.json').read_text())


def test_clean_run_and_guard_limitation_warning_are_not_overflow(tmp_path):
    log = tmp_path / 'action.log'
    command = [sys.executable, '-u', '-c',
               'print("same-step CCD-overflow detection is unavailable"); print("ok")']
    assert run_guarded(command, log) == 0
    assert receipt(log)['state'] == 'complete'
    assert receipt(log)['overflow'] is None
    assert 'ok' in log.read_text()


@pytest.mark.parametrize('diagnostic', [
    'CCD overflow - please increase naccdmax to 2498',
    'nefc overflow - please increase njmax to 6000',
    'narrowphase overflow - please increase nconmax or naconmax',
])
def test_native_warning_rejects_even_if_child_would_exit_zero(tmp_path, diagnostic):
    log = tmp_path / 'action.log'
    assert run_guarded([sys.executable, '-u', '-c', f'print({diagnostic!r})'], log) == OVERFLOW_EXIT_CODE
    assert receipt(log)['state'] == 'rejected_overflow'
    assert receipt(log)['overflow'] == diagnostic


def test_overflow_stops_live_child(tmp_path):
    log = tmp_path / 'action.log'
    command = [sys.executable, '-u', '-c',
               'import signal; print("CCD overflow - please increase naccdmax to 2498", flush=True); signal.pause()']
    assert run_guarded(command, log) == OVERFLOW_EXIT_CODE
    assert receipt(log)['process_exit_code'] < 0


def test_failed_run_and_existing_evidence_are_preserved(tmp_path):
    log = tmp_path / 'action.log'
    assert run_guarded([sys.executable, '-c', 'raise SystemExit(7)'], log) == 7
    assert receipt(log)['state'] == 'failed'
    before = log.read_bytes()
    with pytest.raises(FileExistsError):
        run_guarded([sys.executable, '-c', 'print("overwrite")'], log)
    assert log.read_bytes() == before
