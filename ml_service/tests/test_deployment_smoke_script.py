import importlib.util
import io
import os
from pathlib import Path
import signal

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deployment-smoke.py"
SPEC = importlib.util.spec_from_file_location("deployment_smoke", SCRIPT)
assert SPEC and SPEC.loader
deployment_smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deployment_smoke)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal handling is exercised on Linux CI")
def test_stop_process_uses_sigint_for_clean_uvicorn_exit():
    class Process:
        returncode = None

        def poll(self):
            return None

        def send_signal(self, received_signal):
            self.received_signal = received_signal

        def wait(self, timeout):
            self.returncode = 0

    process = Process()
    assert deployment_smoke.stop_process(process, "FastAPI service", io.StringIO()) is True
    assert process.received_signal == signal.SIGINT
    assert process.returncode == 0
