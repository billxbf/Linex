import asyncio
import os
import shlex
import signal
import sys

import pytest

from polar.runtime import base as runtime_base


class _Runtime(runtime_base.BaseRuntime):
    """Concrete shell around the abstract base so _run_local_command can be exercised alone."""

    @property
    def runtime_id(self) -> str:
        return "test"

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def exec(self, command, *, cwd=None, env=None, timeout_sec=None): ...

    async def upload_file(self, local_path, remote_path) -> None: ...

    async def upload_dir(self, local_path, remote_path) -> None: ...

    async def download_file(self, remote_path, local_path) -> None: ...

    async def download_dir(self, remote_path, local_path) -> None: ...


class _ExitsAsKilled:
    """A subprocess that exits in the same instant the timeout fires: kill() finds no process."""

    returncode = None

    def kill(self):
        raise ProcessLookupError

    async def wait(self):
        await asyncio.sleep(10)
        return 0


def test_timeout_racing_process_exit_reports_timeout_not_error(monkeypatch) -> None:
    async def fake_exec(*args, **kwargs):
        return _ExitsAsKilled()

    monkeypatch.setattr(runtime_base.asyncio, "create_subprocess_exec", fake_exec)
    runtime = _Runtime.__new__(_Runtime)
    runtime._active_process = None

    rc, out, err = asyncio.run(runtime._run_local_command("sleep", "10", timeout=0.01))

    assert (rc, out, err) == (-1, None, None)
    assert runtime._active_process is None


def test_completed_command_does_not_wait_for_inherited_background_output(tmp_path) -> None:
    runtime = _Runtime.__new__(_Runtime)
    pid_file = tmp_path / "background.pid"
    command = f"sleep 10 9>&1 >/dev/null 2>&1 & echo $! > {shlex.quote(str(pid_file))}; printf done"
    try:
        result = asyncio.run(runtime._run_local_command("bash", "-c", command, capture=True, timeout=0.5))
        assert result == (0, "done", None)
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("capture", [False, True])
def test_local_command_preserves_exit_code_and_output(capture) -> None:
    runtime = _Runtime.__new__(_Runtime)
    result = asyncio.run(runtime._run_local_command(
        sys.executable, "-c", "import sys; print('out'); print('err',file=sys.stderr); sys.exit(7)",
        capture=capture,
    ))
    assert result == (7, "out\n" if capture else None, "err\n" if capture else None)


def test_running_command_still_times_out() -> None:
    runtime = _Runtime.__new__(_Runtime)
    result = asyncio.run(runtime._run_local_command(
        sys.executable, "-c", "import time; time.sleep(5)", timeout=0.05, capture=True,
    ))
    assert result == (-1, None, None)
