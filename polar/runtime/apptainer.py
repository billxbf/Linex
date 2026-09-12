"""Apptainer-backed rollout runtime."""

from __future__ import annotations

import hashlib
import logging
import os
import pwd
import shlex
import shutil
from pathlib import Path

from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecResult, RuntimeSpec

logger = logging.getLogger(__name__)


class ApptainerRuntime(BaseRuntime):
    """Apptainer instance used across rollout stages."""

    def __init__(self, spec: RuntimeSpec, session_id: str, session_dir: Path) -> None:
        super().__init__(spec, session_id, session_dir)
        # Use a hash suffix to guarantee uniqueness even when session IDs
        # share a long prefix (e.g. "sk-polar-...-eval" vs "sk-polar-...").
        short_hash = hashlib.sha256(session_id.encode()).hexdigest()[:8]
        safe_name = session_id.replace("/", "-")[:30]
        self._instance_name = f"polar-{safe_name}-{short_hash}"
        self._binary = self._resolve_binary()
        # Rootless: the instance runs as this unprivileged host user inside a user namespace
        # (--fakeroot), so the agent is root only in its own namespace. Capabilities are not
        # namespaced by containers -- a real-root task set the host clock, wrote host sysctls
        # and deleted the checkout -- but they are scoped by the user namespace, so the kernel
        # rejects those on host objects without any per-vector patch.
        # Opt-in (POLAR_APPTAINER_ROOTLESS=1): the launcher must provide uidmap/squashfuse deps,
        # /etc/subuid ranges and a home for the sandbox user. Off, the instance runs as the
        # invoking root with the --drop-caps list and apptainer's own --memory cgroup.
        self._rootless = os.environ.get("POLAR_APPTAINER_ROOTLESS", "0") == "1"
        self._sandbox_uid = int(os.environ.get("POLAR_SANDBOX_UID", "1000"))
        self._sandbox_gid = self._sandbox_uid
        self._sandbox_env: dict[str, str] = {}
        if self._rootless:
            entry = pwd.getpwuid(self._sandbox_uid)
            self._sandbox_gid = entry.pw_gid
            self._sandbox_env = {"HOME": entry.pw_dir, "USER": entry.pw_name, "LOGNAME": entry.pw_name}
        # Rootless apptainer cannot apply --memory (no cgroup delegation); the gateway, which is
        # root, parks every apptainer process of this instance in a cgroup of its own instead.
        self._cgroup = (
            Path(os.environ.get("POLAR_SANDBOX_CGROUP", "/sys/fs/cgroup/polar")) / self._instance_name
            if self._rootless and self.spec.memory_mb is not None
            else None
        )

    @property
    def runtime_id(self) -> str:
        return self._instance_name

    @property
    def supports_gpus(self) -> bool:
        return True

    @property
    def can_disable_internet(self) -> bool:
        return True

    @property
    def supports_memory_limits(self) -> bool:
        # cgroup v2: the launcher must move its own processes out of the cgroup root and enable
        # +memory in cgroup.subtree_control; the gateway then creates one leaf per instance.
        return True

    def _unprivileged(self, argv: list[str]) -> list[str]:
        """Run ``argv`` as the sandbox user, first parking the process in the instance cgroup."""
        if not self._rootless:
            return list(argv)
        prefix = ["setpriv", f"--reuid={self._sandbox_uid}", f"--regid={self._sandbox_gid}",
                  "--clear-groups", "env", *(f"{k}={v}" for k, v in self._sandbox_env.items())]
        if self._cgroup is None:
            return [*prefix, *argv]
        # Root writes its own pid into the leaf, then drops privileges and execs apptainer; every
        # process it spawns inside the instance inherits the cgroup and its memory.max.
        return ["bash", "-c", f'echo $$ > {shlex.quote(str(self._cgroup / "cgroup.procs"))} && exec "$@"',
                "_", *prefix, *argv]

    def _exec_argv(self) -> list[str]:
        return self._unprivileged(
            [self._binary, "exec", "--drop-caps", self._DROP_CAPS, f"instance://{self._instance_name}"]
        )

    def _chown_tree(self, path: Path) -> None:
        if not self._rootless:
            return
        for root, dirs, files in os.walk(path):
            for name in (*dirs, *files):
                try:
                    os.chown(os.path.join(root, name), self._sandbox_uid, self._sandbox_gid, follow_symlinks=False)
                except OSError:
                    pass
        os.chown(path, self._sandbox_uid, self._sandbox_gid)

    def _setup_cgroup(self) -> None:
        assert self._cgroup is not None
        parent = self._cgroup.parent
        parent.mkdir(parents=True, exist_ok=True)
        try:
            (parent / "cgroup.subtree_control").write_text("+memory +pids\n")
        except OSError as exc:  # already enabled, or the parent holds processes
            logger.debug("cgroup subtree_control on %s: %s", parent, exc)
        self._cgroup.mkdir(exist_ok=True)
        (self._cgroup / "memory.max").write_text(f"{self.spec.memory_mb * 1024 * 1024}\n")

    def _teardown_cgroup(self) -> None:
        if self._cgroup is None:
            return
        try:
            self._cgroup.rmdir()
        except OSError as exc:  # a straggler is still exiting; the leaf is tiny, leave it
            logger.debug("cgroup rmdir %s: %s", self._cgroup, exc)

    def _copy_to_bind_mount(self, local_path: str, runtime_path: str) -> bool:
        copied = super()._copy_to_bind_mount(local_path, runtime_path)
        if copied:
            host_path = self.resolve_host_path(runtime_path)
            if host_path is not None and host_path.exists():
                self._chown_tree(host_path) if host_path.is_dir() else os.chown(
                    host_path, self._sandbox_uid, self._sandbox_gid
                )
        return copied

    async def start(self) -> None:
        if self._destroyed:
            raise RuntimeError("apptainer runtime was already destroyed")
        # Use a host-backed overlay directory instead of --writable-tmpfs
        # (default tmpfs overlay is only 64 MB, too small for most workloads).
        self._overlay_dir = self.session_dir / "overlay"
        self._overlay_dir.mkdir(parents=True, exist_ok=True)
        args = [self._binary, "instance", "start", *(["--fakeroot"] if self._rootless else []),
                "--overlay", str(self._overlay_dir), "--drop-caps", self._DROP_CAPS]
        # Task agents run as root here (root-launched apptainer keeps the instance root,
        # no userns remap). Apptainer otherwise auto-mounts the invoking process's $HOME
        # and CWD read-write into the instance; molt runs from the repo checkout, so that
        # silently exposed the host repo to a destructive agent (it deleted it once). A
        # rollout is self-contained -- SIF + overlay + the explicit session-dir bind below --
        # so drop both. $HOME then lives in the overlay, and the prepare step installs the
        # agent CLI there per session instead of into a shared host home.
        # Apptainer also binds the host /tmp and /var/tmp into the instance (`mount tmp = yes`),
        # so every task shared the trainer's /tmp: one agent's cleanup wiped Ray's session dir
        # and the Triton compile scratch mid-backward. Give each session its own host-backed
        # /tmp and /var/tmp (unbounded, unlike --writable-tmpfs) and bind nothing else shared.
        tmp_dir, var_tmp_dir = self.session_dir / "tmp", self.session_dir / "var_tmp"
        for path in (tmp_dir, var_tmp_dir):
            path.mkdir(parents=True, exist_ok=True)
        # Tasks run with the host network namespace, and /proc/sys is writable for root: a task
        # set net.core.somaxconn=0 on the host and the next run's c10d TCPStore never came up.
        args.extend(["--no-mount", "home,cwd,tmp",
                     "--bind", f"{tmp_dir}:/tmp", "--bind", f"{var_tmp_dir}:/var/tmp",
                     "--bind", "/proc/sys:/proc/sys:ro"])
        if self.spec.memory_mb is not None:
            # Without a cap one agent's python grew to 1.4 TB and took the trainer down with it.
            if self._cgroup is not None:
                self._setup_cgroup()
            else:
                args.extend(["--memory", f"{self.spec.memory_mb}m"])
        if self.spec.gpus > 0:
            args.append("--nv")
        network_name: str | None
        if not self.spec.allow_internet:
            network_name = "none"
        else:
            network_name = self.spec.network
        if network_name and network_name != "host":
            args.extend(["--net", "--network", network_name])
        args.extend(["--bind", f"{self.session_dir}:{self.runtime_session_dir}"])
        # Match DockerRuntime's kwargs.volumes contract. Apptainer accepts the
        # same src[:dst[:opts]] bind syntax for the read-only CLI mount used by
        # SWE-Gym.
        for volume in self.spec.kwargs.get("volumes", []):
            args.extend(["--bind", str(volume)])
        args.extend([self.spec.image, self._instance_name])
        # The sandbox user owns the session dir (bind-mounted rw) and the overlay; files the
        # gateway copies in later are re-owned in _copy_to_bind_mount.
        self._chown_tree(self.session_dir)
        # apptainer's stderr goes to a file, not a pipe: the instance daemon inherits the fd, so a
        # pipe would keep communicate() waiting for as long as the instance lives.
        start_log = self.session_dir / "logs" / "apptainer-start.err"
        start_log.parent.mkdir(parents=True, exist_ok=True)
        rc, _, _ = await self._run_local_command(
            "bash", "-c", 'exec "$@" 2>>"$0"', str(start_log), *self._unprivileged(args)
        )
        if rc != 0:
            detail = start_log.read_text(errors="replace").strip()[-2000:] if start_log.exists() else ""
            raise RuntimeError(f"{self._binary} instance start failed with exit code {rc}: {detail}")

    _STOP_TIMEOUT = 30.0
    # Root inside the instance otherwise keeps every capability of the (privileged) launcher, and
    # capabilities are not namespaced: a task agent set the host clock back 36 days mid-run
    # (CAP_SYS_TIME). Drop what a coding task never needs but the host cannot afford to lose;
    # chown/setuid/net_bind/net_raw stay so `su`, `chown` and dev servers keep working. Apptainer
    # applies the flag per process, so it goes on `instance start` and on every `exec`.
    _DROP_CAPS = ",".join((
        "CAP_SYS_TIME", "CAP_SYS_MODULE", "CAP_SYS_BOOT", "CAP_SYS_RAWIO", "CAP_SYS_ADMIN",
        "CAP_NET_ADMIN", "CAP_SYSLOG", "CAP_WAKE_ALARM", "CAP_MAC_ADMIN", "CAP_MAC_OVERRIDE",
        "CAP_LINUX_IMMUTABLE", "CAP_AUDIT_CONTROL", "CAP_AUDIT_READ", "CAP_BLOCK_SUSPEND",
        "CAP_SYS_PACCT", "CAP_SYS_TTY_CONFIG", "CAP_BPF", "CAP_PERFMON", "CAP_CHECKPOINT_RESTORE",
    ))

    async def stop(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        rc, _, stderr = await self._run_local_command(
            *self._unprivileged([self._binary, "instance", "stop", self._instance_name]),
            timeout=self._STOP_TIMEOUT, capture=True,
        )
        if rc != 0:
            logger.warning(
                "%s instance stop failed for %s (rc=%s): %s",
                self._binary, self._instance_name, rc, stderr,
            )
        self._teardown_cgroup()

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecResult:
        effective_env = {**self.spec.env, **(env or {})}
        effective_workdir = cwd or self.spec.workdir or self.runtime_session_dir
        wrapped_command = command
        if effective_workdir:
            wrapped_command = f"cd {shlex.quote(effective_workdir)} && {command}"
        shell_exports = []
        for key in ("HOME", "PATH"):
            if key in effective_env:
                shell_exports.append(f"export {key}={shlex.quote(str(effective_env[key]))};")
        if shell_exports:
            wrapped_command = " ".join(shell_exports + [wrapped_command])
        args = self._exec_argv()
        if effective_env:
            args.append("env")
            args.extend(f"{key}={value}" for key, value in effective_env.items())
        args.extend(["bash", "-lc", wrapped_command])
        rc, stdout, stderr = await self._run_local_command(
            *args, timeout=timeout_sec, capture=True
        )
        return ExecResult(stdout=stdout, stderr=stderr, return_code=rc)

    async def upload_file(self, local_path: str, remote_path: str) -> None:
        if self._copy_to_bind_mount(local_path, remote_path):
            return
        parent = str(Path(remote_path).parent)
        filename = Path(local_path).name
        source_dir = str(Path(local_path).parent)
        result = await self.exec(f"mkdir -p {shlex.quote(parent)}")
        if result.return_code != 0:
            raise RuntimeError(f"failed to create directory {parent} in runtime")
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"tar -cf - -C {shlex.quote(source_dir)} {shlex.quote(filename)} | "
            f"{shlex.join(self._exec_argv())} "
            f"tar -xf - -C {shlex.quote(parent)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(f"apptainer upload_file failed with exit code {rc}")

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        if self._copy_to_bind_mount(local_path, remote_path):
            return
        result = await self.exec(f"mkdir -p {shlex.quote(remote_path)}")
        if result.return_code != 0:
            raise RuntimeError(
                f"failed to create directory {remote_path} in runtime"
            )
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"tar -cf - -C {shlex.quote(local_path)} . | "
            f"{shlex.join(self._exec_argv())} "
            f"tar -xf - -C {shlex.quote(remote_path)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(f"apptainer upload_dir failed with exit code {rc}")

    async def download_file(self, remote_path: str, local_path: str) -> None:
        if self._copy_from_bind_mount(remote_path, Path(local_path)):
            return
        parent = str(Path(remote_path).parent)
        filename = Path(remote_path).name
        local_dir = str(Path(local_path).parent)
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"{shlex.join(self._exec_argv())} "
            f"tar -cf - -C {shlex.quote(parent)} {shlex.quote(filename)} | "
            f"tar -xf - -C {shlex.quote(local_dir)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(
                f"apptainer download_file failed with exit code {rc}"
            )

    async def download_dir(self, remote_path: str, local_path: str) -> None:
        if self._copy_from_bind_mount(remote_path, Path(local_path)):
            return
        Path(local_path).mkdir(parents=True, exist_ok=True)
        rc, _, _ = await self._run_local_command(
            "bash",
            "-c",
            f"{shlex.join(self._exec_argv())} "
            f"tar -cf - -C {shlex.quote(remote_path)} . | "
            f"tar -xf - -C {shlex.quote(local_path)}",
            capture=False,
        )
        if rc != 0:
            raise RuntimeError(
                f"apptainer download_dir failed with exit code {rc}"
            )

    @staticmethod
    def _resolve_binary() -> str:
        override = os.environ.get("POLAR_APPTAINER_BIN")
        if override:
            return override
        for candidate in ("/usr/bin/apptainer", "/bin/apptainer"):
            if Path(candidate).is_file():
                return candidate
        resolved = shutil.which("apptainer")
        if resolved:
            return resolved
        return "apptainer"
