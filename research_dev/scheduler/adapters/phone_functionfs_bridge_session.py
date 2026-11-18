"""The helper-phone session over the direct USB bridge (transport ``functionfs-bridge``).

Moved here unchanged from ``campaigns/burstgpt/tools/weight_stream_microbench.py`` (which re-exports it), so
that drivers and campaigns share one class. It has the method names of ``AdbTcpPhoneWorkerSession``
(``preflight``, ``start``, ``transport_contract``, ``stop``, ``active``); the phone side is the production
session script (``adapters/native/direct_phone_ffn_session.sh``) and the host side the
``llama-ffn-split-usb-bridge`` binary.

Two opt-in additions (absent = the original behaviour):

* ``tolerate_launch_disconnect``: the detached launch command may return adb exit status 255 on hosts where
  the phone leaves adb before the command returns; ``start`` then waits for the device as usual and still
  fails if the phone session ends first. The dropped reply is kept in ``launch_dropped``.
* ``multi_session={"resident_workers", "resident_router", "rows"}``: several NPU sessions through the
  production resident workers and router; the session script takes these three arguments after the
  single-session ones. (A keyword, not a key of the ``functionfs`` section: older harnesses that subclass
  this class keep their own handling of such a key.)
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping

F16_IO_FLAG, SWIGLU_FLAG = 1, 2      # EXECUTE flag bits of the FFN split protocol
MULTI_SESSION_KEYS = ("resident_workers", "resident_router", "rows")
# the ``functionfs`` section of a session: the rig's phone block + the transport identity
FUNCTIONFS_KEYS = ("session_script", "session_script_sha256", "restore_script", "restore_script_sha256",
                   "session_root", "android_gadget_path", "functionfs_gadget_path", "functionfs_root_path",
                   "phone_usb_controller", "kernel_release", "bridge_path", "bridge_sha256", "bridge_port")


class FunctionFsBridgePhoneWorkerSession:
    """The OP15 leg over the production FunctionFS DMA-BUF path.

    The production phone session script (rig.json ``phone.session_script``) starts the pinned worker with
    ``--ffs-root``, swaps the Android gadget for the FunctionFS one (18d1:2d00, no adb while it runs) and restores
    Android USB when the worker exits. On the host, ``llama-ffn-split-usb-bridge`` accepts this harness's TCP v6
    connection and forwards every call over FunctionFS. The worker runs without a budget; the driver ends the leg
    with the bridge's local shutdown request, the bridge ends the phone session (layer -1) and exits with its
    FFNDMABUF receipt. ``stop`` then waits for adb, acknowledges the restoration (restore_android_usb.sh restarts
    adbd without it) and copies the phone session's logs. A start or stop that leaves the gadget up uses the
    production close tool (``usb_close_path``) or the restore script, so the phone never waits for the watchdog."""

    close_request = "shutdown"
    VENDOR_ID, PRODUCT_ID = "18d1", "2d00"
    # set by direct_phone_ffn_session.sh itself; a leg's worker_environment may repeat them, nothing else
    SCRIPT_FIXED_ENVIRONMENT = {"GGML_HEXAGON_MBUF": "4192", "GGML_HEXAGON_NDEV": "1", "S41_DISABLE_GRAPH_CACHE": "1"}

    def __init__(self, worker, functionfs: Mapping[str, Any], *, run: Callable = subprocess.run,
                 popen: Callable = subprocess.Popen, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, probe: Callable | None = None,
                 restored: Callable | None = None, listening: Callable[[int], bool] | None = None,
                 tolerate_launch_disconnect: bool = False,
                 multi_session: Mapping[str, str] | None = None) -> None:
        from research_dev.scheduler.adapters import bridge
        missing = [key for key in FUNCTIONFS_KEYS if key not in functionfs]
        if missing:
            raise ValueError("functionfs section needs " + ",".join(missing))
        worker_dir = str(Path(worker.worker_path).parent)
        environment = dict(worker.worker_environment)
        fixed = dict(self.SCRIPT_FIXED_ENVIRONMENT, ADSP_LIBRARY_PATH=worker_dir)
        if tuple(worker.library_directories) != (worker_dir,) or any(
                key not in fixed and key not in ("GGML_HEXAGON_NHVX", "GGML_HEXAGON_VMEM")
                or key in fixed and value != fixed[key] for key, value in environment.items()):
            raise ValueError("the session script loads the worker's own directory with its fixed environment; "
                             "only GGML_HEXAGON_NHVX / GGML_HEXAGON_VMEM may differ")
        self.worker, self.functionfs = worker, dict(functionfs)
        self.nhvx = environment.get("GGML_HEXAGON_NHVX", "4")
        self.vmem = environment.get("GGML_HEXAGON_VMEM", "3328")
        self._run, self._popen, self._sleep, self._clock = run, popen, sleep, clock
        self._probe = probe or (lambda: bridge.probe_functionfs_usb_device(
            vendor_id=self.VENDOR_ID, product_id=self.PRODUCT_ID))
        self._restored = restored or (lambda timeout_s: bridge.verify_android_usb_restored(
            serial=worker.serial, adb_port=worker.adb_port,
            minimum_speed_mbps=int(self.functionfs.get("minimum_usb_speed_mbps", 5000)), timeout_s=timeout_s))
        self._listening = listening or self._tcp_listening
        self._parse_terminal = bridge.parse_functionfs_bridge_terminal
        self._bridge: subprocess.Popen | None = None
        self._bridge_log: Path | None = None
        self._remote_root: str | None = None
        self._log_dir: Path | None = None
        self._boot_id = ""
        self.restoration: dict[str, Any] | None = None
        self.phone_terminal_status: str | None = None
        self.orphans_killed: list[int] = []
        self.tolerate_launch_disconnect = bool(tolerate_launch_disconnect)
        self.launch_dropped: dict[str, str] | None = None
        if multi_session is not None:
            absent = [key for key in MULTI_SESSION_KEYS if not multi_session.get(key)]
            if absent:
                raise ValueError("multi_session needs " + ",".join(absent))
        self.multi_session = None if multi_session is None else {key: str(multi_session[key])
                                                                 for key in MULTI_SESSION_KEYS}

    @property
    def active(self) -> bool:
        return self._bridge is not None or self._remote_root is not None

    def _root_shell(self, command: str, *, timeout_s: float = 60) -> str:
        worker = self.worker
        return self._run([str(worker.adb_path), "-P", str(worker.adb_port), "-s", worker.serial, "shell",
                          "su -c " + shlex.quote(command)], check=True, capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, timeout=timeout_s).stdout

    @staticmethod
    def _tcp_listening(port: int) -> bool:
        suffix = "0100007F:%04X" % port
        with open("/proc/net/tcp", encoding="ascii") as table:
            return any(len(columns) > 3 and columns[1] == suffix and columns[3] == "0A"
                       for columns in (line.split() for line in table))

    def _receipt(self, phase: str, details: Mapping[str, Any]):
        from research_dev.scheduler.adapters.phone_tcp_session import AdbTcpWorkerReceipt
        return AdbTcpWorkerReceipt(phase, self.worker.device_id, self.worker.serial, self._boot_id, dict(details))

    def _processes(self) -> list[tuple[int, int, str]]:
        rows = []
        for line in self._root_shell("ps -A -o PID,PPID,ARGS").splitlines():
            fields = line.split(None, 2)
            if len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit():
                rows.append((int(fields[0]), int(fields[1]), fields[2]))
        return rows

    def _worker_running(self) -> bool:
        """A process executing the worker (the script's own arguments name it too: not a worker)."""
        return any(args.split()[:1] == [self.worker.worker_path] for _, _, args in self._processes())

    def _kill_orphaned_watchdog(self) -> list[int]:
        """SIGKILL what is left of this session's script: its watchdog subshell outlives the script's cleanup
        (the plain kill does not end it while it waits on ``sleep session_timeout_s``; 2026-10-11, PID re-parented
        to init 20 min after the smoke) and still carries the script's arguments; then the subshell's sleep."""
        rows = self._processes()
        script, root = self.functionfs["session_script"], self._remote_root
        parents = [pid for pid, _, args in rows if script in args and root in args.split()]
        children = [pid for pid, ppid, _ in rows if ppid in parents]
        if parents:
            self._root_shell("kill -KILL " + " ".join(map(str, parents + children)) + " 2>/dev/null; true",
                             timeout_s=10)
        return parents + children

    def _gadget_up(self) -> bool:
        try:
            self._probe()
        except Exception:  # noqa: BLE001 - absent (or not unique): no live FunctionFS function to close
            return False
        return True

    def preflight(self):
        """Refuse a foreign kernel, a running worker, a live FunctionFS device, a busy bridge port and any
        phone or host file that differs from its pin."""
        worker, functionfs = self.worker, self.functionfs
        release = self._root_shell("uname -r").strip()
        if release != functionfs["kernel_release"]:
            raise RuntimeError(f"phone kernel {release!r} is not the FunctionFS kernel {functionfs['kernel_release']}")
        if self._worker_running():
            raise RuntimeError(f"{worker.serial} already runs {worker.worker_path}")
        if self._gadget_up():
            raise RuntimeError("a FunctionFS device (18d1:2d00) is already enumerated on this host")
        if self._listening(int(functionfs["bridge_port"])):
            raise RuntimeError(f"bridge port {functionfs['bridge_port']} is in use")
        pins = dict(worker.expected_sha256_by_path, **{functionfs["session_script"]: functionfs["session_script_sha256"],
                                                       functionfs["restore_script"]: functionfs["restore_script_sha256"]})
        observed = {}
        for line in self._root_shell("sha256sum " + " ".join(shlex.quote(path) for path in pins),
                                     timeout_s=900).splitlines():
            digest, _, path = line.strip().partition("  ")
            observed[path] = "sha256:" + digest
        mismatched = sorted(path for path, digest in pins.items() if observed.get(path) != digest)
        bridge_sha256 = "sha256:" + hashlib.sha256(Path(functionfs["bridge_path"]).read_bytes()).hexdigest()
        if bridge_sha256 != functionfs["bridge_sha256"]:
            mismatched.append(functionfs["bridge_path"])
        if mismatched:
            raise RuntimeError("files differ from their pinned sha256: " + ",".join(mismatched))
        self._boot_id = self._root_shell("cat /proc/sys/kernel/random/boot_id").strip()
        return self._receipt("preflight", {"kernel_release": release, "observed_sha256_by_path": observed,
                                           "bridge_sha256": bridge_sha256, "transport": "functionfs-bridge"})

    def _usbfs_bytes(self) -> int:
        override = self.functionfs.get("usbfs_available_bytes")
        if override is not None:
            return int(override)
        return int(Path("/sys/module/usbcore/parameters/usbfs_memory_mb").read_text().strip()) * 1024 * 1024

    def _remote_command(self, remote_root: str) -> str:
        worker, functionfs = self.worker, self.functionfs
        from research_dev.scheduler.adapters.phone_helpers import layer_spec
        words = [
            "SCHEDULER_ANDROID_GADGET=" + functionfs["android_gadget_path"],
            "SCHEDULER_FUNCTIONFS_GADGET=" + functionfs["functionfs_gadget_path"],
            "SCHEDULER_FUNCTIONFS_ROOT=" + functionfs["functionfs_root_path"],
            "SCHEDULER_PHONE_UDC=" + functionfs["phone_usb_controller"],
            "S41_FFN_F16_IO=1", "S41_FFN_STAGED_DMABUF=0", "S41_FFN_MAX_TOKENS=" + str(worker.max_tokens),
            "S41_FFN_COLUMN_QUANTUM=" + str(worker.column_quantum),
            "S41_FFN_QUEUE_DEPTH=" + str(functionfs.get("queue_depth", 1)),
            "S41_FFN_VMEM=" + self.vmem, "GGML_HEXAGON_NHVX=" + self.nhvx,
            "sh", functionfs["session_script"], worker.worker_path, worker.shard_path, layer_spec(worker.layer_mask),
            str(worker.columns), worker.backend, remote_root, functionfs["restore_script"],
            str(functionfs.get("session_timeout_s", 1800)), "0", worker.artifact_sha256,
        ]
        if self.multi_session is not None:
            words += [self.multi_session[key] for key in MULTI_SESSION_KEYS]
        return ("mkdir -p " + shlex.quote(remote_root) + " && nohup env " + shlex.join(words)
                + " > " + shlex.quote(remote_root + "/launch.log") + " 2>&1 < /dev/null &")

    def start(self, log_path: Path):
        """Launch the phone session, wait for the FunctionFS device, then start the bridge (it accepts the
        driver's connection and opens USB on the driver's HELLO, so readiness here = its port listens)."""
        if self.active:
            raise RuntimeError("functionfs-bridge session is already active")
        if not self._boot_id:
            self.preflight()
        functionfs = self.functionfs
        token = hashlib.sha256(f"ws16:{os.getpid()}:{time.monotonic_ns()}".encode()).hexdigest()[:16]
        self._remote_root = functionfs["session_root"].rstrip("/") + "/" + token
        self._log_dir = log_path.parent
        try:
            try:
                self._root_shell(self._remote_command(self._remote_root), timeout_s=30)
            except subprocess.CalledProcessError as error:
                if not self.tolerate_launch_disconnect or error.returncode != 255:
                    raise
                self.launch_dropped = {"stderr": (error.stderr or "")[-500:],
                                       "stdout": (error.stdout or "")[-200:]}
            deadline = self._clock() + float(functionfs.get("launch_timeout_s", 240))
            next_check, usb, last = self._clock() + 2.0, None, "FunctionFS USB did not enumerate"
            minimum = int(functionfs.get("minimum_usb_speed_mbps", 5000))
            while usb is None:
                if self._clock() > deadline:
                    raise RuntimeError(last)
                try:
                    observed = self._probe()
                    if observed.negotiated_speed_mbps >= minimum:
                        usb = observed
                        break
                    last = f"FunctionFS USB at {observed.negotiated_speed_mbps} Mb/s (< {minimum})"
                except Exception as error:  # noqa: BLE001 - not enumerated yet
                    last = str(error)
                if self._clock() >= next_check:  # adb still answers until the gadget swap
                    next_check = self._clock() + 2.0
                    try:
                        status = self._root_shell("cat " + shlex.quote(self._remote_root + "/terminal.status")
                                                  + " 2>/dev/null || true", timeout_s=5).strip()
                    except Exception:  # noqa: BLE001 - adb is gone: the swap is under way
                        status = ""
                    if status:
                        raise RuntimeError("phone session ended before USB enumeration: status=" + status)
                self._sleep(0.25)
            usbfs_bytes = self._usbfs_bytes()
            command = [functionfs["bridge_path"], "127.0.0.1", str(functionfs["bridge_port"]),
                       functionfs.get("allocator", "devmem"), str(functionfs.get("queue_depth", 1)),
                       str(usbfs_bytes), functionfs.get("transport_generation", "functionfs-dmabuf-async-ring-v2")]
            environment = dict(os.environ)
            if functionfs.get("bridge_library_path"):
                environment["LD_LIBRARY_PATH"] = functionfs["bridge_library_path"]
            self._bridge_log = log_path.with_name("bridge.log")
            with self._bridge_log.open("x") as log:
                self._bridge = self._popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, env=environment)
            listen_deadline = self._clock() + 15
            while not self._listening(int(functionfs["bridge_port"])):
                if self._bridge.poll() is not None or self._clock() > listen_deadline:
                    raise RuntimeError("USB bridge did not listen: " + self._bridge_log.read_text()[-2000:])
                self._sleep(0.1)
        except BaseException as error:
            cleanup = self._release(force_close=True)
            if cleanup:
                raise RuntimeError(f"{error}; cleanup: {'; '.join(cleanup)}") from error
            raise
        return self._receipt("launch", {"remote_root": self._remote_root, "bridge_command": command,
                                        "bridge_pid": self._bridge.pid, "usbfs_available_bytes": usbfs_bytes,
                                        "usb": dataclasses.asdict(usb), "transport": "functionfs-bridge"})

    def transport_contract(self):
        return SimpleNamespace(control_port=int(self.functionfs["bridge_port"]))

    def _close_gadget(self) -> list[str]:
        """End a phone session whose bridge did not: the production close tool, else the restore script is
        out of reach (no adb while the gadget is up) and the watchdog (session_timeout_s) restores USB."""
        close_path = self.functionfs.get("usb_close_path")
        if not close_path:
            return ["FunctionFS gadget still up and no usb_close_path: the phone watchdog restores USB after "
                    + str(self.functionfs.get("session_timeout_s", 1800)) + " s"]
        worker = self.worker
        flags = F16_IO_FLAG | (SWIGLU_FLAG if worker.swiglu else 0)
        result = self._run([close_path, "0x" + self.VENDOR_ID, "0x" + self.PRODUCT_ID, hex(worker.layer_mask),
                            str(worker.n_embd), str(worker.columns), str(worker.max_tokens), str(flags),
                            str(self._usbfs_bytes()),
                            self.functionfs.get("transport_generation", "functionfs-dmabuf-async-ring-v2"),
                            worker.artifact_sha256], check=False, capture_output=True, text=True, timeout=30)
        if result.returncode != 0 or "FFNUSB_CLOSE status=ok" not in result.stdout:
            return ["USB close failed: " + (result.stderr.strip() or result.stdout.strip())[-500:]]
        return []

    def _release(self, *, force_close: bool) -> list[str]:
        """Bridge gone, gadget gone, Android USB back with the host's acknowledgement, phone logs copied."""
        errors: list[str] = []
        if self._bridge is not None and self._bridge.poll() is None:
            self._bridge.terminate()
            try:
                self._bridge.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._bridge.kill()
                self._bridge.wait(timeout=10)
        if self._remote_root is not None:
            deadline = self._clock() + (5 if force_close else 30)
            while self._gadget_up() and self._clock() < deadline:
                self._sleep(0.5)
            if self._gadget_up():
                errors += self._close_gadget()
            else:
                try:  # a start that failed before the gadget swap: the script's own cleanup restores USB
                    self._root_shell("test -f " + shlex.quote(self._remote_root + "/terminal.status")
                                     + " || sh " + shlex.quote(self.functionfs["restore_script"]) + " "
                                     + shlex.quote(self._remote_root), timeout_s=30)
                except Exception:  # noqa: BLE001 - adb not back yet; the restoration wait below decides
                    pass
            try:
                self.restoration = dataclasses.asdict(self._restored(90))
            except Exception as error:  # noqa: BLE001
                errors.append("Android USB not restored: " + str(error))
            else:
                try:
                    self._root_shell("touch " + shlex.quote(self._remote_root + "/host-ack"), timeout_s=10)
                    self.orphans_killed = self._kill_orphaned_watchdog()
                    self.phone_terminal_status = self._root_shell(
                        "cat " + shlex.quote(self._remote_root + "/terminal.status") + " 2>/dev/null || true",
                        timeout_s=10).strip()
                    for name in ("launch.log", "session.log", "worker.log", "restore-watchdog.log"):
                        text = self._root_shell("cat " + shlex.quote(self._remote_root + "/" + name)
                                                + " 2>/dev/null || true", timeout_s=30)
                        (self._log_dir / ("phone-" + name)).write_text(text)
                except Exception as error:  # noqa: BLE001
                    errors.append("phone session acknowledgement failed: " + str(error))
        self._bridge, self._remote_root = None, None
        return errors

    def release(self, *, force_close: bool = True) -> list[str]:
        """Failure-path cleanup: end the bridge and the phone session, wait for adb; returns what failed."""
        return self._release(force_close=force_close)

    def stop(self, *, served_calls: int):
        """The bridge exits after the driver's shutdown request; its receipt must count every call."""
        errors, terminal, returncode = [], None, None
        self.restoration, self.phone_terminal_status = None, None
        if self._bridge is not None:
            try:
                returncode = self._bridge.wait(timeout=60)
            except subprocess.TimeoutExpired:
                errors.append("USB bridge did not exit after the shutdown request")
        if self._bridge_log is not None and self._bridge_log.exists():
            try:
                terminal = self._parse_terminal(self._bridge_log.read_text(errors="backslashreplace").splitlines())
            except Exception as error:  # noqa: BLE001
                errors.append(str(error))
        if returncode not in (None, 0):
            errors.append(f"USB bridge exited {returncode}")
        if terminal is not None and terminal.calls != served_calls:
            errors.append(f"USB bridge served {terminal.calls} calls, the driver sent {served_calls}")
        errors += self._release(force_close=bool(errors))
        if self.phone_terminal_status not in (None, "0"):
            errors.append("phone session exit status " + str(self.phone_terminal_status))
        if errors:
            raise RuntimeError("; ".join(errors))
        return self._receipt("stop", {"bridge_returncode": returncode, "bridge_terminal": dict(terminal.values),
                                      "restoration": self.restoration, "phone_terminal_status":
                                      self.phone_terminal_status, "orphans_killed": self.orphans_killed,
                                      "served_calls": served_calls})
