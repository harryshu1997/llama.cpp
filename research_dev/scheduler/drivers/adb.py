"""One guarded way to run adb for the portable path.

On a shared host the adb server on a port is a shared resource: any client command starts a server when
none listens (under the caller's account, which locks the other users out of the phones they use), and a
server-control command ends whoever owns it. ``AdbGuard`` therefore

* checks that a server already listens on the port before every client call and refuses otherwise
  (``AdbUnavailable``; it never starts a server),
* refuses the commands that control the server or change a phone's mode or boot state
  (``FORBIDDEN_COMMANDS``): those stay a person's decision,
* always names a serial, so a call cannot land on another person's phone.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import shlex
import subprocess
from typing import Callable, Sequence

from ..adapters.contracts import PhysicalAdapterError


FORBIDDEN_COMMANDS = frozenset({
    "start-server", "kill-server", "reconnect", "root", "unroot", "usb", "tcpip", "reboot", "reboot-bootloader",
    "remount", "disable-verity", "enable-verity", "sideload", "install", "uninstall",
})


class AdbUnavailable(PhysicalAdapterError):
    """No adb server listens on the port; nothing was run."""


def tcp_listening(port: int, tables: Sequence[str] = ("/proc/net/tcp", "/proc/net/tcp6")) -> bool:
    """Whether a local socket listens on ``port`` (any address), read from the kernel's socket tables."""
    suffix = ":%04X" % port
    for name in tables:
        try:
            with open(name, encoding="ascii") as table:
                for line in table:
                    columns = line.split()
                    if len(columns) > 3 and columns[1].endswith(suffix) and columns[3] == "0A":
                        return True
        except OSError:
            continue
    return False


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            value.update(block)
    return "sha256:" + value.hexdigest()


class AdbGuard:
    def __init__(self, adb_path: Path, port: int, *, run: Callable = subprocess.run,
                 listening: Callable[[int], bool] = tcp_listening) -> None:
        if type(port) is not int or not 1 <= port <= 65535:
            raise PhysicalAdapterError("adb port is invalid")
        self.adb_path, self.port = Path(adb_path), port
        self._run, self._listening = run, listening

    def available(self) -> bool:
        return bool(self._listening(self.port))

    def _require_listener(self) -> None:
        if not self.available():
            raise AdbUnavailable(f"no adb server listens on port {self.port}; none is started here "
                                 "(the owner of the phones starts it)")

    def devices(self) -> list[tuple[str, str]]:
        """(serial, state) of every device the listening server reports."""
        self._require_listener()
        done = self._run([str(self.adb_path), "-P", str(self.port), "devices"], check=True, capture_output=True,
                         text=True, stdin=subprocess.DEVNULL, timeout=30)
        rows = []
        for line in done.stdout.splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 2:
                rows.append((fields[0], fields[1]))
        return rows

    def run(self, serial: str, *arguments: str, timeout_s: float = 60, check: bool = True):
        """``adb -P port -s serial <arguments>``; refuses server-control and mode-changing commands."""
        if not serial or not arguments:
            raise PhysicalAdapterError("adb call needs a serial and a command")
        if arguments[0] in FORBIDDEN_COMMANDS:
            raise PhysicalAdapterError(f"adb {arguments[0]} is not run by the drivers")
        self._require_listener()
        return self._run([str(self.adb_path), "-P", str(self.port), "-s", serial, *arguments], check=check,
                         capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout_s)

    def shell(self, serial: str, command: str, *, root: bool = False, timeout_s: float = 60,
              check: bool = True) -> str:
        """Output of one phone shell command; ``root`` runs it through ``su -c``."""
        text = "su -c " + shlex.quote(command) if root else command
        return self.run(serial, "shell", text, timeout_s=timeout_s, check=check).stdout

    def remote_sha256(self, serial: str, paths: Sequence[str], *, root: bool = False,
                      timeout_s: float = 900) -> dict[str, str]:
        """sha256 of phone files (absent files are left out)."""
        if not paths:
            return {}
        output = self.shell(serial, "sha256sum " + " ".join(shlex.quote(path) for path in paths) + " 2>/dev/null",
                            root=root, timeout_s=timeout_s, check=False)
        observed = {}
        for line in output.splitlines():
            digest, _, path = line.strip().partition("  ")
            if len(digest) == 64 and path:
                observed[path] = "sha256:" + digest
        return observed

    def push_verified(self, serial: str, local: Path, remote: str, sha256: str, *, allowed_roots: Sequence[str],
                      root: bool = False, timeout_s: float = 3600) -> str:
        """Copy a file to the phone unless it is already there with this digest; returns ``present`` or ``pushed``.

        ``remote`` must lie under one of ``allowed_roots`` (our own directories on the phone): nothing is
        ever written elsewhere.
        """
        if not any(remote.startswith(base.rstrip("/") + "/") for base in allowed_roots):
            raise PhysicalAdapterError(f"{remote} is outside this site's directories on the phone")
        if sha256_file(local) != sha256:
            raise PhysicalAdapterError(f"{local} differs from its pinned sha256")
        if self.remote_sha256(serial, [remote], root=root).get(remote) == sha256:
            return "present"
        self.shell(serial, "mkdir -p " + shlex.quote(remote.rsplit("/", 1)[0]), timeout_s=30)
        self.run(serial, "push", str(local), remote, timeout_s=timeout_s)
        if self.remote_sha256(serial, [remote], root=root).get(remote) != sha256:
            raise PhysicalAdapterError(f"{remote} differs from its sha256 after the push")
        return "pushed"
