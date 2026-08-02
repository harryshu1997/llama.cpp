#!/usr/bin/env python3
"""A6000-side identity and telemetry producer for phone routes."""

from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any

from a6000_phone_route_control import (
    PhoneControl,
    load_config,
    remove_durable,
    write_durable_new,
)
from phone_gateway import (
    GatewayError,
    canonical_bytes,
    integer,
    require,
    string,
)


MIN_AVAILABLE_BYTES = 512 * 1024 * 1024
RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
THERMAL_STATUS = re.compile(r"^\s*Thermal Status:\s*(\d+)\s*$")


def parse_meminfo(raw: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].endswith(":"):
            try:
                values[fields[0][:-1]] = int(fields[1]) * 1024
            except ValueError:
                continue
    require(
        all(key in values for key in ("MemAvailable", "SwapFree", "SwapTotal")),
        "phone memory fields",
    )
    require(
        values["SwapFree"] <= values["SwapTotal"],
        "phone swap counters",
    )
    return {
        "available_bytes": values["MemAvailable"],
        "swap_total_bytes": values["SwapTotal"],
        "swap_used_bytes": values["SwapTotal"] - values["SwapFree"],
    }


def parse_temperatures(raw: str) -> list[dict[str, Any]]:
    result = []
    names = set()
    for line in raw.splitlines():
        if "|" not in line:
            continue
        name, value = line.rsplit("|", 1)
        name = string(name.strip(), "thermal sensor name")
        require(name not in names, "duplicate thermal sensor")
        names.add(name)
        try:
            temperature = int(value)
        except ValueError as error:
            raise GatewayError(f"thermal sensor value: {error}") from error
        require(
            -100_000 <= temperature <= 300_000,
            "thermal sensor range",
        )
        result.append({
            "name": name,
            "temp_millic": temperature,
        })
    result.sort(key=lambda row: row["name"])
    require(result, "phone thermal sensors are empty")
    return result


def parse_thermal_status(raw: str) -> int:
    values = [
        int(match.group(1))
        for line in raw.splitlines()
        if (match := THERMAL_STATUS.fullmatch(line)) is not None
    ]
    require(len(values) == 1, "phone thermal status")
    return values[0]


def parse_interfaces(
    netdev_raw: str,
    addresses_raw: str,
) -> dict[str, dict[str, Any]]:
    counters: dict[str, tuple[int, int]] = {}
    for line in netdev_raw.splitlines():
        if ":" not in line:
            continue
        name, values = line.split(":", 1)
        fields = values.split()
        if len(fields) != 16:
            continue
        try:
            counters[name.strip()] = (int(fields[0]), int(fields[8]))
        except ValueError:
            continue
    addresses: dict[str, list[str]] = {}
    for line in addresses_raw.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[2] != "inet":
            continue
        name = fields[1]
        address = str(ipaddress.IPv4Interface(fields[3]).ip)
        addresses.setdefault(name, []).append(address)
    result = {}
    for name in sorted(set(counters) & set(addresses)):
        rx_bytes, tx_bytes = counters[name]
        result[name] = {
            "ipv4": sorted(set(addresses[name])),
            "rx_bytes": rx_bytes,
            "tx_bytes": tx_bytes,
        }
    require(result, "phone network interfaces are empty")
    return result


def parse_tcp(raw: str) -> list[dict[str, Any]]:
    result = []
    for line in raw.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 10 or fields[3] != "01":
            continue
        try:
            local_hex, local_port = fields[1].split(":")
            remote_hex, remote_port = fields[2].split(":")
            local_address = str(
                ipaddress.IPv4Address(bytes.fromhex(local_hex)[::-1])
            )
            remote_address = str(
                ipaddress.IPv4Address(bytes.fromhex(remote_hex)[::-1])
            )
            row = {
                "local_address": local_address,
                "local_port": int(local_port, 16),
                "remote_address": remote_address,
                "remote_port": int(remote_port, 16),
                "socket_inode": int(fields[9]),
            }
        except (ValueError, ipaddress.AddressValueError) as error:
            raise GatewayError(f"phone TCP table: {error}") from error
        if row["local_port"] > 0 and row["remote_port"] > 0:
            result.append(row)
    result.sort(
        key=lambda row: (
            row["local_address"],
            row["local_port"],
            row["remote_address"],
            row["remote_port"],
            row["socket_inode"],
        )
    )
    return result


def reciprocal_peer(
    op12: dict[str, Any],
    op15: dict[str, Any],
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    op12_addresses = {
        address
        for row in op12["interfaces"].values()
        for address in row["ipv4"]
    }
    op15_addresses = {
        address
        for row in op15["interfaces"].values()
        for address in row["ipv4"]
    }
    op15_sockets = {
        (
            row["local_address"],
            row["local_port"],
            row["remote_address"],
            row["remote_port"],
        ): row["socket_inode"]
        for row in op15["tcp_established"]
    }
    matches = []
    for row in op12["tcp_established"]:
        forward = (
            row["local_address"],
            row["local_port"],
            row["remote_address"],
            row["remote_port"],
        )
        reverse = (
            row["remote_address"],
            row["remote_port"],
            row["local_address"],
            row["local_port"],
        )
        if (
            row["local_address"] in op12_addresses
            and row["remote_address"] in op15_addresses
            and reverse in op15_sockets
        ):
            matches.append((
                forward,
                row["socket_inode"],
                op15_sockets[reverse],
            ))
    require(matches, "no reciprocal OP12-OP15 TCP peer")
    if expected is None:
        (
            (local_address, local_port, remote_address, remote_port),
            op12_inode,
            op15_inode,
        ) = sorted(matches)[0]
        result = {
            "op12_address": local_address,
            "op12_port": local_port,
            "op15_address": remote_address,
            "op15_port": remote_port,
            "op12_socket_inode": op12_inode,
            "op15_socket_inode": op15_inode,
            "schema": "s40-phone-direct-peer-v2",
        }
    else:
        required = (
            expected["op12_address"],
            expected["op12_port"],
            expected["op15_address"],
            expected["op15_port"],
        )
        selected = [row for row in matches if row[0] == required]
        require(len(selected) == 1, "expected OP12-OP15 TCP peer is not active")
        result = {
            **expected,
            "op12_socket_inode": selected[0][1],
            "op15_socket_inode": selected[0][2],
            "schema": "s40-phone-direct-peer-v2",
        }
    return result


class PhoneObserver:
    def __init__(self, control: PhoneControl):
        self.control = control

    def process_snapshot(
        self,
        phone: dict[str, Any],
        record: dict[str, Any],
    ) -> dict[str, Any]:
        serial = phone["serial"]
        pid = integer(record["pid"], "phone process PID", 1)
        require(
            record["lifecycle"] == "LIVE"
            and record["ready"] is True
            and self.control.adb(
                serial,
                "shell",
                f"kill -0 {pid}",
                check=False,
            ).returncode == 0,
            "phone process is not live",
        )
        start_ticks = self.control.process_start_ticks(serial, pid)
        cmdline_sha256 = self.control.process_cmdline_sha256(serial, pid)
        require(
            start_ticks == record["start_ticks"]
            and cmdline_sha256 == record["cmdline_sha256"],
            "phone process identity changed",
        )
        raw = self.control.text(
            serial,
            (
                f"for f in /proc/{pid}/fd/*; do "
                "readlink \"$f\" 2>/dev/null || true; done"
            ),
        )
        socket_inodes = sorted({
            int(match.group(1))
            for line in raw.splitlines()
            if (match := re.fullmatch(r"socket:\[(\d+)\]", line)) is not None
        })
        return {
            "argv": list(record["argv"]),
            "artifact_path": record["artifact_path"],
            "artifact_role": record["artifact_role"],
            "artifact_sha256": record["artifact_sha256"],
            "backend": record["backend"],
            "cmdline_sha256": cmdline_sha256,
            "env": record["env"],
            "head_host": record.get("head_host"),
            "head_port": record.get("head_port"),
            "kind": record["kind"],
            "layer_end": record["layer_end"],
            "layer_start": record["layer_start"],
            "listen_port": record["listen_port"],
            "name": record["name"],
            "pid": pid,
            "process_start_ticks": start_ticks,
            "socket_inodes": socket_inodes,
            "tail_host": record.get("tail_host"),
            "tail_port": record.get("tail_port"),
            "tail_source_port": record["tail_source_port"],
        }

    def snapshot(
        self,
        phone_name: str,
        phone: dict[str, Any],
        state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        serial = phone["serial"]
        started_ns = time.monotonic_ns()
        require(
            self.control.adb(serial, "get-state").stdout.decode(
                "ascii"
            ).strip() == "device",
            f"{serial} is not ready",
        )
        boot_id = self.control.text(
            serial,
            "cat /proc/sys/kernel/random/boot_id",
        )
        require(boot_id == phone["boot_id"], f"{serial} boot changed")
        memory = parse_meminfo(
            self.control.text(serial, "cat /proc/meminfo")
        )
        thermal_status = parse_thermal_status(
            self.control.text(serial, "dumpsys thermalservice")
        )
        temperatures = parse_temperatures(
            self.control.text(
                serial,
                (
                    "for z in /sys/class/thermal/thermal_zone*; do "
                    "n=$(cat \"$z/type\" 2>/dev/null) || continue; "
                    "t=$(cat \"$z/temp\" 2>/dev/null) || continue; "
                    "printf '%s|%s\\n' \"$n\" \"$t\"; done"
                ),
            )
        )
        interfaces = parse_interfaces(
            self.control.text(serial, "cat /proc/net/dev"),
            self.control.text(serial, "ip -o -4 addr show"),
        )
        tcp = parse_tcp(
            self.control.text(serial, "cat /proc/net/tcp")
        )
        processes = (
            []
            if state is None
            else [
                self.process_snapshot(phone, record)
                for record in state["processes"][phone_name]
            ]
        )
        runtime_files = []
        for role, row in sorted(phone["runtime_files"].items()):
            current = self.control.remote_stat(serial, row["path"])
            require(
                current == row["stat"],
                f"{serial} runtime stat changed: {role}",
            )
            runtime_files.append({
                "path": row["path"],
                "role": role,
                "sha256": row["sha256"],
                "stat": current,
            })
        result = {
            **memory,
            "boot_id": boot_id,
            "completed_ns": time.monotonic_ns(),
            "device": self.control.text(
                serial,
                "getprop ro.product.device",
            ),
            "interfaces": interfaces,
            "model": self.control.text(
                serial,
                "getprop ro.product.model",
            ),
            "product": self.control.text(
                serial,
                "getprop ro.product.name",
            ),
            "processes": processes,
            "runtime_files": runtime_files,
            "schema": "s40-phone-runtime-snapshot-v1",
            "serial": serial,
            "started_ns": started_ns,
            "tcp_established": tcp,
            "temperatures": temperatures,
            "thermal_status": thermal_status,
        }
        require(
            result["available_bytes"] >= MIN_AVAILABLE_BYTES,
            f"{serial} memory headroom",
        )
        require(
            result["swap_used_bytes"] == 0,
            f"{serial} swap is in use",
        )
        require(
            result["thermal_status"] == 0,
            f"{serial} thermal status",
        )
        return result

    def active_state(self, model_id: str | None = None) -> dict[str, Any]:
        state = self.control._read_journal()
        require(
            state["state"] == "ACTIVE"
            and (model_id is None or state["model_id"] == model_id),
            "phone route is not active",
        )
        return state

    def observe_route(
        self,
        state: dict[str, Any],
        phones: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model_id = state["model_id"]
        route = self.control.config["routes"].get(model_id)
        require(
            route is not None
            and state["route_instance_id"]
            and state["state"] == "ACTIVE"
            and self.control.config["dependency_package"] is not None
            and self.control.config["dependency_package_sha256"] is not None
            and "direct_peer" in route,
            "phone observer requires a versioned dependency lock",
        )
        if phones is None:
            phones = {
                name: self.snapshot(name, route[name], state)
                for name in ("op12", "op15")
            }
        for name in phones:
            require(
                phones[name]["boot_id"]
                == state["identity_snapshots"][name]["boot_id"],
                "phone observer boot differs from route journal",
            )
        direct_peer = reciprocal_peer(
            phones["op12"],
            phones["op15"],
            route["direct_peer"],
        )
        op12_tail = [
            row for row in phones["op12"]["processes"]
            if row["kind"] == "STAGE_TAIL"
        ]
        op15_relay = [
            row for row in phones["op15"]["processes"]
            if row["kind"] == "DIRECT_RELAY"
        ]
        require(
            len(op12_tail) == 1
            and len(op15_relay) == 1
            and direct_peer["op12_socket_inode"]
            in op12_tail[0]["socket_inodes"]
            and direct_peer["op15_socket_inode"]
            in op15_relay[0]["socket_inodes"],
            "direct peer is not owned by the frozen phone processes",
        )
        return {
            "direct_peer": direct_peer,
            "model_id": model_id,
            "phones": phones,
            "route_instance_id": state["route_instance_id"],
            "schema": "s40-phone-route-observation-v1",
        }

    def identity(self, model_id: str) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        state = self.active_state(model_id)
        observation = self.observe_route(state)
        return {
            "a6000_identity": self.control.config["a6000_identity"],
            "completed_ns": time.monotonic_ns(),
            "direct_peer": observation["direct_peer"],
            "model_id": model_id,
            "phones": observation["phones"],
            "remote_identity_package":
                self.control.config["dependency_package"],
            "remote_identity_package_sha256":
                self.control.config["dependency_package_sha256"],
            "route_instance_id": state["route_instance_id"],
            "schema": "s40-phone-identity-v2",
            "started_ns": started_ns,
        }

    def current_state(self) -> dict[str, Any] | None:
        try:
            return self.active_state()
        except FileNotFoundError:
            return None

    def stop_path(self, run_id: str) -> Path:
        require(RUN_ID.fullmatch(run_id) is not None, "telemetry run ID")
        return self.control.config["state_dir"] / f"telemetry-{run_id}.stop"

    def stop(self, run_id: str) -> dict[str, Any]:
        path = self.stop_path(run_id)
        if not path.exists():
            write_durable_new(
                path,
                {
                    "run_id": run_id,
                    "schema": "s40-phone-telemetry-stop-v1",
                },
            )
        return {
            "run_id": run_id,
            "schema": "s40-phone-telemetry-stop-v1",
            "success": True,
        }

    def telemetry(
        self,
        model_id: str,
        run_id: str,
        interval_ms: int,
    ) -> int:
        require(100 <= interval_ms <= 10_000, "telemetry interval")
        identity = self.identity(model_id)
        stop_path = self.stop_path(run_id)
        if stop_path.exists():
            remove_durable(stop_path)
        stopping = False

        def request_stop(_signum, _frame):
            nonlocal stopping
            stopping = True

        previous_term = signal.signal(signal.SIGTERM, request_stop)
        previous_int = signal.signal(signal.SIGINT, request_stop)
        index = 0
        try:
            header = {
                "identity": identity,
                "interval_ms": interval_ms,
                "run_id": run_id,
                "schema": "s40-phone-telemetry-header-v1",
            }
            sys.stdout.buffer.write(canonical_bytes(header))
            sys.stdout.buffer.flush()
            while not stopping and not stop_path.exists():
                started_ns = time.monotonic_ns()
                state = self.current_state()
                route = self.control.config["routes"][
                    model_id if state is None else state["model_id"]
                ]
                phones = {
                    name: self.snapshot(name, route[name], state)
                    for name in ("op12", "op15")
                }
                active_route = (
                    None
                    if state is None
                    else self.observe_route(state, phones)
                )
                if active_route is not None:
                    require(
                        active_route["phones"] == phones,
                        "phone route observation changed within one sample",
                    )
                    active_route = {
                        key: value
                        for key, value in active_route.items()
                        if key != "phones"
                    }
                index += 1
                sample = {
                    "active_route": active_route,
                    "completed_ns": time.monotonic_ns(),
                    "phones": phones,
                    "run_id": run_id,
                    "sample_index": index,
                    "schema": "s40-phone-telemetry-sample-v2",
                    "started_ns": started_ns,
                }
                sys.stdout.buffer.write(canonical_bytes(sample))
                sys.stdout.buffer.flush()
                deadline = time.monotonic() + interval_ms / 1000
                while (
                    not stopping
                    and not stop_path.exists()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.05)
            state = self.current_state()
            route = self.control.config["routes"][
                model_id if state is None else state["model_id"]
            ]
            phones = {
                name: self.snapshot(name, route[name], state)
                for name in ("op12", "op15")
            }
            active_route = (
                None
                if state is None
                else self.observe_route(state, phones)
            )
            if active_route is not None:
                require(
                    active_route["phones"] == phones,
                    "phone route observation changed within footer",
                )
                active_route = {
                    key: value
                    for key, value in active_route.items()
                    if key != "phones"
                }
            footer = {
                "active_route": active_route,
                "completed_ns": time.monotonic_ns(),
                "phones": phones,
                "run_id": run_id,
                "sample_count": index,
                "schema": "s40-phone-telemetry-footer-v2",
                "started_ns": min(
                    row["started_ns"] for row in phones.values()
                ),
            }
            sys.stdout.buffer.write(canonical_bytes(footer))
            sys.stdout.buffer.flush()
            return 0
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)
            if stop_path.exists():
                remove_durable(stop_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("identity", "telemetry", "stop"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=1000)
    parser.add_argument("--model")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    require(args.config.is_absolute(), "observer config path")
    observer = PhoneObserver(PhoneControl(load_config(args.config)))
    if args.action == "identity":
        require(args.model is not None and args.run_id is None, "identity args")
        sys.stdout.buffer.write(canonical_bytes(observer.identity(args.model)))
        sys.stdout.buffer.flush()
        return 0
    if args.action == "telemetry":
        require(
            args.model is not None and args.run_id is not None,
            "telemetry args",
        )
        return observer.telemetry(
            args.model,
            args.run_id,
            args.interval_ms,
        )
    require(
        args.model is None and args.run_id is not None,
        "telemetry stop args",
    )
    sys.stdout.buffer.write(canonical_bytes(observer.stop(args.run_id)))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GatewayError, OSError, ValueError) as error:
        print(f"A6000 phone observer failed: {error}", file=sys.stderr)
        raise SystemExit(2)
