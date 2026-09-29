"""Run the 74-request F16 trace plus ten-request Llama overlay."""

import argparse
from pathlib import Path

from .common import CONFIRMATION, require


def host_dependency(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    require(
        separator == "="
        and bool(name)
        and name.isascii()
        and ":" not in name
        and bool(raw_path),
        "transport host dependency",
    )
    return name, Path(raw_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large-requests", type=Path, required=True)
    parser.add_argument("--overlay-requests", type=Path, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--capability-catalog", type=Path, required=True)
    parser.add_argument("--observation-store-input", type=Path)
    parser.add_argument("--observation-source-catalog", type=Path)
    parser.add_argument("--adaptive-observation-store-input", type=Path)
    parser.add_argument(
        "--adaptive-observation-source-catalog", type=Path
    )
    parser.add_argument("--qwen-manifest", type=Path, required=True)
    parser.add_argument("--gemma-manifest", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--gemma-model", type=Path, required=True)
    parser.add_argument("--llama-model", type=Path, required=True)
    parser.add_argument("--gguf-manifest-cache", type=Path)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument(
        "--transport-host-dependency",
        type=host_dependency,
        action="append",
        default=[],
    )
    parser.add_argument("--resident-server", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--resident-lib-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--close-helper", type=Path, required=True)
    parser.add_argument("--adb", type=Path, required=True)
    parser.add_argument("--phone-usb-close", type=Path, required=True)
    parser.add_argument("--phone-session", required=True)
    parser.add_argument("--phone-restore", required=True)
    parser.add_argument("--phone-worker", required=True)
    parser.add_argument("--phone-resident-workers")
    parser.add_argument("--phone-resident-router")
    parser.add_argument("--phone-multi-session-port-base", type=int)
    parser.add_argument("--phone-busybox", required=True)
    parser.add_argument("--qwen-phone-model", required=True)
    parser.add_argument("--gemma-phone-model", required=True)
    parser.add_argument(
        "--qwen-ffn-shards",
        default=None,
        metavar="LOCAL_FFN_SHARDS.json=PHONE_DIR",
        help="offline FFN shard index for the Qwen phone shards",
    )
    parser.add_argument(
        "--gemma-ffn-shards",
        default=None,
        metavar="LOCAL_FFN_SHARDS.json=PHONE_DIR",
        help="offline FFN shard index for the Gemma phone shards",
    )
    parser.add_argument("--phone-session-root", required=True)
    parser.add_argument("--phone-whole-server")
    parser.add_argument("--phone-whole-library-directory")
    parser.add_argument("--phone-whole-model")
    parser.add_argument("--phone-whole-state-directory")
    parser.add_argument("--phone-whole-control-transport", choices=("adb-usb", "adb-ncm"), default="adb-usb")
    parser.add_argument("--phone-whole-ncm-adb-endpoint")
    parser.add_argument(
        "--phone-whole-executable-device", default="GPUOpenCL"
    )
    parser.add_argument("--phone-remote-hash-cache", type=Path)
    parser.add_argument("--phone-diagnostic-endpoint", required=True)
    parser.add_argument("--phone-battery-ppm", type=int, required=True)
    parser.add_argument("--phone-usb-serial", required=True)
    parser.add_argument("--phone-android-gadget", required=True)
    parser.add_argument("--phone-functionfs-gadget", required=True)
    parser.add_argument("--phone-functionfs-root", required=True)
    parser.add_argument("--phone-usb-controller", required=True)
    parser.add_argument("--adb-port", type=int, required=True)
    parser.add_argument("--minimum-usb-speed-mbps", type=int, default=5000)
    parser.add_argument("--phone-kernel-release", required=True)
    parser.add_argument("--phone-boot-image-sha256")
    parser.add_argument("--usb-qualification-identity", type=Path)
    parser.add_argument("--nmcli", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection-mode",
        choices=(
            "adaptive-decode",
            "calibration",
            "deadline-first",
            "desktop-baseline",
            "energy-aware",
        ),
        required=True,
    )
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--request-indices")
    parser.add_argument("--arrival-scale", type=int)
    parser.add_argument("--replay-schedule", type=Path)
    parser.add_argument("--adaptive-minimum-remaining-tokens", type=int)
    parser.add_argument("--maximum-phone-sessions", type=int)
    parser.add_argument("--fixed-phone-residency-json")
    parser.add_argument("--include-startup-preparation", action="store_true")
    parser.add_argument(
        "--helper-preparation-fault-injection",
        choices=("post-load-once",),
    )
    parser.add_argument(
        "--energy-attribution-kind",
        choices=("diagnostic", "isolated", "matched_abba"),
        default="diagnostic",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    return parser


def _validate_arguments(args: argparse.Namespace) -> dict[str, Path]:
    require(
        args.execute and args.confirm == CONFIRMATION,
        "physical execution confirmation",
    )
    require(
        args.output.is_absolute()
        and not args.output.exists()
        and args.max_workers > 0,
        "physical output",
    )
    require(
        args.adaptive_minimum_remaining_tokens is None
        or args.adaptive_minimum_remaining_tokens > 0,
        "adaptive minimum remaining tokens",
    )
    require(
        args.maximum_phone_sessions is None
        or args.maximum_phone_sessions > 0,
        "maximum phone sessions",
    )
    require(
        args.arrival_scale is None or args.arrival_scale > 0,
        "arrival scale",
    )
    require(
        args.replay_schedule is None
        or (
            args.request_indices is None
            and args.arrival_scale is None
        ),
        "named replay schedule cannot be combined with replay transforms",
    )
    for path in (
        args.large_requests,
        args.overlay_requests,
        args.trace_manifest,
        args.capability_catalog,
        args.qwen_manifest,
        args.gemma_manifest,
        args.qwen_model,
        args.gemma_model,
        args.llama_model,
        args.server,
        args.resident_server,
        args.cuda_lib_dir,
        args.resident_lib_dir,
        args.bridge,
        args.close_helper,
        args.adb,
        args.phone_usb_close,
        *(() if args.replay_schedule is None else (args.replay_schedule,)),
    ):
        require(path.exists(), f"physical dependency: {path}")
    if args.source_manifest is not None:
        require(args.source_manifest.is_file(), "physical source manifest")
    if args.observation_store_input is not None:
        require(
            args.observation_store_input.is_file(),
            "physical observation store input",
        )
    if args.adaptive_observation_store_input is not None:
        require(
            args.adaptive_observation_store_input.is_file(),
            "physical adaptive observation store input",
        )
    if args.usb_qualification_identity is not None:
        require(
            args.usb_qualification_identity.is_file()
            and type(args.phone_boot_image_sha256) is str,
            "physical transport qualification identity",
        )
    else:
        require(
            args.phone_boot_image_sha256 is None,
            "physical boot identity lacks transport qualification",
        )
    host_dependencies = dict(args.transport_host_dependency)
    require(
        len(host_dependencies) == len(args.transport_host_dependency)
        and all(path.is_file() for path in host_dependencies.values()),
        "transport host dependencies",
    )
    return host_dependencies
