#!/usr/bin/env python3
"""Generality traces: independent BurstGPT windows and arrival-density variants of longtail_eval_v2.

    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.generality_traces build \\
        --burstgpt-csv burstgpt_3.csv --codec build-cuda/bin/llama-token-codec --library-dir build-cuda/bin \\
        --qwen-tokenizer-model Qwen3-14B-Q4_K_M.gguf --gemma-tokenizer-model gemma-4-12B-it-Q4_0.gguf \\
        --llama-tokenizer-model Llama-3.2-1B-Instruct-Q4_0.gguf --out-dir OUT [--variants w2,w3,...]
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.generality_traces verify --out-dir OUT
    python3 -m research_dev.scheduler.campaigns.burstgpt.tools.generality_traces plan

Every variant is one run of build_realistic_trace.py with longtail_eval_v2's recorded rules (BASE_RULES,
from its TRACE_MANIFEST derivation block and talks.md 2026-09-25 14:21 UTC) plus exactly one change:

* window variants (w*): a later qualifying window of the same scan (``--window-rank``), or the first
  qualifying windows of the scan grid shifted by half a window (``--scan-phase-s 900``) because the
  unshifted grid has only three qualifying windows in the whole log. Every numeric rule is unchanged;
  ``verify`` checks that no two windows (base included) share a second of source time.
* density variants (d*): the base window with ``--arrival-scale`` 0.5 / 0.25 (2x / 4x denser) or 2.0
  (half the density). Rows, prompts and outputs are those of the base; only arrivals and ids move.

``build`` first rebuilds longtail_eval_v2 itself and refuses to continue unless it is byte-identical to the
frozen copy (``paper_config_v1/trace``) after the manifest paths are relocated: that proves the local codec
and tokenizers reproduce the desktop derivation. Built files record the future desktop location
(``--desktop-root``/<name>/, CSV at ``--desktop-csv``), so they are the bytes a desktop build would write.
For each variant it also writes a campaign template: ``paper_config_v1/template`` with only the trace paths
changed. Nothing here touches the desktop; ``COPY_TO_DESKTOP.sh`` lists what must be copied there.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from ..build_realistic_trace import canonical, digest_file, long_tail_stats
from ..common import load_object
from ..trace_inputs import (
    GEMMA_ROLE, QWEN_ROLE, REPLAY_START_US, TRACE_HOT_MODEL_ID, apply_named_replay_schedule, merge_rows,
    validate_trace,
)

SCHEMA = "ws6-generality-traces-v1"
BUILDER_MODULE = "research_dev.scheduler.campaigns.burstgpt.build_realistic_trace"
REPO_ROOT = Path(__file__).resolve().parents[5]
PAPER_CONFIG = Path(__file__).resolve().parents[1] / "paper_config_v1"
BASE_NAME = "longtail_eval_v2"
DESKTOP_ROOT = "/mnt/storage/burstgpt-source"
DESKTOP_CSV = DESKTOP_ROOT + "/burstgpt_3.csv"
DESKTOP_TEMPLATE_ROOT = "/mnt/storage/s43-two-phone-eval-20260925"
DESKTOP_HOST = "zhihao@172.20.74.85"
TEMPLATE_PREFIX = "template-eval2-s2-"
TRACE_FILES = ("REQUESTS_SEMANTIC_SOURCE.jsonl", "REQUESTS_OVERLAY.jsonl", "TRACE_MANIFEST.json")
TEMPLATE_FILES = ("campaign.json", "rig.json", "models.json", "evidence.json", "CHANGES.txt")
TRACE_PATH_KEYS = ("large_requests_path", "overlay_requests_path", "replay_schedule_path", "trace_manifest_path")
# longtail_eval_v2's derivation (TRACE_MANIFEST.json "derivation" + talks.md 2026-09-25 14:21 UTC)
BASE_RULES = (
    "--small-model-share", "0.15", "--duration-s", "1800", "--min-requests", "14", "--max-requests", "18",
    "--prompt-cap", "2048", "--output-cap", "1100", "--min-input", "16", "--min-output", "4",
    "--long-tail-threshold", "512", "--long-tail-tolerance", "0.05", "--max-output-tokens", "4000",
    "--min-requests-per-model", "4",
)


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str  # base | window | density
    note: str
    window_rank: int = 1
    scan_phase_s: float = 0.0
    arrival_scale: float = 1.0
    spare: bool = False

    @property
    def trace_name(self) -> str:
        return "burstgpt_" + self.name

    @property
    def template_name(self) -> str:
        return TEMPLATE_PREFIX + self.name

    def builder_flags(self) -> tuple[str, ...]:
        flags: list[str] = []
        if self.window_rank != 1:
            flags += ["--window-rank", str(self.window_rank)]
        if self.scan_phase_s:
            flags += ["--scan-phase-s", repr(self.scan_phase_s)]
        if self.arrival_scale != 1.0:
            flags += ["--arrival-scale", repr(self.arrival_scale)]
        return tuple(flags)

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "note": self.note, "window_rank": self.window_rank,
                "scan_phase_s": self.scan_phase_s, "arrival_scale": self.arrival_scale, "spare": self.spare,
                "trace_name": self.trace_name, "template_name": self.template_name}


BASE = Variant(BASE_NAME, "base", "the frozen evaluation trace (rebuilt only to prove the derivation reproduces)")
VARIANTS = (
    Variant(BASE_NAME + "_w2", "window", "2nd qualifying window of the base scan grid", window_rank=2),
    Variant(BASE_NAME + "_w3", "window", "3rd (last) qualifying window of the base scan grid", window_rank=3),
    Variant(BASE_NAME + "_w4", "window", "1st qualifying window of the half-shifted grid (+900 s)",
            scan_phase_s=900.0),
    Variant(BASE_NAME + "_w5", "window", "2nd qualifying window of the half-shifted grid (+900 s); spare",
            window_rank=2, scan_phase_s=900.0, spare=True),
    Variant(BASE_NAME + "_d2x", "density", "base window, inter-arrival times x0.5 (2x denser)", arrival_scale=0.5),
    Variant(BASE_NAME + "_d4x", "density", "base window, inter-arrival times x0.25 (4x denser)", arrival_scale=0.25),
    Variant(BASE_NAME + "_d0p5x", "density", "base window, inter-arrival times x2 (half the density)",
            arrival_scale=2.0),
)
VARIANT_BY_NAME = {row.name: row for row in (BASE, *VARIANTS)}


class GeneralityError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise GeneralityError(message)


def builder_argv(variant: Variant, *, csv: Path, output_dir: Path, tools: Mapping[str, Path],
                 inventory_from: Path, rules: Sequence[str] = BASE_RULES) -> list[str]:
    """build_realistic_trace.py arguments of one variant (tools: codec, library_dir, qwen, gemma, llama)."""
    return [
        "--burstgpt-csv", str(csv), "--output-dir", str(output_dir), "--trace-name", variant.trace_name,
        "--codec", str(tools["codec"]), "--library-dir", str(tools["library_dir"]),
        "--qwen-tokenizer-model", str(tools["qwen"]), "--gemma-tokenizer-model", str(tools["gemma"]),
        "--llama-tokenizer-model", str(tools["llama"]),
        *rules, *variant.builder_flags(), "--model-inventory-from", str(inventory_from),
    ]


def relocate_manifest(manifest: Mapping[str, Any], trace_dir: str, csv_path: str) -> dict[str, Any]:
    """The manifest a build with --output-dir trace_dir and --burstgpt-csv csv_path writes (paths only)."""
    result = json.loads(json.dumps(manifest))
    for key, name in (("base_trace", TRACE_FILES[0]), ("overlay_trace", TRACE_FILES[1])):
        need(type(result.get(key)) is dict and Path(result[key].get("path", "")).name == name,
             "trace manifest " + key + " path")
        result[key]["path"] = trace_dir.rstrip("/") + "/" + name
    need(type(result.get("derivation")) is dict and "burstgpt_csv" in result["derivation"],
         "trace manifest derivation")
    result["derivation"]["burstgpt_csv"] = csv_path
    return result


def trace_file_names(trace_name: str) -> tuple[str, ...]:
    return (*TRACE_FILES, trace_name + ".json", trace_name + ".md")


def write_sha256sums(directory: Path, names: Sequence[str]) -> dict[str, str]:
    sums = {name: digest_file(directory / name) for name in sorted(names)}
    (directory / "SHA256SUMS").write_text("".join(f"{value}  {name}\n" for name, value in sums.items()))
    return sums


def load_trace(directory: Path, trace_name: str) -> dict[str, Any]:
    """Load and validate a trace dir exactly as the campaign loader does (manifest identity, merge, schedule)."""
    manifest = load_object(directory / "TRACE_MANIFEST.json")
    large, overlay = validate_trace(directory / TRACE_FILES[0], directory / TRACE_FILES[1], manifest)
    merged = merge_rows(large, overlay, {QWEN_ROLE: "hot-model", GEMMA_ROLE: "cold-model"})
    schedule = load_object(directory / (trace_name + ".json"))
    need(schedule.get("trace_name") == trace_name, "replay schedule trace name")
    selected, replay = apply_named_replay_schedule(merged, schedule)
    need(len(selected) == len(merged), "replay schedule covers every request")
    return {"manifest": manifest, "large": large, "overlay": overlay, "merged": merged, "replay": replay}


def trace_summary(trace: Mapping[str, Any]) -> dict[str, Any]:
    rows = [item["row"] for item in trace["merged"]]
    manifest = trace["manifest"]
    window = manifest["derivation"]["window"]
    start = window["window_start_source_s"]
    large = trace["large"]
    replay = trace["replay"]
    source_rows = [{"output": row["source_output_tokens"]} for row in rows]
    tail = long_tail_stats(source_rows, 512)
    return {
        "requests": len(rows),
        "qwen_hot": sum(1 for row in large if row["model_id"] == TRACE_HOT_MODEL_ID),
        "gemma_cold": sum(1 for row in large if row["model_id"] != TRACE_HOT_MODEL_ID),
        "llama_overlay": len(trace["overlay"]),
        "source_models": {name: sum(1 for row in rows if row["source_model"] == name) for name in ("ChatGPT", "GPT-4")},
        "input_tokens": manifest["combined_work"]["input_tokens"],
        "output_tokens": manifest["combined_work"]["output_tokens"],
        "max_output_tokens_per_request": max(row["output_tokens"] for row in rows),
        "clipped": manifest["derivation"]["clipped"],
        "arrival_scale": manifest["derivation"]["arrival_scale"],
        "replay_span_s": replay["replay_span_us"] / 1e6,
        "window_start_source_s": start,
        "window_end_source_s": start + window["window_duration_s"],
        "long_tail_request_share": tail["request_share"],
        "long_tail_output_token_share": tail["output_token_share"],
        "long_tail_log": {key: window["long_tail"]["log"][key] for key in ("request_share", "output_token_share")},
        "window_rank": window.get("window_rank", {}).get("rank", 1),
        "scan_phase_s": window.get("scan_phase_s", 0.0),
    }


def check_density(base: Mapping[str, Any], variant: Mapping[str, Any], scale: float, base_name: str,
                  variant_name: str) -> None:
    """A density variant is the base with scaled arrivals: same rows, prompts, outputs and merge order."""
    moved = {"arrival_us", "event_id"}
    for kind in ("large", "overlay"):
        need(len(base[kind]) == len(variant[kind]), f"density {kind} row count")
        for a, b in zip(base[kind], variant[kind]):
            need({k: v for k, v in a.items() if k not in moved} == {k: v for k, v in b.items() if k not in moved},
                 f"density {kind} row differs beyond arrival and id: {a['event_id']}")
            need(b["event_id"] == a["event_id"].replace(base_name, variant_name, 1), "density event id")
            need(b["arrival_us"] - REPLAY_START_US == round((a["arrival_us"] - REPLAY_START_US) * scale),
                 f"density arrival of {a['event_id']}")
    need([item["combined_index"] for item in base["merged"]]
         == [item["combined_index"] for item in variant["merged"]], "density merge order")
    need({k: v for k, v in base["manifest"]["derivation"]["window"].items()}
         == variant["manifest"]["derivation"]["window"], "density variant window differs from the base")


def check_disjoint(windows: Mapping[str, tuple[float, float]]) -> None:
    rows = sorted(windows.items(), key=lambda item: item[1][0])
    for (name_a, (_, end_a)), (name_b, (start_b, _)) in zip(rows, rows[1:]):
        need(end_a <= start_b, f"windows {name_a} and {name_b} overlap in source time")


def rewrite_template_text(text: str, name: str) -> str:
    old_dir = f"{DESKTOP_ROOT}/{BASE_NAME}/"
    new_dir = f"{DESKTOP_ROOT}/{name}/"
    text = text.replace(old_dir + f"burstgpt_{BASE_NAME}.json", new_dir + f"burstgpt_{name}.json")
    return text.replace(old_dir, new_dir)


def build_template(source: Path, destination: Path, variant: Variant) -> dict[str, str]:
    """paper_config_v1/template with the trace paths pointed at the variant's desktop directory."""
    destination.mkdir(parents=True, exist_ok=False)
    for name in TEMPLATE_FILES:
        raw = (source / name).read_bytes()
        if name == "campaign.json":
            raw = rewrite_template_text(raw.decode("ascii"), variant.name).encode("ascii")
        elif name == "CHANGES.txt":
            raw = (rewrite_template_text(raw.decode("ascii"), variant.name)
                   + f"2026-09-29 (WS6 generality): {variant.template_name} = template-eval2-s2 (paper_config_v1) "
                     f"with the four trace paths at {DESKTOP_ROOT}/{variant.name}/ ({variant.note}); "
                     "nothing else changed.\n").encode("ascii")
        (destination / name).write_bytes(raw)
    check_template(source, destination, variant)
    return {name: digest_file(destination / name) for name in TEMPLATE_FILES}


def check_template(source: Path, destination: Path, variant: Variant) -> None:
    before = json.loads((source / "campaign.json").read_text())
    after = json.loads((destination / "campaign.json").read_text())
    expected = dict(before)
    expected["trace"] = {
        **before["trace"],
        "large_requests_path": f"{DESKTOP_ROOT}/{variant.name}/{TRACE_FILES[0]}",
        "overlay_requests_path": f"{DESKTOP_ROOT}/{variant.name}/{TRACE_FILES[1]}",
        "replay_schedule_path": f"{DESKTOP_ROOT}/{variant.name}/{variant.trace_name}.json",
        "trace_manifest_path": f"{DESKTOP_ROOT}/{variant.name}/{TRACE_FILES[2]}",
    }
    need(after == expected, f"{variant.template_name} campaign differs beyond the trace paths")
    for name in ("rig.json", "models.json", "evidence.json"):
        need((source / name).read_bytes() == (destination / name).read_bytes(), f"{name} must be unchanged")


THREAD_KEYS = ("threads", "threads_batch", "cpu_affinity")


def thread_template(source: Path, destination: Path, threads: int, cpu_affinity: str | None = None) -> dict[str, str]:
    """paper_config_v1/template with llama-server --threads/--threads-batch (and taskset) on the assisted models.

    Without these keys llama-server uses common_cpu_get_num_math() = 8 P-cores on the desktop's i9-12900K;
    the overlay's CPU executor keeps its own threads (4). Only models.json and CHANGES.txt change."""
    need(type(threads) is int and threads > 0, "threads must be a positive integer")
    need(cpu_affinity is None or (cpu_affinity.isascii() and cpu_affinity
                                  and all(c.isdigit() or c in ",-" for c in cpu_affinity)), "cpu affinity list")
    destination.mkdir(parents=True, exist_ok=False)
    models = json.loads((source / "models.json").read_text())
    assisted = [row for row in models["models"] if row["kind"] == "assisted"]
    need(len(assisted) == 2 and not any(key in row["runtime_parameters"] for row in assisted for key in THREAD_KEYS),
         "template already sets threads")
    for row in assisted:
        row["runtime_parameters"].update({"threads": threads, "threads_batch": threads,
                                          **({"cpu_affinity": cpu_affinity} if cpu_affinity else {})})
    for name in TEMPLATE_FILES:
        (destination / name).write_bytes((source / name).read_bytes())
    (destination / "models.json").write_text(json.dumps(models, indent=1, sort_keys=True) + "\n")
    with (destination / "CHANGES.txt").open("a", encoding="ascii") as changes:
        changes.write(f"2026-09-29 (WS6 host sweep): {destination.name} = template-eval2-s2 (paper_config_v1) with "
                      f"threads={threads} threads_batch={threads}"
                      + (f" cpu_affinity={cpu_affinity}" if cpu_affinity else "")
                      + " in the runtime_parameters of qwen3-14b-q4km-dequant-f16 and gemma-4-12b-q40-dequant-f16; "
                        "nothing else changed.\n")
    before = json.loads((source / "models.json").read_text())
    after = json.loads((destination / "models.json").read_text())
    for row in before["models"]:
        if row["kind"] == "assisted":
            row["runtime_parameters"].update({key: value for key, value in (
                ("threads", threads), ("threads_batch", threads), ("cpu_affinity", cpu_affinity)) if value is not None})
    need(before == after, "thread template differs beyond the thread keys")
    for name in ("campaign.json", "rig.json", "evidence.json"):
        need((source / name).read_bytes() == (destination / name).read_bytes(), f"{name} must be unchanged")
    return {name: digest_file(destination / name) for name in TEMPLATE_FILES}


def run_builder(argv: Sequence[str]) -> None:
    completed = subprocess.run([sys.executable, "-m", BUILDER_MODULE, *argv], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise GeneralityError("builder failed: " + (completed.stderr or completed.stdout).strip()[-2000:])


def build_variant(variant: Variant, *, out_dir: Path, csv: Path, tools: Mapping[str, Path], inventory_from: Path,
                  desktop_root: str = DESKTOP_ROOT, desktop_csv: str = DESKTOP_CSV,
                  rules: Sequence[str] = BASE_RULES) -> Path:
    """Build one variant into out_dir/trace/<name> with the desktop paths recorded; returns the directory."""
    staging = out_dir / ".staging" / variant.name
    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    argv = builder_argv(variant, csv=csv, output_dir=staging, tools=tools, inventory_from=inventory_from, rules=rules)
    run_builder(argv)
    manifest = relocate_manifest(json.loads((staging / "TRACE_MANIFEST.json").read_text()),
                                 f"{desktop_root}/{variant.name}", desktop_csv)
    (staging / "TRACE_MANIFEST.json").write_bytes(canonical(manifest))
    final = out_dir / "trace" / variant.name
    need(not final.exists(), f"{final} exists; use a fresh --out-dir")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(final)
    names = trace_file_names(variant.trace_name)
    need(sorted(os.listdir(final)) == sorted(names), f"unexpected files in {final}")
    desktop_argv = builder_argv(
        variant, csv=Path(desktop_csv), output_dir=Path(f"{desktop_root}/{variant.name}"),
        tools={key: Path(f"<{key}>") for key in tools}, inventory_from=Path(f"{desktop_root}/{BASE_NAME}/TRACE_MANIFEST.json"),
        rules=rules)
    sums = write_sha256sums(final, names)
    (final / "VARIANT.json").write_bytes(canonical({
        "schema": SCHEMA, "variant": variant.to_json(), "sha256": sums,
        "desktop_directory": f"{desktop_root}/{variant.name}",
        "builder_command": ["python3", "-m", BUILDER_MODULE, *desktop_argv],
        "note": "Built with the local llama-token-codec against tokenizer models whose sha256 equals the manifest's "
                "derivation.tokenizers; model_inventory copied from the base manifest (same execution artifacts).",
    }))
    return final


def verify(out_dir: Path, base_dir: Path, names: Sequence[str] | None = None) -> dict[str, Any]:
    """Re-check every built variant: loader identity, sha256 sums, density identity, window disjointness."""
    base = load_trace(base_dir, BASE.trace_name)
    base_summary = trace_summary(base)
    rows: dict[str, Any] = {}
    windows = {BASE_NAME: (base_summary["window_start_source_s"], base_summary["window_end_source_s"])}
    for variant in VARIANTS:
        if names is not None and variant.name not in names:
            continue
        directory = out_dir / "trace" / variant.name
        need(directory.is_dir(), f"missing {directory}")
        listed = {}
        for line in (directory / "SHA256SUMS").read_text().splitlines():
            value, name = line.split("  ", 1)
            listed[name] = value
        need(listed == {name: digest_file(directory / name) for name in trace_file_names(variant.trace_name)},
             f"{variant.name} SHA256SUMS")
        trace = load_trace(directory, variant.trace_name)
        summary = trace_summary(trace)
        manifest = trace["manifest"]
        need(manifest["base_trace"]["path"].startswith(f"{DESKTOP_ROOT}/{variant.name}/")
             and manifest["overlay_trace"]["path"].startswith(f"{DESKTOP_ROOT}/{variant.name}/")
             and manifest["derivation"]["burstgpt_csv"] == DESKTOP_CSV, f"{variant.name} manifest paths")
        need(manifest["derivation"]["burstgpt_csv_sha256"] == base["manifest"]["derivation"]["burstgpt_csv_sha256"]
             and manifest["derivation"]["tokenizers"] == base["manifest"]["derivation"]["tokenizers"]
             and manifest["model_inventory"] == base["manifest"]["model_inventory"],
             f"{variant.name} source, tokenizers or model inventory differ from the base")
        need(summary["arrival_scale"] == variant.arrival_scale and summary["window_rank"] == variant.window_rank
             and summary["scan_phase_s"] == variant.scan_phase_s, f"{variant.name} derivation flags")
        if variant.kind == "density":
            check_density(base, trace, variant.arrival_scale, BASE.trace_name, variant.trace_name)
        else:
            tail = manifest["derivation"]["window"]["long_tail"]
            need(abs(tail["window"]["request_share"] - tail["log"]["request_share"]) <= 0.05
                 and abs(tail["window"]["output_token_share"] - tail["log"]["output_token_share"]) <= 0.05
                 and min(summary["source_models"].values()) >= 4 and 14 <= summary["requests"] <= 18
                 and summary["output_tokens"] <= 4000, f"{variant.name} breaks a selection rule")
            windows[variant.name] = (summary["window_start_source_s"], summary["window_end_source_s"])
        rows[variant.name] = {"variant": variant.to_json(), "summary": summary,
                              "sha256": {name: digest_file(directory / name) for name in trace_file_names(variant.trace_name)}}
    check_disjoint(windows)
    return {"schema": SCHEMA, "base": {"name": BASE_NAME, "summary": base_summary}, "variants": rows}


def copy_script(variants: Sequence[Variant]) -> str:
    lines = ["#!/bin/bash", "# Copy the WS6 generality traces and templates to the desktop. NOT run by WS6.",
             "# Run from the WS6 deliverable directory; the desktop paths must not exist yet.", "set -euo pipefail",
             f"H={DESKTOP_HOST}"]
    for variant in variants:
        trace_dir = f"{DESKTOP_ROOT}/{variant.name}"
        template_dir = f"{DESKTOP_TEMPLATE_ROOT}/{variant.template_name}"
        lines += [
            f"ssh $H 'test ! -e {trace_dir} && test ! -e {template_dir} && mkdir {trace_dir}'",
            f"scp trace/{variant.name}/* $H:{trace_dir}/",
            f"ssh $H 'cd {trace_dir} && sha256sum -c SHA256SUMS'",
            f"scp -r templates/{variant.template_name} $H:{DESKTOP_TEMPLATE_ROOT}/",
        ]
    return "\n".join(lines) + "\n"


def summary_markdown(report: Mapping[str, Any]) -> str:
    def row(name: str, value: Mapping[str, Any]) -> str:
        models = value["source_models"]
        return (f"| `{name}` | {value['requests']} ({value['qwen_hot']} Qwen / {value['gemma_cold']} Gemma / "
                f"{value['llama_overlay']} Llama) | {models['GPT-4']}/{value['requests']} | {value['input_tokens']:,} / "
                f"{value['output_tokens']:,} | {value['max_output_tokens_per_request']} | "
                f"{value['long_tail_request_share']:.1%} / {value['long_tail_output_token_share']:.1%} | "
                f"{value['arrival_scale']:g} | {value['replay_span_s']:,.0f} | "
                f"{value['window_start_source_s']:,.0f} | {value['window_rank']} / {value['scan_phase_s']:g} |")
    log = report["base"]["summary"]["long_tail_log"]
    lines = [
        "| trace | requests | GPT-4 share | input / output tokens | max output | long tail req / out | arrival scale "
        "| span s | window start (source s) | rank / grid phase s |",
        "|---|---|---|---|---|---|---|---|---|---|",
        row(BASE_NAME, report["base"]["summary"]),
        *(row(name, value["summary"]) for name, value in report["variants"].items()),
        "",
        f"Log-wide long tail (source output > 512 tokens): {log['request_share']:.1%} of requests / "
        f"{log['output_token_share']:.1%} of output tokens; every window must be within 5 points of both.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--burstgpt-csv", type=Path, required=True)
    build.add_argument("--codec", type=Path, required=True)
    build.add_argument("--library-dir", type=Path, required=True)
    build.add_argument("--qwen-tokenizer-model", type=Path, required=True)
    build.add_argument("--gemma-tokenizer-model", type=Path, required=True)
    build.add_argument("--llama-tokenizer-model", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    build.add_argument("--base-dir", type=Path, default=PAPER_CONFIG / "trace" / BASE_NAME)
    build.add_argument("--template-dir", type=Path, default=PAPER_CONFIG / "template")
    build.add_argument("--variants", default=",".join(row.name.removeprefix(BASE_NAME + "_") for row in VARIANTS))
    threads = sub.add_parser("thread-templates", help="host thread sweep templates (no trace change)")
    threads.add_argument("--out-dir", type=Path, required=True)
    threads.add_argument("--template-dir", type=Path, default=PAPER_CONFIG / "template")
    threads.add_argument("--threads", default="4,8,12,16")
    threads.add_argument("--cpu-affinity", action="append", default=[], metavar="N=LIST",
                         help="taskset --cpu-list for N threads (verify the CPU numbering with lscpu -e first)")
    for name in ("verify", "plan"):
        parser = sub.add_parser(name)
        parser.add_argument("--out-dir", type=Path, required=name == "verify")
        parser.add_argument("--base-dir", type=Path, default=PAPER_CONFIG / "trace" / BASE_NAME)
    args = ap.parse_args(argv)
    if args.command == "plan":
        tools = {key: Path(f"<{key}>") for key in ("codec", "library_dir", "qwen", "gemma", "llama")}
        for variant in VARIANTS:
            print(" ".join(["python3", "-m", BUILDER_MODULE, *builder_argv(
                variant, csv=Path(DESKTOP_CSV), output_dir=Path(f"{DESKTOP_ROOT}/{variant.name}"), tools=tools,
                inventory_from=Path(f"{DESKTOP_ROOT}/{BASE_NAME}/TRACE_MANIFEST.json"))]))
        return 0
    if args.command == "thread-templates":
        affinity = dict(item.split("=", 1) for item in args.cpu_affinity)
        made = {}
        for count in (int(item) for item in args.threads.split(",") if item):
            name = f"{TEMPLATE_PREFIX}t{count}" + ("p" if str(count) in affinity else "")
            made[name] = thread_template(args.template_dir, args.out_dir / name, count, affinity.get(str(count)))
        print(json.dumps(made, indent=2, sort_keys=True))
        return 0
    if args.command == "verify":
        print(json.dumps(verify(args.out_dir, args.base_dir), indent=2, sort_keys=True))
        return 0
    selected = [VARIANT_BY_NAME[BASE_NAME + "_" + name] for name in args.variants.split(",") if name]
    tools = {"codec": args.codec, "library_dir": args.library_dir, "qwen": args.qwen_tokenizer_model,
             "gemma": args.gemma_tokenizer_model, "llama": args.llama_tokenizer_model}
    inventory_from = args.base_dir / "TRACE_MANIFEST.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rebuilt = build_variant(BASE, out_dir=args.out_dir / ".base-check", csv=args.burstgpt_csv, tools=tools,
                            inventory_from=inventory_from)
    for name in trace_file_names(BASE.trace_name):
        need((rebuilt / name).read_bytes() == (args.base_dir / name).read_bytes(),
             f"rebuilt {BASE_NAME} differs from the frozen copy in {name}; the local derivation does not reproduce")
    templates = {}
    for variant in selected:
        build_variant(variant, out_dir=args.out_dir, csv=args.burstgpt_csv, tools=tools, inventory_from=inventory_from)
        templates[variant.template_name] = build_template(
            args.template_dir, args.out_dir / "templates" / variant.template_name, variant)
    report = verify(args.out_dir, args.base_dir, [row.name for row in selected])
    report["base_rebuild"] = "byte-identical to " + str(args.base_dir)
    report["templates"] = templates
    report["builder_rules"] = list(BASE_RULES)
    (args.out_dir / "GENERALITY_TRACES.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "GENERALITY_TRACES.md").write_text(summary_markdown(report))
    (args.out_dir / "COPY_TO_DESKTOP.sh").write_text(copy_script(selected))
    shutil.rmtree(args.out_dir / ".staging", ignore_errors=True)
    shutil.rmtree(args.out_dir / ".base-check", ignore_errors=True)
    print(summary_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
