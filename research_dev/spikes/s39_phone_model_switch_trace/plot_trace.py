#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "s39-phone-model-switch-trace-v2"

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


WINDOW_US = 1_200_000_000
TRIGGER_GAP_US = 60_000_000
MODEL_VALUE = {"ChatGPT": 0, "GPT-4": 1}
MODEL_LABEL = {0: "Gemma", 1: "Qwen"}


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="ascii").splitlines(),
        1,
    ):
        row = json.loads(line)
        model = row.get("source_fields", {}).get("model")
        if model not in MODEL_VALUE:
            raise ValueError(
                f"{path}:{line_number}: unsupported model {model!r}"
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty trace")
    if [row["t_us"] for row in rows] != sorted(row["t_us"] for row in rows):
        raise ValueError(f"{path}: timestamps are not nondecreasing")
    if rows[-1]["t_us"] >= WINDOW_US:
        raise ValueError(f"{path}: request lies outside the 20-minute window")
    return rows


def derive_policy(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    successful = [
        row for row in rows if not row["source_fields"]["burstgpt_failed"]
    ]
    cycles = []
    phone_only = []
    index = 0
    while index < len(successful):
        end = index + 1
        model = successful[index]["source_fields"]["model"]
        while (
            end < len(successful)
            and successful[end]["source_fields"]["model"] == model
        ):
            end += 1
        run = successful[index:end]
        if model == "GPT-4":
            trigger_index = next(
                (
                    offset
                    for offset in range(1, len(run))
                    if run[offset]["t_us"] - run[offset - 1]["t_us"]
                    <= TRIGGER_GAP_US
                ),
                None,
            )
            if trigger_index is None:
                phone_only.append(run[0]["t_us"])
            else:
                cycles.append(
                    {
                        "start_us": run[0]["t_us"],
                        "trigger_us": run[trigger_index]["t_us"],
                        "return_us": (
                            successful[end]["t_us"]
                            if end < len(successful)
                            else WINDOW_US
                        ),
                        "has_return_request": end < len(successful),
                        "request_count": len(run),
                    }
                )
        index = end
    return cycles, phone_only


def minute(value_us: int) -> float:
    return value_us / 60_000_000


def clock(value_us: int) -> str:
    seconds = value_us // 1_000_000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def plot(
    rows: list[dict[str, Any]],
    output_prefix: Path,
    profile_label: str,
) -> tuple[Path, Path]:
    cycles, phone_only = derive_policy(rows)
    times = [minute(row["t_us"]) for row in rows]
    input_tokens = [row["input_tokens"] for row in rows]
    output_tokens = [row["output_tokens"] for row in rows]
    failed = [bool(row["source_fields"]["burstgpt_failed"]) for row in rows]
    successful = [index for index, value in enumerate(failed) if not value]
    success_times = [times[index] for index in successful]
    success_models = [
        MODEL_VALUE[rows[index]["source_fields"]["model"]]
        for index in successful
    ]
    step_times = [0.0, *success_times, 20.0]
    step_models = [0, *success_models, success_models[-1]]

    fig, (model_ax, token_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )
    fig.suptitle(
        f"S39 BurstGPT 20-minute trace: {profile_label}",
        fontsize=16,
        fontweight="bold",
    )

    for axis in (model_ax, token_ax):
        axis.axvspan(0, 20, color="#4C78A8", alpha=0.07)
        for cycle in cycles:
            axis.axvspan(
                minute(cycle["trigger_us"]),
                minute(cycle["return_us"]),
                color="#F58518",
                alpha=0.13,
            )
        axis.grid(
            True,
            axis="both",
            color="#D9D9D9",
            linewidth=0.7,
            alpha=0.75,
        )

    model_ax.step(
        step_times,
        step_models,
        where="post",
        color="#333333",
        linewidth=1.6,
        alpha=0.7,
        label="Most recent requested model",
    )
    gemma_times = [
        times[index]
        for index in successful
        if rows[index]["source_fields"]["model"] == "ChatGPT"
    ]
    qwen_times = [
        times[index]
        for index in successful
        if rows[index]["source_fields"]["model"] == "GPT-4"
    ]
    model_ax.scatter(
        gemma_times,
        [0] * len(gemma_times),
        s=36,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="Gemma request",
    )
    model_ax.scatter(
        qwen_times,
        [1] * len(qwen_times),
        s=44,
        color="#F58518",
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
        label="Qwen request",
    )
    failed_times = [times[index] for index, value in enumerate(failed) if value]
    failed_models = [
        MODEL_VALUE[rows[index]["source_fields"]["model"]]
        for index, value in enumerate(failed)
        if value
    ]
    model_ax.scatter(
        failed_times,
        failed_models,
        marker="x",
        s=60,
        color="#B22222",
        linewidth=2,
        zorder=5,
        label="Failed source request, skipped",
    )

    for cycle in cycles:
        trigger = minute(cycle["trigger_us"])
        model_ax.axvline(
            trigger,
            color="#228B22",
            linestyle="--",
            linewidth=1.5,
        )
        model_ax.text(
            trigger,
            1.42,
            f"G->Q\n{clock(cycle['trigger_us'])}",
            ha="center",
            va="top",
            fontsize=7,
            color="#165C16",
        )
        if cycle["has_return_request"]:
            returned = minute(cycle["return_us"])
            model_ax.axvline(
                returned,
                color="#228B22",
                linestyle=":",
                linewidth=1.35,
            )
            model_ax.text(
                returned,
                -0.43,
                f"Q->G\n{clock(cycle['return_us'])}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#165C16",
            )

    for value_us in phone_only:
        value = minute(value_us)
        model_ax.axvline(
            value,
            color="#777777",
            linestyle=":",
            linewidth=1,
            alpha=0.8,
        )
        model_ax.text(
            value,
            1.16,
            "phone only",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=6,
            color="#666666",
        )

    model_ax.set_yticks([0, 1], [MODEL_LABEL[0], MODEL_LABEL[1]])
    model_ax.set_ylim(-0.48, 1.48)
    model_ax.set_ylabel("Requested model")
    model_ax.set_title(
        "Request sequence and derived server-target intervals",
        loc="left",
        fontsize=11,
    )
    handles, labels = model_ax.get_legend_handles_labels()
    handles.extend(
        [
            Patch(color="#4C78A8", alpha=0.12),
            Patch(color="#F58518", alpha=0.18),
        ]
    )
    labels.extend(["Server target: Gemma", "Server target: Qwen"])
    model_ax.legend(
        handles,
        labels,
        loc="lower right",
        ncols=3,
        fontsize=7.5,
        framealpha=0.95,
    )

    token_ax.plot(
        times,
        input_tokens,
        color="#2A9D8F",
        marker="o",
        markersize=3,
        linewidth=1.25,
        alpha=0.9,
        label="Input tokens",
    )
    token_ax.plot(
        times,
        output_tokens,
        color="#8E5EA2",
        marker="o",
        markersize=3,
        linewidth=1.25,
        alpha=0.9,
        label="Output tokens",
    )
    token_ax.scatter(
        failed_times,
        [0] * len(failed_times),
        marker="x",
        s=50,
        color="#B22222",
        linewidth=2,
        zorder=5,
    )
    token_ax.set_ylabel("Tokens per request")
    token_ax.set_xlabel("Trace time (minutes)")
    token_ax.set_xlim(0, 20)
    token_ax.set_xticks(range(0, 21))
    token_ax.set_ylim(bottom=-20)
    token_ax.legend(loc="upper right", fontsize=9)

    target_switches = len(cycles) + sum(
        cycle["has_return_request"] for cycle in cycles
    )
    fig.text(
        0.5,
        0.012,
        (
            f"Derived policy: {len(cycles)} Qwen promotion cycles and "
            f"{target_switches} target changes. Orange spans are policy "
            "targets; actual readiness and handoff depend on measured load time."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(
        png_path,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "s39-phone-model-switch-trace-v2"},
    )
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        metadata={
            "Creator": "s39-phone-model-switch-trace-v2",
            "Date": None,
        },
    )
    plt.close(fig)
    return png_path, svg_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot an S39 switch trace")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--profile-label", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    png, svg = plot(
        load_trace(arguments.trace),
        arguments.output_prefix,
        arguments.profile_label,
    )
    print(png)
    print(svg)
