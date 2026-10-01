#!/usr/bin/env python3
"""Score GSM8K quality runs and apply the pre-registered noninferiority test (reads run artifacts only).

    python3 -m research_dev.scheduler.campaigns.burstgpt.quality.score score \\
        --suite OUT/SUITE.json --key OUT/QUALITY_KEY.jsonl \\
        --arm legacy=RUN_DIR_SHARD0 --arm legacy=RUN_DIR_SHARD1 ... \\
        --arm paper=... --arm forced=... \\
        --baseline legacy --treatment paper --treatment forced --strict-arm forced \\
        [--aa legacy_repeat] [--shards N] --out REPORT.json --md REPORT.md

    python3 -m ...quality.score interim --suite ... --key ... --arm ... --baseline legacy --treatment ...

    python3 -m ...quality.score pilot --suite PILOT/SUITE.json --key PILOT/QUALITY_KEY.jsonl \
        --arm legacy=PILOT_RUN_DIR --out PILOT.json

A RUN_DIR is a campaign run directory (RESULT.json or RESULT.json.gz and streams/request-NNN.raw); the
shard is taken from RESULT.replay_schedule.trace_name. Each output is the decoded text of its streamed
chunks; the answer is extracted with the frozen rule of gsm8k.extract_answer and compared with the key.
`score` writes the full report: per-arm accuracy, the Tango noninferiority test of the pooled Qwen + Gemma
paired difference for each treatment (fixed-sequence in --treatment order), sensitivity intervals, the
per-model and per-protocol readouts, similarity to the baseline output, phone coverage and the Llama
control. `interim` is the blinded sample-size re-estimation: it prints only discordance totals, never
accuracies or the direction of any difference. `pilot` applies the budget rule to the train-split pilot
run and prints the --output-tokens arguments for the confirmatory build.
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import json
import math
import statistics
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import protocol
from .gsm8k import Extraction, extract_answer
from .stats import (
    PairedCounts, exact_mcnemar_p, newcombe_interval, noninferiority, noninferiority_power,
    noninferiority_sample_size, paired_bootstrap_interval,
)

REPORT_SCHEMA = "research-scheduler-quality-report-v1"
INTERIM_SCHEMA = "research-scheduler-quality-interim-v1"
PILOT_SCHEMA = "research-scheduler-quality-pilot-v1"
ALL_ROLES = (*protocol.CONFIRMATORY_ROLES, *protocol.CONTROL_ROLES)


class QualityScoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class KeyRow:
    request_id: str
    trace_name: str
    shard: int
    role: str
    item_id: str
    reference: Decimal
    combined_request_index: int
    prompt_sha256: str
    output_tokens: int


@dataclass
class Output:
    """One arm's result for one key row; `missing` holds the reason when there is no scorable stream."""

    key: KeyRow
    missing: str | None = None
    tokens: list[int] = field(default_factory=list)
    text: str = ""
    chunk_starts: list[tuple[int, int]] = field(default_factory=list)
    extraction: Extraction | None = None
    correct: bool = False
    executor_id: str | None = None
    assisted_share: float | None = None
    phone_calls: int = 0

    def answer_token_end(self) -> int:
        """Tokens of the shortest stream prefix whose text holds the extracted answer (all tokens if none)."""
        if self.extraction is None or not self.extraction.found:
            return len(self.tokens)
        for token_start, char_start in self.chunk_starts:
            if char_start >= self.extraction.end:
                return token_start
        return len(self.tokens)


def truncated_extraction(output: Output, budget: int) -> Extraction:
    """The frozen extraction on the text of the first `budget` tokens (a chunk crossing the budget is cut)."""
    end_char = len(output.text)
    for index, (_, char_start) in enumerate(output.chunk_starts):
        chunk_end = (output.chunk_starts[index + 1][0] if index + 1 < len(output.chunk_starts)
                     else len(output.tokens))
        if chunk_end > budget:
            end_char = char_start
            break
    return extract_answer(output.text[:end_char])


def load_key(path: Path) -> dict[str, KeyRow]:
    rows = {}
    for line in path.read_text(encoding="ascii").splitlines():
        value = json.loads(line)
        row = KeyRow(request_id=value["request_id"], trace_name=value["trace_name"], shard=value["shard"],
                     role=value["role"], item_id=value["item_id"], reference=Decimal(value["reference"]),
                     combined_request_index=value["combined_request_index"], prompt_sha256=value["prompt_sha256"],
                     output_tokens=value["output_tokens"])
        if row.request_id in rows:
            raise QualityScoreError(f"duplicate key row {row.request_id}")
        rows[row.request_id] = row
    return rows


def read_stream(path: Path) -> tuple[list[int], str, list[tuple[int, int]]]:
    """Tokens, text and (token_start, char_start) of each streamed chunk of a llama-server SSE file."""
    tokens: list[int] = []
    parts: list[str] = []
    starts: list[tuple[int, int]] = []
    characters = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("data:"):
            continue
        encoded = line[5:].strip()
        if not encoded or encoded == "[DONE]":
            continue
        value = json.loads(encoded)
        chunk = value.get("tokens", [])
        content = value.get("content", "")
        if type(chunk) is not list or type(content) is not str:
            raise QualityScoreError(f"malformed stream chunk in {path}")
        if chunk or content:
            starts.append((len(tokens), characters))
        tokens.extend(chunk)
        parts.append(content)
        characters += len(content)
    return tokens, "".join(parts), starts


def load_result(run_dir: Path) -> dict[str, Any]:
    for name in ("RESULT.json.gz", "RESULT.json"):
        path = run_dir / name
        if path.is_file():
            opener = gzip.open if name.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as stream:
                return json.load(stream)
    raise QualityScoreError(f"{run_dir} has no RESULT.json")


def assisted_share(row: dict[str, Any]) -> tuple[float, int]:
    """Approximate share of decode steps with phone FFN calls: the busiest layer's call count over the
    request's decode steps (output tokens - 1), capped at 1. Batched calls count once per request."""
    proof = row.get("physical_execution_proof") or {}
    calls = int(proof.get("phone_call_count", 0) or 0)
    by_layer = [int(entry.get("calls", 0)) for entry in proof.get("phone_calls_by_layer", []) or []]
    steps = max(1, int(row.get("output_tokens", 1)) - 1)
    return (min(1.0, max(by_layer, default=0) / steps) if calls else 0.0), calls


def load_arm(label: str, run_dirs: list[Path], key: dict[str, KeyRow]) -> dict[str, Output]:
    """Outputs of every key row of the traces this arm ran; rows of traces it never ran are absent."""
    by_trace: dict[str, list[KeyRow]] = {}
    for row in key.values():
        by_trace.setdefault(row.trace_name, []).append(row)
    outputs: dict[str, Output] = {}
    seen_traces = set()
    for run_dir in run_dirs:
        result = load_result(run_dir)
        trace_name = (result.get("replay_schedule") or {}).get("trace_name")
        if trace_name not in by_trace:
            raise QualityScoreError(f"{label}: {run_dir} replayed {trace_name!r}, not a trace of this suite")
        if trace_name in seen_traces:
            raise QualityScoreError(f"{label}: two runs of {trace_name}; score only the first complete run")
        seen_traces.add(trace_name)
        results = {row["request_id"]: row for row in result.get("request_results", [])}
        rejected = {row.get("request_id") for row in result.get("rejected_requests", []) if type(row) is dict}
        for key_row in by_trace[trace_name]:
            output = Output(key=key_row)
            outputs[key_row.request_id] = output
            row = results.get(key_row.request_id)
            if row is None:
                output.missing = "rejected" if key_row.request_id in rejected else "absent-from-result"
                continue
            if (row.get("prompt_sha256") != key_row.prompt_sha256
                    or row.get("combined_request_index") != key_row.combined_request_index):
                raise QualityScoreError(f"{label}: {key_row.request_id} prompt or index differs from the key")
            if row.get("output_tokens") != key_row.output_tokens:
                raise QualityScoreError(f"{label}: {key_row.request_id} output budget differs from the key")
            stream = run_dir / "streams" / f"request-{key_row.combined_request_index:03d}.raw"
            if not stream.is_file():
                output.missing = "stream-absent"
                continue
            tokens, text, starts = read_stream(stream)
            if len(tokens) != key_row.output_tokens:
                output.missing = "stream-incomplete"
                continue
            output.tokens, output.text, output.chunk_starts = tokens, text, starts
            output.extraction = extract_answer(text)
            output.correct = output.extraction.found and output.extraction.value == key_row.reference
            output.executor_id = row.get("actual_executor_id")
            output.assisted_share, output.phone_calls = assisted_share(row)
    return outputs


def _analysis_rows(key: dict[str, KeyRow], shards: int, roles: tuple[str, ...]) -> list[KeyRow]:
    return sorted((row for row in key.values() if row.shard < shards and row.role in roles),
                  key=lambda row: row.request_id)


def _pairs(rows: list[KeyRow], baseline: dict[str, Output], treatment: dict[str, Output], *,
           imputation: str) -> list[tuple[KeyRow, bool, bool]]:
    """(row, baseline_correct, treatment_correct). imputation 'worst-case': a missing treatment output is
    wrong and a missing baseline output right; 'complete-case': pairs with a missing side are dropped."""
    pairs = []
    for row in rows:
        b, t = baseline.get(row.request_id), treatment.get(row.request_id)
        b_missing = b is None or b.missing is not None
        t_missing = t is None or t.missing is not None
        if imputation == "complete-case" and (b_missing or t_missing):
            continue
        pairs.append((row, True if b_missing else b.correct, False if t_missing else t.correct))
    return pairs


def _interval_json(interval: tuple[float, float]) -> list[float]:
    return [interval[0], interval[1]]


def _describe(pairs: list[tuple[KeyRow, bool, bool]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0}
    counts = PairedCounts.from_pairs((b, t) for _, b, t in pairs)
    test = noninferiority(counts, protocol.MARGIN, protocol.ALPHA_ONE_SIDED)
    return {**counts.to_json(), "tango_interval": test["interval"], "z_at_margin": test["z_at_margin"],
            "noninferior_at_margin": test["noninferior"]}


def similarity(baseline: Output, treatment: Output) -> dict[str, Any]:
    tb, tt = baseline.tokens, treatment.tokens
    first = next((index for index, (x, y) in enumerate(zip(tb, tt)) if x != y), min(len(tb), len(tt)))
    identical = tb == tt
    end_b, end_t = baseline.answer_token_end(), treatment.answer_token_end()
    same_answer = (baseline.extraction.value if baseline.extraction else None) == (
        treatment.extraction.value if treatment.extraction else None)
    ratio = difflib.SequenceMatcher(None, tb[:end_b], tt[:end_t], autojunk=False).ratio()
    return {"identical": identical, "first_divergence": len(tb) if identical else first,
            "identical_through_answer": identical or first >= end_b, "same_answer": same_answer,
            "answer_region_ratio": ratio, "baseline_answer_token_end": end_b}


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]  # noqa: E731
    return {"min": ordered[0], "p25": pick(0.25), "median": pick(0.5), "p75": pick(0.75), "max": ordered[-1],
            "mean": statistics.fmean(ordered)}


def similarity_summary(rows: list[KeyRow], baseline: dict[str, Output], treatment: dict[str, Output]) -> dict:
    values = []
    for row in rows:
        b, t = baseline.get(row.request_id), treatment.get(row.request_id)
        if b is None or t is None or b.missing or t.missing:
            continue
        values.append(similarity(b, t))
    if not values:
        return {"pairs": 0}
    diverged = [value["first_divergence"] for value in values if not value["identical"]]
    return {"pairs": len(values),
            "identical_share": sum(value["identical"] for value in values) / len(values),
            "identical_through_answer_share": sum(value["identical_through_answer"] for value in values) / len(values),
            "same_answer_share": sum(value["same_answer"] for value in values) / len(values),
            "first_divergence_of_diverged": _quantiles(diverged),
            "answer_region_ratio": _quantiles([value["answer_region_ratio"] for value in values])}


def coverage_summary(rows: list[KeyRow], outputs: dict[str, Output]) -> dict[str, Any]:
    present = [outputs[row.request_id] for row in rows
               if row.request_id in outputs and outputs[row.request_id].missing is None]
    if not present:
        return {"requests": 0}
    steps = [max(1, output.key.output_tokens - 1) for output in present]
    assisted = [output.assisted_share * step for output, step in zip(present, steps)]
    return {"requests": len(present), "assisted_requests": sum(output.phone_calls > 0 for output in present),
            "token_weighted_assisted_share": sum(assisted) / sum(steps),
            "requests_at_per_protocol_coverage": sum(output.assisted_share >= protocol.PER_PROTOCOL_MINIMUM_COVERAGE
                                                     for output in present)}


def accuracy_summary(rows: list[KeyRow], outputs: dict[str, Output]) -> dict[str, Any]:
    present = [outputs[row.request_id] for row in rows
               if row.request_id in outputs and outputs[row.request_id].missing is None]
    missing = len(rows) - len(present)
    rules: dict[str, int] = {}
    for output in present:
        rules[output.extraction.rule] = rules.get(output.extraction.rule, 0) + 1
    return {"items": len(rows), "present": len(present), "missing": missing,
            "missing_share": missing / len(rows) if rows else 0.0,
            "correct": sum(output.correct for output in present),
            "accuracy_present": sum(output.correct for output in present) / len(present) if present else None,
            "unscorable": rules.get("none", 0), "extraction_rules": dict(sorted(rules.items()))}


def compare(label: str, rows: list[KeyRow], control_rows: list[KeyRow], baseline: dict[str, Output],
            treatment: dict[str, Output], *, strict: bool, bootstrap_replicates: int) -> dict[str, Any]:
    worst = _pairs(rows, baseline, treatment, imputation="worst-case")
    counts = PairedCounts.from_pairs((b, t) for _, b, t in worst)
    test = noninferiority(counts, protocol.MARGIN, protocol.ALPHA_ONE_SIDED)
    strata = [[int(t) - int(b) for row, b, t in worst if row.role == role] for role in protocol.CONFIRMATORY_ROLES]
    strata = [stratum for stratum in strata if stratum]
    missing_share = max(accuracy_summary(rows, arm)["missing_share"] for arm in (baseline, treatment))
    coverage = coverage_summary(rows, treatment)
    checks = {"missing_share_within_limit": missing_share <= protocol.MAXIMUM_MISSING_SHARE}
    if strict:
        checks["strict_arm_coverage"] = (coverage.get("token_weighted_assisted_share", 0.0)
                                         >= protocol.STRICT_ARM_MINIMUM_COVERAGE)
    per_protocol_rows = [row for row in rows if row.request_id in treatment
                         and treatment[row.request_id].missing is None
                         and treatment[row.request_id].assisted_share >= protocol.PER_PROTOCOL_MINIMUM_COVERAGE]
    return {
        "treatment": label,
        "strict_arm": strict,
        "validity_checks": checks,
        "valid": all(checks.values()),
        "primary": {**counts.to_json(), **test,
                    "imputation": "worst-case (missing treatment = wrong, missing baseline = right)",
                    "lower_bound_above_descriptive_margin": test["interval"][0] > -protocol.DESCRIPTIVE_STRICTER_MARGIN,
                    "descriptive_margin": protocol.DESCRIPTIVE_STRICTER_MARGIN},
        "sensitivity": {
            "newcombe_interval_95": _interval_json(newcombe_interval(counts, 1.0 - 2.0 * protocol.ALPHA_ONE_SIDED)),
            "bootstrap_interval_95": _interval_json(paired_bootstrap_interval(
                strata, bootstrap_replicates, protocol.BOOTSTRAP_SEED, 1.0 - 2.0 * protocol.ALPHA_ONE_SIDED)),
            "bootstrap": {"replicates": bootstrap_replicates, "seed": protocol.BOOTSTRAP_SEED,
                          "strata": list(protocol.CONFIRMATORY_ROLES)},
            "exact_mcnemar_p_two_sided": exact_mcnemar_p(counts),
            "complete_case": _describe(_pairs(rows, baseline, treatment, imputation="complete-case")),
        },
        "per_model": {role: _describe(_pairs([row for row in rows if row.role == role], baseline, treatment,
                                              imputation="worst-case"))
                      for role in protocol.CONFIRMATORY_ROLES},
        "per_protocol": {"minimum_coverage": protocol.PER_PROTOCOL_MINIMUM_COVERAGE,
                         **_describe(_pairs(per_protocol_rows, baseline, treatment, imputation="complete-case"))},
        "similarity": similarity_summary(rows, baseline, treatment),
        "similarity_per_model": {role: similarity_summary([row for row in rows if row.role == role], baseline,
                                                          treatment) for role in protocol.CONFIRMATORY_ROLES},
        "coverage": coverage,
        "control": {"accuracy": accuracy_summary(control_rows, treatment),
                    "similarity": similarity_summary(control_rows, baseline, treatment)},
        "losses": [row.item_id for row, b, t in worst if b and not t],
        "gains": [row.item_id for row, b, t in worst if t and not b],
    }


def fixed_sequence(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Hypotheses in --treatment order, each at the full one-sided alpha, tested only while every earlier
    one was valid and rejected (fixed-sequence procedure: family-wise error <= alpha)."""
    steps = []
    open_gate = True
    for comparison in comparisons:
        if not open_gate:
            steps.append({"treatment": comparison["treatment"], "status": "NOT_TESTED"})
            continue
        if not comparison["valid"]:
            status = "INVALID"
        else:
            status = "NONINFERIOR" if comparison["primary"]["noninferior"] else "NOT_SHOWN"
        steps.append({"treatment": comparison["treatment"], "status": status})
        open_gate = status == "NONINFERIOR"
    return {"procedure": "fixed-sequence", "alpha_one_sided": protocol.ALPHA_ONE_SIDED, "steps": steps}


def _arm_dirs(values: list[str]) -> dict[str, list[Path]]:
    arms: dict[str, list[Path]] = {}
    for value in values:
        label, _, path = value.partition("=")
        if not label or not path:
            raise SystemExit(f"--arm expects LABEL=RUN_DIR: {value}")
        arms.setdefault(label, []).append(Path(path))
    return arms


def load_suite(path: Path, *, pilot: bool) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="ascii"))
    if suite.get("protocol_id") != protocol.PROTOCOL_ID:
        raise QualityScoreError("suite protocol differs from the scorer's protocol")
    if bool(suite.get("pilot")) != pilot:
        raise QualityScoreError("the pilot suite is scored only by `pilot`, the confirmatory suite never by it")
    return suite


def pilot(args: argparse.Namespace) -> dict[str, Any]:
    """Per-model output budget: the smallest candidate that holds the first answer of all but
    floor(PILOT_TRUNCATION_LIMIT x n) pilot outputs (missing outputs count as not held)."""
    suite = load_suite(args.suite, pilot=True)
    key = load_key(args.key)
    arms = _arm_dirs(args.arm)
    if len(arms) != 1:
        raise SystemExit("the pilot scores one arm (the all-desktop baseline)")
    label, run_dirs = next(iter(arms.items()))
    outputs = load_arm(label, run_dirs, key)
    roles = {}
    arguments = []
    for role in ALL_ROLES:
        rows = [row for row in key.values() if row.role == role]
        present = [outputs[row.request_id] for row in rows
                   if row.request_id in outputs and outputs[row.request_id].missing is None]
        missing = len(rows) - len(present)
        allowed = math.floor(protocol.PILOT_TRUNCATION_LIMIT * len(rows))
        misses = {budget: missing + sum(not truncated_extraction(output, budget).found for output in present)
                  for budget in protocol.OUTPUT_TOKEN_CANDIDATES}
        chosen = next((budget for budget in protocol.OUTPUT_TOKEN_CANDIDATES if misses[budget] <= allowed),
                      protocol.OUTPUT_TOKEN_CANDIDATES[-1])
        rules: dict[str, int] = {}
        for output in present:
            rules[output.extraction.rule] = rules.get(output.extraction.rule, 0) + 1
        roles[role] = {"items": len(rows), "missing": missing, "allowed_misses": allowed,
                       "misses_by_budget": {str(budget): value for budget, value in misses.items()},
                       "chosen_output_tokens": chosen, "capped": misses[chosen] > allowed,
                       "answer_token_end": _quantiles([output.answer_token_end() for output in present
                                                       if output.extraction.found]),
                       "extraction_rules": dict(sorted(rules.items()))}
        arguments.append(f"--output-tokens {role}={chosen}")
    return {"schema": PILOT_SCHEMA, "protocol_id": protocol.PROTOCOL_ID, "arm": label,
            "pilot_output_tokens": protocol.PILOT_OUTPUT_TOKENS, "candidates": list(protocol.OUTPUT_TOKEN_CANDIDATES),
            "roles": roles, "builder_arguments": " ".join(arguments), "suite_key_sha256": suite["key_sha256"]}


def score(args: argparse.Namespace) -> dict[str, Any]:
    suite = load_suite(args.suite, pilot=False)
    key = load_key(args.key)
    shards = args.shards if args.shards is not None else suite["planned_shards"]
    arms = _arm_dirs(args.arm)
    labels = [args.baseline, *args.treatment, *([args.aa] if args.aa else [])]
    for label in labels:
        if label not in arms:
            raise SystemExit(f"no --arm runs for {label}")
    outputs = {label: load_arm(label, arms[label], key) for label in set(labels)}
    rows = _analysis_rows(key, shards, protocol.CONFIRMATORY_ROLES)
    control_rows = _analysis_rows(key, shards, protocol.CONTROL_ROLES)
    comparisons = [compare(label, rows, control_rows, outputs[args.baseline], outputs[label],
                           strict=label in args.strict_arm, bootstrap_replicates=args.bootstrap_replicates)
                   for label in args.treatment]
    report = {
        "schema": REPORT_SCHEMA,
        "protocol": {"id": protocol.PROTOCOL_ID, "margin": protocol.MARGIN,
                     "alpha_one_sided": protocol.ALPHA_ONE_SIDED, "extraction_rule": protocol.EXTRACTION_ID,
                     "shards_analysed": shards, "planned_shards": suite["planned_shards"],
                     "suite_key_sha256": suite["key_sha256"]},
        "arms": {label: {"runs": [str(path) for path in arms[label]],
                         "accuracy": {role: accuracy_summary([row for row in rows + control_rows if row.role == role],
                                                             outputs[label])
                                      for role in ALL_ROLES},
                         "pooled": accuracy_summary(rows, outputs[label])}
                 for label in sorted(set(labels))},
        "baseline": args.baseline,
        "comparisons": comparisons,
        "conclusion": fixed_sequence(comparisons),
    }
    if args.aa:
        aa_pairs = _pairs(rows, outputs[args.baseline], outputs[args.aa], imputation="complete-case")
        report["baseline_repeat"] = {"label": args.aa, **_describe(aa_pairs),
                                     "similarity": similarity_summary(rows, outputs[args.baseline], outputs[args.aa])}
    return report


def interim(args: argparse.Namespace) -> dict[str, Any]:
    """Blinded re-estimation of the number of shards from the pooled discordance (never its direction)."""
    suite = load_suite(args.suite, pilot=False)
    key = load_key(args.key)
    shards = protocol.INTERIM_SHARDS if args.shards is None else args.shards
    arms = _arm_dirs(args.arm)
    rows = _analysis_rows(key, shards, protocol.CONFIRMATORY_ROLES)
    baseline = load_arm(args.baseline, arms[args.baseline], key)
    discordance = {}
    for label in args.treatment:
        treatment = load_arm(label, arms[label], key)
        pairs = _pairs(rows, baseline, treatment, imputation="complete-case")
        discordant = sum(b != t for _, b, t in pairs)
        discordance[label] = {"pairs": len(pairs), "discordant": discordant,
                              "discordance": discordant / len(pairs) if pairs else None}
    observed = [row["discordance"] for row in discordance.values() if row["discordance"] is not None]
    if not observed:
        raise QualityScoreError("no interim pairs")
    planning = max(max(observed), 1.0 / max(1, max(row["pairs"] for row in discordance.values())))
    pairs_per_shard = protocol.GEMMA_PER_SHARD + protocol.QWEN_PER_SHARD
    required = noninferiority_sample_size(planning / 2, planning / 2, protocol.MARGIN, protocol.ALPHA_ONE_SIDED,
                                          protocol.TARGET_POWER, step=pairs_per_shard)
    final_shards = min(protocol.MAXIMUM_SHARDS, max(protocol.PLANNED_SHARDS, math.ceil(required / pairs_per_shard)))
    final_pairs = final_shards * pairs_per_shard
    return {"schema": INTERIM_SCHEMA, "protocol_id": protocol.PROTOCOL_ID, "interim_shards": shards,
            "discordance": discordance, "planning_discordance": planning, "required_pairs": required,
            "final_shards": final_shards,
            "power_at_final": noninferiority_power(final_pairs, planning / 2, planning / 2, protocol.MARGIN,
                                                   protocol.ALPHA_ONE_SIDED),
            "capped": math.ceil(required / pairs_per_shard) > protocol.MAXIMUM_SHARDS,
            "suite_key_sha256": suite["key_sha256"]}


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# Quality report ({report['protocol']['id']})", "",
             f"Margin {report['protocol']['margin']:.3f} (accuracy, pooled Qwen + Gemma), one-sided alpha "
             f"{report['protocol']['alpha_one_sided']}, {report['protocol']['shards_analysed']} shards, baseline "
             f"`{report['baseline']}`.", "", "| arm | pooled acc | Qwen | Gemma | Llama (control) | missing |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    fmt = lambda value: "-" if value is None else f"{100 * value:.1f} %"  # noqa: E731
    for label, arm in report["arms"].items():
        accuracy = arm["accuracy"]
        lines.append(f"| {label} | {fmt(arm['pooled']['accuracy_present'])} | "
                     f"{fmt(accuracy['qwen']['accuracy_present'])} | {fmt(accuracy['gemma']['accuracy_present'])} | "
                     f"{fmt(accuracy['llama']['accuracy_present'])} | {arm['pooled']['missing']} |")
    lines += ["", "| treatment | n | gains | losses | d | 95 % Tango CI | z at -margin | NI | valid |",
              "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |"]
    for comparison in report["comparisons"]:
        primary = comparison["primary"]
        low, high = primary["interval"]
        lines.append(f"| {comparison['treatment']} | {primary['n']} | {primary['gain']} | {primary['loss']} | "
                     f"{100 * primary['difference']:+.2f} | [{100 * low:+.2f}, {100 * high:+.2f}] | "
                     f"{primary['z_at_margin']:.2f} | {primary['noninferior']} | {comparison['valid']} |")
    lines += ["", "| treatment | identical outputs | identical through answer | same answer | coverage |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for comparison in report["comparisons"]:
        sim = comparison["similarity"]
        coverage = comparison["coverage"].get("token_weighted_assisted_share")
        lines.append(f"| {comparison['treatment']} | {fmt(sim.get('identical_share'))} | "
                     f"{fmt(sim.get('identical_through_answer_share'))} | {fmt(sim.get('same_answer_share'))} | "
                     f"{fmt(coverage)} |")
    lines += ["", "Conclusion (fixed sequence): " + ", ".join(
        f"{step['treatment']} {step['status']}" for step in report["conclusion"]["steps"]), ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("score", "interim", "pilot"):
        sub = commands.add_parser(name)
        sub.add_argument("--suite", type=Path, required=True)
        sub.add_argument("--key", type=Path, required=True)
        sub.add_argument("--arm", action="append", required=True, metavar="LABEL=RUN_DIR")
        sub.add_argument("--out", type=Path, required=True)
        if name == "pilot":
            continue
        sub.add_argument("--baseline", required=True)
        sub.add_argument("--treatment", action="append", required=True)
        sub.add_argument("--shards", type=int, default=None)
        if name == "score":
            sub.add_argument("--strict-arm", action="append", default=[])
            sub.add_argument("--aa", default=None, help="a repeat of the baseline arm (noise floor, no test)")
            sub.add_argument("--bootstrap-replicates", type=int, default=protocol.BOOTSTRAP_REPLICATES)
            sub.add_argument("--md", type=Path)
    args = parser.parse_args()
    value = {"score": score, "interim": interim, "pilot": pilot}[args.command](args)
    encoded = json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"
    with args.out.open("x") as stream:
        stream.write(encoded)
    if args.command == "score" and args.md is not None:
        with args.md.open("x") as stream:
            stream.write(markdown(value))
    summary = {"score": lambda: value["conclusion"], "interim": lambda: value,
               "pilot": lambda: value["builder_arguments"]}[args.command]()
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
