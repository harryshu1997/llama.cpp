#!/usr/bin/env python3
"""WS17 approximate-numerics qualification of a quantized phone helper (see _internal/approximate_numerics.py).

    prompts   --trace-dir DIR --tokenizer qwen3-14b-q4_k_m [--count 8] --out PROMPTS.json
    reference --endpoint http://127.0.0.1:PORT --prompts PROMPTS.json [--positions 64] --out REF.json
    force     --endpoint http://127.0.0.1:PORT --reference REF.json --out CAND.json
    rows      --stderr SERVER.stderr [...] --out ROWS.json
    evaluate  --reference REF.json --candidate CAND.json --aa AA.json --rows ROWS.json
              [--policy POLICY.json] [--identity KEY=VALUE ...] --out RECEIPT.json

``reference`` runs greedy generation on the desktop-only server (the qualified GPU placement) and keeps each
position's top-2 log-probabilities. ``force`` replays every position on another server with the reference
prefix forced (one ``n_predict: 1`` request per position, ``cache_prompt`` on, so only the new token is
processed): run it on the phone-assisted server (candidate) and on a second desktop configuration of the same
artifact, e.g. ``-ngl 0`` (A/A). ``rows`` collects the assisted server's ``S41SERVERFFNROW stage=returned``
diagnostics (``S41_SERVER_FFN_ROW_DIAGNOSTIC_STEPS``). ``evaluate`` writes the receipt; exit 0 = PASS, 2 = FAIL.
Prompts are token ids from the trace, so the tokenizer of the qualified artifact must equal the trace's
(``quantized_regime.py tokenizer-check``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research_dev.scheduler._internal import approximate_numerics as an  # noqa: E402


def canonical_sha256(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def path_sha256(prompts: list[dict[str, object]]) -> str:
    return canonical_sha256([[row["prompt_id"], [p["token"] for p in row["positions"]]] for row in prompts])


def prompts_from_trace(trace_dir: Path, tokenizer: str, count: int) -> list[dict[str, object]]:
    rows = []
    for line in (trace_dir / "REQUESTS_SEMANTIC_SOURCE.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("prompt_tokenizer_model") == tokenizer:
            rows.append({"prompt_id": row["event_id"], "prompt_tokens": list(row["prompt_tokens"])})
    if not rows:
        raise SystemExit(f"no trace rows tokenized with {tokenizer}")
    return rows[:count]


def _post(endpoint: str, body: dict[str, object], timeout: float) -> dict[str, object]:
    request = urllib.request.Request(endpoint.rstrip("/") + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _top(entry: dict[str, object]) -> list[list[object]]:
    return [[int(row["id"]), float(row["logprob"])] for row in entry.get("top_logprobs", [])]


def _body(tokens: list[int], n_predict: int) -> dict[str, object]:
    return {"prompt": tokens, "n_predict": n_predict, "n_probs": 2, "temperature": 0.0, "top_k": 1,
            "ignore_eos": True, "cache_prompt": True, "stream": False}


def reference_path(endpoint: str, prompts: list[dict[str, object]], positions: int, timeout: float) -> dict[str, object]:
    out = []
    for prompt in prompts:
        reply = _post(endpoint, _body(prompt["prompt_tokens"], positions), timeout)
        probs = reply.get("completion_probabilities") or []
        if len(probs) != positions:
            raise SystemExit(f"{prompt['prompt_id']}: {len(probs)} positions returned, {positions} requested")
        out.append({"prompt_id": prompt["prompt_id"], "prompt_tokens": prompt["prompt_tokens"],
                    "positions": [{"token": int(entry["id"]), "top": _top(entry)} for entry in probs]})
    return {"schema": an.PATH_SCHEMA, "role": "reference", "endpoint": endpoint, "positions_per_prompt": positions,
            "prompts": out, "path_sha256": path_sha256(out)}


def forced_path(endpoint: str, reference: dict[str, object], timeout: float) -> dict[str, object]:
    out = []
    for prompt in reference["prompts"]:
        tokens = list(prompt["prompt_tokens"])
        positions = []
        for ref in prompt["positions"]:
            reply = _post(endpoint, _body(tokens, 1), timeout)
            probs = reply.get("completion_probabilities") or []
            if len(probs) != 1:
                raise SystemExit(f"{prompt['prompt_id']}: forced request returned {len(probs)} positions")
            positions.append({"token": int(probs[0]["id"]), "top": _top(probs[0])})
            tokens.append(int(ref["token"]))
        out.append({"prompt_id": prompt["prompt_id"], "positions": positions})
    return {"schema": an.PATH_SCHEMA, "role": "forced", "endpoint": endpoint,
            "forced_reference_sha256": reference["path_sha256"], "prompts": out}


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    aa = json.loads(args.aa.read_text())
    expected = path_sha256(reference["prompts"])
    if reference.get("path_sha256") != expected:
        raise SystemExit("reference path digest differs from its content")
    for name, value in (("candidate", candidate), ("A/A", aa)):
        if value.get("forced_reference_sha256") != expected:
            raise SystemExit(f"{name} was not forced along this reference path")
    rows = json.loads(args.rows.read_text())["rows"]
    policy = an.ApproximateNumericsPolicy.from_json(json.loads(args.policy.read_text()) if args.policy else None)
    identity = dict(item.split("=", 1) for item in args.identity)
    identity.update(reference_path_sha256=expected,
                    **{name + "_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                       for name, path in (("reference", args.reference), ("candidate", args.candidate),
                                          ("aa", args.aa), ("rows", args.rows))})
    return an.evaluate(reference=reference, candidate=candidate, aa_candidate=aa, rows=rows, policy=policy,
                       identity=identity)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("reference")
    r.add_argument("--endpoint", required=True)
    r.add_argument("--prompts", type=Path, required=True)
    r.add_argument("--positions", type=int, default=64)
    r.add_argument("--timeout-s", type=float, default=600.0)
    r.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("force")
    f.add_argument("--endpoint", required=True)
    f.add_argument("--reference", type=Path, required=True)
    f.add_argument("--timeout-s", type=float, default=600.0)
    f.add_argument("--out", type=Path, required=True)
    w = sub.add_parser("rows")
    w.add_argument("--stderr", type=Path, nargs="+", required=True)
    w.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate")
    for name in ("reference", "candidate", "aa", "rows"):
        e.add_argument("--" + name, type=Path, required=True)
    e.add_argument("--policy", type=Path)
    e.add_argument("--identity", action="append", default=[], metavar="KEY=VALUE")
    e.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if getattr(args, "out", None) is not None and args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if args.command == "prompts":
        value = {"schema": "ws17-numerics-prompts-v1", "tokenizer": args.tokenizer,
                 "prompts": prompts_from_trace(args.trace_dir, args.tokenizer, args.count)}
    elif args.command == "reference":
        value = reference_path(args.endpoint, json.loads(args.prompts.read_text())["prompts"], args.positions,
                               args.timeout_s)
    elif args.command == "force":
        value = forced_path(args.endpoint, json.loads(args.reference.read_text()), args.timeout_s)
    elif args.command == "rows":
        rows = []
        for path in args.stderr:
            rows.extend(an.parse_row_diagnostics(path.read_text(errors="replace").splitlines()))
        value = {"schema": "ws17-ffn-row-diagnostics-v1", "sources": [str(p) for p in args.stderr], "rows": rows}
    else:
        value = evaluate(args)
    args.out.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n")
    if args.command == "evaluate":
        print(value["status"], json.dumps({"row_screen": value["row_screen"]["status"],
                                           "teacher_forced": value["teacher_forced"]["status"]}))
        return 0 if value["status"] == "PASS" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
