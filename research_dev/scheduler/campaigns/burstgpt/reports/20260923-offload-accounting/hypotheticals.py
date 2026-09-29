#!/usr/bin/env python3
"""Estimate what scheduler changes would have bought on the long-tail trace, from measured constants.

Everything in `CAL` is derived from the completed runs (accounting.json). The simulator replays the
trace arrivals through a pass-level model of the desktop:

  pass_ms(model, B, policy) = T_other(B) + n_cpu_ffn_layers * t_cpu_ffn + phone_ms(B)
  pass_J                    = T_other * (P_cpu_other + P_gpu) + cpu_ffn_ms * (P_cpu_ffn + P_gpu)
                              + phone_ms * (P_cpu_wait + P_gpu)

with T_other, t_cpu_ffn, P_cpu_other and P_cpu_ffn solved from the four measured batch-1 windows per
model (baseline and 100 % phone: ms/token and J/token), P_gpu and P_cpu_wait taken from the measured
idle/decode samples, t_rpc from the S41SERVERFFNSHAPE call statistics, and the batch-2 increment of
T_other from the measured batch-2 both-baseline windows. Model loads, prefill and idle use the measured
state powers and durations. The model is validated by replaying the actual admission order of both
arms (see `validation` in the output) before any variant is evaluated.

MEASURED inputs are marked M, ESTIMATED assumptions E in the comments. Variants:
  admission  : wave (as observed: after a model switch the first request runs alone, pairs form only
               when several requests are re-planned at one completion) | backfill (work-conserving,
               admit the next same-model request whenever a slot and KV room are free)
  ordering   : fifo | affinity (stay on the resident model while it has queued work, switch when its
               queue is empty or the oldest other-model request has waited max_wait_s)
  phone      : none | current (phone only at batch 1 after probing; pairs decode mixed for the
               measured number of tokens, then baseline) | coherent_splitrow (all slots share one
               policy, one phone call per row) | coherent_coalesced (one call per layer per pass)
  layers     : phone-resident FFN layers per model, current (12, 8) or re-provisioned (18, 26)
  max_batch  : concurrent decode slots per server
  probe_tokens: tokens per request spent on exploration at baseline speed

    python3 hypotheticals.py --accounting accounting.json --out hypotheticals.json
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib

# ---------------------------------------------------------------------------------------- constants
P_GPU_W = 33.0          # M: NVML board power during decode, all states (31.8-34.7 W)
P_CPU_WAIT_W = 8.0      # M: RAPL package while the host waits (idle samples 6-8 W)
P_LOAD_W = 26.5         # M: host power during model load (26.0-26.9 W)
P_PREFILL_W = 55.0      # M: host power during prefill (53-56 W)
P_IDLE_W = 27.2         # M: host power with no request active
T_RPC_MS = {"qwen": 10.27, "gemma": 7.06}      # M: S41SERVERFFNSHAPE rpc_mean_ms, tokens=1 full-column calls
N_CPU_FFN_LAYERS = {"qwen": 24, "gemma": 26}   # M: 40-16 GPU layers, 48-22 GPU layers
LAYER_BYTES = {"qwen": 6417137664 / 12, "gemma": 2831056896 / 8}   # M: released bytes / layers
PHONE_CAPACITY_BYTES = 10_078_625_792          # M: phone available_bytes at first residency admission
PHONE_RELOAD_BYTES_PER_S = 270e6               # M: 3.2 GB HTP session load in ~12 s from phone flash
COALESCED_ROW_OVERHEAD = 0.15                  # E: extra phone time per additional row in one call
DISPATCH_GAP_S = 2.5                           # M: mean (ttft - prompt) of non-switch requests, treatment
LLAMA_TOTAL_S = 3.0                            # M: overlay request incl. tiny server load, at ~40 W


def calibrate(accounting: dict) -> dict:
    """Solve the per-layer host model from the measured batch-1 windows of the treatment arm."""
    rows = accounting["treatment"]["window_latency_energy"]

    def pick(model, batch, policy, fraction=0, cotenant=None):
        for row in rows:
            if (row["model"], row["active_batch"], row["policy"], row["fraction_ppm"]) == (model, batch, policy, fraction):
                if cotenant is None or row["cotenant_policy"] == cotenant:
                    return row
        raise KeyError((model, batch, policy, fraction))

    cal = {}
    for model in ("qwen", "gemma"):
        base = pick(model, 1, "baseline", 0, "none")
        phone = pick(model, 1, "phone", 1000000, "none")
        both = pick(model, 2, "baseline", 0, "baseline")
        n_cpu = N_CPU_FFN_LAYERS[model]
        n_phone = accounting["treatment"]["requests"][0] and {"qwen": 12, "gemma": 8}[model]
        rpc_ms = n_phone * T_RPC_MS[model]
        # latency: base = T + n_cpu*t ; phone = T + (n_cpu-n_phone)*t + rpc
        t_cpu = (base["ms_per_token"] - phone["ms_per_token"] + rpc_ms) / n_phone
        t_other = base["ms_per_token"] - n_cpu * t_cpu
        # energy (J): base = T*(Po+G) + n_cpu*t*(Pf+G); phone = T*(Po+G) + (n_cpu-n_phone)*t*(Pf+G) + rpc*(Pw+G)
        pf_plus_g = (base["host_j_per_slot_token"] - phone["host_j_per_slot_token"]
                     + rpc_ms / 1e3 * (P_CPU_WAIT_W + P_GPU_W)) / (n_phone * t_cpu / 1e3)
        po_plus_g = (base["host_j_per_slot_token"] - n_cpu * t_cpu / 1e3 * pf_plus_g) / (t_other / 1e3)
        cal[model] = {
            "t_other_ms": t_other, "t_cpu_ffn_ms_per_layer": t_cpu,
            "t_other_batch_increment_ms": both["ms_per_token"] - base["ms_per_token"],
            "p_cpu_ffn_w": pf_plus_g - P_GPU_W, "p_cpu_other_w": po_plus_g - P_GPU_W,
            "measured": {"b1_baseline_ms": base["ms_per_token"], "b1_phone_ms": phone["ms_per_token"],
                         "b1_baseline_j": base["host_j_per_slot_token"], "b1_phone_j": phone["host_j_per_slot_token"],
                         "b2_both_baseline_pass_ms": both["ms_per_token"], "b2_both_baseline_pass_j": both["host_j_per_slot_token"]},
        }
    return cal


def pass_cost(cal: dict, model: str, batch: int, phone_layers: int, mode: str) -> tuple[float, float]:
    """(ms, J) for one decode pass of `batch` rows; mode in none|splitrow|coalesced."""
    c = cal[model]
    n_cpu = N_CPU_FFN_LAYERS[model]
    t_other = c["t_other_ms"] + c["t_other_batch_increment_ms"] * (batch - 1)
    if mode == "none" or phone_layers == 0:
        phone_ms, cpu_layers = 0.0, n_cpu
    else:
        cpu_layers = n_cpu - phone_layers
        per_layer = T_RPC_MS[model]
        if mode == "splitrow":
            phone_ms = batch * phone_layers * per_layer
        else:
            phone_ms = phone_layers * per_layer * (1 + COALESCED_ROW_OVERHEAD * (batch - 1))
    cpu_ms = cpu_layers * c["t_cpu_ffn_ms_per_layer"]
    ms = t_other + cpu_ms + phone_ms
    joules = (t_other * (c["p_cpu_other_w"] + P_GPU_W) + cpu_ms * (c["p_cpu_ffn_w"] + P_GPU_W)
              + phone_ms * (P_CPU_WAIT_W + P_GPU_W)) / 1e3
    return ms, joules


# ---------------------------------------------------------------------------------------- simulator

class Sim:
    def __init__(self, requests: list[dict], cal: dict, loads: dict, switch_extra_s: float, cfg: dict):
        self.requests = sorted(copy.deepcopy(requests), key=lambda r: r["arrival_s"])
        self.cal, self.loads, self.switch_extra_s, self.cfg = cal, loads, switch_extra_s, cfg
        self.t = 0.0
        self.energy_j = 0.0
        self.resident = None
        self.active: list[dict] = []
        self.queue: list[dict] = []
        self.done: list[dict] = []
        self.switches = 0
        self.phone_tokens = 0
        self.tokens = 0
        self.mixed_tokens = 0
        self.log: list[dict] = []
        self.pending = list(self.requests)

    # ---- helpers
    def _arrive(self):
        while self.pending and self.pending[0]["arrival_s"] <= self.t:
            self.queue.append(self.pending.pop(0))

    def _spend(self, seconds: float, watts: float):
        self.t += seconds
        self.energy_j += seconds * watts

    def _kv_ok(self, cand: dict) -> bool:
        budget = self.cfg["kv_budget"][cand["model"]]
        used = sum(r["input_tokens"] + r["output_tokens"] for r in self.active)
        return used + cand["input_tokens"] + cand["output_tokens"] <= budget

    def _next_model(self):
        large = [r for r in self.queue if r["model"] != "llama"]
        if not large:
            return None
        if self.cfg["ordering"] == "affinity" and self.resident is not None:
            same = [r for r in large if r["model"] == self.resident]
            other = [r for r in large if r["model"] != self.resident]
            if same and (not other or self.t - min(r["arrival_s"] for r in other) < self.cfg["max_wait_s"]):
                return self.resident
            return min(other or same, key=lambda r: r["arrival_s"])["model"]
        return min(large, key=lambda r: r["arrival_s"])["model"]

    def _admit(self, model: str, limit: int):
        cands = sorted((r for r in self.queue if r["model"] == model), key=lambda r: r["arrival_s"])
        for cand in cands:
            if len(self.active) >= limit or not self._kv_ok(cand):
                break
            self.queue.remove(cand)
            cand["remaining"] = cand["output_tokens"]
            cand["start_s"] = self.t
            cand["decoded"] = 0
            # prefill stalls the batch (server processes the prompt before the new slot generates)
            self._spend(cand["server_prompt_ms"] / 1e3, P_PREFILL_W)
            self.active.append(cand)

    def _admit_replay(self):
        """Reproduce the observed admission: requests start in their actual order; a request and its
        observed decode partners form one wave; a partner that has not arrived yet joins on arrival."""
        order = self.cfg["replay_order"]
        while self.replay_pos < len(order):
            rid = order[self.replay_pos]
            cand = next((r for r in self.queue if r["request_id"] == rid), None)
            if cand is None:
                if any(r["request_id"] == rid for r in self.done + self.active):
                    self.replay_pos += 1
                    continue
                return  # not arrived yet
            if self.active and cand["request_id"] not in self.cfg["replay_partners"].get(self.active[0]["request_id"], ()):
                return
            if cand["model"] != self.resident:
                if self.active:
                    return
                self._switch(cand["model"])
            self.queue.remove(cand)
            cand["remaining"] = cand["output_tokens"]
            cand["start_s"] = self.t
            cand["decoded"] = 0
            self._spend(cand["server_prompt_ms"] / 1e3, P_PREFILL_W)
            self.active.append(cand)
            self.replay_pos += 1

    def _switch(self, model: str):
        load_s = self.loads[model]
        extra = self.switch_extra_s
        if self.cfg["layers"][model] * LAYER_BYTES[model] > self.cfg["layers_baseline"][model] * LAYER_BYTES[model]:
            reload_s = self.cfg["layers"][model] * LAYER_BYTES[model] / PHONE_RELOAD_BYTES_PER_S
            extra += max(0.0, reload_s - load_s)
        self._spend(load_s + extra, P_LOAD_W)
        self.resident = model
        self.switches += 1
        self._just_switched = True

    def _wave_limit(self, model: str) -> int:
        """Observed behaviour: after a switch the first request runs alone; otherwise up to max_batch
        requests that are already queued start together, and no back-fill while a wave runs."""
        return 1 if self.cfg.get("switch_solo", True) and self._just_switched else self.cfg["max_batch"][model]

    # ---- main loop
    def run(self) -> dict:
        self._just_switched = False
        self.replay_pos = 0
        while self.pending or self.queue or self.active:
            self._arrive()
            # overlay requests run on their own tiny server, concurrently; account a fixed cost
            for r in [r for r in self.queue if r["model"] == "llama"]:
                self.queue.remove(r)
                self.energy_j += LLAMA_TOTAL_S * 40.0
                r["end_s"] = max(self.t, r["arrival_s"]) + LLAMA_TOTAL_S
                self.done.append(r)
            if self.cfg["admission"] == "replay":
                self._admit_replay()
                if not self.active:
                    if self.pending:
                        self._spend(max(0.05, self.pending[0]["arrival_s"] - self.t), P_IDLE_W)
                        continue
                    break
                self._pass()
                continue
            if not self.active:
                model = self._next_model()
                if model is None:
                    if self.pending:
                        self._spend(max(0.0, self.pending[0]["arrival_s"] - self.t), P_IDLE_W)
                        continue
                    break
                if model != self.resident:
                    self._switch(model)
                else:
                    self._spend(DISPATCH_GAP_S, P_IDLE_W)
                self._admit(model, self._wave_limit(model))
                self._just_switched = False
                continue
            if self.cfg["admission"] == "backfill":
                self._admit(self.resident, self.cfg["max_batch"][self.resident])
            self._pass()
        duration = self.t
        waits = [r["start_s"] - r["arrival_s"] for r in self.done if r["model"] != "llama"]
        return {"duration_s": round(duration, 1), "host_kj": round(self.energy_j / 1e3, 1),
                "switches": self.switches, "tokens": self.tokens, "phone_tokens": self.phone_tokens,
                "phone_share": round(self.phone_tokens / max(1, self.tokens), 3),
                "mixed_tokens": self.mixed_tokens,
                "mean_queue_wait_s": round(sum(waits) / len(waits), 1) if waits else None,
                "max_queue_wait_s": round(max(waits), 1) if waits else None}

    def _pass(self):
        model = self.resident
        batch = len(self.active)
        layers = self.cfg["layers"][model]
        phone_mode = self.cfg["phone"]
        for r in self.active:
            r.setdefault("probe_left", self.cfg["probe_tokens"])
            if batch >= 2 and phone_mode == "current":
                r.setdefault("mixed_left", self.cfg["pair_mixed_tokens"])
        if phone_mode == "none":
            ms, j = pass_cost(self.cal, model, batch, 0, "none")
            phone_rows = 0
        elif phone_mode == "current":
            if batch == 1:
                r = self.active[0]
                if r["probe_left"] > 0:
                    ms, j = pass_cost(self.cal, model, 1, 0, "none")
                    r["probe_left"] -= 1
                    phone_rows = 0
                else:
                    ms, j = pass_cost(self.cal, model, 1, layers, "splitrow")
                    phone_rows = 1
            else:
                # mixed policies cannot batch: one baseline pass + one phone pass per token, for the
                # measured number of mixed tokens per member, then everyone falls back to baseline
                if any(r.get("mixed_left", 0) > 0 for r in self.active):
                    ms_a, j_a = pass_cost(self.cal, model, 1, 0, "none")
                    ms_b, j_b = pass_cost(self.cal, model, 1, layers, "splitrow")
                    ms, j = ms_a + ms_b, j_a + j_b
                    for r in self.active:
                        r["mixed_left"] = max(0, r.get("mixed_left", 0) - 1)
                        r["probe_left"] = max(0, r["probe_left"] - 1)
                    self.mixed_tokens += batch
                    phone_rows = 1
                else:
                    ms, j = pass_cost(self.cal, model, batch, 0, "none")
                    phone_rows = 0
        else:
            mode = "splitrow" if phone_mode == "coherent_splitrow" else "coalesced"
            probing = any(r["probe_left"] > 0 for r in self.active)
            if probing:
                ms, j = pass_cost(self.cal, model, batch, 0, "none")
                for r in self.active:
                    r["probe_left"] = max(0, r["probe_left"] - 1)
                phone_rows = 0
            else:
                ms, j = pass_cost(self.cal, model, batch, layers, mode)
                phone_rows = batch
        self.t += ms / 1e3
        self.energy_j += j
        self.tokens += batch
        self.phone_tokens += phone_rows
        finished = []
        for r in self.active:
            r["remaining"] -= 1
            r["decoded"] += 1
            if r["remaining"] <= 0:
                r["end_s"] = self.t
                finished.append(r)
        for r in finished:
            self.active.remove(r)
            self.done.append(r)


def base_config() -> dict:
    return {"admission": "wave", "ordering": "fifo", "max_wait_s": 1e9, "phone": "current",
            "layers": {"qwen": 12, "gemma": 8}, "layers_baseline": {"qwen": 12, "gemma": 8},
            "max_batch": {"qwen": 2, "gemma": 2}, "kv_budget": {"qwen": 4096, "gemma": 32768},
            "probe_tokens": 16, "pair_mixed_tokens": 44, "switch_solo": True,
            "replay_order": [], "replay_partners": {}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accounting", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    accounting = json.loads(args.accounting.read_text())
    cal = calibrate(accounting)
    requests = accounting["treatment"]["requests"]
    for row in requests:
        if row["server_prompt_ms"] is None:
            row["server_prompt_ms"] = 40.0
    loads = {}
    for model, role in (("qwen", "hot"), ("gemma", "cold")):
        values = [m["load_s"] for m in accounting["treatment"]["model_loads"] if m["role"] == role]
        loads[model] = sum(values) / len(values)
    switch_extra_s = accounting["treatment"].get("switch_extra_s") or 9.0
    order = [r["request_id"] for r in sorted(requests, key=lambda r: (r["scheduled_start_s"], r["arrival_s"]))]
    partners = {r["request_id"]: tuple(p for p in r["decode_partners"].split(",") if p) for r in requests}

    def run(name, **overrides):
        cfg = base_config()
        cfg["admission"] = "replay"
        cfg["replay_order"], cfg["replay_partners"] = order, partners
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key] = {**cfg[key], **value}
            else:
                cfg[key] = value
        result = Sim(requests, cal, loads, switch_extra_s, cfg).run()
        result["name"] = name
        result["config"] = cfg
        return result

    measured = {"treatment": {"duration_s": accounting["treatment"]["duration_s"],
                              "host_kj": sum(v for k, v in accounting["treatment"]["host_kj"].items() if k != "phone-system")},
                "baseline": {"duration_s": accounting["baseline"]["duration_s"],
                             "host_kj": sum(v for k, v in accounting["baseline"]["host_kj"].items() if k != "phone-system")}}
    validation = {"treatment_replay": run("validate:treatment (observed admission, current phone)"),
                  "baseline_replay": run("validate:baseline (observed admission, no phone)", phone="none"),
                  "wave_rule": run("reference: wave rule (first request after a switch alone, pairs otherwise)", admission="wave")}
    # Policy-only variants keep the observed admission (replay). Admission/ordering variants cannot be
    # replayed, so they are compared with the simulator's own "wave" reference as well as with the replay.
    variants = [
        run("A  coherent phone policy in pairs (split-row calls) [replay admission]", phone="coherent_splitrow"),
        run("A' coherent phone policy in pairs (coalesced calls) [replay admission]", phone="coherent_coalesced"),
        run("D  re-provision phone per resident model, 18 Qwen / 26 Gemma layers [replay]", layers={"qwen": 18, "gemma": 26}),
        run("D+A' re-provision + coherent coalesced [replay]", layers={"qwen": 18, "gemma": 26}, phone="coherent_coalesced"),
        run("E  drop probing (probe_tokens 4, no mixed probing in pairs) [replay]", probe_tokens=4, pair_mixed_tokens=4),
        run("C  model-affinity ordering, max wait 900 s [wave admission]", admission="wave", ordering="affinity", max_wait_s=900),
        run("C  model-affinity ordering, max wait 1800 s [wave admission]", admission="wave", ordering="affinity", max_wait_s=1800),
        run("C  model-affinity ordering, unbounded wait [wave admission]", admission="wave", ordering="affinity", max_wait_s=1e9),
        run("C+B affinity (1800 s) + back-fill B<=2, phone as today", ordering="affinity", max_wait_s=1800, admission="backfill", switch_solo=False),
        run("B  work-conserving back-fill, batch<=2, phone as today [fifo]", admission="backfill", switch_solo=False),
        run("B+A' back-fill B<=2 + coherent coalesced phone", admission="backfill", switch_solo=False, phone="coherent_coalesced"),
        run("C+B+A' affinity + back-fill B<=2 + coherent coalesced", ordering="affinity", max_wait_s=900, admission="backfill",
            switch_solo=False, phone="coherent_coalesced"),
        run("F  back-fill batch<=4, no phone", admission="backfill", switch_solo=False, phone="none",
            max_batch={"qwen": 4, "gemma": 4}),
        run("F+A' back-fill batch<=4 + coherent coalesced phone", admission="backfill", switch_solo=False,
            phone="coherent_coalesced", max_batch={"qwen": 4, "gemma": 4}),
        run("F+A'+D back-fill B<=4 + coalesced + 18/26 layers", admission="backfill", switch_solo=False,
            phone="coherent_coalesced", max_batch={"qwen": 4, "gemma": 4}, layers={"qwen": 18, "gemma": 26}),
        run("ALL: affinity + back-fill B<=4 + coalesced + 18/26 layers + probe 4", ordering="affinity", max_wait_s=900,
            admission="backfill", switch_solo=False, phone="coherent_coalesced", max_batch={"qwen": 4, "gemma": 4},
            layers={"qwen": 18, "gemma": 26}, probe_tokens=4),
        run("ALL with Qwen KV 8192", ordering="affinity", max_wait_s=900, admission="backfill", switch_solo=False,
            phone="coherent_coalesced", max_batch={"qwen": 4, "gemma": 4}, layers={"qwen": 18, "gemma": 26},
            probe_tokens=4, kv_budget={"qwen": 8192, "gemma": 32768}),
    ]
    pass_table = []
    for model in ("qwen", "gemma"):
        for batch in (1, 2, 4):
            for layers, mode in ((0, "none"), ({"qwen": 12, "gemma": 8}[model], "splitrow"),
                                 ({"qwen": 12, "gemma": 8}[model], "coalesced"), ({"qwen": 18, "gemma": 26}[model], "coalesced")):
                ms, j = pass_cost(cal, model, batch, layers, mode)
                pass_table.append({"model": model, "batch": batch, "phone_layers": layers, "mode": mode,
                                   "pass_ms": round(ms, 1), "pass_j": round(j, 2),
                                   "ms_per_token": round(ms / batch, 1), "j_per_token": round(j / batch, 2)})
    payload = {"calibration": cal, "constants": {
        "P_GPU_W": P_GPU_W, "P_CPU_WAIT_W": P_CPU_WAIT_W, "P_LOAD_W": P_LOAD_W, "P_PREFILL_W": P_PREFILL_W,
        "P_IDLE_W": P_IDLE_W, "T_RPC_MS": T_RPC_MS, "N_CPU_FFN_LAYERS": N_CPU_FFN_LAYERS,
        "loads_s": loads, "switch_extra_s": switch_extra_s, "COALESCED_ROW_OVERHEAD": COALESCED_ROW_OVERHEAD,
        "PHONE_RELOAD_BYTES_PER_S": PHONE_RELOAD_BYTES_PER_S},
        "measured": measured, "validation": validation, "pass_table": pass_table, "variants": variants}
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print("calibration:", json.dumps({m: {k: round(v, 2) for k, v in c.items() if k != 'measured'} for m, c in cal.items()}))
    print(f"{'variant':<74} {'dur s':>7} {'host kJ':>8} {'dE%':>6} {'sw':>3} {'phone%':>7} {'mixed':>6} {'wait':>7}")
    ref = validation["treatment_replay"]
    wave = validation["wave_rule"]
    for label, row in (("measured treatment", measured["treatment"]), ("measured baseline", measured["baseline"])):
        print(f"{label:<74} {row['duration_s']:7.0f} {row['host_kj']:8.1f}")
    for row in list(validation.values()) + variants:
        base = wave if "[replay" not in row["name"] and "validate" not in row["name"] else ref
        row["reference"] = "wave_rule" if base is wave else "treatment_replay"
        row["delta_energy_pct_vs_reference"] = round(100 * (row["host_kj"] / base["host_kj"] - 1), 1)
        row["delta_duration_pct_vs_reference"] = round(100 * (row["duration_s"] / base["duration_s"] - 1), 1)
        row["delta_energy_pct_vs_replay"] = round(100 * (row["host_kj"] / ref["host_kj"] - 1), 1)
        row["delta_duration_pct_vs_replay"] = round(100 * (row["duration_s"] / ref["duration_s"] - 1), 1)
        print(f"{row['name']:<74} {row['duration_s']:7.0f} {row['host_kj']:8.1f} {row['delta_energy_pct_vs_reference']:+6.1f} "
              f"{row['switches']:>3} {100 * row['phone_share']:6.0f}% {row['mixed_tokens']:>6} {row['mean_queue_wait_s']:>7}  vs {row['reference']}")
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print("pass table (ms, J per pass; per token):")
    for row in pass_table:
        print(f"  {row['model']:<6} B{row['batch']} layers {row['phone_layers']:>2} {row['mode']:<9} {row['pass_ms']:>7} ms {row['pass_j']:>7} J  -> {row['ms_per_token']:>6} ms/tok {row['j_per_token']:>6} J/tok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
