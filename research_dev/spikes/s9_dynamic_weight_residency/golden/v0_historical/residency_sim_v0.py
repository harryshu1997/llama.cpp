#!/usr/bin/env python3
"""S9 deterministic offline residency simulator (V0).

Isolated research tool -- NOT integrated into llama-server. Integer-microsecond
time, stable total event order (heap keyed by (t_us, priority_rank, seq)). No wall
clock, no PRNG for durations. Reads a sim_config (schema-validated separately),
sweeps link goodput, runs eight baselines, and emits a sim_run_manifest dict plus a
byte-identical deterministic_replay_sha256. Every quantity is an integer.

See SIMULATOR_SPEC.md, PREFETCH_POLICY.md, WEIGHT_RESIDENCY_CONTRACT.md.
"""
import hashlib, heapq, json, os, sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
US = 1000000

# priority ranks for the event heap total order (lower = earlier at equal t)
PR_FAULT, PR_ARRIVAL, PR_ACT, PR_STAGE, PR_EXEC, PR_EVICT = 0, 1, 2, 3, 4, 5


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha_over_text(text):
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_over_bytes(b):
    return "sha256:" + hashlib.sha256(b).hexdigest()


class FrameSequencer:
    """TransportFrame reader gate (TRANSPORT_CONTRACT section 7): rejects duplicate,
    reordered (seq gap), and stale-epoch frames fail-closed. Idempotency keys make a
    retried mutating frame a no-op. Per (connection, channel) monotone seq."""
    def __init__(self):
        self.next_seq = {}          # (conn, channel) -> expected seq
        self.applied = set()        # idempotency keys already applied
        self.boot_epoch = {}        # conn -> current boot epoch
        self.residency_epoch = {}   # conn -> current residency epoch

    def accept(self, frame):
        conn = frame["request_id"].split(":")[0] if ":" in frame["request_id"] else frame.get("conn", "c0")
        conn = frame.get("conn", conn)
        ch = frame["channel"]
        key = (conn, ch)
        exp = self.next_seq.get(key, frame["seq"])   # first frame sets the baseline
        # stale epoch
        be = self.boot_epoch.get(conn)
        if be is not None and frame["boot_epoch"] < be:
            return False, "stale_epoch"
        re = self.residency_epoch.get(conn)
        if re is not None and frame.get("residency_epoch") is not None and frame["residency_epoch"] < re:
            return False, "stale_epoch"
        # duplicate seq
        if frame["seq"] < exp:
            return False, "duplicate_message"
        # reordered (gap)
        if frame["seq"] > exp:
            return False, "reordered_message"
        # duplicate idempotency key on a mutating op
        if frame["idempotency_key"] in self.applied:
            return False, "duplicate_message"
        # accept
        self.next_seq[key] = frame["seq"] + 1
        self.applied.add(frame["idempotency_key"])
        self.boot_epoch[conn] = max(be or 0, frame["boot_epoch"])
        if frame.get("residency_epoch") is not None:
            self.residency_epoch[conn] = max(re or 0, frame["residency_epoch"])
        return True, "ok"


def xfer_us(nbytes, goodput):
    return (nbytes * US + goodput - 1) // goodput


def rate_us(nbytes, rate):
    return (nbytes * US + rate - 1) // rate


def pctl(sorted_vals, q_num, q_den):
    if not sorted_vals:
        return 0
    n = len(sorted_vals)
    r = (q_num * n + q_den - 1) // q_den   # nearest-rank, integer
    r = max(1, min(n, r))
    return sorted_vals[r - 1]


class Ledger:
    """Exact LPDDR partition: weights+derived+scratch+activations+state+free == total."""
    def __init__(self, total):
        self.total = total
        self.weights = 0
        self.derived = 0
        self.scratch = 0
        self.activations = 0
        self.state = 0

    @property
    def free(self):
        return self.total - (self.weights + self.derived + self.scratch + self.activations + self.state)

    def check(self):
        used = self.weights + self.derived + self.scratch + self.activations + self.state
        assert 0 <= used <= self.total, f"ledger overflow used={used} total={self.total}"
        assert self.free == self.total - used
        for v in (self.weights, self.derived, self.scratch, self.activations, self.state):
            assert v >= 0

    def can_fit(self, w, d, s):
        return self.free >= w + d + s

    def reserve_weights(self, w, d, s):
        self.weights += w
        self.derived += d
        self.scratch += s
        self.check()

    def free_weights(self, w, d, s):
        self.weights -= w
        self.derived -= d
        self.scratch -= s
        self.check()


class ResEntry:
    """Residency state for one weight set on one device."""
    def __init__(self, ws_id, model):
        self.ws_id = ws_id
        self.model = model
        self.state = "ABSENT"
        self.ready_at = None
        self.generation = 0
        self.pipeline_ver = 0     # lazy-invalidation for delayed/cancelled pipelines
        self.reserved = (0, 0, 0)
        self.pinned = 0           # active state leases depending on this residency
        self.last_use = -1
        self.uses = 0
        self.resumed = False
        self.evict_after_unpin = False   # drain-before-eviction: staled while a lease was live


class Sim:
    def __init__(self, cfg, goodput, policy):
        self.cfg = cfg
        self.policy = policy
        self.t = 0
        self.seq = 0
        self.heap = []
        # single-phone-first, but supports many
        self.phones = {p["device_id"]: p for p in cfg["phones"]}
        self.n_phones = len(self.phones)
        controller = cfg["usb_shared_controller_bytes_per_s"]
        self.eff_goodput = {d: min(goodput, controller // max(1, self.n_phones)) for d in self.phones}
        self.rtt = {d: 500 for d in self.phones}
        self.ledger = {d: Ledger(p["lpddr_total_bytes"]) for d, p in self.phones.items()}
        self.htp_free = {d: p["htp_lane_slots"] for d, p in self.phones.items()}
        self.gpu_free = {d: p["gpu_lane_slots"] for d, p in self.phones.items()}
        self.lane_q = {(d, b): deque() for d in self.phones for b in ("htp", "gpu")}
        self.thermal_ok = {d: p["thermal_eligible"] for d, p in self.phones.items()}
        self.boot_epoch = {d: p["boot_epoch"] for d, p in self.phones.items()}
        # link state (bulk in-flight is delayable by activations)
        self.link_bulk = {d: None for d in self.phones}   # {"ready_ev": (device,ws), "ver": int}
        self.link_act_busy = {d: 0 for d in self.phones}
        self.models = {m["model_id"]: m for m in cfg["models"]}
        self.ws_model = {m["weight_set_id"]: m for m in cfg["models"]}
        self.res = {}                                     # (device, ws_id) -> ResEntry
        self.server_gpu_free = cfg["server"]["gpu_lane_slots"]
        self.server_hbm_free = cfg["server"]["hbm_credit_bytes"]
        self.server_q = []
        self.interf = cfg["interference"]
        self.faults = {}                                  # keyed by (kind, target) -> at_us
        for f in cfg.get("failure_schedule", []):
            self.faults.setdefault((f["kind"], f["target"]), f["at_us"])
        self.pp = cfg["policy_params"]
        self.predict = {}                                 # ws_id -> ewma reuse (permille units)
        # metrics
        self.m = dict(completed=0, dispatched_to_phone=0, fell_back_to_server=0, weight_misses=0,
                      server_gpu_us_used=0, server_gpu_us_freed=0, server_hbm_bytes_freed=0,
                      transfer_bytes=0, prepare_us_total=0, evictions=0, deadline_hits=0,
                      deadline_misses=0, causal_score_total=0, waited_for_weights=0)
        self.completions = []
        # instrumentation for tests
        self.preemptions = 0
        self.quarantined = set()
        self.resumed_sets = set()
        self.live_state_evictions = 0
        self.dispatched_from_on_disk = 0

    # ---- event heap ----
    def push(self, t, pr, kind, payload):
        self.seq += 1
        heapq.heappush(self.heap, (t, pr, self.seq, kind, payload))

    def resolve(self, dev, ws_id):
        key = (dev, ws_id)
        if key not in self.res:
            self.res[key] = ResEntry(ws_id, self.ws_model[ws_id])
        return self.res[key]

    # ---- link with activation-preempts-bulk ----
    def send_activation(self, dev, nbytes):
        eff = self.eff_goodput[dev]
        dur = xfer_us(nbytes, eff) + self.rtt[dev]
        start = max(self.t, self.link_act_busy[dev])
        finish = start + dur
        self.link_act_busy[dev] = finish
        bulk = self.link_bulk[dev]
        if bulk is not None:
            # the in-flight weight transfer is delayed by the activation occupancy
            self.preemptions += 1
            r = self.resolve(dev, bulk["ws_id"])
            r.pipeline_ver += 1                       # invalidate the old stage event
            bulk["ready_shift"] = bulk.get("ready_shift", 0) + dur
            # reschedule the pending transfer_done later by dur
            self.push(bulk["done_time"] + bulk["ready_shift"], PR_STAGE, "res_stage",
                      {"dev": dev, "ws_id": bulk["ws_id"], "stage": "transfer_done",
                       "ver": r.pipeline_ver})
        return finish

    # ---- residency pipeline ----
    def start_prefetch(self, dev, ws_id):
        r = self.resolve(dev, ws_id)
        if r.state not in ("ABSENT",) or ws_id in self.quarantined:
            return
        if self.link_bulk[dev] is not None:
            return                                     # one bulk transfer per link; a later miss retries
        if (self.faults.get(("unknown_profile", dev)) is not None and
                self.faults[("unknown_profile", dev)] <= self.t):
            return
        model = self.ws_model[ws_id]
        phone = self.phones[dev]
        w = model["canonical_bytes"]
        # room? try to evict per policy, lease-safe
        d_htp, d_gpu = model["derived_bytes_htp"], model["derived_bytes_gpu"]
        d = max(d_htp, d_gpu)
        s = model["scratch_bytes"]
        if not self.ledger[dev].can_fit(w, d, s):
            if not self.try_evict(dev, w, d, s):
                return                                 # cannot make room; stays ABSENT -> misses fall back
        r.state = "RECEIVING"
        eff = self.eff_goodput[dev]
        tdur = xfer_us(w, eff) + self.rtt[dev]
        self.m["transfer_bytes"] += w
        done = self.t + tdur
        r.pipeline_ver += 1
        self.link_bulk[dev] = {"ws_id": ws_id, "done_time": done, "ver": r.pipeline_ver}
        self.push(done, PR_STAGE, "res_stage", {"dev": dev, "ws_id": ws_id,
                                                "stage": "transfer_done", "ver": r.pipeline_ver})

    def stage(self, dev, ws_id, stage, ver):
        r = self.resolve(dev, ws_id)
        if ver != r.pipeline_ver:
            return                                     # stale (delayed/cancelled) event
        model = self.ws_model[ws_id]
        phone = self.phones[dev]
        if stage == "transfer_done":
            self.link_bulk[dev] = None
            # fault: partial transfer -> resume from last verified chunk (recover, not restart)
            at = self.faults.get(("partial_transfer", ws_id))
            if at is not None and at <= self.t and not r.resumed:
                r.resumed = True
                self.resumed_sets.add(ws_id)
                resume_bytes = model["canonical_bytes"] // 4
                r.pipeline_ver += 1
                self.push(self.t + xfer_us(resume_bytes, self.eff_goodput[dev]), PR_STAGE,
                          "res_stage", {"dev": dev, "ws_id": ws_id, "stage": "transfer_done",
                                        "ver": r.pipeline_ver})
                return
            # durable staging: write the received bytes to UFS before VERIFYING (the fsync boundary)
            udur = rate_us(model["canonical_bytes"], phone["ufs_write_bytes_per_s"])
            r.pipeline_ver += 1
            self.push(self.t + udur, PR_STAGE, "res_stage",
                      {"dev": dev, "ws_id": ws_id, "stage": "ufs_done", "ver": r.pipeline_ver})
        elif stage == "ufs_done":
            r.state = "VERIFYING"
            # fault: hash mismatch -> QUARANTINED, no publish
            at = self.faults.get(("hash_mismatch", ws_id))
            if at is not None and at <= self.t:
                r.state = "QUARANTINED"
                self.quarantined.add(ws_id)
                return
            vdur = rate_us(model["canonical_bytes"], phone["verify_bytes_per_s"])
            r.pipeline_ver += 1
            self.push(self.t + vdur, PR_STAGE, "res_stage",
                      {"dev": dev, "ws_id": ws_id, "stage": "verify_done", "ver": r.pipeline_ver})
        elif stage == "verify_done":
            r.state = "VERIFIED_ON_DISK"
            mdur = rate_us(model["canonical_bytes"], phone["materialize_bytes_per_s"])
            r.pipeline_ver += 1
            self.push(self.t + mdur, PR_STAGE, "res_stage",
                      {"dev": dev, "ws_id": ws_id, "stage": "materialize_done", "ver": r.pipeline_ver})
        elif stage == "materialize_done":
            r.state = "LPDDR_READY"
            back = model["eligible_backends"][0]
            dbytes = model["derived_bytes_htp"] if back == "htp" else model["derived_bytes_gpu"]
            pr_rate = phone["prepare_htp_bytes_per_s"] if back == "htp" else phone["prepare_gpu_bytes_per_s"]
            pdur = rate_us(dbytes, pr_rate) if dbytes > 0 else 0
            self.m["prepare_us_total"] += pdur
            r.state = "PREPARING_HTP" if back == "htp" else "PREPARING_GPU"
            r.pipeline_ver += 1
            self.push(self.t + pdur, PR_STAGE, "res_stage",
                      {"dev": dev, "ws_id": ws_id, "stage": "prepare_done", "ver": r.pipeline_ver})
        elif stage == "prepare_done":
            r.state = "WARMING"
            r.pipeline_ver += 1
            self.push(self.t + phone["warmup_us"], PR_STAGE, "res_stage",
                      {"dev": dev, "ws_id": ws_id, "stage": "warm_done", "ver": r.pipeline_ver})
        elif stage == "warm_done":
            # publish READY: reserve the physical bytes now (fail-closed if it no longer fits)
            back = model["eligible_backends"][0]
            d = model["derived_bytes_htp"] if back == "htp" else model["derived_bytes_gpu"]
            w, s = model["canonical_bytes"], model["scratch_bytes"]
            if not self.ledger[dev].can_fit(w, d, s):
                r.state = "ABSENT"
                return
            self.ledger[dev].reserve_weights(w, d, s)
            r.reserved = (w, d, s)
            r.state = "READY_HTP" if back == "htp" else "READY_GPU"
            r.ready_at = self.t

    def is_ready(self, dev, ws_id):
        r = self.res.get((dev, ws_id))
        return r is not None and r.state in ("READY_HTP", "READY_GPU")

    def next_use(self, ws_id):
        """clairvoyant foreknowledge: earliest future arrival for ws_id (or 'never')."""
        nxt = None
        for req in self.cfg["workload"]["requests"]:
            if req["weight_set_id"] == ws_id and req["arrival_us"] > self.t:
                if nxt is None or req["arrival_us"] < nxt:
                    nxt = req["arrival_us"]
        return nxt if nxt is not None else (1 << 62)   # never reused -> farthest

    def free_residency(self, dev, ws, r):
        """Reclaim a residency. Routing ALL weight frees through here makes
        live_state_evictions a real guard: freeing a pinned set is an invariant
        violation and is counted (it must never happen -- callers skip pinned)."""
        if r.pinned > 0:
            self.live_state_evictions += 1
        rw, rd, rs = r.reserved
        self.ledger[dev].free_weights(rw, rd, rs)
        r.state = "ABSENT"
        r.reserved = (0, 0, 0)
        r.ready_at = None
        r.evict_after_unpin = False

    # ---- eviction (lease-safe: never evict a pinned/live-state set) ----
    def try_evict(self, dev, w, d, s):
        min_hold, ttl = self.pp["min_hold_us"], self.pp["ttl_us"]
        cands = []
        for (dd, ws), r in sorted(self.res.items()):
            if dd != dev or r.state not in ("READY_HTP", "READY_GPU"):
                continue
            if r.pinned > 0:
                continue                               # lease-safe: never evict live state
            cands.append((dd, ws, r))
        if not cands:
            return False
        # hysteresis: prefer sets held past min_hold; fall back to fresh ones only if forced
        aged = [c for c in cands if c[2].ready_at is not None and (self.t - c[2].ready_at) >= min_hold]
        pool = aged if aged else cands
        if self.policy == "clairvoyant":
            pool.sort(key=lambda c: (-self.next_use(c[1]), c[1]))       # Belady: farthest next use first
        elif self.policy == "lfu_ttl":
            pool.sort(key=lambda c: (0 if (self.t - c[2].last_use) > ttl else 1,
                                     c[2].uses, c[2].last_use, c[1]))    # expired-first, then least-used
        else:                                          # lru / fastest_ready / relief_predictive / static
            pool.sort(key=lambda c: (c[2].last_use, c[1]))
        dd, ws, r = pool[0]
        self.free_residency(dd, ws, r)                 # DRAINING -> EVICTING -> ABSENT (not pinned)
        self.m["evictions"] += 1
        if self.ledger[dev].can_fit(w, d, s):
            return True
        return self.try_evict(dev, w, d, s)

    # ---- server ----
    def run_server(self, req):
        cls = req["service_class"]
        gpu_us = self.cfg["server"]["per_class_gpu_us"].get(cls, 1000)
        hbm = self.cfg["server"]["per_class_hbm_bytes"].get(cls, 0)
        self.m["server_gpu_us_used"] += gpu_us
        if self.server_gpu_free >= 1 and self.server_hbm_free >= hbm:
            self.server_gpu_free -= 1
            self.server_hbm_free -= hbm
            self.push(self.t + gpu_us, PR_EXEC, "exec_done",
                      {"where": "server", "req": req, "hbm": hbm, "start": self.t})
        else:
            self.server_q.append((req, gpu_us, hbm))

    def server_release(self, hbm):
        self.server_gpu_free += 1
        self.server_hbm_free += hbm
        if self.server_q:
            req, gpu_us, hbm2 = self.server_q.pop(0)
            if self.server_gpu_free >= 1 and self.server_hbm_free >= hbm2:
                self.server_gpu_free -= 1
                self.server_hbm_free -= hbm2
                self.push(self.t + gpu_us, PR_EXEC, "exec_done",
                          {"where": "server", "req": req, "hbm": hbm2, "start": self.t})
            else:
                self.server_q.insert(0, (req, gpu_us, hbm2))

    # ---- phone dispatch (lane-bounded) ----
    def dispatch_phone(self, dev, req):
        """Returns True iff dispatched to the phone. Fail-closed: never dispatch a
        non-READY set; require activation+state to fit the ledger."""
        ws = req["weight_set_id"]
        if not self.is_ready(dev, ws):
            self.dispatched_from_on_disk += 1          # GUARD: on-disk/in-pipeline is never dispatchable
            return False
        act, st = req["input_bytes"], req["state_bytes"]
        if not self.ledger[dev].can_fit(0, 0, act + st):
            return False                               # no RAM for activation+state -> caller falls back
        model = self.models[req["model_id"]]
        back = model["eligible_backends"][0]
        r = self.resolve(dev, ws)
        self.ledger[dev].activations += act
        self.ledger[dev].state += st
        self.ledger[dev].check()
        r.pinned += 1                                  # pins residency: no live-state eviction
        r.last_use = self.t
        r.uses += 1
        in_done = self.send_activation(dev, req["input_bytes"])   # preempts in-flight bulk
        comp = model["phone_compute_us"]
        if self.interf["measured"]:
            comp = comp * self.interf["htp_gpu_slowdown_permille"] // 1000
        # bounded lane: acquire a slot on activation arrival; queue if busy
        self.push(in_done, PR_EXEC, "lane_arrive",
                  {"dev": dev, "back": back, "req": req, "comp": comp, "act": act, "st": st})
        cls = req["service_class"]
        self.m["server_gpu_us_freed"] += self.cfg["server"]["per_class_gpu_us"].get(cls, 0)
        self.m["server_hbm_bytes_freed"] += self.cfg["server"]["per_class_hbm_bytes"].get(cls, 0)
        return True

    def lane_arrive(self, payload):
        dev, back = payload["dev"], payload["back"]
        free = self.htp_free if back == "htp" else self.gpu_free
        if free[dev] >= 1:
            free[dev] -= 1
            self.push(self.t + payload["comp"], PR_EXEC, "exec_done",
                      {"where": dev, "req": payload["req"], "act": payload["act"],
                       "st": payload["st"], "back": back})
        else:
            self.lane_q[(dev, back)].append(payload)   # bounded lane: wait for a slot

    def exec_done(self, payload):
        req = payload["req"]
        where = payload["where"]
        self.m["completed"] += 1
        self.completions.append(self.t - req["arrival_us"])
        if req["deadline_us"] is not None:
            if self.t <= req["deadline_us"]:
                self.m["deadline_hits"] += 1
            else:
                self.m["deadline_misses"] += 1
        if where == "server":
            self.server_release(payload["hbm"])
            return
        dev, back = where, payload["back"]
        self.ledger[dev].activations -= payload["act"]
        self.ledger[dev].state -= payload["st"]
        self.ledger[dev].check()
        r = self.resolve(dev, req["weight_set_id"])
        r.pinned -= 1
        if r.pinned == 0 and r.evict_after_unpin:      # drain-before-eviction finished
            self.free_residency(dev, req["weight_set_id"], r)
            self.m["evictions"] += 1
        free = self.htp_free if back == "htp" else self.gpu_free
        free[dev] += 1
        q = self.lane_q[(dev, back)]
        if q:
            nxt = q.popleft()
            free[dev] -= 1
            self.push(self.t + nxt["comp"], PR_EXEC, "exec_done",
                      {"where": dev, "req": nxt["req"], "act": nxt["act"], "st": nxt["st"], "back": back})

    # ---- causal score (relief_predictive) ----
    def causal_score(self, dev, req):
        model = self.models[req["model_id"]]
        cls = req["service_class"]
        relief = self.cfg["server"]["per_class_gpu_us"].get(cls, 0)
        reuse = self.predict.get(req["weight_set_id"], 0)          # permille
        expected_reuse = relief * reuse // 1000
        avoided_cold = (xfer_us(model["canonical_bytes"], self.eff_goodput[dev]) +
                        rate_us(model["canonical_bytes"], self.phones[dev]["verify_bytes_per_s"]))
        transfer_cost = xfer_us(model["canonical_bytes"], self.eff_goodput[dev])
        prep = model["derived_bytes_gpu"]
        prep_cost = rate_us(prep, self.phones[dev]["prepare_gpu_bytes_per_s"]) if prep else 0
        risk = 0 if self.thermal_ok[dev] else 10 ** 9
        return relief + avoided_cold + expected_reuse - transfer_cost - prep_cost - risk

    # ---- eligibility (hard gates) ----
    def eligible(self, dev, req):
        model = self.models[req["model_id"]]
        # unknown profile / soc
        if self.faults.get(("unknown_profile", dev)) is not None and self.faults[("unknown_profile", dev)] <= self.t:
            return False, "unknown_profile"
        # unsupported backend for the model
        back = model["eligible_backends"][0]
        if back not in ("htp", "gpu"):
            return False, "unsupported_backend"
        if self.faults.get(("unsupported_backend", req["weight_set_id"])) is not None and \
                self.faults[("unsupported_backend", req["weight_set_id"])] <= self.t:
            return False, "unsupported_backend"
        # arbitrary-model gate
        if not model["partial_load_supported"] and model["canonical_bytes"] > self.ledger[dev].total:
            return False, "insufficient_ram"
        # thermal
        if not self.thermal_ok[dev]:
            return False, "thermal"
        return True, "ok"

    # ---- policy: on arrival ----
    def on_arrival(self, req):
        dev = next(iter(self.phones))     # single-phone-first placement
        ws = req["weight_set_id"]
        if ws in self.predict:
            self.predict[ws] = (self.predict[ws] * (1000 - self.pp["predictor_ewma_permille"]) +
                                1000 * self.pp["predictor_ewma_permille"]) // 1000
        else:
            self.predict[ws] = self.pp["predictor_ewma_permille"]

        if self.policy == "server_only":
            self.m["fell_back_to_server"] += 1
            self.run_server(req)
            return

        ok, reason = self.eligible(dev, req)

        if self.policy == "per_request_fetch":
            # DIAGNOSTIC losing baseline: fetch then WAIT (blocks on the critical path)
            if not ok:
                self.m["fell_back_to_server"] += 1
                self.run_server(req)
                return
            if self.is_ready(dev, ws):
                if self.dispatch_phone(dev, req):
                    self.m["dispatched_to_phone"] += 1
                else:
                    self.m["fell_back_to_server"] += 1
                    self.run_server(req)
            else:
                self.start_prefetch(dev, ws)
                self.m["waited_for_weights"] += 1
                self.push_wait(dev, req)               # dispatched/fell-back counted at wake
            return

        # cache / static / clairvoyant: NEVER wait; hit -> phone, miss -> server + bg prefetch
        dispatched = False
        if ok and self.is_ready(dev, ws):
            if self.policy == "relief_predictive":
                self.m["causal_score_total"] += self.causal_score(dev, req)
            dispatched = self.dispatch_phone(dev, req)
            if dispatched:
                self.m["dispatched_to_phone"] += 1
        if not dispatched:
            if not self.is_ready(dev, ws):
                self.m["weight_misses"] += 1
            self.m["fell_back_to_server"] += 1
            self.run_server(req)
            if (self.policy in ("lru", "lfu_ttl", "fastest_ready", "relief_predictive")
                    and ok and not self.is_ready(dev, ws)):
                if self.policy == "relief_predictive":
                    sc = self.causal_score(dev, req)
                    self.m["causal_score_total"] += sc
                    if sc <= 0:
                        return
                self.start_prefetch(dev, ws)

    def push_wait(self, dev, req):
        """per_request_fetch: poll until the set is READY, then dispatch (blocking)."""
        ws = req["weight_set_id"]
        if self.is_ready(dev, ws):
            if self.dispatch_phone(dev, req):
                self.m["dispatched_to_phone"] += 1
            else:
                self.m["fell_back_to_server"] += 1
                self.run_server(req)
            return
        r = self.res.get((dev, ws))
        if r is not None and r.state == "QUARANTINED":
            self.m["fell_back_to_server"] += 1
            self.run_server(req)                       # fetch failed; recover on server
            return
        self.push(self.t + 1000, PR_ARRIVAL, "wait", {"dev": dev, "req": req})

    def fault_event(self, kind, target):
        if kind == "thermal_trip" and target in self.phones:
            self.thermal_ok[target] = False
        elif kind == "stale_epoch" and target in self.phones:
            self.boot_epoch[target] += 1              # invalidate certs: drop READY sets
            for (dd, ws), r in sorted(self.res.items()):
                if dd == target and r.state in ("READY_HTP", "READY_GPU"):
                    if r.pinned > 0:
                        r.evict_after_unpin = True     # lease-safe: drain first, free at unpin
                    else:
                        self.free_residency(dd, ws, r)
        elif kind == "link_drop" and target in self.phones:
            self.eff_goodput[target] = max(1, self.eff_goodput[target] // 4)
        # hash_mismatch / partial_transfer / unknown_profile / unsupported_backend
        # are consumed inline in stage()/eligible()/start_prefetch().

    # ---- prefetch scheduling for static/clairvoyant at t=0 ----
    def preplace(self):
        if self.policy in ("static_placement", "clairvoyant"):
            seen = []
            for req in self.cfg["workload"]["requests"]:
                if req["weight_set_id"] not in seen:
                    seen.append(req["weight_set_id"])
            dev = next(iter(self.phones))
            for ws in seen:
                self.start_prefetch(dev, ws)

    def run(self):
        # schedule faults
        for (kind, target), at in sorted(self.faults.items()):
            self.push(at, PR_FAULT, "fault", {"kind": kind, "target": target})
        self.preplace()
        for req in sorted(self.cfg["workload"]["requests"], key=lambda r: (r["arrival_us"], r["rank"])):
            self.push(req["arrival_us"], PR_ARRIVAL, "arrival", req)
        while self.heap:
            t, pr, seq, kind, payload = heapq.heappop(self.heap)
            self.t = t
            if kind == "arrival":
                self.on_arrival(payload)
            elif kind == "wait":
                self.push_wait(payload["dev"], payload["req"])
            elif kind == "res_stage":
                self.stage(payload["dev"], payload["ws_id"], payload["stage"], payload["ver"])
            elif kind == "lane_arrive":
                self.lane_arrive(payload)
            elif kind == "exec_done":
                self.exec_done(payload)
            elif kind == "fault":
                self.fault_event(payload["kind"], payload["target"])
        # final ledger integrity
        for d in self.phones:
            self.ledger[d].check()
        return self.result()

    def result(self):
        comps = sorted(self.completions)
        r = dict(self.m)
        r["policy"] = self.policy
        r["is_upper_bound"] = (self.policy == "clairvoyant")
        r["is_diagnostic_losing"] = (self.policy == "per_request_fetch")
        r["completion_p50_us"] = pctl(comps, 1, 2)
        r["completion_p95_us"] = pctl(comps, 19, 20)
        # order keys deterministically for the schema
        keys = ["policy", "is_upper_bound", "is_diagnostic_losing", "completed",
                "dispatched_to_phone", "fell_back_to_server", "weight_misses",
                "server_gpu_us_used", "server_gpu_us_freed", "server_hbm_bytes_freed",
                "transfer_bytes", "prepare_us_total", "evictions", "deadline_hits",
                "deadline_misses", "completion_p50_us", "completion_p95_us",
                "causal_score_total", "waited_for_weights"]
        return {k: r[k] for k in keys}


def run_sweep(cfg):
    results = []
    for g in cfg["link_goodput_sweep_bytes_per_s"]:
        row = {"goodput_bytes_per_s": g, "baselines": []}
        for pol in cfg["baselines"]:
            sim = Sim(cfg, g, pol)
            row["baselines"].append(sim.run())
        results.append(row)
    return results


def code_version():
    return sha_over_bytes(open(__file__, "rb").read())


def build_manifest(cfg, cfg_path, results):
    cfg_hash = sha_over_text(canonical(cfg))
    replay = sha_over_text(canonical(results))
    return {
        "schema_version": 1, "run_id": "run-" + cfg["config_id"], "kind": "residency_sim",
        "fixtures_only": True, "config_id": cfg["config_id"], "config_hash": cfg_hash,
        "code_version": code_version(), "seed": cfg["seed"],
        "inputs": [{"role": "sim_config", "path": os.path.relpath(cfg_path, HERE),
                    "sha256": sha_over_bytes(open(cfg_path, "rb").read())}],
        "goodput_results": results,
        "deterministic_replay_sha256": replay, "gate_results": {},
    }


def main(argv):
    if not argv:
        print("usage: residency_sim.py CONFIG.json [--out MANIFEST.json]")
        return 2
    cfg_path = argv[0]
    cfg = json.load(open(cfg_path))
    results = run_sweep(cfg)
    manifest = build_manifest(cfg, cfg_path, results)
    out = None
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    text = canonical(manifest) + "\n"
    if out:
        open(out, "w").write(text)
        print("wrote", out)
    print("replay:", manifest["deterministic_replay_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
