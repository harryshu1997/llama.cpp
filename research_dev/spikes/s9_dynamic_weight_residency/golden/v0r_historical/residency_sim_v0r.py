#!/usr/bin/env python3
"""S9-V0-R deterministic offline residency simulator (bundle version 2).

Isolated research tool -- NOT integrated into llama-server. Integer-microsecond time,
stable total event order (heap keyed by (t_us, priority_rank, seq)). No wall clock, no
PRNG for durations. This is the REPAIRED simulator; the pre-repair V0 lives frozen at
golden/v0_historical/residency_sim_v0.py.

Repairs over V0 (see SIMULATOR_SPEC.md section "V0-R"):
  - every request reaches exactly one terminal outcome:
    completed_phone / completed_server / fallback / rejected / timed_out (checked).
  - pipelines/dispatches/completions are epoch-bound (boot, residency generation,
    route, state); a stale pipeline stage or completion is dropped/rejected fail-closed.
  - horizon_us + bounded prefetch/server/lane queues are enforced.
  - real multi-device/backend candidate selection over per-domain links (streams in the
    same contention_domain share bandwidth; separate domains are additive -- this
    replaces V0's "divide one controller by phone count").
  - UFS/LPDDR-canonical/derived/scratch/activation/state bytes are reserved when each
    stage acquires them and rolled back on every failure; sticky state persists until an
    explicit reset.
  - resume re-sends only the remaining verified range while RETAINING link ownership and
    counting retry bytes; an activation still preempts the resumed transfer.
  - the V0 dimensionally-invalid causal score is replaced by a documented LEXICOGRAPHIC
    objective over integer keys; relief and cold-cost are also summed separately (both in
    microseconds -- dimensionally valid).
  - interference is applied ONLY over the actual resource-overlap window, and only when
    interference.measured is true.
  - D2H (result return) is an EXPLICIT completion blocker on the link (the directional
    H2D/D2H rate decomposition remains the S9-V1 blocker).
"""
import hashlib
import heapq
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
US = 1000000

PR_FAULT, PR_ARRIVAL, PR_ACT, PR_STAGE, PR_LANE, PR_EXEC, PR_D2H, PR_EVICT, PR_WAIT = range(9)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha_text(text):
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_bytes(b):
    return "sha256:" + hashlib.sha256(b).hexdigest()


def xfer_us(nbytes, goodput):
    return (nbytes * US + goodput - 1) // goodput


def rate_us(nbytes, rate):
    return (nbytes * US + rate - 1) // rate


def pctl(vals, qn, qd):
    if not vals:
        return 0
    n = len(vals)
    r = max(1, min(n, (qn * n + qd - 1) // qd))
    return vals[r - 1]


class FrameSequencer:
    """TransportFrame reader gate (TRANSPORT_CONTRACT s7): rejects duplicate, reordered,
    and stale-epoch frames fail-closed; idempotency keys make a retried mutating frame a
    no-op. Per (connection, channel) monotone seq."""
    def __init__(self):
        self.next_seq = {}
        self.applied = set()
        self.boot_epoch = {}
        self.residency_epoch = {}

    def accept(self, frame):
        conn = frame.get("conn", "c0")
        ch = frame["channel"]
        key = (conn, ch)
        exp = self.next_seq.get(key, frame["seq"])
        be = self.boot_epoch.get(conn)
        if be is not None and frame["boot_epoch"] < be:
            return False, "stale_epoch"
        re = self.residency_epoch.get(conn)
        if re is not None and frame.get("residency_epoch") is not None and frame["residency_epoch"] < re:
            return False, "stale_epoch"
        if frame["seq"] < exp:
            return False, "duplicate_message"
        if frame["seq"] > exp:
            return False, "reordered_message"
        if frame["idempotency_key"] in self.applied:
            return False, "duplicate_message"
        self.next_seq[key] = frame["seq"] + 1
        self.applied.add(frame["idempotency_key"])
        self.boot_epoch[conn] = max(be or 0, frame["boot_epoch"])
        if frame.get("residency_epoch") is not None:
            self.residency_epoch[conn] = max(re or 0, frame["residency_epoch"])
        return True, "ok"


class Ledger:
    """Exact LPDDR partition + a separate UFS budget. LPDDR:
    canonical+derived+scratch+activations+state+free == lpddr_total. UFS holds durable
    staged copies. Every reserve checks fit; callers roll back on any downstream failure."""
    def __init__(self, lpddr_total, ufs_total):
        self.total = lpddr_total
        self.canonical = 0
        self.derived = 0
        self.scratch = 0
        self.activations = 0
        self.state = 0
        self.ufs_total = ufs_total
        self.ufs_used = 0

    @property
    def free(self):
        return self.total - (self.canonical + self.derived + self.scratch + self.activations + self.state)

    def check(self):
        used = self.canonical + self.derived + self.scratch + self.activations + self.state
        assert 0 <= used <= self.total, f"lpddr overflow used={used} total={self.total}"
        assert self.free == self.total - used
        for v in (self.canonical, self.derived, self.scratch, self.activations, self.state):
            assert v >= 0
        assert 0 <= self.ufs_used <= self.ufs_total, f"ufs overflow {self.ufs_used}/{self.ufs_total}"

    def can_fit_lpddr(self, nbytes):
        return self.free >= nbytes

    def can_fit_ufs(self, nbytes):
        return self.ufs_total - self.ufs_used >= nbytes


class ResEntry:
    def __init__(self, ws_id, model):
        self.ws_id = ws_id
        self.model = model
        self.state = "ABSENT"
        self.backend = None
        self.ready_at = None
        self.generation = 0
        self.pipeline_ver = 0
        self.res_ufs = 0
        self.res_canonical = 0
        self.res_derived = 0
        self.res_scratch = 0
        self.pinned = 0
        self.last_use = -1
        self.uses = 0
        self.resumed = False
        self.evict_after_unpin = False
        self.sticky_state = 0        # persistent mutable state pinned to this residency


class Sim:
    def __init__(self, cfg, goodput, policy):
        self.cfg = cfg
        self.policy = policy
        self.goodput = goodput
        self.horizon = cfg["horizon_us"]
        self.t = 0
        self.seq = 0
        self.heap = []
        self.phones = {p["device_id"]: p for p in cfg["phones"]}
        self.pdev = sorted(self.phones)                       # deterministic device order
        self.rtt = {d: 500 for d in self.phones}
        self.ledger = {d: Ledger(p["lpddr_total_bytes"], p["ufs_total_bytes"]) for d, p in self.phones.items()}
        self.htp_free = {d: p["htp_lane_slots"] for d, p in self.phones.items()}
        self.gpu_free = {d: p["gpu_lane_slots"] for d, p in self.phones.items()}
        self.lane_q = {(d, b): [] for d in self.phones for b in ("htp", "gpu")}
        self.lane_busy_until = {(d, b): 0 for d in self.phones for b in ("htp", "gpu")}
        self.thermal_ok = {d: p["thermal_eligible"] for d, p in self.phones.items()}
        self.boot_epoch = {d: p["boot_epoch"] for d, p in self.phones.items()}
        self.route_epoch = {d: 0 for d in self.phones}
        self.state_epoch = {d: 0 for d in self.phones}
        self.draining = {d: False for d in self.phones}
        # per-contention-domain link (separate USB buses do NOT contend)
        self.domain_of = {d: p["contention_domain"] for d, p in self.phones.items()}
        self.domains = sorted(set(self.domain_of.values()))
        self.domain_bulk = {dom: None for dom in self.domains}
        self.domain_link_busy = {dom: 0 for dom in self.domains}
        self.prefetch_q = {dom: [] for dom in self.domains}
        self.pf_depth = cfg["queues"]["prefetch_depth"]
        self.sv_depth = cfg["queues"]["server_depth"]
        self.models = {m["model_id"]: m for m in cfg["models"]}
        self.ws_model = {m["weight_set_id"]: m for m in cfg["models"]}
        self.res = {}
        self.server_gpu_free = cfg["server"]["gpu_lane_slots"]
        self.server_hbm_free = cfg["server"]["hbm_credit_bytes"]
        self.server_hbm_total = cfg["server"]["hbm_credit_bytes"]
        self.server_q = []
        self.interf = cfg["interference"]
        self.faults = {}
        for f in cfg.get("failure_schedule", []):
            self.faults.setdefault((f["kind"], f["target"]), f["at_us"])
        self.pp = cfg["policy_params"]
        self.predict = {}
        self.static_dev = {}
        self.req_by_id = {r["request_id"]: r for r in cfg["workload"]["requests"]}
        self.terminal = {}                                     # request_id -> outcome
        self.req_res = {}                                      # request_id -> (dev, ws, act, st, back, sticky)
        # metrics
        self.m = dict(dispatched_to_phone=0, weight_misses=0, server_gpu_us_used=0,
                      server_gpu_us_freed=0, server_hbm_bytes_freed=0, transfer_bytes=0,
                      retry_bytes=0, d2h_bytes=0, ufs_bytes=0, prepare_us_total=0, evictions=0,
                      deadline_hits=0, deadline_misses=0, objective_relief_us_total=0,
                      objective_cost_us_total=0, waited_for_weights=0, stale_pipeline_drops=0,
                      stale_completion_rejects=0, prefetch_drops=0, oversub_rollbacks=0)
        self.outc = dict(completed_phone=0, completed_server=0, fallback=0, rejected=0, timed_out=0)
        self.dispatched_by_device = {d: 0 for d in self.phones}
        self.completions = []
        self.preemptions = 0
        self.quarantined = set()
        self.resumed_sets = set()
        self.live_state_evictions = 0
        self.dispatched_from_on_disk = 0

    # ---- heap ----
    def push(self, t, pr, kind, payload):
        self.seq += 1
        heapq.heappush(self.heap, (t, pr, self.seq, kind, payload))

    def fault_active(self, kind, target):
        at = self.faults.get((kind, target))
        return at is not None and at <= self.t

    def resolve(self, dev, ws_id):
        key = (dev, ws_id)
        if key not in self.res:
            self.res[key] = ResEntry(ws_id, self.ws_model[ws_id])
        return self.res[key]

    def is_ready(self, dev, ws_id):
        r = self.res.get((dev, ws_id))
        return r is not None and r.state in ("READY_HTP", "READY_GPU")

    # ---- terminal bookkeeping (exactly once per request) ----
    def finish(self, req, outcome):
        rid = req["request_id"]
        if rid in self.terminal:
            return
        if self.t > self.horizon and outcome in ("completed_phone", "completed_server", "fallback"):
            outcome = "timed_out"
        self.terminal[rid] = outcome
        self.outc[outcome] += 1
        if outcome in ("completed_phone", "completed_server", "fallback"):
            lat = self.t - req["arrival_us"]
            self.completions.append(lat)
            if req["deadline_us"] is not None:
                if self.t <= req["deadline_us"]:
                    self.m["deadline_hits"] += 1
                else:
                    self.m["deadline_misses"] += 1

    # ---- link (per contention domain; activation/result preempt bulk, retain ownership) ----
    def _link_send(self, dev, nbytes, is_result):
        dom = self.domain_of[dev]
        dur = xfer_us(nbytes, self.goodput) + self.rtt[dev]
        start = max(self.t, self.domain_link_busy[dom])
        finish = start + dur
        self.domain_link_busy[dom] = finish
        if is_result:
            self.m["d2h_bytes"] += nbytes
        bulk = self.domain_bulk[dom]
        if bulk is not None:
            self.preemptions += 1
            r = self.resolve(bulk["dev"], bulk["ws_id"])
            r.pipeline_ver += 1
            bulk["shift"] = bulk.get("shift", 0) + dur
            self.push(bulk["done_time"] + bulk["shift"], PR_STAGE, "res_stage",
                      {"dev": bulk["dev"], "ws_id": bulk["ws_id"], "stage": "transfer_done",
                       "ver": r.pipeline_ver, "gen": r.generation, "boot": self.boot_epoch[bulk["dev"]]})
        return finish

    # ---- pipeline reservation helpers (reserve on acquire; roll back on failure) ----
    def rollback_pipeline(self, dev, r):
        L = self.ledger[dev]
        L.canonical -= r.res_canonical
        L.derived -= r.res_derived
        L.scratch -= r.res_scratch
        L.ufs_used -= r.res_ufs
        r.res_canonical = r.res_derived = r.res_scratch = r.res_ufs = 0
        L.check()

    def free_residency(self, dev, ws, r):
        """Reclaim a READY residency. Routing all weight frees through here makes
        live_state_evictions a real guard (freeing a pinned/sticky set is a violation)."""
        if r.pinned > 0 or r.sticky_state > 0:
            self.live_state_evictions += 1
        L = self.ledger[dev]
        L.canonical -= r.res_canonical
        L.derived -= r.res_derived
        L.scratch -= r.res_scratch
        L.ufs_used -= r.res_ufs
        r.res_canonical = r.res_derived = r.res_scratch = r.res_ufs = 0
        r.state = "ABSENT"
        r.ready_at = None
        r.evict_after_unpin = False
        L.check()

    # ---- prefetch admission + transfer ----
    def start_prefetch(self, dev, ws_id):
        r = self.resolve(dev, ws_id)
        if r.state != "ABSENT" or ws_id in self.quarantined:
            return False
        if self.draining[dev] or self.fault_active("unknown_profile", dev):
            return False
        dom = self.domain_of[dev]
        if self.domain_bulk[dom] is not None:
            if len(self.prefetch_q[dom]) < self.pf_depth:
                if (dev, ws_id) not in self.prefetch_q[dom]:
                    self.prefetch_q[dom].append((dev, ws_id))
            else:
                self.m["prefetch_drops"] += 1                # bounded queue; not silent
            return False
        model = self.ws_model[ws_id]
        w = model["canonical_bytes"]
        if not self.ledger[dev].can_fit_ufs(w):
            if not self.evict_for_ufs(dev, w):
                return False
        self.ledger[dev].ufs_used += w                       # reserve durable UFS staging on RECEIVING
        r.res_ufs = w
        r.state = "RECEIVING"
        r.resumed = False
        tdur = xfer_us(w, self.goodput)
        if self.interf["measured"]:
            tdur += self.overlap_extra(dev, tdur, ("htp", "gpu"), self.interf["transfer_compute_slowdown_permille"])
        self.m["transfer_bytes"] += w
        start = max(self.t, self.domain_link_busy[dom])
        done = start + tdur + self.rtt[dev]
        r.pipeline_ver += 1
        self.domain_bulk[dom] = {"dev": dev, "ws_id": ws_id, "done_time": done, "ver": r.pipeline_ver}
        self.push(done, PR_STAGE, "res_stage", {"dev": dev, "ws_id": ws_id, "stage": "transfer_done",
                  "ver": r.pipeline_ver, "gen": r.generation, "boot": self.boot_epoch[dev]})
        return True

    def link_free(self, dom):
        self.domain_bulk[dom] = None
        while self.prefetch_q[dom]:
            dev, ws = self.prefetch_q[dom].pop(0)
            if self.res.get((dev, ws)) and self.res[(dev, ws)].state == "ABSENT":
                if self.start_prefetch(dev, ws):
                    break

    def stage_stale(self, dev, r, ver, gen, boot):
        return ver != r.pipeline_ver or gen != r.generation or boot != self.boot_epoch[dev]

    def stage(self, p):
        dev, ws_id, st = p["dev"], p["ws_id"], p["stage"]
        r = self.resolve(dev, ws_id)
        if self.stage_stale(dev, r, p["ver"], p["gen"], p["boot"]):
            self.m["stale_pipeline_drops"] += 1
            return
        model = self.ws_model[ws_id]
        phone = self.phones[dev]
        dom = self.domain_of[dev]
        if st == "transfer_done":
            # partial-transfer fault: resume the REMAINING verified range, retain link ownership
            if self.fault_active("partial_transfer", ws_id) and not r.resumed:
                r.resumed = True
                self.resumed_sets.add(ws_id)
                remaining = model["canonical_bytes"] // 4            # re-send the unverified tail
                self.m["retry_bytes"] += remaining
                self.m["transfer_bytes"] += remaining
                rdur = xfer_us(remaining, self.goodput)
                r.pipeline_ver += 1
                start = max(self.t, self.domain_link_busy[dom])
                self.domain_bulk[dom] = {"dev": dev, "ws_id": ws_id,
                                         "done_time": start + rdur + self.rtt[dev], "ver": r.pipeline_ver}
                self.push(start + rdur + self.rtt[dev], PR_STAGE, "res_stage",
                          {"dev": dev, "ws_id": ws_id, "stage": "transfer_done", "ver": r.pipeline_ver,
                           "gen": r.generation, "boot": self.boot_epoch[dev]})
                return
            self.link_free(dom)
            self.m["ufs_bytes"] += model["canonical_bytes"]
            udur = rate_us(model["canonical_bytes"], phone["ufs_write_bytes_per_s"])
            r.pipeline_ver += 1
            self.push(self.t + udur, PR_STAGE, "res_stage", {"dev": dev, "ws_id": ws_id,
                      "stage": "ufs_done", "ver": r.pipeline_ver, "gen": r.generation, "boot": self.boot_epoch[dev]})
        elif st == "ufs_done":
            r.state = "VERIFYING"
            if self.fault_active("hash_mismatch", ws_id):
                r.state = "QUARANTINED"
                self.quarantined.add(ws_id)
                self.rollback_pipeline(dev, r)               # free the staged UFS bytes
                return
            vdur = rate_us(model["canonical_bytes"], phone["verify_bytes_per_s"])
            r.pipeline_ver += 1
            self.push(self.t + vdur, PR_STAGE, "res_stage", {"dev": dev, "ws_id": ws_id,
                      "stage": "verify_done", "ver": r.pipeline_ver, "gen": r.generation, "boot": self.boot_epoch[dev]})
        elif st == "verify_done":
            r.state = "VERIFIED_ON_DISK"
            w = model["canonical_bytes"]
            if not self.ledger[dev].can_fit_lpddr(w) and not self.evict_for_lpddr(dev, w):
                self.m["oversub_rollbacks"] += 1
                self.rollback_pipeline(dev, r)               # transient LPDDR oversubscription
                r.state = "ABSENT"
                return
            self.ledger[dev].canonical += w
            r.res_canonical = w
            self.ledger[dev].check()
            r.state = "MATERIALIZING"
            mdur = rate_us(w, phone["materialize_bytes_per_s"])
            r.pipeline_ver += 1
            self.push(self.t + mdur, PR_STAGE, "res_stage", {"dev": dev, "ws_id": ws_id,
                      "stage": "materialize_done", "ver": r.pipeline_ver, "gen": r.generation, "boot": self.boot_epoch[dev]})
        elif st == "materialize_done":
            r.state = "LPDDR_READY"
            back = self.choose_backend(model)
            r.backend = back
            d = model["derived_bytes_htp"] if back == "htp" else model["derived_bytes_gpu"]
            if d > 0:
                if not self.ledger[dev].can_fit_lpddr(d) and not self.evict_for_lpddr(dev, d):
                    self.m["oversub_rollbacks"] += 1
                    self.rollback_pipeline(dev, r)
                    r.state = "ABSENT"
                    return
                self.ledger[dev].derived += d
                r.res_derived = d
                self.ledger[dev].check()
            pr_rate = phone["prepare_htp_bytes_per_s"] if back == "htp" else phone["prepare_gpu_bytes_per_s"]
            pdur = rate_us(d, pr_rate) if d > 0 else 0
            self.m["prepare_us_total"] += pdur
            r.state = "PREPARING_HTP" if back == "htp" else "PREPARING_GPU"
            r.pipeline_ver += 1
            self.push(self.t + pdur, PR_STAGE, "res_stage", {"dev": dev, "ws_id": ws_id,
                      "stage": "prepare_done", "ver": r.pipeline_ver, "gen": r.generation, "boot": self.boot_epoch[dev]})
        elif st == "prepare_done":
            r.state = "WARMING"
            r.pipeline_ver += 1
            self.push(self.t + phone["warmup_us"], PR_STAGE, "res_stage", {"dev": dev, "ws_id": ws_id,
                      "stage": "warm_done", "ver": r.pipeline_ver, "gen": r.generation, "boot": self.boot_epoch[dev]})
        elif st == "warm_done":
            s = model["scratch_bytes"]
            if not self.ledger[dev].can_fit_lpddr(s) and not self.evict_for_lpddr(dev, s):
                self.m["oversub_rollbacks"] += 1
                self.rollback_pipeline(dev, r)
                r.state = "ABSENT"
                return
            self.ledger[dev].scratch += s
            r.res_scratch = s
            self.ledger[dev].check()
            r.state = "READY_HTP" if r.backend == "htp" else "READY_GPU"
            r.ready_at = self.t

    # ---- eviction (lease-safe; policy-dependent victim) ----
    def victims(self, dev):
        c = []
        for (dd, ws), r in sorted(self.res.items()):
            if dd == dev and r.state in ("READY_HTP", "READY_GPU") and r.pinned == 0 and r.sticky_state == 0:
                c.append((dd, ws, r))
        return c

    def pick_victim(self, cands):
        ttl = self.pp["ttl_us"]
        if self.policy == "clairvoyant":
            cands.sort(key=lambda c: (-self.next_use(c[1]), c[1]))
        elif self.policy == "lfu_ttl":
            cands.sort(key=lambda c: (0 if (self.t - c[2].last_use) > ttl else 1, c[2].uses, c[2].last_use, c[1]))
        else:
            cands.sort(key=lambda c: (c[2].last_use, c[1]))
        return cands[0]

    def evict_for_lpddr(self, dev, need):
        min_hold = self.pp["min_hold_us"]
        while not self.ledger[dev].can_fit_lpddr(need):
            cands = self.victims(dev)
            aged = [c for c in cands if c[2].ready_at is not None and (self.t - c[2].ready_at) >= min_hold]
            pool = aged if aged else cands
            if not pool:
                return False
            dd, ws, r = self.pick_victim(pool)
            self.free_residency(dd, ws, r)
            self.m["evictions"] += 1
        return True

    def evict_for_ufs(self, dev, need):
        while not self.ledger[dev].can_fit_ufs(need):
            cands = self.victims(dev)
            if not cands:
                return False
            dd, ws, r = self.pick_victim(cands)
            self.free_residency(dd, ws, r)
            self.m["evictions"] += 1
        return True

    def next_use(self, ws_id):
        nxt = None
        for req in self.cfg["workload"]["requests"]:
            if req["weight_set_id"] == ws_id and req["arrival_us"] > self.t:
                if nxt is None or req["arrival_us"] < nxt:
                    nxt = req["arrival_us"]
        return nxt if nxt is not None else (1 << 62)

    def choose_backend(self, model):
        return model["eligible_backends"][0]

    # ---- interference: only over the actual resource-overlap window ----
    def overlap_extra(self, dev, dur, competing_backends, slowdown_permille):
        end = self.t + dur
        busy = 0
        for b in competing_backends:
            busy = max(busy, self.lane_busy_until.get((dev, b), 0))
        busy = max(busy, self.domain_link_busy[self.domain_of[dev]])
        overlap = max(0, min(end, busy) - self.t)
        return overlap * (slowdown_permille - 1000) // 1000

    # ---- server (bounded queue) ----
    def route_server(self, req, bucket):
        cls = req["service_class"]
        hbm = self.cfg["server"]["per_class_hbm_bytes"].get(cls, 0)
        if hbm > self.server_hbm_total:
            self.finish(req, "rejected")                     # impossible HBM demand: fail-closed
            return
        gpu_us = self.cfg["server"]["per_class_gpu_us"].get(cls, 1000)
        self.m["server_gpu_us_used"] += gpu_us
        if self.server_gpu_free >= 1 and self.server_hbm_free >= hbm:
            self.server_gpu_free -= 1
            self.server_hbm_free -= hbm
            self.push(self.t + gpu_us, PR_EXEC, "server_done", {"req": req, "hbm": hbm, "bucket": bucket})
        elif len(self.server_q) < self.sv_depth:
            self.server_q.append((req, gpu_us, hbm, bucket))
        else:
            self.finish(req, "rejected")                     # server queue saturated

    def server_done(self, p):
        self.server_gpu_free += 1
        self.server_hbm_free += p["hbm"]
        self.finish(p["req"], p["bucket"])
        if self.server_q:
            req, gpu_us, hbm, bucket = self.server_q[0]
            if self.server_gpu_free >= 1 and self.server_hbm_free >= hbm:
                self.server_q.pop(0)
                self.server_gpu_free -= 1
                self.server_hbm_free -= hbm
                self.push(self.t + gpu_us, PR_EXEC, "server_done", {"req": req, "hbm": hbm, "bucket": bucket})

    # ---- eligibility over (device, backend) ----
    def eligible(self, dev, back, req):
        model = self.models[req["model_id"]]
        if self.draining[dev]:
            return False, "draining"
        if self.fault_active("unknown_profile", dev):
            return False, "unknown_profile"
        if self.fault_active("unsupported_backend", req["weight_set_id"]):
            return False, "unsupported_backend"
        if back not in model["eligible_backends"]:
            return False, "unsupported_backend"
        if not self.thermal_ok[dev]:
            return False, "thermal"
        d = model["derived_bytes_htp"] if back == "htp" else model["derived_bytes_gpu"]
        if model["canonical_bytes"] + d + model["scratch_bytes"] > self.ledger[dev].total:
            return False, "insufficient_ram"
        return True, "ok"

    def candidates(self, req):
        model = self.models[req["model_id"]]
        cs = []
        for dev in self.pdev:
            for back in model["eligible_backends"]:
                ok, _ = self.eligible(dev, back, req)
                if ok:
                    cs.append((dev, back))
        return cs

    def cold_cost_us(self, dev, back, model):
        phone = self.phones[dev]
        w = model["canonical_bytes"]
        d = model["derived_bytes_htp"] if back == "htp" else model["derived_bytes_gpu"]
        pr_rate = phone["prepare_htp_bytes_per_s"] if back == "htp" else phone["prepare_gpu_bytes_per_s"]
        return (xfer_us(w, self.goodput) + rate_us(w, phone["ufs_write_bytes_per_s"]) +
                rate_us(w, phone["verify_bytes_per_s"]) + rate_us(w, phone["materialize_bytes_per_s"]) +
                (rate_us(d, pr_rate) if d > 0 else 0) + phone["warmup_us"])

    def objective_key(self, dev, back, req):
        """Documented LEXICOGRAPHIC objective over INTEGER keys (higher is better); no
        cross-unit summation. Ordering: SLO-feasible, server relief (gpu-us), lower
        cold cost (us), higher predicted reuse (permille), lower eviction loss (bytes),
        lower interference risk, earlier device order."""
        model = self.models[req["model_id"]]
        cls = req["service_class"]
        relief = self.cfg["server"]["per_class_gpu_us"].get(cls, 0)
        cold = self.cold_cost_us(dev, back, model)
        reuse = self.predict.get(req["weight_set_id"], 0)
        ready = self.is_ready(dev, req["weight_set_id"])
        need = 0 if ready else (model["canonical_bytes"] +
                                (model["derived_bytes_htp"] if back == "htp" else model["derived_bytes_gpu"]))
        evict_loss = max(0, need - self.ledger[dev].free)
        slo = 1
        if req["deadline_us"] is not None:
            eta = self.t + (0 if ready else cold) + model["phone_compute_us"] + xfer_us(req["output_bytes"], self.goodput)
            slo = 1 if eta <= req["deadline_us"] else 0
        interf_risk = 1 if (self.interf["measured"] and self.domain_link_busy[self.domain_of[dev]] > self.t) else 0
        return (slo, relief, -cold, reuse, -evict_loss, -interf_risk, -self.pdev.index(dev))

    # ---- dispatch ----
    def dispatch_phone(self, dev, back, req):
        ws = req["weight_set_id"]
        if not self.is_ready(dev, ws):
            self.dispatched_from_on_disk += 1
            return False
        if self.draining[dev]:
            return False
        act, st = req["input_bytes"], req["state_bytes"]
        model = self.models[req["model_id"]]
        sticky = model["state_policy"] in ("sticky", "rebuildable")
        r = self.resolve(dev, ws)
        new_state = 0 if (sticky and r.sticky_state > 0) else st   # sticky state already resident is reused
        if not self.ledger[dev].can_fit_lpddr(act + new_state):
            return False
        self.ledger[dev].activations += act
        self.ledger[dev].state += new_state
        self.ledger[dev].check()
        r.pinned += 1
        r.last_use = self.t
        r.uses += 1
        self.dispatched_by_device[dev] += 1
        self.m["server_gpu_us_freed"] += self.cfg["server"]["per_class_gpu_us"].get(req["service_class"], 0)
        self.m["server_hbm_bytes_freed"] += self.cfg["server"]["per_class_hbm_bytes"].get(req["service_class"], 0)
        self.m["objective_relief_us_total"] += self.cfg["server"]["per_class_gpu_us"].get(req["service_class"], 0)
        self.req_res[req["request_id"]] = {"dev": dev, "ws": ws, "act": act, "st": new_state,
                                           "back": back, "sticky": sticky}
        in_done = self._link_send(dev, act, is_result=False)  # H2D activation preempts bulk
        self.push(in_done, PR_LANE, "lane_arrive", {"dev": dev, "back": back, "req": req,
                  "boot": self.boot_epoch[dev], "gen": r.generation, "route": self.route_epoch[dev],
                  "state": self.state_epoch[dev]})
        return True

    def lane_arrive(self, p):
        dev, back, req = p["dev"], p["back"], p["req"]
        free = self.htp_free if back == "htp" else self.gpu_free
        model = self.models[req["model_id"]]
        comp = model["phone_compute_us"]
        if self.interf["measured"]:
            other = ("gpu",) if back == "htp" else ("htp",)
            comp += self.overlap_extra(dev, comp, other, self.interf["htp_gpu_slowdown_permille"])
        if free[dev] >= 1:
            free[dev] -= 1
            self.lane_busy_until[(dev, back)] = self.t + comp
            self.push(self.t + comp, PR_EXEC, "exec_done", {**p, "comp": comp})
        else:
            self.lane_q[(dev, back)].append({**p, "comp": comp})

    def exec_done(self, p):
        dev, back, req = p["dev"], p["back"], p["req"]
        r = self.resolve(dev, req["weight_set_id"])
        stale = (p["boot"] != self.boot_epoch[dev] or p["gen"] != r.generation or
                 p["route"] != self.route_epoch[dev] or p["state"] != self.state_epoch[dev])
        rr = self.req_res.get(req["request_id"])
        if stale:
            self.m["stale_completion_rejects"] += 1           # fail-closed: never trust a stale completion
            if rr:
                self.ledger[dev].activations -= rr["act"]
                self.ledger[dev].state -= rr["st"]
                self.ledger[dev].check()
            r.pinned = max(0, r.pinned - 1)
            self._release_lane(dev, back)
            self.finish(req, "rejected")
            return
        out = self._link_send(dev, req["output_bytes"], is_result=True)  # D2H result: explicit blocker
        self.push(out, PR_D2H, "d2h_done", p)
        self._release_lane(dev, back)

    def _release_lane(self, dev, back):
        free = self.htp_free if back == "htp" else self.gpu_free
        free[dev] += 1
        q = self.lane_q[(dev, back)]
        if q:
            nxt = q.pop(0)
            free[dev] -= 1
            self.lane_busy_until[(dev, back)] = self.t + nxt["comp"]
            self.push(self.t + nxt["comp"], PR_EXEC, "exec_done", nxt)

    def d2h_done(self, p):
        dev, req = p["dev"], p["req"]
        rr = self.req_res.get(req["request_id"], {})
        r = self.resolve(dev, req["weight_set_id"])
        self.ledger[dev].activations -= rr.get("act", 0)
        if rr.get("sticky"):
            r.sticky_state += rr.get("st", 0)                 # persists until explicit reset
        else:
            self.ledger[dev].state -= rr.get("st", 0)
        self.ledger[dev].check()
        r.pinned = max(0, r.pinned - 1)
        if r.pinned == 0 and r.sticky_state == 0 and r.evict_after_unpin:
            self.free_residency(dev, req["weight_set_id"], r)
            self.m["evictions"] += 1
        self.finish(req, "completed_phone")

    # ---- arrival policy ----
    def preplace(self):
        seen = []
        for req in self.cfg["workload"]["requests"]:
            if req["weight_set_id"] not in seen:
                seen.append(req["weight_set_id"])
        for i, ws in enumerate(seen):
            model = self.ws_model[ws]
            elig = [d for d in self.pdev if self.eligible(d, model["eligible_backends"][0], self._probe_req(ws))[0]]
            self.static_dev[ws] = elig[i % len(elig)] if elig else (self.pdev[0] if self.pdev else None)
        if self.policy in ("static_placement", "clairvoyant"):
            for ws in seen:
                dev = self.static_dev[ws] if self.policy == "static_placement" else self.pdev[0]
                if self.policy == "clairvoyant" and self.next_use(ws) >= self.pp["reuse_horizon_us"]:
                    continue
                self.start_prefetch(dev, ws)

    def _probe_req(self, ws):
        model = self.ws_model[ws]
        return {"model_id": model["model_id"], "weight_set_id": ws, "service_class": "decode",
                "input_bytes": 0, "output_bytes": 0, "state_bytes": 0, "deadline_us": None}

    def on_arrival(self, req):
        ws = req["weight_set_id"]
        self.predict[ws] = (self.predict[ws] * (1000 - self.pp["predictor_ewma_permille"]) +
                            1000 * self.pp["predictor_ewma_permille"]) // 1000 if ws in self.predict \
            else self.pp["predictor_ewma_permille"]

        if self.policy == "server_only":
            self.route_server(req, "completed_server")
            return

        cands = self.candidates(req)
        if not cands:
            self.route_server(req, "fallback")               # ineligible everywhere -> server (or rejected in route)
            return

        if self.policy == "per_request_fetch":
            self._per_request_fetch(req, cands)
            return

        ready = [(d, b) for (d, b) in cands if self.is_ready(d, ws)]
        pick = self._select(req, cands, ready)
        if pick is not None and self.is_ready(pick[0], ws):
            self.m["objective_cost_us_total"] += 0
            if self.dispatch_phone(pick[0], pick[1], req):
                self.m["dispatched_to_phone"] += 1
                return
        # miss: server fallback + background prefetch to the selected device
        self.m["weight_misses"] += 1
        self.route_server(req, "fallback")
        self._maybe_prefetch(req, cands, pick)

    def _select(self, req, cands, ready):
        ws = req["weight_set_id"]
        if self.policy == "static_placement":
            dev = self.static_dev.get(ws)
            for (d, b) in cands:
                if d == dev:
                    return (d, b)
            return cands[0]
        if self.policy in ("lru", "lfu_ttl"):
            for (d, b) in ready:
                if d == self.static_dev.get(ws):
                    return (d, b)
            return ready[0] if ready else (cands[0] if self.static_dev.get(ws) is None else
                                           next(((d, b) for (d, b) in cands if d == self.static_dev.get(ws)), cands[0]))
        if self.policy == "fastest_ready":
            pool = ready if ready else cands
            pool = sorted(pool, key=lambda c: (self.models[req["model_id"]]["phone_compute_us"], c[0]))
            return pool[0]
        if self.policy == "clairvoyant":
            pool = ready if ready else cands
            return sorted(pool, key=lambda c: (0 if self.is_ready(c[0], ws) else 1, self.pdev.index(c[0])))[0]
        # relief_predictive: lexicographic objective
        pool = ready if ready else cands
        return max(pool, key=lambda c: self.objective_key(c[0], c[1], req))

    def _maybe_prefetch(self, req, cands, pick):
        ws = req["weight_set_id"]
        if self.policy == "static_placement":
            dev = self.static_dev.get(ws)
            if dev is not None:
                self.start_prefetch(dev, ws)
            return
        if self.policy == "relief_predictive":
            best = max(cands, key=lambda c: self.objective_key(c[0], c[1], req))
            if self.objective_key(best[0], best[1], req)[1] <= 0:   # no server relief -> do not prefetch
                return
            self.start_prefetch(best[0], ws)
            return
        if self.policy == "clairvoyant":
            if self.next_use(ws) >= self.pp["reuse_horizon_us"]:
                return
            self.start_prefetch(self.pdev[0], ws)
            return
        dev = pick[0] if pick is not None else cands[0][0]
        self.start_prefetch(dev, ws)

    def _per_request_fetch(self, req, cands):
        ws = req["weight_set_id"]
        dev = cands[0][0]
        back = cands[0][1]
        if self.is_ready(dev, ws):
            if not self.dispatch_phone(dev, back, req):
                self.route_server(req, "fallback")
            else:
                self.m["dispatched_to_phone"] += 1
            return
        self.start_prefetch(dev, ws)
        self.m["waited_for_weights"] += 1
        self.push(self.t + 1000, PR_WAIT, "wait", {"dev": dev, "back": back, "req": req})

    def wait(self, p):
        dev, back, req = p["dev"], p["back"], p["req"]
        ws = req["weight_set_id"]
        if req["request_id"] in self.terminal:
            return
        if self.t > self.horizon:
            self.finish(req, "timed_out")
            return
        if self.is_ready(dev, ws):
            if self.dispatch_phone(dev, back, req):
                self.m["dispatched_to_phone"] += 1
            else:
                self.route_server(req, "fallback")
            return
        r = self.res.get((dev, ws))
        if r is not None and r.state == "QUARANTINED":
            self.route_server(req, "fallback")
            return
        if r is not None and r.state == "ABSENT":
            self.start_prefetch(dev, ws)
        self.push(self.t + 1000, PR_WAIT, "wait", p)

    # ---- faults ----
    def fault_event(self, kind, target):
        if kind == "thermal_trip" and target in self.phones:
            self.thermal_ok[target] = False
        elif kind == "drain" and target in self.phones:
            self.draining[target] = True
        elif kind == "reset_state" and target in self.phones:
            for (dd, ws), r in sorted(self.res.items()):
                if dd == target and r.sticky_state > 0 and r.pinned == 0:
                    self.ledger[dd].state -= r.sticky_state
                    r.sticky_state = 0
                    self.ledger[dd].check()
        elif kind == "stale_epoch" and target in self.phones:
            self.boot_epoch[target] += 1
            self.route_epoch[target] += 1
            self.state_epoch[target] += 1
            for (dd, ws), r in sorted(self.res.items()):
                if dd == target and r.state in ("READY_HTP", "READY_GPU"):
                    if r.pinned > 0 or r.sticky_state > 0:
                        r.evict_after_unpin = True
                    else:
                        self.free_residency(dd, ws, r)
        elif kind == "link_drop" and target in self.phones:
            pass
        # hash_mismatch / partial_transfer / unknown_profile / unsupported_backend consumed inline

    def run(self):
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
                self.wait(payload)
            elif kind == "res_stage":
                self.stage(payload)
            elif kind == "lane_arrive":
                self.lane_arrive(payload)
            elif kind == "exec_done":
                self.exec_done(payload)
            elif kind == "d2h_done":
                self.d2h_done(payload)
            elif kind == "server_done":
                self.server_done(payload)
            elif kind == "fault":
                self.fault_event(payload["kind"], payload["target"])
        # horizon sweep: every request that never terminalized times out
        self.t = self.horizon
        for req in self.cfg["workload"]["requests"]:
            if req["request_id"] not in self.terminal:
                self.finish(req, "timed_out")
        for d in self.phones:
            self.ledger[d].check()
        return self.result()

    def result(self):
        comps = sorted(self.completions)
        total_req = len(self.cfg["workload"]["requests"])
        assert sum(self.outc.values()) == total_req, f"terminal buckets {self.outc} != {total_req} requests"
        r = dict(self.m)
        r.update(self.outc)
        r["policy"] = self.policy
        r["requests"] = total_req
        r["is_upper_bound"] = (self.policy == "clairvoyant")
        r["is_diagnostic_losing"] = (self.policy == "per_request_fetch")
        r["dispatched_by_device"] = dict(sorted(self.dispatched_by_device.items()))
        r["completion_p50_us"] = pctl(comps, 1, 2)
        r["completion_p95_us"] = pctl(comps, 19, 20)
        keys = ["policy", "is_upper_bound", "is_diagnostic_losing", "requests",
                "completed_phone", "completed_server", "fallback", "rejected", "timed_out",
                "dispatched_to_phone", "dispatched_by_device", "weight_misses",
                "server_gpu_us_used", "server_gpu_us_freed", "server_hbm_bytes_freed",
                "transfer_bytes", "retry_bytes", "d2h_bytes", "ufs_bytes", "prepare_us_total",
                "evictions", "stale_pipeline_drops", "stale_completion_rejects", "prefetch_drops",
                "oversub_rollbacks", "deadline_hits", "deadline_misses", "completion_p50_us",
                "completion_p95_us", "objective_relief_us_total", "objective_cost_us_total",
                "waited_for_weights"]
        return {k: r[k] for k in keys}


def run_sweep(cfg):
    results = []
    for g in cfg["link_goodput_sweep_bytes_per_s"]:
        row = {"goodput_bytes_per_s": g, "baselines": []}
        for pol in cfg["baselines"]:
            row["baselines"].append(Sim(cfg, g, pol).run())
        results.append(row)
    return results


def code_version():
    return sha_bytes(open(__file__, "rb").read())


def build_manifest(cfg, cfg_path, results):
    cfg_hash = sha_text(canonical(cfg))
    replay = sha_text(canonical(results))
    return {
        "schema_version": 2, "run_id": "run-" + cfg["config_id"], "kind": "residency_sim",
        "fixtures_only": True, "config_id": cfg["config_id"], "config_hash": cfg_hash,
        "code_version": code_version(), "seed": cfg["seed"],
        "inputs": [{"role": "sim_config", "path": os.path.relpath(cfg_path, HERE),
                    "sha256": sha_bytes(open(cfg_path, "rb").read())}],
        "goodput_results": results, "deterministic_replay_sha256": replay, "gate_results": {},
    }


def main(argv):
    if not argv:
        print("usage: residency_sim.py CONFIG.json [--out MANIFEST.json]")
        return 2
    cfg_path = argv[0]
    cfg = json.load(open(cfg_path))
    results = run_sweep(cfg)
    manifest = build_manifest(cfg, cfg_path, results)
    text = canonical(manifest) + "\n"
    if "--out" in argv:
        open(argv[argv.index("--out") + 1], "w").write(text)
        print("wrote", argv[argv.index("--out") + 1])
    print("replay:", manifest["deterministic_replay_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
