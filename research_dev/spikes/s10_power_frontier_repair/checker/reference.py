"""Independent brute-force optimum reference for the tiny offline scheduling model.

This module is written ONLY from the written specification of the model (schema_version 2).
It is deliberately an obviously-complete brute force: clarity and evident completeness are
preferred over speed.  It exists as an independent cross-check against the oracle, so it
shares no code with, and was derived without reading, any other implementation.

Method
------
  * enumerate every assignment of nodes to devices that appear in their own routes;
  * enumerate every set partition of the SERVER nodes that carry a non-null batch_key
    (nodes with a null batch_key, and all nodes placed on a phone, are always singletons);
  * enumerate every integer start time for every action, restricted only to
        start in [max(member release_us), horizon_us - duration]
    plus start >= server_power.wake_us for SERVER actions.  Those restrictions follow
    directly from legality rules 5, 8 and 9 and remove no legal schedule.
  * reject anything violating a legality rule or the activation-memory bound;
  * return the lexicographic minimum objective over all legal schedules.

Deadlines are NEVER used to bound the search: only the horizon bounds it.  There is no
branch-and-bound and no incumbent pruning.  The depth-first search rejects a partial
schedule only when a legality rule (6: precedence, 7: one lane per device) is ALREADY
violated by the actions placed so far, which can only remove illegal schedules.  Every
complete assignment is additionally re-checked against the full legality rule set at the
leaf, so correctness does not depend on the search order or on the early rejections.

Units are integers throughout: time in us, power in mW, energy in nJ = mW*us.

Public surface
--------------
  REFERENCE_MAX_NODES, REFERENCE_MAX_ACTIONS, REFERENCE_MAX_ENUM
  ReferenceOutOfDomain
  optimum(inst, max_enum=REFERENCE_MAX_ENUM) -> 4-tuple
"""

import itertools
import json

# ---------------------------------------------------------------------------
# Declared bounds.  These are honest, explicit limits on what this brute force
# will attempt.  Outside them we refuse to answer rather than answer slowly or
# wrongly.
# ---------------------------------------------------------------------------

# Maximum number of nodes in an instance.
REFERENCE_MAX_NODES = 8

# Maximum number of actions in any single (assignment, partition) configuration.
REFERENCE_MAX_ACTIONS = 8

# Maximum total number of start-time combinations the search may consider, summed
# over every (assignment, partition) configuration.  A configuration contributes
# the product of its per-action start-domain sizes (an upper bound on the number of
# leaves the DFS can reach for it).  The two frozen fixtures need about 1.67e6
# (transition_delay_counterexample) and about 4.6e5 (activation_delay_counterexample),
# so this leaves ample headroom while staying an explicit ceiling.
REFERENCE_MAX_ENUM = 8000000


class ReferenceOutOfDomain(Exception):
    """The instance is outside the declared brute-force domain; no answer is given."""


# ---------------------------------------------------------------------------
# Small combinatorial helpers
# ---------------------------------------------------------------------------

def _set_partitions(items):
    """Yield every set partition of `items` as a list of lists (deterministic order)."""
    if not items:
        yield []
        return
    first = items[0]
    rest = list(items[1:])
    for smaller in _set_partitions(rest):
        # Put `first` into each existing block ...
        for i in range(len(smaller)):
            yield smaller[:i] + [[first] + smaller[i]] + smaller[i + 1:]
        # ... or into a fresh block of its own.
        yield [[first]] + smaller


# ---------------------------------------------------------------------------
# Instance view
# ---------------------------------------------------------------------------

class _Instance(object):
    """Flattened, index-based view of the instance JSON."""

    def __init__(self, inst):
        self.horizon = int(inst["horizon_us"])
        self.mem_bound = int(inst["activation_mem_bound_bytes"])

        sp = inst["server_power"]
        self.p8_mw = int(sp["p8_mw"])
        self.p0_mw = int(sp["p0_mw"])
        self.wake_us = int(sp["wake_us"])
        self.idle_entry_us = int(sp["idle_entry_us"])
        self.transition_nj = int(sp["transition_nj"])

        self.devices = inst["devices"]
        self.batch_profiles = inst.get("batch_profiles", {})

        nodes = inst["nodes"]
        self.n = len(nodes)
        self.node_ids = [nd["id"] for nd in nodes]
        self.index_of = {}
        for i, nid in enumerate(self.node_ids):
            self.index_of[nid] = i

        self.release = [int(nd["release_us"]) for nd in nodes]
        self.model_id = [nd["model_id"] for nd in nodes]
        self.weight_set_id = [nd["weight_set_id"] for nd in nodes]
        self.batch_key = [nd.get("batch_key") for nd in nodes]
        self.tokens = [int(nd["tokens"]) for nd in nodes]
        self.output_bytes = [int(nd["output_bytes"]) for nd in nodes]
        self.routes = [nd["routes"] for nd in nodes]

        # Direct predecessors / successors, as node indices.
        self.preds = []
        for nd in nodes:
            self.preds.append([self.index_of[p] for p in nd["predecessors"]])
        self.succs = [[] for _ in range(self.n)]
        for i in range(self.n):
            for p in self.preds[i]:
                self.succs[p].append(i)

        # Precedence pairs (predecessor index, successor index).
        self.edges = []
        for i in range(self.n):
            for p in self.preds[i]:
                self.edges.append((p, i))

        # Nodes that can hold activation memory.
        self.act_nodes = [i for i in range(self.n) if self.output_bytes[i] > 0]

        # Legal devices per node: the device must appear in the node's own routes
        # and must be a device of the instance.
        self.legal_devices = []
        for i in range(self.n):
            devs = sorted(d for d in self.routes[i].keys() if d in self.devices)
            self.legal_devices.append(devs)

        # Requests.
        self.requests = inst["requests"]
        self.req_terminal = [self.index_of[r["terminal_node"]] for r in self.requests]
        self.req_deadline = [int(r["deadline_us"]) for r in self.requests]
        self.n_requests = len(self.requests)


# ---------------------------------------------------------------------------
# Configuration enumeration (device assignment + batching partition)
# ---------------------------------------------------------------------------

def _block_duration(ix, dev, members):
    """Duration of an action, or None if the action is illegal (rule 2 / rule 3)."""
    if len(members) > 1:
        # Rule 2: multi-member actions are SERVER-only, must agree on batch_key
        # (non-null), model_id and weight_set_id, and no member may be a predecessor
        # of another member.  Grouping upstream guarantees the first three; the
        # predecessor condition is checked here.
        if dev != "SERVER":
            return None
        key = ix.batch_key[members[0]]
        if key is None:
            return None
        for m in members:
            if (ix.batch_key[m] != key
                    or ix.model_id[m] != ix.model_id[members[0]]
                    or ix.weight_set_id[m] != ix.weight_set_id[members[0]]):
                return None
        member_set = set(members)
        for m in members:
            for p in ix.preds[m]:
                if p in member_set:
                    return None

    key = ix.batch_key[members[0]]
    if dev == "SERVER" and key is not None:
        # Rule 3: SERVER action whose members carry a non-null batch_key is timed by
        # the batch profile -- this holds for singletons too.
        profile = ix.batch_profiles.get(key)
        if profile is None:
            return None
        token_key = str(sum(ix.tokens[m] for m in members))
        if token_key not in profile:
            return None
        return int(profile[token_key])

    # Otherwise the action must be a singleton timed by its route.
    if len(members) != 1:
        return None
    return int(ix.routes[members[0]][dev]["duration_us"])


def _iter_configurations(ix):
    """Yield configurations: lists of actions (device, members tuple, duration).

    A configuration fixes the device of every node and the batching partition of the
    SERVER nodes that carry a non-null batch_key.  Start times are not fixed here.
    """
    if any(len(d) == 0 for d in ix.legal_devices):
        # Some node has no legal device: no legal schedule exists at all.
        return

    for assignment in itertools.product(*ix.legal_devices):
        # Nodes that may be batched: on SERVER with a non-null batch_key.
        batchable = [i for i in range(ix.n)
                     if assignment[i] == "SERVER" and ix.batch_key[i] is not None]

        # A legal action may only group nodes agreeing on (batch_key, model_id,
        # weight_set_id), so any legal partition refines this grouping.  Enumerating
        # set partitions within each group and taking the product over groups is
        # therefore exhaustive over all legal partitions.
        groups = {}
        for i in batchable:
            gkey = (ix.batch_key[i], ix.model_id[i], ix.weight_set_id[i])
            groups.setdefault(gkey, []).append(i)
        group_keys = sorted(groups.keys())
        per_group = [list(_set_partitions(groups[g])) for g in group_keys]

        singles = [i for i in range(ix.n) if i not in set(batchable)]

        for combo in itertools.product(*per_group) if per_group else [()]:
            actions = []
            ok = True

            for i in singles:
                dev = assignment[i]
                dur = _block_duration(ix, dev, (i,))
                if dur is None:
                    ok = False
                    break
                actions.append((dev, (i,), dur))
            if not ok:
                continue

            for blocks in combo:
                for block in blocks:
                    members = tuple(sorted(block))
                    dur = _block_duration(ix, "SERVER", members)
                    if dur is None:
                        ok = False
                        break
                    actions.append(("SERVER", members, dur))
                if not ok:
                    break
            if not ok:
                continue

            yield actions


def _start_domain(ix, action):
    """Inclusive [lo, hi] integer start domain of an action, or None if empty.

    lo comes from rule 5 (member releases) and rule 8 (SERVER wake).
    hi comes from rule 9 (horizon).  Nothing else narrows it: no deadline and no
    successor-derived reasoning is used.
    """
    dev, members, dur = action
    lo = 0
    for m in members:
        if ix.release[m] > lo:
            lo = ix.release[m]
    if dev == "SERVER" and ix.wake_us > lo:
        lo = ix.wake_us
    hi = ix.horizon - dur
    if lo > hi:
        return None
    return (lo, hi)


# ---------------------------------------------------------------------------
# Full legality check (applied at every leaf, independent of search order)
# ---------------------------------------------------------------------------

def _fully_legal(ix, actions, starts, node_action):
    n_act = len(actions)

    for ai in range(n_act):
        dev, members, dur = actions[ai]
        s = starts[ai]
        f = s + dur
        # Rule 5: start at or after every member's release.
        for m in members:
            if s < ix.release[m]:
                return False
        # Rule 8: SERVER actions start at or after wake.
        if dev == "SERVER" and s < ix.wake_us:
            return False
        # Rule 9: finish within the horizon.
        if f > ix.horizon:
            return False

    # Rule 6: every predecessor finishes at or before its successor starts.
    for (p, q) in ix.edges:
        pa = node_action[p]
        qa = node_action[q]
        if starts[pa] + actions[pa][2] > starts[qa]:
            return False

    # Rule 7: actions on the same device do not overlap.
    for ai in range(n_act):
        for bi in range(ai + 1, n_act):
            if actions[ai][0] != actions[bi][0]:
                continue
            a_s = starts[ai]
            a_f = a_s + actions[ai][2]
            b_s = starts[bi]
            b_f = b_s + actions[bi][2]
            if a_s < b_f and b_s < a_f:
                return False

    return True


# ---------------------------------------------------------------------------
# Evaluation of a complete schedule
# ---------------------------------------------------------------------------

def _evaluate(ix, actions, starts, node_action):
    """Return the 4-tuple objective, or None if the activation bound is exceeded."""
    node_start = [0] * ix.n
    node_finish = [0] * ix.n
    for ai in range(len(actions)):
        dev, members, dur = actions[ai]
        s = starts[ai]
        f = s + dur
        for m in members:
            node_start[m] = s
            node_finish[m] = f

    # --- activation memory -------------------------------------------------
    intervals = []
    for i in ix.act_nodes:
        s = node_start[i]
        succ = ix.succs[i]
        if succ:
            e = node_finish[succ[0]]
            for j in succ[1:]:
                if node_finish[j] > e:
                    e = node_finish[j]
        else:
            e = node_finish[i]
        if e > s:
            intervals.append((s, e, ix.output_bytes[i]))
    if intervals:
        points = set()
        for (s, e, b) in intervals:
            points.add(s)
            points.add(e)
        peak = 0
        for t in points:
            tot = 0
            for (s, e, b) in intervals:
                if s <= t < e:
                    tot += b
            if tot > peak:
                peak = tot
        if peak > ix.mem_bound:
            return None

    # --- energy ------------------------------------------------------------
    windows = []
    phone_nj = 0
    for ai in range(len(actions)):
        dev, members, dur = actions[ai]
        s = starts[ai]
        f = s + dur
        if dev == "SERVER":
            w_lo = s - ix.wake_us
            if w_lo < 0:
                w_lo = 0
            w_hi = f + ix.idle_entry_us
            if w_hi > ix.horizon:
                w_hi = ix.horizon
            windows.append((w_lo, w_hi))
        else:
            phone_nj += ix.devices[dev]["active_mw"] * (f - s)
            for m in members:
                phone_nj += int(ix.routes[m][dev]["extra_energy_nj"])

    active_us = 0
    window_count = 0
    if windows:
        windows.sort()
        cur_lo, cur_hi = windows[0]
        for (w_lo, w_hi) in windows[1:]:
            if w_lo <= cur_hi:
                # Overlapping or touching: merge.
                if w_hi > cur_hi:
                    cur_hi = w_hi
            else:
                active_us += cur_hi - cur_lo
                window_count += 1
                cur_lo, cur_hi = w_lo, w_hi
        active_us += cur_hi - cur_lo
        window_count += 1

    server_nj = (ix.p8_mw * ix.horizon
                 + (ix.p0_mw - ix.p8_mw) * active_us
                 + ix.transition_nj * window_count)
    total_nj = server_nj + phone_nj

    # --- outcomes ----------------------------------------------------------
    misses = 0
    total_lateness = 0
    for r in range(ix.n_requests):
        terminal_finish = node_finish[ix.req_terminal[r]]
        lateness = terminal_finish - ix.req_deadline[r]
        if lateness > 0:
            misses += 1
            total_lateness += lateness

    return (misses, total_lateness, -(ix.n_requests - misses), total_nj)


# ---------------------------------------------------------------------------
# Depth-first enumeration of start times for one configuration
# ---------------------------------------------------------------------------

def _order_actions(ix, actions):
    """Deterministic action order; topological w.r.t. precedence when one exists.

    This is a search-order choice only.  It never changes the set of schedules that
    are explored: the leaf re-checks full legality regardless of order.
    """
    n_act = len(actions)
    node_action = {}
    for ai in range(n_act):
        for m in actions[ai][1]:
            node_action[m] = ai

    succ = [set() for _ in range(n_act)]
    indeg = [0] * n_act
    for (p, q) in ix.edges:
        pa = node_action[p]
        qa = node_action[q]
        if pa != qa and qa not in succ[pa]:
            succ[pa].add(qa)
            indeg[qa] += 1

    order = []
    ready = sorted(ai for ai in range(n_act) if indeg[ai] == 0)
    while ready:
        ai = ready.pop(0)
        order.append(ai)
        for bi in sorted(succ[ai]):
            indeg[bi] -= 1
            if indeg[bi] == 0:
                ready.append(bi)
        ready.sort()
    if len(order) != n_act:
        # A cycle among actions (possible only with zero-duration actions); fall back
        # to the natural order.  The leaf check still enforces rule 6 exactly.
        order = list(range(n_act))
    return order, node_action


def _search_configuration(ix, actions, best):
    """Explore every start-time combination for `actions`; return the improved best."""
    n_act = len(actions)

    domains = []
    for a in actions:
        dom = _start_domain(ix, a)
        if dom is None:
            return best
        domains.append(dom)

    order, node_action = _order_actions(ix, actions)

    starts = [0] * n_act
    placed = []  # action indices already assigned, in placement order

    def dfs(k):
        nonlocal best
        if k == n_act:
            if not _fully_legal(ix, actions, starts, node_action):
                return
            obj = _evaluate(ix, actions, starts, node_action)
            if obj is not None and (best is None or obj < best):
                best = obj
            return

        ai = order[k]
        dev, members, dur = actions[ai]
        lo, hi = domains[ai]

        # Early rejection for rule 6 against predecessors already placed: a start
        # earlier than a placed predecessor's finish is already illegal.  This only
        # removes illegal schedules.
        for bi in placed:
            for m in actions[bi][1]:
                for s_node in ix.succs[m]:
                    if node_action[s_node] == ai:
                        f = starts[bi] + actions[bi][2]
                        if f > lo:
                            lo = f
        if lo > hi:
            return

        # Same-device actions already placed, for rule 7 early rejection.
        busy = []
        for bi in placed:
            if actions[bi][0] == dev:
                busy.append((starts[bi], starts[bi] + actions[bi][2]))

        for s in range(lo, hi + 1):
            f = s + dur
            conflict = False
            for (b_s, b_f) in busy:
                if s < b_f and b_s < f:
                    conflict = True
                    break
            if conflict:
                continue
            # Rule 6 against successors already placed (only when both endpoints are
            # fixed): reject a start whose finish lands after a placed successor.
            bad = False
            for m in members:
                for s_node in ix.succs[m]:
                    sa = node_action.get(s_node)
                    if sa is not None and sa in placed_set and f > starts[sa]:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue

            starts[ai] = s
            placed.append(ai)
            placed_set.add(ai)
            dfs(k + 1)
            placed.pop()
            placed_set.discard(ai)

    placed_set = set()
    dfs(0)
    return best


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def optimum(inst, max_enum=REFERENCE_MAX_ENUM):
    """Return the lexicographically minimal 4-tuple objective over all legal schedules.

    Raises ReferenceOutOfDomain if the instance is outside the declared bounds.
    Raises RuntimeError if no legal schedule exists.
    """
    if isinstance(inst, str):
        with open(inst, "r") as fh:
            inst = json.load(fh)

    ix = _Instance(inst)

    if ix.n > REFERENCE_MAX_NODES:
        raise ReferenceOutOfDomain(
            "node count %d exceeds REFERENCE_MAX_NODES=%d" % (ix.n, REFERENCE_MAX_NODES))

    # Pass 1: enumerate configurations, check the declared bounds up front, and keep
    # only the configurations that have a non-empty start domain.
    configurations = []
    total_enum = 0
    for actions in _iter_configurations(ix):
        if len(actions) > REFERENCE_MAX_ACTIONS:
            raise ReferenceOutOfDomain(
                "action count %d exceeds REFERENCE_MAX_ACTIONS=%d"
                % (len(actions), REFERENCE_MAX_ACTIONS))
        combos = 1
        feasible = True
        for a in actions:
            dom = _start_domain(ix, a)
            if dom is None:
                feasible = False
                break
            combos *= (dom[1] - dom[0] + 1)
        if not feasible:
            continue
        total_enum += combos
        if total_enum > max_enum:
            raise ReferenceOutOfDomain(
                "enumeration would exceed max_enum=%d (already %d start-time "
                "combinations)" % (max_enum, total_enum))
        configurations.append(actions)

    # Pass 2: exhaustive search.
    best = None
    for actions in configurations:
        best = _search_configuration(ix, actions, best)

    if best is None:
        raise RuntimeError("no legal schedule exists for instance %r"
                           % (inst.get("instance_id"),))
    return best
