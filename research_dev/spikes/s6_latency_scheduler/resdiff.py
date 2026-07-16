#!/usr/bin/env python3
# Canonical cross-backend residual diff for the S6-L correctness oracle.
#
# oplayerprof dumps the output residual of one MID gemma layer to a raw little-endian
# float32 array <dump>_<mode>_B<b>_C<c>.f32 (rows = n_out, cols = n_embd, row-major).
# This tool diffs a reference dump (e.g. CPU) against one or more candidate dumps
# (stock-OpenCL / xmem / HTP) captured at the SAME injected input and KV state.
#
# It is fail-closed. It rejects, with a nonzero exit, any file that is:
#   - empty,
#   - truncated or carrying trailing bytes (size not a whole rows*n_embd*4),
#   - dimension-mismatched against the reference,
#   - non-finite (inf/nan anywhere).
# It then reports finite/max-abs/rel-L2/residual-component argmax agreement and exits
# nonzero if any candidate's rel-L2 exceeds the gate (default 5e-3) or argmax agreement
# is not exact. A candidate over the gate is PERF_ONLY, not a scheduler-eligible backend.
#
# Pure standard library (no numpy) so it runs anywhere the raw .f32 files are pulled.

import argparse
import math
import os
import struct
import sys


def load_f32(path, n_embd):
    """Return (rows, flat list of floats). Raises ValueError on any structural defect."""
    if not os.path.isfile(path):
        raise ValueError("missing file")
    nbytes = os.path.getsize(path)
    if nbytes == 0:
        raise ValueError("empty file")
    elem = 4
    row_bytes = n_embd * elem
    if nbytes % row_bytes != 0:
        raise ValueError("size %d not a multiple of n_embd*4=%d (truncated or trailing bytes)"
                         % (nbytes, row_bytes))
    rows = nbytes // row_bytes
    if rows == 0:
        raise ValueError("zero rows for n_embd=%d" % n_embd)
    n = rows * n_embd
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) != nbytes:
        raise ValueError("short read (%d of %d bytes)" % (len(raw), nbytes))
    vals = list(struct.unpack("<%df" % n, raw))
    for v in vals:
        if not math.isfinite(v):
            raise ValueError("non-finite value in file")
    return rows, vals


def row_argmax(vals, r, n_embd):
    base = r * n_embd
    best_i, best_v = 0, vals[base]
    for i in range(1, n_embd):
        v = vals[base + i]
        if v > best_v:
            best_v, best_i = v, i
    return best_i


def compare(ref_rows, ref, cand_rows, cand, n_embd):
    if ref_rows != cand_rows:
        raise ValueError("dimension mismatch: ref rows=%d candidate rows=%d" % (ref_rows, cand_rows))
    num = 0.0
    den = 0.0
    max_abs = 0.0
    for k in range(ref_rows * n_embd):
        d = cand[k] - ref[k]
        num += d * d
        den += ref[k] * ref[k]
        a = abs(d)
        if a > max_abs:
            max_abs = a
    rel_l2 = math.sqrt(num) / math.sqrt(den) if den > 0.0 else (0.0 if num == 0.0 else float("inf"))
    agree = 0
    for r in range(ref_rows):
        if row_argmax(ref, r, n_embd) == row_argmax(cand, r, n_embd):
            agree += 1
    return rel_l2, max_abs, agree, ref_rows


def main():
    ap = argparse.ArgumentParser(description="cross-backend residual diff (fail-closed)")
    ap.add_argument("--embd", type=int, required=True, help="n_embd (row width of the .f32 dumps)")
    ap.add_argument("--gate", type=float, default=5e-3, help="rel-L2 production gate (default 5e-3)")
    ap.add_argument("ref", help="reference .f32 dump (e.g. CPU)")
    ap.add_argument("cand", nargs="+", help="candidate .f32 dump(s) to diff against ref")
    a = ap.parse_args()

    if a.embd <= 0:
        print("error: --embd must be positive", file=sys.stderr)
        return 2

    try:
        ref_rows, ref = load_f32(a.ref, a.embd)
    except ValueError as e:
        print("REJECT ref %s: %s" % (a.ref, e), file=sys.stderr)
        return 2

    rc = 0
    print("ref %s: rows=%d n_embd=%d gate=%g" % (a.ref, ref_rows, a.embd, a.gate))
    for c in a.cand:
        try:
            cr, cv = load_f32(c, a.embd)
            rel_l2, max_abs, agree, rows = compare(ref_rows, ref, cr, cv, a.embd)
        except ValueError as e:
            print("REJECT cand %s: %s" % (c, e), file=sys.stderr)
            rc = 2
            continue
        pass_gate = (rel_l2 <= a.gate) and (agree == rows)
        verdict = "PASS" if pass_gate else "FAIL"
        print("%s  %-40s rel_l2=%.3e max_abs=%.3e argmax=%d/%d"
              % (verdict, c, rel_l2, max_abs, agree, rows))
        if not pass_gate:
            rc = max(rc, 1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
