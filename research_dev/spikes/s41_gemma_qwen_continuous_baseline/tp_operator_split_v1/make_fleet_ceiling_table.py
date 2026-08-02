#!/usr/bin/env python3
"""Render the N-phone idealized fleet-ceiling table as SVG + PNG.

Every number here is a CEILING derived from the single measured N=1 point:
linear power scaling with offloaded work, latency held at parity, and no
coordination overhead. Only the N=1 row has any measured backing, and even
there the energy term is unmeasured. Rendering, not evidence.
"""

import subprocess
import sys

import cairosvg

OUT = "fleet_ceiling_table"

# Okabe-Ito colorblind-safe palette
INK       = "#1a1d21"
MUTED     = "#5b6470"
RULE      = "#d4d9e0"
HEADER_BG = "#2b3440"
STRIPE    = "#f6f8fa"
GREEN     = "#009E73"
VERMILION = "#D55E00"
BLUE      = "#0072B2"

COLUMNS = [
    ("Phones",                    90, "middle"),
    ("Columns/\nphone",          125, "end"),
    ("FFN columns\noffloaded",   150, "end"),
    ("Compute-only\nserver work saved", 185, "end"),
    ("Maximum\nspeedup ceiling", 165, "end"),
    ("Ideal server\npower",      140, "end"),
    ("Phone\npower",             115, "end"),
    ("Best-case\nfleet saving",  155, "end"),
]

ROWS = [
    ("1",  "1,792", "10.3%",  "6.9%", "1.07x", "116.1 W",   "4.5 W",   3.3),
    ("2",  "1,792", "20.6%", "13.7%", "1.16x", "107.6 W",     "9 W",   6.5),
    ("4",  "1,792", "41.2%", "27.5%", "1.38x",  "90.5 W",    "18 W",  13.0),
    ("8",  "1,792", "82.4%", "54.9%", "2.22x",  "56.2 W",    "36 W",  26.0),
    ("16", "1,088",  "100%", "66.7%", "3.00x",  "41.6 W",    "72 W",   8.9),
    ("32",   "544",  "100%", "66.7%", "3.00x",  "41.6 W",   "144 W", -48.8),
]

PAD_X, PAD_TOP = 44, 26
HEAD_H, ROW_H = 74, 46
TITLE_H = 74
TABLE_W = sum(c[1] for c in COLUMNS)
W = TABLE_W + 2 * PAD_X
FOOT_H = 96
H = PAD_TOP + TITLE_H + HEAD_H + ROW_H * len(ROWS) + FOOT_H


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def saving_fill(v):
    """Shade positive savings green by magnitude; negative is vermillion."""
    if v < 0:
        return VERMILION, "#ffffff", "700"
    alpha = 0.10 + 0.62 * min(v / 26.0, 1.0)
    text = "#ffffff" if alpha > 0.45 else INK
    weight = "700" if v >= 26.0 else "600"
    return f"rgba(0,158,115,{alpha:.3f})", text, weight


def build():
    p = []
    a = p.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
      f'viewBox="0 0 {W} {H}" font-family="DejaVu Sans, Helvetica, Arial, sans-serif">')
    a(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

    # ---- title ----
    y = PAD_TOP + 26
    a(f'<text x="{PAD_X}" y="{y}" font-size="23" font-weight="700" fill="{INK}">'
      f'Phone FFN offload &#8212; idealized fleet ceiling</text>')
    a(f'<text x="{PAD_X}" y="{y + 25}" font-size="14" fill="{MUTED}">'
      f'Qwen3-14B, M=1 decode, 17,408 FFN columns. Upper bound only: assumes linear power scaling, '
      f'latency parity, zero coordination cost.</text>')

    top = PAD_TOP + TITLE_H

    # ---- header ----
    a(f'<rect x="{PAD_X}" y="{top}" width="{TABLE_W}" height="{HEAD_H}" fill="{HEADER_BG}"/>')
    x = PAD_X
    for label, cw, anchor in COLUMNS:
        lines = label.split("\n")
        tx = x + cw / 2 if anchor == "middle" else x + cw - 14
        y0 = top + HEAD_H / 2 - (len(lines) - 1) * 9 + 5
        for i, ln in enumerate(lines):
            a(f'<text x="{tx:.0f}" y="{y0 + i * 18:.0f}" font-size="13.5" font-weight="600" '
              f'fill="#ffffff" text-anchor="{anchor}">{esc(ln)}</text>')
        x += cw

    # ---- rows ----
    best = max(r[7] for r in ROWS)
    for ri, row in enumerate(ROWS):
        ry = top + HEAD_H + ri * ROW_H
        is_best = row[7] == best
        if is_best:
            a(f'<rect x="{PAD_X}" y="{ry}" width="{TABLE_W}" height="{ROW_H}" fill="#eef7f4"/>')
        elif ri % 2 == 1:
            a(f'<rect x="{PAD_X}" y="{ry}" width="{TABLE_W}" height="{ROW_H}" fill="{STRIPE}"/>')
        a(f'<line x1="{PAD_X}" y1="{ry}" x2="{PAD_X + TABLE_W}" y2="{ry}" '
          f'stroke="{RULE}" stroke-width="1"/>')

        ty = ry + ROW_H / 2 + 5
        x = PAD_X
        for ci, (label, cw, anchor) in enumerate(COLUMNS):
            val = row[ci]
            if ci == 7:
                fill, tcol, wgt = saving_fill(val)
                a(f'<rect x="{x + 6}" y="{ry + 7}" width="{cw - 20}" height="{ROW_H - 14}" '
                  f'rx="5" fill="{fill}"/>')
                a(f'<text x="{x + cw - 20:.0f}" y="{ty:.0f}" font-size="15" font-weight="{wgt}" '
                  f'fill="{tcol}" text-anchor="end">{val:+.1f}%</text>')
            else:
                tx = x + cw / 2 if anchor == "middle" else x + cw - 14
                wgt = "700" if ci == 0 else ("600" if (ci == 4 and is_best) else "400")
                col = BLUE if (ci == 4 and is_best) else INK
                a(f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" font-weight="{wgt}" '
                  f'fill="{col}" text-anchor="{anchor}">{esc(str(val))}</text>')
            x += cw

    bot = top + HEAD_H + ROW_H * len(ROWS)
    a(f'<line x1="{PAD_X}" y1="{bot}" x2="{PAD_X + TABLE_W}" y2="{bot}" '
      f'stroke="{HEADER_BG}" stroke-width="2"/>')

    # marker for the peak row
    peak = [i for i, r in enumerate(ROWS) if r[7] == best][0]
    my = top + HEAD_H + peak * ROW_H + ROW_H / 2 + 5
    a(f'<text x="{PAD_X - 12}" y="{my:.0f}" font-size="15" font-weight="700" '
      f'fill="{GREEN}" text-anchor="end">&#9654;</text>')

    # ---- footnote ----
    fy = bot + 26
    a(f'<text x="{PAD_X}" y="{fy}" font-size="12.5" fill="{MUTED}">'
      f'Offloading gate+up only; the down projection stays on the server, capping compute savings at 66.7%. '
      f'Server power scaled from a 124.7 W measured baseline.</text>')
    a(f'<text x="{PAD_X}" y="{fy + 19}" font-size="12.5" fill="{MUTED}">'
      f'Beyond 16 phones the column budget is exhausted, so added phones raise fleet power without '
      f'removing further server work.</text>')
    a(f'<text x="{PAD_X}" y="{fy + 40}" font-size="12.5" fill="{VERMILION}">'
      f'<tspan font-weight="700">Modelled, not measured.</tspan>'
      f'<tspan> The only measured configuration is N=1, where the observed speedup is '
      f'~0.5&#8211;0.9% (not 1.07x).</tspan></text>')
    a(f'<text x="{PAD_X}" y="{fy + 59}" font-size="12.5" fill="{VERMILION}">'
      f'Fleet energy has never been measured at any N; the one measured energy acquisition was '
      f'+30.5% per layer, a loss.</text>')

    a('</svg>')
    return "\n".join(p)


svg = build()
with open(f"{OUT}.svg", "w") as f:
    f.write(svg)
cairosvg.svg2png(bytestring=svg.encode(), write_to=f"{OUT}.png", scale=2.0)
print(f"wrote {OUT}.svg and {OUT}.png")
