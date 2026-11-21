"""Look for structure in the fitted preset window table (window_fit.py output).

    python -I tools/sem/window_structure.py [/tmp/bcpack-shared/window_table.json]

Prints the table by slot with support, the lane-0 code that reaches each slot from block 0,
luma/colour of both endpoints, and checks: luma vs slot order, swapped-pair symmetry
(slot j's (c1, c0) equal to another slot's (c0, c1)), mirror symmetry j <-> 256-j, and whether
usage frequency follows slot order.
"""
from __future__ import annotations

import json
import sys

import numpy as np

MOD = [32, 64, 32, 32, 64, 32]


def rgb(v):
    r0, g0, b0, r1, g1, b1 = v
    f5 = lambda x: (x << 3) | (x >> 2)
    f6 = lambda x: (x << 2) | (x >> 4)
    return (f5(r0), f6(g0), f5(b0)), (f5(r1), f6(g1), f5(b1))


def luma(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def code_from_block0(slot):
    """Lane-0 symbol that makes block 0 reference ring slot `slot` (distance D = 256 - slot)."""
    D = (256 - slot) % 256
    return 2 * D if D <= 127 else 2 * (255 - D) + 1


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bcpack-shared/window_table.json"
    T = {int(k): v for k, v in json.load(open(path)).items()}
    good = {k: v for k, v in T.items() if v["textures"] >= 3 and v.get("stable")}
    print("slots fitted %d; >=3 textures and split-half stable: %d" % (len(T), len(good)))
    print("slot  code0  textures blocks  cost stable  value(R0 G0 B0 / R1 G1 B1)     c0 rgb          c1 rgb          luma0 luma1")
    for k in sorted(T):
        v = T[k]
        c0, c1 = rgb(v["value"])
        print("%4d  %5d  %8d %6d %5.1f %-6s %-30s %-15s %-15s %5.0f %5.0f" % (
            k, code_from_block0(k), v["textures"], v["blocks"], v["cost"], v.get("stable"),
            v["value"], c0, c1, luma(c0), luma(c1)))
    ks = sorted(good)
    if len(ks) > 3:
        L = np.array([(luma(rgb(good[k]["value"])[0]) + luma(rgb(good[k]["value"])[1])) / 2 for k in ks])
        F = np.array([good[k]["textures"] for k in ks])
        print("corr(slot, mean luma) %.2f   corr(slot, usage) %.2f   corr(code0, usage) %.2f" % (
            np.corrcoef(ks, L)[0, 1], np.corrcoef(ks, F)[0, 1],
            np.corrcoef([code_from_block0(k) for k in ks], F)[0, 1]))
    vals = {k: tuple(good[k]["value"]) for k in ks}
    near = lambda a, b: max(min((x - y) % m, (y - x) % m) for x, y, m in zip(a, b, MOD)) <= 1
    sw = [(j, k) for j in ks for k in ks if j < k and near(vals[j][3:] + vals[j][:3], vals[k])]
    print("swapped-pair matches (c1,c0 of j == c0,c1 of k, +-1):", sw)
    mir = [(j, 256 - j) for j in ks if (256 - j) in vals and j < 256 - j and near(vals[j], vals[256 - j])]
    print("mirror j <-> 256-j matches:", mir)
    dup = [(j, k) for j in ks for k in ks if j < k and near(vals[j], vals[k])]
    print("near-duplicate entries (+-1):", dup)
    flat = [k for k in ks if near(vals[k][:3], vals[k][3:])]
    print("entries with c0 ~ c1 (+-1): %d of %d" % (len(flat), len(ks)))


if __name__ == "__main__":
    main()
