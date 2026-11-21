"""Reproduce the endpoint-rule evidence table (index-free segment-distance metric).

    /tmp/bcpack-sem-indices/venv/bin/python -I tools/sem/score_rule.py --n 40 --side 256

(Use a Python/numpy without the 3.14 temporary-elision bug; see docs/findings/sem-endpoints.md.)

For each sampled BCPack/plain-mip-1 pair (seeded, textures with mip-1 std >= 12) prints the
median distance from the mip-1 2x2 mean to the decoded c0-c1 segment for
  rule      back-reference (lane 0) + zigzag residual mod 2^width, d=0 -> zero predictor
  rule+root same, with the value at the pre-start reference solved per texture (only when
            exactly one pre-start reference exists; coordinate search, 2 passes)
  left      best earlier 2D predictor (left neighbour, zigzag, wrap)
  ctl       rule endpoints scored against shuffled targets (chance level)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import endpoints as ep  # noqa: E402
import run_hyp  # noqa: E402


def solve_root(s, tgt, ok):
    def run(P0):
        return ep.score(ep.decode_window(s, "zigA", "zero", history=[P0] * 256), tgt, ok)["med"]
    P = [0] * 6
    best = run(tuple(P))
    for _ in range(2):
        for k in range(6):
            for o in range(int(ep.MOD[k])):
                Q = list(P); Q[k] = o
                m = run(tuple(Q))
                if m < best:
                    best, P = m, Q
    return best, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--side", type=int, default=256)
    ap.add_argument("--no-root", action="store_true")
    a = ap.parse_args()
    rows = []
    for p in run_hyp.pick(a.n, a.side):
        try:
            s = common.decode_chunk(p["mip0_chunks"][0][0])
        except Exception as e:
            print("skip", p["texture_id"], e)
            continue
        tgt, ok, xy = common.pair_targets(p["texture_id"])
        n = s.shape[1]
        dist = [ep.back_distance(v) for v in s[0]]
        npre = sum(1 for i in range(n) if dist[i] > i)
        r = ep.score(ep.decode_window(s, "zigA", "zero"), tgt, ok)
        lf = ep.score(ep.decode(s, np.maximum(xy, 0), "zigA", "left", "own"), tgt, ok)
        root = (float("nan"), None)
        if npre == 1 and not a.no_root:
            root = solve_root(s, tgt, ok)
        rows.append((r["med"], root[0], lf["med"], r["med_ctl"], npre))
        print("%-46s pre %3d  rule %6.1f  rule+root %6.1f %-26s left %6.1f  ctl %6.1f" % (
            p["texture_id"][-46:], npre, r["med"], root[0], root[1] or "", lf["med"], r["med_ctl"]))
    R = np.array(rows, dtype=float)
    z = R[:, 4] == 0
    print("textures %d; with no pre-start refs %d: rule median %.1f (<4: %d)" % (
        len(R), z.sum(), np.median(R[z, 0]), (R[z, 0] < 4).sum()))
    one = R[:, 4] == 1
    if one.any() and not a.no_root:
        print("one pre-start ref %d: rule %.1f -> rule+root %.1f (<7: %d)" % (
            one.sum(), np.median(R[one, 0]), np.nanmedian(R[one, 1]), (R[one, 1] < 7).sum()))
    print("all: left median %.1f, ctl median %.1f" % (np.median(R[:, 2]), np.median(R[:, 3])))


if __name__ == "__main__":
    main()
