"""End-to-end mip-consistency check of the endpoint rule with the harness scorer.

    python -I tools/sem/mip_endpoints.py --n 40 --side 256

BC1 blocks = decoded c0, c1 (back-reference rule, zero window) + a fixed index checkerboard
(0,1 / 1,0 in every 2x2 quadrant), so each mip-1 pixel is predicted by the endpoint midpoint.
That uses no information from the reference.  Scored with harness mipscore.score(order='gfx10')
next to psnr_flat and two controls: endpoints from the best non-reference hypothesis (left
neighbour) and the same rule endpoints with blocks shuffled.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import endpoints as ep  # noqa: E402
import mipscore  # noqa: E402
import run_hyp  # noqa: E402

CHECKER = 0
for py in range(4):
    for px in range(4):
        CHECKER |= ((px + py) & 1) << (2 * (py * 4 + px))


def blocks_from(E):
    out = bytearray()
    for r0, g0, b0, r1, g1, b1 in E.tolist():
        out += struct.pack("<HHI", r0 << 11 | g0 << 5 | b0, r1 << 11 | g1 << 5 | b1, CHECKER)
    return bytes(out)


WIN = ep.load_window()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--side", type=int, default=256)
    a = ap.parse_args()
    rows = []
    for p in run_hyp.pick(a.n, a.side):
        tid = p["texture_id"]
        try:
            s = common.decode_chunk(p["mip0_chunks"][0][0])
        except RuntimeError as e:            # simulator does not converge on this chunk
            print("skip", tid, e)
            continue
        tgt, ok, xy = common.pair_targets(tid)
        n = s.shape[1]
        dist = [ep.back_distance(v) for v in s[0]]
        npre = sum(1 for i in range(n) if dist[i] > i)
        E0 = ep.decode_window(s, "zigA", "zero")
        r0 = mipscore.score(tid, blocks_from(E0), "gfx10")
        E = ep.decode_window(s, "zigA", "zero", history=WIN)
        r = mipscore.score(tid, blocks_from(E), "gfx10")
        El = ep.decode(s, np.maximum(xy, 0), "zigA", "left", "own")
        rl = mipscore.score(tid, blocks_from(El), "gfx10")
        perm = np.random.default_rng(0).permutation(n)
        rs = mipscore.score(tid, blocks_from(E[perm]), "gfx10")
        unk = sum(1 for i in range(n) if dist[i] > i and WIN[(i - dist[i]) % 256] == (0,) * 6)
        rows.append((npre, r["psnr"], r["psnr_flat"], rl["psnr"], rs["psnr"], r0["psnr"], unk))
        print("%-46s pre %3d unfitted %3d  rule+window %5.1f dB  (zero window %5.1f)  flat %5.1f  left %5.1f  shuffled %5.1f" % (
            tid[-46:], npre, unk, r["psnr"], r0["psnr"], r["psnr_flat"], rl["psnr"], rs["psnr"]))
    R = np.array(rows)
    for name, m in (("no pre-start refs", R[:, 0] == 0), ("pre-start refs all in fitted slots", (R[:, 0] > 0) & (R[:, 6] == 0)),
                    ("some unfitted slot", R[:, 6] > 0), ("all", R[:, 0] >= 0)):
        if m.any():
            print("%s (n=%d): median rule+window %.1f dB, flat %.1f, left %.1f, shuffled %.1f, zero window %.1f; rule > flat+3 in %d" % (
                name, m.sum(), *np.median(R[m, 1:6], 0), (R[m, 1] > R[m, 2] + 3).sum()))


if __name__ == "__main__":
    main()
