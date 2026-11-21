"""End-to-end mip-consistency check of the endpoint rule with the harness scorer.

    python -I tools/sem/mip_endpoints.py --n 40 --side 256            # textures with detail
    python -I tools/sem/mip_endpoints.py --n 0 --side 512 --min-std 0  # every pair up to 512 px

BC1 blocks = decoded c0, c1 (back-reference rule; preset window from
/tmp/bcpack-shared/window_table.json, see window_fit.py) + a fixed index checkerboard
(0,1 / 1,0 in every 2x2 quadrant), so each mip-1 pixel is predicted by the endpoint midpoint.
That uses no information from the reference.  Scored with harness mipscore.score(order='gfx10')
next to psnr_flat and two controls: the same endpoints with an all-zero window, and with blocks
shuffled.
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


def one(p):
    """Score one pair texture (worker)."""
    tid = p["texture_id"]
    try:
        s = common.decode_chunk(p["mip0_chunks"][0][0])
        tgt, ok, xy = common.pair_targets(tid)
    except Exception as e:               # e.g. the simulator does not converge on this chunk
        return tid, None, str(e)
    n = min(s.shape[1], len(ok))
    s = s[:, :n]
    dist = [ep.back_distance(v) for v in s[0]]
    npre = sum(1 for i in range(n) if dist[i] > i)
    unk = sum(1 for i in range(n) if dist[i] > i and WIN[(i - dist[i]) % 256] == (0,) * 6)
    E0 = ep.decode_window(s, "zigA", "zero")
    r0 = mipscore.score(tid, blocks_from(E0), "gfx10")
    E = ep.decode_window(s, "zigA", "zero", history=WIN)
    r = mipscore.score(tid, blocks_from(E), "gfx10")
    perm = np.random.default_rng(0).permutation(n)
    rs = mipscore.score(tid, blocks_from(E[perm]), "gfx10")
    return tid, (npre, r["psnr"], r["psnr_flat"], rs["psnr"], r0["psnr"], unk), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="pairs to score (0 = all within --side)")
    ap.add_argument("--side", type=int, default=256)
    ap.add_argument("--min-std", type=float, default=12.0, help="pick textures with mip-1 std >= this (0 = no filter)")
    a = ap.parse_args()
    from multiprocessing import Pool
    if a.min_std > 0:
        ps = run_hyp.pick(a.n or 10 ** 9, a.side, min_std=a.min_std)
    else:
        ps = [p for p in common.pairs() if max(p["w"], p["h"]) <= a.side]
        np.random.default_rng(1).shuffle(ps)
        ps = ps[:a.n] if a.n else ps
    rows = []
    with Pool(os.cpu_count()) as pool:
        for tid, r, err in pool.imap_unordered(one, ps, chunksize=1):
            if r is None:
                print("skip", tid, err)
                continue
            rows.append(r)
            print("%-46s pre %3d unfitted %3d  rule+window %5.1f dB  (zero window %5.1f)  flat %5.1f  shuffled %5.1f" % (
                tid[-46:], r[0], r[5], r[1], r[4], r[2], r[3]), flush=True)
    R = np.array(rows)
    for name, m in (("no pre-start refs", R[:, 0] == 0), ("pre-start refs all in fitted slots", (R[:, 0] > 0) & (R[:, 5] == 0)),
                    ("some unfitted slot", R[:, 5] > 0), ("all", R[:, 0] >= 0)):
        if m.any():
            d = R[m, 1] - R[m, 2]
            print("%s (n=%d): median rule+window %.1f dB, flat %.1f, shuffled %.1f, zero window %.1f; "
                  "beats flat: %d (by >3 dB: %d); rule-flat quartiles %.1f/%.1f/%.1f" % (
                      name, m.sum(), *np.median(R[m, 1:5], 0), (d > 0).sum(), (d > 3).sum(),
                      *np.percentile(d, [25, 50, 75])))
    print("rule+window psnr deciles (all):", np.round(np.percentile(R[:, 1], np.arange(10, 100, 10)), 1).tolist())


if __name__ == "__main__":
    main()
