"""Full BC1 decode of a BCPack chunk (endpoints + indices) and harness mip-consistency scoring.

    python -I tools/sem/decode_bc1.py --side 512                 # score every pair up to 512 px
    python -I tools/sem/decode_bc1.py --side 4096 --n 0

decode(syms, window, quad) -> BC1 bytes in stream (= stored Gfx10) order:
  endpoints: back-reference rule (endpoints.decode_window) with the fitted preset window
             (/tmp/bcpack-shared/window_table.json, tools/sem/window_fit_exact.py)
  indices:   lanes 8-11 = TL, TR, BL, BR 2x2 groups; symbol -> 4 BC1 codes (TL, TR, BL, BR pixel
             of the group) from sem-indices' quad_table.json
Scored with harness mipscore.score(texture, bytes, 'gfx10') next to psnr_flat.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import endpoints as ep  # noqa: E402
import mipscore  # noqa: E402
from window_fit_exact import quad_codes  # noqa: E402

WIN = ep.load_window()
QT = quad_codes()
# bit shift of pixel (pos in group) for each group: pixel (x, y) -> bit 2*(4y+x)
SHIFT = np.array([[2 * (4 * (2 * (g // 2) + pos // 2) + 2 * (g % 2) + pos % 2) for pos in range(4)]
                  for g in range(4)], dtype=np.uint64)


def decode(s, window=None, quad=QT):
    E = ep.decode_window(s, "zigA", "zero", history=WIN if window is None else window)
    n = s.shape[1]
    idx = np.zeros(n, np.uint64)
    for g in range(4):
        codes = quad[s[8 + g]].astype(np.uint64)           # (n, 4)
        idx |= np.bitwise_or.reduce(codes << SHIFT[g][None, :], axis=1)
    c0 = (E[:, 0] << 11) | (E[:, 1] << 5) | E[:, 2]
    c1 = (E[:, 3] << 11) | (E[:, 4] << 5) | E[:, 5]
    out = np.zeros(n, dtype=[("c0", "<u2"), ("c1", "<u2"), ("idx", "<u4")])
    out["c0"], out["c1"], out["idx"] = c0, c1, idx
    return out.tobytes()


def one(p):
    tid = p["texture_id"]
    try:
        s = common.decode_chunk(p["mip0_chunks"][0][0])
    except Exception as e:
        return tid, None, str(e)
    n = s.shape[1]
    dist = [ep.back_distance(v) for v in s[0]]
    unk = sum(1 for i in range(n) if dist[i] > i and WIN[(i - dist[i]) % 256] == (0,) * 6)
    r = mipscore.score(tid, decode(s), "gfx10")
    return tid, (r["psnr"], r["psnr_flat"], unk, p["w"] * p["h"]), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=512)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    ps = [p for p in common.pairs() if max(p["w"], p["h"]) <= a.side]
    ps.sort(key=lambda p: -p["w"] * p["h"])
    if a.n:
        ps = ps[:a.n]
    rows, fails = [], []
    with Pool(a.jobs) as pool:
        for tid, r, err in pool.imap_unordered(one, ps, chunksize=1):
            if r is None:
                fails.append((tid, err))
                continue
            rows.append((tid,) + r)
    R = np.array([r[1:] for r in rows], dtype=float)
    ps_, fl, unk = R[:, 0], R[:, 1], R[:, 2]
    print("pairs %d scored %d failed %d" % (len(ps), len(rows), len(fails)))
    for t, e in fails[:10]:
        print("  fail", t, e)
    print("all pre-start refs in fitted slots: %d / %d" % ((unk == 0).sum(), len(R)))
    for name, m in (("all", unk >= 0), ("fully covered", unk == 0)):
        d = ps_[m] - fl[m]
        print("%s (n=%d): psnr median %.1f p10 %.1f p90 %.1f | flat median %.1f | beats flat %d, by >3 dB %d, psnr>=30: %d, >=25: %d" % (
            name, m.sum(), np.median(ps_[m]), np.percentile(ps_[m], 10), np.percentile(ps_[m], 90),
            np.median(fl[m]), (d > 0).sum(), (d > 3).sum(), (ps_[m] >= 30).sum(), (ps_[m] >= 25).sum()))
    print("psnr deciles:", np.round(np.percentile(ps_, np.arange(10, 100, 10)), 1).tolist())
    worst = sorted(rows, key=lambda r: r[1] - r[2])[:8]
    print("worst (psnr - flat):")
    for r in worst:
        print("  %-50s psnr %5.1f flat %5.1f unfitted-refs %d" % (r[0][-50:], r[1], r[2], r[3]))


if __name__ == "__main__":
    main()
