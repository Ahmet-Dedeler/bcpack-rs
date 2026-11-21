"""Score endpoint hypotheses on BCPack/mip-1 pairs (index-free segment-distance metric).

    python3 -I tools/sem/run_hyp.py --n 24 --side 256        # sample pairs, all hypotheses
    python3 -I tools/sem/run_hyp.py --control                # positive control: plain mip 0 BC1

Output: per hypothesis, median over textures of (median distance, frac<12), and the shuffled
control.  Game data is read in place; nothing is written except caches in /tmp.
"""
from __future__ import annotations

import argparse
import itertools
import os
import struct
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import endpoints as ep  # noqa: E402

HYPS = [(r, p, e) for r in ("zigA", "zigB", "twos", "raw")
        for p in ("none", "prev", "left", "up", "avg", "med")
        for e in ("own", "e0")]


def pick(n, side, seed=1, min_std=12.0):
    rng = np.random.default_rng(seed)
    ps = [p for p in common.pairs() if max(p["w"], p["h"]) <= side and p["w"] * p["h"] >= 128 * 128]
    rng.shuffle(ps)
    out = []
    for p in ps:
        try:
            tgt, ok, xy = common.pair_targets(p["texture_id"])
        except Exception:
            continue
        if tgt[ok].std(0).max() < min_std:
            continue
        out.append(p)
        if len(out) >= n:
            break
    return out


def one(args):
    p, hyps, order = args
    tid = p["texture_id"]
    try:
        s = common.decode_chunk(p["mip0_chunks"][0][0])
    except Exception as e:
        return tid, None, str(e)
    tgt, ok, xy = common.pair_targets(tid)
    n = min(s.shape[1], len(ok))
    s, tgt, ok, xy = s[:, :n], tgt[:n], ok[:n], xy[:n]
    xy = np.where(xy < 0, 0, xy)
    res = {}
    for h in hyps:
        E = ep.decode(s, xy, *h)
        res[h] = ep.score(E, tgt, ok)
    return tid, res, None


def control_one(p):
    """Plain mip 0 of a control texture through the same metric (true endpoints)."""
    import mipscore
    tid = p["texture_id"]
    data = mipscore.plain_mip0_bytes(tid)
    tgt, ok, xy = common.pair_targets(tid)
    n = min(len(data) // 8, len(ok))
    c = np.array([struct.unpack_from("<HH", data, 8 * i) for i in range(n)])
    E = np.stack([(c[:, 0] >> 11) & 31, (c[:, 0] >> 5) & 63, c[:, 0] & 31,
                  (c[:, 1] >> 11) & 31, (c[:, 1] >> 5) & 63, c[:, 1] & 31], 1)
    return tid, ep.score(E, tgt[:n], ok[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--side", type=int, default=256)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--hyps", default="")
    args = ap.parse_args()
    if args.control:
        ps = [p for p in common.pairs("control") if max(p["w"], p["h"]) <= args.side and p["w"] >= 64 and p["h"] >= 64]
        rng = np.random.default_rng(1); rng.shuffle(ps)
        with Pool(os.cpu_count()) as pool:
            rs = pool.map(control_one, ps[:args.n])
        m = np.array([[r["med"], r["f12"], r["med_ctl"], r["f12_ctl"]] for _, r in rs])
        print("CONTROL true plain mip0, n=%d: med %.1f f12 %.2f | shuffled med %.1f f12 %.2f" % (len(rs), *np.median(m, 0)))
        return
    ps = pick(args.n, args.side)
    hyps = HYPS if not args.hyps else [tuple(h.split(":")) for h in args.hyps.split(",")]
    with Pool(os.cpu_count()) as pool:
        rs = pool.map(one, [(p, hyps, "stored") for p in ps])
    good = [r for r in rs if r[1] is not None]
    for tid, _, err in rs:
        if err:
            print("skip", tid, err)
    print("textures scored:", len(good))
    rows = []
    for h in hyps:
        m = np.array([[r[1][h]["med"], r[1][h]["f12"], r[1][h]["med_ctl"], r[1][h]["f12_ctl"]] for r in good])
        rows.append((np.median(m[:, 0]), h, np.median(m, 0)))
    rows.sort()
    for _, h, m in rows:
        print("%-5s %-5s %-4s  med %6.1f f12 %.2f | shuffled med %6.1f f12 %.2f" % (*h, *m))


if __name__ == "__main__":
    main()
