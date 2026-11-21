"""Fit the fixed preset back-reference window (256 slots), jointly across textures.

    python -I tools/sem/window_fit.py --n 0 --side 4096 --top 256 --min-blocks 4 \
        --out /tmp/bcpack-shared/window_table.json

Every block's reference chain ends at exactly one root: d == 0 (absolute zero) or a pre-start
reference to ring slot (i - d) mod 256.  Then fields = (root value + sum of residuals along the
chain) mod 2^w, so each slot's value can be fitted independently from the blocks rooted at it.

Per slot:
  joint      multi-start coordinate search on the robust (capped) segment distance between the
             decoded c0-c1 segment and the mip-1 2x2 target, over all textures using the slot
             (search on <= --max-joint subsampled blocks, scored on all)
  within2    textures whose own best value scores within 2 units of the joint value
  halves     the same joint fit on two disjoint halves of the textures; 'stable' = both halves
             give the joint value to within 1 step per field (circular distance)
Output JSON: {slot: {value, cost, textures, blocks, within2, half_a, half_b, stable, per}}
(derived decoder constants; no game bytes).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import endpoints as ep  # noqa: E402

MOD = [32, 64, 32, 32, 64, 32]


def chains(s):
    """-> root (n,) int: -1 = zero root, else ring slot; S (n,6) residual sums mod 2^w."""
    n = s.shape[1]
    l0 = s[0].tolist()
    res = []
    for k in range(6):
        v = s[2 + k]
        res.append(np.where(v & 1, -((v + 1) >> 1), v >> 1).tolist())
    root = [0] * n
    S = [None] * n
    for i in range(n):
        d = ep.back_distance(l0[i])
        if d == 0:
            root[i], base = -1, (0,) * 6
        elif d <= i:
            root[i], base = root[i - d], S[i - d]
        else:
            root[i], base = (i - d) % 256, (0,) * 6
        S[i] = tuple((base[k] + res[k][i]) % MOD[k] for k in range(6))
    return np.array(root), np.array(S, dtype=np.int64)


def load(p):
    """Worker: decode one pair, return only the blocks rooted at pre-start slots."""
    try:
        s = common.decode_chunk(p["mip0_chunks"][0][0])
        tgt, ok, xy = common.pair_targets(p["texture_id"])
    except Exception as e:
        return p["texture_id"], None, str(e)
    n = min(s.shape[1], len(ok))
    root, S = chains(s[:, :n])
    m = (root >= 0) & ok[:n]
    return p["texture_id"], (root[m], S[m], tgt[:n][m]), None


def cost(W, S, tgt, cap=40.0):
    E = (S + np.asarray(W)) % np.array(MOD)
    c0, c1 = ep.to888(E)
    d = ep.seg_dist(c0, c1, tgt)
    return np.minimum(d, cap)


def fit(S, tgt, start=None, passes=3):
    W = list(start or [0] * 6)
    best = cost(W, S, tgt).mean()
    for _ in range(passes):
        changed = False
        for k in range(6):
            for o in range(MOD[k]):
                Q = list(W); Q[k] = o
                c = cost(Q, S, tgt).mean()
                if c < best - 1e-9:
                    best, W, changed = c, Q, True
        if not changed:
            break
    return W, float(best)


def joint_fit(items, max_joint, rng, max_seed_tex=48):
    Sall = np.concatenate([x[1] for x in items]); Tall = np.concatenate([x[2] for x in items])
    if len(Sall) > max_joint:
        k = rng.choice(len(Sall), max_joint, replace=False)
        Ss, Ts = Sall[k], Tall[k]
    else:
        Ss, Ts = Sall, Tall
    Wj, best = fit(Ss, Ts)
    # seeds: the most common per-texture optima (largest textures first) + independent starts
    from collections import Counter
    big = sorted(items, key=lambda x: -len(x[1]))[:max_seed_tex]
    cnt = Counter(tuple(fit(S, tgt, start=Wj)[0]) for _, S, tgt in big)
    seeds = {sd for sd, _ in cnt.most_common(16)}
    seeds |= {tuple(fit(S, tgt)[0]) for _, S, tgt in big[:8]}
    for sd in seeds:
        W2, c2 = fit(Ss, Ts, start=list(sd), passes=2)
        if c2 < best:
            Wj, best = W2, c2
    return Wj, float(cost(Wj, Sall, Tall).mean()), len(Sall)


def circ(a, b):
    return max(min((x - y) % m, (y - x) % m) for x, y, m in zip(a, b, MOD))


def fit_slot(args):
    sl, max_joint, items = args
    rng = np.random.default_rng(sl)
    Wj, cj, nb = joint_fit(items, max_joint, rng)
    per = []
    for tid, S, tgt in items:
        Wt, ct = fit(S, tgt, start=Wj)
        per.append((tid, Wt, round(ct, 1), round(float(cost(Wj, S, tgt).mean()), 1),
                    round(float(cost([0] * 6, S, tgt).mean()), 1), len(S)))
    within2 = sum(1 for p in per if p[3] <= p[2] + 2)
    ha = hb = None
    stable = None
    if len(items) >= 2:
        idx = rng.permutation(len(items))
        A = [items[i] for i in idx[::2]]; B = [items[i] for i in idx[1::2]]
        ha = joint_fit(A, max_joint, rng)[0]
        hb = joint_fit(B, max_joint, rng)[0]
        stable = circ(ha, Wj) <= 1 and circ(hb, Wj) <= 1
    rec = dict(value=Wj, cost=round(cj, 2), textures=len(items), blocks=nb, within2=within2,
               half_a=ha, half_b=hb, stable=stable, per=per)
    return sl, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="number of pairs (0 = all)")
    ap.add_argument("--side", type=int, default=4096)
    ap.add_argument("--slots", default="")
    ap.add_argument("--min-blocks", type=int, default=4)
    ap.add_argument("--top", type=int, default=256)
    ap.add_argument("--max-joint", type=int, default=20000)
    ap.add_argument("--out", default="/tmp/bcpack-shared/window_table.json")
    a = ap.parse_args()
    ps = [p for p in common.pairs() if max(p["w"], p["h"]) <= a.side]
    rng = np.random.default_rng(5)
    rng.shuffle(ps)
    if a.n:
        ps = ps[:a.n]
    # big textures first so the pool stays busy
    ps.sort(key=lambda p: -p["w"] * p["h"])
    by_slot = defaultdict(list)
    nfail = 0
    with Pool(os.cpu_count()) as pool:
        for tid, d, err in pool.imap_unordered(load, ps, chunksize=1):
            if d is None:
                nfail += 1
                continue
            root, S, tgt = d
            for sl in np.unique(root).tolist():
                m = root == sl
                if m.sum() >= a.min_blocks:
                    by_slot[sl].append((tid, S[m], tgt[m]))
    print("pairs %d, failed %d, slots with data %d" % (len(ps), nfail, len(by_slot)), flush=True)
    slots = [int(x) for x in a.slots.split(",")] if a.slots else \
        sorted(by_slot, key=lambda s: -len(by_slot[s]))[:a.top]
    jobs = [(sl, a.max_joint, by_slot[sl]) for sl in slots if by_slot[sl]]
    jobs.sort(key=lambda j: -len(j[2]))
    out = {}
    with Pool(os.cpu_count()) as pool:
        for sl, rec in pool.imap_unordered(fit_slot, jobs, chunksize=1):
            out[sl] = rec
            print("slot %3d: value %-26s cost %5.2f textures %4d blocks %7d within2 %4d stable %s" % (
                sl, rec["value"], rec["cost"], rec["textures"], rec["blocks"], rec["within2"], rec["stable"]), flush=True)
    with open(a.out, "w") as f:
        json.dump({str(k): out[k] for k in sorted(out)}, f, indent=1)
    print("wrote", a.out, len(out), "slots")


if __name__ == "__main__":
    main()
