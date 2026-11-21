"""Fit the initial (pre-start) back-reference window slot by slot, jointly across textures.

    python -I tools/sem/window_fit.py --n 400 --side 512 [--slots 254,193,128]

Every block's reference chain ends at exactly one root: d == 0 (absolute zero) or a pre-start
reference to ring slot (i - d) mod 256.  Then fields = (root value + sum of residuals along the
chain) mod 2^w, so each slot's value can be fitted independently from the blocks rooted at it.
For each slot: per-texture fit (coordinate search on the robust segment distance) and a joint fit
over all textures that use it.  If the window is a fixed table, per-texture fits agree with the
joint fit and the joint fit scores as well as the per-texture ones.
Writes /tmp/bcpack-sem-endpoints/window_fit.json.
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
    try:
        s = common.decode_chunk(p["mip0_chunks"][0][0])
        tgt, ok, xy = common.pair_targets(p["texture_id"])
    except Exception:
        return None
    n = min(s.shape[1], len(ok))
    root, S = chains(s[:, :n])
    return p["texture_id"], root, S, tgt[:n], ok[:n]


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
    return W, best


BY = {}


def fit_slot(args):
    """Joint multi-start fit of one slot (runs in a worker; BY is inherited via fork)."""
    sl, max_joint = args
    items = BY[sl]
    rng = np.random.default_rng(sl)
    Sall = np.concatenate([x[1] for x in items]); Tall = np.concatenate([x[2] for x in items])
    if len(Sall) > max_joint:                      # subsample for the search, score on all
        k = rng.choice(len(Sall), max_joint, replace=False)
        Ss, Ts = Sall[k], Tall[k]
    else:
        Ss, Ts = Sall, Tall
    Wj, _ = fit(Ss, Ts)
    seeds = {tuple(fit(S, tgt, start=Wj)[0]) for _, S, tgt in items}
    best = cost(Wj, Ss, Ts).mean()
    for sd in seeds:
        W2, c2 = fit(Ss, Ts, start=list(sd), passes=2)
        if c2 < best:
            Wj, best = W2, c2
    cj = float(cost(Wj, Sall, Tall).mean())
    lines = ["slot %3d: %d textures, %d blocks; joint W=%s mean capped dist %.1f" % (sl, len(items), len(Sall), Wj, cj)]
    per = []
    for tid, S, tgt in items:
        Wt, ct = fit(S, tgt, start=Wj)
        cjt = cost(Wj, S, tgt).mean(); cz = cost([0] * 6, S, tgt).mean()
        per.append((tid, Wt, round(float(ct), 1), round(float(cjt), 1), round(float(cz), 1)))
        lines.append("    %-44s own W=%-26s own %.1f  joint %.1f  zero %.1f  (%d blocks)" % (tid[-44:], Wt, ct, cjt, cz, len(S)))
    near = sum(1 for p in per if p[3] <= p[2] + 2)
    lines[0] += "; joint within 2 of own: %d/%d" % (near, len(per))
    return sl, "\n".join(lines), dict(joint=Wj, joint_cost=cj, near=near, per=per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--side", type=int, default=512)
    ap.add_argument("--slots", default="")
    ap.add_argument("--min-blocks", type=int, default=24)
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--max-joint", type=int, default=20000)
    a = ap.parse_args()
    ps = [p for p in common.pairs() if max(p["w"], p["h"]) <= a.side]
    rng = np.random.default_rng(5)
    rng.shuffle(ps)
    with Pool(16) as pool:
        data = [d for d in pool.map(load, ps[:a.n]) if d]
    by_slot = defaultdict(list)
    for tid, root, S, tgt, ok in data:
        for sl in set(root.tolist()) - {-1}:
            m = (root == sl) & ok
            if m.sum() >= a.min_blocks:
                by_slot[sl].append((tid, S[m], tgt[m]))
    slots = [int(x) for x in a.slots.split(",")] if a.slots else sorted(by_slot, key=lambda s: -len(by_slot[s]))[:a.top]
    global BY
    BY = by_slot
    with Pool(min(16, len(slots))) as pool:
        results = pool.map(fit_slot, [(sl, a.max_joint) for sl in slots if by_slot[sl]])
    out = {}
    for sl, txt, rec in results:
        print(txt, flush=True)
        out[sl] = rec
    with open(os.path.join(common.SCR, "window_fit.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
