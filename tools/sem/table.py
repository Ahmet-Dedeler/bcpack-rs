"""Infer the quadrant codebook: symbol -> four BC1 codes (TL, TR, BL, BR of the 2x2 group).

For every good block (see arrange.py) we have, per quadrant, the 4 upsampled mip-1 values in
palette-weight space u (4-vector). For a candidate pattern with weights v, the expected
squared error is n*(E||u||^2 - 2 v.E[u] + ||v||^2), so only E[u] per symbol is needed.
Symbols 0..3 are fixed (uniform; code = sym ^ 2). The rest are assigned by a min-cost
bijection (Hungarian) between symbols and the 252 non-uniform patterns.
Weights: BC1 code 0 -> 0, 1 -> 1, 2 -> 1/3, 3 -> 2/3.
usage: venv/bin/python -I tools/sem/table.py [--lanes 8,9,10,11] [--split] -> $SEM_SCRATCH/table.npy
"""
import os
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")
QPOS = [(1, 1), (1, 2), (2, 1), (2, 2)]
CW = np.array([0, 1, 1 / 3, 2 / 3])
PATS = np.array([[(p >> (2 * k)) & 3 for k in range(4)] for p in range(256)])   # p0..p3 codes


def stats(lanes, sel=None, min_fit=0.95):
    z = np.load(os.path.join(SCR, "quadring_gfx10.npz"))
    S = z["syms"].astype(np.int64)
    Yr = z["rgb"].astype(np.float32) @ np.array([.299, .587, .114], np.float32)
    mw, cnt = np.load(os.path.join(SCR, "meanw.npy"))
    mw = np.where(cnt >= 30, mw, np.nan)
    L = S[8:12].T
    Wq = mw[L]
    Yq = np.stack([Yr[:, r, c] for r, c in QPOS], 1)
    ok = ~np.isnan(Wq).any(1)
    Wf = np.where(ok[:, None], Wq, 0)
    wm = Wf.mean(1); ym = Yq.mean(1)
    sxx = ((Wf - wm[:, None]) ** 2).sum(1); sxy = ((Wf - wm[:, None]) * (Yq - ym[:, None])).sum(1)
    syy = ((Yq - ym[:, None]) ** 2).sum(1)
    d = sxy / np.maximum(sxx, 1e-9); a = ym - d * wm
    fit = sxy ** 2 / np.maximum(sxx * syy, 1e-9)
    good = ok & (sxx > 0.05) & (np.abs(d) > 32) & (fit > min_fit)
    if sel is not None:
        good &= sel
    ANCH = "--anchor" in sys.argv
    if not ANCH:
        idx = np.where(good)[0]
    W = np.where(L < 4, np.array([1 / 3, 2 / 3, 0, 1])[np.minimum(L, 3)], np.nan)
    Eu = np.zeros((256, 4)); n = np.zeros(256)
    for q, (r, c) in enumerate(QPOS):
        if 8 + q not in lanes:
            continue
        if ANCH:   # (a, d) from the uniform quadrants among the other three only (exact weights)
            oth = [k for k in range(4) if k != q]
            Wo, Yo = W[:, oth], Yq[:, oth]
            okk = ~np.isnan(Wo)
            nn = okk.sum(1)
            wmo = np.where(okk, Wo, 0).sum(1) / np.maximum(nn, 1)
            ymo = np.where(okk, Yo, 0).sum(1) / np.maximum(nn, 1)
            sx = (np.where(okk, Wo - wmo[:, None], 0) ** 2).sum(1)
            sy = np.where(okk, (Wo - wmo[:, None]) * (Yo - ymo[:, None]), 0).sum(1)
            d = sy / np.maximum(sx, 1e-9); a = ymo - d * wmo
            g = (sx > 1e-6) & (np.abs(d) > 32)
            if sel is not None:
                g &= sel
            idx = np.where(g)[0]
        sub = []
        for dy in (0, 1):
            for dx in (0, 1):
                rr = r + (1 if dy else -1); cc = c + (1 if dx else -1)
                sub.append(9 / 16 * Yr[idx, r, c] + 3 / 16 * Yr[idx, rr, c] + 3 / 16 * Yr[idx, r, cc]
                           + 1 / 16 * Yr[idx, rr, cc])
        u = (np.stack(sub, 1) - a[idx, None]) / d[idx, None]
        s = L[idx, q]
        for k in range(4):
            Eu[:, k] += np.bincount(s, weights=u[:, k], minlength=256)
        n += np.bincount(s, minlength=256)
    return Eu / np.maximum(n, 1)[:, None], n


def assign(Eu, n):
    V = CW[PATS]                                    # 256 patterns x 4 weights
    cost = n[:, None] * (-2 * Eu @ V.T + (V * V).sum(1)[None, :])
    table = -np.ones(256, int)
    uni = {s: (s ^ 2) * 0x55 for s in range(4)}
    for s, p in uni.items():
        table[s] = p
    rs = [s for s in range(256) if s not in uni]
    cs = [p for p in range(256) if p not in uni.values()]
    C = cost[np.ix_(rs, cs)]
    r, c = linear_sum_assignment(C)
    for i, j in zip(r, c):
        table[rs[i]] = cs[j]
    # margin: best alternative pattern cost - assigned cost (per occurrence)
    return table, cost


def show(table, n):
    for s in np.argsort(-n)[:64]:
        p = PATS[table[s]]
        print("sym %3d n=%6d codes TL TR / BL BR = %d %d / %d %d" % (s, n[s], *p))


def main():
    lanes = [int(x) for x in sys.argv[sys.argv.index("--lanes") + 1].split(",")] if "--lanes" in sys.argv else [8, 9, 10, 11]
    if "--split" in sys.argv:   # stability: two halves of the blocks and per-lane runs
        z = np.load(os.path.join(SCR, "quadring_gfx10.npz"))
        B = z["syms"].shape[1]
        h = np.arange(B) < B // 2
        t1, _ = assign(*stats(lanes, h)); t2, _ = assign(*stats(lanes, ~h))
        Eu, n = stats(lanes)
        for thr in (1000, 300, 100, 30):
            m = n >= thr
            print("symbols n>=%d: %d, halves agree on %d" % (thr, m.sum(), (t1[m] == t2[m]).sum()))
        tl = [assign(*stats([l]))[0] for l in (8, 9, 10, 11)]
        for thr in (1000, 300):
            m = n >= thr
            agree = sum(((tl[0] == t) & m).sum() for t in tl[1:])
            print("n>=%d per-lane tables vs lane 8: %d / %d" % (thr, agree, 3 * m.sum()))
        return
    Eu, n = stats(lanes)
    table, cost = assign(Eu, n)
    np.save(os.path.join(SCR, "table.npy"), np.stack([table, n.astype(int)]))
    show(table, n)


if __name__ == "__main__":
    main()
