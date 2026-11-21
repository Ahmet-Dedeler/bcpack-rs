"""Per-symbol mean palette weight from uniform 'anchor' quadrants in the same block.

Found: quadrant symbols 0..3 are uniform quadrants with palette weights
  sym 2 -> 0 (c0), sym 0 -> 1/3, sym 1 -> 2/3, sym 3 -> 1 (c1)
i.e. BC1 index = sym ^ 2 (ratio test: blocks holding uniform 2,0,3 give
(Y0-Y2)/(Y3-Y2) = 0.333, blocks holding 2,1,3 give 0.667).
For every block whose other quadrants contain >= 2 distinct anchor weights, fit
Y = a + d*w on the anchors and read w(s) = (Y_s - a)/d for the remaining quadrant.
Prints per lane, per symbol: n, median w, IQR, and the nearest k/12.
usage: venv/bin/python -I tools/sem/anchor.py [quad_gfx10.npz] [--lane 8] [--top 60]
"""
import os
import sys
import numpy as np

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")
AW = {0: 1 / 3, 1: 2 / 3, 2: 0.0, 3: 1.0}


def estimate(path, min_d=24.0):
    z = np.load(path)
    S = z["syms"].astype(np.int64)
    Y = z["rgb"].astype(float) @ np.array([.299, .587, .114])
    L = S[8:12].T
    isA = L < 4
    W = np.where(isA, np.array([1 / 3, 2 / 3, 0, 1])[np.minimum(L, 3)], np.nan)
    out = {q: {} for q in range(4)}
    for q in range(4):
        others = [r for r in range(4) if r != q]
        Wo, Yo = W[:, others], Y[:, others]
        ok = ~np.isnan(Wo)
        n = ok.sum(1)
        wm = np.where(ok, Wo, 0).sum(1) / np.maximum(n, 1)
        ym = np.where(ok, Yo, 0).sum(1) / np.maximum(n, 1)
        sxx = (np.where(ok, Wo - wm[:, None], 0) ** 2).sum(1)
        sxy = (np.where(ok, (Wo - wm[:, None]) * (Yo - ym[:, None]), 0)).sum(1)
        d = sxy / np.maximum(sxx, 1e-9)
        a = ym - d * wm
        good = (sxx > 1e-6) & (np.abs(d) > min_d)
        w = (Y[good, q] - a[good]) / d[good]
        s = L[good, q]
        for sym in np.unique(s):
            v = w[s == sym]
            out[q][int(sym)] = (len(v), float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)))
    return out


def main():
    args = [x for x in sys.argv[1:] if x.endswith(".npz")]
    path = args[0] if args else os.path.join(SCR, "quad_gfx10.npz")
    lanes = [int(sys.argv[sys.argv.index("--lane") + 1]) - 8] if "--lane" in sys.argv else range(4)
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 64
    out = estimate(path)
    np.save(os.path.join(SCR, "anchor_w.npy"), out, allow_pickle=True)
    for q in lanes:
        print("lane", 8 + q)
        for sym in range(top):
            if sym in out[q]:
                n, med, lo, hi = out[q][sym]
                print("  sym %3d n=%6d w=%.3f [%.2f,%.2f] ~%2d/12" % (sym, n, med, lo, hi, round(med * 12)))


if __name__ == "__main__":
    main()
