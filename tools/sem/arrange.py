"""Estimate the 2x2 pixel arrangement behind each quadrant symbol.

Per block, fit Y = a + d*w over the 4 quadrants using the measured per-symbol mean weights
(meanw.npy from meanw.py), keep good fits, then bilinearly upsample the 4x4 mip-1 window
to the 4 mip-0 pixel positions of each quadrant, map to weight space with (a, d), and
average the within-quadrant deviation per (lane, symbol).  Pixel order in the output:
p0 = top-left, p1 = top-right, p2 = bottom-left, p3 = bottom-right of the quadrant.
usage: venv/bin/python -I tools/sem/arrange.py [--pooled] [--top 80]
"""
import os
import sys
import numpy as np

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")
QPOS = [(1, 1), (1, 2), (2, 1), (2, 2)]


def profiles(min_fit=0.95):
    z = np.load(os.path.join(SCR, "quadring_gfx10.npz"))
    S = z["syms"].astype(np.int64)
    Yr = z["rgb"].astype(np.float32) @ np.array([.299, .587, .114], np.float32)   # B x 4 x 4
    mw, cnt = np.load(os.path.join(SCR, "meanw.npy"))
    mw = np.where(cnt >= 30, mw, np.nan)
    L = S[8:12].T
    Wq = mw[L]                                                  # B x 4
    Yq = np.stack([Yr[:, r, c] for r, c in QPOS], 1)
    ok = ~np.isnan(Wq).any(1)
    wm = np.nanmean(Wq, 1); ym = Yq.mean(1)
    sxx = ((Wq - wm[:, None]) ** 2).sum(1); sxy = ((Wq - wm[:, None]) * (Yq - ym[:, None])).sum(1)
    syy = ((Yq - ym[:, None]) ** 2).sum(1)
    d = sxy / np.maximum(sxx, 1e-9); a = ym - d * wm
    fit = sxy ** 2 / np.maximum(sxx * syy, 1e-9)
    good = ok & (sxx > 0.05) & (np.abs(d) > 32) & (fit > min_fit)
    acc = np.zeros((4, 256, 4)); n = np.zeros((4, 256))
    idx = np.where(good)[0]
    for q, (r, c) in enumerate(QPOS):
        sub = []
        for dy in (0, 1):
            for dx in (0, 1):
                rr = r + (1 if dy else -1); cc = c + (1 if dx else -1)
                sub.append(9 / 16 * Yr[idx, r, c] + 3 / 16 * Yr[idx, rr, c] + 3 / 16 * Yr[idx, r, cc]
                           + 1 / 16 * Yr[idx, rr, cc])
        sub = np.stack(sub, 1)
        wp = (sub - a[idx, None]) / d[idx, None]
        dev = wp - wp.mean(1, keepdims=True)
        s = L[idx, q]
        for k in range(4):
            acc[q, :, k] += np.bincount(s, weights=dev[:, k], minlength=256)
        n[q] += np.bincount(s, minlength=256)
    return acc / np.maximum(n, 1)[..., None], n, mw, int(good.sum())


def main():
    prof, n, mw, ng = profiles()
    np.save(os.path.join(SCR, "arrange.npy"), prof)
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 80
    print("good blocks", ng)
    tot = n.sum(0)
    for s in np.argsort(-tot)[:top]:
        line = "sym %3d mw=%5.2f " % (s, mw[s] * 12)
        for q in range(4):
            p = prof[q, s] * 100
            line += " | L%d n=%5d [%+4.0f %+4.0f / %+4.0f %+4.0f]" % (8 + q, n[q, s], p[0], p[1], p[2], p[3])
        print(line)


if __name__ == "__main__":
    main()
