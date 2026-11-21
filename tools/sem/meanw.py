"""Pooled per-symbol mean weight (anchor method, mean of clipped estimates) and a test of
generative hypotheses for the quadrant symbol -> 4 codes mapping.

Every hypothesis maps symbol s -> four BC1 codes; its mean weight is compared with the
measured one (weighted correlation over symbols with n >= 30, weights = n).
usage: venv/bin/python -I tools/sem/meanw.py [quad_gfx10.npz]
"""
import itertools
import os
import sys
import numpy as np

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")
BCW = np.array([0, 3, 1, 2]) / 3.0          # BC1 code -> weight (4-colour)


def measured(path, min_d=24.0):
    z = np.load(path)
    S = z["syms"].astype(np.int64)
    Y = z["rgb"].astype(float) @ np.array([.299, .587, .114])
    L = S[8:12].T
    W = np.where(L < 4, np.array([1 / 3, 2 / 3, 0, 1])[np.minimum(L, 3)], np.nan)
    sums = np.zeros(256); cnt = np.zeros(256)
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
        w = np.clip((Y[good, q] - a[good]) / d[good], -0.2, 1.2)
        s = L[good, q]
        sums += np.bincount(s, weights=w, minlength=256)
        cnt += np.bincount(s, minlength=256)
    return sums / np.maximum(cnt, 1), cnt


def hyps():
    digits = np.array([[(s >> (2 * k)) & 3 for k in range(4)] for s in range(256)])
    v2c = np.array([2, 3, 0, 1])                  # uniform symbol v -> BC1 code (found)
    rank = np.array([1, 2, 0, 3])                 # symbol digit -> weight rank (same thing)
    out = {}
    for order in ("lsb", "msb"):
        dg = digits if order == "lsb" else digits[:, ::-1]
        d0 = dg[:, 0]
        for op in ("xor", "add", "sub"):
            for space in ("code", "rank"):
                p = []
                for k in range(4):
                    dk = dg[:, k]
                    if k == 0:
                        v = d0
                    elif op == "xor":
                        v = d0 ^ dk
                    elif op == "add":
                        v = (d0 + dk) & 3
                    else:
                        v = (d0 - dk) & 3
                    p.append(v)
                p = np.stack(p, 1)
                if space == "code":
                    wts = BCW[v2c[p]]
                else:
                    wts = rank[p] / 3.0 if False else np.array([1, 2, 0, 3])[p] / 3.0
                out["first-pixel+%s %s %s" % (op, space, order)] = wts.mean(1)
        # plain packing with every code permutation (control: already ruled out by corr^2)
        for perm in itertools.permutations(range(4)):
            out["plain %s perm%s" % (order, perm)] = (np.array(perm)[dg] / 3.0).mean(1)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCR, "quad_gfx10.npz")
    mw, cnt = measured(path)
    np.save(os.path.join(SCR, "meanw.npy"), np.stack([mw, cnt]))
    m = cnt >= 30
    res = []
    for name, hw in hyps().items():
        w = cnt[m]
        x, y = hw[m], mw[m]
        xm, ym = (w * x).sum() / w.sum(), (w * y).sum() / w.sum()
        c = (w * (x - xm) * (y - ym)).sum() / np.sqrt((w * (x - xm) ** 2).sum() * (w * (y - ym) ** 2).sum())
        res.append((c, name))
    res.sort(reverse=True)
    for c, n in res[:12]:
        print("%.3f  %s" % (c, n))
    print("symbols measured:", int(m.sum()))


if __name__ == "__main__":
    main()
