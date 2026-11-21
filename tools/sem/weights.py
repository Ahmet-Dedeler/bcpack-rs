"""Score index-value -> palette-weight hypotheses against mip-1 quadrant luminance.

Each lane 8-11 symbol covers one 2x2 quadrant = one mip-1 pixel. Whatever the pixel order
inside the quadrant, the quadrant's mean palette weight only depends on the multiset of its
four 2-bit codes and on the code -> weight map. Per block we correlate the 4 quadrant mean
weights with the 4 mip-1 luminances (corr^2; endpoint order/sign drops out) and average
over blocks whose mip-1 2x2 spread is > 8.

usage: python3 -I tools/sem/weights.py [quad_gfx10.npz]
"""
import itertools
import os
import sys
import numpy as np

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")


def load(path):
    z = np.load(path)
    S = z["syms"].astype(np.int64)
    Y = z["rgb"].astype(float) @ np.array([.299, .587, .114])
    m = (Y.max(1) - Y.min(1)) > 8
    return S[:, m], Y[m]


def r2(M, Yb):
    a = M - M.mean(1, keepdims=True)
    b = Yb - Yb.mean(1, keepdims=True)
    num = (a * b).sum(1) ** 2
    den = (a * a).sum(1) * (b * b).sum(1)
    ok = den > 1e-9
    return float((num[ok] / den[ok]).mean()), float(ok.mean())


def main():
    S, Yb = load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCR, "quad_gfx10.npz"))
    L = S[8:12].T
    fields = np.stack([(L >> (2 * k)) & 3 for k in range(4)], -1)
    res = []
    for perm in itertools.permutations([0, 1, 2, 3]):
        w = np.array(perm) / 3.
        res.append((r2(w[fields].mean(-1), Yb), "code->weight*3 %s" % (perm,)))
    res.sort(reverse=True)
    for r in res:
        print("%.4f (defined %.3f)  %s" % (r[0][0], r[0][1], r[1]))
    print("%.4f  symbol value itself as weight" % r2(L.astype(float), Yb)[0])
    print("%.4f  random weight per symbol (control)" % r2(np.random.default_rng(0).random(256)[L], Yb)[0])


if __name__ == "__main__":
    main()
