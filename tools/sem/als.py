"""Learn one scalar 'mean palette weight' per quadrant symbol by alternating least squares.

Model per block b and quadrant q: Y[b,q] = a_b + d_b * w[lane, s_bq] (mip-1 luminance).
If lanes 8-11 carry a fixed code (codebook / bijection of 2x2 patterns), a per-symbol weight
explains the data far better than chance; if the meaning depends on context it won't.
usage: venv/bin/python -I tools/sem/als.py [quad_gfx10.npz] [--shared]
"""
import os
import sys
import numpy as np

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weights import load, r2  # noqa: E402


def als(L, Yb, shared, iters=30, seed=0, init=None):
    rng = np.random.default_rng(seed)
    nl = 1 if shared else 4
    key = L if shared else L + 256 * np.arange(4)[None, :]
    w = rng.standard_normal(256 * nl) if init is None else init.copy()
    for _ in range(iters):
        M = w[key]
        mc = M - M.mean(1, keepdims=True)
        yc = Yb - Yb.mean(1, keepdims=True)
        den = (mc * mc).sum(1)
        d = np.where(den > 1e-9, (mc * yc).sum(1) / np.maximum(den, 1e-9), 0.0)
        a = Yb.mean(1) - d * M.mean(1)
        # update w: minimise sum (Y - a - d w)^2
        num = np.bincount(key.ravel(), weights=(d[:, None] * (Yb - a[:, None])).ravel(), minlength=w.size)
        dd = np.bincount(key.ravel(), weights=np.repeat(d * d, 4), minlength=w.size)
        w = np.where(dd > 1e-9, num / np.maximum(dd, 1e-9), w)
        w = (w - w.mean()) / (w.std() + 1e-12)
    return w, r2(w[key], Yb)[0]


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    S, Yb = load(args[0] if args else os.path.join(SCR, "quad_gfx10.npz"))
    L = S[8:12].T
    n = len(L)
    tr, te = slice(0, n // 2), slice(n // 2, n)
    shared = "--shared" in sys.argv
    w, s_tr = als(L[tr], Yb[tr], shared)
    key = L[te] if shared else L[te] + 256 * np.arange(4)[None, :]
    print("train r2 %.4f  held-out r2 %.4f" % (s_tr, r2(w[key], Yb[te])[0]))
    np.save(os.path.join(SCR, "als_w%s.npy" % ("_shared" if shared else "")), w)
    order = np.argsort(w[:256])
    print("lane 8 symbols sorted by learned weight:", order.tolist())


if __name__ == "__main__":
    main()
