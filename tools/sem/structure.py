"""Look for a generative rule in a fitted quadrant table.
- symbol order vs occurrence frequency (Spearman), vs frequency in plain BC1 controls
- pairs (2k, 2k+1): related by which symmetry (flip H/V, transpose, rotate 180, endpoint swap)?
usage: venv/bin/python -I tools/sem/structure.py TABLE.npy
"""
import os
import sys
import numpy as np

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")
N = "01ab"


def codes(p):
    return [(p >> (2 * k)) & 3 for k in range(4)]


def pat(c):
    return sum(v << (2 * k) for k, v in enumerate(c))


SYM = {
    "hflip": lambda c: [c[1], c[0], c[3], c[2]],
    "vflip": lambda c: [c[2], c[3], c[0], c[1]],
    "transpose": lambda c: [c[0], c[2], c[1], c[3]],
    "antitranspose": lambda c: [c[3], c[1], c[2], c[0]],
    "rot180": lambda c: [c[3], c[2], c[1], c[0]],
    "rot90": lambda c: [c[2], c[0], c[3], c[1]],
    "rot270": lambda c: [c[1], c[3], c[0], c[2]],
    "swap_ends": lambda c: [[1, 0, 3, 2][v] for v in c],   # c0<->c1 (and a<->b)
}


def rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


def main():
    T = np.load(sys.argv[1])[0].astype(int)
    F = np.load(os.path.join(SCR, "cooc.npz"))["F"].astype(float)
    s = np.arange(256)
    print("Spearman(symbol, -frequency in BCPack data): %.3f" % np.corrcoef(rank(s), rank(-F))[0, 1])
    b = np.load(os.path.join(SCR, "real", "blocks.npy"))
    idx = (b >> np.uint64(32)).astype(np.int64)
    pf = np.zeros(256)
    for oy, ox in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        q = 0
        for k, (dy, dx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            q = q | (((idx >> (2 * ((oy + dy) * 4 + ox + dx))) & 3) << (2 * k))
        pf += np.bincount(q, minlength=256)
    print("Spearman(symbol, -frequency of its pattern in plain BC1 controls): %.3f"
          % np.corrcoef(rank(s), rank(-pf[T]))[0, 1])
    from collections import Counter
    rel = Counter()
    for k in range(0, 256, 2):
        a, c = codes(T[k]), codes(T[k + 1])
        hit = [n for n, f in SYM.items() if pat(f(a)) == T[k + 1]]
        rel[tuple(hit) if hit else ("none",)] += 1
    print("pairs (2k, 2k+1) by symmetry:", dict(rel))
    # class of each symbol: sorted multiset
    cls = ["".join(sorted(N[v] for v in codes(T[x]))) for x in s]
    first = {}
    for x in s:
        first.setdefault(cls[x], x)
    print("first symbol of each code multiset (in order):",
          sorted(first.items(), key=lambda kv: kv[1])[:40])


if __name__ == "__main__":
    main()
