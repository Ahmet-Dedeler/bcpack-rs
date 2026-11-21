"""Fit the 256-entry quadrant codebook (a bijection symbol -> 2x2 code pattern) with exact
endpoints, by iterated conditional assignment.

Data: tools/sem/fullres.py (blocks with known endpoints, four-colour only).
Per quadrant occurrence and candidate pattern P (codes TL TR BL BR) the cost is
  mip1:   |mean(pal[P]) - mip1 pixel|^2 / (2 * sm^2)                       (sm = 6)
  edges:  sum over the 8 pixel pairs that cross the quadrant boundary of
          log(1 + |pal[P_k] - neighbour colour|^2 / sg^2)                  (sg = 10)
          (neighbour colours from the current table; pairs into unknown blocks skipped)
  intra:  the same robust cost for the 4 pixel pairs inside the quadrant
Costs are summed per symbol into a 256 x 256 matrix and a min-cost bijection is solved
(symbols 0-3 fixed to the uniform patterns, code = sym ^ 2). Repeat until stable.
usage: venv/bin/python -I tools/sem/fit_rgb.py [--iters 6] [--init table.npy] [--limit N]
writes $SEM_SCRATCH/rgb_table.npy  ([table, per-symbol margin, per-symbol occurrences])
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import linear_sum_assignment

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")
PATS = np.array([[(p >> (2 * k)) & 3 for k in range(4)] for p in range(256)])
FIXED = {s: (s ^ 2) * 0x55 for s in range(4)}
SM, SG = 6.0, 10.0
T_CUR = None
USE_MIP1 = "--no-mip1" not in sys.argv


def store_split(files, parity):
    """Keep files whose texture's store has the given index parity in the sorted store list."""
    import json
    stores = {}
    for f in files:
        tid = str(np.load(f)["tid"])
        stores[f] = tid.rpartition(":")[0]
    names = sorted(set(stores.values()))
    rank = {n: i for i, n in enumerate(names)}
    return [f for f in files if rank[stores[f]] % 2 == parity]


def rho(d2):
    return np.log1p(d2 / (SG * SG))


def tex_cost(path):
    T = T_CUR
    z = np.load(path)
    qs = z["qsym"].astype(np.int64); pal = z["pal"]; K = z["known"] & z["four"]; m1 = z["mip1"]
    bh, bw = K.shape
    if not K.any():
        return np.zeros((256, 256)), np.zeros(256)
    # current full-res code image and colour image
    codes = PATS[T[qs]]                                     # 2bh x 2bw x 4
    img_codes = np.zeros((4 * bh, 4 * bw), np.int64)
    for k, (dy, dx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        img_codes[dy::2, dx::2] = codes[..., k]
    palpx = np.repeat(np.repeat(pal, 4, 0), 4, 1)           # 4bh x 4bw x 4 x 3
    img = np.take_along_axis(palpx, img_codes[..., None, None].repeat(3, -1), 2)[..., 0, :]
    kpx = np.repeat(np.repeat(K, 4, 0), 4, 1)
    # quadrant occurrences in known 4-colour blocks
    qy, qx = np.nonzero(np.repeat(np.repeat(K, 2, 0), 2, 1))
    by, bx = qy // 2, qx // 2
    P4 = pal[by, bx]                                        # n x 4 x 3
    n = len(qy)
    cost = np.zeros((n, 256))
    # mip1 term
    meanP = P4[:, PATS, :].mean(2)                          # n x 256 x 3
    if USE_MIP1:
        cost += ((meanP - m1[qy, qx][:, None, :]) ** 2).sum(-1) / (2 * SM * SM)
    del meanP
    H, Wd = 4 * bh, 4 * bw
    # boundary pairs: pixel k of the quadrant at (2qy+dy, 2qx+dx); outward neighbours
    for k, (dy, dx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        py, px = 2 * qy + dy, 2 * qx + dx
        A = np.zeros((n, 4))
        for ny, nx in [(py, px - 1 if dx == 0 else px + 1), (py - 1 if dy == 0 else py + 1, px)]:
            ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < Wd)
            nyc, nxc = np.clip(ny, 0, H - 1), np.clip(nx, 0, Wd - 1)
            ok &= kpx[nyc, nxc]
            nb = img[nyc, nxc]
            d2 = ((P4 - nb[:, None, :]) ** 2).sum(-1)       # n x 4 codes
            A += np.where(ok[:, None], rho(d2), 0)
        cost += A[:, PATS[:, k]]
    D = rho(((P4[:, :, None, :] - P4[:, None, :, :]) ** 2).sum(-1))   # n x 4 x 4
    for i, j in [(0, 1), (2, 3), (0, 2), (1, 3)]:
        cost += D[np.arange(n)[:, None], PATS[None, :, i], PATS[None, :, j]]
    s = qs[qy, qx]
    M = np.zeros((256, 256))
    np.add.at(M, s, cost)
    return M, np.bincount(s, minlength=256).astype(float)


def init_pool(T, use_mip1):
    global T_CUR, USE_MIP1
    T_CUR = T
    USE_MIP1 = use_mip1


def costs(T, files):
    with Pool(os.cpu_count(), initializer=init_pool, initargs=(T, USE_MIP1)) as p:
        M = np.zeros((256, 256)); N = np.zeros(256)
        for m, c in p.imap_unordered(tex_cost, files, chunksize=2):
            M += m; N += c
    return M, N


def assign(M):
    T = np.zeros(256, int)
    rs = [s for s in range(256) if s not in FIXED]
    cs = [p for p in range(256) if p not in FIXED.values()]
    r, c = linear_sum_assignment(M[np.ix_(rs, cs)])
    for s, p in FIXED.items():
        T[s] = p
    for i, j in zip(r, c):
        T[rs[i]] = cs[j]
    return T


def margin(M, T):
    """Per symbol: cost increase of the cheapest swap of its pattern with another symbol's
    (bijection-respecting; >= 0 at the assignment optimum). Divide by n for per occurrence."""
    own = M[np.arange(256), T]
    D = M[:, T] + M[:, T].T - own[:, None] - own[None, :]   # D[a, b]: swap a <-> b
    np.fill_diagonal(D, np.inf)
    for s in FIXED:
        D[s, :] = np.inf; D[:, s] = np.inf
    return D.min(1)


def main():
    iters = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 6
    files = sorted(glob.glob(os.path.join(SCR, "fullres", "*.npz")))
    if "--limit" in sys.argv:
        k = int(sys.argv[sys.argv.index("--limit") + 1]); files = files[::max(1, len(files) // k)][:k]
    if "--stores" in sys.argv:
        files = store_split(files, {"even": 0, "odd": 1}[sys.argv[sys.argv.index("--stores") + 1]])
    print("files", len(files), "mip1 term", USE_MIP1, flush=True)
    init = sys.argv[sys.argv.index("--init") + 1] if "--init" in sys.argv else os.path.join(SCR, "smooth_table.npy")
    if init == "random":
        T = np.zeros(256, int)
        for s_, p_ in FIXED.items():
            T[s_] = p_
        rest = [p for p in range(256) if p not in FIXED.values()]
        T[4:] = np.random.default_rng(1).permutation(rest)
    else:
        T = np.load(init)[0].astype(int)
    for it in range(iters):
        M, N = costs(T, files)
        T2 = assign(M)
        ch = int((T2 != T).sum())
        print("iter %d: total cost %.4g, symbols changed %d" % (it, M[np.arange(256), T2].sum(), ch), flush=True)
        T = T2
        if ch == 0:
            break
    mg = margin(M, T)
    outp = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(SCR, "rgb_table.npy")
    np.save(outp, np.stack([T, mg, N]))
    names = "01ab"
    for s in np.argsort(-N)[:256]:
        p = PATS[T[s]]
        print("sym %3d n %8d  %s%s/%s%s  margin/occ %.3f" % (s, N[s], *[names[x] for x in p], mg[s] / max(N[s], 1)))


if __name__ == "__main__":
    main()
