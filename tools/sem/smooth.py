"""Solve the 256-entry quadrant codebook as a bijection by local search.

Objective (lower is better) for a table T: symbol -> pattern (BC1 codes TL, TR, BL, BR):
  smooth  sum over adjacent quadrant pairs inside a block (left|right, top/bottom; counts from
          all decoded blocks) of -PMI(code_a, code_b) for the two pixel pairs that touch across
          the border. PMI comes from plain BC1 controls (cross-quadrant pixel pairs, c0 > c1).
  mean    kappa * n_s * max(0, |12*mean(T[s]) - 12*mw[s]| - tol)^2   (mw: anchor mean weights
          measured against mip 1, meanw.py; down-weighted for rarely measured symbols)
Symbols 0-3 are fixed (uniform, code = sym ^ 2). Search: start from a Hungarian assignment on
the mean term + smoothness unary, then pairwise swaps until no swap improves.
Writes $SEM_SCRATCH/smooth_table.npy and prints per-symbol swap margins.
usage: venv/bin/python -I tools/sem/smooth.py [--kappa 4] [--tol 0.4] [--seed 0]
"""
import os
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")
CW = np.array([0, 1, 1 / 3, 2 / 3])
PATS = np.array([[(p >> (2 * k)) & 3 for k in range(4)] for p in range(256)])


def arg(name, default, typ=float):
    return typ(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def cooc():
    path = os.path.join(SCR, "cooc.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["H"], z["V"], z["F"]
    S = np.load(os.path.join(SCR, "quad_gfx10.npz"))["syms"].astype(np.int64)[8:12]
    H = (np.bincount(S[0] * 256 + S[1], minlength=65536) + np.bincount(S[2] * 256 + S[3], minlength=65536)).reshape(256, 256)
    V = (np.bincount(S[0] * 256 + S[2], minlength=65536) + np.bincount(S[1] * 256 + S[3], minlength=65536)).reshape(256, 256)
    F = np.bincount(S.ravel(), minlength=256)
    np.savez(path, H=H, V=V, F=F)
    return H, V, F


def pair_costs():
    J = np.load(os.path.join(SCR, "pairJ_cross.npy"))
    m = J.sum(1)
    C = -np.log(J / np.outer(m, m))                        # -PMI(a, b)
    P = PATS
    # horizontal: left pattern's right column (TR, BR) vs right pattern's left column (TL, BL)
    Ch = C[P[:, 1][:, None], P[:, 0][None, :]] + C[P[:, 3][:, None], P[:, 2][None, :]]
    # vertical: top pattern's bottom row (BL, BR) vs bottom pattern's top row (TL, TR)
    Cv = C[P[:, 2][:, None], P[:, 0][None, :]] + C[P[:, 3][:, None], P[:, 1][None, :]]
    return Ch, Cv


def objective_parts(T, H, V, Ch, Cv, U):
    sm = (H * Ch[T][:, T]).sum() + (V * Cv[T][:, T]).sum()
    un = U[np.arange(256), T].sum()
    return sm, un


def total(T, H, V, Ch, Cv, U):
    sm, un = objective_parts(T, H, V, Ch, Cv, U)
    return sm + un


def search(T, H, V, Ch, Cv, U, fixed, max_iter=20000, verbose=True):
    """Best-improvement pairwise swaps. Linearised swap deltas from the 'row' matrix
    R[s, p] (cost of symbol s holding pattern p against everybody else), candidates verified
    with the exact objective."""
    T = T.copy()
    free = np.array([s for s in range(256) if s not in fixed])
    cur = total(T, H, V, Ch, Cv, U)
    for it in range(max_iter):
        R = H @ Ch[:, T].T + H.T @ Ch[T, :] + V @ Cv[:, T].T + V.T @ Cv[T, :] + U
        Rf = R[np.ix_(free, T[free])]                       # Rf[i, j] = R[free_i, T[free_j]]
        dg = np.diag(Rf)
        D = Rf + Rf.T - dg[:, None] - dg[None, :]
        np.fill_diagonal(D, np.inf)
        order = np.argsort(D, axis=None)[:200]
        done = False
        for k in order:
            i, j = divmod(int(k), len(free))
            if D[i, j] >= 0:
                break
            a, b = free[i], free[j]
            T2 = T.copy(); T2[a], T2[b] = T[b], T[a]
            c2 = total(T2, H, V, Ch, Cv, U)
            if c2 < cur - 1e-6:
                T, cur, done = T2, c2, True
                break
        if verbose and (it % 50 == 0 or not done):
            sm, un = objective_parts(T, H, V, Ch, Cv, U)
            print("iter %d total %.0f smooth %.0f unary %.0f" % (it, cur, sm, un), flush=True)
        if not done:
            break
    return T


def margins(T, H, V, Ch, Cv, U, fixed):
    """Per symbol: cost increase of the best swap involving it (>0 at a local optimum)."""
    free = np.array([s for s in range(256) if s not in fixed])
    R = H @ Ch[:, T].T + H.T @ Ch[T, :] + V @ Cv[:, T].T + V.T @ Cv[T, :] + U
    Rf = R[np.ix_(free, T[free])]
    dg = np.diag(Rf)
    D = Rf + Rf.T - dg[:, None] - dg[None, :]
    np.fill_diagonal(D, np.inf)
    m = np.full(256, np.inf)
    m[free] = D.min(1)
    return m


def main():
    kappa, tol = arg("--kappa", 4.0), arg("--tol", 0.4)
    H, V, F = cooc()
    Ch, Cv = pair_costs()
    mw, cnt = np.load(os.path.join(SCR, "meanw.npy"))
    pm12 = CW[PATS].mean(1) * 12
    conf = np.minimum(1.0, cnt / 100.0)
    ex = np.maximum(0, np.abs(pm12[None, :] - 12 * mw[:, None]) - tol)
    U = kappa * (F * conf)[:, None] * ex ** 2                     # 256 symbols x 256 patterns
    fixed = {s: (s ^ 2) * 0x55 for s in range(4)}
    # init: Hungarian on unary + "smooth unary" (pattern's cost against the average neighbour)
    rs = [s for s in range(256) if s not in fixed]
    cs = [p for p in range(256) if p not in fixed.values()]
    r, c = linear_sum_assignment(U[np.ix_(rs, cs)])
    T = np.zeros(256, int)
    for s, p in fixed.items():
        T[s] = p
    for i, j in zip(r, c):
        T[rs[i]] = cs[j]
    init = sys.argv[sys.argv.index("--init") + 1] if "--init" in sys.argv else None
    if init:
        T = np.load(init)[0] if init.endswith(".npy") else T
    T = search(T, H, V, Ch, Cv, U, fixed)
    np.save(os.path.join(SCR, "smooth_table.npy"), np.stack([T, F]))
    mg = margins(T, H, V, Ch, Cv, U, fixed)
    names = "01ab"
    for s in np.argsort(-F)[:80]:
        p = PATS[T[s]]
        print("sym %3d freq %8d mw12 %5.2f  %s%s/%s%s  mean12 %2d  margin/occ %.3f" % (
            s, F[s], mw[s] * 12, *[names[x] for x in p], round(pm12[T[s]]), mg[s] / max(F[s], 1)))


if __name__ == "__main__":
    main()
