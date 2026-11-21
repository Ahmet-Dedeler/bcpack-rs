"""Codebook readout v2: multiset from the anchor mean weight (twelfths), arrangement from
the baseline-corrected upsampled mip-1 profile.

1. mean weight per symbol m (meanw.py, anchor method, mean of clipped estimates); keep
   symbols with n >= 300 and |12m - round(12m)| < 0.3.
2. per lane, profile dev (arrange.py) minus the uniform baseline b(w) = (1-w)*P[sym2] + w*P[sym3]
   of that lane; pool lanes (absolute frame, table shared by lanes).
3. among the 4-colour patterns whose mean is round(12m)/12, pick the one whose centred
   weights correlate best with the pooled corrected profile; report the margin to the
   runner-up, and duplicates.
Also: held-out corr^2 of the measured mean-weight table against mip-1 (vs ceiling 0.90).
usage: venv/bin/python -I tools/sem/table2.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arrange import profiles  # noqa: E402
from weights import load, r2  # noqa: E402

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")
CW = np.array([0, 1, 1 / 3, 2 / 3])
PATS = np.array([[(p >> (2 * k)) & 3 for k in range(4)] for p in range(256)])
NAME = {0: "0", 1: "1", 2: "a", 3: "b"}   # a = 2/3 c0 + 1/3 c1 (code 2), b = code 3


def main():
    mw, cnt = np.load(os.path.join(SCR, "meanw.npy"))
    S, Yb = load(os.path.join(SCR, "quad_gfx10.npz"))
    L = S[8:12].T
    h = len(L) // 2
    tab = np.where(cnt >= 30, mw, 0.5)
    print("held-out-free corr^2 with measured mean-weight table: %.4f (chance 0.456, true-index ceiling 0.90)"
          % r2(tab[L[h:]], Yb[h:])[0])
    prof, n, _, _ = profiles()
    corr = np.zeros((256, 4)); tot = np.zeros(256)
    for q in range(4):
        P2, P3 = prof[q, 2], prof[q, 3]
        for s in range(256):
            w = mw[s]
            corr[s] += n[q, s] * (prof[q, s] - ((1 - w) * P2 + w * P3))
            tot[s] += n[q, s]
    corr /= np.maximum(tot, 1)[:, None]
    V = CW[PATS]
    pm12 = np.round(V.mean(1) * 12).astype(int)
    out = {}
    for s in np.argsort(-cnt):
        if cnt[s] < 300:
            continue
        m12 = mw[s] * 12
        if s < 4:
            out[s] = ((s ^ 2) * 0x55, 99.0)
            continue
        if abs(m12 - round(m12)) > 0.3:
            continue
        cand = np.where(pm12 == round(m12))[0]
        cand = cand[[len(set(PATS[p])) > 1 for p in cand]]
        c = corr[s] - corr[s].mean()
        sc = []
        for p in cand:
            v = V[p] - V[p].mean()
            sc.append((v @ c) / (np.linalg.norm(v) * np.linalg.norm(c) + 1e-12))
        o = np.argsort(sc)[::-1]
        out[s] = (cand[o[0]], sc[o[0]] - (sc[o[1]] if len(o) > 1 else -1))
    used = {}
    for s, (p, mg) in out.items():
        used.setdefault(p, []).append(s)
    np.save(os.path.join(SCR, "table2.npy"), out, allow_pickle=True)
    print("symbol  n      12*mean  pattern(TL TR/BL BR)  margin  dup")
    for s, (p, mg) in out.items():
        a = PATS[p]
        print("%4d %7d  %6.2f    %s %s / %s %s      %.2f   %s" % (
            s, cnt[s], mw[s] * 12, *[NAME[x] for x in a], mg, used[p] if len(used[p]) > 1 else ""))
    print("assigned", len(out), "distinct patterns", len(used))


if __name__ == "__main__":
    main()
