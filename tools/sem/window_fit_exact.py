"""Exact per-channel fit of the fixed preset back-reference window (all 256 slots).

    python -I tools/sem/window_fit_exact.py --side 4096 --out /tmp/bcpack-shared/window_table.json

Needs the solved index codebook (sem-indices `quad_table.json`: lane 8-11 symbol -> the 4 BC1
codes of a 2x2 pixel group TL, TR, BL, BR).

Model.  A block rooted at slot s has endpoint field c = (v_c + delta_c) mod 2^w, with delta the
known residual sum along its reference chain.  With exact indices, each decoded pixel channel is
a fixed mix of that channel's c0 and c1 only (BC1 palette, 4-colour if c0 > c1 as 16-bit words,
else 3-colour with code 3 = black).  The mean of a 2x2 pixel group is compared with the mip-1
pixel it maps to.  So the cost decouples per channel: R depends on (R0, R1) only, G on (G0, G1),
B on (B0, B1), and each pair is searched exhaustively (32x32, 64x64, 32x32) for the exact optimum.
The 4/3-colour mode couples the channels, so it is iterated: start with 4-colour, fit, recompute
each block's mode from the fitted words, refit (until no change, max 4 rounds).

Per slot output: value, support (textures, blocks), cost (RMS of capped per-channel error over
2x2 groups, 8-bit units), margin (per channel: cost increase of the best value outside +-1 of the
optimum, as a fraction), split-half agreement (by texture), and the fraction of textures whose own
exact optimum equals the joint value (+-1).  Derived decoder constants, no game bytes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import endpoints as ep  # noqa: E402
import mipscore  # noqa: E402

QUAD_TABLE = os.environ.get("BCPACK_QUAD_TABLE",
                            "/Users/ahmet/Code/bcpack-wt/sem-indices/tools/sem/quad_table.json")
MOD = [32, 64, 32, 32, 64, 32]
E5 = np.array([(v << 3) | (v >> 2) for v in range(32)])
E6 = np.array([(v << 2) | (v >> 4) for v in range(64)])
CAP = 48.0


def quad_codes():
    """sym -> (4 groups? no: 4 pixels) BC1 codes, as an int array [256, 4] (pixel order TL TR BL BR)."""
    with open(QUAD_TABLE) as f:
        d = json.load(f)
    t = np.zeros((256, 4), np.int64)
    for e in d["entries"]:
        t[e["sym"]] = e["codes"]
    return t


def quad_targets(texture_id):
    """Per stored block: mip-1 pixel (RGB) for each of the 4 quadrants (TL TR BL BR), + ok mask."""
    os.makedirs(os.path.join(common.SCR, "qtgt"), exist_ok=True)
    fn = os.path.join(common.SCR, "qtgt", texture_id.replace("/", "_") + ".npz")
    if os.path.exists(fn):
        z = np.load(fn)
        return z["q"], z["ok"]
    tgt, ok, xy = common.pair_targets(texture_id)
    rows, w1, h1 = mipscore.mip1_rgb(texture_id)
    m1 = np.array(rows, dtype=np.float64)
    n = len(ok)
    q = np.zeros((n, 4, 3))
    for i in np.where(ok)[0]:
        x, y = xy[i]
        q[i] = m1[2 * y:2 * y + 2, 2 * x:2 * x + 2].reshape(4, 3)
    np.savez(fn, q=q, ok=ok)
    return q, ok


def load(p):
    """Worker: decode one pair; return rooted blocks only: slot, delta (n,6), code counts (n,4,4), targets (n,4,3)."""
    tid = p["texture_id"]
    try:
        s = common.decode_chunk(p["mip0_chunks"][0][0])
        q, ok = quad_targets(tid)
    except Exception as e:
        return tid, None, str(e)
    from window_fit import chains
    n = min(s.shape[1], len(ok))
    root, S = chains(s[:, :n])
    m = (root >= 0) & ok[:n]
    if not m.any():
        return tid, (np.zeros(0, int), np.zeros((0, 6), int), np.zeros((0, 4, 4)), np.zeros((0, 4, 3))), None
    QT = quad_codes()
    codes = np.stack([QT[s[8 + g, :n][m]] for g in range(4)], 1)          # (k, 4 groups, 4 pixels)
    cnt = np.zeros((codes.shape[0], 4, 4))
    for c in range(4):
        cnt[:, :, c] = (codes == c).sum(2)
    return tid, (root[m], S[m], cnt, q[:n][m]), None


def chan_cost(ch, d0, d1, cnt, t, mode4, cap=CAP):
    """Cost matrix [M, M] for channel ch (0 R, 1 G, 2 B) over candidate (v0, v1).
    d0, d1: (k,) deltas; cnt: (k, 4, 4) code counts per group; t: (k, 4) targets; mode4: (k,) bool."""
    M = MOD[ch]
    E = E6 if M == 64 else E5
    v = np.arange(M)
    A = E[(v[:, None] + d0[None, :]) % M]          # (M, k) expanded c0 per candidate v0
    B = E[(v[:, None] + d1[None, :]) % M]          # (M, k) expanded c1 per candidate v1
    out = np.zeros((M, M))
    n0, n1, n2, n3 = cnt[..., 0], cnt[..., 1], cnt[..., 2], cnt[..., 3]  # (k, 4)
    for i in range(M):
        a = A[i][None, :]                           # (1, k)
        p2 = np.where(mode4, (2 * a + B + 1) // 3, (a + B) // 2)      # (M, k)
        p3 = np.where(mode4, (a + 2 * B + 1) // 3, 0)
        val = (n0[None] * a[..., None] + n1[None] * B[..., None] + n2[None] * p2[..., None]
               + n3[None] * p3[..., None]) / 4.0     # (M, k, 4)
        e = np.minimum((val - t[None]) ** 2, cap * cap)
        out[i] = e.sum((1, 2))
    return out


def best_pair(C):
    i, j = np.unravel_index(np.argmin(C), C.shape)
    M = C.shape[0]
    far = C.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            far[(i + di) % M, (j + dj) % M] = np.inf
    margin = (far.min() - C[i, j]) / max(C[i, j], 1e-9)
    return int(i), int(j), float(margin)


def words(V, delta):
    E = (np.asarray(V) + delta) % np.array(MOD)
    c0 = (E[:, 0] << 11) | (E[:, 1] << 5) | E[:, 2]
    c1 = (E[:, 3] << 11) | (E[:, 4] << 5) | E[:, 5]
    return c0, c1


def fit_blocks(delta, cnt, tgt, rounds=4):
    mode4 = np.ones(len(delta), bool)
    V = [0] * 6
    margins = [0.0] * 3
    total = 0.0
    for _ in range(rounds):
        total = 0.0
        for ch in range(3):
            C = chan_cost(ch, delta[:, ch], delta[:, ch + 3], cnt, tgt[:, :, ch], mode4)
            i, j, mg = best_pair(C)
            V[ch], V[ch + 3] = i, j
            margins[ch] = mg
            total += C[i, j]
        c0, c1 = words(V, delta)
        new = c0 > c1
        if (new == mode4).all():
            break
        mode4 = new
    rms = float(np.sqrt(total / (len(delta) * 4 * 3))) if len(delta) else float("nan")
    return V, rms, margins


def circ(a, b):
    return max(min((x - y) % m, (y - x) % m) for x, y, m in zip(a, b, MOD))


def fit_slot(args):
    sl, items, max_blocks = args
    rng = np.random.default_rng(sl)
    D = np.concatenate([x[1] for x in items]); Cn = np.concatenate([x[2] for x in items])
    T = np.concatenate([x[3] for x in items])
    nb = len(D)
    if nb > max_blocks:
        k = rng.choice(nb, max_blocks, replace=False)
        D, Cn, T = D[k], Cn[k], T[k]
    V, rms, margins = fit_blocks(D, Cn, T)
    rec = dict(value=V, rms=round(rms, 2), margin=[round(m, 3) for m in margins],
               textures=len(items), blocks=int(nb))
    if len(items) >= 2:
        idx = rng.permutation(len(items))
        halves = []
        for part in (idx[::2], idx[1::2]):
            sub = [items[i] for i in part]
            d = np.concatenate([x[1] for x in sub]); c = np.concatenate([x[2] for x in sub])
            t = np.concatenate([x[3] for x in sub])
            if len(d) > max_blocks // 2:
                kk = rng.choice(len(d), max_blocks // 2, replace=False)
                d, c, t = d[kk], c[kk], t[kk]
            halves.append(fit_blocks(d, c, t)[0])
        rec["half_a"], rec["half_b"] = halves
        rec["halves_agree"] = circ(halves[0], V) <= 1 and circ(halves[1], V) <= 1
    own = 0
    for tid, d, c, t in items[:40]:
        if circ(fit_blocks(d, c, t, rounds=2)[0], V) <= 1:
            own += 1
    rec["own_agree"] = [own, min(len(items), 40)]
    return sl, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=4096)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-blocks", type=int, default=6000)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="/tmp/bcpack-shared/window_table.json")
    a = ap.parse_args()
    ps = [p for p in common.pairs() if max(p["w"], p["h"]) <= a.side]
    ps.sort(key=lambda p: -p["w"] * p["h"])
    if a.n:
        ps = ps[:a.n]
    from collections import defaultdict
    by = defaultdict(list)
    nfail = 0
    with Pool(a.jobs) as pool:
        for tid, d, err in pool.imap_unordered(load, ps, chunksize=1):
            if d is None:
                nfail += 1
                continue
            root, S, cnt, q = d
            for sl in np.unique(root).tolist():
                m = root == sl
                by[sl].append((tid, S[m], cnt[m], q[m]))
    print("pairs %d failed %d slots with data %d" % (len(ps), nfail, len(by)), flush=True)
    jobs = sorted(((sl, by[sl], a.max_blocks) for sl in by), key=lambda j: -sum(len(x[1]) for x in j[1]))
    out = {}
    with Pool(a.jobs) as pool:
        for sl, rec in pool.imap_unordered(fit_slot, jobs, chunksize=1):
            out[sl] = rec
            print("slot %3d value %-26s rms %5.2f margin %-22s tex %4d blocks %7d halves %s own %s" % (
                sl, rec["value"], rec["rms"], rec["margin"], rec["textures"], rec["blocks"],
                rec.get("halves_agree"), rec["own_agree"]), flush=True)
    doc = {"format": "BCPack BC1 preset back-reference window: slot -> endpoint fields "
                     "[R0, G0, B0, R1, G1, B1] read by a block whose lane-0 reference lands before the chunk "
                     "start (slot = (i - d) mod 256). Fitted by tools/sem/window_fit_exact.py; derived constants.",
           "slots": {str(k): out[k] for k in sorted(out)}}
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
    print("wrote", a.out, len(out))


if __name__ == "__main__":
    main()
