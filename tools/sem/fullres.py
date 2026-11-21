"""Build full-resolution fitting data from exact endpoints (sem-endpoints back-reference rule).

For each cached pair texture (cache.py): endpoints via the back-reference rule (lane 0 =
distance unzigzag(v) mod 256, lanes 2-7 zigzag residuals mod 2^w, d = 0 -> absolute).
Blocks whose reference chain reaches before block 0 are marked unknown (the preset window is
not solved yet). Output $SEM_SCRATCH/fullres/<n>.npz:
  qsym  int16 [2bh, 2bw]     quadrant symbol grid (lanes 8-11 placed TL TR / BL BR)
  pal   float32 [bh, bw, 4, 3]  BC1 palette (codes 0..3) in RGB 0-255
  known bool [bh, bw]; four bool [bh, bw] (c0 > c1); mip1 float32 [2bh, 2bw, 3]
usage: python3 -I tools/sem/fullres.py [--limit N]
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH = os.environ.get("BCPACK_BENCH", "/Users/ahmet/Code/bcpack-wt/harness/tools/bench")
sys.path.insert(0, BENCH)
import numpy as np  # noqa: E402
import mipscore  # noqa: E402
from quadref import placement  # noqa: E402

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")
W = [5, 6, 5, 5, 6, 5]


def back_distance(v):
    return (v >> 1) if not (v & 1) else 255 - (v >> 1)


def endpoints(syms):
    n = syms.shape[1]
    l0 = syms[0].tolist()
    lanes = [syms[2 + k].tolist() for k in range(6)]
    E = [None] * n
    known = [False] * n
    for i in range(n):
        d = back_distance(l0[i])
        if d == 0:
            P, kn = (0,) * 6, True
        elif d <= i:
            P, kn = E[i - d], known[i - d]
        else:
            P, kn = (0,) * 6, False
        e = []
        for k in range(6):
            v = lanes[k][i]
            r = -((v + 1) >> 1) if v & 1 else v >> 1
            e.append((P[k] + r) % (1 << W[k]))
        E[i] = tuple(e)
        known[i] = kn
    return np.array(E, np.int64), np.array(known)


def palette(E):
    def exp5(v): return (v << 3) | (v >> 2)
    def exp6(v): return (v << 2) | (v >> 4)
    c0 = np.stack([exp5(E[:, 0]), exp6(E[:, 1]), exp5(E[:, 2])], 1).astype(np.float32)
    c1 = np.stack([exp5(E[:, 3]), exp6(E[:, 4]), exp5(E[:, 5])], 1).astype(np.float32)
    w0 = (E[:, 0] << 11) | (E[:, 1] << 5) | E[:, 2]
    w1 = (E[:, 3] << 11) | (E[:, 4] << 5) | E[:, 5]
    four = w0 > w1
    p2 = np.where(four[:, None], (2 * c0 + c1) / 3, (c0 + c1) / 2)
    p3 = np.where(four[:, None], (c0 + 2 * c1) / 3, 0 * c0)
    return np.stack([c0, c1, p2, p3], 1), four


def job(f):
    out = os.path.join(SCR, "fullres", os.path.basename(f))
    if os.path.exists(out):
        return 1
    z = np.load(f)
    meta = json.loads(str(z["meta"]))
    syms = z["syms"].astype(np.int64)
    n = syms.shape[1]
    try:
        xy, bw, bh = placement(meta["texture_id"], n, "gfx10")
        ref, w1, h1 = mipscore.mip1_rgb(meta["texture_id"])
    except Exception:  # noqa: BLE001
        return 0
    if w1 != 2 * bw or h1 != 2 * bh:
        return 0
    E, known = endpoints(syms)
    pal, four = palette(E)
    qsym = np.zeros((2 * bh, 2 * bw), np.int16)
    P = np.zeros((bh, bw, 4, 3), np.float32)
    K = np.zeros((bh, bw), bool); F4 = np.zeros((bh, bw), bool); have = np.zeros((bh, bw), bool)
    for i, (x, y) in enumerate(xy):
        if x < 0:
            continue
        qsym[2 * y, 2 * x], qsym[2 * y, 2 * x + 1] = syms[8, i], syms[9, i]
        qsym[2 * y + 1, 2 * x], qsym[2 * y + 1, 2 * x + 1] = syms[10, i], syms[11, i]
        P[y, x] = pal[i]; K[y, x] = known[i]; F4[y, x] = four[i]; have[y, x] = True
    K &= have
    np.savez(out, qsym=qsym, pal=P, known=K, four=F4, mip1=np.array(ref, np.float32).reshape(h1, w1, 3),
             tid=meta["texture_id"], chunk=meta["chunk_id"])
    return 2


def main():
    os.makedirs(os.path.join(SCR, "fullres"), exist_ok=True)
    fs = sorted(glob.glob(os.path.join(SCR, "syms", "*.npz")))
    if "--limit" in sys.argv:
        k = int(sys.argv[sys.argv.index("--limit") + 1])
        fs = fs[::max(1, len(fs) // k)][:k]
    with Pool(os.cpu_count()) as p:
        r = p.map(job, fs, chunksize=4)
    print("written", sum(x == 2 for x in r), "cached", sum(x == 1 for x in r), "skipped", sum(x == 0 for x in r))


if __name__ == "__main__":
    main()
