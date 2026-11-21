"""Attach mip-1 reference colours to cached lane symbols.

For each cached pair texture (tools/sem/cache.py), mip 0 block b in STREAM order is placed
with an order hypothesis (gfx10 = the stored Gfx10 tile order, the same layout plain
chunks use; linear = row-major). Its 2x2 quadrant q (TL, TR, BL, BR) covers exactly one
mip-1 pixel, (2bx+qx, 2by+qy). Output $SEM_SCRATCH/quad_<order>.npz:
  syms  int16 [12, B]     all cached blocks, concatenated
  rgb   uint8 [B, 4, 3]   mip-1 pixel under each quadrant (TL, TR, BL, BR); 0 for padding
  valid bool  [B]         block lies inside the level
  tex   int32 [B]         texture number (index into the cache), pos int32 [B] (bx, by packed)

usage: python3 -I tools/sem/quadref.py [--order gfx10|linear] [--limit N]
"""
import glob
import json
import os
import sys
import argparse
from multiprocessing import Pool

BENCH = os.environ.get("BCPACK_BENCH", "/Users/ahmet/Code/bcpack-wt/harness/tools/bench")
sys.path.insert(0, BENCH)
import numpy as np  # noqa: E402
import ttpf  # noqa: E402
import mipscore  # noqa: E402

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")
RING = "--ring" in sys.argv   # store the 4x4 mip-1 window around each block instead (quadring_*.npz)


def placement(texture_id, nblocks, order):
    """For stream block i -> (bx, by) or (-1, -1)."""
    st, tex, levels = mipscore.get_tex(texture_id)
    p = levels[0]
    pwb, phb = p.pw // 4, p.ph // 4
    bw, bh = -(-p.w // 4), -(-p.h // 4)
    ox, oy = p.ox // 4, p.oy // 4
    if order == "gfx10":
        pat = ttpf.PATTERNS[(ttpf.tile_kind(tex.tile_mode), 8)]
        lin = ttpf.untile_order(nblocks, pwb, phb, pat)
        xy = [(-1, -1) if d < 0 else (d % pwb - ox, d // pwb - oy) for d in lin]
    else:
        xy = [(i % bw, i // bw) for i in range(nblocks)]
    return [(x, y) if 0 <= x < bw and 0 <= y < bh else (-1, -1) for x, y in xy], bw, bh


def job(args):
    f, order = args
    z = np.load(f)
    meta = json.loads(str(z["meta"]))
    syms = z["syms"]
    n = syms.shape[1]
    try:
        ref, w1, h1 = mipscore.mip1_rgb(meta["texture_id"])
        xy, bw, bh = placement(meta["texture_id"], n, order)
    except Exception as e:  # noqa: BLE001
        return None
    rgb = np.zeros((n, 4, 3), np.uint8)
    ring = np.zeros((n, 4, 4, 3), np.uint8) if RING else None
    refa = np.array(ref, np.uint8).reshape(h1, w1, 3) if RING else None
    valid = np.zeros(n, bool)
    pos = np.full(n, -1, np.int32)
    for i, (x, y) in enumerate(xy):
        if x < 0 or 2 * y + 1 >= h1 or 2 * x + 1 >= w1:
            continue
        valid[i] = True
        pos[i] = (y << 16) | x
        rgb[i, 0] = ref[2 * y][2 * x]; rgb[i, 1] = ref[2 * y][2 * x + 1]
        rgb[i, 2] = ref[2 * y + 1][2 * x]; rgb[i, 3] = ref[2 * y + 1][2 * x + 1]
        if RING:
            ys = np.clip(np.arange(2 * y - 1, 2 * y + 3), 0, h1 - 1)
            xs = np.clip(np.arange(2 * x - 1, 2 * x + 3), 0, w1 - 1)
            ring[i] = refa[ys][:, xs]
    return int(os.path.basename(f)[:5]), syms.astype(np.int16), (ring if RING else rgb), valid, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default="gfx10")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ring", action="store_true")
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(SCR, "syms", "*.npz")))
    if a.limit:
        fs = fs[::max(1, len(fs) // a.limit)][:a.limit]
    S, R, V, T, P = [], [], [], [], []
    with Pool(os.cpu_count()) as p:
        for r in p.imap(job, [(f, a.order) for f in fs], chunksize=4):
            if r is None:
                continue
            t, s, rgb, v, pos = r
            S.append(s); R.append(rgb); V.append(v); P.append(pos)
            T.append(np.full(s.shape[1], t, np.int32))
    np.savez(os.path.join(SCR, ("quadring_%s.npz" if RING else "quad_%s.npz") % a.order), syms=np.concatenate(S, 1),
             rgb=np.concatenate(R), valid=np.concatenate(V), tex=np.concatenate(T),
             pos=np.concatenate(P))
    print(len(S), sum(len(v) for v in V))


if __name__ == "__main__":
    main()
