"""Ceiling for the quadrant-weight test: same corr^2 statistic on plain (zlib) control
textures, where the true BC1 indices are known.  Also writes real quadrant patterns +
mip-1 luminance to $SEM_SCRATCH/ceiling.npz for codebook matching.
usage: venv/bin/python -I tools/sem/ceiling.py [--n 300]
"""
import json
import os
import sys
import random
import struct
from multiprocessing import Pool

BENCH = os.environ.get("BCPACK_BENCH", "/Users/ahmet/Code/bcpack-wt/harness/tools/bench")
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import mipscore  # noqa: E402

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")


def job(row):
    tid = row["texture_id"]
    try:
        st, tex, levels = mipscore.get_tex(tid)
        data = mipscore.plain_mip0_bytes(tid)
        lin, bw, bh = mipscore.detile_blocks(data, levels[0], tex.tile_mode)
        ref, w1, h1 = mipscore.mip1_rgb(tid)
    except Exception:  # noqa: BLE001
        return None
    if w1 != 2 * bw or h1 != 2 * bh:
        return None
    blk = np.frombuffer(lin, dtype="<u8")[:bw * bh].reshape(bh, bw)
    ref = np.array(ref, dtype=np.float32).reshape(h1, w1, 3)
    rgb = np.stack([ref[0::2, 0::2], ref[0::2, 1::2], ref[1::2, 0::2], ref[1::2, 1::2]], 2)
    return blk.ravel(), rgb.reshape(-1, 4, 3)


def main():
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 300
    rows = [json.loads(l) for l in open("/tmp/bcpack-shared/mip_pairs.jsonl")]
    rows = [r for r in rows if r["kind"] == "control" and r["dxgi"] in (71, 72) and r["mip0_exact_chunk"]
            and 2048 <= r["mip0_len"] // 8 <= 16384 and r["tile_mode"] in (2, 3)]
    random.Random(2).shuffle(rows)
    B, R = [], []
    with Pool(os.cpu_count()) as p:
        for res in p.imap(job, rows[:n]):
            if res is not None:
                B.append(res[0]); R.append(res[1])
    B = np.concatenate(B); R = np.concatenate(R)
    np.savez(os.path.join(SCR, "ceiling.npz"), blocks=B, rgb=R)
    from weights import r2
    Y = R @ np.array([.299, .587, .114], dtype=np.float32)
    m = (Y.max(1) - Y.min(1)) > 8
    idx = (B >> np.uint64(32)).astype(np.int64)
    c0 = (B & np.uint64(0xffff)).astype(np.int64); c1 = ((B >> np.uint64(16)) & np.uint64(0xffff)).astype(np.int64)
    px = np.stack([(idx >> (2 * k)) & 3 for k in range(16)], 1)
    w4 = np.array([0, 3, 1, 2]) / 3.0
    w3 = np.array([0, 2, 1, 1]) / 2.0  # 3-colour: idx3 = black, treat as mid (rough)
    W = np.where((c0 > c1)[:, None], w4[px], w3[px])
    q = [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
    M = np.stack([W[:, qq].mean(1) for qq in q], 1)
    print("blocks", len(B), "busy", m.sum())
    print("ceiling corr^2 (true indices, BC1 weights): %.4f" % r2(M[m], Y[m])[0])
    rng = np.random.default_rng(0)
    print("chance (random weight per real quadrant pattern): %.4f" % r2(rng.random(256)[
        np.stack([(idx >> (8 * 0)) & 0, ] * 4, 1) * 0 + rng.integers(0, 256, (len(B), 4))][m], Y[m])[0])
    pat = np.stack([sum(px[:, qq[k]] << (2 * k) for k in range(4)) for qq in q], 1)
    from als import als
    nn = m.sum() // 2
    Lm, Ym = pat[m], Y[m]
    w, s = als(Lm[:nn], Ym[:nn], False, iters=20)
    key = Lm[nn:] + 256 * np.arange(4)[None, :]
    print("ALS per-pattern weight on TRUE raw patterns: train %.4f held-out %.4f" % (s, r2(w[key], Ym[nn:])[0]))


if __name__ == "__main__":
    main()
