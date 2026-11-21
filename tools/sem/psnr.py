"""Full decode -> harness mip-consistency PSNR.

For each cached pair texture whose endpoints are fully known (no reference before block 0),
build the BC1 chunk in stream order: endpoints by the back-reference rule (fullres.endpoints),
indices from a quadrant table (symbol -> codes TL TR BL BR of the 2x2 group; quadrant lanes
8-11 = TL TR BL BR of the block). Score with mipscore.score(tid, bytes, 'stream').
Variants: --table path.npy (row 0 = table) ; controls: 'shuffled' (random bijection keeping
symbols 0-3) and 'mid' (every index = code 2).
usage: python3 -I tools/sem/psnr.py --table /tmp/bcpack-sem-indices/rgb_table.npy [--n 200]
"""
import glob
import json
import os
import struct
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH = os.environ.get("BCPACK_BENCH", "/Users/ahmet/Code/bcpack-wt/harness/tools/bench")
sys.path.insert(0, BENCH)
import numpy as np  # noqa: E402
import mipscore  # noqa: E402
from fullres import endpoints  # noqa: E402

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")
PATS = [[(p >> (2 * k)) & 3 for k in range(4)] for p in range(256)]
TABLES = {}


def index_word(pq):
    """pq: 4 patterns (TL TR BL BR quadrants) -> 32-bit BC1 index word."""
    w = 0
    for q, (oy, ox) in enumerate([(0, 0), (0, 2), (2, 0), (2, 2)]):
        p = PATS[pq[q]]
        for k, (dy, dx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            y, x = oy + dy, ox + dx
            w |= p[k] << (2 * (4 * y + x))
    return w


def build(syms, table, swap3=False):
    E, known = endpoints(syms)
    out = bytearray()
    n = syms.shape[1]
    for i in range(n):
        e = E[i]
        c0 = int((e[0] << 11) | (e[1] << 5) | e[2]); c1 = int((e[3] << 11) | (e[4] << 5) | e[5])
        if table is None:
            idx = 0xAAAAAAAA
        else:
            idx = index_word([int(table[syms[8 + q, i]]) for q in range(4)])
        out += struct.pack("<HHI", c0, c1, idx)
    return bytes(out), bool(known.all())


def job(args):
    f, tables = args
    z = np.load(f)
    meta = json.loads(str(z["meta"]))
    syms = z["syms"].astype(np.int64)
    res = {"tid": meta["texture_id"], "blocks": syms.shape[1]}
    for name, t in tables.items():
        data, ok = build(syms, t)
        if not ok:
            return None
        r = mipscore.score(meta["texture_id"], data, "stream")
        res[name] = r["psnr"]; res["flat"] = r["psnr_flat"]
    return res


def main():
    tpath = sys.argv[sys.argv.index("--table") + 1]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 200
    T = np.load(tpath)[0].astype(int)
    rng = np.random.default_rng(0)
    sh = T.copy(); perm = rng.permutation(np.arange(4, 256)); sh[4:] = T[perm]
    TABLES.update({"table": T, "shuffled": sh, "mid": None})
    for extra in sys.argv[1:]:
        if extra.startswith("--also="):
            nm, p = extra[7:].split(":")
            TABLES[nm] = np.load(p)[0].astype(int)
    names = list(TABLES)
    fs = []
    for f in sorted(glob.glob(os.path.join(SCR, "fullres", "*.npz"))):
        z = np.load(f)
        if z["known"].all():
            fs.append(os.path.join(SCR, "syms", os.path.basename(f)))
    if "--stores" in sys.argv:
        from fit_rgb import store_split
        par = {"even": 0, "odd": 1}[sys.argv[sys.argv.index("--stores") + 1]]
        keep = set(os.path.basename(f) for f in store_split(
            [os.path.join(SCR, "fullres", os.path.basename(f)) for f in fs], par))
        fs = [f for f in fs if os.path.basename(f) in keep]
    fs = fs[::max(1, len(fs) // n)][:n]
    with Pool(os.cpu_count()) as p:
        rows = [r for r in p.map(job, [(f, TABLES) for f in fs], chunksize=2) if r]
    print("textures", len(rows))
    for nm in names + ["flat"]:
        v = np.array([r[nm] for r in rows])
        print("%-9s median %.2f  p10 %.2f  p90 %.2f" % (nm, np.median(v), np.percentile(v, 10), np.percentile(v, 90)))
    d = np.array([r["table"] - r["flat"] for r in rows])
    print("table - flat: median %+.2f dB, > +3 dB in %d / %d" % (np.median(d), (d > 3).sum(), len(d)))
    json.dump(rows, open(os.path.join(SCR, "psnr_rows.json"), "w"))


if __name__ == "__main__":
    main()
