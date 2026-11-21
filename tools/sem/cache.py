"""Decode the mip-0 BCPack chunk of mip-pair textures into per-lane symbols and cache them.

usage: python3 -I tools/sem/cache.py [--max-blocks 4096] [--limit N] [--jobs J]

Uses the probe agent's cycle-level MDU simulator (copied here as mdusim.py; feeder =
lane-order window 8, symvec = lockstep, N = blocks on all 12 lanes). Output goes to
$SEM_SCRATCH (default /tmp/bcpack-sem-indices)/syms/<n>.npz, never into git:
  syms   uint16 [12, blocks]   lane symbols in emission order
  meta   json string (texture row from mip_pairs.jsonl + convergence flags)
"""
import json
import os
import sys
import argparse
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "bench"))
BENCH = os.environ.get("BCPACK_BENCH", "/Users/ahmet/Code/bcpack-wt/harness/tools/bench")
sys.path.insert(0, BENCH)
import numpy as np  # noqa: E402
import ttpf  # noqa: E402
from mdusim import simulate  # noqa: E402
from mdu_tables import LEGAL  # noqa: E402

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")
BC1_AB = [7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7]


def decode_chunk(chunk_id):
    rel, idx = ttpf.parse_chunk_id(chunk_id)
    st = ttpf.open_store(rel)
    c = st.chunks[idx]
    d = st.chunk_bytes(idx, strip_pad=False)
    blocks = c.vlen // 8
    r = simulate(d, BC1_AB, [blocks] * 12, stream_len=len(d),
                 max_cycles=40 * (12 * blocks + len(d)) + 1000)
    ok = all(l["done"] for l in r["lanes"]) and all(l["final"] in LEGAL for l in r["lanes"])
    return r, ok, blocks


def job(args):
    n, row = args
    out = os.path.join(SCR, "syms", "%05d.npz" % n)
    if os.path.exists(out):
        return n, "cached"
    cid = row["mip0_chunks"][0][0]
    try:
        r, ok, blocks = decode_chunk(cid)
    except Exception as e:  # noqa: BLE001
        return n, "error %r" % e
    if not ok or any(len(s) != blocks for s in r["syms"]):
        return n, "noconv"
    row = dict(row, chunk_id=cid, blocks=blocks)
    np.savez_compressed(out, syms=np.array(r["syms"], dtype=np.uint16), meta=json.dumps(row))
    return n, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-blocks", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    rows = [json.loads(l) for l in open("/tmp/bcpack-shared/mip_pairs.jsonl")]
    rows = [r for r in rows if r["kind"] == "pair" and r["mip0_len"] // 8 <= a.max_blocks]
    rows.sort(key=lambda r: r["texture_id"])
    todo = list(enumerate(rows))
    if a.limit:
        step = max(1, len(todo) // a.limit)
        todo = todo[::step][:a.limit]
    from collections import Counter
    c = Counter()
    with Pool(a.jobs) as p:
        for n, s in p.imap_unordered(job, todo):
            c[s.split()[0]] += 1
    print(dict(c))


if __name__ == "__main__":
    main()
