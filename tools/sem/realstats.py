"""Collect real BC1 blocks from plain (zlib) mip-0 control textures for reference statistics.

usage: python3 -I tools/sem/realstats.py [--n 400]
Writes $SEM_SCRATCH/real/blocks.npy (uint64 BC1 blocks, stored order, concatenated) and
real/meta.json (per texture: id, w, h, tile_mode, offset, count). Never into git.
"""
import json
import os
import sys
import random
import argparse
from multiprocessing import Pool

BENCH = os.environ.get("BCPACK_BENCH", "/Users/ahmet/Code/bcpack-wt/harness/tools/bench")
sys.path.insert(0, BENCH)
import numpy as np  # noqa: E402
import mipscore  # noqa: E402

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")


def job(row):
    try:
        d = mipscore.plain_mip0_bytes(row["texture_id"])
    except Exception:  # noqa: BLE001
        return None
    return row, np.frombuffer(d, dtype="<u8").copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open("/tmp/bcpack-shared/mip_pairs.jsonl")]
    rows = [r for r in rows if r["kind"] == "control" and r["dxgi"] in (71, 72)
            and r["mip0_exact_chunk"] and 2048 <= r["mip0_len"] // 8 <= 16384]
    random.Random(1).shuffle(rows)
    rows = rows[:a.n]
    out, meta, off = [], [], 0
    with Pool(os.cpu_count()) as p:
        for res in p.imap(job, rows):
            if res is None:
                continue
            row, b = res
            meta.append(dict(texture_id=row["texture_id"], w=row["w"], h=row["h"],
                             tile_mode=row["tile_mode"], off=off, n=len(b)))
            out.append(b); off += len(b)
    os.makedirs(os.path.join(SCR, "real"), exist_ok=True)
    np.save(os.path.join(SCR, "real", "blocks.npy"), np.concatenate(out))
    json.dump(meta, open(os.path.join(SCR, "real", "meta.json"), "w"))
    print(len(meta), off)


if __name__ == "__main__":
    main()
