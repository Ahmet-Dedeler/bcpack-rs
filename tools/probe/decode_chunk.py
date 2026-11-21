"""Decode one BC1 BCPack chunk to its 12 per-lane symbol streams.

usage: python3 -I decode_chunk.py store.ttpf CHUNK_INDEX [--out file.npy-like.txt]
Prints per-lane fragment list, convergence, and symbol histograms. Uses the
same feeder/symvec hypotheses as scan.py (window 8, lockstep, N = blocks).
"""
import os
import sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chunks import iter_chunks
from mdusim import simulate
from mdu_tables import LEGAL
from scan import BC1_AB

def decode_chunk_symbols(path, chunk_index):
    """Return (blocks, syms) for one BC1 chunk; syms is a list of 12 per-lane
    symbol lists (lane order 0..11). Raises ValueError if the chunk does not
    converge exactly (all lanes done, legal end states, budget consumed)."""
    for i, codec, mm, off, size, blocks in iter_chunks(path):
        if i != chunk_index:
            continue
        d = mm[off:off + size]
        r = simulate(d, BC1_AB, [blocks] * 12, stream_len=size)
        ok = all(l["done"] and l["final"] in LEGAL for l in r["lanes"]) and \
            len(d.rstrip(b"\0")) <= r["pos"] <= size
        if not ok:
            raise ValueError("chunk %d did not converge: %s" % (chunk_index, r["why"]))
        return blocks, r["syms"]
    raise KeyError("no BC1 chunk %d in %s" % (chunk_index, path))


if __name__ == "__main__":
    store, want = sys.argv[1], int(sys.argv[2])
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    for i, codec, mm, off, size, blocks in iter_chunks(store):
        if i != want:
            continue
        d = mm[off:off + size]
        r = simulate(d, BC1_AB, [blocks] * 12, stream_len=size)
        print("chunk", i, "blocks", blocks, "size", size, "consumed", r["pos"],
              "stripped", len(d.rstrip(b"\0")), r["why"])
        for ln, syms in zip(r["lanes"], r["syms"]):
            print("lane %2d done=%s legal=%s n=%d frags=%s" % (
                ln["idx"], ln["done"], ln["final"] in LEGAL, len(syms),
                [(f[0], "%02x" % f[1], f[2], "eab%d" % f[3], "q%d" % f[4]) for f in ln["frags"]]))
            print("        top symbols", Counter(syms).most_common(8), "first", syms[:16])
        if out:
            with open(out, "w") as f:
                for syms in r["syms"]:
                    f.write(" ".join(map(str, syms)) + "\n")
