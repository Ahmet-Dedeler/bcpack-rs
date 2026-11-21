"""Coordinate search of per-lane widths using the per-fragment end signal.

For each lane j and width w (1..9), run the sim with lane j's width set to w and
report lane j's fragment ends ("L" = fragment drained exactly to x == L, the
encoder's start state, i.e. decoded correctly; "X" = drained below L = wrong).
python3 -I widths7.py STORE CHUNK CONFIG LANES(comma)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scan7
from multiprocessing import Pool
rel, ch, cfg0 = sys.argv[1], int(sys.argv[2]), scan7.parse_config(sys.argv[3])
lanes = [int(x) for x in sys.argv[4].split(",")]
codec = int(sys.argv[5], 0) if len(sys.argv) > 5 else 0x1C
data, blocks = scan7.read_chunk(rel, ch, codec)

def one(job):
    j, w = job
    cfg = list(cfg0); cfg[j] = (w, cfg[j][1])
    ok, r, s, lok = scan7.run_chunk(data, blocks, cfg)
    e = r["lanes"][j]["ends"]
    nL = sum(1 for x in e if x[0] == "L"); nX = sum(1 for x in e if x[0] == "X")
    allL = sum(1 for l in r["lanes"] for x in l["ends"] if x[0] == "L")
    allX = sum(1 for l in r["lanes"] for x in l["ends"] if x[0] == "X")
    return j, w, ok, lok[j], nL, nX, e[:4], allL, allX, sum(lok)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("fork")
    jobs = [(j, w) for j in lanes for w in range(1, 10)]
    with Pool(6) as p:
        for j, w, ok, lj, nL, nX, e, aL, aX, nleg in p.imap(one, jobs):
            print("lane %2d w=%d ok=%d legal=%d L=%d X=%d first=%s | all L=%d X=%d legal=%d" % (j, w, ok, lj, nL, nX, e, aL, aX, nleg), flush=True)
