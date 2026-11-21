"""Corpus scan: run the cycle-level MDU simulation on every BC1 chunk of the
given stores and report exact convergence.

PASS means: every lane reached done (out_target symbols emitted, byte budget
of every fragment fully consumed), every lane's state after its last symbol is
a legal end state (L, L>>8, L>>16), and the bytes consumed equal the chunk
length (stored length minus trailing zero padding; consumed may exceed the
stripped length only by trailing zero bytes).

usage: python3 -I scan.py [--max-blocks N] [--limit K] store.ttpf ...
"""
import argparse
import os
import sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chunks import iter_chunks, stream_len
from mdusim import simulate
from mdu_tables import LEGAL

BC1_AB = [7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7]


class LambdaModel:
    """Control: the fitted lambda(q) model family of docs/BCPACK_SPEC.md
    (table index eab = alphabet bits - 1), exact-division inverse."""

    def __init__(self, eab, q):
        import models
        fr = models.spec_freqs(min(eab + 1, 8), q)
        self.tab = models.tables(fr)

    def forward(self, sym):
        if sym == 0:
            return self.tab[0][1] - 1, 1
        k = sym.bit_length()
        if k >= len(self.tab):
            return 1, 65535
        base, f, c, st = self.tab[k]
        return f, (base + f * (sym - st)) & 0xFFFF

    def inverse(self, cf):
        k = len(self.tab) - 1
        while self.tab[k][0] > cf:
            k -= 1
        base, f, c, st = self.tab[k]
        return 0 if k == 0 else st + (cf - base) // f


def run(job):
    path, idx, off, size, blocks, target_mode, variant = job
    import mdusim
    window, symvec, dN, flip = 8, "lockstep", 0, None
    for v in variant.split(","):
        if v.startswith("window="): window = int(v[7:])
        elif v.startswith("symvec="): symvec = v[7:]
        elif v.startswith("dN="): dN = int(v[3:])
        elif v.startswith("flip="): flip = float(v[5:])
        elif v == "lambda": mdusim.Model = LambdaModel
    with open(path, "rb") as f:
        f.seek(off)
        d = f.read(size)
    sl = len(d.rstrip(b"\0"))
    if flip is not None:
        d = bytearray(d); d[min(max(sl - 1, 0), int(sl * flip))] ^= 0x01; d = bytes(d)
    N = blocks if target_mode == "blocks" else target_mode
    N += dN
    r = simulate(d, BC1_AB, [N] * 12, stream_len=size, window=window, symvec=symvec,
                 max_cycles=40 * (12 * N + size) + 1000)
    ok_done = all(l["done"] for l in r["lanes"])
    ok_legal = all(l["final"] in LEGAL for l in r["lanes"])
    ok_len = sl <= r["pos"] <= size
    return dict(store=os.path.basename(path), idx=idx, blocks=blocks, size=size, sl=sl,
                pos=r["pos"], why=r["why"], done=ok_done, legal=ok_legal, lenok=ok_len,
                ndone=sum(l["done"] for l in r["lanes"]),
                nfr=[len(l["frags"]) for l in r["lanes"]],
                passed=ok_done and ok_legal and ok_len)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stores", nargs="+")
    ap.add_argument("--max-blocks", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--target", default="blocks")
    ap.add_argument("--variant", default="", help="comma list: window=K symvec=free dN=D flip=F lambda")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    tm = "blocks" if args.target == "blocks" else int(args.target)
    jobs = []
    for p in args.stores:
        for i, codec, mm, off, size, blocks in iter_chunks(p):
            if blocks <= args.max_blocks:
                jobs.append((p, i, off, size, blocks, tm, args.variant))
    if args.limit:
        step = max(1, len(jobs) // args.limit)
        jobs = jobs[::step][:args.limit]
    from collections import Counter
    byb = Counter(); passb = Counter(); fails = []
    with Pool(os.cpu_count()) as pool:
        for r in pool.imap_unordered(run, jobs, chunksize=1):
            byb[r["blocks"]] += 1
            if r["passed"]:
                passb[r["blocks"]] += 1
            else:
                fails.append(r)
            if not args.quiet or not r["passed"]:
                print(("PASS " if r["passed"] else "FAIL ") + str(r), flush=True)
    print("VARIANT", repr(args.variant))
    print("SUMMARY passed %d / %d" % (sum(passb.values()), sum(byb.values())))
    for b in sorted(byb):
        print("  blocks %6d: %d / %d" % (b, passb[b], byb[b]))
