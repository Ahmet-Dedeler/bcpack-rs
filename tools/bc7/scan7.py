"""Entropy-layer convergence scan for non-BC1 BCPack codecs (BC7 first).

Runs the cycle-level MDU simulation (copy of the probe's mdusim.py: silicon-rans
lane decoder + lane-ordered 8-byte/clock feeder + lockstep symbol vector) on
chunks of a given codec with a candidate lane configuration, and reports exact
convergence:

  PASS = every lane done (emitted its target symbol count with every fragment
  budget consumed), every lane's final state legal (L, L>>8, L>>16), and bytes
  consumed == chunk length with trailing zero padding stripped (consumed may
  exceed the stripped length only by trailing zero bytes).

A config is "W:T,W:T,..." one entry per lane: W = lane width in bits
(alphabet_bits = W-1), T = symbol count: "B" (blocks), "2B", "B/2", or an int.

usage:
  python3 -I scan7.py --codec 0x1c --config 4:2B,4:2B,8:2B,... --blocks 512 --limit 40
  python3 -I scan7.py --codec 0x1c --config ... --ids ids.txt --jobs 6 [--verbose]

Game data is read in place (index: /tmp/bcpack-shared/<codec>_index.jsonl or
chunk table walk); nothing is written except to stdout / --json.
"""
import argparse
import json
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdusim import simulate
from mdu_tables import LEGAL
from chunks import iter_chunks

GAME_ROOT = open("/tmp/bcpack-shared/game_root.txt").read().strip()
BLOCK_BYTES = {0x04: 8, 0x0C: 16, 0x10: 8, 0x14: 16, 0x18: 16, 0x1C: 16}


def parse_config(spec):
    """'4:2B,8:B,...' -> list of (width, count_expr)."""
    out = []
    for part in spec.split(","):
        w, t = part.split(":")
        out.append((int(w), t))
    return out


def count_of(expr, blocks):
    if expr.endswith("B"):
        k = expr[:-1]
        return blocks * (int(k) if k else 1)
    if expr.startswith("B/"):
        return blocks // int(expr[2:])
    return int(expr)


def run_chunk(data, blocks, config, window=8, symvec="lockstep", order=None):
    ab = [w - 1 for w, _ in config]
    tg = [count_of(t, blocks) for _, t in config]
    size = len(data)
    stripped = len(data.rstrip(b"\0"))
    r = simulate(data, ab, tg, stream_len=size, window=window, symvec=symvec,
                 order=order, max_cycles=64 * (sum(tg) + size) + 1000)
    lanes_ok = [l["done"] and l["final"] in LEGAL for l in r["lanes"]]
    ok = all(lanes_ok) and stripped <= r["pos"] <= size
    return ok, r, stripped, lanes_ok


def load_chunks(codec, ids=None, blocks=None, limit=None, seed=1, multistream=None):
    """Return list of (id, path, chunk_index, blocks). Uses the shared index for
    BC7, otherwise walks every store's chunk table."""
    name = {0x1C: "bc7"}.get(codec)
    rows = []
    idx = "/tmp/bcpack-shared/%s_index.jsonl" % name if name else None
    if idx and os.path.exists(idx):
        for line in open(idx):
            r = json.loads(line)
            if multistream is not None and r.get("sixng_multistream") != multistream:
                continue
            rows.append((r["id"], r["store"], r["chunk"], r["blocks"]))
    else:
        rows = list_codec_chunks(codec)
    if ids:
        want = set(ids)
        rows = [r for r in rows if r[0] in want]
    if blocks:
        rows = [r for r in rows if r[3] in blocks]
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def list_codec_chunks(codec):
    cache = "/tmp/bcpack-other-formats/codec_%02x_index.json" % codec
    if os.path.exists(cache):
        return [tuple(x) for x in json.load(open(cache))]
    rows = []
    for dp, _, fs in os.walk(GAME_ROOT):
        for f in fs:
            if not f.endswith(".ttpf"):
                continue
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, GAME_ROOT)
            try:
                for i, c, mm, off, size, blocks in iter_chunks(p, codecs=(codec,)):
                    rows.append((rel + ":%d" % i, rel, i, blocks))
            except AssertionError:
                pass
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(rows, open(cache, "w"))
    return rows


_CODEC = 0x1C


def read_chunk(rel, chunk, codec=None):
    codec = codec or _CODEC
    for i, c, mm, off, size, blocks in iter_chunks(os.path.join(GAME_ROOT, rel), codecs=(codec,)):
        if i == chunk:
            return bytes(mm[off:off + size]), blocks
    raise KeyError((rel, chunk))


def job(args):
    cid, rel, chunk, blocks, config, opts = args
    data, b2 = read_chunk(rel, chunk, opts.get("codec"))
    ok, r, stripped, lanes_ok = run_chunk(data, blocks, config, window=opts.get("window", 8),
                                          symvec=opts.get("symvec", "lockstep"))
    res = dict(id=cid, blocks=blocks, ok=ok, why=r["why"], pos=r["pos"], stripped=stripped,
               size=len(data), lanes_ok=lanes_ok,
               nsyms=[l["nsyms"] for l in r["lanes"]],
               frags=[[(f[0], f[1], f[2]) for f in l["frags"]] for l in r["lanes"]])
    if opts.get("keep_syms"):
        res["syms"] = r["syms"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec", default="0x1c")
    ap.add_argument("--config", required=True)
    ap.add_argument("--blocks", default=None, help="comma list of block counts to keep")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--limit", type=int, default=40, help="0 = all")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--symvec", default="lockstep")
    ap.add_argument("--single", action="store_true", help="skip SixNG multistream chunks")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--progress", type=int, default=0)
    a = ap.parse_args()
    global _CODEC
    codec = int(a.codec, 0)
    _CODEC = codec
    config = parse_config(a.config)
    ids = [l.strip() for l in open(a.ids)] if a.ids else None
    blocks = [int(x) for x in a.blocks.split(",")] if a.blocks else None
    rows = load_chunks(codec, ids, blocks, a.limit, a.seed, False if a.single else None)
    opts = dict(codec=codec, window=a.window, symvec=a.symvec)
    jobs = [(cid, rel, ch, b, config, opts) for cid, rel, ch, b in rows]
    jobs.sort(key=lambda j: -j[3])  # big chunks first for load balance
    res = []
    with Pool(a.jobs) as p:
        for k, r in enumerate(p.imap_unordered(job, jobs, chunksize=1)):
            res.append(r)
            if a.progress and (k + 1) % a.progress == 0:
                print("progress %d/%d pass %d" % (k + 1, len(jobs), sum(x["ok"] for x in res)), flush=True)
    npass = sum(r["ok"] for r in res)
    lane_pass = [sum(r["lanes_ok"][j] for r in res) for j in range(len(config))]
    for r in res:
        if a.verbose or not r["ok"]:
            print("%s %s blocks=%d why=%s pos=%d stripped=%d size=%d lanes=%s nsyms=%s" % (
                "PASS" if r["ok"] else "FAIL", r["id"], r["blocks"], r["why"], r["pos"],
                r["stripped"], r["size"], "".join("1" if x else "0" for x in r["lanes_ok"]),
                r["nsyms"]))
    print("config", a.config)
    print("PASS %d / %d   per-lane ok %s" % (npass, len(res), lane_pass))
    if a.json:
        json.dump(res, open(a.json, "w"))


if __name__ == "__main__":
    main()
