# bcpack-rs by Ahmet Dedeler (https://github.com/Ahmet-Dedeler, https://ahmetdedeler.com)
# Source: https://github.com/Ahmet-Dedeler/bcpack-rs
# SPDX-License-Identifier: MIT OR Apache-2.0
# If you copy, vendor or port this file, keep this header and credit bcpack-rs in your README.
#
# Retail-data check (not run by the test suites). Needs your own texture pools and the
# research branches checked out. Configure with environment variables:
#   BCPACK_GAME_ROOT   texture pool root (read by the harness)
#   BCPACK_BENCH       harness branch tools/bench
#   BCPACK_PROBE       probe branch tools/probe
#   BCPACK_SEM_ENDPOINTS, BCPACK_SEM_INDICES   sem-endpoints / sem-indices branch tools/sem
#   BCPACK_WINDOW      window fit JSON used by the research pipeline (same values as data/bc1_window.json)
#   BCPACK_BIN         the bcpack binary (default target/release/bcpack)
#   BCPACK_BC1_INDEX   harness corpus index bc1_index.jsonl (crosscheck)
#   BCPACK_SCRATCH     scratch directory (default: system temp dir)
"""Cross-check: Rust `bcpack decode-bc1` vs the Python research pipeline
(probe mdusim + sem-endpoints decode_window + sem-indices quad table), byte for byte.
usage: venv/bin/python -I crosscheck.py N [seed] [maxblocks]"""
import tempfile, sys, os, json, random, subprocess, struct, hashlib, time
from multiprocessing import Pool
sys.path.insert(0, os.environ["BCPACK_BENCH"])
sys.path.insert(0, os.environ["BCPACK_PROBE"])
sys.path.insert(0, os.environ["BCPACK_SEM_ENDPOINTS"])
import numpy as np
import ttpf, mdusim
from endpoints import decode_window, load_window
BIN = os.environ.get("BCPACK_BIN", "target/release/bcpack")
SCR = os.path.join(os.environ.get("BCPACK_SCRATCH", tempfile.gettempdir()), "bcpack-xc")
os.makedirs(SCR, exist_ok=True)
QT = json.load(open(os.path.join(os.environ["BCPACK_SEM_INDICES"], "quad_table.json")))["entries"]
CODES = {e["sym"]: e["codes"] for e in QT}
def _load_hist(path):
    """Window JSON as written by the endpoint fit: {"slots": {slot: {"value": [...]}}} or {slot: {...}}."""
    d = json.load(open(path))
    src = d["slots"] if isinstance(d.get("slots"), dict) else d
    W = [(0, 0, 0, 0, 0, 0)] * 256
    for k, v in src.items():
        if k.isdigit():
            W[int(k)] = tuple(v.get("value") or v["joint"])
    return W
HIST = _load_hist(os.environ["BCPACK_WINDOW"])

def qmask(q, s):
    oy, ox = [(0, 0), (0, 2), (2, 0), (2, 2)][q]
    w = 0
    for k, (dy, dx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        w |= CODES[s][k] << (2 * (4 * (oy + dy) + ox + dx))
    return w
MASK = [[qmask(q, s) for s in range(256)] for q in range(4)]

def research(raw, blocks):
    r = mdusim.simulate(raw, [7,7,4,5,4,4,5,4,7,7,7,7], [blocks]*12, stream_len=len(raw))
    ok = all(l["done"] and l["final"] in mdusim.LEGAL for l in r["lanes"]) and \
        len(raw.rstrip(b"\0")) <= r["pos"] <= len(raw)
    if not ok:
        return None
    syms = np.array(r["syms"], dtype=np.int64)
    E = decode_window(syms, resid="zigA", zero_mode="zero", history=HIST)
    out = bytearray()
    for i in range(blocks):
        e = [int(v) for v in E[i]]
        c0 = (e[0] << 11) | (e[1] << 5) | e[2]; c1 = (e[3] << 11) | (e[4] << 5) | e[5]
        w = MASK[0][r["syms"][8][i]] | MASK[1][r["syms"][9][i]] | MASK[2][r["syms"][10][i]] | MASK[3][r["syms"][11][i]]
        out += struct.pack("<HHI", c0, c1, w)
    return bytes(out)

def job(cid):
    c = ttpf.chunk_info(cid)
    raw = ttpf.extract(cid, strip_pad=False)
    blocks = c.vlen // 8
    fn = os.path.join(SCR, cid.replace("/", "_") + ".bin")
    open(fn, "wb").write(raw)
    t = time.time()
    p = subprocess.run([BIN, "decode-bc1", fn, str(blocks), fn + ".bc1"], capture_output=True)
    rust = open(fn + ".bc1", "rb").read() if p.returncode == 0 else None
    py = research(raw, blocks)
    for f in (fn, fn + ".bc1"):
        if os.path.exists(f): os.unlink(f)
    return cid, blocks, rust is not None, py is not None, rust == py and rust is not None, \
        hashlib.sha256(rust).hexdigest()[:16] if rust else None

if __name__ == "__main__":
    n = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    maxb = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
    byst = {}
    for line in open(os.environ["BCPACK_BC1_INDEX"]):
        d = json.loads(line)
        if d["blocks"] <= maxb:
            byst.setdefault(d["store"], []).append(d["id"])
    rng = random.Random(seed)
    stores = sorted(byst)
    ids = []
    while len(ids) < n:
        for s in stores:
            if len(ids) < n and byst[s]:
                ids.append(byst[s].pop(rng.randrange(len(byst[s]))))
    with Pool(int(os.environ.get("JOBS", "6"))) as p:
        res = p.map(job, ids, chunksize=1)
    same = sum(r[4] for r in res); both_fail = sum((not r[2]) and (not r[3]) for r in res)
    print("chunks %d stores %d: identical %d, both failed %d, mismatches %d" % (
        len(res), len(set(i.rpartition(':')[0] for i in ids)), same, both_fail, len(res) - same - both_fail))
    for r in res:
        if not r[4] and (r[2] or r[3]): print("MISMATCH", r)
    json.dump(res, open(os.path.join(os.path.dirname(SCR), "%s_result.json" % os.path.splitext(os.path.basename(__file__))[0]), "w"))
