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
"""End-to-end: Rust decode of pair textures' BCPack mip 0 -> harness mipscore vs plain mip 1.
usage: python3 -I mip_e2e.py N [seed] [max_side]"""
import tempfile, sys, os, json, random, subprocess, re
from multiprocessing import Pool
sys.path.insert(0, os.environ["BCPACK_BENCH"])
import ttpf, mipscore
BIN = os.environ.get("BCPACK_BIN", "target/release/bcpack")
SCR = os.path.join(os.environ.get("BCPACK_SCRATCH", tempfile.gettempdir()), "bcpack-mip"); os.makedirs(SCR, exist_ok=True)

def job(tid):
    r = mipscore.pairs()[tid]
    parts = []; unf = 0; pre = 0
    for cid, _, _ in r["mip0_chunks"]:
        info = ttpf.chunk_info(cid)
        fn = os.path.join(SCR, cid.replace("/", "_") + ".bin")
        open(fn, "wb").write(ttpf.extract(cid, strip_pad=False))
        p = subprocess.run([BIN, "decode-bc1", fn, str(info.vlen // 8), fn + ".bc1"], capture_output=True, text=True)
        os.unlink(fn)
        if p.returncode != 0:
            return {"tid": tid, "error": p.stderr.strip()[-200:]}
        m = re.search(r"(\d+) references before block 0, (\d+) blocks depend", p.stderr)
        pre += int(m.group(1)); unf += int(m.group(2))
        parts.append(open(fn + ".bc1", "rb").read()); os.unlink(fn + ".bc1")
    s = mipscore.score(tid, b"".join(parts), "stream")
    return {"tid": tid, "psnr": s["psnr"], "flat": s["psnr_flat"], "unfitted": unf, "prestart": pre,
            "blocks": sum(len(x) for x in parts) // 8}

def stats(v):
    v = sorted(v)
    if not v: return "n=0"
    q = lambda f: v[min(len(v) - 1, int(f * len(v)))]
    return "n=%d median %.2f p10 %.2f p90 %.2f" % (len(v), q(.5), q(.1), q(.9))

if __name__ == "__main__":
    n = int(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    side = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    tids = [t for t, r in mipscore.pairs().items() if r["kind"] == "pair" and max(r["w"], r["h"]) <= side]
    random.Random(seed).shuffle(tids); tids = tids[:n]
    with Pool(int(os.environ.get("JOBS", "6"))) as p:
        res = p.map(job, tids, chunksize=1)
    json.dump(res, open(os.path.join(os.path.dirname(SCR), "%s_result.json" % os.path.splitext(os.path.basename(__file__))[0]), "w"))
    ok = [r for r in res if "psnr" in r]
    err = [r for r in res if "error" in r]
    print("textures %d, decode errors %d" % (len(res), len(err)))
    for e in err[:5]: print("  ", e)
    print("all:                 ", stats([r["psnr"] for r in ok]))
    print("flat baseline:       ", stats([r["flat"] for r in ok]))
    clean = [r for r in ok if r["unfitted"] == 0]
    dirty = [r for r in ok if r["unfitted"] > 0]
    print("no unfitted slot:    ", stats([r["psnr"] for r in clean]))
    print("uses unfitted slot:  ", stats([r["psnr"] for r in dirty]))
    for name, grp in (("all", ok), ("clean", clean), ("dirty", dirty)):
        bad = sum(1 for r in grp if r["psnr"] < r["flat"] + 3)
        print("%-6s fail (< flat + 3 dB): %d / %d; below 25 dB: %d" % (name, bad, len(grp), sum(r["psnr"] < 25 for r in grp)))
