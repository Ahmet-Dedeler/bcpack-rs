"""Render decoded mip 0 (and mip 1 upscaled 2x for comparison) to PNG for eyeballing.
Output goes to $SEM_SCRATCH only; never commit images of game assets.
usage: python3 -I tools/sem/render.py --table T.npy TEXTURE_ID [TEXTURE_ID ...]
"""
import glob
import json
import os
import struct
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH = os.environ.get("BCPACK_BENCH", "/Users/ahmet/Code/bcpack-wt/harness/tools/bench")
sys.path.insert(0, BENCH)
import numpy as np  # noqa: E402
import mipscore  # noqa: E402
import ttpf  # noqa: E402
from psnr import build  # noqa: E402

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")


def write_png(path, rows):
    h, w = len(rows), len(rows[0])
    raw = b"".join(b"\0" + bytes(v for px in r for v in px) for r in rows)
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main():
    T = np.load(sys.argv[sys.argv.index("--table") + 1])[0].astype(int)
    want = [a for a in sys.argv[1:] if ":t" in a]
    idx = {}
    for f in glob.glob(os.path.join(SCR, "syms", "*.npz")):
        z = np.load(f)
        idx[json.loads(str(z["meta"]))["texture_id"]] = f
    for tid in want:
        z = np.load(idx[tid])
        data, ok = build(z["syms"].astype(np.int64), T)
        st, tex, levels = mipscore.get_tex(tid)
        a = tex.vstart + levels[0].off
        loc = st.locate(a, a + levels[0].len)
        (i, f0, e0), = loc
        lin, bw, bh = mipscore.detile_blocks(data[f0:e0], levels[0], tex.tile_mode)
        rows = ttpf.bc1_decode_grid(lin, bw, bh)
        base = tid.replace("/", "_").replace(":", "_")
        write_png(os.path.join(SCR, base + "_mip0_decoded.png"), rows)
        ref, w1, h1 = mipscore.mip1_rgb(tid)
        up = [[ref[y // 2][x // 2] for x in range(2 * w1)] for y in range(2 * h1)]
        write_png(os.path.join(SCR, base + "_mip1_plain_x2.png"), up)
        print(os.path.join(SCR, base + "_mip0_decoded.png"), mipscore.score(tid, data, "stream"))


if __name__ == "__main__":
    main()
