"""Mip-consistency scorer (layers 3-4): decode a candidate mip 0, box-downsample 2x,
compare with the plain (zlib/stored) BC1 mip 1 of the same texture.

    python3 -I tools/bench/mipscore.py index        # -> /tmp/bcpack-shared/mip_pairs.jsonl
    python3 -I tools/bench/mipscore.py controls     # positive + negative controls
    python3 -I tools/bench/mipscore.py score TEXTURE_ID FILE --order stream|gfx10|linear
    python3 -I tools/bench/mipscore.py map TEXTURE_ID     # chunks that make up mip 0

Library:
    from mipscore import score
    score(texture_id, candidate_bytes, block_order="stream")

texture_id = "<store relpath>:t<texture index>" (ETPF descriptor index).

block_order:
  'gfx10'  candidate = mip 0 exactly as stored in the virtual stream (Gfx10 tiled,
           padded to whole tiles: len = mip0_len in mip_pairs.jsonl)
  'stream' candidate = the full decoded virtual bytes of the BCPack chunk(s) that
           cover mip 0, concatenated in chunk order (what a chunk decoder outputs);
           the mip 0 slice is cut out and treated as 'gfx10'
  'linear' candidate = ceil(w/4) x ceil(h/4) BC1 blocks, row-major

Result: psnr (dB, RGB, 8-bit), mse, pixels compared, plus psnr_flat: the score of
predicting mip 1 by its own mean colour.  A candidate must beat psnr_flat clearly
to mean anything (flat normal maps make even a constant guess score well).

Detiling is a port of paryzius ttpf-viewer (src/tile.rs, src/layout.rs); see ttpf.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttpf  # noqa: E402

SHARED = "/tmp/bcpack-shared"
SCRATCH = os.environ.get("BCPACK_SCRATCH", "/tmp/bcpack-harness")
PAIRS = os.path.join(SHARED, "mip_pairs.jsonl")
BC1_DXGI = (70, 71, 72)
PLAIN = (0x20, 0x00)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Index of textures
# ---------------------------------------------------------------------------
def texture_levels(st: ttpf.Store, t: ttpf.Texture):
    if t.dxgi not in BC1_DXGI or t.array_size != 1 or t.tile_mode > 3:
        return None
    mips = t.mip_field + 1
    ch = ttpf.mip_chain(8, t.tile_mode, t.width, t.height, mips)
    return ch


def index_store(rel: str):
    st = ttpf.Store(rel)
    out = []
    starts = sorted(x.vstart for x in st.textures)
    import bisect
    for t in st.textures:
        ch = texture_levels(st, t)
        if not ch:
            continue
        levels, size = ch
        nxt_i = bisect.bisect_right(starts, t.vstart)
        end = starts[nxt_i] if nxt_i < len(starts) else st.vsize
        lv = []
        try:
            for k, p in enumerate(levels):
                a = t.vstart + p.off
                loc = st.locate(a, a + p.len)
                codecs = sorted({st.chunks[i].codec for i, _, _ in loc})
                lv.append({"k": k, "w": p.w, "h": p.h, "off": a, "len": p.len,
                           "chunks": [[f"{rel}:{i}", f, e] for i, f, e in loc], "codecs": codecs})
        except ValueError:
            continue
        if len(lv) < 2:
            continue
        m0, m1 = lv[0], lv[1]
        m0_bc1 = m0["codecs"] == [0x04]
        m0_plain = all(c in PLAIN for c in m0["codecs"])
        m1_plain = all(c in PLAIN for c in m1["codecs"])
        kind = "pair" if (m0_bc1 and m1_plain) else "control" if (m0_plain and m1_plain) else None
        if kind is None:
            continue
        exact = (len(m0["chunks"]) == 1 and m0["chunks"][0][1] == 0 and
                 m0["chunks"][0][2] == st.chunks[int(m0["chunks"][0][0].rpartition(":")[2])].vlen)
        out.append({"texture_id": f"{rel}:t{t.index}", "kind": kind, "store": rel, "index": t.index,
                    "hash": f"{t.hash:016x}", "w": t.width, "h": t.height, "mips": len(levels),
                    "tile_mode": t.tile_mode, "dxgi": t.dxgi, "vstart": t.vstart,
                    "chain_size": size, "fits_before_next": t.vstart + size <= end,
                    "mip0_off": m0["off"], "mip0_len": m0["len"], "mip0_chunks": m0["chunks"],
                    "mip0_exact_chunk": exact,
                    "mip1_off": m1["off"], "mip1_len": m1["len"], "mip1_chunks": m1["chunks"]})
    st.close()
    return out


def build_index(jobs: int):
    rows = []
    with Pool(jobs) as pool:
        for r in pool.imap_unordered(index_store, ttpf.list_stores()):
            rows += r
    rows.sort(key=lambda r: (r["store"], r["index"]))
    with open(PAIRS + ".tmp", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    os.replace(PAIRS + ".tmp", PAIRS)
    c = Counter((r["kind"], r["mip0_exact_chunk"] if r["kind"] == "pair" else None) for r in rows)
    print(json.dumps({str(k): v for k, v in c.items()}, indent=1))
    return rows


_PAIRS: Optional[Dict[str, dict]] = None


def pairs() -> Dict[str, dict]:
    global _PAIRS
    if _PAIRS is None:
        _PAIRS = {}
        with open(PAIRS) as fh:
            for line in fh:
                r = json.loads(line)
                _PAIRS[r["texture_id"]] = r
    return _PAIRS


def get_tex(texture_id: str):
    rel, _, t = texture_id.rpartition(":")
    st = ttpf.open_store(rel)
    tex = st.textures[int(t[1:])]
    levels, _ = texture_levels(st, tex)
    return st, tex, levels


# ---------------------------------------------------------------------------
# Block grids
# ---------------------------------------------------------------------------
def detile_blocks(data: bytes, place: ttpf.Place, tile_mode: int) -> Tuple[bytes, int, int]:
    """Stored (tiled) level bytes -> row-major BC1 blocks of the level (cropped)."""
    pat = ttpf.PATTERNS[(ttpf.tile_kind(tile_mode), 8)]
    pwb, phb = place.pw // 4, place.ph // 4
    n = len(data) // 8
    order = ttpf.untile_order(n, pwb, phb, pat)
    lin = bytearray(pwb * phb * 8)
    for i, d in enumerate(order):
        if d >= 0:
            lin[d * 8:d * 8 + 8] = data[i * 8:i * 8 + 8]
    bw, bh = -(-place.w // 4), -(-place.h // 4)
    ox, oy = place.ox // 4, place.oy // 4
    out = bytearray()
    for y in range(oy, oy + bh):
        out += lin[(y * pwb + ox) * 8:(y * pwb + ox + bw) * 8]
    return bytes(out), bw, bh


def mip1_rgb(texture_id: str) -> Tuple[List[List[Tuple[int, int, int]]], int, int]:
    st, tex, levels = get_tex(texture_id)
    p = levels[1]
    a = tex.vstart + p.off
    data = st.read_virtual(a, a + p.len)
    blocks, bw, bh = detile_blocks(data, p, tex.tile_mode)
    rows = ttpf.bc1_decode_grid(blocks, bw, bh)
    return rows, p.w, p.h


def plain_mip0_bytes(texture_id: str) -> bytes:
    """Stored (gfx10) mip 0 bytes of a texture whose mip 0 is plain (controls)."""
    st, tex, levels = get_tex(texture_id)
    a = tex.vstart + levels[0].off
    return st.read_virtual(a, a + levels[0].len)


def downsampled_mip0(blocks: bytes, bw: int, bh: int):
    """Row-major BC1 blocks -> dict-free 2x box-downsampled grid (2*bh x 2*bw)."""
    W = 2 * bw
    out = [[(0, 0, 0)] * W for _ in range(2 * bh)]
    for by in range(bh):
        r0, r1 = out[2 * by], out[2 * by + 1]
        for bx in range(bw):
            c0, c1, idx = struct.unpack_from("<HHI", blocks, (by * bw + bx) * 8)
            pal = ttpf.bc1_palette(c0, c1)
            for qy in range(2):
                row = r0 if qy == 0 else r1
                for qx in range(2):
                    s0 = s1 = s2 = 0
                    for py in (2 * qy, 2 * qy + 1):
                        for px in (2 * qx, 2 * qx + 1):
                            c = pal[(idx >> (2 * (py * 4 + px))) & 3]
                            s0 += c[0]; s1 += c[1]; s2 += c[2]
                    row[2 * bx + qx] = ((s0 + 2) >> 2, (s1 + 2) >> 2, (s2 + 2) >> 2)
    return out


def mip0_chunk_map(texture_id: str) -> List[dict]:
    """Which chunks make up mip 0, in order.  One entry per chunk piece:

      store, chunk_index, chunk_id, codec
      chunk_bytes  [from, to) inside the chunk's decoded (virtual) bytes
      mip0_bytes   [from, to) inside mip 0 as stored (Gfx10 tiled, padded)
      blocks       [first, end) 8-byte BC1 block indices inside mip 0 in STORED
                   (tiled) order; detile with ``detile_blocks`` / ``assemble_mip0``

    For all 4,809 indexed pairs this is a single piece covering a whole chunk."""
    st, tex, levels = get_tex(texture_id)
    p0 = levels[0]
    a = tex.vstart + p0.off
    out, pos = [], 0
    for i, f, e in st.locate(a, a + p0.len):
        n = e - f
        out.append({"store": st.rel, "chunk_index": i, "chunk_id": f"{st.rel}:{i}",
                    "codec": st.chunks[i].codec, "chunk_bytes": [f, e],
                    "mip0_bytes": [pos, pos + n], "blocks": [pos // 8, (pos + n) // 8]})
        pos += n
    return out


def assemble_mip0(texture_id: str, decoded: Dict[str, bytes]) -> Tuple[bytes, bytes, int, int]:
    """Build mip 0 from decoded chunks.  ``decoded`` maps chunk_id -> that chunk's full
    decoded bytes (vlen); plain (zlib/stored) pieces are read from the store if missing.
    Returns (gfx10_bytes, linear_row_major_blocks, blocks_w, blocks_h)."""
    st, tex, levels = get_tex(texture_id)
    parts = []
    for m in mip0_chunk_map(texture_id):
        d = decoded.get(m["chunk_id"])
        if d is None:
            d = st.chunk_virtual(m["chunk_index"])
        f, e = m["chunk_bytes"]
        parts.append(d[f:e])
    g = b"".join(parts)
    lin, bw, bh = detile_blocks(g, levels[0], tex.tile_mode)
    return g, lin, bw, bh


def write_ppm(path: str, blocks: bytes, bw: int, bh: int):
    """Row-major BC1 blocks -> binary PPM (for eyeballing; keep it out of git)."""
    rows = ttpf.bc1_decode_grid(blocks, bw, bh)
    with open(path, "wb") as fh:
        fh.write(b"P6 %d %d 255\n" % (bw * 4, bh * 4))
        for r in rows:
            fh.write(bytes(v for px in r for v in px))


def psnr_of(mse: float) -> float:
    return 99.0 if mse <= 1e-12 else 10 * math.log10(255 * 255 / mse)


@lru_cache(maxsize=8)
def _mip1_cached(texture_id: str):
    return mip1_rgb(texture_id)


def score(texture_id: str, candidate: bytes, block_order: str = "stream") -> dict:
    """PSNR of 2x-downsampled candidate mip 0 against the plain BC1 mip 1."""
    st, tex, levels = get_tex(texture_id)
    p0 = levels[0]
    if block_order == "stream":
        a = tex.vstart + p0.off
        loc = st.locate(a, a + p0.len)
        need = sum(st.chunks[i].vlen for i, _, _ in loc)
        if len(candidate) != need:
            raise ValueError(f"stream candidate has {len(candidate)} bytes, chunks cover {need}")
        parts, base = [], 0
        for i, f, e in loc:
            parts.append(candidate[base + f:base + e])
            base += st.chunks[i].vlen
        candidate = b"".join(parts)
        block_order = "gfx10"
    if block_order == "gfx10":
        if len(candidate) != p0.len:
            raise ValueError(f"gfx10 candidate has {len(candidate)} bytes, mip 0 is {p0.len}")
        blocks, bw, bh = detile_blocks(candidate, p0, tex.tile_mode)
    elif block_order == "linear":
        bw, bh = -(-p0.w // 4), -(-p0.h // 4)
        if len(candidate) < bw * bh * 8:
            raise ValueError(f"linear candidate has {len(candidate)} bytes, need {bw * bh * 8}")
        blocks = candidate[:bw * bh * 8]
    else:
        raise ValueError(block_order)
    down = downsampled_mip0(blocks, bw, bh)
    ref, w1, h1 = _mip1_cached(texture_id)
    se = 0
    n = 0
    sums = [0, 0, 0]
    for y in range(h1):
        dr, rr = down[y], ref[y]
        for x in range(w1):
            a, b = dr[x], rr[x]
            se += (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
            sums[0] += b[0]; sums[1] += b[1]; sums[2] += b[2]
            n += 1
    mean = [s / n for s in sums]
    sef = sum((rr[x][c] - mean[c]) ** 2 for rr in ref[:h1] for x in range(w1) for c in range(3))
    mse = se / (3 * n)
    return {"texture_id": texture_id, "order": block_order, "pixels": n, "mse": mse,
            "psnr": psnr_of(mse), "psnr_flat": psnr_of(sef / (3 * n))}


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
def _control_one(tid: str) -> dict:
    data = plain_mip0_bytes(tid)
    rng = random.Random(tid)
    out = {"texture_id": tid}
    out["true_gfx10"] = score(tid, data, "gfx10")["psnr"]
    r = score(tid, data, "linear")  # right blocks, wrong order
    out["true_as_linear"] = r["psnr"]
    out["psnr_flat"] = r["psnr_flat"]
    out["random_blocks"] = score(tid, bytes(rng.getrandbits(8) for _ in range(len(data))), "gfx10")["psnr"]
    # same texture's blocks, shuffled block order (keeps the colour statistics)
    blk = [data[i:i + 8] for i in range(0, len(data), 8)]
    rng.shuffle(blk)
    out["shuffled_blocks"] = score(tid, b"".join(blk), "gfx10")["psnr"]
    return out


def _bcpackrs_one(tid: str) -> dict:
    r = pairs()[tid]
    st, tex, levels = get_tex(tid)
    parts = []
    for cid, _, _ in r["mip0_chunks"]:
        info = ttpf.chunk_info(cid)
        os.makedirs(SCRATCH, exist_ok=True)
        fd, path = tempfile.mkstemp(dir=SCRATCH, suffix=".bin")
        outp = path + ".bc1"
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(ttpf.extract(cid))
            p = subprocess.run([os.path.join(REPO, "target/release/bcpack"), "decode-bc1", path, outp,
                                str(info.vlen // 8)], capture_output=True, timeout=300)
            if p.returncode != 0:
                return {"texture_id": tid, "error": p.stderr.decode()[-200:]}
            parts.append(open(outp, "rb").read())
        finally:
            for f in (path, outp):
                if os.path.exists(f):
                    os.unlink(f)
    res = score(tid, b"".join(parts), "stream")
    return {"texture_id": tid, "bcpack_rs": res["psnr"], "psnr_flat": res["psnr_flat"]}


def _stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {}
    q = lambda f: round(xs[min(len(xs) - 1, int(f * len(xs)))], 2)
    return {"n": len(xs), "min": round(xs[0], 2), "p10": q(0.1), "median": q(0.5), "p90": q(0.9),
            "max": round(xs[-1], 2), "mean": round(sum(xs) / len(xs), 2)}


def controls(n: int, max_side: int, jobs: int, seed: int):
    rows = list(pairs().values())
    rng = random.Random(seed)
    ctl = [r["texture_id"] for r in rows if r["kind"] == "control" and max(r["w"], r["h"]) <= max_side]
    prs = [r["texture_id"] for r in rows if r["kind"] == "pair" and max(r["w"], r["h"]) <= max_side]
    ctl = rng.sample(ctl, min(n, len(ctl)))
    prs = rng.sample(prs, min(n, len(prs)))
    with Pool(jobs) as pool:
        c = pool.map(_control_one, ctl, chunksize=1)
        b = pool.map(_bcpackrs_one, prs, chunksize=1)
    rep = {"positive_true_mip0_gfx10": _stats([x["true_gfx10"] for x in c]),
           "true_mip0_read_as_linear": _stats([x["true_as_linear"] for x in c]),
           "negative_random_blocks": _stats([x["random_blocks"] for x in c]),
           "negative_shuffled_true_blocks": _stats([x["shuffled_blocks"] for x in c]),
           "flat_mean_colour_baseline(controls)": _stats([x["psnr_flat"] for x in c]),
           "positive_minus_flat": _stats([x["true_gfx10"] - x["psnr_flat"] for x in c]),
           "positive_below_25dB": sum(1 for x in c if x["true_gfx10"] < 25),
           "negative_bcpack_rs_decode_bc1(pairs)": _stats([x.get("bcpack_rs") for x in b]),
           "flat_mean_colour_baseline(pairs)": _stats([x.get("psnr_flat") for x in b]),
           "bcpack_rs_errors": sum(1 for x in b if "error" in x),
           "max_side": max_side, "seed": seed}
    with open(os.path.join(SCRATCH, "mip_controls.json"), "w") as fh:
        json.dump({"report": rep, "controls": c, "bcpack_rs": b}, fh, indent=1)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("index"); s.add_argument("--jobs", type=int, default=os.cpu_count())
    s = sub.add_parser("controls")
    s.add_argument("--n", type=int, default=200); s.add_argument("--max-side", type=int, default=512)
    s.add_argument("--jobs", type=int, default=os.cpu_count()); s.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("map"); s.add_argument("texture_id")
    s = sub.add_parser("score")
    s.add_argument("texture_id"); s.add_argument("file")
    s.add_argument("--order", default="stream", choices=["stream", "gfx10", "linear"])
    a = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    if a.cmd == "index":
        build_index(a.jobs)
    elif a.cmd == "controls":
        controls(a.n, a.max_side, a.jobs, a.seed)
    elif a.cmd == "map":
        print(json.dumps(mip0_chunk_map(a.texture_id), indent=1))
    else:
        print(json.dumps(score(a.texture_id, open(a.file, "rb").read(), a.order), indent=1))


if __name__ == "__main__":
    main()
