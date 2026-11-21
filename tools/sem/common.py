"""Shared helpers for the endpoint-semantics work (agent sem-endpoints).

* decode_chunk(chunk_id) -> 12 x blocks symbol array (cached in /tmp/bcpack-sem-endpoints/syms)
  using the probe agent's cycle-level MDU simulator (copied here as mdusim.py / mdu_tables.py).
* pair_targets(texture_id) -> per-block target colours from the plain mip 1, in STORED (tiled)
  block order of mip 0, so they line up index-for-index with the BCPack chunk's block stream.

Nothing derived from game data is written into git; caches live in /tmp only.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH = os.environ.get("BCPACK_BENCH", "/Users/ahmet/Code/bcpack-wt/harness/tools/bench")
sys.path.insert(0, BENCH)

import mdusim  # noqa: E402
import ttpf  # noqa: E402
import mipscore  # noqa: E402

SCR = "/tmp/bcpack-sem-endpoints"
BC1_AB = [7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7]
WIDTHS = [8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8]


def decode_chunk(chunk_id: str, blocks: int | None = None) -> np.ndarray:
    os.makedirs(os.path.join(SCR, "syms"), exist_ok=True)
    fn = os.path.join(SCR, "syms", chunk_id.replace("/", "_") + ".npy")
    if os.path.exists(fn):
        return np.load(fn)
    raw = ttpf.extract(chunk_id, strip_pad=False)   # the feeder may prefetch into the pad
    data_len = len(ttpf.extract(chunk_id))
    if blocks is None:
        blocks = ttpf.chunk_info(chunk_id).vlen // 8
    r = mdusim.simulate(raw, BC1_AB, [blocks] * 12, stream_len=len(raw))
    ok = (all(ln["done"] and ln["final"] in mdusim.LEGAL for ln in r["lanes"])
          and all(len(s) == blocks for s in r["syms"]) and r["pos"] == data_len)
    if not ok:
        raise RuntimeError(f"{chunk_id}: no convergence ({r.get('why')})")
    a = np.array(r["syms"], dtype=np.int32)
    np.save(fn, a)
    return a


def pairs(kind="pair"):
    out = []
    with open("/tmp/bcpack-shared/mip_pairs.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["kind"] == kind:
                out.append(d)
    return out


def stored_to_linear(texture_id: str):
    """Return (order, bw, bh, pwb, phb): order[i] = linear index (row-major in the padded
    level, pwb wide) of stored block i, or -1 for padding."""
    st, tex, levels = mipscore.get_tex(texture_id)
    p = levels[0]
    pat = ttpf.PATTERNS[(ttpf.tile_kind(tex.tile_mode), 8)]
    pwb, phb = p.pw // 4, p.ph // 4
    n = p.len // 8
    order = ttpf.untile_order(n, pwb, phb, pat)
    return order, pwb, phb, p, tex


def pair_targets(texture_id: str):
    """Per stored block i of mip 0: target mean RGB (0..255 floats) of the 2x2 mip-1 pixels
    it maps to, plus validity mask and the (x, y) block coordinates."""
    os.makedirs(os.path.join(SCR, "tgt"), exist_ok=True)
    fn = os.path.join(SCR, "tgt", texture_id.replace("/", "_") + ".npz")
    if os.path.exists(fn):
        z = np.load(fn)
        return z["tgt"], z["ok"], z["xy"]
    order, pwb, phb, p, tex = stored_to_linear(texture_id)
    rows, w1, h1 = mipscore.mip1_rgb(texture_id)
    m1 = np.array(rows, dtype=np.float64)  # h1 x w1 x 3 (cropped)
    n = len(order)
    tgt = np.zeros((n, 3))
    ok = np.zeros(n, bool)
    xy = np.full((n, 2), -1, np.int32)
    ox, oy = p.ox // 4, p.oy // 4
    for i, d in enumerate(order):
        if d < 0:
            continue
        x, y = d % pwb - ox, d // pwb - oy
        xy[i] = (x, y)
        if 0 <= 2 * x + 1 < m1.shape[1] and 0 <= 2 * y + 1 < m1.shape[0]:
            tgt[i] = m1[2 * y:2 * y + 2, 2 * x:2 * x + 2].reshape(4, 3).mean(0)
            ok[i] = True
    np.savez(fn, tgt=tgt, ok=ok, xy=xy)
    return tgt, ok, xy


def rgb565(c):
    c = np.asarray(c)
    r = (c >> 11) & 31
    g = (c >> 5) & 63
    b = c & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1)
