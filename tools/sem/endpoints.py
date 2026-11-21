"""Endpoint reconstruction hypotheses for BC1 BCPack lanes 2-7 and an index-free scorer.

Lanes 2..7 (widths 5,6,5,5,6,5) are taken as R0 G0 B0 R1 G1 B1 residuals.

A hypothesis = (residual map, predictor, block order).  It yields c0, c1 (RGB565 fields) per
block.  Index-free score: in 4-colour BC1 every palette entry lies on the segment c0-c1 in RGB,
so the block's mean colour (~ the 2x2 mip-1 mean) must lie near that segment.  We report the
median distance (0..255 RGB units) from the mip-1 target to the segment, plus the fraction of
blocks within 12 units.  Controls: same endpoints with targets shuffled across blocks.
"""
from __future__ import annotations

import numpy as np

W = np.array([5, 6, 5, 5, 6, 5])
MOD = 1 << W


# ---------------------------------------------------------------- residual maps
def unzig_a(s):   # 0,1,2,3,4 -> 0,-1,+1,-2,+2
    return np.where(s & 1, -((s + 1) >> 1), s >> 1)


def unzig_b(s):   # 0,1,2,3,4 -> 0,+1,-1,+2,-2
    return np.where(s & 1, (s + 1) >> 1, -(s >> 1))


def twos(s, w):   # two's complement at field width
    return np.where(s >= (1 << (w - 1)), s - (1 << w), s)


def signmag_lsb(s):  # LSB = sign, rest magnitude
    return np.where(s & 1, -(s >> 1), s >> 1)


def plain(s):
    return s


RESID = {
    "zigA": lambda s, w: unzig_a(s),
    "zigB": lambda s, w: unzig_b(s),
    "twos": lambda s, w: twos(s, w),
    "raw": lambda s, w: s,
}


# ---------------------------------------------------------------- predictors
def decode(syms, xy, resid="zigA", pred="prev", e1="own", wrap=True, order_xy=None):
    """syms: 12 x n.  Returns E (n x 6) endpoint fields (R0 G0 B0 R1 G1 B1).

    pred: 'none'  absolute (residual = value)
          'prev'  previous block in stream order
          'left' / 'up'  2D neighbour (xy coordinates; missing -> previous-in-stream / 0)
          'med'   LOCO-I median of left, up, up-left
          'avg'   (left + up) / 2
    e1:   'own'   endpoint 1 predicted like endpoint 0 from its own history
          'e0'    endpoint 1 predicted from this block's endpoint 0
    """
    n = syms.shape[1]
    R = np.stack([RESID[resid](syms[2 + k].astype(np.int64), W[k]) for k in range(6)], 1)
    E = np.zeros((n, 6), np.int64)
    if pred == "none":
        E = R.copy()
        if e1 == "e0":
            E[:, 3:] = E[:, :3] + R[:, 3:]
        return E % MOD if wrap else np.clip(E, 0, MOD - 1)
    xy = np.asarray(xy)
    if pred != "prev" and not pred.startswith("ref"):
        mx, my = xy[:, 0].max() + 1, xy[:, 1].max() + 1
        grid = -np.ones((my + 1, mx + 1), np.int64)
    prevE = np.zeros(6, np.int64)
    zero = np.zeros(6, np.int64)
    l0 = syms[0]
    for i in range(n):
        if pred == "prev":
            P = prevE
        elif pred.startswith("ref"):
            # lane 0 = back-reference: predictor = endpoints of block i - (l0 >> 1)
            k = l0[i] >> 1
            P = E[i - k] if 0 < k <= i else zero
        else:
            x, y = xy[i]
            def g(xx, yy):
                if xx < 0 or yy < 0:
                    return None
                j = grid[yy, xx]
                return None if j < 0 else E[j]
            Lf, Up, UL = g(x - 1, y), g(x, y - 1), g(x - 1, y - 1)
            if pred == "left":
                P = Lf if Lf is not None else (Up if Up is not None else prevE)
            elif pred == "up":
                P = Up if Up is not None else (Lf if Lf is not None else prevE)
            else:
                if Lf is None or Up is None:
                    P = Lf if Lf is not None else (Up if Up is not None else prevE)
                elif pred == "avg":
                    P = (Lf + Up) >> 1
                else:  # med
                    c = UL if UL is not None else Lf
                    mn, mxv = np.minimum(Lf, Up), np.maximum(Lf, Up)
                    P = np.where(c >= mxv, mn, np.where(c <= mn, mxv, Lf + Up - c))
        e = np.empty(6, np.int64)
        e[:3] = P[:3] + R[i, :3]
        if e1 == "e0":
            e[3:] = e[:3] + R[i, 3:]
        else:
            e[3:] = P[3:] + R[i, 3:]
        e = e % MOD if wrap else np.clip(e, 0, MOD - 1)
        E[i] = e
        prevE = e
        if pred != "prev" and not pred.startswith("ref"):
            grid[xy[i, 1], xy[i, 0]] = i
    return E


def to888(E):
    E = np.asarray(E)
    r0, g0, b0, r1, g1, b1 = (E[:, k] for k in range(6))
    f5 = lambda v: (v << 3) | (v >> 2)
    f6 = lambda v: (v << 2) | (v >> 4)
    c0 = np.stack([f5(r0), f6(g0), f5(b0)], 1).astype(float)
    c1 = np.stack([f5(r1), f6(g1), f5(b1)], 1).astype(float)
    return c0, c1


def seg_dist(c0, c1, t):
    d = c1 - c0
    dd = (d * d).sum(1)
    u = np.where(dd > 0, ((t - c0) * d).sum(1) / np.maximum(dd, 1e-9), 0.0)
    u = np.clip(u, 0, 1)
    p = c0 + u[:, None] * d
    return np.sqrt(((t - p) ** 2).sum(1))


def score(E, tgt, ok, rng=None):
    c0, c1 = to888(E)
    d = seg_dist(c0[ok], c1[ok], tgt[ok])
    rng = rng or np.random.default_rng(0)
    perm = rng.permutation(ok.sum())
    dctl = seg_dist(c0[ok], c1[ok], tgt[ok][perm])
    return dict(med=float(np.median(d)), f12=float((d < 12).mean()),
                med_ctl=float(np.median(dctl)), f12_ctl=float((dctl < 12).mean()))


def back_distance(v: int) -> int:
    """Lane 0 symbol -> back-reference distance: unzigzag(v) mod 256.
    even v = 2k -> k (1..127); odd v = 2k+1 -> 255-k (255..128); v = 0 -> 0."""
    return (v >> 1) if not (v & 1) else 255 - (v >> 1)


def fields_of(blocks: bytes):
    """BC1 block bytes -> list of endpoint field tuples (R0 G0 B0 R1 G1 B1)."""
    import struct
    out = []
    for o in range(0, len(blocks) - 7, 8):
        c0, c1 = struct.unpack_from("<HH", blocks, o)
        out.append(((c0 >> 11) & 31, (c0 >> 5) & 63, c0 & 31, (c1 >> 11) & 31, (c1 >> 5) & 63, c1 & 31))
    return out


def decode_window(syms, resid="zigA", zero_mode="zero", before="zero", history=None):
    """Back-reference decoder (pure Python ints).

    Endpoint fields of block i = (fields of block i - d) + residual (mod 2^width), with
    d = back_distance(lane0[i]).  zero_mode: what d == 0 means: 'prev' (d = 1), 'zero'
    (predict from 0), 'ring' (d = 256).  A reference before block 0 reads `history` (endpoint
    tuples of the BC1 blocks that precede the chunk in the output stream, oldest first) or 0."""
    n = syms.shape[1]
    lanes = [syms[2 + k].tolist() for k in range(6)]
    l0 = syms[0].tolist()
    mods = [int(m) for m in MOD]
    half = [m // 2 for m in mods]
    E = [None] * n
    z = (0, 0, 0, 0, 0, 0)
    for i in range(n):
        d = back_distance(l0[i])
        if d == 0:
            d = {"prev": 1, "zero": 0, "ring": 256}[zero_mode]
        if d == 0:
            P = z
        elif d <= i:
            P = E[i - d]
        elif history is not None and d - i <= len(history):
            P = history[i - d]
        else:
            P = z
        e = []
        for k in range(6):
            v = lanes[k][i]
            if resid == "zigA":
                r = -((v + 1) >> 1) if v & 1 else v >> 1
            elif resid == "zigB":
                r = (v + 1) >> 1 if v & 1 else -(v >> 1)
            elif resid == "twos":
                r = v - mods[k] if v >= half[k] else v
            else:
                r = v
            e.append((P[k] + r) % mods[k])
        E[i] = tuple(e)
    return np.array(E, dtype=np.int64)


def load_window(path="/tmp/bcpack-shared/window_table.json", min_textures=1):
    """Initial window (256 slots, R0 G0 B0 R1 G1 B1) from window_fit.py output; unfitted slots = 0.
    history[slot] is what a pre-start reference to ring slot (i - d) mod 256 reads.
    Accepts the current format ({slot: {value, textures, ...}}) and the older one ({slot: {joint}})."""
    import json
    W = [(0, 0, 0, 0, 0, 0)] * 256
    for pth in (path, "/tmp/bcpack-shared/window_fit.json"):
        try:
            with open(pth) as f:
                d = json.load(f)
        except FileNotFoundError:
            continue
        d = d.get("slots", d)
        for k, v in d.items():
            if v.get("textures", 99) >= min_textures:
                W[int(k)] = tuple(v.get("value") or v["joint"])
        break
    return W
