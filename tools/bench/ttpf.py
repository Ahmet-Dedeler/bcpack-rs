"""RDS5 (.ttpf) texture-store reader shared by the BCPack bench tools.

Python 3 stdlib only; safe to run with ``python3 -I``.

Chunk table (u64 per entry, little-endian, at 0x18; n entries, last one is an end
sentinel).  Two public descriptions exist and agree once aligned:

* SixNG formats/05-ttpf.md: entry i = (u32 a, u32 b) describes the chunk that
  STARTS at file offset ``(a & 0xFFFFFF) << 8`` with codec ``(a >> 24) & 0x3F``.
* paryzius ttpf-viewer: entry i holds where block i ENDS; bits 26-29 give the
  codec of the NEXT block; the first block (ETPF index, stored) starts at the
  ETPF position.

We number chunks the SixNG way (chunk i starts at entry i, runs to entry i+1),
use the 26-bit file offset (bits 0-25) and the viewer's virtual offsets
``((a >> 30) | (b & 0xFFFFF) << 2) << 12`` WITHOUT SixNG's +0x1000 bias: with
this convention the last entry's virtual offset equals the header's virtual
size and ETPF texture starts are ``(d2 & 0x1FFFFF) << 12`` directly.
``b >> 20`` (12 bits) is the number of zero pad bytes at the end of the
chunk's stored bytes (checked on the corpus, see docs/findings/harness.md).

Codec ids are SixNG's 6-bit value: 0x20 zlib, 0x00 stored, 0x04 BC1 BCPack,
0x0C BC3, 0x10 BC4, 0x14 BC5, 0x18 BC6H, 0x1C BC7 (BCPack id = BC number * 4).
"""
from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

def _game_root() -> str:
    """Game install root: $BCPACK_GAME_ROOT, else the path stored (outside git) in
    /tmp/bcpack-shared/game_root.txt."""
    env = os.environ.get("BCPACK_GAME_ROOT")
    if env:
        return env
    try:
        with open("/tmp/bcpack-shared/game_root.txt") as fh:
            return fh.read().strip()
    except OSError:
        return ""


GAME_ROOT = _game_root()

CODEC_NAMES = {0x00: "stored", 0x20: "zlib", 0x04: "BC1", 0x08: "BC2", 0x0C: "BC3",
               0x10: "BC4", 0x14: "BC5", 0x18: "BC6H", 0x1C: "BC7"}
BCPACK_CODECS = (0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C)


def list_stores(root: str = GAME_ROOT) -> List[str]:
    """Relative paths of every .ttpf store under data/ttdb and pack/*/data/ttdb."""
    out = []
    for base in [os.path.join(root, "data", "ttdb")] + sorted(
            os.path.join(root, "pack", p, "data", "ttdb")
            for p in (os.listdir(os.path.join(root, "pack")) if os.path.isdir(os.path.join(root, "pack")) else [])):
        for dp, _dn, fn in os.walk(base):
            for f in fn:
                if f.lower().endswith(".ttpf"):
                    out.append(os.path.relpath(os.path.join(dp, f), root))
    return sorted(out)


@dataclass
class Chunk:
    index: int        # chunk number = chunk-table entry index
    file_off: int
    file_len: int     # stored bytes incl. trailing pad
    pad: int          # b >> 20: trailing zero pad bytes
    vstart: int       # virtual (unpacked) offset
    vlen: int         # virtual (unpacked) size
    codec: int

    @property
    def data_len(self) -> int:
        """Stored length minus trailing pad."""
        return self.file_len - self.pad


@dataclass
class Texture:
    index: int
    hash: int
    width: int
    height: int
    mip_field: int    # d0 bits 28-31 (mips - 1 on plain textures)
    dxgi: int
    flags: int        # d1 bits 0-2
    array_size: int
    tile_mode: int
    vstart: int
    d1: int
    d3: int


class Store:
    """One RDS5 store: chunk table + ETPF texture index.  Reads on demand."""

    def __init__(self, rel: str, root: str = GAME_ROOT):
        self.rel = rel
        self.path = os.path.join(root, rel)
        self.fd = os.open(self.path, os.O_RDONLY)
        self.size = os.fstat(self.fd).st_size
        head = os.pread(self.fd, 0x18, 0)
        if head[:4] != b"RDS5":
            raise ValueError(f"{rel}: not RDS5")
        n, vsize_lo, x0c, etpf256 = struct.unpack_from("<IIII", head, 4)
        self.n_entries = n
        self.vsize = vsize_lo | ((x0c & 0xFF) << 32)
        self.etpf_off = etpf256 << 8
        tab = os.pread(self.fd, 8 * n, 0x18)
        ents = [struct.unpack_from("<II", tab, 8 * i) for i in range(n)]
        self.chunks: List[Chunk] = []
        for i in range(n - 1):
            a, b = ents[i]
            a2, b2 = ents[i + 1]
            off = (a & 0x3FFFFFF) << 8
            end = (a2 & 0x3FFFFFF) << 8
            v = ((a >> 30) | ((b & 0xFFFFF) << 2)) << 12
            v2 = ((a2 >> 30) | ((b2 & 0xFFFFF) << 2)) << 12
            self.chunks.append(Chunk(i, off, end - off, b >> 20, v, v2 - v, (a >> 24) & 0x3F))
        self._vstarts = [c.vstart for c in self.chunks]
        self._textures: Optional[List[Texture]] = None

    def close(self):
        os.close(self.fd)

    # ---- chunks -------------------------------------------------------------
    def read(self, off: int, n: int) -> bytes:
        return os.pread(self.fd, n, off)

    def chunk_bytes(self, i: int, strip_pad: bool = True) -> bytes:
        c = self.chunks[i]
        return self.read(c.file_off, c.data_len if strip_pad else c.file_len)

    def chunk_virtual(self, i: int) -> bytes:
        """Unpacked bytes of a zlib/stored chunk (raises for BCPack)."""
        c = self.chunks[i]
        if c.codec == 0x20:
            out = zlib.decompress(self.read(c.file_off, c.file_len))
        elif c.codec == 0x00:
            out = self.read(c.file_off, c.vlen)
        else:
            raise ValueError(f"chunk {i} codec {c.codec:#x} is not plain")
        if len(out) < c.vlen:
            raise ValueError(f"chunk {i}: inflated {len(out)} < vlen {c.vlen}")
        return out[:c.vlen]

    def locate(self, a: int, b: int) -> List[Tuple[int, int, int]]:
        """Chunks covering virtual range [a, b): list of (chunk, from, to) relative to chunk start."""
        import bisect
        out = []
        pos = a
        while pos < b:
            i = bisect.bisect_right(self._vstarts, pos) - 1
            if i < 0 or i >= len(self.chunks):
                raise ValueError("virtual range outside chunk table")
            c = self.chunks[i]
            if c.vlen <= 0:
                raise ValueError("empty chunk in range")
            end = min(b, c.vstart + c.vlen)
            out.append((i, pos - c.vstart, end - c.vstart))
            pos = end
        return out

    def read_virtual(self, a: int, b: int, cache: Optional[Dict[int, bytes]] = None) -> bytes:
        parts = []
        for i, f, t in self.locate(a, b):
            if cache is not None and i in cache:
                d = cache[i]
            else:
                d = self.chunk_virtual(i)
                if cache is not None:
                    cache[i] = d
            parts.append(d[f:t])
        return b"".join(parts)

    # ---- textures -----------------------------------------------------------
    @property
    def textures(self) -> List[Texture]:
        if self._textures is None:
            at = self.etpf_off
            hdr = self.read(at, 16)
            if hdr[:4] != b"ETPF":
                raise ValueError(f"{self.rel}: no ETPF at {at:#x}")
            t = struct.unpack_from("<I", hdr, 8)[0]
            desc = self.read(at + 16, 16 * t)
            hashes = self.read(at + 16 + 16 * t, 8 * t)
            out = []
            for i in range(t):
                d0, d1, d2, d3 = struct.unpack_from("<IIII", desc, 16 * i)
                out.append(Texture(
                    index=i, hash=struct.unpack_from("<Q", hashes, 8 * i)[0],
                    width=(d0 & 0x3FFF) + 1, height=((d0 >> 14) & 0x3FFF) + 1,
                    mip_field=d0 >> 28, dxgi=(d1 & 0x3FF) >> 3, flags=d1 & 7,
                    array_size=((d1 >> 10) & 0x3FF) + 1, tile_mode=(d1 >> 24) & 0xF,
                    vstart=(d2 & 0x1FFFFF) << 12, d1=d1, d3=d3))
            self._textures = out
        return self._textures


@lru_cache(maxsize=64)
def open_store(rel: str, root: str = GAME_ROOT) -> Store:
    return Store(rel, root)


def parse_chunk_id(chunk_id: str) -> Tuple[str, int]:
    rel, _, idx = chunk_id.rpartition(":")
    return rel, int(idx)


def extract(chunk_id: str, strip_pad: bool = True, root: str = GAME_ROOT) -> bytes:
    """Stored bytes of a chunk, e.g. extract('data/ttdb/graphics/ttdb_graphics.ttpf:1547').

    Never write the result into a git repository."""
    rel, idx = parse_chunk_id(chunk_id)
    return open_store(rel, root).chunk_bytes(idx, strip_pad)


def chunk_info(chunk_id: str, root: str = GAME_ROOT) -> Chunk:
    rel, idx = parse_chunk_id(chunk_id)
    return open_store(rel, root).chunks[idx]


# ---------------------------------------------------------------------------
# Gfx10 tiling and mip chains (port of paryzius ttpf-viewer src/tile.rs + layout.rs)
# ---------------------------------------------------------------------------
PATTERNS = {  # (tile, element bytes) -> bit order, lowest bit first
    ("64K", 16): "xyxyyxyxyxyx", ("4K", 16): "yyxxyxyx",
    ("64K", 8): "xyxxyyxyxyxyx", ("4K", 8): "xyyxxyxyx",
}


def tile_kind(tile_mode: int) -> str:
    return "64K" if tile_mode == 0 else "4K"


@lru_cache(maxsize=64)
def _tile_lut(pattern: str) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int]:
    n = len(pattern)
    lx, ly = [], []
    for j in range(1 << n):
        x = y = xi = yi = 0
        for b, c in enumerate(pattern):
            bit = (j >> b) & 1
            if c == "x":
                x |= bit << xi; xi += 1
            else:
                y |= bit << yi; yi += 1
        lx.append(x); ly.append(y)
    xs = pattern.count("x")
    return tuple(lx), tuple(ly), 1 << xs, 1 << (n - xs)


def untile_order(n_elems: int, w: int, h: int, pattern: str) -> List[int]:
    """For each stored element index i (< n_elems), its linear index y*w+x in a w x h
    element grid, or -1 if it falls outside.  Same maths as ttpf-viewer untile()."""
    n = len(pattern)
    t = pattern[:min(n, max(n_elems, 1).bit_length() - 1)]
    lx, ly, tw, th = _tile_lut(t)
    tl = len(t)
    mask = (1 << tl) - 1
    tpr = max(1, -(-w // tw))
    out = []
    for i in range(n_elems):
        k = i >> tl
        x = (k % tpr) * tw + lx[i & mask]
        y = (k // tpr) * th + ly[i & mask]
        out.append(y * w + x if (x < w and y < h) else -1)
    return out


@dataclass
class Place:
    w: int; h: int          # level size in pixels
    off: int; len: int      # bytes from image start, bytes to read
    pw: int; ph: int        # decoded region (whole tiles), pixels
    ox: int; oy: int        # level origin inside region, pixels


def mip_chain(elem_bytes: int, tile_mode: int, w: int, h: int, mips: int):
    """Gfx10 mip chain of a plain BCn image (4x4 blocks).  Returns (levels largest
    first, total size) or None.  Port of ttpf-viewer layout::chain."""
    tile = tile_kind(tile_mode)
    pat = PATTERNS.get((tile, elem_bytes))
    if not pat or mips <= 0:
        return None
    _, _, bw, bh = _tile_lut(pat)  # tile size in elements
    tw, th = bw * 4, bh * 4
    tile_bytes = bw * bh * elem_bytes
    max_in_tail = 12 if tile == "64K" else 8
    px = lambda i: (max(w >> i, 1), max(h >> i, 1))
    el = lambda i: (-(-px(i)[0] // 4), -(-px(i)[1] // 4))
    tail = mips
    if mips > 1:
        for i in range(mips):
            if el(i)[0] <= bw // 2 and el(i)[1] <= bh and mips - i <= max_in_tail:
                tail = i
                break
    mw, mh = (4, 4) if elem_bytes == 16 else (8, 4)
    levels: List[Optional[Place]] = [None] * mips
    pos = 0
    if tail < mips:
        pos = tile_bytes
        for i in range(tail, mips):
            slot = max_in_tail - 1 - (i - tail)
            at = (16 << slot) if slot > 6 else (slot << 8)
            morton = at >> 8
            mx = my = 0
            for bit in range(8):
                my |= ((morton >> (2 * bit)) & 1) << bit
                mx |= ((morton >> (2 * bit + 1)) & 1) << bit
            ox, oy = mx * mw, my * mh
            if ox + el(i)[0] > bw or oy + el(i)[1] > bh:
                return None
            lw, lh = px(i)
            levels[i] = Place(lw, lh, 0, tile_bytes, tw, th, ox * 4, oy * 4)
    for i in range(tail - 1, -1, -1):
        ew, eh = el(i)
        nx, ny = -(-ew // bw), -(-eh // bh)
        ln = nx * ny * tile_bytes
        lw, lh = px(i)
        levels[i] = Place(lw, lh, pos, ln, nx * tw, ny * th, 0, 0)
        pos += ln
    return levels, pos


# ---------------------------------------------------------------------------
# BC1 decode
# ---------------------------------------------------------------------------
def _c565(c: int) -> Tuple[int, int, int]:
    r = (c >> 11) & 31; g = (c >> 5) & 63; b = c & 31
    return ((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2))


@lru_cache(maxsize=1 << 16)
def bc1_palette(c0: int, c1: int) -> Tuple[Tuple[int, int, int], ...]:
    a, b = _c565(c0), _c565(c1)
    if c0 > c1:
        p2 = tuple((2 * x + y + 1) // 3 for x, y in zip(a, b))
        p3 = tuple((x + 2 * y + 1) // 3 for x, y in zip(a, b))
    else:
        p2 = tuple((x + y) // 2 for x, y in zip(a, b))
        p3 = (0, 0, 0)
    return (a, b, p2, p3)


def bc1_decode_grid(blocks: bytes, bw: int, bh: int) -> List[List[Tuple[int, int, int]]]:
    """Decode bw*bh row-major BC1 blocks to a (4*bh) x (4*bw) grid of RGB tuples."""
    W = bw * 4
    rows = [[(0, 0, 0)] * W for _ in range(bh * 4)]
    for by in range(bh):
        for bx in range(bw):
            o = (by * bw + bx) * 8
            c0, c1, idx = struct.unpack_from("<HHI", blocks, o)
            pal = bc1_palette(c0, c1)
            for py in range(4):
                row = rows[by * 4 + py]
                for px_ in range(4):
                    row[bx * 4 + px_] = pal[(idx >> (2 * (py * 4 + px_))) & 3]
    return rows
