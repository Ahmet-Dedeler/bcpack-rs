"""Minimal RDS5 (ttpf) chunk reader for BCPack probing.

Chunk table per SixNG notes (formats/05-ttpf.md):
  entry (a, b) at 0x18 + 8*i
  file offset  = (a & 0xFFFFFF) << 8
  codec        = (a >> 24) & 0x3F       (0x04 BC1, 0x1C BC7, 0x20 zlib)
  virtual off  = 0x1000 + (b & 0xFFFFF) * 0x4000 + (a >> 30) * 0x1000
A chunk runs to the next entry in both spaces. Stored chunks are zero padded
to 256 bytes, so the stored length is an upper bound on the stream length.

Reads only local files given on the command line; embeds no game data.
"""
import mmap
import struct
import sys

BLOCK_BYTES = {0x04: 8, 0x0C: 16, 0x10: 8, 0x14: 16, 0x18: 16, 0x1C: 16}


def iter_chunks(path, codecs=(0x04,)):
    """Yield (index, codec, file_offset, stored_bytes, blocks) for chosen codecs."""
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        assert mm[:4] == b"RDS5", path
        n = struct.unpack_from("<I", mm, 4)[0]
        ents = []
        for i in range(n):
            a, b = struct.unpack_from("<II", mm, 0x18 + 8 * i)
            ents.append(((a & 0xFFFFFF) << 8,
                         0x1000 + (b & 0xFFFFF) * 0x4000 + (a >> 30) * 0x1000,
                         (a >> 24) & 0x3F, a, b))
        for i in range(n - 1):
            off, voff, codec, a, b = ents[i]
            if a == 0 and b == 0:
                continue
            if codec not in codecs:
                continue
            noff, nvoff = ents[i + 1][0], ents[i + 1][1]
            if noff <= off:
                continue
            blocks = (nvoff - voff) // BLOCK_BYTES[codec]
            yield i, codec, mm, off, noff - off, blocks


def stream_len(mm, off, size):
    """Stored length with trailing zero padding stripped (lower bound on true length)."""
    end = off + size
    while end > off and mm[end - 1] == 0:
        end -= 1
    return end - off


if __name__ == "__main__":
    for p in sys.argv[1:]:
        for i, codec, mm, off, size, blocks in iter_chunks(p):
            print(i, hex(codec), off, size, stream_len(mm, off, size), blocks)
