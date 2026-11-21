//! RDS5 (`.ttpf`) texture store reader, Gfx10 mip layout and detiling, BC1
//! pixel decode, and DDS / PNG writers.
//!
//! Layout as documented by the SixNG format notes and paryzius' ttpf-viewer
//! (whose Gfx10 tiling and mip-tail code this follows):
//!
//! * header: `RDS5`, u32 entry count n, u32 virtual size low, u32 (bits 0-7 =
//!   virtual size high), u32 ETPF offset / 256; chunk table of n u64 entries at 0x18;
//! * entry i = (u32 a, u32 b): chunk i starts at file offset `(a & 0x3FFFFFF) << 8`,
//!   codec `(a >> 24) & 0x3F`, virtual offset `((a >> 30) | (b & 0xFFFFF) << 2) << 12`,
//!   `b >> 20` trailing zero pad bytes; it runs to entry i + 1 in both spaces;
//! * codecs: 0x00 stored, 0x20 zlib, 0x04 BC1 BCPack (0x08..0x1C other BCn BCPack);
//! * ETPF index: `ETPF`, count at +8, then 16-byte descriptors and 8-byte hashes.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

pub const CODEC_STORED: u8 = 0x00;
pub const CODEC_ZLIB: u8 = 0x20;
pub const CODEC_BC1: u8 = 0x04;

/// One chunk of the store.
#[derive(Clone, Debug)]
pub struct Chunk {
    pub index: usize,
    pub file_off: u64,
    /// Stored bytes including the trailing pad.
    pub file_len: u64,
    pub pad: u64,
    pub vstart: u64,
    pub vlen: u64,
    pub codec: u8,
}

impl Chunk {
    pub fn data_len(&self) -> u64 {
        self.file_len.saturating_sub(self.pad)
    }
}

/// One ETPF texture descriptor.
#[derive(Clone, Debug)]
pub struct Texture {
    pub index: usize,
    pub hash: u64,
    pub width: u32,
    pub height: u32,
    pub mips: u32,
    pub dxgi: u32,
    pub array_size: u32,
    pub tile_mode: u32,
    pub vstart: u64,
}

/// An opened store (whole file in memory via a read).
pub struct Store {
    pub data: Vec<u8>,
    pub vsize: u64,
    pub chunks: Vec<Chunk>,
    pub etpf_off: u64,
}

fn u32le(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

impl Store {
    pub fn open(path: &Path) -> io::Result<Store> {
        let mut data = Vec::new();
        File::open(path)?.read_to_end(&mut data)?;
        Store::from_bytes(data)
    }

    pub fn from_bytes(data: Vec<u8>) -> io::Result<Store> {
        if data.len() < 0x18 || &data[..4] != b"RDS5" {
            return Err(bad("not an RDS5 store"));
        }
        let n = u32le(&data, 4) as usize;
        let vsize = u32le(&data, 8) as u64 | ((u32le(&data, 12) as u64 & 0xFF) << 32);
        let etpf_off = (u32le(&data, 16) as u64) << 8;
        if 0x18 + 8 * n > data.len() {
            return Err(bad("chunk table past end of file"));
        }
        let ent = |i: usize| (u32le(&data, 0x18 + 8 * i), u32le(&data, 0x1C + 8 * i));
        let mut chunks = Vec::with_capacity(n.saturating_sub(1));
        for i in 0..n.saturating_sub(1) {
            let (a, b) = ent(i);
            let (a2, b2) = ent(i + 1);
            let off = ((a & 0x3FF_FFFF) as u64) << 8;
            let end = ((a2 & 0x3FF_FFFF) as u64) << 8;
            let v = (((a >> 30) as u64) | (((b & 0xFFFFF) as u64) << 2)) << 12;
            let v2 = (((a2 >> 30) as u64) | (((b2 & 0xFFFFF) as u64) << 2)) << 12;
            chunks.push(Chunk {
                index: i,
                file_off: off,
                file_len: end.saturating_sub(off),
                pad: (b >> 20) as u64,
                vstart: v,
                vlen: v2.saturating_sub(v),
                codec: ((a >> 24) & 0x3F) as u8,
            });
        }
        Ok(Store { data, vsize, chunks, etpf_off })
    }

    /// Stored bytes of a chunk (with the trailing pad when `with_pad`).
    pub fn chunk_bytes(&self, i: usize, with_pad: bool) -> io::Result<&[u8]> {
        let c = &self.chunks[i];
        let len = if with_pad { c.file_len } else { c.data_len() };
        let (a, b) = (c.file_off as usize, (c.file_off + len) as usize);
        self.data.get(a..b).ok_or_else(|| bad(format!("chunk {i} past end of file")))
    }

    /// Unpacked bytes of a plain (stored or zlib) chunk.
    pub fn plain_chunk(&self, i: usize) -> io::Result<Vec<u8>> {
        let c = &self.chunks[i];
        let raw = self.chunk_bytes(i, true)?;
        let mut out = match c.codec {
            CODEC_ZLIB => {
                let mut v = Vec::with_capacity(c.vlen as usize);
                flate2::read::ZlibDecoder::new(raw).read_to_end(&mut v)?;
                v
            }
            CODEC_STORED => raw.to_vec(),
            x => return Err(bad(format!("chunk {i}: codec {x:#x} is not plain"))),
        };
        if (out.len() as u64) < c.vlen {
            return Err(bad(format!("chunk {i}: {} bytes < vlen {}", out.len(), c.vlen)));
        }
        out.truncate(c.vlen as usize);
        Ok(out)
    }

    /// Chunks covering the virtual range [a, b): (chunk, from, to) relative to the chunk.
    pub fn locate(&self, a: u64, b: u64) -> io::Result<Vec<(usize, u64, u64)>> {
        let mut out = Vec::new();
        let mut pos = a;
        while pos < b {
            let i = self.chunks.partition_point(|c| c.vstart <= pos);
            if i == 0 {
                return Err(bad("virtual range before first chunk"));
            }
            let c = &self.chunks[i - 1];
            if c.vlen == 0 || pos >= c.vstart + c.vlen {
                return Err(bad("virtual range not covered"));
            }
            let end = b.min(c.vstart + c.vlen);
            out.push((i - 1, pos - c.vstart, end - c.vstart));
            pos = end;
        }
        Ok(out)
    }

    pub fn textures(&self) -> io::Result<Vec<Texture>> {
        let at = self.etpf_off as usize;
        let d = &self.data;
        if at + 16 > d.len() || &d[at..at + 4] != b"ETPF" {
            return Err(bad("no ETPF index"));
        }
        let t = u32le(d, at + 8) as usize;
        let desc = at + 16;
        let hashes = desc + 16 * t;
        if hashes + 8 * t > d.len() {
            return Err(bad("ETPF index past end of file"));
        }
        Ok((0..t)
            .map(|i| {
                let o = desc + 16 * i;
                let (d0, d1, d2) = (u32le(d, o), u32le(d, o + 4), u32le(d, o + 8));
                let h = u32le(d, hashes + 8 * i) as u64 | (u32le(d, hashes + 8 * i + 4) as u64) << 32;
                Texture {
                    index: i,
                    hash: h,
                    width: (d0 & 0x3FFF) + 1,
                    height: ((d0 >> 14) & 0x3FFF) + 1,
                    mips: (d0 >> 28) + 1,
                    dxgi: (d1 & 0x3FF) >> 3,
                    array_size: ((d1 >> 10) & 0x3FF) + 1,
                    tile_mode: (d1 >> 24) & 0xF,
                    vstart: ((d2 & 0x1F_FFFF) as u64) << 12,
                }
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Gfx10 tiling and mip chain (port of ttpf-viewer src/tile.rs + src/layout.rs)
// ---------------------------------------------------------------------------

/// Swizzle bit pattern, lowest address bit first, for 8-byte elements (BC1).
pub fn pattern_8b(tile_mode: u32) -> &'static str {
    if tile_mode == 0 {
        "xyxxyyxyxyxyx" // 64 KiB tiles
    } else {
        "xyyxxyxyx" // 4 KiB tiles
    }
}

fn tile_lut(pattern: &str) -> (Vec<u32>, Vec<u32>, u32, u32) {
    let n = pattern.len();
    let mut lx = Vec::with_capacity(1 << n);
    let mut ly = Vec::with_capacity(1 << n);
    for j in 0..(1u32 << n) {
        let (mut x, mut y, mut xi, mut yi) = (0, 0, 0, 0);
        for (b, c) in pattern.bytes().enumerate() {
            let bit = (j >> b) & 1;
            if c == b'x' {
                x |= bit << xi;
                xi += 1;
            } else {
                y |= bit << yi;
                yi += 1;
            }
        }
        lx.push(x);
        ly.push(y);
    }
    let xs = pattern.bytes().filter(|&c| c == b'x').count() as u32;
    (lx, ly, 1 << xs, 1 << (n as u32 - xs))
}

/// For each stored element i < n, its linear index y*w+x in a w x h grid, or None.
pub fn untile_order(n: usize, w: usize, h: usize, pattern: &str) -> Vec<Option<usize>> {
    let bits = (usize::BITS - n.max(1).leading_zeros() - 1) as usize;
    let t = &pattern[..pattern.len().min(bits)];
    let (lx, ly, tw, th) = tile_lut(t);
    let tl = t.len();
    let mask = (1usize << tl) - 1;
    let tpr = w.div_ceil(tw as usize).max(1);
    (0..n)
        .map(|i| {
            let k = i >> tl;
            let x = (k % tpr) * tw as usize + lx[i & mask] as usize;
            let y = (k / tpr) * th as usize + ly[i & mask] as usize;
            if x < w && y < h {
                Some(y * w + x)
            } else {
                None
            }
        })
        .collect()
}

/// Placement of one mip level inside the texture's virtual bytes.
#[derive(Clone, Debug)]
pub struct Place {
    pub w: u32,
    pub h: u32,
    pub off: u64,
    pub len: u64,
    /// Decoded region in pixels (whole tiles) and the level origin inside it.
    pub pw: u32,
    pub ph: u32,
    pub ox: u32,
    pub oy: u32,
}

/// Gfx10 mip chain of an 8-byte-block (BC1) texture: levels largest first.
pub fn mip_chain_bc1(tile_mode: u32, w: u32, h: u32, mips: u32) -> Option<(Vec<Place>, u64)> {
    let elem: u64 = 8;
    let pat = pattern_8b(tile_mode);
    let is64k = tile_mode == 0;
    if mips == 0 {
        return None;
    }
    let (_, _, bw, bh) = tile_lut(pat);
    let (tw, th) = (bw * 4, bh * 4);
    let tile_bytes = bw as u64 * bh as u64 * elem;
    let max_in_tail = if is64k { 12 } else { 8 };
    let px = |i: u32| ((w >> i).max(1), (h >> i).max(1));
    let el = |i: u32| (px(i).0.div_ceil(4), px(i).1.div_ceil(4));
    let mut tail = mips;
    if mips > 1 {
        for i in 0..mips {
            if el(i).0 <= bw / 2 && el(i).1 <= bh && mips - i <= max_in_tail {
                tail = i;
                break;
            }
        }
    }
    let (mw, mh) = (8u32, 4u32);
    let mut levels: Vec<Option<Place>> = vec![None; mips as usize];
    let mut pos: u64 = 0;
    if tail < mips {
        pos = tile_bytes;
        for i in tail..mips {
            let slot = max_in_tail - 1 - (i - tail);
            let at: u32 = if slot > 6 { 16 << slot } else { slot << 8 };
            let morton = at >> 8;
            let (mut mx, mut my) = (0u32, 0u32);
            for bit in 0..8 {
                my |= ((morton >> (2 * bit)) & 1) << bit;
                mx |= ((morton >> (2 * bit + 1)) & 1) << bit;
            }
            let (ox, oy) = (mx * mw, my * mh);
            if ox + el(i).0 > bw || oy + el(i).1 > bh {
                return None;
            }
            let (lw, lh) = px(i);
            levels[i as usize] = Some(Place { w: lw, h: lh, off: 0, len: tile_bytes, pw: tw, ph: th, ox: ox * 4, oy: oy * 4 });
        }
    }
    for i in (0..tail).rev() {
        let (ew, eh) = el(i);
        let (nx, ny) = (ew.div_ceil(bw), eh.div_ceil(bh));
        let ln = nx as u64 * ny as u64 * tile_bytes;
        let (lw, lh) = px(i);
        levels[i as usize] = Some(Place { w: lw, h: lh, off: pos, len: ln, pw: nx * tw, ph: ny * th, ox: 0, oy: 0 });
        pos += ln;
    }
    Some((levels.into_iter().map(|p| p.unwrap()).collect(), pos))
}

/// Stored (tiled) level bytes -> row-major BC1 blocks of the level, cropped.
pub fn detile_bc1(data: &[u8], place: &Place, tile_mode: u32) -> (Vec<u8>, usize, usize) {
    let (pwb, phb) = ((place.pw / 4) as usize, (place.ph / 4) as usize);
    let n = data.len() / 8;
    let order = untile_order(n, pwb, phb, pattern_8b(tile_mode));
    let mut lin = vec![0u8; pwb * phb * 8];
    for (i, d) in order.iter().enumerate() {
        if let Some(d) = d {
            lin[d * 8..d * 8 + 8].copy_from_slice(&data[i * 8..i * 8 + 8]);
        }
    }
    let (bw, bh) = (place.w.div_ceil(4) as usize, place.h.div_ceil(4) as usize);
    let (ox, oy) = ((place.ox / 4) as usize, (place.oy / 4) as usize);
    let mut out = Vec::with_capacity(bw * bh * 8);
    for y in oy..oy + bh {
        out.extend_from_slice(&lin[(y * pwb + ox) * 8..(y * pwb + ox + bw) * 8]);
    }
    (out, bw, bh)
}

// ---------------------------------------------------------------------------
// BC1 pixels, DDS, PNG
// ---------------------------------------------------------------------------

fn c565(c: u16) -> [u8; 3] {
    let (r, g, b) = ((c >> 11) & 31, (c >> 5) & 63, c & 31);
    [((r << 3) | (r >> 2)) as u8, ((g << 2) | (g >> 4)) as u8, ((b << 3) | (b >> 2)) as u8]
}

/// Standard BC1 palette (RGBA): 4-colour when c0 > c1, else 3-colour + transparent black.
pub fn bc1_palette(c0: u16, c1: u16) -> [[u8; 4]; 4] {
    let (a, b) = (c565(c0), c565(c1));
    let mut p = [[a[0], a[1], a[2], 255], [b[0], b[1], b[2], 255], [0, 0, 0, 255], [0, 0, 0, 0]];
    for k in 0..3 {
        let (x, y) = (a[k] as u32, b[k] as u32);
        if c0 > c1 {
            p[2][k] = ((2 * x + y + 1) / 3) as u8;
            p[3][k] = ((x + 2 * y + 1) / 3) as u8;
        } else {
            p[2][k] = ((x + y) / 2) as u8;
        }
    }
    if c0 > c1 {
        p[3][3] = 255;
    }
    p
}

/// Row-major BC1 blocks (bw x bh) -> RGBA pixels cropped to w x h.
pub fn bc1_to_rgba(blocks: &[u8], bw: usize, bh: usize, w: usize, h: usize) -> Vec<u8> {
    let mut img = vec![0u8; w * h * 4];
    for by in 0..bh {
        for bx in 0..bw {
            let o = (by * bw + bx) * 8;
            let c0 = u16::from_le_bytes([blocks[o], blocks[o + 1]]);
            let c1 = u16::from_le_bytes([blocks[o + 2], blocks[o + 3]]);
            let idx = u32le(blocks, o + 4);
            let pal = bc1_palette(c0, c1);
            for py in 0..4 {
                for pxl in 0..4 {
                    let (x, y) = (bx * 4 + pxl, by * 4 + py);
                    if x < w && y < h {
                        let c = pal[((idx >> (2 * (py * 4 + pxl))) & 3) as usize];
                        img[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&c);
                    }
                }
            }
        }
    }
    img
}

/// A DDS file (legacy header, FourCC DXT1) holding one BC1 level.
pub fn dds_bc1(blocks: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut d = Vec::with_capacity(128 + blocks.len());
    let mut put = |v: u32| d.extend_from_slice(&v.to_le_bytes());
    put(0x2053_4444); // "DDS "
    put(124); // header size
    put(0x1 | 0x2 | 0x4 | 0x1000 | 0x80000); // CAPS HEIGHT WIDTH PIXELFORMAT LINEARSIZE
    put(h);
    put(w);
    put(blocks.len() as u32);
    put(0); // depth
    put(1); // mip count
    for _ in 0..11 {
        put(0);
    }
    put(32); // pixel format size
    put(0x4); // DDPF_FOURCC
    put(u32::from_le_bytes(*b"DXT1"));
    for _ in 0..5 {
        put(0);
    }
    put(0x1000); // DDSCAPS_TEXTURE
    for _ in 0..4 {
        put(0);
    }
    d.extend_from_slice(blocks);
    d
}

/// Encode RGBA pixels as PNG.
pub fn png_rgba(rgba: &[u8], w: u32, h: u32) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header().map_err(|e| bad(e.to_string()))?;
        wr.write_image_data(rgba).map_err(|e| bad(e.to_string()))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untile_is_a_permutation() {
        for &(tm, w, h) in &[(0u32, 128usize, 64usize), (1, 64, 64), (1, 16, 8)] {
            let n = w * h;
            let o = untile_order(n, w, h, pattern_8b(tm));
            let mut seen = vec![false; n];
            for d in o.into_iter().flatten() {
                assert!(!seen[d]);
                seen[d] = true;
            }
            assert!(seen.iter().all(|&s| s));
        }
    }

    #[test]
    fn palette_modes() {
        let p = bc1_palette(0xFFFF, 0x0000);
        assert_eq!(p[2][..3], [170, 170, 170]);
        let p = bc1_palette(0x0000, 0xFFFF);
        assert_eq!(p[2][..3], [127, 127, 127]);
        assert_eq!(p[3], [0, 0, 0, 0]);
    }
}
