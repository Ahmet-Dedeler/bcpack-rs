// bcpack-rs by Ahmet Dedeler (https://github.com/Ahmet-Dedeler, https://ahmetdedeler.com)
// Source: https://github.com/Ahmet-Dedeler/bcpack-rs
// SPDX-License-Identifier: MIT OR Apache-2.0
// If you copy, vendor or port this file, keep this header and credit bcpack-rs in your README.
//! Pixel-decode tests: hand-computed blocks, invariants over all BC7 controls,
//! synthetic stream round trips, and the `ttpf-export` / `scan` CLI paths.
//! Everything here is synthetic; no game bytes.

use bcpack::bc7::{self, Control};
use bcpack::encode::Rng;
use bcpack::format::Mode;
use bcpack::pixels;

fn bc4_block(e0: u8, e1: u8, codes: [u8; 16]) -> [u8; 8] {
    let mut v = e0 as u64 | (e1 as u64) << 8;
    for (p, &c) in codes.iter().enumerate() {
        v |= (c as u64) << (16 + 3 * p);
    }
    v.to_le_bytes()
}

fn bc1_block(c0: u16, c1: u16, idx: u32) -> [u8; 8] {
    let mut b = [0u8; 8];
    b[..2].copy_from_slice(&c0.to_le_bytes());
    b[2..4].copy_from_slice(&c1.to_le_bytes());
    b[4..8].copy_from_slice(&idx.to_le_bytes());
    b
}

fn idx_word(codes: impl Fn(usize) -> u32) -> u32 {
    let mut w = 0u32;
    for p in 0..16 {
        w |= codes(p) << (2 * p);
    }
    w
}

const CODES: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0];

#[test]
fn pixel_decode_rejects_malformed_input() {
    use std::io::ErrorKind::InvalidInput;
    for (mode, input, bw, bh, w, h) in [
        (Mode::Bc7, &[0; 15][..], 1, 1, 4, 4),
        (Mode::Bc4, &[0; 8][..], 1, 1, 5, 4),
        (Mode::Bc3, &[][..], usize::MAX, 2, 4, 4),
        (Mode::Bc6h, &[0; 16][..], 1, 1, 4, 4),
    ] {
        let err = pixels::blocks_to_rgba(mode, false, input, bw, bh, w, h).unwrap_err();
        assert_eq!(err.kind(), InvalidInput);
    }
}

#[test]
fn bc4_palettes_by_hand() {
    // e0 > e1: 8-value palette, a[i] = ((8-i)*e0 + (i-1)*e1) / 7.
    let img = pixels::bc4_to_rgba(&bc4_block(255, 0, CODES), 1, 1, 4, 4, false).unwrap();
    let expect = [
        255, 0, 218, 182, 145, 109, 72, 36, 36, 72, 109, 145, 182, 218, 0, 255,
    ];
    for p in 0..16 {
        assert_eq!(
            &img[p * 4..p * 4 + 4],
            &[expect[p], expect[p], expect[p], 255]
        );
    }
    // e0 <= e1: 6-value palette with divisor 5, then 0 and 255.
    let img = pixels::bc4_to_rgba(&bc4_block(0, 255, CODES), 1, 1, 4, 4, false).unwrap();
    let expect = [
        0, 255, 51, 102, 153, 204, 0, 255, 255, 0, 204, 153, 102, 51, 255, 0,
    ];
    for p in 0..16 {
        assert_eq!(
            &img[p * 4..p * 4 + 4],
            &[expect[p], expect[p], expect[p], 255]
        );
    }
    // Cropping: a 2x3 image keeps only in-bounds pixels (block pixels 0, 1, 4, 5, 8, 9).
    let img = pixels::bc4_to_rgba(&bc4_block(255, 0, CODES), 1, 1, 2, 3, false).unwrap();
    assert_eq!(img.len(), 2 * 3 * 4);
    assert_eq!(&img[0..4], &[255, 255, 255, 255]);
    assert_eq!(&img[4..8], &[0, 0, 0, 255]);
    assert_eq!(&img[8..12], &[145, 145, 145, 255]);
}

#[test]
fn bc4_snorm_by_hand() {
    // -128 and -127 both represent -1.0. The preview spans all of RGBA8.
    let img = pixels::bc4_to_rgba(&bc4_block(0x80, 0x81, CODES), 1, 1, 4, 4, true).unwrap();
    assert_eq!(&img[..8], &[0, 0, 0, 255, 0, 0, 0, 255]);
    assert_eq!(&img[24..32], &[0, 0, 0, 255, 255, 255, 255, 255]);

    // e0 = -5 <= e1 = 5: signed interpolants are -3, -1, 1, 3, then -127/127.
    let img = pixels::bc4_to_rgba(&bc4_block(0xFB, 0x05, CODES), 1, 1, 4, 4, true).unwrap();
    let expect = [
        122, 133, 124, 126, 129, 131, 0, 255, 255, 0, 131, 129, 126, 124, 133, 122,
    ];
    for p in 0..16 {
        assert_eq!(
            &img[p * 4..p * 4 + 4],
            &[expect[p], expect[p], expect[p], 255]
        );
    }
    // e0 = 5 > e1 = -5: 8-value signed palette with divisor 7.
    let img = pixels::bc4_to_rgba(&bc4_block(0x05, 0xFB, CODES), 1, 1, 4, 4, true).unwrap();
    let expect = [
        133, 122, 131, 130, 128, 127, 125, 124, 124, 125, 127, 128, 130, 131, 122, 133,
    ];
    for p in 0..16 {
        assert_eq!(
            &img[p * 4..p * 4 + 4],
            &[expect[p], expect[p], expect[p], 255]
        );
    }
}

#[test]
fn bc1_block_by_hand() {
    // c0 = 565 red > c1 = 565 blue: 4-colour palette.
    let img = pixels::bc1_to_rgba(
        &bc1_block(0xF800, 0x001F, idx_word(|p| p as u32 % 4)),
        1,
        1,
        4,
        4,
    );
    let pal = [
        [255, 0, 0, 255],
        [0, 0, 255, 255],
        [170, 0, 85, 255],
        [85, 0, 170, 255],
    ];
    for p in 0..16 {
        assert_eq!(&img[p * 4..p * 4 + 4], &pal[p % 4]);
    }
    // c0 <= c1: 3-colour punch-through, index 3 is transparent black.
    let img = pixels::bc1_to_rgba(&bc1_block(0x0000, 0xFFFF, idx_word(|_| 3)), 1, 1, 4, 4);
    assert!(img.chunks(4).all(|px| px == [0, 0, 0, 0]));
    let img = pixels::bc1_to_rgba(&bc1_block(0x0000, 0xFFFF, idx_word(|_| 2)), 1, 1, 4, 4);
    assert!(img.chunks(4).all(|px| px == [127, 127, 127, 255]));
}

#[test]
fn bc3_block_by_hand() {
    // Alpha half-block e0=255 > e1=0, colour half-block red/blue with c0 > c1.
    let mut block = [0u8; 16];
    block[..8].copy_from_slice(&bc4_block(255, 0, CODES));
    block[8..].copy_from_slice(&bc1_block(0xF800, 0x001F, idx_word(|p| p as u32 % 4)));
    let img = pixels::bc3_to_rgba(&block, 1, 1, 4, 4).unwrap();
    let alpha = [
        255, 0, 218, 182, 145, 109, 72, 36, 36, 72, 109, 145, 182, 218, 0, 255,
    ];
    let pal = [[255, 0, 0], [0, 0, 255], [170, 0, 85], [85, 0, 170]];
    for p in 0..16 {
        let c = pal[p % 4];
        assert_eq!(&img[p * 4..p * 4 + 4], &[c[0], c[1], c[2], alpha[p]]);
    }
    // The BC3 colour block is always 4-colour: even with c0 <= c1, index 3 is
    // an opaque mix, never transparent.
    let mut block = [0u8; 16];
    block[..8].copy_from_slice(&bc4_block(255, 255, [0; 16]));
    block[8..].copy_from_slice(&bc1_block(0x0000, 0xFFFF, idx_word(|_| 3)));
    let img = pixels::bc3_to_rgba(&block, 1, 1, 4, 4).unwrap();
    assert!(img.chunks(4).all(|px| px == [170, 170, 170, 255]));
}

#[test]
fn bc5_block_by_hand() {
    let mut block = [0u8; 16];
    block[..8].copy_from_slice(&bc4_block(255, 0, CODES));
    block[8..].copy_from_slice(&bc4_block(0, 255, CODES));
    let img = pixels::bc5_to_rgba(&block, 1, 1, 4, 4, false).unwrap();
    let r = [
        255, 0, 218, 182, 145, 109, 72, 36, 36, 72, 109, 145, 182, 218, 0, 255,
    ];
    let g = [
        0, 255, 51, 102, 153, 204, 0, 255, 255, 0, 204, 153, 102, 51, 255, 0,
    ];
    for p in 0..16 {
        assert_eq!(&img[p * 4..p * 4 + 4], &[r[p], g[p], 0, 255]);
    }
    block[..8].copy_from_slice(&bc4_block(0x80, 0x7F, CODES));
    block[8..].copy_from_slice(&bc4_block(0x7F, 0x81, CODES));
    let img = pixels::bc5_to_rgba(&block, 1, 1, 4, 4, true).unwrap();
    assert_eq!(&img[..8], &[0, 255, 0, 255, 255, 0, 0, 255]);
}

/// Spec weight tables, copied independently of `pixels`.
const W2: [u32; 4] = [0, 21, 43, 64];
const W3: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const W4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

fn interp(e0: u32, e1: u32, w: u32) -> u8 {
    ((e0 * (64 - w) + e1 * w + 32) >> 6) as u8
}

/// Bit-replicated endpoint expansion: p-bit value in the top bits of `raw`.
fn rep(raw: u8, p: u32) -> u8 {
    if p == 0 {
        255
    } else if p >= 8 {
        raw
    } else {
        raw | (raw >> p)
    }
}

/// The spec's endpoint expansion: in the p-bit modes the p-bit (stored as the
/// lowest bit of red in the unpacked state) extends every component of the
/// endpoint, so each channel reads base+1 bits.
fn spec_ep8(c: &Control, st: &bc7::State, row: usize, ch: usize, k: usize) -> u8 {
    let raw = st.ep[row][2 * ch + k];
    if !c.has_pbits() {
        let p = c.precision()[ch] as u32;
        return if p == 0 { 255 } else { rep(raw, p) };
    }
    if ch == 0 {
        return rep(raw, c.precision()[0] as u32);
    }
    let base = c.precision()[ch] as u32;
    if base == 0 {
        return 255;
    }
    let pbit = (st.ep[row][k] >> (8 - c.precision()[0])) & 1;
    let v = (raw >> (8 - base)) as u32;
    let full = ((v << 1) | pbit as u32) << (8 - base - 1);
    rep(full as u8, base + 1)
}

#[test]
fn bc7_mode6_block_by_hand() {
    // Mode 6: 1 subset, RGBA 7.7.7.7 + p-bit per endpoint (effectively 8 bits),
    // 4-bit indices. Endpoint raws below have p-bit 0, so every channel expands
    // from its top-aligned raw value directly.
    let c = Control {
        mode: 6,
        partition: 0,
        rotation: 0,
        index_selection: 0,
    };
    let mut st = bc7::State {
        subsets: 1,
        ..Default::default()
    };
    st.ep[0] = [0x00, 0xFE, 0x80, 0x00, 0x40, 0xC0, 0xFE, 0x00];
    for (p, ix) in st.ix[0].iter_mut().enumerate().take(16) {
        *ix = p as u8; // 4-bit indices, top-aligned in 4 bits
    }
    let mut block = [0u8; 16];
    bc7::pack(c, &st, &mut block);
    let px = pixels::bc7_block_rgba(&block);
    // With p-bit 0 the expansion is: R (p=8): 0x00, 0xFE; G/B/A (p=7->8 with
    // pbit 0): raw with the low bit clear.
    let e = [(0x00u8, 0xFEu8), (0x80, 0x00), (0x40, 0xC0), (0xFE, 0x00)];
    for (p, row) in px.iter().enumerate() {
        let w = W4[p];
        for (ch, &got) in row.iter().enumerate() {
            assert_eq!(
                got,
                interp(e[ch].0 as u32, e[ch].1 as u32, w),
                "pixel {p} channel {ch}"
            );
        }
    }
    // spec_ep8 agrees with the hand constants on every endpoint.
    for (ch, ech) in e.iter().enumerate() {
        assert_eq!(spec_ep8(&c, &st, 0, ch, 0), ech.0);
        assert_eq!(spec_ep8(&c, &st, 0, ch, 1), ech.1);
    }
}

#[test]
fn bc7_mode0_three_subsets_by_hand() {
    // Mode 0: 3 subsets, partition 5, RGB 4 bits + p-bit on red, 3-bit indices.
    let c = Control {
        mode: 0,
        partition: 5,
        rotation: 0,
        index_selection: 0,
    };
    let mut st = bc7::State {
        subsets: 3,
        ..Default::default()
    };
    // Flat endpoints per subset (e0 == e1), so every pixel of a subset is one
    // colour. Red's p-bit is forced to 0 (even 5-bit value), so every RGB
    // channel expands from its top-aligned raw with base+1 bits.
    let raw = |s: usize, ch: usize| -> u8 {
        let base = bc7::CHANNEL_BITS[0][ch] as u32; // mode 0: RGB 4 bits, no alpha
        let mut v = (s as u32 * 9 + ch as u32 * 3 + 1) % (1 << base);
        if ch == 0 {
            v &= !1; // p-bit 0
        }
        (v << (8 - base)) as u8
    };
    for (s, ep) in st.ep.iter_mut().enumerate().take(3) {
        for ch in 0..4 {
            let r = raw(s, ch);
            ep[2 * ch] = r;
            ep[2 * ch + 1] = r;
        }
    }
    for ix in st.ix[0].iter_mut().take(16) {
        *ix = 0; // weight 0 -> endpoint 0
    }
    let mut block = [0u8; 16];
    bc7::pack(c, &st, &mut block);
    let px = pixels::bc7_block_rgba(&block);
    for (p, row) in px.iter().enumerate() {
        let s = pixels::PARTITION_3[5][p] as usize;
        for (ch, &got) in row.iter().enumerate() {
            let base = bc7::CHANNEL_BITS[0][ch] as u32;
            let want = if base == 0 {
                255
            } else {
                rep(raw(s, ch), base + 1)
            };
            assert_eq!(got, want, "pixel {p} ch {ch}");
        }
    }
}

#[test]
fn bc7_mode4_dual_index_sets_by_hand() {
    // Mode 4: 1 subset, rotation 1 (R is the scalar), index selection 1
    // (colour uses the 3-bit set 0, the scalar R uses the 2-bit set 1).
    let c = Control {
        mode: 4,
        partition: 0,
        rotation: 1,
        index_selection: 1,
    };
    let mut st = bc7::State {
        subsets: 1,
        ..Default::default()
    };
    // precision after rotation: R 6 bits, G/B/A 5 bits.
    let prec = [6u32, 5, 5, 5];
    let raws = [0x00u8, 0xFC, 0x00, 0xF8, 0x20, 0xE0, 0x10, 0xF0];
    st.ep[0] = raws;
    for p in 0..16 {
        st.ix[0][p] = ((p % 8) as u8) << 1; // 3-bit set, top-aligned
        st.ix[1][p] = ((p % 4) as u8) << 2; // 2-bit set, top-aligned
    }
    let mut block = [0u8; 16];
    bc7::pack(c, &st, &mut block);
    let px = pixels::bc7_block_rgba(&block);
    for (p, row) in px.iter().enumerate() {
        for (ch, &got) in row.iter().enumerate() {
            let (set, tab) = if ch == 0 {
                (1usize, &W2[..])
            } else {
                (0, &W3[..])
            };
            let ix = if set == 1 { p % 4 } else { p % 8 };
            let w = tab[ix];
            let e0 = rep(raws[2 * ch], prec[ch]) as u32;
            let e1 = rep(raws[2 * ch + 1], prec[ch]) as u32;
            assert_eq!(got, interp(e0, e1, w), "pixel {p} ch {ch}");
        }
    }
}

#[test]
fn bc7_flat_endpoints_give_flat_pixels_for_every_control() {
    let mut rng = Rng(99);
    for c in Control::all() {
        let subsets = c.subsets();
        let prec = c.precision();
        let mut st = bc7::State {
            subsets: subsets as u8,
            ..Default::default()
        };
        for ep in st.ep.iter_mut().take(subsets) {
            for (ch, &p) in prec.iter().enumerate() {
                let v = if p == 0 {
                    0
                } else {
                    rng.below(1u64 << p) as u32
                };
                let raw = if p == 0 { 0 } else { (v << (8 - p)) as u8 };
                ep[2 * ch] = raw;
                ep[2 * ch + 1] = raw;
            }
        }
        for set in st.ix.iter_mut().take(2) {
            for ix in set.iter_mut().take(16) {
                *ix = rng.below(16) as u8;
            }
        }
        let mut block = [0u8; 16];
        bc7::pack(c, &st, &mut block);
        let px = pixels::bc7_block_rgba(&block);
        let part = (c.partition as usize) & 63;
        for (p, row) in px.iter().enumerate() {
            let s = match subsets {
                3 => pixels::PARTITION_3[part][p] as usize,
                2 => pixels::PARTITION_2[part][p] as usize,
                _ => 0,
            };
            for (ch, &got) in row.iter().enumerate() {
                assert_eq!(
                    got,
                    spec_ep8(&c, &st, s, ch, 0),
                    "control {c:?} pixel {p} ch {ch}"
                );
            }
        }
    }
    // A reserved block (no mode bit) is transparent black.
    assert_eq!(pixels::bc7_block_rgba(&[0u8; 16]), [[0u8; 4]; 16]);
}

#[test]
fn synthetic_streams_decode_to_pixels() {
    let window = bcpack::bc1::Window::default_table();
    let quads = bcpack::bc1::QuadTable::default_table();
    for mode in [Mode::Bc1, Mode::Bc3, Mode::Bc4, Mode::Bc5, Mode::Bc7] {
        let blocks = 37;
        let (stream, _) = bcpack::encode::synthetic(mode, 23, blocks, 11);
        let out = bcpack::decode_chunk(mode, &stream, blocks, &window, &quads).unwrap();
        assert_eq!(out.blocks.len(), blocks * mode.block_bytes());
        let bb = mode.block_bytes();
        // 37 blocks in one row of blocks -> 148x4 pixels.
        let img = pixels::blocks_to_rgba(mode, false, &out.blocks, blocks, 1, 148, 4).unwrap();
        assert_eq!(img.len(), 148 * 4 * 4);
        // Per-block decode must agree with the batch decode.
        for b in 0..blocks {
            let one =
                pixels::blocks_to_rgba(mode, false, &out.blocks[b * bb..(b + 1) * bb], 1, 1, 4, 4)
                    .unwrap();
            assert_eq!(
                &img[b * 16..b * 16 + 4],
                &one[0..4],
                "{} block {b}",
                mode.name()
            );
        }
        // Signed decode of BC4/BC5 runs over the same blocks.
        if matches!(mode, Mode::Bc4 | Mode::Bc5) {
            let img = pixels::blocks_to_rgba(mode, true, &out.blocks, blocks, 1, 148, 4).unwrap();
            assert_eq!(img.len(), 148 * 4 * 4);
        }
    }
}

// ---------------------------------------------------------------------------
// Store builder for the ttpf / cli tests
// ---------------------------------------------------------------------------

#[cfg(feature = "ttpf")]
struct StoreTex {
    dxgi: u32,
    width: u32,
    height: u32,
    mips: u32,
    tile_mode: u32,
    chunk: usize,
}

/// Build a minimal RDS5 store: one 256-byte-aligned chunk per entry, plus an
/// optional ETPF index.
#[cfg(feature = "ttpf")]
fn build_store(chunks: &[(Mode, Vec<u8>, u64)], textures: &[StoreTex]) -> Vec<u8> {
    let n = chunks.len() + 1;
    let table_end = 0x18 + 8 * n;
    let mut data = vec![0u8; table_end];
    let mut offs = Vec::new();
    let mut vstarts = Vec::new();
    let mut pads = Vec::new();
    let mut pos = table_end.div_ceil(256) * 256;
    let mut vpos = 0u64;
    for (_, stream, vlen) in chunks {
        offs.push(pos);
        vstarts.push(vpos);
        data.resize(pos + stream.len(), 0);
        data[pos..pos + stream.len()].copy_from_slice(stream);
        let pad = (pos + stream.len()).div_ceil(256) * 256 - (pos + stream.len());
        data.resize(pos + stream.len() + pad, 0);
        pads.push(pad as u64);
        pos += stream.len() + pad;
        vpos += vlen;
    }
    let end_off = pos;
    let mut put_entry = |i: usize, off: u64, vstart: u64, codec: u8, pad: u64| {
        let a = (((off >> 8) & 0x3FF_FFFF) as u32)
            | ((codec as u32) << 24)
            | ((((vstart >> 12) & 3) as u32) << 30);
        let b = (((vstart >> 14) & 0xFFFFF) as u32) | (((pad & 0xFFF) as u32) << 20);
        data[0x18 + 8 * i..0x18 + 8 * i + 4].copy_from_slice(&a.to_le_bytes());
        data[0x18 + 8 * i + 4..0x18 + 8 * i + 8].copy_from_slice(&b.to_le_bytes());
    };
    for (i, (mode, _, _)) in chunks.iter().enumerate() {
        put_entry(i, offs[i] as u64, vstarts[i], mode.codec(), pads[i]);
    }
    put_entry(n - 1, end_off as u64, vpos, 0, 0);
    data[0..4].copy_from_slice(b"RDS5");
    data[4..8].copy_from_slice(&(n as u32).to_le_bytes());
    data[8..12].copy_from_slice(&(vpos as u32).to_le_bytes());
    data[12..16].copy_from_slice(&(((vpos >> 32) & 0xFF) as u32).to_le_bytes());
    let etpf_off = data.len().div_ceil(256) * 256;
    data.resize(etpf_off, 0);
    data[16..20].copy_from_slice(&((etpf_off >> 8) as u32).to_le_bytes());
    if !textures.is_empty() {
        data.extend_from_slice(b"ETPF");
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(&(textures.len() as u32).to_le_bytes());
        data.extend_from_slice(&[0; 4]);
        for t in textures {
            let d0 = ((t.width - 1) & 0x3FFF)
                | (((t.height - 1) & 0x3FFF) << 14)
                | (((t.mips - 1) & 0xF) << 28);
            let d1 = ((t.dxgi & 0x7F) << 3) | ((t.tile_mode & 0xF) << 24);
            let d2 = ((vstarts[t.chunk] >> 12) & 0x1F_FFFF) as u32;
            data.extend_from_slice(&d0.to_le_bytes());
            data.extend_from_slice(&d1.to_le_bytes());
            data.extend_from_slice(&d2.to_le_bytes());
            data.extend_from_slice(&[0; 4]);
        }
        for _ in textures {
            data.extend_from_slice(&[0; 8]);
        }
    }
    data
}

/// One 16x8 mip-0 tile (4096 stored bytes) of `mode`, as a store + texture.
#[cfg(feature = "ttpf")]
fn tile_store(mode: Mode, dxgi: u32, seed: u64) -> (Vec<u8>, usize) {
    let bb = mode.block_bytes();
    let vlen = 4096u64;
    let blocks = (vlen / bb as u64) as usize;
    let (stream, _) = bcpack::encode::synthetic(mode, seed, blocks, 128);
    let store = build_store(
        &[(mode, stream, vlen)],
        &[StoreTex {
            dxgi,
            width: 16,
            height: 8,
            mips: 1,
            tile_mode: 1,
            chunk: 0,
        }],
    );
    (store, blocks)
}

#[cfg(feature = "ttpf")]
#[test]
fn png_round_trip_all_formats() {
    use bcpack::ttpf;
    let window = bcpack::bc1::Window::default_table();
    let quads = bcpack::bc1::QuadTable::default_table();
    for (mode, dxgi) in [
        (Mode::Bc1, 71),
        (Mode::Bc3, 77),
        (Mode::Bc4, 80),
        (Mode::Bc5, 83),
        (Mode::Bc7, 98),
    ] {
        let (store_bytes, blocks) = tile_store(mode, dxgi, 31);
        let store = ttpf::Store::from_bytes(store_bytes).unwrap();
        let tex = &store.textures().unwrap()[0];
        let m = bcpack::texture_mip0(&store, tex, &window, &quads).unwrap();
        assert_eq!(
            (m.width, m.height, m.bw, m.bh, m.bcpack_chunks),
            (16, 8, 4, 2, 1)
        );
        assert_eq!(m.blocks.len(), 4 * 2 * mode.block_bytes());
        assert_eq!(blocks * mode.block_bytes(), 4096);
        let (pmode, signed) = pixels::mode_from_dxgi(dxgi).unwrap();
        let rgba = pixels::blocks_to_rgba(pmode, signed, &m.blocks, m.bw, m.bh, 16, 8).unwrap();
        let png = ttpf::png_rgba(&rgba, 16, 8).unwrap();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        // Decode the PNG back and compare pixel for pixel.
        let dec = png::Decoder::new(std::io::Cursor::new(&png));
        let mut reader = dec.read_info().unwrap();
        assert_eq!((reader.info().width, reader.info().height), (16, 8));
        let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!(info.color_type, png::ColorType::Rgba);
        assert_eq!(&buf[..info.buffer_size()], &rgba[..], "{}", mode.name());
    }
}

#[cfg(feature = "cli")]
#[test]
fn cli_exports_png_and_dds_for_every_format_and_scans_all_formats() {
    use std::{fs, process::Command};
    let dir = std::env::temp_dir().join(format!("bcpack-pixels-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let run = |args: &[&str]| {
        let out = Command::new(env!("CARGO_BIN_EXE_bcpack"))
            .args(args)
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    };
    let window = bcpack::bc1::Window::default_table();
    let quads = bcpack::bc1::QuadTable::default_table();
    for (mode, dxgi, srgb) in [
        (Mode::Bc1, 71, false),
        (Mode::Bc1, 72, true),
        (Mode::Bc3, 78, true),
        (Mode::Bc4, 80, false),
        (Mode::Bc4, 81, false),
        (Mode::Bc5, 83, false),
        (Mode::Bc5, 84, false),
        (Mode::Bc7, 99, true),
    ] {
        let (store_bytes, _) = tile_store(mode, dxgi, 41);
        fs::write(dir.join("store.ttpf"), &store_bytes).unwrap();
        run(&["ttpf-export", "store.ttpf", "0", "out.png"]);
        let png = fs::read(dir.join("out.png")).unwrap();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        // IHDR: width 16, height 8, 8-bit RGBA.
        assert_eq!(&png[16..20], &16u32.to_be_bytes());
        assert_eq!(&png[20..24], &8u32.to_be_bytes());
        assert_eq!(png[24], 8);
        assert_eq!(png[25], 6);
        let mut reader = png::Decoder::new(std::io::Cursor::new(&png))
            .read_info()
            .unwrap();
        assert_eq!(reader.info().srgb.is_some(), srgb, "DXGI {dxgi}");
        let store = bcpack::ttpf::Store::from_bytes(store_bytes).unwrap();
        let m =
            bcpack::texture_mip0(&store, &store.textures().unwrap()[0], &window, &quads).unwrap();
        let signed = matches!(dxgi, 81 | 84);
        let rgba = pixels::blocks_to_rgba(mode, signed, &m.blocks, m.bw, m.bh, 16, 8).unwrap();
        let mut buf = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!(&buf[..info.buffer_size()], &rgba, "DXGI {dxgi}");
        run(&["ttpf-export", "store.ttpf", "0", "out.dds"]);
        let dds = fs::read(dir.join("out.dds")).unwrap();
        let fourcc = if mode == Mode::Bc1 { b"DXT1" } else { b"DX10" };
        assert_eq!(&dds[84..88], fourcc, "{}", mode.name());
        if mode != Mode::Bc1 {
            assert_eq!(&dds[128..132], &dxgi.to_le_bytes());
        }
    }
    // Virtual offsets have 4 KiB granularity, so each chunk spans 4 KiB.
    let mut chunks = Vec::new();
    for (i, mode) in [
        Mode::Bc1,
        Mode::Bc4,
        Mode::Bc1,
        Mode::Bc7,
        Mode::Bc1,
        Mode::Bc3,
        Mode::Bc5,
    ]
    .into_iter()
    .enumerate()
    {
        let blocks = 4096usize / mode.block_bytes();
        let (stream, _) = bcpack::encode::synthetic(mode, 5 + i as u64, blocks, 128);
        chunks.push((mode, stream, 4096u64));
    }
    fs::write(dir.join("multi.ttpf"), build_store(&chunks, &[])).unwrap();
    let out = run(&["scan", "--emit", "emit", "multi.ttpf"]);
    assert!(out.contains("TOTAL bc1: 3/3 converged"), "{out}");
    for name in ["bc3", "bc4", "bc5", "bc7"] {
        assert!(
            out.contains(&format!("  {name}: 1/1 converged")),
            "{name}: {out}"
        );
        assert!(
            out.contains(&format!("TOTAL {name}: 1/1 converged")),
            "{name}: {out}"
        );
    }
    // Emitted blocks match decode_chunk output.
    for (i, (mode, stream, _)) in chunks.iter().enumerate() {
        let blocks = 4096usize / mode.block_bytes();
        let expect = bcpack::decode_chunk(*mode, stream, blocks, &window, &quads)
            .unwrap()
            .blocks;
        assert_eq!(
            fs::read(dir.join("emit").join(format!("multi_{i}.{}", mode.name()))).unwrap(),
            expect
        );
    }
    let out = run(&["scan", "--every", "2", "--emit", "sample", "multi.ttpf"]);
    assert!(out.contains("TOTAL bc1: 2/2 converged"), "{out}");
    assert!(dir.join("sample/multi_0.bc1").is_file());
    assert!(!dir.join("sample/multi_2.bc1").exists());
    assert!(dir.join("sample/multi_4.bc1").is_file());
    for name in ["bc3", "bc4", "bc5", "bc7"] {
        assert!(
            out.contains(&format!("TOTAL {name}: 1/1 converged")),
            "{name}: {out}"
        );
    }
    let out = run(&["scan", "--max-blocks", "256", "multi.ttpf"]);
    assert!(out.contains("TOTAL 3/3 converged"), "{out}");
    for name in ["bc3", "bc5", "bc7"] {
        assert!(
            out.contains(&format!("TOTAL {name}: 1/1 converged")),
            "{name}: {out}"
        );
    }
    fs::remove_dir_all(&dir).unwrap();
}
