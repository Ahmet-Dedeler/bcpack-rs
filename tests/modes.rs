// bcpack-rs by Ahmet Dedeler (https://github.com/Ahmet-Dedeler, https://ahmetdedeler.com)
// Source: https://github.com/Ahmet-Dedeler/bcpack-rs
// SPDX-License-Identifier: MIT OR Apache-2.0
// If you copy, vendor or port this file, keep this header and credit bcpack-rs in your README.
//! BC3, BC4, BC5 and BC7 block rebuild: synthetic checks only (no game bytes).

use bcpack::bc7::{self, Control};
use bcpack::bcn::{self, Tables};
use bcpack::encode::Rng;
use bcpack::format::{self, Mode};
use bcpack::rans::LaneOutput;

fn lanes(n_lanes: usize, target: usize) -> LaneOutput {
    LaneOutput {
        lanes: n_lanes,
        target,
        symbols: vec![0; n_lanes * target],
        consumed: 0,
        cycles: 0,
        stats: vec![],
    }
}

#[test]
fn bcn_pack_unpack_round_trip() {
    let mut rng = Rng(11);
    for layout in [&bcn::BC3, &bcn::BC4, &bcn::BC5] {
        let bb = layout.mode.block_bytes();
        for _ in 0..2000 {
            let block: Vec<u8> = (0..bb).map(|_| rng.below(256) as u8).collect();
            let st = bcn::unpack(layout, &block);
            let mut out = vec![0u8; bb];
            bcn::pack(layout, &st, &mut out);
            assert_eq!(out, block, "{}", layout.mode.name());
        }
    }
}

#[test]
fn bc7_pack_unpack_round_trip() {
    let mut rng = Rng(12);
    let mut seen = [0usize; 8];
    for _ in 0..20000 {
        let mut block: Vec<u8> = (0..16).map(|_| rng.below(256) as u8).collect();
        let mode = rng.below(8) as u32;
        block[0] = (block[0] & !((1u16 << (mode + 1)) - 1) as u8) | 1 << mode;
        let (c, st) = bc7::unpack(&block).expect("valid mode");
        assert_eq!(c.mode as u32, mode);
        seen[mode as usize] += 1;
        let mut out = [0u8; 16];
        bc7::pack(c, &st, &mut out);
        assert_eq!(&out[..], &block[..], "mode {mode}");
    }
    assert!(seen.iter().all(|&n| n > 0));
    assert!(bc7::unpack(&[0u8; 16]).is_none());
    assert_eq!(Control::all().len(), 285);
}

#[test]
fn bc4_references_by_hand() {
    // Pair table: symbol s -> (s & 7, s >> 3); empty dictionary except entry 0.
    let mut pairs = [[0u8; 2]; 64];
    for (s, p) in pairs.iter_mut().enumerate() {
        *p = [(s & 7) as u8, (s >> 3) as u8];
    }
    let mut t = Tables::empty(&bcn::BC4);
    t.pairs = bcn::PairTable::from_pairs(pairs);
    t.dictionary[0].0.ep[0] = 100;
    t.dictionary[0].1 = bcn::Unknown { ep: 2, ix: !0 };
    let n = 3;
    let mut l = lanes(12, n);
    let mut set = |lane: usize, i: usize, v: u16| l.symbols[lane * n + i] = v;
    // block 0: no references, e0 = +10, e1 = -3, pixel pair k = (k, 7 - k)
    set(2, 0, format::zigzag(10));
    set(3, 0, format::zigzag(-3));
    for k in 0..8 {
        set(4 + k, 0, (k | (7 - k) << 3) as u16);
    }
    // block 1: both references = block 0; e0 +1, every position +1
    let (d, x) = format::reference_codes(1, 1);
    set(0, 1, d as u16);
    set(1, 1, x as u16);
    set(2, 1, format::zigzag(1));
    for k in 0..8 {
        set(4 + k, 1, 1 | 1 << 3);
    }
    // block 2: endpoint reference 3 = dictionary entry 0 (block -1), no index reference
    let (d, x) = format::reference_codes(3, 0);
    set(0, 2, d as u16);
    set(1, 2, x as u16);
    let out = bcn::assemble(&l, &t);
    let st = |i: usize| bcn::unpack(&bcn::BC4, &out.blocks[i * 8..i * 8 + 8]);
    assert_eq!(out.blocks[0], 10 ^ 0x80);
    assert_eq!(out.blocks[1], (-3i8 as u8) ^ 0x80);
    assert_eq!(st(0).ix[..4], [0, 7, 1, 6]);
    assert_eq!(st(1).ep[..2], [11, (-3i8) as u8]);
    assert_eq!(st(1).ix[..4], [1, 0, 2, 7]);
    assert_eq!(st(2).ep[0], 100);
    assert_eq!(out.dictionary_refs, 1);
    // entry 0's e1 is unknown, so block 2 is counted
    assert_eq!(out.unfitted_blocks, 1);
}

#[test]
fn bc7_mode6_by_hand() {
    let mut t = bc7::Tables::empty();
    let c6 = Control {
        mode: 6,
        ..Default::default()
    };
    t.control[bc7::control_code(20, 0)] = Some(c6);
    let n = 2;
    let mut l = lanes(16, 2 * n);
    let mut set = |lane: usize, i: usize, half: usize, v: u16| l.symbols[lane * 2 * n + 2 * i + half] = v;
    for i in 0..n {
        set(15, i, 0, 20);
    }
    // block 0: R0 = 0x55 (8 bits with the p-bit), G0 = 3 (7 bits); first pixel index +5
    set(2, 0, 0, format::zigzag(0x55));
    set(3, 0, 0, format::zigzag(3));
    set(11, 0, 0, bc7::index_symbol(format::zigzag(5), 0, 0));
    // block 1: both references = block 0, R0 + 1, first pixel index - 1
    let (d, x) = format::reference_codes(1, 1);
    let (de, do_) = format::deinterleave(d);
    let (xe, xo) = format::deinterleave(x);
    set(0, 1, 0, de);
    set(0, 1, 1, do_);
    set(1, 1, 0, xe);
    set(1, 1, 1, xo);
    set(2, 1, 0, format::zigzag(1));
    set(11, 1, 0, bc7::index_symbol(format::zigzag(-1), 0, 0));
    let out = bc7::assemble(&l, &t);
    assert_eq!(out.unknown_controls, 0);
    assert_eq!(out.unfitted_blocks, 0);
    let (c, s0) = bc7::unpack(&out.blocks[..16]).unwrap();
    assert_eq!(c, c6);
    assert_eq!(s0.ep[0][0], 0x55);
    assert_eq!(s0.ep[0][2], 3 << 1);
    assert_eq!(s0.ix[0][0], 5);
    let (_, s1) = bc7::unpack(&out.blocks[16..]).unwrap();
    assert_eq!(s1.ep[0][0], 0x56);
    assert_eq!(s1.ep[0][2], 3 << 1);
    assert_eq!(s1.ix[0][0], 4);
}

#[test]
fn shipped_tables_load() {
    for mode in [Mode::Bc3, Mode::Bc4, Mode::Bc5] {
        let t = Tables::default_for(mode).unwrap();
        // the pair tables are complete bijections
        let mut seen = std::collections::HashSet::new();
        for s in 0..64 {
            assert!(t.pairs.known[s]);
            assert!(seen.insert(t.pairs.pairs[s]));
        }
        assert_eq!(t.dictionary.len(), 255);
    }
    let t = bc7::Tables::default_table();
    let known: Vec<Control> = t.control.iter().flatten().copied().collect();
    let distinct: std::collections::HashSet<Control> = known.iter().copied().collect();
    assert_eq!(known.len(), distinct.len());
    assert!(known.len() >= 205);
    for c in &known {
        assert!(Control::all().contains(c));
    }
}
