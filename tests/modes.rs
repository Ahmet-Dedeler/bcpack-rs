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
        if mode == Mode::Bc3 {
            let mut seen_q = std::collections::HashSet::new();
            for s in 0..256 {
                assert!(t.quads.known[s]);
                assert!(seen_q.insert(t.quads.quads[s]));
            }
        }
        assert_eq!(t.dictionary.len(), 255);
        for (k, &(_, unk)) in t.dictionary.iter().enumerate() {
            assert_eq!(
                unk,
                bcn::Unknown::default(),
                "{} dictionary entry {k} has unknown fields",
                mode.name()
            );
        }
    }
    let t = bc7::Tables::default_table();
    let known: Vec<Control> = t.control.iter().flatten().copied().collect();
    let distinct: std::collections::HashSet<Control> = known.iter().copied().collect();
    assert_eq!(known.len(), distinct.len());
    assert_eq!(known.len(), 285);
    for c in &known {
        assert!(Control::all().contains(c));
    }
    assert_eq!(t.dictionary.len(), 255);
    for (k, (st, kn)) in t.dictionary.iter().enumerate() {
        assert_eq!(st.subsets, 1, "bc7 dictionary entry {k} subsets");
        assert!(kn.subsets, "bc7 dictionary entry {k} kn.subsets");
        assert_eq!(kn.ep, [[8; 8]; 3], "bc7 dictionary entry {k} kn.ep");
        assert_eq!(kn.ix, [[4; 16]; 2], "bc7 dictionary entry {k} kn.ix");
    }
}

#[test]
fn bc7_dictionary_multiset_not_unfitted() {
    let t = bc7::Tables::default_table();
    let c7 = Control {
        mode: 7,
        partition: 0,
        rotation: 0,
        index_selection: 0,
    };
    let c0 = Control {
        mode: 0,
        partition: 0,
        rotation: 0,
        index_selection: 0,
    };
    let code7 = t.code_of(c7).expect("mode 7 control code");
    let code0 = t.code_of(c0).expect("mode 0 control code");

    let n = 3;
    let mut l = lanes(16, 2 * n);
    let set = |l: &mut LaneOutput, lane: usize, i: usize, half: usize, v: u16| {
        l.symbols[lane * 2 * n + 2 * i + half] = v;
    };
    let set_ctl = |l: &mut LaneOutput, i: usize, code: usize| {
        l.symbols[15 * 2 * n + 2 * i] = (code & 31) as u16;
        l.symbols[15 * 2 * n + 2 * i + 1] = ((code >> 5) & 15) as u16;
    };
    let set_refs = |l: &mut LaneOutput, i: usize, ep_ref: usize, ix_ref: usize| {
        let (d, x) = format::reference_codes(ep_ref, ix_ref);
        let (de, do_) = format::deinterleave(d);
        let (xe, xo) = format::deinterleave(x);
        l.symbols[2 * i] = de;
        l.symbols[2 * i + 1] = do_;
        l.symbols[2 * n + 2 * i] = xe;
        l.symbols[2 * n + 2 * i + 1] = xo;
    };

    // block 0: 2-subset mode 7, both references hit dictionary entry 4
    set_ctl(&mut l, 0, code7);
    set_refs(&mut l, 0, 5, 5);
    // block 1: 3-subset mode 0, endpoint ref hits dictionary entry 10, index ref hits block 0
    set_ctl(&mut l, 1, code0);
    set_refs(&mut l, 1, 12, 1);
    // block 2: 2-subset mode 7, both references hit block 1 (checks no transitive unfitted bit)
    set_ctl(&mut l, 2, code7);
    set_refs(&mut l, 2, 1, 1);
    set(&mut l, 2, 2, 0, format::zigzag(1));

    let out = bc7::assemble(&l, t);
    assert_eq!(out.unknown_controls, 0);
    assert_eq!(out.dictionary_refs, 3);
    assert_eq!(out.unfitted_blocks, 0);
    for i in 0..n {
        let blk = &out.blocks[i * 16..(i + 1) * 16];
        let (c, st) = bc7::unpack(blk).expect("valid bc7 block");
        let mut repacked = [0u8; 16];
        bc7::pack(c, &st, &mut repacked);
        assert_eq!(&repacked[..], blk);
    }

    // Partial custom table with unknown subsets still flags multi-subset blocks as unfitted.
    let mut partial = t.clone();
    partial.dictionary[4].0.subsets = 0;
    partial.dictionary[4].1.subsets = false;
    let out_partial = bc7::assemble(&l, &partial);
    assert_eq!(out_partial.unfitted_blocks, 1);
}

#[test]
fn synthetic_streams_round_trip() {
    let window = bcpack::bc1::Window::default_table();
    let quads = bcpack::bc1::QuadTable::default_table();
    for mode in [Mode::Bc3, Mode::Bc4, Mode::Bc5, Mode::Bc7] {
        for (blocks, fragment) in [(1, 1), (67, 5), (517, 900)] {
            let (mut stream, symbols) = bcpack::encode::synthetic(mode, 7, blocks, fragment);
            let consumed = stream.len();
            stream.extend_from_slice(&[0; 13]);
            let mut expected = lanes(
                mode.alphabet_bits().len(),
                blocks * mode.symbols_per_block(),
            );
            expected.symbols = symbols.concat();
            let decoded = mode.decode_symbols(&stream, blocks).unwrap();
            assert_eq!(decoded.symbols, expected.symbols);
            assert_eq!(decoded.consumed, consumed);
            let reference = if mode == Mode::Bc7 {
                let r = bc7::assemble(&expected, bc7::Tables::default_table());
                assert_eq!(r.unknown_controls, 0);
                r.blocks
            } else {
                bcn::assemble(&expected, Tables::default_for(mode).unwrap()).blocks
            };
            let rebuilt = bcpack::decode_chunk(mode, &stream, blocks, &window, &quads).unwrap();
            assert_eq!(rebuilt.blocks, reference);
            assert_eq!(rebuilt.blocks.len(), blocks * mode.block_bytes());
            assert_eq!(rebuilt.unfitted_blocks, 0);
            stream.push(1);
            assert!(mode.decode_symbols(&stream, blocks).is_err());
        }
    }
}

#[cfg(feature = "ttpf")]
#[test]
fn compressed_mip_tail_to_dx10_dds() {
    use bcpack::ttpf::{self, Chunk, Store, Texture};
    let window = bcpack::bc1::Window::default_table();
    let quads = bcpack::bc1::QuadTable::default_table();
    for (mode, dxgi) in [
        (Mode::Bc3, 78),
        (Mode::Bc4, 81),
        (Mode::Bc5, 84),
        (Mode::Bc7, 99),
    ] {
        for tile_mode in [0, 1] {
            let tile = if tile_mode == 0 { 65536 } else { 4096 };
            let stride = mode.block_bytes();
            let (stream, _) = bcpack::encode::synthetic(mode, 17, tile / stride, 128);
            let raw = bcpack::decode_chunk(mode, &stream, tile / stride, &window, &quads)
                .unwrap()
                .blocks;
            let chunk = Chunk {
                index: 0,
                file_off: 0,
                file_len: stream.len() as u64,
                pad: 0,
                vstart: 0,
                vlen: tile as u64,
                codec: mode.codec(),
            };
            let store = Store {
                data: stream,
                vsize: tile as u64,
                chunks: vec![chunk],
                etpf_off: 0,
            };
            let tex = Texture {
                index: 0,
                hash: 0,
                width: 5,
                height: 7,
                mips: 2,
                dxgi,
                array_size: 1,
                tile_mode,
                vstart: 0,
            };
            let mip = bcpack::texture_mip0(&store, &tex, &window, &quads).unwrap();
            assert_eq!(
                (mip.width, mip.height, mip.bw, mip.bh, mip.bcpack_chunks),
                (5, 7, 2, 2, 1)
            );
            // The first tail level starts halfway through the tile. Low address bits
            // are y0,y1,x0,x1 for 4 KiB/16 B and x0,y0 for the other tested layouts.
            let indices = if tile_mode == 1 && stride == 16 {
                [0, 4, 1, 5]
            } else {
                [0, 1, 2, 3]
            };
            let mut expected = Vec::new();
            for index in indices {
                let offset = tile / 2 + index * stride;
                expected.extend_from_slice(&raw[offset..offset + stride]);
            }
            assert_eq!(mip.blocks, expected);
            let dds = ttpf::dds(&mip.blocks, mip.width, mip.height, dxgi).unwrap();
            let word = |i| u32::from_le_bytes(dds[i..i + 4].try_into().unwrap());
            assert_eq!((word(12), word(16), word(20)), (7, 5, 4 * stride as u32));
            assert_eq!(&dds[84..88], b"DX10");
            assert_eq!(
                [word(128), word(132), word(136), word(140), word(144)],
                [dxgi, 3, 0, 1, 0]
            );
            assert_eq!(dds[148..], expected);
        }
    }
}

#[cfg(feature = "cli")]
#[test]
fn cli_decodes_prints_and_benchmarks_each_format() {
    use std::{fs, process::Command};
    let dir = std::env::temp_dir().join(format!("bcpack-cli-{}", std::process::id()));
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
    for mode in [Mode::Bc3, Mode::Bc4, Mode::Bc5, Mode::Bc7] {
        let name = mode.name();
        let (stream, syms) = bcpack::encode::synthetic(mode, 19, 3, 4);
        fs::write(dir.join("chunk"), &stream).unwrap();
        run(&[&format!("decode-{name}"), "chunk", "3", "blocks"]);
        let w = bcpack::bc1::Window::default_table();
        let q = bcpack::bc1::QuadTable::default_table();
        let expected = bcpack::decode_chunk(mode, &stream, 3, &w, &q).unwrap();
        assert_eq!(fs::read(dir.join("blocks")).unwrap(), expected.blocks);
        let printed = run(&["symbols", "--format", name, "chunk", "3"]);
        let lanes: Vec<Vec<u16>> = printed
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|s| s.parse().unwrap())
                    .collect()
            })
            .collect();
        assert_eq!(lanes, syms);
        assert!(run(&["bench", "--format", name, "chunk", "3", "0.001"])
            .contains(&format!("MB/s {} out", name.to_ascii_uppercase())));
    }
    fs::remove_file(dir.join("chunk")).unwrap();
    fs::remove_file(dir.join("blocks")).unwrap();
    fs::remove_dir(dir).unwrap();
}
