// bcpack-rs by Ahmet Dedeler (https://github.com/Ahmet-Dedeler, https://ahmetdedeler.com)
// Source: https://github.com/Ahmet-Dedeler/bcpack-rs
// SPDX-License-Identifier: MIT OR Apache-2.0
// If you copy, vendor or port this file, keep this header and credit bcpack-rs in your README.
//! Synthetic round-trip tests, table checksum and golden hashes.
//! Everything here is generated data; no game bytes.

use bcpack::bc1::{self, QuadTable, Window, BC1_ALPHABET_BITS};
use bcpack::encode::{self, Fragment};
use bcpack::rans::{self, DecodeError, SliceSource};
use bcpack::tables;
use sha2::{Digest, Sha256};

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn verilog_table_checksum() {
    // Canonical text: one line per table (base, freq, rf, rs, rn), values comma
    // separated in [eab][q][seg] order. Same definition as tools/gen_tables.py.
    let mut text = String::new();
    fn line<T: ToString + Copy>(t: &[[[T; 10]; 16]; 9], out: &mut String) {
        let v: Vec<String> = t
            .iter()
            .flatten()
            .flatten()
            .map(|x| x.to_string())
            .collect();
        out.push_str(&v.join(","));
        out.push('\n');
    }
    line(&tables::BASE_TABLE, &mut text);
    line(&tables::FREQ_TABLE, &mut text);
    line(&tables::RF_TABLE, &mut text);
    line(&tables::RS_TABLE, &mut text);
    line(&tables::RN_TABLE, &mut text);
    assert_eq!(hex(&Sha256::digest(text.as_bytes())), tables::TABLES_SHA256);
}

/// Window and codebook used by the golden tests, independent of the fitted data files.
fn test_tables() -> (Window, QuadTable) {
    let mut w = Window {
        slots: [[0; 6]; 256],
        fitted: [true; 256],
    };
    for s in 0..256 {
        for k in 0..6 {
            w.slots[s][k] = ((s * 7 + k * 13) % (1 << bc1::ENDPOINT_WIDTHS[k])) as u8;
        }
    }
    let mut codes = [[0u8; 4]; 256];
    for (s, c) in codes.iter_mut().enumerate() {
        for (k, code) in c.iter_mut().enumerate() {
            *code = ((s >> (2 * k)) & 3) as u8;
        }
    }
    (w, QuadTable::from_codes(codes))
}

#[test]
fn synthetic_round_trip_many() {
    for seed in 1..=40u64 {
        let blocks = [1usize, 2, 7, 64, 300, 1024, 1536][seed as usize % 7];
        let max_frag = [3usize, 40, 200, 2000][seed as usize % 4];
        let (stream, syms) = encode::synthetic_bc1(seed, blocks, max_frag);
        let out =
            bc1::decode_symbols(&stream, blocks).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        assert_eq!(out.consumed, stream.len());
        for (l, expected) in syms.iter().enumerate() {
            assert_eq!(out.lane(l), expected, "seed {seed} lane {l}");
        }
        // The reference (generic) clocked model must agree with the fast one.
        let mut src = SliceSource {
            data: &stream,
            pos: 0,
        };
        let r = rans::decode_lanes(&mut src, &BC1_ALPHABET_BITS, blocks, stream.len()).unwrap();
        assert_eq!(r.symbols, out.symbols);
        assert_eq!(r.cycles, out.cycles);
    }
}

/// Compare both clocked decoders, including the fields that do not affect the
/// reconstructed texture. Return the oracle's result for fixture assertions.
fn compare_decoders(data: &[u8], bits: &[u8], target: usize) -> Result<(), DecodeError> {
    let mut src = SliceSource { data, pos: 0 };
    let reference = rans::decode_lanes(&mut src, bits, target, data.len());
    let fast = rans::decode_lanes_fast(data, bits, target);
    assert_eq!(
        fast,
        reference,
        "{} lanes, {target} symbols, {} bytes",
        bits.len(),
        data.len()
    );
    reference.map(|_| ())
}

fn split_fragments(rng: &mut encode::Rng, ab: u8, blocks: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut left = blocks;
    while left > 0 {
        let count = left.min(1 + rng.below(1024) as usize);
        let q = rng.below(16) as u8;
        let z = rng.below(4).min(ab as u64) as u8;
        let symbols = (0..count)
            .map(|_| encode::sample_symbol(rng, ab - z, q))
            .collect();
        let frag = encode::encode_fragment(ab, &Fragment { symbols, q, z }).unwrap();
        let flags = frag[0] & !0x30;
        let payload = &frag[if frag[0] & 0x20 != 0 { 3 } else { 2 }..];
        // Split an encoded payload without flushing its state. This also
        // exercises DRAIN with a partially loaded initial state.
        let cut = 1 + rng.below((payload.len() - 1).min(256) as u64) as usize;
        // Up to three starting bytes cannot emit a symbol. The next header
        // can select a different model while retaining this partial state.
        let first_flags = if cut <= 3 { flags ^ 0x0f } else { flags };
        bytes.extend([first_flags, (cut - 1) as u8]);
        bytes.extend_from_slice(&payload[..cut]);
        let tail = payload.len() - cut;
        if tail > 256 {
            bytes.push(flags | 0x30);
            bytes.extend_from_slice(&((tail - 257) as u16).to_le_bytes());
        } else {
            bytes.extend([flags | 0x10, (tail - 1) as u8]);
        }
        bytes.extend_from_slice(&payload[cut..]);
        left -= count;
    }
    bytes
}

#[test]
fn clocked_decoders_agree_across_fragment_boundaries() {
    let wide = [4, 4, 8, 8, 8, 8, 8, 7, 7, 5, 6, 8, 8, 8, 8, 5];
    let mut rng = encode::Rng(0xe7c10c);
    for n in 1..=16 {
        let varied: Vec<_> = (0..n).map(|_| rng.below(9) as u8).collect();
        let bits = match n {
            12 => &BC1_ALPHABET_BITS[..],
            16 => &wide[..],
            _ => &varied,
        };
        for blocks in [1, 19, 256, 1024] {
            let lanes: Vec<_> = bits
                .iter()
                .map(|&ab| split_fragments(&mut rng, ab, blocks))
                .collect();
            let data = encode::interleave(bits, blocks, &lanes).unwrap();
            compare_decoders(&data, bits, blocks).unwrap();
            for len in [0, 1, data.len() / 2, data.len() - 1] {
                compare_decoders(&data[..len], bits, blocks).unwrap_err();
            }
            for _ in 0..8 {
                let mut damaged = data.clone();
                let pos = rng.below(data.len() as u64) as usize;
                damaged[pos] ^= 1 << rng.below(8);
                let _ = compare_decoders(&damaged, bits, blocks);
            }
        }
    }
}

#[test]
fn clocked_decoders_agree_on_invalid_inputs() {
    for (bits, target) in [
        (&[][..], 1),
        (&[0; 17][..], 1),
        (&[9][..], 1),
        (&[0][..], 0),
        (&[0][..], usize::MAX),
    ] {
        assert!(matches!(
            compare_decoders(&[], bits, target),
            Err(DecodeError::Invalid(_))
        ));
    }
    assert_eq!(
        compare_decoders(&[0xc0], &[0], 1),
        Err(DecodeError::BadHeader { lane: 0, eab: 9 })
    );
    assert_eq!(
        compare_decoders(&[0x10, 0, 1], &[0], 1),
        Err(DecodeError::NoProgress { consumed: 3 })
    );
    assert_eq!(
        compare_decoders(&[0x10, 3, 0xff, 0xff, 0xff, 0xff], &[0], 1),
        Err(DecodeError::IllegalEndState { lanes: vec![0] })
    );
    // Steady-state PROC lane stalled at x == L with a prefetched byte and
    // >= 8 bytes remaining: must trigger NoProgress inside the PROC burst.
    assert_eq!(
        compare_decoders(
            &[0x10, 0x0c, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0],
            4
        ),
        Err(DecodeError::NoProgress { consumed: 7 })
    );
}

#[test]
fn padding_truncation_and_garbage() {
    let (mut stream, _) = encode::synthetic_bc1(99, 512, 100);
    let n = stream.len();
    // zero padding after the stream is fine
    stream.extend([0u8; 300]);
    assert_eq!(bc1::decode_symbols(&stream, 512).unwrap().consumed, n);
    // non-zero bytes after it are not
    stream[n + 10] = 1;
    assert!(matches!(
        bc1::decode_symbols(&stream, 512),
        Err(DecodeError::TrailingData { .. })
    ));
    // truncation fails, the wrong block count fails
    assert!(bc1::decode_symbols(&stream[..n - 5], 512).is_err());
    assert!(bc1::decode_symbols(&stream[..n], 511).is_err());
    assert!(bc1::decode_symbols(&stream[..n], 513).is_err());
}

#[test]
fn hand_built_fragments() {
    // Lanes with explicit models, narrowing and multi-fragment layouts,
    // including 3-byte headers (fragments longer than 256 bytes).
    let mut rng = encode::Rng(7);
    let blocks = 3000;
    let mut lanes = Vec::new();
    for (i, &ab) in BC1_ALPHABET_BITS.iter().enumerate() {
        let parts: Vec<usize> = match i % 3 {
            0 => vec![blocks],
            1 => vec![1, 999, 2000],
            _ => vec![1500, 1500],
        };
        let mut frags = Vec::new();
        for (j, n) in parts.into_iter().enumerate() {
            let z = (j as u8) % 2;
            let eab = encode::header_model(ab, z).unwrap();
            let q = ((i * 5 + j * 3) % 16) as u8;
            let symbols = (0..n)
                .map(|_| encode::sample_symbol(&mut rng, eab, q))
                .collect();
            frags.push(Fragment { symbols, q, z });
        }
        lanes.push(frags);
    }
    let stream = encode::encode_lanes(&BC1_ALPHABET_BITS, &lanes).unwrap();
    let out = bc1::decode_symbols(&stream, blocks).unwrap();
    for (l, frags) in lanes.iter().enumerate() {
        let want: Vec<u16> = frags.iter().flat_map(|f| f.symbols.clone()).collect();
        assert_eq!(out.lane(l), &want[..]);
    }
}

#[test]
fn semantics_by_hand() {
    // Build lanes directly: block 0 absolute, block 1 refs block 0 (+1 on R0),
    // block 2 refs the window slot (2 - 255) mod 256 = 3.
    let (w, q) = test_tables();
    let n = 3;
    let mut sym = vec![0u16; 12 * n];
    let set = |sym: &mut Vec<u16>, lane: usize, i: usize, v: u16| sym[lane * n + i] = v;
    for k in 0..6 {
        set(&mut sym, 2 + k, 0, 4); // +2 on every field
    }
    set(&mut sym, 0, 1, 2); // d = 1
    set(&mut sym, 2, 1, 2); // R0 +1
    set(&mut sym, 0, 2, 1); // d = 255
    set(&mut sym, 3, 2, 1); // G0 -1
    for l in 8..12 {
        set(&mut sym, l, 0, 0b11_10_01_00); // TL TR BL BR codes 0 1 2 3
    }
    let lanes = rans::LaneOutput {
        lanes: 12,
        target: n,
        symbols: sym,
        consumed: 0,
        cycles: 0,
        stats: vec![],
    };
    let out = bc1::assemble(&lanes, &w, &q);
    let blk = |i: usize| {
        let b = &out.blocks[i * 8..i * 8 + 8];
        (
            u16::from_le_bytes([b[0], b[1]]),
            u16::from_le_bytes([b[2], b[3]]),
            u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        )
    };
    assert_eq!(blk(0).0, 2 << 11 | 2 << 5 | 2);
    assert_eq!(blk(1).0, 3 << 11 | 2 << 5 | 2);
    let s3 = w.slots[3];
    assert_eq!(
        blk(2).0,
        (s3[0] as u16) << 11 | ((s3[1] as u16 + 63) % 64) << 5 | s3[2] as u16
    );
    // every quadrant holds codes 0 1 / 2 3 -> rows 0 1 0 1, 2 3 2 3, ...
    assert_eq!(blk(0).2, 0b11_10_11_10_01_00_01_00_11_10_11_10_01_00_01_00);
    assert_eq!(out.prestart_refs, 1);
}

#[test]
fn prediction_chains_cross_history_wraps() {
    let (mut w, _) = test_tables();
    w.fitted[1] = false;
    let q = QuadTable::from_codes([[0; 4]; 256]);
    let n = 1025;
    let mut symbols = vec![0u16; 12 * n];
    symbols[..n].fill(1); // d = 255, the longest reference distance.
    symbols[2 * n..8 * n].fill(2); // +1 on every endpoint field.
    let lanes = rans::LaneOutput {
        lanes: 12,
        target: n,
        symbols,
        consumed: 0,
        cycles: 0,
        stats: vec![],
    };
    let out = bc1::assemble(&lanes, &w, &q);
    for (i, block) in out.blocks.chunks(8).enumerate() {
        // Each chain starts in slots 1..=255 and gains one every 255 blocks.
        let slot = i % 255 + 1;
        let steps = (i / 255 + 1) as u16;
        let mut fields = [0u16; 6];
        for (k, field) in fields.iter_mut().enumerate() {
            *field = (w.slots[slot][k] as u16 + steps) & ((1 << bc1::ENDPOINT_WIDTHS[k]) - 1);
        }
        let c0 = fields[0] << 11 | fields[1] << 5 | fields[2];
        let c1 = fields[3] << 11 | fields[4] << 5 | fields[5];
        assert_eq!(&block[..2], &c0.to_le_bytes(), "block {i}");
        assert_eq!(&block[2..4], &c1.to_le_bytes(), "block {i}");
        assert_eq!(&block[4..], &[0; 4], "block {i}");
    }
    assert_eq!(out.blocks.len(), n * 8);
    assert_eq!(out.prestart_refs, 255);
    assert_eq!(out.unfitted_blocks, 5); // Blocks 0, 255, 510, 765 and 1020.
}

#[test]
fn golden_hashes() {
    // Synthetic streams decoded with the test tables. A change here means the
    // decoder's output changed.
    let (w, q) = test_tables();
    let mut got = Vec::new();
    for &(seed, blocks) in &[(1u64, 256usize), (2, 1024), (3, 4096)] {
        let (stream, _) = encode::synthetic_bc1(seed, blocks, 500);
        let out = bc1::decode(&stream, blocks, &w, &q).unwrap();
        got.push(format!(
            "{seed}:{blocks}:{}",
            hex(&Sha256::digest(&out.blocks))
        ));
    }
    let want = include_str!("golden.txt")
        .lines()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    if std::env::var("BCPACK_BLESS").is_ok() {
        std::fs::write(
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden.txt"),
            got.join("\n") + "\n",
        )
        .unwrap();
        return;
    }
    assert_eq!(got, want);
}

#[test]
fn default_tables_load() {
    let w = Window::default_table();
    assert!(w.fitted_count() >= 16);
    let q = QuadTable::default_table();
    let mut seen = std::collections::HashSet::new();
    for c in q.codes.iter() {
        assert!(seen.insert(*c), "codebook must be a bijection");
    }
    // symbols 0-3 are the uniform quadrants, code = sym ^ 2 (measured exactly)
    for s in 0..4 {
        assert_eq!(q.codes[s], [(s ^ 2) as u8; 4]);
    }
}

/// Writes tests/vectors/*.json for python/bcpack.py selftest (run with BCPACK_BLESS=1).
#[test]
fn python_vectors() {
    let (w, q) = test_tables();
    for &(seed, blocks) in &[(11u64, 64usize), (12, 700)] {
        let (stream, syms) = encode::synthetic_bc1(seed, blocks, 150);
        let out = bc1::decode(&stream, blocks, &w, &q).unwrap();
        let path = format!(
            "{}/tests/vectors/synthetic_{seed}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let doc = serde_json::json!({
            "about": "synthetic BCPack BC1 stream from src/encode.rs; bc1_sha256 uses the test tables of tests/roundtrip.rs",
            "blocks": blocks,
            "chunk_hex": hex(&stream),
            "symbols": syms,
            "bc1_sha256": hex(&Sha256::digest(&out.blocks)),
        });
        if std::env::var("BCPACK_BLESS").is_ok() {
            std::fs::write(&path, serde_json::to_string(&doc).unwrap()).unwrap();
        } else {
            let have: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            assert_eq!(have, doc, "{path} is stale (BCPACK_BLESS=1 cargo test)");
        }
    }
}

#[test]
fn validation_and_error_guards() {
    // alphabet_bits > 8 must be rejected cleanly
    assert!(matches!(
        rans::decode_lanes_fast(&[0u8; 8], &[16], 1),
        Err(rans::DecodeError::Invalid(_))
    ));
    // encoder rejects invalid z / q instead of producing corrupt headers or panicking
    let bad_q = encode::Fragment {
        symbols: vec![0],
        q: 16,
        z: 0,
    };
    assert!(encode::encode_fragment(7, &bad_q).is_err());
    let bad_z = encode::Fragment {
        symbols: vec![0],
        q: 0,
        z: 4,
    };
    assert!(encode::encode_fragment(7, &bad_z).is_err());
    assert!(encode::encode_lanes(&[7], &[]).is_err());
    assert!(encode::interleave(&[7, 7], 1, &[vec![]]).is_err());
}

#[cfg(feature = "ttpf")]
#[test]
fn ttpf_codec_mask_high_file_offset_and_mip_bounds() {
    // Entry 0 sits at file_off = 0x1_0000_0000 (bit 24 set in `a`) with CODEC_BC1 (0x04).
    // Bits 24..25 belong to file_off, so codec must stay 0x04, not 0x05.
    let mut data = vec![0u8; 0x18 + 16];
    data[..4].copy_from_slice(b"RDS5");
    data[4..8].copy_from_slice(&2u32.to_le_bytes());
    let a0: u32 = (1u32 << 24) | ((bcpack::ttpf::CODEC_BC1 as u32) << 24);
    let a1: u32 = (1u32 << 24) | 1u32;
    data[0x18..0x1C].copy_from_slice(&a0.to_le_bytes());
    data[0x1C..0x20].copy_from_slice(&0u32.to_le_bytes());
    data[0x20..0x24].copy_from_slice(&a1.to_le_bytes());
    data[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
    let store = bcpack::ttpf::Store::from_bytes(data).unwrap();
    assert_eq!(store.chunks.len(), 1);
    assert_eq!(store.chunks[0].file_off, 0x1_0000_0000);
    assert_eq!(store.chunks[0].codec, bcpack::ttpf::CODEC_BC1);
    assert!(store.chunk_bytes(99, true).is_err());

    // Invalid mip inputs must return None, not panic on shift overflow
    assert!(bcpack::ttpf::mip_chain_bc1(0, 0, 64, 1).is_none());
    assert!(bcpack::ttpf::mip_chain_bc1(0, 64, 0, 1).is_none());
    assert!(bcpack::ttpf::mip_chain_bc1(0, 64, 64, 33).is_none());
}

/// BC1 lane 1 is the index reference and the endpoint reference is that
/// plus lane 0's signed delta (issue #10, tyr0nsur's two-block case).
#[test]
fn bc1_lane1_index_reference() {
    use bcpack::{bc1, encode};
    let run = |reference: u16| {
        let lanes: Vec<_> = (0..12)
            .map(|i| {
                vec![encode::Fragment {
                    symbols: vec![
                        if i >= 2 { 2 + i as u16 % 4 } else { 0 },
                        if i == 1 { reference } else { 0 },
                    ],
                    q: 7,
                    z: 0,
                }]
            })
            .collect();
        let stream = encode::encode_lanes(&bc1::BC1_ALPHABET_BITS, &lanes).unwrap();
        bc1::decode(&stream, 2, &bc1::Window::default_table(), &bc1::QuadTable::default_table())
            .unwrap()
            .blocks
    };
    let first = [0xa1, 0x17, 0x5d, 0xf0, 0x50, 0x50, 0xfa, 0xaf];
    let a = run(0);
    assert_eq!(&a[..8], &first);
    assert_eq!(&a[8..], &[0x00, 0x00, 0x00, 0x00, 0xaa, 0xaa, 0xaa, 0xaa]);
    let b = run(1);
    assert_eq!(&b[..8], &first);
    assert_eq!(&b[8..], &[0xa1, 0x17, 0x5d, 0xf0, 0x0a, 0x0a, 0x5f, 0xf5]);
}
