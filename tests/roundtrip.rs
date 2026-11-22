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
        for k in 0..4 {
            c[k] = ((s >> (2 * k)) & 3) as u8;
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
        for l in 0..12 {
            assert_eq!(out.lane(l), &syms[l][..], "seed {seed} lane {l}");
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
