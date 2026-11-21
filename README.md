# bcpack-rs

Decoder for **BCPack**, the rANS entropy codec that the Xbox Series X|S Media Decompression Unit (MDU) uses for block-compressed (`BC1`-`BC7`) textures. Rust crate and CLI, a stdlib-only Python reference decoder, and a written spec ([`docs/BCPACK_SPEC.md`](docs/BCPACK_SPEC.md)).

Licensed under `MIT OR Apache-2.0`. The repository contains no game data; all tests use synthetic streams.

## Status

**BC1 decodes.** `bcpack decode-bc1` turns a BCPack BC1 chunk into plain BC1 blocks. The entropy layer is exact. The block layer uses two small constant tables fitted from data (a 256-slot preset endpoint window and a 256-entry index codebook); both are now essentially complete.

| Part | Status |
|---|---|
| Entropy layer (rANS) | **Exact.** Microsoft's MDU tables from the public [`silicon-rans`](https://github.com/microsoft/silicon-rans) Verilog (MIT), a clock-for-clock port of its lane decoder, bytes fed in lane order (8 per clock) with all 12 lanes in lockstep. Every lane must consume exactly its byte budget and end in a legal state, so a wrong rule fails loudly |
| BC1 endpoints | **Verified.** Lane 0 is a back-reference "copy block *i − d*", lanes 2-7 add zigzag corrections to R0 G0 B0 R1 G1 B1. References before the chunk start read a fixed 256-entry preset window |
| BC1 preset window | **Fitted.** 254 of 256 slots (`data/bc1_window.json`); slots 0 and 4 were never referenced in the fitting data. The decoder reports any block that depends on an unfitted slot |
| BC1 indices | **Fitted.** Each 2x2 quadrant is a symbol into a fixed, frequency-ordered table of all 256 index patterns (`data/bc1_quad_table.json`). Two independent fits agree on 248/256 entries (99.76% of quadrants); 8 rare mirror pairs are uncertain |
| BC1 block order | Standard Gfx10 tiled order (stream order = stored order) |
| BC7 | Entropy layer solved (research branches): 16 lanes, widths `4,4,8,8,8,8,8,7,7,5,6,8,8,8,8,5`. 4,600+ chunks converge exactly, 0 of 1,180 controls do. Block mapping (modes, partitions) in progress. Not in this decoder yet |
| BC6H, BC5, BC4, BC3 | Entropy layer solved (research branches): every chunk in the corpus converges exactly. Block mappings in progress. Not in this decoder yet |

### Verified numbers (this decoder)

All measured on retail Xbox Series texture pools with the commands in [Reproducing](#reproducing). No game data is in this repository.

| Check | Result |
|---|---|
| Exact convergence (`bcpack scan --every 20`, every 20th BC1 chunk of all 57 pools) | **5,162/5,162** chunks (97.9 M blocks): all 12 lanes end in a legal state with exactly `blocks` symbols, and the bytes consumed equal the stored length minus the pad in 5,162/5,162. 5,161 touch no unfitted preset-window slot |
| Rust vs the Python research pipeline (probe simulator + endpoint rule + codebook), byte for byte | **520/520** chunks identical, sampled round-robin across 52 stores (1,024-16,384 blocks), plus **12/12** chunks of 32,768-65,536 blocks |
| Mip consistency, 600 random pair textures (BCPack mip 0 decoded, downsampled 2x, compared with the plain mip 1) | 600/600 decode, none touching an unfitted window slot. Median **35.4 dB**, p10 30.2, p90 41.8 (flat mean-colour baseline 22.6); 3 below 25 dB, 18 not at least 3 dB above flat. With only the first 16 window slots the same set scored median 27.6, p10 7.3 |
| Held-out check of the fitted tables (fit on half the texture pools, decode the other half) | Preset window fitted on even pools, scored on odd: 1,514 textures, median **34.7 dB**, p10 29.7. Reverse: 1,481 textures, median **34.0 dB**, p10 28.7. In-sample on all: 34.4 dB. The two half-fits agree within ±1 on 228/249 slots (99.9% of blocks by use). Index codebook fitted on half the pools: 36.6 / 36.1 dB held out, 248/256 entries identical between halves. 200 random textures larger than 512 px: 200/200 decode, median 37.4 dB. Plain BC1 ceiling on the same test: 37.9 dB |
| Speed | **34-40 MB/s of BC1 output per core** (4.3-5 M blocks/s, 25 MB/s compressed input), CPU time, `bcpack bench` on 4,096- and 65,536-block chunks. Chunks decode independently, so `scan` scales across cores (8 threads on a heavily shared machine: about 110 MB/s). Below the 50 MB/s per-core target |

The plain-BC1 ceiling of the mip-consistency check is about 37.9 dB median. The 5 chunks the research simulator reported as non-converging decode here (see `docs/findings/decoder.md`).

### Research branches

The work happened in parallel branches. Each has a findings doc in `docs/findings/`. This branch (`decoder`) consolidates them.

| Branch | What it holds |
|---|---|
| `patent` | Reference decoder with tables generated from Microsoft's `mdu_minirans.v`, patent (`US11234023B2`) notes |
| `prior-work` | Public BCPack facts collected from other projects, with contradictions noted |
| `probe` | Cycle-level multi-lane MDU simulator, BC1 convergence scan over 15,000 chunks |
| `harness` | Shared test bench: corpus index, convergence oracle, mip-consistency scorer |
| `sem-indices` | The 256-entry quadrant index codebook and its held-out checks |
| `sem-endpoints` | Endpoint rule, preset-window fitter, full BC1 decode + scoring |
| `other-formats` | BC7 entropy layer |
| `decoder` | This code: the consolidated Rust decoder, Python reference, spec |

None of the branches contain game files or decoded textures. Scripts read textures from your own copy at run time.

## Usage

```bash
cargo build --release
bcpack decode-bc1 chunk.bin 4096 out.bc1        # chunk bytes (zero pad allowed) + block count
bcpack symbols chunk.bin 4096                   # the 12 lane symbol streams
bcpack ttpf-export pool.ttpf 318 out.png        # mip 0 of a BC1 texture, decoded + detiled (.png or .dds)
bcpack scan --jobs 8 --every 10 pools/*.ttpf    # convergence and speed over stores
bcpack bench chunk.bin 4096                     # single-core speed (CPU time)
```

`--window file.json` and `--quads file.json` replace the built-in tables, so a newer fit can be tried without rebuilding. `tools/import_tables.py` converts fit outputs into the `data/` format.

Library:

```rust
let window = bcpack::bc1::Window::default_table();
let quads = bcpack::bc1::QuadTable::default_table();
let out = bcpack::bc1::decode(&chunk, blocks, &window, &quads)?;   // out.blocks: blocks * 8 bytes
```

To embed only the decoder, turn off the default features. The library then depends on `serde_json` alone:

```toml
bcpack-rs = { git = "https://github.com/Ahmet-Dedeler/bcpack-rs", default-features = false }
```

Features: `ttpf` adds the RDS5 store reader, Gfx10 detiling, whole-texture decode and PNG output (`flate2`, `png`). `cli` (default) adds the `bcpack` tool on top (`rayon`).

Python reference (slow, readable): `python3 -I python/bcpack.py decode-bc1 chunk.bin 4096 out.bc1`.

## Layout

- `src/rans.rs`: the 12-lane clocked rANS decoder (a reference version generic over the byte source, and a fast slice version; tests check they agree)
- `src/bc1.rs`: lane symbols to BC1 blocks (back references, preset window, quadrant codebook)
- `src/ttpf.rs`: RDS5 stores, ETPF texture index, Gfx10 mip layout and detiling, BC1 to RGBA, DDS/PNG
- `src/encode.rs`: synthetic encoder for tests (reverse rANS per fragment, interleaved by replaying the feeder)
- `src/tables.rs`, `python/bcpack_tables.py`: generated from `mdu_minirans.v` by `tools/gen_tables.py`
- `data/`: the fitted preset window and index codebook
- `python/bcpack.py`: reference decoder, pure Python

## Tests

```bash
cargo test --release
python3 -I python/bcpack.py selftest
```

The Rust tests check the Verilog table checksum, that the reciprocal inverse equals exact division for every model and cf, synthetic round trips through the encoder (random symbols, all 16 models, alphabet narrowing, 2- and 3-byte headers, multi-fragment lanes, padding, truncation), that the fast and reference decoders agree clock for clock, the block assembly on a hand-built case, and golden SHA-256 hashes of decoded synthetic output. The Python self-test decodes the same synthetic vectors (`tests/vectors/`) and must match. Synthetic tests show the decoder is self-consistent; agreement with hardware comes from the retail checks above.

## Reproducing

The retail checks need your own copy of the texture pools; point the harness at it with `BCPACK_GAME_ROOT`. The bench scripts live on the `harness` branch (`tools/bench`).

- convergence: `bcpack scan --jobs 8 --every 20 $BCPACK_GAME_ROOT/data/ttdb/**/*.ttpf $BCPACK_GAME_ROOT/pack/*/data/ttdb/**/*.ttpf`
- mip consistency: decode each pair texture's mip-0 chunk with `bcpack decode-bc1` and score with `tools/bench/mipscore.py score TEXTURE_ID out.bc1 --order stream`

## Credits

- Microsoft [`silicon-rans`](https://github.com/microsoft/silicon-rans) (MIT): the reference MDU rANS decoder and its exact model tables. See `THIRD_PARTY_NOTICES.md`. The same design is described in patent US11234023B2.
- SixNG format notes (holdmysocks, walkthestars, TrinitronClient and contributors): RDS5/ETPF container, chunk table, Gfx10 layouts, BCPack header observations.
- Xbox2Windows (R0Y00) and MarlomMedeiros' corpus-wide BCPack measurements (issue #15): fixed 8-byte rounds, demand-driven refills, the 2/3-byte length flag, lane-dependent constant models.
- Leonida (Kam471): per-codec chunk counts and size statistics across all pools.
- paryzius' ttpf-viewer: exact Gfx10 detiling and mip-tail placement, ported in `src/ttpf.rs`.
