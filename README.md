# bcpack-rs

Reverse engineering of **BCPack**, the rANS entropy codec that the Xbox Series X|S Media Decompression Unit (MDU) uses for block-compressed (`BC1`–`BC7`) textures. Rust crate + CLI, a stdlib-only Python port, and a written spec.

Licensed under `MIT OR Apache-2.0`. The repository contains no game data; all tests use synthetic streams.

## Status

**BC1 decodes.** The format is solved: 2,995 of 3,000 test textures decode fully. A held-out check of the preset window table is still running. The fast Rust decoder is still being written; today the working pipeline is the Python research code on the branches listed below.

| Part | Status |
|---|---|
| Entropy layer (rANS) | **Solved.** Microsoft's exact MDU tables from the public [`silicon-rans`](https://github.com/microsoft/silicon-rans) Verilog (MIT). Bytes are fed in lane order with all lanes in lockstep, and every lane must end in a legal state, so a wrong guess fails loudly |
| BC1 endpoints | **Solved.** Lane 0 is a back-reference "copy block *i − d*", lanes 2–7 add zigzag corrections to R0 G0 B0 R1 G1 B1. References before the chunk start read a fixed 256-entry preset window |
| BC1 indices | **Solved.** Each 2×2 quadrant is a symbol into a fixed, frequency-ordered table of all 256 index patterns. Two independent fits agree on 248/256 entries (99.76% of quadrants); 8 rare mirror pairs are still uncertain |
| BC1 block order | Standard Gfx10 tiled order |
| BC1 quality, held out | Fit on even stores, scored on odd: **36.6 dB** median vs the plain mip 1 (318/319 above baseline). Reverse split 36.1 dB. Plain BC1 ceiling is 37.9 dB. Fitting with no mip term at all still gives 35.0 dB, so the fit is not just learning the scorer |
| BC1 coverage | The preset window is fully fitted (254/254 referenced slots). Full decode works on **2,995 of 3,000** test textures: median 34.4 dB vs the plain mip 1 (ceiling 37.9 dB), 2,641 at 30 dB or more. Held-out check of the window table pending |
| BC7 | Entropy layer solved: 16 lanes, widths `4,4,8,8,8,8,8,7,7,5,6,8,8,8,8,5`. 4,600+ chunks converge exactly, 0 of 1,180 controls do. Block mapping (modes, partitions) in progress |
| BC6H, BC5, BC4, BC3 | Entropy layer solved: every chunk in the corpus converges exactly. Block mappings in progress |

### Research branches

The work happened in parallel branches. Each has a findings doc in `docs/findings/`.

| Branch | What it holds |
|---|---|
| `patent` | Reference decoder with tables generated from Microsoft's `mdu_minirans.v`, patent (`US11234023B2`) notes |
| `prior-work` | Public BCPack facts collected from other projects, with contradictions noted |
| `probe` | Cycle-level multi-lane MDU simulator, BC1 convergence scan over 15,000 chunks |
| `harness` | Shared test bench: corpus index, convergence oracle, mip-consistency scorer |
| `sem-indices` | The 256-entry quadrant index codebook (`tools/sem/quad_table.json`) and its held-out checks |
| `sem-endpoints` | Endpoint rule, preset-window fitter, full BC1 decode + scoring |
| `other-formats` | BC7 entropy layer |

None of the branches contain game files or decoded textures. Scripts read textures from your own copy at run time.

Credits: Microsoft `silicon-rans` (MIT) for the rANS tables, and the SixNG and Xbox2Windows communities for prior format notes.

## Layout

- `src/lib.rs` – rANS lane decoder state machine, static model tables, fragment header parser
- `src/main.rs` – `bcpack inspect <chunk>` (dump lane and fragment headers) and `bcpack decode-bc1` (an early guess at the lane schedule that does not produce valid BC1; it is being replaced by the solved decoder)
- `python/bcpack.py` – the same decoder in pure Python (runs under `python -I`)
- [`docs/BCPACK_SPEC.md`](docs/BCPACK_SPEC.md) – early format notes. Parts of it predate the solved layout; the findings docs on the research branches are current

## Tests

```bash
cargo test
python3 -I python/bcpack.py --selftest
```

Both suites check round trips on synthetic streams built with the same model tables, so they confirm the decoder is self-consistent, not that it matches hardware.
