# bcpack-rs

Clean-room Rust crate, CLI, and stdlib-only Python implementation of the Xbox Series X|S Media Decompression Unit (MDU) **BCPack** texture rANS entropy codec (`BC1`–`BC7`).

Licensed under `MIT OR Apache-2.0`. Contains **zero** proprietary game bytes or assets.

## Key Features

- **Exact MDU rANS Engine (`M = 2^16`, `L = 2^24`)**: Implements the 2-phase hardware rANS decoder state machine (`DSTATE_IDLE`..`DSTATE_DRAINING`), 16 static probability models (`q = 0..15`) across power-of-two segments (`base_table[eab][q]` / `freq_table[eab][q]`), adjusted symbol widths (`eab = alphabet_bits - z`), slot-0 reservation (`fwd_cf = 1`, `fwd_f = fwd_fa - 1`), and early-trimmed terminal states (`L`, `L >> 8`, `L >> 16`).
- **Multi-Fragment & Multi-Lane Interleaving**: Supports 2-byte (`hdr3 = 0`, `len = b1 + 1`) and 3-byte (`hdr3 = 1`, `len = u16_le + 257`) fragment headers, 1-symbol uniform prefix fragments (`0x10`, `0x50`, `0x90`, `0xd0`) transitioning in-band to delta fragments, Round-6 index lane launch (`L8..L11`), and lockstep `symvec` output synchronization.
- **Rust Crate + CLI + Python Module**:
  - Rust library (`bcpack`) and CLI (`bcpack inspect`, `bcpack decode-bc1`)
  - Pure-Python stdlib-only module (`python/bcpack.py`, compatible with `python -I`)
  - Format specification in [`docs/BCPACK_SPEC.md`](docs/BCPACK_SPEC.md)

## Quick Start

```bash
# Run Rust unit tests
cargo test

# Run stdlib-only Python unit tests
python3 -I python/bcpack.py --selftest
```
