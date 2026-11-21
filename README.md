# bcpack-rs

Work-in-progress reverse engineering of **BCPack**, the rANS entropy codec that the Xbox Series X|S Media Decompression Unit (MDU) uses for block-compressed (`BC1`–`BC7`) textures. Rust crate + CLI, a stdlib-only Python port, and a written spec.

Licensed under `MIT OR Apache-2.0`. The repository contains no game data; all tests use synthetic streams.

## Status

**This does not decode textures yet.** What is verified and what is not:

| Part | Status |
|---|---|
| Fragment header bitfields (`z`, `hdr3`, `flush_per_frag`, `q`) and the 2-byte vs 3-byte length forms | Verified: parsed fragment byte budgets add up exactly to the compressed size of real multi-fragment BC1 chunks |
| Power-of-two segmented static models, slot-0 reservation, legal end states `{L, L>>8, L>>16}` | Taken from the public patent (`US11234023B2`, Fig. 15) and consistent with every single-symbol fragment observed |
| Constant-lane and 1-symbol prefix (`0x10`/`0x50`/`0x90`/`0xd0`) fragments | Verified: all 7,162 observed `len == 4` fragments end in a legal state. These are the trivial cases (one symbol or a constant lane) |
| Exponential decay schedule `λ(q)` for the 16 models | Fitted to the frequency tables observed so far, not confirmed for every `(eab, q)` pair |
| Multi-symbol lanes, lane interleaving schedule, BC1 block reassembly | **Not working.** `bcpack decode-bc1` runs on real chunks but its output is not valid BC1 (index words are mostly zero in the top bits), so the lane schedule or symbol-to-block mapping is still wrong |
| BC3–BC7 | Lane layouts in the spec are guesses; nothing is decoded |

If you know the MDU lane schedule or have a known plaintext/compressed BC1 pair, issues and PRs are very welcome.

## Layout

- `src/lib.rs` – rANS lane decoder state machine, static model tables, fragment header parser
- `src/main.rs` – `bcpack inspect <chunk>` (dump lane and fragment headers) and `bcpack decode-bc1` (experimental, see status)
- `python/bcpack.py` – the same decoder in pure Python (runs under `python -I`)
- [`docs/BCPACK_SPEC.md`](docs/BCPACK_SPEC.md) – format notes, with the evidence behind each claim

## Tests

```bash
cargo test
python3 -I python/bcpack.py --selftest
```

Both suites check round trips on synthetic streams built with the same model tables, so they confirm the decoder is self-consistent, not that it matches hardware.
