# Probe findings: BC1 BCPack layers 1 and 2 converge exactly

Agent: `probe`. Date: 2026-11-21. Scope: the rANS read schedule (layer 1) and
the probability models (layer 2) for BC1 (codec 0x04) chunks in retail Xbox
Series texture pools. No game bytes are in this repository. Every script reads
the local store files given on its command line.

## Result

Whole BC1 chunks decode with **exact convergence on all 12 lanes**. Every lane
emits exactly `blocks` symbols. Every fragment's byte budget is consumed in
full. Every lane's state after its last symbol is a legal end state
(`L`, `L>>8`, `L>>16`). The total bytes read equal the chunk's stored length
minus its trailing zero padding (in a handful of cases one more byte, a real
trailing `0x00`).

| Run | Passed |
|---|---|
| `ttdb_maps_0_7`, 60 chunks sampled evenly, 1024-4096 blocks | 60 / 60 |
| 7 stores (maps, terrain, props, interiors, peds, graphics, vehicles), 600 chunks, 1024-16384 blocks | 600 / 600 |
| `ttdb_maps_0_7`, chunks of 32768 and 65536 blocks (36 sampled) | 36 / 36 |
| **All 57 stores, 15,000 BC1 chunks sampled evenly (every 6.7th of ~100k), every size** | **15,000 / 15,000** |

Corpus run by block count (`scan.py --quiet --limit 15000 --max-blocks 200000 <all 57 stores>`, about 8 min on 18 cores): 1024: 1311/1311, 1536: 2227/2227, 2048: 341/341, 2560: 22/22, 3072: 25/25, 3584: 2/2, 4096: 3980/3980, 8192: 470/470, 16384: 3751/3751, 32768: 320/320, 65536: 2512/2512, 131072: 39/39. No failures, so there is no failure pattern to report. The 1536- and 3072-block chunks, earlier described as "several sub-streams", are single streams with `out_target = blocks`. Their lanes simply split into several fragments.

So the varying lanes **do** converge exactly. They do so under Microsoft's
published tables, not under the fitted `lambda(q)` family that
`docs/BCPACK_SPEC.md` describes (see controls below).

## What the decoder is

1. **Per-lane decoder.** `tools/probe/mdusim.py` transliterates the
   `decoder` module of Microsoft's MIT-licensed `silicon-rans`
   (`mdu_minirans.v`) cycle by cycle. It runs two phases per clock, with a
   1-byte input buffer and a 1-symbol output buffer. The states are
   HDR0/HDR1/HDR2/PROCESSING/DRAINING. Header fields: the flush bit, a 4-bit
   model `q`, a 2-bit narrowing `z`, and an 8-bit length (`+1`) or 16-bit
   length (`+257`). The legal-end-state trim is included. The initial state
   is not "4 big-endian bytes": it is fed through the normal renorm path
   starting from 0. The two come to the same thing unless the first payload
   byte is 0.
2. **Exact tables.** `tools/probe/mdu_tables.py` parses `base_table`,
   `freq_table`, `rf_table`, `rs_table` and `rn_table` (9 x 16 x 10) straight
   out of the Verilog. The inverse uses the reciprocal path:
   `dist=cf-base+rn; steps=((dist*rf)>>15)>>rs`. That path was checked
   against exact division for every `cf` in 1..65535 and every (eab, q): 0
   mismatches. **Table index `eab` means an alphabet of `eab+1` bits**
   (entry 0 is labelled "alphabet size = 1 bit"). So `q=15` gives
   `K = 2^(eab+1)`.
3. **Lane alphabet widths** (the Verilog input `alphabet_bits` is the lane's
   symbol width minus 1). Order: L0, L1, R0, G0, B0, R1, G1, B1, then the 4
   index rows: `[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7]`. These widths
   reproduce every constant-lane K the scene had reported: `de` gives K=32
   on 8-bit lanes, `dc` gives K=4 on 5-bit, `dd` gives K=8 on 6-bit.
4. **Feeder (HYPOTHESIS, the published Verilog leaves it out).** On each
   clock, phase 0, the lanes that raise `din_req` are served bytes from the
   single stream in lane order 0..11, at most **8 bytes per clock**
   (`MDU_RANS_FEED_WINDOW = 8`). A lane that asks and is not served stalls
   for that clock (`blocked_on_input`).
5. **Symbol-vector assembly (HYPOTHESIS, also left out).** There is one slot
   per lane, and `sym_ready` = "my slot is empty". The vector is released
   only when every slot is full. That is lockstep: one symbol per lane per
   vector, `blocks` vectors per chunk.
6. **`out_target` = chunk block count** for all 12 lanes. The block count
   comes from the virtual-offset delta divided by 8. Lanes are multi-fragment
   (up to 43 fragments in one lane observed), and later fragment headers
   appear in-band exactly where the simulator asks for them.

Nothing else is assumed. The 48-byte column-interleaved header, the
secondary-header interleave in rounds 6+, the "round 7 starts at offset 56"
rule and the demand-driven refills are not modelled by hand. They all fall
out of points 1, 4 and 5.

## Controls (feeder and symvec are discriminated, not merely compatible)

The same chunks were run with exactly one thing changed
(`scan.py --variant ...`):

120 chunks sampled evenly across all 57 stores, up to 16384 blocks
(`scan.py --quiet --limit 120 --max-blocks 16384 --variant V <all stores>`):

| Variant | What changes | Passed |
|---|---|---|
| (baseline) | nothing | **120 / 120** |
| `window=7` / `window=9` / `window=16` | bytes served per clock | 0 / 120 each |
| `symvec=free` | no lockstep; each lane drains its own output | 0 / 120 |
| `dN=1` / `dN=-1` | `out_target = blocks +/- 1` | 0 / 120 each |
| `order=rev` | feeder serves lanes 11..0 | 0 / 120 |
| `order=sec-first` | feeder serves index lanes 8-11 before 0-7 | 0 / 120 |
| `lambda` | fitted lambda(q) tables from BCPACK_SPEC.md | 0 / 120 (1/20 on an earlier maps_1_7 sample) |
| `split=2` | 65536-block chunk as 2 x 32768 streams | 0 / 36 |
| `flip=0.25` / `0.5` / `0.75` | XOR bit 0 of one mid-stream byte | 2, 2, 5 / 120 |

Notes:
- **`flip`** XORs bit 0 of one byte in the middle of the stream. A few chunks
  survive it. rANS decoding is contractive on lanes dominated by one
  high-probability symbol, so a 1-unit perturbation in the low state bits can
  be absorbed. The two surviving cases examined were a data byte of a `q=12`
  lane (lane 2 of maps_1_7 #12065) and an index-lane data byte. This is a
  known rANS property, not evidence against the schedule. The structural
  controls (window, symvec, N, order, tables, split) are the meaningful ones.
- **`lambda`** swaps in the fitted `lambda(q)` family from
  `docs/BCPACK_SPEC.md`. It passes only on chunks whose lanes all use
  (eab, q) pairs where that family happens to equal the real table
  (uniform, all-floor, or one of its hand-copied exact overrides).
- **`split=2`** runs a 65536-block chunk as two independent 32768-symbol
  streams, as reported for BC7. It fails 36/36 on BC1. BC1 65536-block
  chunks are one stream with `out_target = 65536`.
- Chance rate: a wrong hypothesis must make all 12 lanes end in one of 3
  legal 32-bit states at exactly the right symbol count, and must also land
  on the exact byte length. Even for one lane, the per-lane probability of a
  legal state by accident is about 3 / 2^24. The variant runs above give 0
  passes for every structural change.

## First exact hit (historical, single-lane)

Before the full simulator existed, maps_0_7 chunk 2247 (1536 blocks) was the
first varying lane to converge. All 8 main lanes have budget 4. The 4 index
lanes have identical records (mode `0x9a`, `b1=0x63`, so 100 payload bytes),
so bytes 48.. arrive in groups of 4 identical bytes. Only table `eab=5, q=10`
decodes that lane to exactly `x = 0x01000000` with all 100 bytes consumed
(256 symbols, i.e. its first fragment). None of the other 143 (eab, q) pairs
does.

## Ruled out / corrected

- The fitted `lambda(q)` tables in `docs/BCPACK_SPEC.md` section 3.3 are
  wrong for most (eab, q). Use the Verilog tables. The spec's `eab = d - z`
  is right only once `d` is the lane width minus 1 (table index). The spec
  counts `eab` as alphabet bits.
- The spec's 16-bit length form (`u16 + 257`) is correct (Verilog HDR2). One
  scene report gave `len16 + 1`; that does not hold.
- "Index lanes launch at round 6" is not a separate rule. It is just the
  8-byte feed window being saturated by the 8 main lanes for 6 clocks.
- BC1 65536-block chunks are **not** two 32768-block streams.
- `symvec = free` (independent lanes, no lockstep) fails everywhere. The
  lockstep release matters for byte order.

## Open (handed to semantics agents)

- Layer 3: what the symbols mean. In chunk 2247 the endpoint lanes look like
  small residuals: mostly 0, then 1, 2, 3 with geometric decay, which
  suggests zigzag-coded deltas. The index lanes are 6- or 8-bit row codes,
  and lane 1 is all 0 there. Lane 0 is unknown.
- Layer 4: block order.
- Other codecs (BC7 etc.) need their own lane count and alphabet widths.
  The same simulator should apply once those are known.

## Environment note

All probe scripts are pure Python (no numpy), so the numpy temporary-elision bug in the system
Python 3.14 does not affect them.

## Reproduce

```
cd tools/probe
python3 -I scan.py --limit 60 --max-blocks 4096 <store>.ttpf            # convergence
python3 -I scan.py --quiet --limit 20 --variant window=7 <store>.ttpf   # a control
python3 -I decode_chunk.py <store>.ttpf <chunk_index> [--out syms.txt]  # per-lane symbols
```

From Python: `decode_chunk.decode_chunk_symbols(path, chunk_index) -> (blocks, syms[12])`.
It raises if the chunk does not converge.

Files: `tools/probe/chunks.py` (RDS5 chunk table), `mdu_tables.py` (Verilog
tables plus single-lane decoder), `mdusim.py` (cycle-level multi-lane sim),
`scan.py` (corpus scan and controls), `decode_chunk.py` (symbol dump),
`models.py` (the old fitted lambda family, kept only as a control).
