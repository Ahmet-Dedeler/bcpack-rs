# Findings: harness (shared test bench)

Agent `harness`, 2026-11-21. Bench code in `tools/bench/` (see its README for every command).
All numbers below come from the retail Xbox Series texture pools (57 RDS5 stores:
`data/ttdb/**` and `pack/*/data/ttdb/**`), read in place; no game bytes are in git.

Shared outputs: `/tmp/bcpack-shared/bc1_index.jsonl` (102,725 lines),
`/tmp/bcpack-shared/bc7_index.jsonl` (5,167), `/tmp/bcpack-shared/corpus_summary.json`,
`/tmp/bcpack-shared/mip_pairs.jsonl` (194,143: 4,809 pairs + 189,334 controls),
`/tmp/bcpack-shared/const_tables/` (constant-lane lookup tables, rebuilt automatically).

## 1. Corpus (`python3 -I tools/bench/corpus.py`, then `--summary`)

| codec | chunks |
|---|---|
| 0x20 zlib | 918,272 |
| 0x04 BC1 BCPack | 102,732 (102,725 indexed; 2 cut off by end of file, 5 zero-length) |
| 0x1C BC7 | 5,167 |
| 0x14 BC5 / 0x18 BC6H / 0x10 BC4 / 0x0C BC3 / 0x00 stored | 334 / 268 / 77 / 35 / 5 |

BC1 BCPack: 10.39 GiB stored (pad excluded), 14.29 GiB virtual. BC7: 1.07 GiB stored.

BC1 block counts per chunk (vlen / 8): 4096: 27,624 · 16384: 24,902 · 65536: 16,184 ·
1536: 15,381 · 1024: 10,221 · 8192: 2,829 · 2048: 2,286 · 32768: 2,051 · 262144: 597 ·
131072: 268 · 2560: 175 · 3072: 166 · 524288: 27 · 3584: 8 · 512: 6.

Non-constant main lanes per BC1 chunk (out of 8; lane 1 is always constant):

| non-const lanes | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| chunks | 2,680 | 822 | 1,900 | 4,037 | 3,066 | 17,874 | 10,874 | 61,472 |

(`n_var`, which also counts budget-4 "len4" lanes as trivial: 0: 3,267 · 1: 820 · 2: 2,416 ·
3: 4,742 · 4: 4,708 · 5: 17,939 · 6: 14,893 · 7: 53,940.)

Per lane (BC1), top mode bytes and proven-constant counts:

| lane | top modes | distinct modes | proven constant (K) |
|---|---|---|---|
| 0 | 32 31 33 34 1f de | 66 | 5,452 (de, K=32) |
| 1 | de | 1 | 102,725 / 102,725 (K=32) |
| 2 | dc 3a 39 1e 3b 3c | 81 | 15,268 (dc, K=4) |
| 3 | 3b 3a 3c dd 1c 5c | 85 | 9,040 (dd, K=8) |
| 4 | dc 3a 3c 1c 3b 5c | 85 | 24,292 (dc, K=4) |
| 5 | 3a dc 3b 1e 39 3c | 79 | 15,524 (dc, K=4) |
| 6 | 3b 3c 3a dd 1c 1d | 84 | 9,098 (dd, K=8) |
| 7 | dc 3a 3c 9d 1c 3b | 83 | 26,486 (dc, K=4) |

"Proven constant" = re-encoding N copies under the SixNG constant model (M=2^16, L=2^24,
repeated symbol at start 1 with freq M-K) from x=L reproduces both the record's budget and its
32-bit state, with K the mode's K (dc 4, dd 8, de 32/16, db 2). Hits at any other (mode, K)
are kept in the index (`const` field) but not trusted: there are 10-34 per lane, the expected
chance rate for ~800k records x 8 K values.

Independent cross-check on BC7 (`bc7_index.jsonl`), reproducing MarlomMedeiros' published
counts (Xbox2Windows issue 15) with our own code: de K=32 on lanes 2-6 (64/94/333/85/335),
de **K=16 on lane 7** (374/374, no K=32 hit), db K=2 on lanes 0-1 (90/94). Lane 2-7 de
symbol counts are 2x blocks in 847 records (Marlom: 827 on 46 stores). This validates the constant-lane
tables and the interleaved header parse.

## 2. Oracle baseline (`python3 -I tools/bench/oracle.py --adapter bcpack-rs --sample 500 --seed 1`)

bcpack-rs got a `trace` subcommand (`target/release/bcpack trace <chunk> <blocks>`, per-lane
JSON; lane bytes are now counted in `RansLane::bytes_read`). Real score on 500 random BC1
chunks (seed 1), expectation `default=auto`:

| | lanes | budget ok | legal state | count ok | pass |
|---|---|---|---|---|---|
| trivial | 1,047 | 671 | 611 | 691 | 611 |
| varying | 4,953 | 146 | 10 | 4,710 | **10** |

Whole chunks passing: 0/500; chunks whose lanes sum to data_len: 23/500; errors: 0.
The 10 varying passes all have payload 5-6 bytes (modes db/dc/9c/9d, 1-2 renorm bytes,
near-constant lanes); by payload: 5-8 B 10/98, 9-64 B 0/166, 65-1024 B 0/778, >1024 B 0/3,911.
Index lanes 8-11: 0/2,000. Shuffle control (`--control shuffle`, bytes after the header
shuffled): varying 0/4,953. So bcpack-rs decodes nothing beyond near-constant lanes; the
`count ok` column is high only because bcpack-rs stops at the requested count.

`python3 -I tools/bench/oracle.py --selftest`: synthetic constant chunk passes, wrong symbol
count fails, flipped state byte fails.

## 3. Mip scorer controls (`python3 -I tools/bench/mipscore.py controls --n 200 --max-side 512`)

`mip_pairs.jsonl`: 4,809 BC1 textures (40 stores) with mip 0 in BCPack and plain mip 1; for
**every one of them mip 0 is exactly one BCPack chunk** (`mip0_exact_chunk`), so a chunk decode
is directly scoreable. Sizes: 256x256 1,348, 1024x1024 1,246, 128x256 441, 512x512 383, ...;
tile modes 2: 4,258, 3: 506, 0: 45. Pair chunks: 4096 blocks 1,477, 65536 1,250, 2048 750.

Controls, PSNR in dB over 200 random textures (<= 512 px side, seed 1):

| | median | p10 | p90 |
|---|---|---|---|
| POSITIVE: true mip 0 (plain), gfx10 order | 37.9 | 30.8 | 47.0 |
| true mip 0 read as linear (wrong order) | 18.7 | 9.2 | 34.4 |
| NEGATIVE: random BC1 blocks | 8.0 | 6.1 | 11.2 |
| NEGATIVE: true blocks, shuffled order | 17.9 | 8.5 | 33.5 |
| mean-colour guess (`psnr_flat`) | 21.6 | 12.7 | 38.8 |
| NEGATIVE: bcpack-rs decode-bc1 on 179 pair chunks (stream order) | 7.3 | 4.3 | 9.9 |

Positive minus flat: median +14.2 dB. 5/200 positives are below 25 dB (one 64x64 2-mip
texture at 7.3 dB, imposter atlases at 12-18 dB): content where mip 1 is not a box filter of
mip 0. bcpack-rs failed to finish on 21/200 pair chunks. The scorer separates correct from
wrong data and wrong block order; always read `psnr` next to `psnr_flat`.

## 4. The 30 easiest BC1 chunks

`python3 -I tools/bench/corpus.py --easiest 30` sorts by non-constant lanes, then size. All
30 (and all 2,680 zero-non-constant chunks but one) are 1536-block chunks whose 8 main lanes
are all constant; their data is entirely in the index lanes (~7 KB, ~37 bits/block). SixNG
says 1536-block chunks hold several streams, so here is the list, then a second list of
single-stream chunks (not 1536/3072 blocks, lane 1 constant with N = blocks).

### 4a. Overall
| chunk id | blocks | data_len | non-const | var | lane modes (c=const, l=len4, v=var) |
|---|---|---|---|---|---|
| `data/ttdb/terrain/ttdb_terrain_image_based_0_1.ttpf:1726` | 1536 | 7041 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/maps/ttdb_maps_2_7.ttpf:18889` | 1536 | 7069 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_image_based_1_1.ttpf:4474` | 1536 | 7164 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_image_based_1_1.ttpf:322` | 1536 | 7186 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_6_7.ttpf:17638` | 1536 | 7189 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_image_based_0_1.ttpf:4922` | 1536 | 7237 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/interiors/ttdb_interiors_1_3.ttpf:1302` | 1536 | 7243 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_image_based_0_1.ttpf:3347` | 1536 | 7250 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_image_based_1_1.ttpf:2624` | 1536 | 7252 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_6_7.ttpf:23858` | 1536 | 7257 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_image_based_0_1.ttpf:1365` | 1536 | 7264 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_image_based_1_1.ttpf:742` | 1536 | 7269 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_image_based_0_1.ttpf:2402` | 1536 | 7282 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_image_based_1_1.ttpf:4502` | 1536 | 7298 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_image_based_0_1.ttpf:154` | 1536 | 7315 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_image_based_0_1.ttpf:235` | 1536 | 7325 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/peds/peds_loose.ttpf:10126` | 1536 | 7335 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/peds/peds_loose.ttpf:7800` | 1536 | 7424 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `pack/story000/data/ttdb/maps/ttdb_maps.ttpf:1030` | 1536 | 7437 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/maps/ttdb_maps_5_7.ttpf:16812` | 1536 | 7474 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_5_7.ttpf:14071` | 1536 | 7492 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/peds/ttdb_peds_1_3.ttpf:16343` | 1536 | 7524 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_4_7.ttpf:22933` | 1536 | 7554 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_0_7.ttpf:10450` | 1536 | 7581 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_4_7.ttpf:648` | 1536 | 7585 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/maps/ttdb_maps_5_7.ttpf:28275` | 1536 | 7589 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_6_7.ttpf:10490` | 1536 | 7598 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_4_7.ttpf:6838` | 1536 | 7621 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_0_7.ttpf:1141` | 1536 | 7659 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_4_7.ttpf:3819` | 1536 | 7694 | 0 | 0 | de:c de:c dc:c dd:c dc:c dc:c dd:c dc:c |

### 4b. Single-stream
| chunk id | blocks | data_len | non-const | var | lane modes (c=const, l=len4, v=var) |
|---|---|---|---|---|---|
| `data/ttdb/vehicles/ttdb_vehicles_loose_1_1.ttpf:7749` | 16384 | 91015 | 1 | 1 | de:c de:c 1d:v dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/props/ttdb_props_3_3.ttpf:23834` | 3584 | 18228 | 2 | 1 | 5f:v de:c dc:c dd:c dc:c dc:c 1f:l dc:c |
| `data/ttdb/maps/ttdb_maps_5_7.ttpf:42092` | 65536 | 371243 | 2 | 1 | 1c:v de:c dc:c dd:c dc:c 9b:l dd:c dc:c |
| `data/ttdb/peds/ttdb_peds_3_3.ttpf:17602` | 16384 | 78310 | 2 | 2 | 9a:v de:c dc:c dd:v dc:c dc:c dd:c dc:c |
| `pack/core_restofworld/data/ttdb/peds/ttdb_peds.ttpf:11` | 16384 | 81939 | 2 | 2 | d8:v de:c dc:c dd:c dc:c 9b:v dd:c dc:c |
| `data/ttdb/peds/decorations_3_3.ttpf:5862` | 65536 | 304708 | 2 | 2 | de:c de:c dc:c 1d:v dc:c dc:c 1f:v dc:c |
| `data/ttdb/maps/ttdb_maps_7_7.ttpf:480` | 65536 | 351217 | 2 | 2 | 5c:v de:c dc:c dd:c dc:c da:v dd:c dc:c |
| `data/ttdb/terrain/ttdb_terrain_5_7.ttpf:2052` | 65536 | 382182 | 2 | 2 | 1c:v de:c 1c:v dd:c dc:c dc:c dd:c dc:c |
| `data/ttdb/peds/ttdb_peds_1_3.ttpf:2203` | 1024 | 4453 | 3 | 1 | 38:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_0_3.ttpf:8506` | 1024 | 4466 | 3 | 1 | 38:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_0_3.ttpf:17712` | 1024 | 4478 | 3 | 1 | 38:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_0_3.ttpf:19725` | 1024 | 4538 | 3 | 1 | 37:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_0_3.ttpf:16675` | 1024 | 4688 | 3 | 1 | 38:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_1_3.ttpf:20570` | 1024 | 4756 | 3 | 1 | 38:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_3_3.ttpf:16209` | 1024 | 4775 | 3 | 1 | 3a:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_3_3.ttpf:2151` | 1024 | 4810 | 3 | 1 | 36:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_1_3.ttpf:15370` | 1024 | 4837 | 3 | 1 | 39:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_1_3.ttpf:6590` | 1024 | 4845 | 3 | 1 | 39:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_1_3.ttpf:786` | 1024 | 4870 | 3 | 1 | 37:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_2_3.ttpf:14652` | 1024 | 4938 | 3 | 1 | 35:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_1_3.ttpf:9527` | 1024 | 5029 | 3 | 1 | 35:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/props/ttdb_props_3_3.ttpf:8231` | 1024 | 5031 | 3 | 1 | 34:v de:c dc:c dd:c 10:l dc:c dd:c 10:l |
| `data/ttdb/peds/ttdb_peds_0_3.ttpf:1331` | 1024 | 5125 | 3 | 1 | 33:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_0_3.ttpf:16376` | 1024 | 5183 | 3 | 1 | 33:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_0_3.ttpf:4624` | 1024 | 5185 | 3 | 1 | 33:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_1_3.ttpf:9398` | 1024 | 5199 | 3 | 1 | 33:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_0_3.ttpf:7158` | 1024 | 5215 | 3 | 1 | 33:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_0_3.ttpf:7162` | 1024 | 5220 | 3 | 1 | 36:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_0_3.ttpf:5898` | 1024 | 5230 | 3 | 1 | 33:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |
| `data/ttdb/peds/ttdb_peds_1_3.ttpf:20275` | 1024 | 5260 | 3 | 1 | 5c:v de:c dc:c 10:l dc:c dc:c 10:l dc:c |

## 5. Surprises and contradictions

1. **Constant lanes inside 1536-block chunks carry sub-stream counts.** In the 2,680
   all-constant chunks lane 1 encodes 1,536 symbols but lanes 0 and 2-7 encode 192 (841
   chunks) or 256/257 (258), with occasional 193 or 96/128. Lane 1 is not a sub-stream lane.
2. **Streams cap at 65,536 blocks.** Lane 1's N is 65,536 in every 131,072 (268),
   262,144 (597) and 524,288 (27) block chunk, so those chunks hold 2, 4 or 8 streams. For
   chunks <= 65,536 blocks outside 1536/3072, N = blocks in every case (86,286).
3. **BC1 endpoint constant lanes carry N = blocks symbols, not 2N** (lanes 2-7: 1x in
   5,369 / 1,014 / 14,269 / 6,085 / 1,049 / 16,369 records outside 1536/3072). The 2x count
   holds on BC7 only.
4. **Lane 0 is constant (de, K=32) only in 1536/3072-block chunks** (5,440 of 5,452), with
   N = 192 (3,555), 128 (735), 257 (666) or 96 (441): again sub-stream counts, never blocks.
5. **dc/dd records are mostly not pure constants**: lane 4 has 45,151 dc records but only
   24,292 re-encode as pure constant runs (the rest have exceptions). The 0x1f/0x5e/0xdb
   constant parameters in docs/BCPACK_SPEC.md §3.2 never show up as pure constant lanes in BC1.
6. **Pad**: `b >> 20` is exactly the number of trailing zero bytes, 102,725/102,725 BC1 and
   5,167/5,167 BC7. Pads go up to 4,094 bytes, so stored chunks are not merely "zero-padded to
   256 bytes" (SixNG §6.1). 1,322 BC1 chunks also end in 1-5 zero data bytes right before the
   pad, so do not strip zeros by value.
7. **Chunk-table conventions**: SixNG adds 0x1000 to virtual offsets; ttpf-viewer does not. With
   the viewer's convention the end-sentinel's virtual offset equals the header's virtual size and
   ETPF texture starts need no bias. Both describe the same table (SixNG entry i = start of chunk
   i; viewer = end of chunk i-1, codec of the next chunk). The bench uses the viewer convention
   and the 26-bit file offset.
8. **3-byte length form**: `mdu_minirans.v` gives `(b2<<8 | b1) + 257`; MarlomMedeiros reports
   `len16 + 1` on BC7. Fragment-0 sums of the 8 main lanes never exceed the chunk under either
   convention (0 negatives in 102,725 BC1 / 5,167 BC7), so this check cannot tell them apart. The
   index stores both (`budget`, `budget_alt_plus1`); the oracle uses the Verilog value.

## 6. Not done / limits

- Index-lane (lane 8+) records are not parsed in the index: where they start depends on the
  read schedule, which is still open.
- Multi-fragment budgets are taken from the candidate's own fragment list; the oracle checks
  fragment 0 of main lanes against the header and the chunk-wide byte sum independently.
- No Rust/multiprocessing fast path in `mipscore.score` itself (pure Python, ~1 s for 512x512,
  slow for 4096x4096); the controls run in parallel across textures.
- Arrays, cube maps and tile modes 9-12 are not indexed as mip pairs (plain textures only).
