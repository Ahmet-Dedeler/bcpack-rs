# Prior work on BCPack: merged facts table

Agent `prior-work`, 2026-11-21. Reading only; no game files were used. Every number below is copied from the cited
source, except rows tagged "ours", which we computed here (reproduction commands given).
Source clones: `/Users/ahmet/Code/bcpack-refs/*` (state as of 2026-11-21 evening). Issue/PR text fetched with `gh`.

Source trust, best to worst:

| Tier | Source | Why |
|---|---|---|
| A | Microsoft `silicon-rans` Verilog (MIT, 2023-11) and patent US11234023B2 | Primary hardware description. Decoder only: feeder and symbol-vector assembly are deliberately left out |
| A- | Microsoft Learn GDK pages (DirectStorage overview, `DSTORAGE_BCPACK_MODE`) | Official, but only the algorithm class and mode enum |
| B+ | MarlomMedeiros comments on Xbox2Windows #15 and #175/#214 (2026-11-21) | Corpus-wide counts with runnable checks and negative controls; not independently reproduced yet; data not shareable |
| B | SixNG `formats/05-ttpf.md`, `06-bcpack.md` (holdmysocks/walkthestars/TrinitronClient, 2026-11-20/21); Leonida `docs/textures.md` (Kam471, PR #82) | Careful, count-backed notes; container facts independently matched by two other projects |
| B- | R0Y00 / Vonsek / descent81 / raulmoracode PRs on `tools/bcpack_probe.py` | Correct but synthetic-only; each PR says so |
| C | mesterember104-cpu comment (declared AI agent GLM) | Counts fine, interpretation (stride 6) contradicted by later evidence |
| D | `bcpack-rs` (our repo), press/forum posts | Unverified or hearsay |

---

## 0. Headline findings (read this first)

1. **Microsoft published the decoder RTL.** `github.com/microsoft/silicon-rans` (MIT, by Derek Gladding, one commit 2023-11-17,
   `mdu_minirans.v`, 1,225 lines). It has the exact `base/freq/rf/rs/rn` tables for 9 alphabet sizes x 16 models, the 2/3-byte header FSM
   and the legal-end-state trim. None of the scene's BCPack projects cites it (Xbox2Windows, SixNG, Leonida; we did not find a mention in
   any of them). It is already cloned to `/Users/ahmet/Code/bcpack-refs/microsoft_silicon-rans`. Its README says the "symbol vector assembly"
   and "feeder" are left out, so the schedule (layer 1) and semantics (layer 3) are exactly what is still unknown, which matches where the
   community is stuck.
2. **Our `bcpack-rs` probability tables are only partly right.** Compared with Microsoft's `base_table`, **51 of 128** (eab 1..8 x q 0..15)
   models differ (max base error 64; 21 differ by more than 2). Constant-lane models (all the verified ones) are right, so the existing
   "7,162/7,162 len==4 fragments converge" result is unaffected, but any multi-symbol lane using a mis-fitted model cannot converge.
   Fix = replace the fitted tables with the Verilog tables. (ours; reproduce: `python3 -I docs/findings/prior-work-assets/cmp_verilog_tables.py <mdu_minirans.v> <worktree>/python`)
3. **The "lane-dependent K" puzzle is explained by Microsoft's own RTL.** `eab = alphabet_bits(lane config) - z`, clamped to 9 (z = `hdr0[7:6]`).
   A constant lane has `K = 2^eab` (sym 0 gets `M-K` at start slot 1). So the same mode byte gives different K on lanes whose configured alphabet
   width differs. Implied lane widths (ours, inferred, from MarlomMedeiros' K table and the Verilog constant models): BC7 `0xde` lanes 2-6 d=8 (K=32),
   lane 7 d=7 (K=16); `0xdb` lanes 0-1 d=4 (K=2); `0xdc` secondary s1 d=5 (K=4); `0xdd` secondary s2 d=6 (K=8). This also says SixNG's single mode->K table
   was a BC1 special case, and that `bcpack-rs`' `BC1_LANE_AB` need a per-codec table.
4. **The container is solved and three projects agree.** RDS5 chunk table, codec ids (BC number x 4; zlib 0x20), per-chunk padding, tiling (Gfx10 `SW_64KB_D`
   / `SW_4KB_S`) are all confirmed independently by SixNG, Leonida, ttpf-viewer. Block order inside a BCPack stream (layer 4) is still unknown.
5. **Nobody has decoded a single block.** Every project says so explicitly (SixNG "0 % decoded"; Leonida "not decodable from public facts"; Xbox2Windows keeps #15's
   decoding half open).

---

## 1. Facts table

Confidence: **C** = confirmed-with-evidence, **I** = inferred, **S** = speculation. "Who" = first public source we found, not necessarily the discoverer.

### 1a. Container (layer: container)

| # | Fact | Conf | Evidence | Source | Who |
|---|---|---|---|---|---|
| K1 | Stores are `RDS5` files; chunk entry `(a,b)`: file offset `(a&0xFFFFFF)<<8`; codec = bits 24-29 of `a`; virtual offset `0x1000 + (b&0xFFFFF)*0x4000 + (a>>30)*0x1000`; trailing zero pad = `b>>20` bytes | C | 2,810 chunks contiguous; pad: 535/535 sampled BCPack and 541/541 sampled zlib chunks end exactly at pad | SixNG `formats/05-ttpf.md` (TrinitronClient fork, PR #2/#5) | deki (`jaylikesbunda`, PR #2) + holdmysocks; Leonida re-derived the same fields as a u64 (offset 0-25, codec 26-29, virt 30-51, pad 52-63) |
| K2 | Codec id = BCPack mode x 4: `0x04` BC1, `0x0C` BC3, `0x10` BC4, `0x14` BC5, `0x18` BC6H, `0x1C` BC7, `0x20` zlib, `0x00` stored; matches `DSTORAGE_BCPACK_MODE` (1..7) | C | Every chunk joined to its texture descriptor DXGI format: 0 mismatches over 1,026,833 chunks (Leonida); `0x0C` 35/35 BC3, `0x10` 77/77 BC4 (SixNG) | Leonida `docs/textures.md`; SixNG 05 | Kam471; holdmysocks |
| K3 | Chunk counts per codec. SixNG (46 `data/ttdb` stores): BC1 100,026, BC3 35, BC4 77, BC5 325, BC6H 268, BC7 5,120, zlib 908,643, stored 5. Leonida (all 57 pools incl. 11 `pack/*`): BC1 102,725, BC3 35, BC4 77, BC5 333, BC6H 268, BC7 5,167, zlib 918,223, raw 5 | C | The two tables differ by exactly the 11 pack stores (SixNG rows sum to 1,014,499; +12,334 = 1,026,833 total; ours) | SixNG 05; Leonida textures.md | holdmysocks; Kam471 |
| K4 | BCPack is 11.6 GiB of 57.7 GiB chunk payload; BC1 is 10.4 GiB of it; saving BC1 27.3 %, BC7 10.2 %, BC5 18.8 %, BC6H 14.2 %, BC4 6.2 % (incl. tile padding) | C | 57 pools, 1,026,833 chunks | Leonida textures.md | Kam471 |
| K5 | A "stream" is not a whole texture: a chunk is a 4 KiB-4 MiB range of the virtual space. Mips go smallest first, so lower mips often sit in zlib chunks | C | descriptor/virtual-offset geometry | SixNG 05 ("Lower-mip recovery") | holdmysocks |
| K6 | Stores are padded to a 256-byte boundary; BCPack padding is zero-filled | C | min(16,pad) bytes zero on all 108,229 padded BCPack chunks (102,368 BC1, 35 BC3, 76 BC4, 332 BC5, 268 BC6H, 5,150 BC7) | Leonida textures.md | Kam471 |
| K7 | No magic, version, length or format at chunk offset 0; payload entropy 7.95-7.99 bits/byte; no zlib/gzip/bzip2/LZMA wrapper | C | 300 sampled chunks per codec (Leonida); "7.9-8.0 bits/byte, flat autocorrelation" (SixNG) | Leonida; SixNG 06 | Kam471; holdmysocks |
| K8 | Texture layout: AMD Gfx10. tile mode 0 = `SW_64KB_D`, modes 1-3 = `SW_4KB_S`; in-tile bit orders (low to high) 16-byte blocks 64K `x0 y0 x1 y1 y2 x2 y3 x3 y4 x4 y5 x5`, 4K `y0 y1 x0 x1 y2 x2 y3 x3`; 8-byte blocks 64K `x0 y0 x1 x2 y1 y2 x3 y3 x4 y4 x5 y5 x6`, 4K `x0 y0 y1 x1 x2 y2 x3 y3 x4`; tiles row-major, no pipe/bank XOR | C | "mip n+1 equals mip n downsampled (mean error 0-4 levels)" on BC1/3/4/5/7 in every mode; 64K order beats `SW_64KB_S` on 161/200 (BC1/BC4) and 191/199 (BC3/5/7) by block-edge error | SixNG 05; ttpf-viewer `src/tile.rs`, `layout.rs` | holdmysocks; paryzius |
| K9 | Mip-tail: mips whose element size <= half the block are packed in one shared block (8 mips in 4 KiB blocks, 12 in 64 KiB); tail mip k origin from `off = m>6 ? 16<<m : m<<8` | C | same downsample check; also Xbox2Windows draws 10 real low-res tail mips on D3D12/WARP against CPU reference (PR #16) | SixNG 05; Xbox2Windows `tools/bc7_tiles.py` | holdmysocks; R0Y00 |
| K10 | Arrays and cube maps have their own layouts (level-by-level; cubes store (tile mode - 8) complete copies, smallest first) | C | 563/567 cubes and 118/130 arrays decode | SixNG 05 | holdmysocks |
| K11 | Standard order of hardware stages: DEFLATE, then BCPack, then swizzle. `IntermediateSize` is the zlib output size and BCPack input size. BCPack file offset must be 16-byte aligned | C (vendor doc) | Microsoft docs | Learn DirectStorage overview, `DSTORAGE_REQUEST` | Microsoft |
| K12 | `ETPF` index: `0x27A` version, descriptors 16 bytes (w-1 14 bits, h-1 14 bits, mips-1 4 bits; DXGI in d1 bits 3-9; array size-1 bits 10-19; tile mode bits 24-27) | C | 357,115/357,115 descriptors | SixNG 05, ttpf-viewer README | holdmysocks, paryzius |
| K13 | BC1 and BC4 chunks always have byte 1 = `0xde` (102,725/102,725 and 77/77); BC3 and BC5 always `0xdb` (35/35, 333/333); BC6H byte 0-1 in `0x30`-`0x36` (268/268); BC7 byte 1 in `0x30`-`0x33` in 4,978/5,167 | C | whole-corpus header scan | Leonida textures.md | Kam471 |

### 1b. Header / records (layer: schedule)

| # | Fact | Conf | Evidence | Source | Who |
|---|---|---|---|---|---|
| H1 | The first 48 bytes hold the 6-byte header records of 8 "main" lanes, **column-interleaved**: byte `r*8 + lane` is lane `lane`'s byte in round r (rounds 0-5) | C | BC1 header checked on 42,000 chunks (prefetch rule 41,988/42,000); BC7 refill positions verified byte by byte on a real 4,096-block chunk | SixNG 06; Marlom #15 | holdmysocks; MarlomMedeiros |
| H2 | Mode byte `hdr0`: bits 7:6 = `z` (narrow alphabet by z bits), bit 5 = `hdr3` (3-byte header), bit 4 = `flush_per_frag`, bits 3:0 = `q` (model 0-15). Length: `hdr3=0` -> `input_remaining = b1 + 1`; `hdr3=1` -> `u16(b1,b2) + 257`. `input_remaining` counts the 4 state bytes | C | Microsoft RTL states it; Marlom: 0 negative stream budgets across all 5,120 BC7 chunks, 0 violations over 11,225 bit5=0 records, 3,354 bit5=1 records with zero high length byte; SixNG independently: `0x30-0x3f` lanes have 16-bit length, `0x1c-0x1e` 8-bit; `bcpack-rs` budgets sum exactly to chunk size on real multi-fragment chunks | `mdu_minirans.v` DSTATE_HDR0-2; Marlom #15; bcpack-rs README | Microsoft; MarlomMedeiros; Ahmet-Dedeler (earlier agent) |
| H3 | Record shape: `[mode][len8][state4 BE]` (bit5=0, total `len8+3` bytes) or `[mode][len16 LE][state_hi3]` + 4th state byte read in a later round (bit5=1) | C | as H2 | Marlom #15 | MarlomMedeiros |
| H4 | Rounds are **fixed 8-byte slots** (feeder width 8 bytes: `MDU_FEEDER_INPUT_COUNT_LOG2 = 3`). Round 6 (bytes 48-55) carries 4th state bytes of bit5=1 lanes + first prefetch/refill bytes, and **secondary lane headers claim any slot a main lane frees**. Round 7 always starts at offset 56 | C (one 4,096-block chunk, byte by byte) | Marlom's table: `prefetch_mask=0xef`, lane 3 refills at offsets 51 and 59, lane 7 at 54 and 61, secondary `s0` mode at 55, len at 62, `s1` mode at 63 | Marlom #15; RTL constant | MarlomMedeiros |
| H5 | Per-lane prefetch: a lane keeps one byte in `input_buf`; constant lanes with `n-1 > 3` take it after the header; length-coded lanes always | C | 41,988/42,000 BC1 chunks (lane 1) | SixNG 06 | holdmysocks |
| H6 | Later refills are **demand-driven**: `refill[i+1]` appears in the stream where `refill[i]` is consumed at its decode step, not in fixed rounds | C (2 streams: one 16,384-block, one dual-32,768) | five refills each landed as multi-lane byte clusters within ~0.001 of predicted decode-step fraction | Marlom #15 | MarlomMedeiros |
| H7 | Per-lane refill consumption is FIFO; reads beyond a lane's budget are "unbudgeted reads" | C | matches real streams | Marlom audit of PR #175 | MarlomMedeiros |
| H8 | The hardware has **18 decoder instances** (`MDU_RANS_NUM_DEC 18`), 9-bit symbol output (`SYMOUT_W 9`), 32-bit state, 16-bit probability, 8-bit renorm | C (vendor RTL) | RTL defines | `mdu_minirans.v` lines 26-39 | Microsoft |
| H9 | Two-phase datapath (phase 1: read input, inverse distribution, write symbol; phase 0: forward distribution, update state); stall on feeder starvation or symvec back-pressure | C | RTL comments | `mdu_minirans.v` | Microsoft |
| H10 | Secondary lanes ("s0, s1, ...") exist after the 8 main lanes; BC1 has 4 index lanes in this slot, BC7 has secondary lanes with modes `0xdc` (K=4), `0xdd` (K=8), `0xde` (K=16), `0x1b`... | C for existence; I for roles | Marlom: `0xdc` 3,072 symbols verified in a 1,536-block prefix, `0xdd` likewise | Marlom #15; SixNG 06 | MarlomMedeiros; holdmysocks |
| H11 | 65,536-block chunks split into two independent 32,768-block streams; 1,536/3,072-block BC1 chunks hold several smaller streams (chunk block count != stream block count) | C | Marlom (65,536); SixNG ("1,536- and 3,072-block chunks") | Marlom #15; SixNG 06 | both |

### 1c. rANS coder and probability models (layer: models)

| # | Fact | Conf | Evidence | Source | Who |
|---|---|---|---|---|---|
| M1 | Byte-wise rANS, 32-bit state, `L = 2^24`, probability precision `M = 2^16`, encoded in reverse, decoded forward, encoder start state `x = L`; decode: `s = x & 0xFFFF`, `x = f*(x>>16) + s - cf`, refill while `x < L` | C | RTL; SixNG byte-for-byte re-encode of constant lanes; Marlom 1,252/1,252 | RTL; SixNG 06; Marlom | all |
| M2 | Legal end states: `L`, `L>>8`, `L>>16` (compressor may trim). RTL only uses this in the `error_overrun` check; the FSM moves to the next fragment on `state == L` | C | RTL lines ~1070 and DSTATE_PROCESSING | `mdu_minirans.v` | Microsoft |
| M3 | Models are static, chosen by `q` (0-15) per fragment; segments are power-of-two: seg 0 = sym 0, seg k = syms `[2^(k-1), 2^k-1]` with one frequency per segment; slot 0 is never used (sym 0 gets `f-1` at `cf=1`) | C | RTL `fwd_f = (cfa==0)? fa-1 : fa; fwd_cf = (cfa==0)? 1 : cfa` | `mdu_minirans.v` | Microsoft; used by `bcpack-rs` |
| M4 | Constant lane = all symbols equal sym 0 with `f = M - K`, `K = 2^eab` for constant models (`eab=1,q>=11; 2,q>=12; 3,q>=13; 4-5,q>=14` etc.). `eab = alphabet_bits - z` | C | Verilog tables; SixNG K(`dc`)=4, K(`dd`)=8, K(`de`)=32; Marlom per-lane K | RTL; SixNG; Marlom | all |
| M5 | Per-lane K for constant lanes: `0xde` K=32 on lanes 2-6 (908 hits), K=16 on lane 7 (373/373; one extracted prefix decodes bit-exact to `LOWER` with K=16 and fails with K=32 at an escape near step 2579); `0xdb` K=2 on lanes 0-1 (184 hits); `0xdc` K=4 and `0xdd` K=8 on secondary lanes (3,072 symbols verified) | C (BC7, one package) | counts as stated; chance floor ~3e-6..5e-6 per record, expected false hits ~0.004 among 1,252 | Marlom #15 (two comments) | MarlomMedeiros |
| M6 | Every mode-`0xde` record with `budget-1 = 3` matches the pure-constant encoder-final state at its lane's K: 1,252/1,252 across 38 stores (167 distinct states; one repeats 466x); refill-consistent population among `0xde` records 1,281 of 1,737 | C | as stated | Marlom #15 | MarlomMedeiros |
| M7 | Symbol counts: endpoint lanes carry **2 x blocks** symbols (843 verified hits; N=1024 for a 512-block prefix, 3,072 for 1,536); other verified lanes carry fixed 128-symbol runs; 827 records equal 2x chunk blocks, 4 are 1x, 421 other verified records carry counts 7-9,703 untied to blocks | C (BC7) | as stated | Marlom #15 | MarlomMedeiros |
| M8 | BC1: lane 1 is always a constant `de` lane whose record depends only on the block count N | C | 41,988/42,000 + whole-corpus byte-1 scan | SixNG 06; Leonida | holdmysocks; Kam471 |
| M9 | BC1 endpoint planes: constant lanes have 5-/6-bit types `0x10` records `10 03 20 00 xx 00` (5-bit planes) or `10 03 40 00 xx 00` (6-bit planes). Read either as parameters (SixNG) or as 1-symbol uniform first fragments (`bcpack-rs`, matches Verilog: state `(L/f)<<16 + cf` returns to L after one symbol) | C for record shape; I for meaning | 7,162/7,162 `len==4` fragments end in a legal state across q=0..15 | SixNG 06; bcpack-rs SPEC 2.2 | both |
| M10 | Exact Microsoft model tables exist: `base_table[9][16][10]`, `freq_table`, plus division-free inverse tables `rf/rs/rn`. Our fitted exponential tables disagree on 51/128 models | C | `cmp_verilog_tables.py`: max base diff 64, 21 models off by >2 | `mdu_minirans.v` lines 95-900; ours | Microsoft; ours |
| M11 | Non-constant ("varying") lane zero-refill states (`0xda`, `0x5f`, `0x1f`) do **not** fit the constant recurrence for any K in {1,2,4..1024} nor any start offset. Signatures: `0xda` states advance in exact +4.0101-bit steps; `0x5f` states cluster in delta-2 pairs | C for the negative; S for the signatures | Marlom | Marlom #15 (second comment) | MarlomMedeiros |
| M12 | Type-`0x1b` records on BC7 lanes 0-1 are formally chain states at K=2 with degenerate n~128 | S | Marlom | Marlom #15 (first comment) | MarlomMedeiros |

### 1d. Semantics (layer: semantics) and block order

| # | Fact | Conf | Evidence | Source | Who |
|---|---|---|---|---|---|
| S1 | BCPack separates colour endpoints from palette indices and codes them with rANS; it is an entropy coder over BCn data and the output is standard BCn | C (vendor) | Microsoft doc: "color endpoints are separated from palette indices (that is, weights) and compressed using a rANS algorithm" | Learn DirectStorage overview | Microsoft |
| S2 | BC1 lane roles: lane 0 `0x30-0x37` carries 1.3-6 bits/block (per-block data, role unknown); lane 1 constant; lanes 2-7 = endpoint planes R0 G0 B0 R1 G1 B1 (565); 4 index lanes `0x33-0x38` (~7.2 bits per row per block, one lane per row of 2-bit indices) | C for planes (SixNG: "confirmed"), I for index lanes and lane 0 | solved constant planes by re-encoding every block; index inference from lane byte budgets | SixNG 06 | holdmysocks |
| S3 | BC1 indices cost ~30 bits/block, so BCPack saves almost nothing on indices; the gain is in endpoint planes | I | stream size accounting | SixNG 06 | holdmysocks |
| S4 | Index streams are coded in reverse: two near-identical textures share only the last ~60 bytes | I | 15 tail-sharing groups out of 3,259 BC1 textures | SixNG 06 | holdmysocks |
| S5 | BC7: endpoint lanes carry 2 x blocks symbols (two subsets or two endpoints per block?); lane-to-BC7-field assignment unresolved; per-lane mode alphabets suggest each lane streams a different field | I / S | Marlom M5-M7 | Marlom #15 | MarlomMedeiros |
| S6 | BC7 27-stream study (`ttdb_graphics.ttpf`): 27 BC7 chunks, block counts from virtual extents, stored size ~12 % under virtual; reads "stride-6 lane 1 carries plane type 25/27" | S (interpretation superseded by H1) | counts reproducible only locally | mesterember #15 comment | mesterember104-cpu (AI agent) |
| B1 | Block order inside a stream (linear vs swizzled), and whether Gfx10 tiling is applied before or after BCPack | unknown | none. MS: stage order DEFLATE -> BCPack -> swizzle, so a BCPack stream should decode to *linear* block rows before swizzle; the swizzle-tile bit order is solved (K8) | Learn overview | (inference ours) |

---

## 2. Contradictions between sources, with a judgment

| # | A says | B says | Judgment and why |
|---|---|---|---|
| X1 | SixNG `06` "K by mode: `dc` 4, `dd` 8, `de` 32" (one table for all lanes) | Marlom: `0xde` is K=32 on lanes 2-6 but **K=16 on lane 7**; `0xdb` K=2 on lanes 0-1 | **Marlom is right; SixNG's table is a BC1 special case.** Microsoft's RTL has `eab = alphabet_bits(lane) - z`, so K depends on the lane's configured width. Lane 7 simply has d=7 where lanes 2-6 have d=8. Also fits SixNG: BC1 lane 1 `de` K=32 needs d=8. The probe's `EXCEPTIONS = {dc:4, dd:8, de:32}` (Xbox2Windows) is therefore lane-blind; fix it as a table keyed by (codec, lane) |
| X2 | PR #175 round model: Round 2 starts at `48 + popcount(prefetch_mask)`; "skip Round 1, take `refill[0]` in Round 2" | Marlom audit: round 7 always starts at 56; secondary headers fill freed round-6 slots; 59/59 constant lanes take `refill[0]` in round 6; `--next-mask` only valid for escapes at decode step 0 | **Marlom is right on real data** (byte-by-byte check on a 4,096-block chunk where PR #175's formula lands on a `s0` mode byte at 55). PR #175 is correct only for its synthetic fixtures. Consistent with the RTL's fixed 8-byte feeder width (H4) |
| X3 | `bcpack-rs` SPEC: `hdr0 = z[7:6], hdr3[5], flush[4], q[3:0]` | SixNG: "3-bit parameter (top) + 5-bit type (bottom)" | **Same thing.** SixNG's 3-bit parameter = z (2 bits) + `hdr3`; its 5-bit type = `flush<<4 | q`. The RTL confirms the `bcpack-rs` split. This also resolves SixNG's "varying plane lanes: 16-bit length fits `3c` but gives impossible lengths for `1c`": `0x3c` has bit5 set, `0x1c` has bit5 clear (8-bit length) |
| X4 | mesterember104-cpu: BC7 header is 8 x 6-byte contiguous records (stride 6), "which is the SixNG section 6 layout"; lane 1 = plane type in 25/27 streams | SixNG 06 / Marlom: records are 6 bytes but the 48 bytes are column-interleaved (stride 8) | **Interleaved is right.** SixNG never said contiguous. mesterember's "stride-6 lane 1" is byte 6 of the prefix, which under H1 is lane 6, round 0 mode. Marlom's byte-level verification (H4) post-dates and contradicts it. PR #96 added `--lane-stride 6` for the diagnostic; harmless but unneeded |
| X5 | mesterember: 27 BC7 streams in `ttdb_graphics.ttpf` with 14x4096 + 10x16384 blocks | R0Y00: reading table from byte 16 with the final sentinel, 13x4096 + 11x16384 | **Open, small.** The difference is the last mode-7 segment: its block count must come from the logical end (virtual extent to the next entry or the final sentinel), not from its stored length. R0Y00's count is the more careful one. Use "next entry's virtual offset" and treat the table terminator as an end sentinel (K1) |
| X6 | SixNG 05 chunk counts (BC1 100,026, BC7 5,120, zlib 908,643) | Leonida (BC1 102,725, BC7 5,167, zlib 918,223) | **Not a conflict**: SixNG counts the 46 `data/ttdb` stores, Leonida all 57 pools including 11 `pack/*` (difference 12,334 = total 1,026,833 - 1,014,499; ours). Marlom's "46 files, 5,120 chunks" matches SixNG. `bcpack-rs` "100,020 BC1 chunks scanned across 46 stores" is off from SixNG's 100,026 by 6; likely a parse skip |
| X7 | `bcpack-rs` SPEC: codec chunk entry "bits 24..29" etc., model tables "fitted" | RTL base_table | **RTL wins.** 51/128 fitted models differ (M10). Fitted `lambda(q)` should be thrown away |
| X8 | `bcpack-rs` `BC1_LANE_AB = (8,8,5,6,5,5,6,5, 8,8,8,8)` and SPEC "0xde / 0x1e = eab 5, q 14" | RTL: `eab = d - z`; `0xde` has z=3, so eab=5 needs d=8; `0x1e` (z=0) needs d=5 | **Plausible and self-consistent** if lane 1 has d=8 (BC1 constant lane, K=32 observed) and endpoint lane 2-7 modes `0x1e`/`0x10` have d=5/6. Needs per-lane check on BC1 constant-plane lanes; unresolved which lanes use which d. Do not trust `BC1_LANE_AB` for BC7 (X1 shows d differs per lane: 4/5/6/7/8) |
| X9 | `bcpack-rs` README "0x10 records are 1-symbol fragments, remaining N-1 symbols come in a following `9c ...` fragment in Round 7" | SixNG: "`10 03 40 00 xx 00` are parameters, not states; their data is elsewhere in the stream" | **Compatible.** The record is a valid 1-symbol fragment under the RTL; what is unverified is bcpack-rs's claim that the *next* fragment header is read in round 7 (no real chunk was decoded). Treat the first half as C, the follow-on schedule as I |
| X10 | `bcpack-rs` "all BC7 lanes follow `hdr0` rules" | Marlom: 456 of 1,737 `0xde` records with `budget-1 > 3` do not fit the lane-K model; type `0x1b` on lanes 0-1 does not either | **Open.** Possibly second chains (multi-fragment lanes) or the `z`/`q` of later fragments; not a contradiction of H2 |
| X11 | Xbox2Windows README/#15 title implies BC7 reference needed | Everyone else has no plain texture | Agree: no public plaintext. `bc7_blocks.py` supplies *synthetic* BC7 blocks only, useful as an encode-side test oracle (see 5) |

---

## 3. Explicitly ruled out (do not retry)

From SixNG 06 section 6.2 (holdmysocks), BC1:
- Static frequency tables carried in the header; Huffman; byte-aligned planes.
- Shift-add and LZMA-style range-coder recurrences.
- Static 4-symbol tables for the index lanes.
- 81 adaptive index models (adaptive frequency tables with rates 1-8; LZMA-style binary trees with shifts 1-8) x 5 contexts x every start offset, both per lane and with the 4 index lanes joint. Caveat from SixNG itself: every chunk with index lanes also has lane 0 and a varying plane lane with unknown model, so the index lanes could not be isolated; this negative result is weaker than it looks.
- A general-purpose wrapper: zlib, deflate, LZMA, zstd, LZ4, bz2 (SixNG); gzip, bzip2 (Leonida).

From Marlom (BC7):
- Variable-lane zero-refill states (`0xda`, `0x5f`, `0x1f`) as the *constant* recurrence for any K in {1,2,4,...,1024} or any start offset.
- PR #175's round model on real streams (X2): "round 2 at `48 + popcount`", "skip round 1 then refill[0] in round 2", `--next-mask` as a general predictor.
- A single global mode->K map (X1).

Structural:
- Stride-6 contiguous header layout (X4).
- "The mode byte carries a free 3-bit parameter" (X3).
- No plain copy of any affected texture exists in the game (SixNG 06 intro; Leonida), so no known-plaintext route; SixNG's "one-block-difference pair" idea has only 15 tail-sharing groups among 3,259 BC1 textures checked.
- A full blind joint search is infeasible: SixNG estimate ~3 x 10^8 full decodes per chunk (~6 x 10^12 symbol steps, ~2 weeks on one desktop) with no early pruning.
- Decoder-from-docs: Leonida PR #82 concludes a clean-room decoder "not feasible from public facts" (this predates our discovery of the Microsoft Verilog, which is public).

---

## 4. Open unknowns, ranked by how much they block decoding

1. **Models/layout of the varying lanes** (`0x30-0x3f` len16 lanes, `0x1c`, `0xda`, `0x5f`, `0x1f`, ...): which (eab, q, z) sequence each uses per fragment, and fragment boundaries. The model *tables* are now known exactly from the RTL; what is unknown is which `q`/`z` each fragment selects (this is in the stream's `hdr0` bytes, readable) and how fragment lengths chain. Per H2 the record is readable, so this should be mostly a schedule problem now, not a model-search problem. Highest-value next step: re-test Marlom's "does not fit the constant recurrence" result with the *exact* RTL tables for `q` read from the byte instead of K-sweeps.
2. **Read schedule for non-constant lanes (feeder).** Demand-driven refill order across lanes (H6) is confirmed only qualitatively; exact "when does a lane consume byte, in which lane order, per cycle" is exactly what the Verilog omits (feeder + symvec assembly). Needs both: lane order within a feeder window of 8 and the stall rule.
3. **Lane count and roles beyond 8 main lanes**: how many secondary lanes per codec (BC1: 4 index; BC7: unknown), with 18 decoder instances in hardware as upper bound; their launch round.
4. **Per-lane alphabet widths `d`** (needed to turn `z` into eab): partly inferred (X1/X8), need a table per codec per lane.
5. **Symbol-to-field mapping**: for BC1 (R0 G0 B0 R1 G1 B1, lane 0, 4 index rows, any delta/prediction between blocks) and BC7 (modes, partitions, rotation, index bits). Cannot be tested without a decoder; hard check = mip consistency.
6. **What lane 0 codes** (BC1) and what the type `0x1b` lanes on BC7 lanes 0-1 are.
7. **Inter-block prediction / delta coding** of endpoints (BC1 constant planes imply that endpoints are coded as vs. a base; unconfirmed). `bcpack-rs` guesses a first-symbol base + deltas (SPEC 2.2).
8. **Block order** within a stream (linear expected, swizzle applied afterwards per MS stage order).
9. **BC3/BC4/BC5/BC6H streams**: studied only to header level (and only 35/77/325/268 chunks); BC5 shares `0xdb` lane-1 constant with BC3.
10. Which of the 456 `0xde` records with `budget-1 > 3` and the 29 header-only matches are multi-fragment chains.
11. Ancillary: the 64-bit texture hash, ETPF flag bits, 3 cube maps with too many copies (SixNG 05), 4-/2-byte layouts (not BCPack related).

---

## 5. Reusable assets

All under `/Users/ahmet/Code/bcpack-refs/` unless stated. "Needs game data" means it reads local retail pools.

| Asset | What it does / verifies | Notes |
|---|---|---|
| `microsoft_silicon-rans/mdu_minirans.v` (MIT) | Authoritative decoder: tables, header FSM, `legal_end_state`, 8-byte feeder constants | **Use this as the model oracle.** Parse `base_table`/`freq_table` with `docs/findings/prior-work-assets/cmp_verilog_tables.py` (in our worktree) as a starting point |
| `docs/findings/prior-work-assets/cmp_verilog_tables.py` (ours) | Compares `bcpack-rs`' fitted bases with the Verilog (51/128 differ) | No game data. `python3 -I cmp_verilog_tables.py <mdu_minirans.v> <worktree>/python` |
| `R0Y00_Xbox2Windows/tools/bcpack_probe.py` | Per-lane constant-lane verifier: header budget + state match, forward decode with supplied refills, `--blocks`, `--prefetch-mask`, `--next-mask`, `--lane-stride`, `--scan-lanes` | **Fix needed before use on real BC7**: `EXCEPTIONS` is lane-blind (K=16 for lane 7 `0xde`, K=2 for `0xdb`) and models `blocks` not `2*blocks` for endpoint lanes. Marlom says unmodified probe verifies lanes 2-6 on real prefixes with `--blocks` = recovered symbol count |
| `R0Y00_Xbox2Windows/tests/bcpack_check.py` | 8 unittest cases, run `python3 -B tests/bcpack_check.py` -> "Ran 8 tests ... OK" (verified here). Covers synthetic constant lanes, supplied-refill decode (100,000 symbols mode `dd`: refills `93 E3`, state `030B87C1`, symbol indices 9,129 and 54,564), stride-6 scan, tail slots | Synthetic only. Good as encode/decode oracle |
| `R0Y00_Xbox2Windows/tools/bc7_blocks.py` (+`tests/bc7_blocks_check.py`) | Exact BC7 field layout (`layout`, `pack`, `unpack`, `decode`) for all 8 modes; `fixture()` builds 512 blocks with every mode/partition/rotation/index selection; decoded identically to `texture2ddecoder` (65,536 random blocks per mode) and Pillow | PR #106. Useful as a ground-truth pack/unpack for BC7 semantics experiments |
| `R0Y00_Xbox2Windows/tools/bc7_tiles.py` | `linearize_4k_s_tail`: Gfx10 `SW_4KB_S` mip-tail reader (checked against D3D12 hardware and WARP on 10 real low-res mips) | Needed for the *mip consistency* check of the lower mips |
| `paryzius_ttpf-viewer/src/{tile,layout,archive}.rs` | Exact Gfx10 detiling (bit patterns, tiled 4K/64K), mip layout, RDS5/ETPF parse, array and cube-map layout | Rust; can extract any plain mip as the "mip 1" reference for the downsample check |
| `holdmysocks_SixNG` (GPL-3.0) `crates/sixng-core/src/{ttpf,gfx10,texture}.rs` | Chunk table, per-chunk `bcpack_ranges()`, "largest mip outside BCPack", Gfx10 | GPL: do not copy code into the MIT/Apache repo; use as reference. No BCPack decoder anywhere in it (README: "support is not planned soon") |
| `TrinitronClient_GTA6-Asset-Info/formats/05-ttpf.md`, `06-bcpack.md` | Spec text for the container and the BC1 header experiments | GPL-3.0 documents. `walkthestars` copy lacks PR #2/#5 updates (padding, codec ids, arrays/cubes) |
| `ItsNotRexed_Leonida/tools/{tex_headers,tile_probe,tex_export,tex_mode_probe}.py` | Texture descriptors/DDS synthesis/tile probes (GPL-3.0) | not BCPack-specific |
| Marlom's scripts (unpublished) | Corpus-wide checks behind every M5-M7 number ("happy to share them") | Ask via #15 later (coordinator only; we do not post) |
| Hot Chips 2020 slide PDF (`hc32.hotchips.org/.../HotChips2020_GPU_Microsoft_Jeff_Andrews_v2.pdf`) | Decompression block slide; the server returned an HTML "Contact Support" page to `curl` and WebFetch, so slides were **not read**. Numbers in 7 come from secondary reports | retry in a browser if it matters |

---

## 6. People actively working on it (as of 2026-11-21 ~19:10 UTC)

| Handle | Who / background | BCPack activity | Last seen |
|---|---|---|---|
| `holdmysocks` (Brandon Villar) | Author of SixNG (GPL); `TrinitronClient/GTA6-Asset-Info` | Wrote `06-bcpack.md` (BC1 lane/rANS notes, 0 % decoded); keeps merging `05-ttpf.md` fixes (PR #4/#5, issue #6) in the TrinitronClient fork | pushes 2026-11-21 03:42 UTC |
| `walkthestars` | Original publisher of the notes at `walkthestars/GTA6-Asset-Info` (issues disabled) | Notes first published 2026-11-20 | 2026-11-20 20:52 |
| `jaylikesbunda` (deki) | Contributor | PR #2 to the notes: ttpf padding and codec ids | 2026-11-21 |
| `R0Y00` (Royoo) | IC design student (STM32/Zynq/FPGA/SoC, CQUPT); owner of Xbox2Windows | Opened #15, wrote the probe (PR #16, #17), merged the rest, responds fast | 2026-11-21 19:05 |
| `MarlomMedeiros` | Single-repo account (created 2017), no public repos of note | **The most important contributor**: corpus-wide BC7 findings in #15 (K table, header bit5, 8-byte rounds, demand refill), audit of #175, comment on #214. Offers the scripts | 2026-11-21 18:54 |
| `Vonsek` | Very active contributor (many Xbox2Windows PRs) | PR #106 (BC7 block reference), #175 (per-lane refill), #214 (supplied-refill decode) | 2026-11-21 19:09 |
| `descent81` | Contributor | PR #69 (second refill byte check) | 2026-11-21 |
| `raulmoracode` | Contributor | PR #96 (stride-6 diagnostic), repo meta/AGENTS.md | 2026-11-21 18:42 |
| `mesterember104-cpu` | Account created 2026-11-09, declared AI agent (GLM) | One #15 comment: BC7 27-stream stats, stride-6 idea (superseded) | 2026-11-21 13:39 |
| `Kam471` | Leonida contributor | PR #82: whole-corpus BCPack gap census and codec mapping | 2026-11-21 19:09 |
| `ItsNotRexed` (NotRexed) | Leonida maintainer, high-school student porting games | Merged #82; owns `rpf9-explorer`-based export | 2026-11-21 19:11 |
| `paryzius` | `ttpf-viewer` author | Exact Gfx10 detile + RDS5/ETPF spec, no BCPack work | 2026-11-21 17:47 |
| `LooperSalty` (AnyXboxX), `californiaAlmonds` (OpenScarlett), `penwin2000` (open-vfs) | Other 2026-11-21 projects touching the same package | mention BCPack only as unverified (AnyXboxX `docs/recherche/paquet/ANALYSE_ETPF.md`); no decoding work found | 2026-11-21 |

Search of GitHub issues/PRs for "BCPack" (2026-11-21) found nothing else; only Xbox2Windows #15 and #132/#93 (mentions), Sewer56/dxt-lossless-transform #120 (attribution), AnyXboxX #14, open-vfs #14 (container). `microsoft/Xbox-GDK-Samples` has only the enum (`BcpackMode = NONE` in samples).

---

## 7. Web sources beyond GTA

| Source | Date | Trust | What it gives |
|---|---|---|---|
| Microsoft Learn: DirectStorage Overview (Xbox Series) | page updated 2026-11-14 | A- | Stage order DEFLATE -> BCPack -> swizzle; BCPack is "a custom entropy coder designed specifically for BCn data" with endpoints separated from weights and rANS-compressed; hardware decompression units (multiple, run in parallel for requests >512 KiB); `IntermediateSize`; 16-byte file alignment. https://learn.microsoft.com/en-us/gaming/gdk/docs/features/console/storage/directstorage/directstorage-overview |
| Microsoft Learn: `DSTORAGE_BCPACK_MODE` | 2022-08, updated 2025-11 | A- | NONE=0, BC1..BC5=1..5, BC6H=6, BC7=7 (BC2 present in enum, absent in the game). 4-bit field in `DSTORAGE_REQUEST_OPTIONS`. https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/dstorage/enums/dstorage_bcpack_mode |
| `microsoft/silicon-rans` | 2023-11-09 / 17 | A | See section 0. https://github.com/microsoft/silicon-rans |
| Patent US11234023B2 (Gladding, Gopalakrishnan, Kumbhani, Lin; filed 2019-06-28, issued 2022-01-25) | 2022 | A | Same decoder in patent form incl. Fig. 14/15. The agent `bcpack-patent` is reading it; the Verilog is the same author (Gladding). https://patents.google.com/patent/US11234023B2/en |
| Xbox News "A Closer Look at Xbox Velocity Architecture" | 2020-07-14 | A- | Marketing-level: decompression block, BCPack as texture-specific codec |
| Hot Chips 2020 (Microsoft, Jeff Andrews) | 2020-08-17 | A (not read) | Secondary reports: 2 decompression engines, >6 GB/s output. PDF not retrievable here |
| Eurogamer/Digital Foundry (Goossen) via Beyond3D quotes | 2020 | B- | "a high-speed hardware decompression block that can deliver over 6GB/s"; BCPack "tailored to the GPU textures" |
| Richard Geldreich ("dark horse") via GamingBolt | 2020-03-23 | C | Opinion that BCPack could beat Kraken on textures; no technical detail |
| Wikipedia S3TC | n/a | C | Says BCPack is a Microsoft texture compression for BCn textures (secondary) |
| `Sewer56/dxt-lossless-transform` (created 2024-12-17, pushed 2026-11-16) | 2024-2026 | B | Not BCPack: BC1/2/3/7 lossless *pre*-transforms (split colours/indices) before zstd. Same general idea of endpoint/index separation; useful only as a design analogy and for fast estimators. No evidence it is related to MDU |
| Xbox GDK public samples (`microsoft/Xbox-GDK-Samples`) | n/a | A | Only the API surface (`BcpackMode`), no codec data |
| XBTC (Xbox Texture Compressor) developer docs | n/a | A- (not read) | Official tool that produces the packed texture archives; docs moved (301 to `docs.xbox.com`) and could not be read; likely NDA-limited |

Not found anywhere (searched 2026-11-21, standard web + GitHub code/issue search): any other game's modding community hitting BCPack (Forza, Halo Infinite,
Starfield, Flight Simulator), any xboxdevwiki/Xenia/xbox-emulator BCPack discussion, any open-source BCPack decoder, any "bcpack" repo other than ours
(`ykhwong/bcpack`, `pomncr/bcpacking`, `Bluecraft-Server/BcPack` are unrelated).

---

## 8. Claims in `bcpack-rs` (our repo), graded

| Claim (README / `docs/BCPACK_SPEC.md`) | Grade | Note |
|---|---|---|
| Fragment header bitfields; budgets sum to chunk size | **C** | Matches the Microsoft RTL and Marlom's 5,120-chunk validation |
| 2-byte vs 3-byte length forms (`b1+1`, `u16+257`) | **C** | RTL |
| Slot-0 reservation, legal end states `{L, L>>8, L>>16}` | **C** | RTL |
| Constant lane `f = M-K`, `K = 2^eab`: `0xdb` 2, `0xdc` 4, `0xdd` 8, `0x5e` 16, `0xde` 32 | **C** with caveat | True per RTL; the *lane that uses which eab* varies (X1) |
| `lambda(q)` fitted exponential tables | **Wrong for 51/128 models** | Replace with RTL tables (M10) |
| `BC1_LANE_AB` | **I/unverified** | See X8 |
| Two-phase feeder, Round 6 launches index lanes, Rounds 7-12 barrier, `blocked_on_output` | **S** | Not supported by real data; contradicted in part by Marlom (X2). The Round-6 secondary launch claim is partly consistent (secondary headers fill round 6/7) |
| `0x10` records are one-symbol uniform base fragments | **C** for the record, **I** for the follow-on | X9 |
| `decode-bc1` output | **Wrong** | README itself: ~92 % of index words have zero top byte; no lane schedule |
| Lane layouts for BC3-BC7 | **S** | guesses |
| 3,674 real BC1 chunks, 7,162 `len==4` fragments converge 7,162/7,162 | **C but weak** | One-symbol fragments pass trivially for a correct constant model; they test table correctness (constant ones), not the schedule |
