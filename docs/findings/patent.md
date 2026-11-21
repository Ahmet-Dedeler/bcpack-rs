# Patent US11234023B2 and the MDU rANS reference Verilog: what they establish

Agent: `patent`. Sources read in full:

- US11234023B2, "Features of range asymmetric number system encoding and decoding" (Microsoft Technology Licensing; inventors Gladding, Gopalakrishnan, Kumbhani, Lin; filed 2019-06-28, granted 2022-01-25). Full description and claims from Google Patents, plus all 24 drawing sheets (full-resolution PNGs from patentimages.storage.googleapis.com and the 6.4 MB PDF). Local copies are in `/tmp/bcpack-patent/` (description.txt, claims.txt, fig/D000NN.png, fig/rot/ for the upright code listings).
- `mdu_minirans.v` from github.com/microsoft/silicon-rans (MIT, commit 2f6827d, 2023-11-17). This is the same `module decoder` that the patent prints as FIGS. 15a-15k. The difference is that the Verilog includes the lookup-table values, which the patent leaves out ("assignment of values ... omitted for brevity").

Deliverables in this worktree:

- `python/gen_mdu_tables.py` reads the five tables out of the .v file programmatically and writes `python/mdu_tables.py`. No table value was typed by hand.
- `python/patent_ref.py` is a stdlib-only port of the Verilog. It has a cycle-accurate two-phase lane, a functional lane decoder, a synthetic encoder and a multi-lane array with a pluggable feeder. It is independent of `bcpack.py`.
- Reproduce everything below with `python3 -I python/patent_ref.py` (self-tests) and `python3 -I python/patent_ref.py --diff-bcpack` (table diff).

---

## 1. The most important facts (with citations)

1. **The patent and the Verilog describe one rANS decoder lane and nothing about textures.** The patent calls the symbols "residual data for media (e.g., video, image, audio, texture for graphics)". It never mentions BC1-BC7, endpoints, blocks or Xbox. The only tie to the console is the `MDU_` macro prefix (FIGS. 15a-15g; `MDU_RANS_LOWER_LIMIT`, `MDU_COMPRESSED_ITEM_WIDTH`). The patent explicitly leaves out the **feeder** and the **decoder array / symbol-vector assembly**: "code is not shown for a feeder module ... and decoder array module (which coordinates operations of multiple instances of RANS decoder, when output symbols are interleaved in the encoded data)" (description, section VI.M, first paragraph). The Verilog README says the same: both are "left as an exercise for the reader".
2. **The exact probability model is now known.** It covers all 9 alphabet widths (1-9 bits) × 16 models. Each model is piecewise-constant over power-of-two segments. Segment 0 is symbol 0, and segment k is symbols [2^(k-1), 2^k), all with the same per-symbol frequency `freq[k]`. `base[k]` is the cumulative start, and the total is 2^16 (FIG. 15b declarations; FIG. 15f/15g forward path; mdu_minirans.v l.95-905 values). All 144 models pass two checks: `base[k+1] == base[k] + freq[k]·2^(k-1)` with a sum of 65536, and the reciprocal-multiply inverse is exact for every cf in 1..65535 (`patent_ref.selftest` step 1).
3. **Slot 0 is reserved and L is a sentinel.** The forward path uses `fwd_f = fa-1, fwd_cf = 1` whenever `fwd_cfa == 0`, i.e. for symbol 0 (FIG. 15g). So cf = 0 is never produced. An encoded state therefore always has nonzero low 16 bits and never equals `L = 2^24` in mid-stream. The decoder treats `x > L` (strict) as "a symbol is available" (FIG. 15e `next_sym_buf_full`), treats `x < L` as "refill a byte" (FIG. 15c `want_to_feed_rans`), and treats `x == L` as "this sub-stream is finished".
4. **Byte order and refill.** The state is 32 bits. A refill is `x = {x[23:0], byte}` (big-endian shift-in), at most one byte per two-phase iteration (FIG. 15c). The decoder has no separate "load initial state" step. After a flush the state is 0, and the normal refill rule shifts in bytes until `x >= 2^24`. That is 4 bytes when the first byte is nonzero (patent text, FIG. 11b discussion: "the first four bytes ... as four iterations through phase 1"). Those bytes count against the fragment's length.
5. **Header layout** (FIG. 15d/15i): `hdr0[7:6] = z`, `hdr0[5] = hdr3` (3-byte header), `hdr0[4] = flush_per_frag`, `hdr0[3:0] = q`. The length is `b1 + 1` (2-byte header) or `(b2<<8 | b1) + 257` (3-byte header). Note that the patent prose says "1 ... 257 bytes" for the 1-byte form, but the code gives 1..256.
6. **Width convention.** The tables are indexed `[eab][q][seg]`, and index 0 is the **1-bit** alphabet (Verilog comments). Segment k is enabled iff `eab >= k-1` (FIG. 15d/15e comparator chain). So the RTL's `eab` and the decoder input `alphabet_bits` are **bits - 1**. Width reduction: `eab = alphabet_bits - z`, and a 4-bit underflow is clamped to 9 (FIG. 15d). The patent's own example "base_table[8][12]" is 1-based: 8-bit alphabet, 12th model, which is q = 11 in the Verilog. The values match exactly: `[0,7575,14276,25440,41008,56352,64256,65344,65408,0]`.
7. **Fragment end is driven by bytes, not by symbol count** (FIG. 15j):
   - In PROCESSING, once the fragment's last byte has been merged: `x == L` means go straight to HDR0; `x != L` means go to DRAINING.
   - DRAINING keeps decoding symbols without refills while `x > L`. When `x <= L`, it goes to the next header, or to IDLE with `done` if `out_target` is reached.
   - There is no per-fragment symbol count. The new fragment's model (q, z) applies from the first symbol that needs a byte from the new fragment.
8. **The flush bit acts at the end of the fragment that carries it** (FIG. 15i stores `flush_per_frag` at HDR0, and FIG. 15j applies it when *that* fragment ends). The patent prose (section VI.L, "state re-initialization flag") describes it as applying to the fragment that carries it. The RTL does the opposite: fragment k's bit decides whether fragment k+1 starts from state 0. The first fragment of a lane always starts from 0, because IDLE→go zeroes the state (FIG. 15h).
9. **Legal end states** (FIG. 15g): `error_overrun` fires only when the last symbol (`output_remaining == 0`) is handed downstream and the state after its forward update is not one of {L, L>>8, L>>16}. The comment says "we can terminate slightly early due to trimming in the compressor". The mechanism: the encoder starts at L. When it renormalises before the first-encoded symbol, it emits one 0x00 byte if f ≤ 256 and a second one if f ≤ 1. These are the last bytes in decode order, and the compressor drops them. So L>>16 can only occur when the last symbol has f = 1, and L>>8 only when f ≤ 256.
10. **The byte budget must match exactly.** After `out_target` is reached, a lane still merges bytes while `x < L`. If any byte is left over once `x >= L`, the RTL can never reach `done` (it deadlocks). So "consumes exactly its budget and ends in a legal state" is a correct hard check.
11. **The per-lane configuration is not in the bitstream.** `alphabet_bits` (default width) and `out_target` (symbol count) are module inputs (FIG. 15a), set by whatever launches the decoder. They must come from the texture format and the block count. Only q, z, flush and the length are in-band.
12. **Array-level constants** (Verilog `define`s only; not in the patent text):
   - `MDU_RANS_NUM_DEC = 18`: decoders per array.
   - `MDU_RANS_FEED_WINDOW = 8`: "number of items fed to the decoder array each update cycle".
   - `MDU_FEEDER_INPUT_COUNT = 8` bytes, `MDU_COMPRESSED_ITEM_WIDTH = 8`, `MDU_MAX_INPUT_LOG2 = 18`, `MDU_RANS_SYMOUT_W = 9`.

   The phase comments in FIG. 15a / l.46-64 give the array protocol:
   - PHASE 1: the decoder reads its input buffer and writes its symbol; the symvec assembly outputs a complete symvec.
   - PHASE 0: the feeder writes input buffers; the decoder applies the forward update; the symvec assembly transfers the symbol from the decoder's output buffer to the symvec buffer.
   - A decoder stalls if the feeder starves it, or if its output buffer is full and downstream is not ready.

## 2. What this changes versus what we believed

- **The fitted λ(q) tables in `docs/BCPACK_SPEC.md` §3.3 / `bcpack.py` are wrong for 51 of 128 models** (bits 1..8). Running `python3 -I python/patent_ref.py --diff-bcpack` prints all of them. The differences are 1-5 counts per entry, which is enough to break exact decoding of any varying lane that uses those models. The 9-bit models are not covered at all. The truncated-exponential formula is only an approximation of the real tables. Use `python/mdu_tables.py` instead.
- **One of the spec's "exact solved" examples is wrong.** The spec gives (4-bit, q=2) as `[7539, 6867, 5975, 4531, 2632]`. The true values are `[7539, 6867, 5973, 4532, 2632]`. The other EXACT_TABLES entries in bcpack.py match.
- **All constant-lane parameters in spec §3.2 are correct** (checked against the tables). The constant models are 1-bit q≥11, 2-bit q≥12, 3-bit q≥13, 4/5-bit q≥14, and 6/7/8-bit q=15 only.
- **The 9-bit alphabet exists** (seg 9 = symbols 256-511, eab index 8). The spec clamps everything to 8 bits.
- **Flush timing is reversed** relative to the spec text (fact 8). This matters wherever bit4 differs between consecutive fragments of a lane.
- **The spec's one-symbol uniform record formula is wrong for symbol 0.** Spec §2.2 says a uniform 5-bit one-symbol record holds `0x20000000 | sym<<11`. That is true for sym ≥ 1, but for sym = 0 the RTL gives `0x20040005` (f = 2047, cf = 1). For 6-bit it gives `0x40100011` instead of `0x40000000`. A real `10 03 20 00 00 00` record therefore does **not** decode to symbol 0 under the RTL. Any such records in the corpus need re-checking.
- **The legal end states are explained.** {L, L>>8, L>>16} come from trimming up to two trailing 0x00 bytes. The spec's alternative explanation (encoder starts at 0x100) is mathematically equivalent but only applies when the final symbol has f = 1.

## 3. Decoder spec (exact, one lane)

Configuration: `alphabet_bits` (= width-1, 0..8) and `out_target`. Input is the lane's own byte sequence. In hardware a feeder interleaves the lanes; that interleave is not specified anywhere we have.

```
x = 0; out = []
loop over fragments:
  hdr0 = next byte; z = hdr0>>6; hdr3 = hdr0>>5&1; flush = hdr0>>4&1; q = hdr0&15
  b1 = next byte; len = hdr3 ? (next byte<<8 | b1) + 257 : b1 + 1
  eab = (alphabet_bits - z) & 15; if eab >= 10: eab = 9        # 9 would index past the tables
  model = TABLES[eab][q]
  draining = false
  repeat:                                   # one two-phase iteration
    if !draining and x < L and len > 0: x = (x & 0xFFFFFF) << 8 | next byte; len--
    sym = none
    if x > L and |out| < out_target: sym = inverse(model, x & 0xFFFF); out += sym
    if !draining and len == 0:
       if |out| == out_target: DONE (check fwd(x,sym) in {L, L>>8, L>>16})
       elif x == L: next fragment
       else: draining = true
    elif draining and x <= L:
       if |out| == out_target: DONE else next fragment
    if sym != none: x = fwd(model, x, sym)
  on "next fragment": if flush (of the fragment just finished): x = 0

forward(sym): seg = bitlength(sym); start = seg ? 1<<(seg-1) : 0
   cfa = (base[seg] + freq[seg]*(sym-start)) & 0xFFFF
   (f, cf) = cfa == 0 ? (freq[seg]-1, 1) : (freq[seg], cfa)
fwd(x, sym) = f*(x>>16) + (x & 0xFFFF) - cf
inverse(cf_in): seg = highest k in 9..1 with cf_in >= base[k] and (k == 1 or eab >= k-1), else 0
   steps = (((cf_in - base[seg] + rn[seg]) & 0xFFFF) * rf[seg] >> 15) >> rs[seg]
   sym = (seg ? 1<<(seg-1) : 0) + steps          # == start + (cf_in - base[seg]) // freq[seg]
```

Two notes on edge cases:

- `x == L` with bytes left in PROCESSING, or `x >= L` with output complete and bytes left, is a hardware deadlock. A valid encoder never produces either.
- When state is retained (flush = 0) across a fragment boundary, the carried state is < L. A carried state of exactly L would deadlock.

## 4. Transcription map: patent figures and Verilog

I read every listing sheet upright: `fig/rot/D00014..D00024.png` = FIGS. 15a-15k, and `fig/D00013.png` = FIG. 14. Line by line, they match `mdu_minirans.v` except for the omitted tables. Rather than retyping roughly 300 lines of public MIT code, here is where each figure lives in the .v file:

| Patent | Content | mdu_minirans.v |
|---|---|---|
| FIG. 14 | Bitstream: g fragments, each header (z 1421, model 1422, re-init flag 1423, extra-length flag 1424, length 1425, optional extra length 1426) + payload 1430 | (header FSM l.1141-1175) |
| 15a (1501) | phase/stall comments, `include "mdu_defines.vh"`, ports clk, nrst, alphabet_bits[3:0], go, out_target[31:0], array_stall, done, error_overrun, din_valid, din, din_ready, din_req | l.24-85 (the patent omits the `define block l.26-42) |
| 15b (1502) | sym_valid, sym[SYMOUT_W-1:0], sym_ready; `eab`, `current_q`; tables `[9][16][10]` with sel1/sel2 (rs 5-bit, rn 1-bit) | l.86-905 (values only in the .v) |
| 15c (1503) | ctrl_state enum IDLE..DRAINING = 0..5, registers, want/will_feed_rans, rans_state_with_input, need_ib_load, stall detect | l.906-946 |
| 15d (1504) | din_req/din_ready, hdr0 fields, eab clamp, cf_in, inverse chain seg 9..5 | l.948-985 |
| 15e (1505) | inverse chain seg 4..0, rf/rs/rn, add_mul, inv_steps, new_sym, sym_valid, next_sym_buf_full, new_output_remaining | l.986-1020 |
| 15f (1506) | forward priority encoder on sym[8:0], fwd_base, fwd_fa | l.1022-1061 |
| 15g (1507) | fwd_steps/cfa/f/cf/p, new_rans_state_p0, legal_end_state, reset block, error_overrun | l.1063-1100 |
| 15h (1508) | PHASE 0 (state propagate, input, output), PHASE 1 IDLE | l.1102-1140 |
| 15i (1509) | HDR0, HDR1, HDR2 | l.1141-1175 |
| 15j (1510) | PROCESSING, DRAINING | l.1176-1216 |
| 15k (1511) | default, `phase <= ~phase`, endmodule | l.1217-1224 |

The patent also prints one table excerpt (section VI.M):

- `base_table[8][12] = [0, 7575, 14276, 25440, 41008, 56352, 64256, 65344, 65408, 0]`
- `freq_table[8][12] = [7575, 6701, 5582, 3892, 1918, 494, 34, 1, 1, 0]`

Both equal Verilog `[7][11]`.

Typos in the patent listing:

- FIG. 15b says "assignment of values to base_table omitted" under `freq_table`.
- The prose says "eab_unclamped indicates the adjusted symbol width after clamping".
- The prose says "1...257 bytes" for the 1-byte length form.

## 5. Classification of `docs/BCPACK_SPEC.md` claims

| # | Claim (spec section) | Verdict | Evidence |
|---|---|---|---|
| 1 | BCPack = MDU entropy layer over BC1-BC7 (§1) | NOT-IN-PATENT | The patent is generic rANS; "texture" appears only as an example; the MDU_ prefix is the only link |
| 2 | Codec id = BC#·4 in chunk bits 24-29 (§1) | NOT-IN-PATENT | container format |
| 3 | Per-format lane counts and field roles, e.g. BC1 8 main + 4 index lanes (§1 table) | NOT-IN-PATENT | Verilog only says the array has 18 decoders |
| 4 | Each lane is a sequence of g ≥ 1 fragments; a new header follows in-band when bytes are exhausted and x ≤ L with output remaining (§2) | SUPPORTED, refined | FIG. 14, 15j; precisely: PROCESSING needs x == L, otherwise DRAINING continues while x > L |
| 5 | hdr0 = z[7:6], hdr3[5], flush[4], q[3:0] (§2.1) | SUPPORTED | FIG. 15d, 15i |
| 6 | eab = min(9, d - z); alphabet = 2^eab (§2.1) | SUPPORTED with convention fix | RTL eab and alphabet_bits are bits-1 (tables index 0 = 1 bit); clamp is on the 4-bit wrapped value |
| 7 | hdr3=0: len = b1+1, 1..256 (§2.1) | SUPPORTED | FIG. 15i (patent prose wrongly says 257) |
| 8 | hdr3=1: len = (b1 \| b2<<8) + 257 (§2.1) | SUPPORTED | FIG. 15i |
| 9 | flush=1 zeroes state "at the end of the previous fragment" and reloads 4 bytes of the new fragment (§2.1) | CONTRADICTED (timing) | FIG. 15i/15j: the bit applies at the end of the fragment carrying it; reload is the ordinary refill from 0 (4 bytes if the first is nonzero) |
| 10 | flush=0 carries the state over (§2.1) | SUPPORTED | FIG. 15j; carried state must be < L |
| 11 | 16 static models selected by q (§2.1, §3) | SUPPORTED | FIG. 15b/15i, patent VI.H/VI.L |
| 12 | `10 03 ...` = uniform q=0 one-symbol fragment, x = 0x20000000 \| sym<<11 (§2.2) | PARTLY CONTRADICTED | true for sym ≥ 1; sym 0 gives 0x20040005 (5-bit) / 0x40100011 (6-bit) because slot 0 is reserved |
| 13 | After that symbol x returns to L, input 0, so HDR0 (§2.2) | SUPPORTED (mechanism) | FIG. 15j PROCESSING branch; "Round 7" timing is NOT-IN-PATENT |
| 14 | Power-of-two segments, seg k = [2^(k-1), 2^k), uniform freq per segment (§3.1) | SUPPORTED | FIG. 15d-15g; also the 9-bit width (seg 9), which the spec omits |
| 15 | Piecewise-linear cumulative frequency fwd_cfa = base + f·(sym-start) (§3.1) | SUPPORTED | FIG. 15g |
| 16 | Slot-0 reservation fwd_f = fa-1, cf = 1 (§3.2) | SUPPORTED | FIG. 15g |
| 17 | Constant-lane parameters 0xdb/0xdc/0xdd/0x5e/0xde/0x1f with f = 65536-K (§3.2) | SUPPORTED (by Verilog tables) | freq0 values 65535/65533/65529/65521/65505/65473/65281 |
| 18 | Models are a truncated exponential with λ(q) (§3.3) | NOT-IN-PATENT; CONTRADICTED by tables | 51/128 fitted models differ; (4-bit, q=2) "exact" row is wrong |
| 19 | Legal end states {L, L>>8, L>>16} (§3.4) | SUPPORTED | FIG. 15g |
| 20 | Explanation: encoder starts from trimmed 0x100 with an f=1 symbol (§3.4) | SUPPORTED-EQUIVALENT, incomplete | RTL: encoder starts at L and trims up to 2 trailing 0x00; L>>8 needs f ≤ 256, L>>16 needs f = 1 |
| 21 | Rounds 0-5 = 48-byte column-interleaved main header, 1 byte per lane per round (§4.1) | NOT-IN-PATENT | consistent with FEED_WINDOW = 8 and with each decoder requesting at most one byte per two clocks |
| 22 | hdr3=0 lanes read 2 header + 4 state bytes, hdr3=1 lanes 3 + 3 then the 4th later (§4.1-4.2) | NOT-IN-PATENT (consistent) | the RTL reads header bytes back to back, then state bytes on demand; it also **prefetches** one byte whenever its buffer is empty and bytes remain (`need_ib_load`) |
| 23 | Index lanes launch at round 6; round 12 alignment barrier (§4.2-4.3) | NOT-IN-PATENT | feeder/array not disclosed |
| 24 | Lanes holding a valid symbol stall on output while others are in header states (§4.3) | NOT-IN-PATENT (mechanism SUPPORTED) | `blocked_on_output` is FIG. 15c; the symvec policy that drives sym_ready is not disclosed |
| 25 | Empirical convergence of 7,162 len==4 fragments (§5) | NOT-IN-PATENT | all are 1-symbol or constant models, which are identical under exact tables; not evidence for varying lanes |
| 26 | Byte-budget conservation on multi-fragment chunks (§5) | NOT-IN-PATENT (consistent) | header length semantics match FIG. 15i |

## 6. What neither source tells us (still needed)

- **Feeder / read schedule.** Which lane receives which byte of the shared chunk stream. The available hints are FEED_WINDOW = 8 items per update cycle, NUM_DEC = 18, and decoder-side demand (header bytes back to back, then one-byte prefetch whenever the buffer is empty and the fragment has bytes left, requests only in phase 0 and only when not blocked on output). `patent_ref.ArraySim` lets you plug in a feeder policy and run lanes cycle-accurately against a shared stream. Its default lane-order policy is a hypothesis, not a fact.
- **Symbol-vector assembly.** Phase-0 transfer from each decoder's output buffer into a symvec, phase-1 output of a "complete symvec". It is not stated how many symbols per lane a symvec holds, or how decoders map to symvec slots. Because a blocked decoder stops requesting bytes, this policy directly shapes the byte interleave.
- **Per-lane configuration.** `alphabet_bits` and `out_target` per decoder (not in the stream).
- **Semantics.** How symbols become BC fields: endpoint prediction/deltas, index packing, BC7 mode handling. The patent only says symbols are typically "residual data", and its generic encoder (FIG. 3) mentions prediction. There is nothing BC-specific.
- **Block order** (linear vs tiled): not mentioned.

## 7. Related patents

I checked:

- **Family:** continuation US20220109891A1, EP3991303B1, divisional EP4607803A2/A3, WO2020263438A1, CN114073009B / CN122178922A, KR102922298B1 / KR20260019670A. Same specification.
- **Other Gladding/Microsoft patents:** US9819359B1 (parallel symbol decoder), US20180246645A1 (multi-format pipelined decompressor), WO2020176239A1 (in-place safe decompression), US20180173642A1 (pipelined LZ decompression).
- **Co-inventor searches** (Kumbhani, Hsu-Kuei Lin, Gopalakrishnan at Microsoft 2017-2022): no other relevant hits.
- **Keyword searches** (texture + block compression + entropy at Microsoft, BCn + rANS): nothing relevant.

None of them describes BC texture layout, lane assignment, a feeder schedule or endpoint transforms. The cited references (Duda 2014/2015, Giesen "rANS in practice", Google ANS coefficient coding, etc.) are generic.

## 8. Tests and a direct test on game data

`python3 -I python/patent_ref.py`, all synthetic:

- **models:** all 144 (width, q) models are exact; every cf in 1..65535 lands in its symbol's [cf, cf+f) slot through the hardware reciprocal path.
- **single-fragment round trips:** 144/144, one per model.
- **multi-fragment lanes:** 196/196 round trips (random width, z, q, flush/retain, 1-3000 symbols, 2- and 3-byte headers). Functional decoder == cycle-accurate decoder on symbols, bytes consumed and end state; no `error_overrun`. 104 random specs were skipped because a retained-state boundary was not representable: the next symbol needs no refill byte, so the RTL would drain it under the old model.
- **retained-state lanes:** 31/31 all-retained 3-fragment lanes (flush=0 at every boundary).
- **corruption:** a one-byte corruption is rejected.
- **two lanes, one stream:** two lanes behind one shared stream (lane-order feeder) round-trip.

A direct test on real chunks was **not** possible in the time box. It needs layer 1 (the feeder interleave) to split a chunk into per-lane byte streams, and the per-lane `alphabet_bits` / `out_target`. Once another agent has a candidate lane split, run `patent_ref.decode_lane(lane_bytes, width-1, n_symbols)` on it. It gives the exact hardware result, and `done && legal_end && bytes_used == len` is the convergence check. `ArraySim` can also test feeder hypotheses directly against a whole chunk.
