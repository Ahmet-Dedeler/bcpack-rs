# Findings: sem-indices (BC1 index lanes 8-11, block order)

Agent `sem-indices`, 2026-11-21. Data: retail Xbox Series texture pools, read in place. Nothing
derived from game data is committed. Scripts are in `tools/sem/`, and scratch output goes to
`/tmp/bcpack-sem-indices/`.

**numpy warning.** System `python3` (Homebrew 3.14 + numpy 2.2.6) corrupts arrays inside functions
(a temporary-elision bug; repro `/tmp/bcpack-sem-indices/elide_test.py`). Every numpy analysis
here runs under `/tmp/bcpack-sem-indices/venv/bin/python -I` (uv venv, Python 3.13, numpy 2.5.5).
The data-extraction scripts (`cache.py`, `quadref.py`, `realstats.py`) only fill arrays element-wise
and run under `python3 -I`.

## Pipeline

```sh
python3 -I tools/sem/cache.py --max-blocks 16384     # decode mip-0 chunk of 3,276 pair textures (probe mdusim, all converge but 6)
python3 -I tools/sem/quadref.py --order gfx10        # + mip-1 pixels under each quadrant (also --order linear, --ring)
PY=/tmp/bcpack-sem-indices/venv/bin/python
$PY -I tools/sem/order.py gfx10 linear                # block order + lane geometry
$PY -I tools/sem/weights.py                           # raw 2-bit packing hypotheses (all fail)
$PY -I tools/sem/ceiling.py --n 300                   # same statistic on plain textures (ceiling)
$PY -I tools/sem/meanw.py                             # per-symbol mean weight, generative hypotheses
$PY -I tools/sem/anchor.py --lane 8                   # per-lane mean-weight table
$PY -I tools/sem/table2.py                            # codebook readout (mean + arrangement), validation
```

Test set: 3,276 mip-pair textures (mip 0 = one BCPack chunk with <= 16,384 blocks, mip 1 plain BC1),
20.3 M blocks. Each lane 8-11 symbol turned out to cover one 2x2 pixel group, which is exactly one
mip-1 pixel. So mip 1 measures each symbol's mean palette weight directly, without endpoints.

## 1. Block order = stored Gfx10 tile order (solved)

`order.py`: mip-1 luminance difference between two quadrant pixels, when the two lane symbols are
equal vs when they differ.

| order | all 4 symbols equal: mip-1 2x2 spread | lanes 8,9 equal: abs(dY) | lanes 8,9 differ |
|---|---|---|---|
| **gfx10** (stored tile order, as in plain chunks) | **0.14** vs 9.29 otherwise | **0.36** | 4.92 |
| linear row-major | 6.37 vs 9.14 | 3.98 | 4.51 |

Per texture, gfx10 beats linear in **3,243 / 3,244** textures. The stream's block i is block i of
the Gfx10-tiled mip, the same layout as zlib chunks. Detile it with `mipscore.detile_blocks` /
`ttpf.untile_order` (harness), or score with `mipscore.score(..., 'stream')`.

## 2. Lanes 8-11 are 2x2 quadrants TL, TR, BL, BR, not rows (solved)

- Equality rates follow 2x2 adjacency. Horizontal neighbours 8-9 / 10-11 are equal in 10.1% of
  blocks, vertical neighbours 8-10 / 9-11 in 9.6%, diagonals 8-11 / 9-10 in 4.2%. Rows would make
  9-10 adjacent, but 9-10 is at the diagonal rate.
- Equal symbols give near-identical mip-1 pixels (table above) for every pair.
- Per-symbol weights learned by ALS on half the blocks (`als.py`), with held-out corr^2 for each
  lane-to-quadrant assignment: identity (8 TL, 9 TR, 10 BL, 11 BR) 0.578. Transposed 0.470,
  reversed 0.451, swapped 0.461, other 0.439, chance 0.456.

## 3. A symbol is NOT a raw packing of four 2-bit indices (ruled out)

A quadrant's mean palette weight depends only on the multiset of its four codes. So any bit order
or pixel order gives the same mean, and only the code-to-weight map matters. `weights.py`: per
block corr^2 of the 4 quadrant mean weights against the 4 mip-1 luminances, on blocks with
mip-1 spread > 8:

- all 24 code-to-weight permutations of 4 x 2-bit fields: 0.415-0.447
- random weight per symbol (control): 0.454
- **ceiling** (`ceiling.py`, the true indices of 300 plain textures, same statistic): **0.901**

The following also fail or are far below the codebook table (`meanw.py`, weighted correlation of
mean weight over 246 symbols): "first pixel raw, others XOR/add/sub relative to it" scores 0.78
LSB-first and 0.32 MSB-first. Plain packing scores 0.38 at best.

## 4. Symbols index a fixed codebook of 2x2 patterns, shared by all four lanes

- **Symbols 0-3 are the uniform quadrants, BC1 code = sym XOR 2** (exact): 0 = all code 2
  (2/3 c0 + 1/3 c1), 1 = all code 3, 2 = all c0, 3 = all c1. Ratio test in blocks holding three
  of the uniform symbols (`anchor.py` docstring; a one-off in the log): (Y0 - Y2)/(Y3 - Y2) has
  median 0.333 (IQR 0.32-0.47, n=226), and (Y1 - Y2)/(Y3 - Y2) has median 0.667 (IQR 0.66-1.00, n=198).
- Every symbol has a stable mean weight that lands on whole twelfths (c0 = 0, c1 = 12). The
  **table is the same for all four lanes** (`anchor.py`, 16x16 tables per lane agree). Using that
  table alone as the symbol-to-weight map gives **corr^2 0.877** (chance 0.456, true-index ceiling
  0.901, `table2.py`). So a fixed per-symbol meaning explains quadrant brightness close to the
  ceiling, with no visible context dependence (lane 0 or neighbours).
- The ordering is a structured, frequency-like table, not arithmetic. Symbols come in mirror sets
  of 2 or 4 with equal mean (orientation variants of one pattern class). Readout with the best
  margins (`table2.py`, arrangement from baseline-corrected upsampled mip 1; codes TL TR / BL BR,
  a = code 2, b = code 3):

| sym | 12*mean | pattern | | sym | 12*mean | pattern |
|---|---|---|---|---|---|---|
| 57 | 1.99 | a a / 0 0 | | 79 | 1.92 | 0 0 / a a |
| 23 | 10.03 | 1 1 / b b | | 22 | 10.05 | b b / 1 1 |
| 110 | 1.94 | 0 a / 0 a | | 103 | 2.00 | a 0 / a 0 |
| 24 | 9.99 | 1 b / 1 b | | 25 | 10.00 | b 1 / b 1 |
| 186 | 0.73 | a 0 / 0 0 | | 194 | 0.73 | 0 a / 0 0 |
| 191 | 0.71 | 0 0 / a 0 | | 196 | 0.68 | (0 0 / 0 a) |
| 56 | 11.27 | 1 1 / 1 b | | 58 | 11.27 | 1 b / 1 1 |
| 59 | 11.26 | 1 1 / b 1 | | 55 | 11.31 | (b 1 / 1 1) |

  Lower confidence (mean solid, arrangement margin < 0.2): 4/5 = a a / b b and b b / a a,
  10/11 = b a / b a and a b / a b, 14/21 = a a / 1 1 and 1 1 / a a, 26/27 = a 1 / a 1 and 1 a / 1 a,
  31/32 = 0 0 / 1 1 and 1 1 / 0 0, 34/36 = 1 0 / 1 0 and 0 1 / 0 1, 53/62 = b b / 0 0 and 0 0 / b b,
  91/101 = 0 b / 0 b and b 0 / b 0. (The automatic readout swaps a few of these. The manual readout
  above follows the per-lane profiles in `arrange.py`.)
- Full readout of 169 symbols (n >= 300): `/tmp/bcpack-shared/sem-indices/codebook.json`, with mean
  weight, tentative pattern and margin. It gives only 94 distinct patterns. Mip 1 resolves the mean
  and the dominant gradient of a 2x2 group, but not second-order arrangement (e.g. "a a / 1 1" vs
  "0 b / 1 1"), so the exact codebook for the remaining ~200 symbols is NOT determined.

## 5. The full 256-entry table (solved; 248/256 stable across disjoint data)

**Table: `tools/sem/quad_table.json`** (derived structure, no game bytes). Symbol -> BC1 codes of
the 2x2 group in order TL, TR, BL, BR (0 = c0, 1 = c1, 2 = 2/3 c0 + 1/3 c1, 3 = 1/3 c0 + 2/3 c1).
It is a bijection onto all 4^4 patterns. Lane 8/9/10/11 = TL/TR/BL/BR group of the block, and BC1
index bit 2*(4y+x) holds pixel (x, y). Every entry carries `stable_halves`, `stable_edgeonly`, a
swap `margin` and its `share` of occurrences.

### Method (`tools/sem/fullres.py`, `tools/sem/fit_rgb.py`)
- Endpoints come from the sem-endpoints back-reference rule (lane 0 = distance unzigzag(v) mod 256,
  lanes 2-7 zigzag residuals mod 2^w). A block counts as known only when its reference chain never
  reaches before block 0, because that preset window is unsolved. That leaves 5.26 M known blocks
  out of 20.3 M, 5.07 M of them four-colour, and 621 textures fully known.
- Per quadrant occurrence and candidate pattern, the cost has three parts:
  - a mip-1 term, |mean palette colour - mip-1 pixel|^2 / 72;
  - a robust edge term, log(1 + d^2/100) over the 8 pixel pairs that cross the quadrant border,
    with neighbours rendered by the current table (pairs into unknown blocks skipped);
  - the same edge cost for the 4 pairs inside the quadrant.
- Costs are summed per symbol into a 256 x 256 matrix, then a min-cost bijection is solved
  (Hungarian, symbols 0-3 fixed) and the loop repeats. Convergence takes 2-5 iterations.

### Anti-overfitting checks (all scores from the harness `mipscore.score(tid, bytes, 'stream')`)
Scoring script: `tools/sem/psnr.py`. It scores only textures whose endpoints are fully known.
Controls on the same textures: the same table with symbols 4-255 shuffled, all indices = code 2
("mid"), and the harness mean-colour guess (`psnr_flat`). Stores are split by index parity in the
sorted store list (`fit_rgb.store_split`).

| table fitted on | scored on (held out) | n | median | p10 | p90 | > flat + 3 dB |
|---|---|---|---|---|---|---|
| even stores, edge + mip-1 terms | odd stores | 319 | **36.63** | 31.51 | 44.68 | 318 / 319 |
| odd stores, edge + mip-1 terms | even stores | 302 | **36.11** | 31.58 | 44.65 | 302 / 302 |
| even stores, **edges only, random init** (no mip-1 term at all) | odd stores | 319 | **35.01** | 29.37 | 43.85 | 317 / 319 |
| controls on the odd stores: shuffled table / mid / flat | | 319 | 28.77 / 26.80 / 19.26 | | | |

Positive control (plain mip 0, harness, 200 textures): median 37.9 dB.
Commands: `python3 -I tools/sem/psnr.py --table /tmp/bcpack-sem-indices/tab_even.npy --n 400 --stores odd`
(and `tab_odd.npy --stores even`, `tab_even_edge.npy --stores odd`). The fits:
`$PY -I tools/sem/fit_rgb.py --stores even --iters 5 --out .../tab_even.npy`, the same with
`--stores odd`, and `--stores even --no-mip1 --init random --iters 8`.

### Stability (`tools/sem/compare.py`)
- **Even-store fit vs odd-store fit: 248 / 256 entries identical, 99.76% of all quadrant
  occurrences.** All 192 most frequent symbols agree. The 8 that differ are rare (frequency rank
  192+, 0.24% of occurrences). Each is a swap inside a symmetric pair: 229 <-> 235 (`10a1`/`a110`),
  230 <-> 231 (`1a01`/`011a`), 247 <-> 248 (`0110`/`1001`), 254 <-> 255 (`100b`/`0b10`), codes written
  TL TR BL BR. Both halves give these large margins with opposite answers, so treat them as
  unresolved.
- Edge-only fit from a random start vs the mip-1 fit: 244 / 256 identical (all top 32; 96.3% of
  occurrences). That run still oscillated on 13 entries.
- Final table = all stores, initialised from the even fit: 252/256 equal to each half fit, with
  swap margins of 0.55-3.2 cost units per occurrence on every frequent entry.

### Structure (`tools/sem/structure.py`)
- No arithmetic rule. The order is a **frequency-sorted trained codebook**: Spearman(symbol, -its
  frequency in BCPack data) = 0.79, and Spearman(symbol, -frequency of its pattern in plain BC1
  textures of the same game) = 0.78. Uniform patterns come first, then interpolant-only patterns
  (codes 2/3), then patterns mixing an endpoint with its neighbouring interpolant, with 0/1 mixes last.
- Orientation variants of a class sit in runs of 2 or 4 (e.g. 55/56/58/59 = three c1 + one code 3,
  one per corner; 186/191/194/196 = three c0 + one code 2). The pairs (2k, 2k+1) are NOT consistently
  related by one symmetry (82/128 pairs match none), so the order inside a class can't be predicted
  by a rule. This parallels the endpoint pre-start window, also a fixed trained table.
- The part-1 readouts from mip 1 alone (section 4) all agree with the final table (57 `aa/00`,
  79 `00/aa`, 22/23, 24/25, 103/110, 186/191/194/196, 55/56/58/59, 4/5, 10/11, 14/21, 26/27, 31/32,
  34/36, 53/62, 91/101).

### 3-colour blocks (c0 <= c1 as decoded; `tools/sem/three.py --table ... --minsep 40`)
On known 3-colour blocks with |c0 - c1| >= 40 (96,518 blocks), the quadrant mean colour error vs
mip 1 is: standard BC1 3-colour semantics (code 2 = midpoint, code 3 = black) **5.58**, code 3 read as
the midpoint 5.80, code 3 = c1 5.89, palette forced to 4-colour 6.35. The same table on 4-colour
blocks gives 3.37. So the standard BC1 decode is the best of these, and the decoder need not treat
these blocks specially. They fit worse than 4-colour blocks, though, and a separate 3-colour table
was NOT fitted. Open.

### Rendered checks (not committed)
Three held-out textures (odd stores, table fitted on even stores) are written by `tools/sem/render.py`
to `/tmp/bcpack-sem-indices/*_mip0_decoded.png`, next to `*_mip1_plain_x2.png`. They score 34.4,
37.6 and 32.7 dB (flat 8.6, 9.2, 12.3).

## What is left
1. The 8 symmetric-pair entries (0.24% of occurrences), and fitting the table on all blocks once the
   endpoint pre-start window is solved: 74% of blocks are still unused.
2. 3-colour blocks: check whether a per-mode table or code remap closes the 5.6 vs 3.4 gap.
3. The endpoint pre-start window (sem-endpoints) limits full-chunk decodes: only 621 / 3,276 pair
   textures are fully known today.
