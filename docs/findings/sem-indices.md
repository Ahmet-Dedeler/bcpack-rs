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

## What is left, and the best leads

1. **Exact codebook (256 entries).** Mip 1 cannot separate arrangements with equal mean and equal
   gradient. Leads:
   (a) Once `sem-endpoints` decodes endpoints, fit each symbol's pattern by mip-consistency PSNR
   (greedy per symbol, starting from the most frequent). Endpoints fix the sign/scale per block,
   so 3-colour blocks can be handled too.
   (b) Inter-quadrant smoothness at mip-0 resolution: adjacent pixels across quadrant borders
   should match. Score candidate patterns per symbol by the border agreement with neighbours whose
   patterns are already known (the uniform symbols 0-3 alone are about 17% of quadrants).
   (c) Look for the table in Xbox GDK / DirectStorage / MDU firmware or driver binaries. A
   256-entry table of 2x2 patterns starting with the uniform patterns in code order 2,3,0,1 is a
   recognisable signature: as bytes, `aa ff 00 55`.
2. **3-colour blocks** (c0 <= c1, about 15% of plain BC1 blocks). It is not tested whether the same
   codebook applies there (code 2 = 1/2, code 3 = black).
3. **Lane 0** correlates with which secondary symbols appear. l0 in {2, 14, 16, 18, 30, 32} favours
   the vertical splits 23/79, l0 in {4, 8, 12, 52} the horizontal splits 24/110. But MI(lane 0,
   lane 8) is only 0.038 bits, and the codebook needs no lane-0 context to reach 0.877 of the 0.901
   ceiling. Hand that to `sem-endpoints`.
