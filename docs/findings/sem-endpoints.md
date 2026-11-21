# Findings: sem-endpoints (BC1 endpoint reconstruction from BCPack lanes 0-7)

Agent `sem-endpoints`, 2026-11-21. Code: `tools/sem/` (`common.py` decode + mip-1 targets,
`endpoints.py` hypotheses + scorer, `run_hyp.py` hypothesis grid, `score_rule.py` evidence table).
Lane symbols come from the probe agent's simulator (`mdusim.py`, copied), with exact convergence
required (all 12 lanes done, legal end state, consumed == stripped data length).

**Python warning:** system python3 3.14 + numpy 2.2.6 corrupts arrays (temporary-elision bug, found by
sem-indices). Every number here was produced with `/tmp/bcpack-sem-indices/venv/bin/python -I`
(py3.13, numpy 2.5.5).

## 1. The rule (status: verified, except the initial window)

For block i (stream order = stored Gfx10 tile order of the chunk's virtual bytes):

```
d = back_distance(lane0[i])          # even v=2k -> k ; odd v=2k+1 -> 255-k ; i.e. unzigzag(v) mod 256
P = (0,0,0,0,0,0)        if d == 0                     # absolute
    fields(block i-d)    if 1 <= d <= i
    WINDOW[i-d]          if d > i                      # pre-start reference: OPEN (see section 3)
for k, w in enumerate([5,6,5,5,6,5]):                  # lanes 2..7 = R0 G0 B0 R1 G1 B1
    s = lane[2+k][i]
    r = -(s+1)//2 if s & 1 else s//2                   # zigzag: 0,-1,+1,-2,+2,...
    field[k] = (P[k] + r) mod 2^w
c0 = R0<<11 | G0<<5 | B0 ;  c1 = R1<<11 | G1<<5 | B1   # BC1 colour words as decoded, no swap
```

* Lane 0 is the back-reference distance into a 256-block window. Its values are mostly small and even
  because the encoder usually refers to a recent block.
* No component mixes with another: endpoint 1 is predicted from the reference's endpoint 1, not from
  endpoint 0. There is no swap flag (an odd lane 0 is a far reference, not a swap; tested and ruled out).
* Lane 1 (constant 0 everywhere) contributes nothing here.
* Residuals wrap: the encoder takes the shortest modular step (e.g. G 20 -> 54 is coded -30).

## 2. Evidence

Metric (index-free, so it works before the index lanes are solved): every 4-colour BC1 palette entry
lies on the segment c0-c1, so the block mean (about the 2x2 mean of the plain mip 1 at the block's
position) must lie on or near that segment. Score = median distance (RGB 0-255 units) over the blocks of
a texture. Positive control: true plain BC1 mip 0 of 40 control textures gives **0.8** (the same
endpoints against shuffled targets give 8.3): `python -I tools/sem/run_hyp.py --control --n 40`.

### 2a. How lane 0 was found (`/tmp`-only exploratory script, logic in `run_hyp.py`/`endpoints.py`)
For zero-residual blocks of 40 pair textures, the earlier block d whose mip-1 colour best matches
the block, for each lane-0 value v. Even v: best d = v/2 for every v from 2 to 148 (e.g. v=52 gives d=26, dist
6.2 vs 37.3 for d=1). Odd v: best d = 256-(v+1)/2 (v=1: 255, v=3: 254, ..., v=149: 181).

### 2b. Hypothesis grid (`python -I tools/sem/run_hyp.py --n 24`; median over 24 textures)
Every combination of residual map {zigzag A, zigzag B, two's complement, raw} x predictor {none,
previous block, left, up, (left+up)/2, LOCO-I median} x endpoint-1 {own history, from endpoint 0}
scores 110-245 (chance level is about 100-120). Best of those: zigzag A + up = 110.1. All ruled out.
With the back-reference predictor (`tools/sem/score_rule.py`), zigzag A is the only residual map that works
(zigzag B / two's complement: medians 200 / 121 on the same textures).

### 2c. Back-reference rule (`python -I tools/sem/score_rule.py --n 40 --side 256`)
| textures | n | rule (window = 0) | rule + solved root | left-neighbour baseline | shuffled control |
|---|---|---|---|---|---|
| no pre-start reference | 8 | median **0.4**, 8/8 < 4 | | 44-222 | 4-60 |
| exactly one pre-start reference | 11 | 24.2 | median **2.4**, 11/11 < 7 | | |
| 2 or more pre-start references | 21 | mostly fail (chains rooted in unknown window values) | not solved | | |

Textures with 0 pre-start references decode to control-level quality. The failures are entirely
explained by references before block 0 (`d > i`). The rest of a chain only adds residuals mod 2^w, so a
wrong window value shifts every block in its chain by a constant vector. Solving that one vector per texture
(coordinate search) brings all 11 one-reference textures to 0-6.5.

### 2d. End-to-end with the harness mip scorer (`python -I tools/sem/mip_endpoints.py --n 40`)
BC1 blocks = decoded c0/c1 + a FIXED index checkerboard (0/1 in every 2x2 quadrant, so each mip-1
pixel is predicted by the endpoint midpoint; no information from the reference is used), scored by
`mipscore.score(tid, bytes, 'gfx10')`.
| textures | n | rule | psnr_flat | left-neighbour endpoints | rule, blocks shuffled |
|---|---|---|---|---|---|
| no pre-start reference | 8 | median **26.3 dB**; beats flat by more than 3 dB in 8/8 | 18.6 | 9.8 | 15.8 |
| all | 40 | 10.3 (pre-start chains broken) | 16.8 | 8.8 | 9.1 |
The midpoint caps the score (real indices are unknown here); correct full mip 0 scores about 38 dB.

## 3. The initial window: a FIXED preset table (verified for slot 242; contents only partly fitted)

A reference with d > i reads ring slot `(i - d) mod 256` of a window whose initial contents are not in
the chunk (all bytes are accounted for by the 12 lanes). What was tested:

| hypothesis for the initial window | result | script |
|---|---|---|
| all zero | wrong: 21/40 textures with 2+ pre-start references fail, 11 single-reference ones need non-zero roots | `score_rule.py` |
| tail of the previous chunk (zlib, holds the same texture's mip 1) | partial only, median 39 -> 28, not exact | exploratory, same decoder with `history=` |
| tail of the previous BC1 BCPack chunk in store order (decoded recursively, 1 or 3 deep) | no: median 110 -> 103 / 93, exact for 0 more textures | exploratory, same decoder with `history=` |
| the chunk's own last 256 blocks (wrap, iterated to a fixed point) | no: 110 -> 75, exact for 0 more | exploratory |
| **fixed table, same value per slot for every chunk** | **yes for slot 242**: one joint value fits 64/64 textures (151,252 blocks) within 2 units of each texture's own optimum; joint fit < 8 for 59/64; a zero root < 8 for 0/64 | `window_fit.py` |

Usage is very uneven, so a few slots cover most chunks. In 599 chunks: slot 254 (415 refs), 193 (390), 128 (293),
242 (285), 89 (112), 99, 232, 121, 255, ... The first block's lane 0 is 0 (no reference) in 165/599.

Fitted slot values (R0 G0 B0 R1 G1 B1, from `/tmp/bcpack-shared/window_fit.json`, joint over all
textures that root a chain there). Confidence comes from the joint cost; the coordinate search can get stuck
in local minima, which is why the multi-start version matters:

| slot | value | mean capped dist (joint) | textures | status |
|---|---|---|---|---|
| 242 | 16 29 31 / 16 34 31 | 2.4 | 64 | solid (multi-start) |
| 121 | 31 61 30 / 31 62 29 | 2.6 | 32 | solid (single start) |
| 255 | 0 0 1 / 0 0 0 | 1.6 | 24 | solid (single start) |
| 82  | 26 11 31 / 8 9 31 | 4.4 | 12 | likely |
| 254, 193, 128, 232, 99, 89, 118, 91 | see json | 9-33 single start | 12-63 | not converged yet (see `wf2.txt` multi-start run) |

Entries look like a trained codebook of common endpoint pairs (many have B = 31, typical of normal maps).
They do not look like a formula of the slot index. An exact search of the store's decompressed plain data for
the slot-121/254/82 colour pairs found nothing systematic.

## 4. Notes for other agents
* Indices (sem-indices): references copy ENDPOINTS only. Index lanes are not residuals against the
  referenced block: blocks that reference an identical-looking block carry the same raw index symbols, not zeros.
  The endpoint order is as decoded (c0 may be <= c1, i.e. 3-colour mode is possible). With endpoints now known
  for any chunk whose chains avoid unknown slots, per-quadrant index hypotheses can be scored exactly.
* Block order: stream order = stored Gfx10 order (consistent with sem-indices). The 2D neighbours in stored
  order were tried as predictors and are wrong; the predictor is purely the lane-0 back-reference.

## 5. Next steps
1. Fit all 255 window slots: run `window_fit.py` with `--slots` over every used slot, multi-start, on many
   more textures (all 4,809 pairs, plus control textures are not needed). Better: once indices are solved, fit with exact block
   means (a linear constraint) instead of segment distance. Ship the table as a decoder constant.
2. Re-check the rule on 1536-block multi-stream chunks (lanes 0/2-7 carry 96-257 symbols there) and on
   >65,536-block chunks. The window probably resets per sub-stream (not tested; no pair texture uses them).
