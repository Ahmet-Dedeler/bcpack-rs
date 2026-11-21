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

## 3. Open: the initial window (pre-start references)
* A window of zeros is wrong: the solved roots are non-zero.
* The previous chunk's decoded tail (zlib, holding the same texture's mip 1) as the window: partial
  improvement (median 39 -> 28 over 40 textures) but not exact. Probably it only helps because the colours are similar.
* Pre-start references concentrate on a few ring slots (`(i-d) mod 256`) across unrelated chunks:
  254 (415 refs), 193 (390), 128 (293), 242 (285), 89 (112) in 599 chunks. The first block's lane 0 is
  0 in 165/599 chunks, then 241, 255, 48, 28, 126 (about 30 each). This points to a FIXED preset table that
  the encoder searches, rather than leftover state.
* Solved root values per (texture, slot) are in the `score_rule.py` output. Same-slot estimates roughly agree
  (slot 121: R0=31, R1=31 in 5/5 textures, B 28-31, G0 49-60, G1 near 0 after wrap).
