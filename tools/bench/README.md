# BCPack test bench

One shared bench so every decoder claim is checked the same way. Python 3 stdlib only; every
script runs under `python3 -I`. Game data is read in place from `$BCPACK_GAME_ROOT`, or from the path in
`/tmp/bcpack-shared/game_root.txt` (kept out of git) and nothing derived from it is ever written into git:
indexes go to `/tmp/bcpack-shared/`, scratch to `/tmp/bcpack-harness/` (`BCPACK_SCRATCH`).

| File | What it does |
|---|---|
| `ttpf.py` | RDS5 store reader: chunk table, ETPF texture index, `extract(chunk_id)`, Gfx10 mip chain + detiling (port of paryzius ttpf-viewer), BC1 decode |
| `lanes.py` | 6-byte main-lane record parser (field layout from `mdu_minirans.v`), legal end states, constant-lane tables (`const_match`) |
| `corpus.py` | Writes `bc1_index.jsonl`, `bc7_index.jsonl`, `corpus_summary.json` |
| `oracle.py` | Exact-convergence oracle (library + CLI), bcpack-rs adapter, optional `RansLaneDecoder` helper with a pluggable model |
| `mipscore.py` | Mip-consistency scorer: `mip_pairs.jsonl`, `score()`, positive/negative controls |

Ids: a chunk is `"<store relpath>:<chunk index>"`, a texture is `"<store relpath>:t<ETPF index>"`.

## Commands

```sh
cd /Users/ahmet/Code/bcpack-wt/harness      # or your own worktree
cargo build --release                        # target/release/bcpack (needed for the adapter)

# 1. corpus index (about 5 s on 18 cores; builds the constant-lane tables on first run)
python3 -I tools/bench/corpus.py
python3 -I tools/bench/corpus.py --summary          # histograms
python3 -I tools/bench/corpus.py --easiest 30       # fewest non-constant lanes, smallest

# 2. oracle
python3 -I tools/bench/oracle.py --selftest                          # synthetic +/- control
python3 -I tools/bench/oracle.py --adapter bcpack-rs --sample 500 --seed 1 --json /tmp/bcpack-harness/o.json
python3 -I tools/bench/oracle.py --adapter bcpack-rs --sample 500 --seed 1 --control shuffle
python3 -I tools/bench/oracle.py --cmd "python3 -I mydec.py {path} {blocks}" --easiest 30
PYTHONPATH=/path/to/mod python3 tools/bench/oracle.py --python mymod:decode --ids ids.txt
#   --expect 'default=auto'       const lanes: their recovered N; others: N (= lane1_N or blocks)
#   --expect 'default=N,2-7=2N'   per-lane symbol counts; 'none' disables the count check

# 3. mip consistency
python3 -I tools/bench/mipscore.py index                 # -> /tmp/bcpack-shared/mip_pairs.jsonl
python3 -I tools/bench/mipscore.py controls --n 200      # positive + negative controls
python3 -I tools/bench/mipscore.py score TEXTURE_ID decoded.bin --order stream
python3 -I tools/bench/mipscore.py map TEXTURE_ID        # chunk ids + block ranges making up mip 0
```

Assembling a whole mip 0 from decoded chunks (semantics agents):

```python
m = mipscore.mip0_chunk_map(tid)        # [{chunk_id, chunk_bytes, mip0_bytes, blocks}, ...] in order
gfx10, linear, bw, bh = mipscore.assemble_mip0(tid, {chunk_id: decoded_chunk_bytes})
mipscore.score(tid, gfx10, "gfx10")      # == score(tid, decoded_chunk_bytes, "stream")
mipscore.write_ppm("/tmp/bcpack-<you>/x.ppm", linear, bw, bh)   # eyeball; never commit
```

Library use:

```python
import sys; sys.path.insert(0, "tools/bench")
import ttpf, oracle, mipscore
raw = ttpf.extract("data/ttdb/maps/ttdb_maps_2_7.ttpf:47982")   # stored bytes, pad stripped
rec = oracle.index()["data/ttdb/maps/ttdb_maps_2_7.ttpf:47982"]  # its index line
res, agg = oracle.evaluate(ids, ("python", "mymod:decode"))
mipscore.score("data/ttdb/maps/ttdb_maps_2_7.ttpf:t17290", decoded_chunk_bytes, "stream")
```

## Candidate decoder contract (oracle)

Python: `decode(chunk: bytes, blocks: int, meta: dict) -> dict`; command: prints the same JSON
on stdout, non-zero exit on failure. `{path}` is a temp file holding the chunk (pad stripped).

```json
{"lanes": [{"lane": 0, "bytes_consumed": 1234, "final_state": 16777216, "n_symbols": 4096,
            "budget": 1234, "fragments": [{"mode": 50, "len": 1231}], "symbols": [0, 3]}]}
```

`budget` (header + payload bytes the lane's own fragment headers declare) and `fragments` are
optional but strongly recommended; without them only single-fragment main lanes can be
budget-checked. Lanes 0-7 are the 48-byte header lanes; number index/extension lanes 8+.

A lane passes when bytes consumed == budget, the final state is L, L>>8 or L>>16, and the
symbol count matches the expectation. The chunk also needs sum(bytes_consumed) == data_len.
Lanes are reported as trivial (proven constant, expected count <= 1, or payload <= 4 bytes) or
varying; only varying lanes count, and the per-payload-size breakdown shows how much
information each pass carries. Always compare against `--control shuffle`.

## mipscore block orders

* `stream`: the decoded virtual bytes of the BCPack chunk(s) covering mip 0, in chunk order.
  For all 4,809 pairs mip 0 is exactly one chunk, so this is simply "the decoded chunk".
* `gfx10`: mip 0 as stored (tiled), `mip0_len` bytes.
* `linear`: row-major BC1 blocks.

Compare `psnr` with `psnr_flat` (mean-colour guess) for the same texture: many textures are flat
normal maps where a constant guess already scores 30+ dB.
