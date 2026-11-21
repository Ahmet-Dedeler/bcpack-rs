"""Exact-convergence oracle for BCPack candidate decoders (layers 1-2).

A candidate decoder is either

* a Python callable ``f(chunk: bytes, blocks: int, meta: dict) -> dict`` or
* an external command template, e.g. ``"target/release/bcpack trace {path} {blocks}"``,
  that prints one JSON object on stdout (non-zero exit = failure),

returning a trace::

    {"lanes": [{"lane": 0, "bytes_consumed": 123, "final_state": 16777216,
                "n_symbols": 4096,
                "budget": 123,                     # optional: header+payload bytes the
                                                   # lane's own fragment headers declare
                "fragments": [{"mode": 0x32, "len": 120}, ...],   # optional
                "symbols": [...]},                 # optional
               ...]}

Lanes 0-7 are the main lanes (records in the 48-byte header); any further lanes
(index / extension lanes) are numbered 8, 9, ... in the candidate's own order.
The oracle is model-agnostic: it never decodes anything itself.  Plug in any
probability model (e.g. a port of microsoft_silicon-rans/mdu_minirans.v tables)
on the candidate side; ``RansLaneDecoder`` below is an optional helper that runs
the reference fragment FSM with a pluggable model.

Checks per lane (all must hold for a lane to PASS):
  budget  bytes_consumed == declared budget (candidate's ``budget`` if given, else
          header + fragment-0 budget for main lanes); for main lanes the
          candidate's fragment 0 must also match the oracle's own header parse
  state   final_state in {L, L>>8, L>>16}, L = 2^24 (mdu_minirans.v legal_end_state)
  count   n_symbols == expected (see --expect), when an expectation exists
Chunk-level (candidate independent): sum(bytes_consumed) == data_len (stored
length minus pad): every byte must belong to exactly one lane.

Lanes are split into TRIVIAL (proven constant in the corpus index, expected
count <= 1, or payload <= 4 bytes) and VARYING (everything else).  Only varying
multi-symbol lanes count as evidence.

CHANCE RATE.  With a fixed symbol count and a decoder that stops at its budget,
a wrong model leaves the 32-bit state roughly log-uniform on [2^24, 2^32), so
P(state legal) is ~1/(2^24 * ln 256) ~ 1e-8 per lane; a free symbol count
("decode until x == L") raises this to O(1) and proves nothing.  Budget-only
matches are tautological for decoders that read until the budget is spent.
``--control shuffle`` reruns the candidate on chunks whose bytes after the
48-byte header are shuffled: any varying-lane pass there is an oracle/candidate
bug or chance, and the real score must be compared against it.

CLI:
  python3 -I tools/bench/oracle.py --adapter bcpack-rs --sample 500 --seed 1
  python3 -I tools/bench/oracle.py --cmd "mydec {path} {blocks}" --easiest 30
  python3 -I tools/bench/oracle.py --python mymod:decode --ids ids.txt --json out.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lanes as LN  # noqa: E402
import ttpf  # noqa: E402

SHARED = "/tmp/bcpack-shared"
SCRATCH = os.environ.get("BCPACK_SCRATCH", "/tmp/bcpack-harness")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Index access
# ---------------------------------------------------------------------------
_INDEX: Optional[Dict[str, dict]] = None


def index(path: str = os.path.join(SHARED, "bc1_index.jsonl")) -> Dict[str, dict]:
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                _INDEX[r["id"]] = r
    return _INDEX


# ---------------------------------------------------------------------------
# Expected symbol counts
# ---------------------------------------------------------------------------
def parse_expect(spec: str) -> Callable[[dict, int], Optional[int]]:
    """``spec`` = comma list of ``lanes=formula``; lanes: 'default', '3', '2-7', '8+';
    formula: 'N' (stream blocks), '2N', '128', 'none', or 'auto' (proven constant
    lanes use their recovered N, others N).  N = corpus lane1_N if known, else blocks.
    Default spec: 'default=auto'."""
    rules = []
    for part in spec.split(","):
        k, _, v = part.partition("=")
        k, v = k.strip(), v.strip()
        if k == "default":
            sel = lambda i: True
        elif k.endswith("+"):
            lo = int(k[:-1]); sel = (lambda lo: lambda i: i >= lo)(lo)
        elif "-" in k:
            a, b = map(int, k.split("-")); sel = (lambda a, b: lambda i: a <= i <= b)(a, b)
        else:
            sel = (lambda j: lambda i: i == j)(int(k))
        rules.append((sel, v))

    def expect(rec: dict, lane: int) -> Optional[int]:
        N = rec.get("lane1_N") or rec["blocks"]
        v = "auto"
        for sel, f in reversed(rules):
            if sel(lane):
                v = f
                break
        if v == "none":
            return None
        if v == "auto":
            if lane < 8 and rec["lanes"][lane]["cls"] == "const":
                return rec["lanes"][lane]["const_N"]
            return N
        if v.endswith("N"):
            return int(v[:-1] or 1) * N
        return int(v)
    return expect


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_chunk(rec: dict, trace: Optional[dict], expect, error: Optional[str] = None) -> dict:
    out = {"id": rec["id"], "blocks": rec["blocks"], "data_len": rec["data_len"],
           "error": error, "lanes": []}
    if trace is None:
        out["chunk_pass"] = False
        out["sum_ok"] = False
        return out
    lanes = trace.get("lanes", [])
    total = sum(int(l.get("bytes_consumed", 0)) for l in lanes)
    out["bytes_total"] = total
    out["sum_ok"] = total == rec["data_len"]
    seen = set()
    for l in lanes:
        i = int(l["lane"])
        seen.add(i)
        exp_n = expect(rec, i)
        hdr = rec["lanes"][i] if i < 8 else None
        frags = l.get("fragments") or []
        budget = l.get("budget")
        frag0_ok = None
        if hdr is not None:
            if frags:
                f0 = frags[0]
                frag0_ok = (int(f0.get("mode", -1)) == hdr["mode"] and int(f0.get("len", -1)) == hdr["budget"])
            if budget is None and len(frags) <= 1:
                budget = hdr["hdr_bytes"] + hdr["budget"]
        budget_ok = budget is not None and int(l["bytes_consumed"]) == int(budget) and frag0_ok is not False
        state_ok = LN.is_legal_end_state(int(l.get("final_state", -1)))
        count_ok = None if exp_n is None else int(l.get("n_symbols", -1)) == exp_n
        payload = (hdr["budget"] if (hdr and len(frags) <= 1) else
                   sum(int(f.get("len", 0)) for f in frags) if frags else None)
        trivial = bool((hdr is not None and hdr["cls"] == "const")
                       or (exp_n is not None and exp_n <= 1)
                       or (payload is not None and payload <= 4))
        ok = budget_ok and state_ok and count_ok is not False
        out["lanes"].append({"lane": i, "trivial": trivial, "budget_ok": budget_ok,
                             "frag0_ok": frag0_ok, "state_ok": state_ok, "count_ok": count_ok,
                             "expected_n": exp_n, "n_symbols": l.get("n_symbols"),
                             "bytes_consumed": l.get("bytes_consumed"), "budget": budget,
                             "final_state": l.get("final_state"), "payload": payload, "pass": ok})
    for i in range(8):  # a main lane the candidate did not report fails
        if i not in seen:
            out["lanes"].append({"lane": i, "trivial": rec["lanes"][i]["cls"] == "const",
                                 "pass": False, "missing": True, "budget_ok": False,
                                 "state_ok": False, "count_ok": False})
    out["chunk_pass"] = out["sum_ok"] and all(x["pass"] for x in out["lanes"])
    return out


def aggregate(results: List[dict]) -> dict:
    agg = {"chunks": len(results),
           "chunks_error": sum(1 for r in results if r["error"]),
           "chunks_sum_ok": sum(1 for r in results if r.get("sum_ok")),
           "chunks_all_lanes_pass": sum(1 for r in results if r.get("chunk_pass"))}
    for kind in ("trivial", "varying"):
        ls = [l for r in results for l in r["lanes"] if l["trivial"] == (kind == "trivial")]
        agg[kind] = {"lanes": len(ls),
                     "budget_ok": sum(1 for l in ls if l["budget_ok"]),
                     "state_ok": sum(1 for l in ls if l["state_ok"]),
                     "count_ok": sum(1 for l in ls if l.get("count_ok")),
                     "pass": sum(1 for l in ls if l["pass"])}
    v = [l for r in results for l in r["lanes"] if not l["trivial"]]
    agg["varying_pass_by_lane"] = {str(k): f"{sum(1 for l in v if l['lane'] == k and l['pass'])}/{sum(1 for l in v if l['lane'] == k)}"
                                   for k in sorted({l["lane"] for l in v})}
    def bucket(p):
        return "unknown" if p is None else "5-8" if p <= 8 else "9-64" if p <= 64 else "65-1024" if p <= 1024 else ">1024"
    agg["varying_pass_by_payload_bytes"] = {
        b: f"{sum(1 for l in v if bucket(l.get('payload')) == b and l['pass'])}/{sum(1 for l in v if bucket(l.get('payload')) == b)}"
        for b in ("5-8", "9-64", "65-1024", ">1024", "unknown")}
    agg["headline"] = (f"varying multi-symbol lanes converged exactly: {agg['varying']['pass']}/"
                       f"{agg['varying']['lanes']}; whole chunks: {agg['chunks_all_lanes_pass']}/{agg['chunks']}")
    return agg


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
def shuffled(chunk: bytes, seed: str) -> bytes:
    rng = random.Random(seed)
    body = bytearray(chunk[48:])
    rng.shuffle(body)
    return chunk[:48] + bytes(body)


def run_cmd(template: str, chunk: bytes, blocks: int, rec: dict, timeout: float = 120) -> dict:
    os.makedirs(SCRATCH, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="chunk_", suffix=".bin", dir=SCRATCH)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(chunk)
        cmd = template.format(path=path, blocks=blocks, id=rec["id"], data_len=len(chunk))
        p = subprocess.run(shlex.split(cmd), capture_output=True, timeout=timeout, cwd=REPO)
        if p.returncode != 0:
            raise RuntimeError(f"exit {p.returncode}: {p.stderr.decode(errors='replace')[-300:]}")
        return json.loads(p.stdout)
    finally:
        os.unlink(path)


ADAPTERS = {"bcpack-rs": os.path.join(REPO, "target", "release", "bcpack") + " trace {path} {blocks}"}

_CAND = None


def _init(cand_spec, expect_spec, control, idx_path):
    global _CAND, _EXPECT, _CONTROL
    index(idx_path)
    _EXPECT = parse_expect(expect_spec)
    _CONTROL = control
    kind, val = cand_spec
    if kind == "python":
        mod, _, fn = val.partition(":")
        _CAND = ("python", getattr(importlib.import_module(mod), fn))
    else:
        _CAND = ("cmd", val)


def _work(cid: str) -> dict:
    rec = index()[cid]
    chunk = ttpf.extract(cid)
    if _CONTROL == "shuffle":
        chunk = shuffled(chunk, cid)
    try:
        if _CAND[0] == "python":
            trace = _CAND[1](chunk, rec["blocks"], rec)
        else:
            trace = run_cmd(_CAND[1], chunk, rec["blocks"], rec)
        return score_chunk(rec, trace, _EXPECT)
    except Exception as e:  # candidate raised -> every lane fails
        return score_chunk(rec, None, _EXPECT, error=f"{type(e).__name__}: {e}"[:400])


def evaluate(ids: List[str], cand_spec, expect_spec="default=auto", control=None, jobs=None,
             idx_path=os.path.join(SHARED, "bc1_index.jsonl")):
    """Library entry point. cand_spec = ("cmd", template) or ("python", "module:function")."""
    with Pool(jobs or os.cpu_count(), initializer=_init,
              initargs=(cand_spec, expect_spec, control, idx_path)) as pool:
        results = pool.map(_work, ids, chunksize=1)
    return results, aggregate(results)


def select(args) -> List[str]:
    idx = index(args.index)
    if args.ids:
        with open(args.ids) as fh:
            return [l.split()[0] for l in fh if l.strip() and not l.startswith("#")]
    rows = list(idx.values())
    if args.max_blocks:
        rows = [r for r in rows if r["blocks"] <= args.max_blocks]
    if args.easiest:
        rows.sort(key=lambda r: (r["n_nonconst"], r["n_var"], r["data_len"]))
        return [r["id"] for r in rows[:args.easiest]]
    rng = random.Random(args.seed)
    return [r["id"] for r in rng.sample(rows, min(args.sample, len(rows)))]


# ---------------------------------------------------------------------------
# Optional helper: reference single-lane fragment FSM with a pluggable model
# ---------------------------------------------------------------------------
class ConstantModel:
    """SixNG constant-lane model: symbol 0 at start 1 with freq M-K, escapes freq 1."""

    def __init__(self, K: int):
        self.f = LN.RANS_M - K

    def decode(self, cf: int):
        """cf (low 16 bits of state) -> (symbol, freq, start)."""
        if 1 <= cf < 1 + self.f:
            return 0, self.f, 1
        return (1 + cf - (1 + self.f)) if cf else 0, 1, cf


class RansLaneDecoder:
    """Decode one lane given its own bytes in read order (header + payload of each
    fragment), following mdu_minirans.v: state starts at 0 and is fed while < L;
    a symbol is emitted while state > L and symbols remain; fragment ends when the
    payload is used up; legal end states {L, L>>8, L>>16}.  ``model_for(hdr0, eab)``
    returns an object with ``decode(cf) -> (sym, freq, start)``.  Pure helper:
    the read SCHEDULE across lanes is the candidate's business."""

    def __init__(self, model_for, alphabet_bits: int):
        self.model_for = model_for
        self.alphabet_bits = alphabet_bits

    def decode(self, data: bytes, n_symbols: int) -> dict:
        pos, x, syms, frags = 0, 0, [], []
        while len(syms) < n_symbols and pos < len(data):
            h0 = data[pos]; pos += 1
            z, hdr3, flush = h0 >> 6, (h0 >> 5) & 1, (h0 >> 4) & 1
            if hdr3:
                n = (data[pos] | (data[pos + 1] << 8)) + 257; pos += 2
            else:
                n = data[pos] + 1; pos += 1
            eab = self.alphabet_bits - z
            eab = eab if 0 <= eab < 10 else 9
            model = self.model_for(h0, eab)
            frags.append({"mode": h0, "len": n})
            end = pos + n
            while True:
                while x < LN.RANS_L and pos < end:
                    x = ((x << 8) | data[pos]) & 0xFFFFFFFF; pos += 1
                if not (x > LN.RANS_L and len(syms) < n_symbols):
                    break
                s, f, c = model.decode(x & 0xFFFF)
                syms.append(s)
                x = f * (x >> 16) + (x & 0xFFFF) - c
                if pos >= end and x <= LN.RANS_L:
                    break
            if flush and len(syms) < n_symbols:
                x = 0
        return {"bytes_consumed": pos, "final_state": x, "n_symbols": len(syms),
                "fragments": frags, "symbols": syms, "budget": sum(3 if (f["mode"] >> 5) & 1 else 2 for f in frags) + sum(f["len"] for f in frags)}


def selftest() -> bool:
    """Synthetic positive/negative control for the oracle itself (no game data):
    8 constant lanes (K=32, N=1000, budget 4) in a 48-byte interleaved header,
    decoded with RansLaneDecoder -> every lane must pass; a candidate reporting a
    wrong symbol count or a corrupted state byte must fail."""
    K, N = 32, 1000
    st, rn = LN.const_table(K)
    assert rn[N] == 0
    rec6 = bytes([0xde, 3]) + st[N].to_bytes(4, "big")
    chunk = bytes(rec6[r] for r in range(6) for _ in range(8))
    lanes = []
    for i, r in enumerate(LN.main_records(chunk)):
        p = LN.parse_record(r); p["lane"] = i
        m = LN.const_match(p["budget"], p["state"])
        p.update(cls="const", const_K=m[0][0], const_N=m[0][1]) if m else p.update(cls="var")
        lanes.append(p)
    rec = {"id": "synthetic:0", "blocks": N, "data_len": 48, "lanes": lanes, "lane1_N": N}
    dec = RansLaneDecoder(lambda h, e: ConstantModel(K), 8)
    good = {"lanes": [dict(dec.decode(r, N), lane=i) for i, r in enumerate(LN.main_records(chunk))]}
    exp = parse_expect("default=auto")
    ok1 = score_chunk(rec, good, exp)["chunk_pass"]
    bad = json.loads(json.dumps(good)); bad["lanes"][3]["n_symbols"] += 1
    ok2 = not score_chunk(rec, bad, exp)["chunk_pass"]
    r3 = bytearray(LN.main_records(chunk)[5]); r3[5] ^= 1
    bad2 = json.loads(json.dumps(good)); bad2["lanes"][5] = dict(dec.decode(bytes(r3), N), lane=5)
    ok3 = not score_chunk(rec, bad2, exp)["chunk_pass"]
    print(json.dumps({"synthetic_all_pass": ok1, "wrong_count_fails": ok2, "corrupt_state_fails": ok3}))
    return ok1 and ok2 and ok3


def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--adapter", choices=sorted(ADAPTERS))
    g.add_argument("--cmd", help="command template with {path} {blocks} [{id} {data_len}]")
    g.add_argument("--python", help="module:function (module importable from PYTHONPATH)")
    ap.add_argument("--index", default=os.path.join(SHARED, "bc1_index.jsonl"))
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--easiest", type=int, default=0)
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--ids", help="file with one chunk id per line")
    ap.add_argument("--expect", default="default=auto", help=parse_expect.__doc__)
    ap.add_argument("--control", choices=["shuffle"], default=None)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--json", help="write per-chunk results + aggregate here")
    a = ap.parse_args()
    cand = ("python", a.python) if a.python else ("cmd", a.cmd or ADAPTERS[a.adapter])
    ids = select(a)
    results, agg = evaluate(ids, cand, a.expect, a.control, a.jobs, a.index)
    agg["candidate"] = cand[1]
    agg["expect"] = a.expect
    agg["control"] = a.control
    errs = Counter(r["error"].split(":")[0] for r in results if r["error"])
    agg["error_kinds"] = dict(errs)
    print(json.dumps(agg, indent=1))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"aggregate": agg, "results": results}, fh)


if __name__ == "__main__":
    main()
