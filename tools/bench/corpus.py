"""Index every BCPack BC1 (0x04) and BC7 (0x1C) chunk of every RDS5 store.

Usage:
    python3 -I tools/bench/corpus.py [--out /tmp/bcpack-shared] [--jobs 18]
    python3 -I tools/bench/corpus.py --summary          # histograms from existing index
    python3 -I tools/bench/corpus.py --easiest 30       # easiest BC1 chunks

Writes ``bc1_index.jsonl`` and ``bc7_index.jsonl`` (one JSON object per chunk) and
``corpus_summary.json``.  Fields per chunk:

  id            "<store relpath>:<chunk index>"  (use with ttpf.extract)
  store, chunk, file_off, file_len, pad, data_len (= file_len - pad), vstart, vlen
  blocks        vlen / 8 (BC1) or vlen / 16 (BC7)
  pad_ok        the last `pad` stored bytes are all zero
  zeros_before_pad  zero bytes immediately before the pad (data bytes that happen to be 0)
  sixng_multistream  blocks in (1536, 3072): SixNG says these hold several streams
  lanes         8 main-lane records (fragment 0), see lanes.parse_record, plus
                cls: "const" (proven constant: const=[[K, N], ...]),
                     "len4"  (budget 4, no renorm bytes, not proven constant),
                     "var"   (everything else)
  n_nonconst    lanes with cls != const   (the brief's "varying lanes")
  n_var         lanes with cls == var (excludes len4)
  main_sum      sum over the 8 lanes of hdr_bytes + budget (fragment 0 only)
  rest          data_len - main_sum (bytes left for later fragments/index lanes)
  lane1_N       N of a proven-constant lane 1 (stream symbol count hint), else null
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lanes as L  # noqa: E402
import ttpf  # noqa: E402

SHARED = "/tmp/bcpack-shared"


def index_store(rel: str):
    st = ttpf.Store(rel)
    bc1, bc7 = [], []
    codec_counts = Counter(c.codec for c in st.chunks)
    for c in st.chunks:
        if c.codec not in (0x04, 0x1C) or c.file_len <= 0:
            continue
        if c.file_off + c.file_len > st.size or c.file_len < 48:
            codec_counts[f"skipped_{c.codec:#04x}"] += 1  # cut off by end of file / too short
            continue
        head = st.read(c.file_off, 48)
        tl = min(c.file_len, max(256, c.pad + 64))
        tail = st.read(c.file_off + c.file_len - tl, tl)
        pz = c.pad
        pad_ok = tail[tl - pz:] == b"\0" * pz          # the pad bytes are all zero
        body = tail[:tl - pz]
        extra_zeros = len(body) - len(body.rstrip(b"\0"))  # zero data bytes right before the pad
        bpb = 8 if c.codec == 0x04 else 16
        rec = {"id": f"{rel}:{c.index}", "store": rel, "chunk": c.index, "codec": c.codec,
               "file_off": c.file_off, "file_len": c.file_len, "pad": c.pad,
               "data_len": c.data_len, "vstart": c.vstart, "vlen": c.vlen,
               "blocks": c.vlen // bpb, "pad_ok": pad_ok, "zeros_before_pad": extra_zeros,
               "sixng_multistream": c.vlen // bpb in (1536, 3072)}
        lanes = []
        main_sum = 0
        for li, r in enumerate(L.main_records(head)):
            p = L.parse_record(r)
            p["lane"] = li
            m = L.const_match(p["budget"], p["state"]) if not p["hdr3"] else []
            good = [x for x in m if x[0] in L.MODE_K.get(p["mode"], ())]
            if m:
                p["const"] = [list(x) for x in m]
            if good:
                p["cls"] = "const"
                p["const_K"], p["const_N"] = good[0]
            elif not p["hdr3"] and p["budget"] == 4:
                p["cls"] = "len4"
            else:
                p["cls"] = "var"
            main_sum += p["hdr_bytes"] + p["budget"]
            lanes.append(p)
        rec["lanes"] = lanes
        rec["n_nonconst"] = sum(1 for p in lanes if p["cls"] != "const")
        rec["n_var"] = sum(1 for p in lanes if p["cls"] == "var")
        rec["main_sum"] = main_sum
        rec["rest"] = c.data_len - main_sum
        rec["main_sum_alt"] = sum(p["hdr_bytes"] + p.get("budget_alt_plus1", p["budget"]) for p in lanes)
        l1 = lanes[1]
        rec["lane1_N"] = l1["const_N"] if l1["cls"] == "const" else None
        (bc1 if c.codec == 0x04 else bc7).append(rec)
    st.close()
    return rel, bc1, bc7, dict(codec_counts)


def summarize(bc1, bc7, codec_totals=None):
    def h(xs):
        return dict(sorted(Counter(xs).items(), key=lambda kv: (str(type(kv[0])), kv[0])))
    s = {}
    for name, rows in (("bc1", bc1), ("bc7", bc7)):
        d = {"chunks": len(rows),
             "stored_bytes": sum(r["data_len"] for r in rows),
             "virtual_bytes": sum(r["vlen"] for r in rows),
             "pad_ok": sum(r["pad_ok"] for r in rows),
             "zeros_before_pad_hist": h(min(r["zeros_before_pad"], 8) for r in rows),
             "pad_hist_top": dict(Counter(r["pad"] for r in rows).most_common(8)),
             "pad_max": max((r["pad"] for r in rows), default=0),
             "n_nonconst_hist": h(r["n_nonconst"] for r in rows),
             "n_var_hist": h(r["n_var"] for r in rows),
             "blocks_hist": h(r["blocks"] for r in rows),
             "rest_negative": sum(1 for r in rows if r["rest"] < 0),
             "rest_negative_alt_plus1": sum(1 for r in rows if r["data_len"] - r["main_sum_alt"] < 0),
             "sixng_multistream": sum(r["sixng_multistream"] for r in rows)}
        per_lane = {}
        for li in range(8):
            modes = Counter(f"{r['lanes'][li]['mode']:02x}" for r in rows)
            cls = Counter(r["lanes"][li]["cls"] for r in rows)
            ks = Counter(r["lanes"][li]["const_K"] for r in rows if r["lanes"][li]["cls"] == "const")
            chance = Counter(f"{r['lanes'][li]['mode']:02x}" for r in rows
                             if r["lanes"][li]["cls"] != "const" and r["lanes"][li].get("const"))
            nrel = Counter()
            for r in rows:
                p = r["lanes"][li]
                if p["cls"] != "const":
                    continue
                N = p["const_N"]
                b = r["blocks"]
                nrel["1x" if N == b else "2x" if N == 2 * b else "0.5x" if 2 * N == b else "other"] += 1
            per_lane[li] = {"modes_top": dict(modes.most_common(12)), "n_modes": len(modes),
                            "cls": dict(cls), "const_K": dict(ks), "nonmode_const_hits": sum(chance.values()), "const_N_vs_blocks": dict(nrel)}
        d["per_lane"] = per_lane
        # mode -> K consistency (does a mode always mean one K on a lane?)
        mk = Counter()
        for r in rows:
            for p in r["lanes"]:
                if p["cls"] == "const":
                    mk[(p["lane"], f"{p['mode']:02x}", p["const_K"], len(p["const"]))] += 1
        d["const_lane_mode_K"] = {f"L{a} {m} K={k} matches={n}": c for (a, m, k, n), c in sorted(mk.items())}
        # mode-byte records with budget 4 for a mode family, per lane
        s[name] = d
    if codec_totals:
        s["codec_chunk_counts"] = codec_totals
    return s


def load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh]


def easiest(bc1, n=30):
    rows = [r for r in bc1 if r["pad_ok"]]
    rows.sort(key=lambda r: (r["n_nonconst"], r["n_var"], r["data_len"]))
    return rows[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=SHARED)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--easiest", type=int, default=0)
    a = ap.parse_args()
    b1p, b7p = os.path.join(a.out, "bc1_index.jsonl"), os.path.join(a.out, "bc7_index.jsonl")
    if a.summary or a.easiest:
        bc1 = load(b1p)
        if a.summary:
            print(json.dumps(summarize(bc1, load(b7p)), indent=1))
        for r in easiest(bc1, a.easiest):
            print(r["id"], r["blocks"], r["data_len"], r["n_nonconst"], r["n_var"],
                  " ".join(f"{p['mode']:02x}:{p['cls'][0]}" for p in r["lanes"]))
        return
    os.makedirs(a.out, exist_ok=True)
    with Pool(min(a.jobs, len(L.CONST_KS))) as pool:  # build constant tables once
        pool.map(L._pool_safe, L.CONST_KS, chunksize=1)
    stores = ttpf.list_stores()
    bc1, bc7 = [], []
    codec_totals = Counter()
    with Pool(a.jobs) as pool:
        for rel, r1, r7, cc in pool.imap_unordered(index_store, stores):
            bc1 += r1
            bc7 += r7
            codec_totals.update(cc)
            print(f"{rel}: bc1={len(r1)} bc7={len(r7)}", file=sys.stderr)
    key = lambda r: (r["store"], r["chunk"])
    bc1.sort(key=key)
    bc7.sort(key=key)
    for p, rows in ((b1p, bc1), (b7p, bc7)):
        with open(p + ".tmp", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        os.replace(p + ".tmp", p)
    s = summarize(bc1, bc7, {(f"{k:#04x}" if isinstance(k, int) else k): v for k, v in sorted(codec_totals.items(), key=str)})
    s["stores"] = len(stores)
    with open(os.path.join(a.out, "corpus_summary.json"), "w") as fh:
        json.dump(s, fh, indent=1)
    print(json.dumps({k: s[k] for k in ("stores", "codec_chunk_counts")}, indent=1))
    print(f"wrote {b1p} ({len(bc1)}) and {b7p} ({len(bc7)})")


if __name__ == "__main__":
    main()
