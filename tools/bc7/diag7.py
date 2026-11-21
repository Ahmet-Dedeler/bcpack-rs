"""Per-lane diagnostic for one chunk: python3 -I diag7.py STORE_REL CHUNK CONFIG [codec]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scan7
from mdu_tables import LEGAL
from collections import Counter
rel, ch, cfg = sys.argv[1], int(sys.argv[2]), scan7.parse_config(sys.argv[3])
codec = int(sys.argv[4], 0) if len(sys.argv) > 4 else 0x1C
data, blocks = scan7.read_chunk(rel, ch, codec)
ok, r, stripped, lok = scan7.run_chunk(data, blocks, cfg)
print("blocks", blocks, "size", len(data), "stripped", stripped, "pos", r["pos"], r["why"], "ok", ok, "cycles", r["cycles"])
for l, s in zip(r["lanes"], r["syms"]):
    print("lane %2d done=%d legal=%d nsyms=%d out_rem=%d nbytes=%d st=%d final=%s frags=%s" % (
        l["idx"], l["done"], l["final"] in LEGAL, len(s), l["out_rem"], l["nbytes"], l["state"],
        hex(l["final"]) if l["final"] is not None else None,
        [(f[0], "%02x" % f[1], f[2]) for f in l["frags"]]))
    print("     ends", l["ends"][:12]); print("     top", Counter(s).most_common(10), "first", s[:24])
