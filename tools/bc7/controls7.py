"""Negative controls for a converging lane config: perturb one lane at a time
(width -1/+1, count B and 4B instead of 2B) and count how many chunks still PASS.
Only chunks where the perturbed lane is not trivially constant are counted
(lane consumed > 6 bytes under the true config).
python3 -I controls7.py CONFIG N_CHUNKS JOBS [blocks]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scan7
from multiprocessing import Pool
cfg = scan7.parse_config(sys.argv[1]); n = int(sys.argv[2]); jobs = int(sys.argv[3])
blocks = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [512, 768, 1024]
codec = int(sys.argv[5], 0) if len(sys.argv) > 5 else 0x1C
scan7._CODEC = codec
rows = scan7.load_chunks(codec, None, blocks, n, 3, False)

def one(row):
    cid, rel, ch, b = row
    d, _ = scan7.read_chunk(rel, ch, codec)
    ok, r, s, lok = scan7.run_chunk(d, b, cfg)
    out = {"id": cid, "true": ok, "var": [l["nbytes"] > 6 for l in r["lanes"]], "pert": {}}
    for j, (w, t) in enumerate(cfg):
        if not out["var"][j]:
            continue
        alt = [("w-1", w - 1, t), ("w+1", w + 1, t), ("cnt/2", w, "B" if t == "2B" else t + "/2"), ("cnt*2", w, "4B" if t == "2B" else "2" + t)]
        for name, w2, t2 in alt:
            if w2 < 1 or w2 > 9:
                continue
            c2 = list(cfg); c2[j] = (w2, t2)
            try:
                ok2 = scan7.run_chunk(d, b, c2)[0]
            except Exception:
                ok2 = False
            out["pert"]["%d:%s" % (j, name)] = ok2
    return out

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("fork")
    from collections import defaultdict
    tot = defaultdict(lambda: [0, 0])
    ntrue = 0
    with Pool(jobs) as p:
        for o in p.imap_unordered(one, rows):
            ntrue += o["true"]
            for k, v in o["pert"].items():
                tot[k][0] += v; tot[k][1] += 1
    print("true config PASS %d / %d" % (ntrue, len(rows)))
    allp = sum(v[0] for v in tot.values()); allt = sum(v[1] for v in tot.values())
    for k in sorted(tot, key=lambda k: (int(k.split(':')[0]), k)):
        print("  lane %-10s pass %d / %d" % (k, tot[k][0], tot[k][1]))
    print("ALL perturbations pass %d / %d" % (allp, allt))
