"""Try many configs on a few chunks; print lanes-legal pattern per config.
python3 -I quick7.py N_CHUNKS BLOCKS CFG1 CFG2 ..."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scan7
from multiprocessing import Pool
n, blocks = int(sys.argv[1]), [int(x) for x in sys.argv[2].split(',')]
rows = scan7.load_chunks(0x1C, None, blocks, n, 7, False)
datas = [scan7.read_chunk(rel, ch) for _, rel, ch, _ in rows]
def one(cfg):
    out = []
    for d, b in datas:
        ok, r, s, lok = scan7.run_chunk(d, b, scan7.parse_config(cfg))
        out.append(("P" if ok else "") + "".join("1" if x else "0" for x in lok) + "/%s/%d" % (r["why"][:4], r["pos"] - s))
    return cfg, out
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("fork")
    cfgs = sys.argv[3:]
    with Pool(6) as p:
        for cfg, out in p.imap(one, cfgs):
            print(cfg, " ".join(out), flush=True)
