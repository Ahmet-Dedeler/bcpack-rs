"""Export the fitted quadrant codebook to tools/sem/quad_table.json (derived structure, no game bytes).

usage: venv/bin/python -I tools/sem/export_table.py FINAL.npy EVEN.npy ODD.npy EDGE.npy
Each entry: symbol -> codes [TL, TR, BL, BR] of the 2x2 pixel group (BC1 2-bit codes:
0 = c0, 1 = c1, 2 = 2/3 c0 + 1/3 c1, 3 = 1/3 c0 + 2/3 c1), plus confidence fields:
  stable_halves   same entry in the fits on even stores and on odd stores (disjoint data)
  stable_edgeonly same entry in the fit without the mip-1 term (random init)
  margin          cost increase per occurrence of the cheapest swap in the final fit
  share           fraction of all quadrant occurrences using the symbol
"""
import json
import os
import sys
import numpy as np

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quad_table.json")


def main():
    fin, ev, od, ed = [np.load(p) for p in sys.argv[1:5]]
    T = fin[0].astype(int)
    F = np.load(os.path.join(SCR, "cooc.npz"))["F"].astype(float)
    ents = []
    for s in range(256):
        p = int(T[s])
        ents.append({
            "sym": s, "codes": [(p >> (2 * k)) & 3 for k in range(4)],
            "stable_halves": bool(ev[0][s] == od[0][s] == T[s]),
            "stable_edgeonly": bool(ed[0][s] == T[s]),
            "margin": None if s < 4 else round(float(fin[1][s] / max(fin[2][s], 1)), 3),
            "share": round(float(F[s] / F.sum()), 6),
        })
    doc = {
        "format": "BCPack BC1 index lanes 8-11: one symbol per 2x2 pixel group (lane 8 = TL, 9 = TR, "
                  "10 = BL, 11 = BR quadrant of the 4x4 block); symbol -> 4 BC1 codes in order "
                  "TL, TR, BL, BR of the group. BC1 index word bit 2*(4y+x) holds pixel (x, y).",
        "source": "fitted (tools/sem/fit_rgb.py) on decoded retail-pool data with exact endpoints; "
                  "symbols 0-3 measured exactly (uniform, code = sym ^ 2)",
        "bijection": len(set(T.tolist())) == 256,
        "entries": ents,
    }
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=0)
    print("wrote", OUT, "bijection", doc["bijection"],
          "stable_halves", sum(e["stable_halves"] for e in ents),
          "stable_edgeonly", sum(e["stable_edgeonly"] for e in ents))


if __name__ == "__main__":
    main()
