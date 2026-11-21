"""Block-order and lane-geometry test from mip-1 (no endpoints needed).

For each order hypothesis file (quadref.py --order X), compare the mip-1 luminance difference
between two quadrant pixels when the two lane symbols are equal vs different. Correct order
+ correct lane geometry: equal symbols -> near-identical mip-1 pixels.
usage: venv/bin/python -I tools/sem/order.py [gfx10 linear ...]
"""
import os
import sys
import numpy as np

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")


def main():
    for order in (sys.argv[1:] or ["gfx10", "linear"]):
        z = np.load(os.path.join(SCR, "quad_%s.npz" % order))
        S = z["syms"].astype(int); V = z["valid"]
        Y = z["rgb"].astype(float) @ np.array([.299, .587, .114])
        L = S[8:12]
        eq4 = (L[0] == L[1]) & (L[1] == L[2]) & (L[2] == L[3])
        sp = Y.max(1) - Y.min(1)
        print("%s: blocks %d; mip-1 2x2 spread when all 4 symbols equal %.2f, otherwise %.2f"
              % (order, V.sum(), sp[V & eq4].mean(), sp[V & ~eq4].mean()))
        for a, b in [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]:
            for qa, qb in [(a, b)]:
                e = L[a] == L[b]; dd = np.abs(Y[:, qa] - Y[:, qb])
                print("   lanes %d,%d equal (%.3f of blocks): |dY| %.2f vs %.2f when different"
                      % (8 + a, 8 + b, e[V].mean(), dd[V & e].mean(), dd[V & ~e].mean()))


if __name__ == "__main__":
    main()
