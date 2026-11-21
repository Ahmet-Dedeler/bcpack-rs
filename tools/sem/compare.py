"""Agreement between quadrant tables from disjoint fits.
usage: venv/bin/python -I tools/sem/compare.py A.npy B.npy [C.npy ...]
Prints entries equal in all tables, and the share of quadrant occurrences (all decoded
blocks, cooc.npz frequencies) covered by agreeing entries; agreement by frequency rank.
"""
import os
import sys
import numpy as np

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")


def main():
    Ts = [np.load(p)[0].astype(int) for p in sys.argv[1:]]
    F = np.load(os.path.join(SCR, "cooc.npz"))["F"].astype(float)
    agree = np.all([t == Ts[0] for t in Ts[1:]], axis=0)
    print("entries agreeing in all %d tables: %d / 256; occurrence-weighted %.4f"
          % (len(Ts), agree.sum(), (F * agree).sum() / F.sum()))
    order = np.argsort(-F)
    for lo, hi in [(0, 32), (32, 64), (64, 128), (128, 192), (192, 256)]:
        sel = order[lo:hi]
        print("  frequency rank %3d-%3d: %2d / %d agree (share of occurrences %.4f)"
              % (lo, hi - 1, agree[sel].sum(), hi - lo, F[sel].sum() / F.sum()))
    for i in range(len(Ts)):
        for j in range(i + 1, len(Ts)):
            print("  pair %d-%d: %d / 256" % (i, j, (Ts[i] == Ts[j]).sum()))
    return agree


if __name__ == "__main__":
    main()
