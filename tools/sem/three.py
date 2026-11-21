"""3-colour blocks (c0 <= c1 as decoded): which index semantics fits mip 1?
For known blocks, quadrant mean colour under the table vs the mip-1 pixel (mean abs error,
RGB 0-255), for several interpretations of the four codes:
  bc1      standard BC1 3-colour palette (code 2 = midpoint, code 3 = black)
  four     palette built as if 4-colour (1/3, 2/3 interpolants) with the same c0, c1
  black?   ... plus 'bc1 with code 3 = midpoint' (no black) to see if black is ever used
Reference: the same error on 4-colour blocks.
usage: venv/bin/python -I tools/sem/three.py --table T.npy [--minsep 40]  (|c0-c1| in RGB)
"""
import glob
import os
import sys
import numpy as np

SCR = os.environ.get("SEM_SCRATCH", "/tmp/bcpack-sem-indices")
MINSEP = float(sys.argv[sys.argv.index("--minsep") + 1]) if "--minsep" in sys.argv else 0.0
PATS = np.array([[(p >> (2 * k)) & 3 for k in range(4)] for p in range(256)])


def main():
    T = np.load(sys.argv[sys.argv.index("--table") + 1])[0].astype(int)
    acc = {}
    cnt3 = 0
    for f in sorted(glob.glob(os.path.join(SCR, "fullres", "*.npz"))):
        z = np.load(f)
        K, F4, pal, qs, m1 = z["known"], z["four"], z["pal"], z["qsym"].astype(int), z["mip1"]
        c0, c1 = pal[..., 0, :], pal[..., 1, :]
        variants = {
            "bc1": pal,
            "four": np.stack([c0, c1, (2 * c0 + c1) / 3, (c0 + 2 * c1) / 3], -2),
            "no_black": np.stack([c0, c1, (c0 + c1) / 2, (c0 + c1) / 2], -2),
            "code3=c1": np.stack([c0, c1, (c0 + c1) / 2, c1], -2),
        }
        sep = np.linalg.norm(c0 - c1, axis=-1) >= MINSEP
        for kind, mask in (("3col", K & ~F4 & sep), ("4col", K & F4 & sep)):
            by, bx = np.nonzero(mask)
            if kind == "3col":
                cnt3 += len(by)
            for q, (dy, dx) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                s = qs[2 * by + dy, 2 * bx + dx]
                codes = PATS[T[s]]                       # n x 4
                tgt = m1[2 * by + dy, 2 * bx + dx]
                for name, P in variants.items():
                    if kind == "4col" and name != "bc1":
                        continue
                    pp = P[by, bx]                        # n x 4 x 3
                    mean = np.take_along_axis(pp, codes[..., None].repeat(3, -1), 1).mean(1)
                    e = np.abs(mean - tgt).mean(1)
                    a = acc.setdefault((kind, name), [0.0, 0])
                    a[0] += e.sum(); a[1] += len(e)
    print("3-colour known blocks:", cnt3)
    for k, (s, n) in sorted(acc.items()):
        print("%-5s %-9s quadrant mean abs err %.2f  (n=%d)" % (k[0], k[1], s / max(n, 1), n))


if __name__ == "__main__":
    main()
