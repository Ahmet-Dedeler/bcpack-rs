"""Static power-of-two segmented model family from docs/BCPACK_SPEC.md section 3.

freqs[k] is the per-symbol frequency of segment k (k=0: symbol 0; k>=1:
symbols [2^(k-1), 2^k)). Symbol 0 reserves slot 0 (cf=1, f=freqs[0]-1).
"""
import math

M = 1 << 16
L = 1 << 24
LEGAL = (L, L >> 8, L >> 16)

LAMBDAS = [0.0, 0.739926, 1.496350, 2.307231, 3.221322, 4.307485, 5.676284,
           7.508501, 10.112494, 14.036894, 20.348947, 31.445855, 53.681684,
           109.26, 328.11561, 1.0e9]

EXACT = {
    (2, 1): [21168, 17592, 13388], (4, 1): [5665, 5407, 5048, 4398, 3347],
    (2, 2): [26354, 18130, 10526], (3, 2): [14406, 11948, 9065, 5263],
    (4, 2): [7539, 6867, 5975, 4531, 2632], (2, 3): [31900, 17918, 7859],
    (3, 4): [22624, 15126, 8437, 2728], (2, 5): [43800, 14920, 3408],
    (3, 6): [33414, 16436, 6031, 906], (4, 7): [24559, 15361, 7808, 2124, 188],
    (2, 8): [60308, 4812, 208], (2, 9): [63575, 1901, 30],
    (3, 10): [60388, 4744, 200, 1], (3, 11): [64250, 1262, 10, 1],
    (8, 11): [7575, 6701, 5582, 3892, 1918, 494, 34, 1, 1],
    (3, 12): [65456, 74, 1, 1], (4, 12): [63248, 2208, 34, 1, 1],
    (5, 12): [53291, 9957, 1104, 14, 1, 1], (5, 13): [63380, 2084, 22, 1, 1, 1],
    (6, 13): [53649, 9731, 1042, 4, 1, 1, 1], (6, 14): [65147, 327, 1, 1, 1, 1, 1],
}


def counts(eab):
    return [1] + [1 << (k - 1) for k in range(1, eab + 1)]


def spec_freqs(eab, q, use_exact=True, lam=None):
    """Frequency vector exactly as src/lib.rs ProbabilityModel::new builds it."""
    n = eab + 1
    c = counts(eab)
    if lam is None:
        if q == 0:
            return [M >> eab] * n
        if q == 15 or (eab == 1 and q >= 11) or (eab == 2 and q >= 12) or \
           (eab == 3 and q >= 13) or (eab in (4, 5) and q >= 14):
            return [M - ((1 << eab) - 1)] + [1] * eab
        if use_exact and (eab, q) in EXACT:
            return list(EXACT[(eab, q)])
        lam = LAMBDAS[q]
    ns = float(1 << eab)
    den = 1.0 - math.exp(-lam)
    raw = [0.0] * n
    for k in range(1, n):
        t = (1 << (k - 1)) / ns
        raw[k] = (1.0 - math.exp(-lam * t)) / den * 65536.0
    top = 65536
    fs = [0] * n
    for k in range(n - 1, 0, -1):
        f = max(1, int(round((top - raw[k]) / c[k])))
        fs[k] = f
        top -= f * c[k]
    fs[0] = max(top, 2)
    return fs


def tables(freqs):
    """Return per-segment (base, freq, count, seg_start)."""
    eab = len(freqs) - 1
    c = counts(eab)
    out = []
    acc = 0
    for k in range(eab + 1):
        out.append((acc, freqs[k], c[k], 0 if k == 0 else 1 << (k - 1)))
        acc += freqs[k] * c[k]
    assert acc == M, (freqs, acc)
    return out


def decode_lane(freqs, data, max_syms=1 << 20, want=None):
    """Decode one self-contained fragment: data = 4 BE state bytes + refills.

    Uses the reference rule (decode, then refill while x < L). Stops when the
    input is exhausted and x <= L (drain end). Returns (n_symbols, final_x,
    bytes_used, symbols) . A clean convergence is final_x in LEGAL with all
    bytes used.
    """
    tab = tables(freqs)
    x = int.from_bytes(data[:4], "big")
    pos = 4
    n = len(data)
    syms = []
    while len(syms) < max_syms:
        if pos >= n and x <= L:
            break
        s = x & 0xFFFF
        if s == 0:
            return len(syms), x, pos, syms, "slot0"
        # find segment
        k = len(tab) - 1
        while tab[k][0] > s:
            k -= 1
        base, f, cnt, st = tab[k]
        if k == 0:
            sym, ff, cf = 0, f - 1, 1
        else:
            step = (s - base) // f
            sym, ff, cf = st + step, f, base + step * f
        syms.append(sym)
        x = ff * (x >> 16) + s - cf
        while x < L and pos < n:
            x = (x << 8) | data[pos]
            pos += 1
        if want is not None and len(syms) >= want:
            break
    return len(syms), x, pos, syms, "ok"
