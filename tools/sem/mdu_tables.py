"""Load the exact MDU rANS tables from Microsoft's MIT-licensed silicon-rans
Verilog (mdu_minirans.v) and provide a faithful single-lane decoder.

Table index `eab` (0..8) selects an alphabet of eab+1 bits (the Verilog labels
entry 0 "alphabet size = 1 bit"; segment 9 is enabled for eab >= 8 -> 9-bit
symbols). Segment k: k=0 -> symbol 0, k>=1 -> symbols [2^(k-1), 2^k).
"""
import re

VPATH = "/Users/ahmet/Code/bcpack-refs/microsoft_silicon-rans/mdu_minirans.v"
L = 1 << 24
LEGAL = (L, L >> 8, L >> 16)


def load(path=VPATH):
    src = open(path).read()
    out = {}
    for name in ("base_table", "freq_table", "rf_table", "rs_table", "rn_table"):
        m = re.search(r"assign\s+%s\s*=\s*'\{(.*?)\}\s*;\s*//\s*end %s" % (name, name), src, re.S)
        body = re.sub(r"//[^\n]*", "", m.group(1))
        nums = [int(x) for x in re.findall(r"\d+", body)]
        assert len(nums) == 9 * 16 * 10, (name, len(nums))
        out[name] = [[nums[(e * 16 + q) * 10:(e * 16 + q) * 10 + 10] for q in range(16)] for e in range(9)]
    return out


T = load()


class Model:
    """One (eab, q) model with the Verilog's exact inverse and forward paths."""

    def __init__(self, eab, q, tabs=T):
        self.eab, self.q = eab, q
        self.base = tabs["base_table"][eab][q]
        self.freq = tabs["freq_table"][eab][q]
        self.rf = tabs["rf_table"][eab][q]
        self.rs = tabs["rs_table"][eab][q]
        self.rn = tabs["rn_table"][eab][q]
        self.nseg = eab + 2  # segments 0..eab+1

    def inverse(self, cf_in):
        b = self.base
        seg = 0
        for k in range(self.nseg - 1, 0, -1):
            if cf_in >= b[k]:
                seg = k
                break
        base_x = 0 if seg == 0 else 1 << (seg - 1)
        dist = ((cf_in - b[seg]) + self.rn[seg]) & 0xFFFF
        add_mul = (dist * self.rf[seg]) & 0xFFFFFFFF
        steps = ((add_mul >> 15) >> self.rs[seg]) & 0x1FF
        return (base_x + steps) & 0x1FF

    def forward(self, sym):
        if sym == 0:
            seg, st = 0, 0
        else:
            seg = sym.bit_length()
            st = 1 << (seg - 1)
        fa = self.freq[seg]
        cfa = (self.base[seg] + fa * (sym - st)) & 0xFFFF
        if cfa == 0:
            return (fa - 1) & 0xFFFF, 1
        return fa, cfa

    def step(self, x):
        s = self.inverse(x & 0xFFFF)
        f, cf = self.forward(s)
        return s, (f * (x >> 16) + (x & 0xFFFF) - cf) & 0xFFFFFFFF


def decode_fragment(model, payload, out_target=1 << 30, x0=0):
    """Single-lane, schedule-free emulation of the Verilog PROCESSING/DRAINING
    path for one flushed fragment whose bytes are `payload` (budget = len).

    Returns dict(n, x, used, end, syms). end: 'done' if out_target reached,
    'next' if the fragment ended with output remaining (state==L at input end,
    or drained to <= L), 'stuck' if input not exhausted when output target hit.
    """
    x = x0
    pos = 0
    rem = len(payload)
    n = 0
    syms = []
    draining = False
    while True:
        # PHASE 1: maybe feed one byte, then maybe emit
        if not draining and x < L and rem > 0:
            x = ((x & 0xFFFFFF) << 8) | payload[pos]
            pos += 1
            rem -= 1
        emit = x > L and n < out_target
        if emit:
            s, xn = model.step(x)
            syms.append(s)
            n += 1
        if not draining:
            if rem == 0:
                if n < out_target:
                    if x == L and not emit:
                        return dict(n=n, x=x, used=pos, end="next", syms=syms)
                    draining = True
                else:
                    return dict(n=n, x=xn if emit else x, used=pos, end="done", syms=syms)
        else:
            if x <= L:
                return dict(n=n, x=x, used=pos, end="next" if n < out_target else "done", syms=syms)
        if emit:
            x = xn
        elif not draining and rem > 0 and x >= L:
            # output target reached but input remains: hardware would hang
            if n >= out_target:
                return dict(n=n, x=x, used=pos, end="stuck", syms=syms)
