"""Cycle-level multi-lane MDU rANS simulation.

The per-lane decoder is a direct transliteration of Microsoft's silicon-rans
`decoder` module (mdu_minirans.v): two phases per clock, a one-byte input
buffer, a one-symbol output buffer, HDR0/HDR1/HDR2/PROCESSING/DRAINING
states, flush bit, 2-bit alphabet narrowing, 8/16-bit length (+1 / +257).

The feeder and symbol-vector assembly are NOT in the published Verilog.
They are HYPOTHESES here:
  feeder:  each clock (phase 0), lanes that raise din_req are served bytes
           from the single compressed stream in lane-index order, at most
           `window` (MDU_RANS_FEED_WINDOW = 8) bytes per clock. A lane that
           requests and is not served is blocked_on_input (stalls that clock).
  symvec:  one slot per lane; a lane's sym_ready is "its slot is empty".
           When every lane that is not finished has a full slot (or
           `symvec='free'`: slots drain immediately), the vector is emitted.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdu_tables import T, Model, L, LEGAL

IDLE, HDR0, HDR1, HDR2, PROC, DRAIN = range(6)


class Lane:
    def __init__(self, idx, alphabet_bits, out_target):
        self.idx = idx
        self.ab = alphabet_bits
        self.out_target = out_target
        self.state = IDLE
        self.ib = 0
        self.ib_full = False
        self.sym = 0
        self.sym_full = False
        self.in_rem = 0
        self.out_rem = 0
        self.p0 = 0
        self.p1 = 0
        self.hdr3 = False
        self.flush = False
        self.q = 0
        self.eab = 0
        self.model = None
        self.done = False
        self.stall_p1 = False
        self.go_pending = True
        self.frags = []        # (stream_offset_of_hdr0, mode, budget, eab, q)
        self.syms = []
        self.bytes = []        # stream offsets this lane loaded
        self.error = None
        self.overrun = False
        self.final_p0 = None

    def need_ib_load(self):
        if self.ib_full:
            return False
        if self.state in (HDR0, HDR1, HDR2):
            return True
        return self.state == PROC and self.in_rem > (1 if self.ib_full else 0)


def simulate(stream, alphabet_bits, out_targets, window=8, symvec="lockstep",
             max_cycles=None, stream_len=None, order=None):
    n = len(alphabet_bits)
    lanes = [Lane(i, alphabet_bits[i], out_targets[i]) for i in range(n)]
    slots = [False] * n
    vec = [None] * n
    pos = 0
    end = len(stream) if stream_len is None else stream_len
    cycle = 0
    vectors = 0
    last_prog, last_change = None, 0
    if max_cycles is None:
        max_cycles = 64 * (sum(out_targets) + end) + 1000
    while cycle < max_cycles:
        # ---------------- PHASE 0 ----------------
        served = 0
        for ln in (lanes if order is None else [lanes[i] for i in order]):
            sym_ready = not slots[ln.idx]
            blocked_out = ln.sym_full and not sym_ready
            need = ln.need_ib_load()
            din_req = need and not blocked_out
            din_valid = False
            if din_req and served < window and pos < end:
                din_valid = True
            blocked_in = need and not din_valid
            stall_p0 = blocked_in or blocked_out
            ln.stall_p1_next = stall_p0
            if stall_p0:
                continue
            if ln.state in (PROC, DRAIN):
                if ln.sym_full:
                    f, cf = ln.model.forward(ln.sym)
                    ln.p0 = (f * (ln.p1 >> 16) + (ln.p1 & 0xFFFF) - cf) & 0xFFFFFFFF
                    if ln.out_rem == 0:
                        ln.final_p0 = ln.p0
                        if ln.p0 not in LEGAL:
                            ln.overrun = True
                else:
                    ln.p0 = ln.p1
            if din_req and din_valid:
                ln.ib = stream[pos]
                ln.bytes.append(pos)
                pos += 1
                served += 1
                ln.ib_full = True
            if ln.sym_full and sym_ready:
                slots[ln.idx] = True
                vec[ln.idx] = ln.sym
                ln.syms.append(ln.sym)
                ln.sym_full = False
        # ---------------- PHASE 1 ----------------
        for ln in lanes:
            ln.stall_p1 = ln.stall_p1_next
            if ln.stall_p1:
                continue
            st = ln.state
            if st == IDLE:
                if ln.go_pending:
                    ln.go_pending = False
                    ln.state = HDR0
                    ln.out_rem = ln.out_target
                    ln.p0 = ln.p1 = 0
            elif st == HDR0:
                if ln.ib_full:
                    b = ln.ib
                    ln.in_rem = 0
                    ln.state = HDR1
                    ln.q = b & 15
                    ln.hdr3 = bool(b & 0x20)
                    ln.flush = bool(b & 0x10)
                    e = ln.ab - (b >> 6)
                    if e < 0:
                        e &= 15
                    ln.eab = e if e < 10 else 9
                    ln.ib_full = False
                    ln.frags.append([ln.bytes[-1], b, None, ln.eab, ln.q])
                    if ln.eab > 8:
                        ln.error = "eab>8"
                        return finish(lanes, pos, end, cycle, vectors, "eab_out_of_range")
            elif st == HDR1:
                if ln.ib_full:
                    ln.ib_full = False
                    ln.model = Model(ln.eab, ln.q)
                    if ln.hdr3:
                        ln.in_rem = ln.ib
                        ln.state = HDR2
                    else:
                        ln.in_rem = ln.ib + 1
                        ln.state = PROC
                        ln.frags[-1][2] = ln.in_rem
            elif st == HDR2:
                if ln.ib_full:
                    ln.in_rem = ((ln.ib << 8) | (ln.in_rem & 0xFF)) + 257
                    ln.frags[-1][2] = ln.in_rem
                    ln.state = PROC
                    ln.ib_full = False
            elif st in (PROC, DRAIN):
                if st == PROC:
                    want = ln.p0 < L and ln.in_rem > 0
                    will = want and ln.ib_full
                else:
                    will = False
                x = (((ln.p0 & 0xFFFFFF) << 8) | ln.ib) if will else ln.p0
                new_in = ln.in_rem - 1 if will else ln.in_rem
                nsf = x > L and ln.out_rem > 0
                new_out = ln.out_rem - 1 if nsf else ln.out_rem
                ln.p1 = x
                ln.out_rem = new_out
                ln.sym = ln.model.inverse(x & 0xFFFF)
                ln.sym_full = nsf
                if st == PROC:
                    ln.in_rem = new_in
                    if will:
                        ln.ib_full = False
                    if new_in == 0:
                        if new_out > 0:
                            if x == L:
                                ln.state = HDR0
                                if ln.flush:
                                    ln.p1 = 0
                            else:
                                ln.state = DRAIN
                        else:
                            ln.done = True
                            ln.state = IDLE
                            ln.final_p0 = _final(ln, x, nsf)
                else:
                    if x <= L:
                        if new_out > 0:
                            ln.state = HDR0
                            if ln.flush:
                                ln.p0 = ln.p1 = 0
                        else:
                            ln.state = IDLE
                            ln.done = True
                            ln.final_p0 = _final(ln, x, nsf)
        # ---------------- symvec ----------------
        if symvec == "free":
            for i in range(n):
                slots[i] = False
        else:
            active = [i for i in range(n) if not (lanes[i].done and not lanes[i].sym_full and not slots[i])]
            if any(slots) and all(slots[i] for i in active):
                for i in range(n):
                    slots[i] = False
                vectors += 1
        cycle += 1
        prog = (pos, sum(len(l.syms) for l in lanes), sum(l.state for l in lanes))
        if prog != last_prog:
            last_prog, last_change = prog, cycle
        elif cycle - last_change > 64:
            return finish(lanes, pos, end, cycle, vectors, "deadlock")
        if all(ln.done and not ln.sym_full for ln in lanes):
            # flush final symbols in slots
            return finish(lanes, pos, end, cycle, vectors, "all_done")
        if pos >= end and all((ln.done or ln.need_ib_load()) for ln in lanes) and \
                all(not ln.sym_full or slots[ln.idx] for ln in lanes if not ln.done) and \
                not any(ln.state in (PROC, DRAIN) and not ln.need_ib_load() for ln in lanes if not ln.done):
            return finish(lanes, pos, end, cycle, vectors, "starved")
    return finish(lanes, pos, end, cycle, vectors, "timeout")


def _final(ln, x, nsf):
    """State after the last symbol's forward update (what error_overrun checks)."""
    if not nsf:
        return x
    f, cf = ln.model.forward(ln.sym)
    return (f * (x >> 16) + (x & 0xFFFF) - cf) & 0xFFFFFFFF


def finish(lanes, pos, end, cycle, vectors, why):
    return dict(why=why, pos=pos, end=end, cycles=cycle, vectors=vectors,
                lanes=[dict(idx=l.idx, done=l.done, nsyms=len(l.syms), out_rem=l.out_rem,
                            p1=l.p1, final=l.final_p0, frags=l.frags, overrun=l.overrun, nbytes=len(l.bytes),
                            state=l.state) for l in lanes],
                syms=[l.syms for l in lanes])
