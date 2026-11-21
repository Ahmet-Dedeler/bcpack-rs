#!/usr/bin/env python3
"""Reference model of the Xbox Series MDU rANS lane decoder (stdlib only).

This is a faithful port of Microsoft's reference Verilog `mdu_minirans.v`
(https://github.com/microsoft/silicon-rans, MIT), which is the same module that
patent US11234023B2 prints as FIGS. 15a-15k (the patent omits the table values;
the Verilog has them). Tables live in `mdu_tables.py`, generated from the .v file
by `gen_mdu_tables.py` (never typed by hand).

It is deliberately independent of `bcpack.py` so the two can be compared.

Contents
--------
* `Model`        - one static probability model (alphabet width, q): forward
                   (encode/state-update) and inverse (symbol lookup) exactly as
                   the Verilog computes them, including 16-bit wraparound, the
                   reserved slot 0 and the reciprocal-multiply division.
* `LaneCycleSim` - cycle-accurate two-phase port of `module decoder` (FIG. 15
                   / mdu_minirans.v), one lane, with explicit feeder and
                   downstream handshakes. Records the clock at which each input
                   byte is requested and accepted.
* `decode_lane`  - functional (non-cycle) equivalent of one lane; must agree
                   with LaneCycleSim on symbols, bytes consumed and end state.
* `encode_lane`  - synthetic encoder (not in the patent or the Verilog) that
                   produces a lane byte stream the decoder accepts, supporting
                   multiple fragments, per-fragment model / width adjust,
                   flushed or retained state, and end trimming.
* `ArraySim`     - several LaneCycleSim instances behind a pluggable feeder
                   policy, for testing read-schedule hypotheses. The feeder and
                   the symbol-vector assembly are NOT in the patent or the
                   Verilog; the policy shipped here is a hypothesis.

Run `python3 -I python/patent_ref.py` for the self-tests.

Notation: `eab` follows the Verilog register: eab = (alphabet bits - 1), so
table index 0 is the 1-bit alphabet and segment k (symbols [2^(k-1), 2^k)) is
enabled iff eab >= k - 1. The decoder input `alphabet_bits` uses the same
"bits - 1" encoding (it is subtracted from directly to form eab).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # allow both `python3 python/patent_ref.py` and `import patent_ref`
    from mdu_tables import BASE_TABLE, FREQ_TABLE, RF_TABLE, RN_TABLE, RS_TABLE
except ImportError:  # pragma: no cover
    sys.path.insert(0, __import__("os").path.dirname(__file__))
    from mdu_tables import BASE_TABLE, FREQ_TABLE, RF_TABLE, RN_TABLE, RS_TABLE

# `define values from mdu_minirans.v
LOWER_LIMIT = 1 << 24          # MDU_RANS_LOWER_LIMIT
PROB_BITS = 16                 # MDU_RANS_PROB_BITS
SYMOUT_W = 9                   # MDU_RANS_SYMOUT_W
MAX_INPUT_LOG2 = 18            # MDU_MAX_INPUT_LOG2 (input_remaining width)
FEED_WINDOW = 8                # MDU_RANS_FEED_WINDOW: items fed to the array per update cycle
NUM_DEC = 18                   # MDU_RANS_NUM_DEC: decoders in the array
LEGAL_END_STATES = (LOWER_LIMIT, LOWER_LIMIT >> 8, LOWER_LIMIT >> 16)

M16 = 0xFFFF
M32 = 0xFFFFFFFF

# ctrl_state encoding (FIG. 15c)
IDLE, HDR0, HDR1, HDR2, PROCESSING, DRAINING = range(6)
STATE_NAMES = ("IDLE", "HDR0", "HDR1", "HDR2", "PROCESSING", "DRAINING")


# ---------------------------------------------------------------------------
# Probability model (FIGS. 15b, 15d-15g)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Model:
    """Static model `q` for effective alphabet `eab` (= bits - 1)."""

    eab: int
    q: int

    @property
    def base(self) -> Tuple[int, ...]:
        return BASE_TABLE[self.eab][self.q]

    @property
    def freq(self) -> Tuple[int, ...]:
        return FREQ_TABLE[self.eab][self.q]

    @property
    def bits(self) -> int:
        return self.eab + 1

    def forward(self, sym: int) -> Tuple[int, int]:
        """(fwd_f, fwd_cf) for `sym` (FIG. 15f/15g, mdu_minirans.v l.620-665)."""
        sym &= (1 << SYMOUT_W) - 1
        seg = sym.bit_length()                     # priority encoder on sym[8:0]
        segstart = 0 if seg == 0 else 1 << (seg - 1)
        fa = self.freq[seg]
        steps = (sym - segstart) & ((1 << SYMOUT_W) - 1)
        cfa = (self.base[seg] + fa * steps) & M16
        if cfa == 0:                               # symbol 0: slot 0 is reserved
            return (fa - 1) & M16, 1
        return fa, cfa

    def inverse(self, cf_in: int) -> int:
        """Symbol for the low 16 state bits (FIG. 15d/15e, l.562-611)."""
        b, eab = self.base, self.eab
        seg = 0
        for k in range(9, 0, -1):                  # comparator chain, seg 9 first
            gate = True if k == 1 else eab >= k - 1
            if cf_in >= b[k] and gate:
                seg = k
                break
        inv_base_x = 0 if seg == 0 else 1 << (seg - 1)
        rn = RN_TABLE[self.eab][self.q][seg]
        rf = RF_TABLE[self.eab][self.q][seg]
        rs = RS_TABLE[self.eab][self.q][seg]
        dist_x = (cf_in - b[seg] + rn) & M16
        add_mul = (dist_x * rf) & M32
        inv_steps = ((add_mul >> 15) & 0x1FFFF) >> rs
        return (inv_base_x + inv_steps) & ((1 << SYMOUT_W) - 1)

    def decode_step(self, x: int) -> Tuple[int, int]:
        """Decode one symbol from state x (> L); return (sym, new_state)."""
        sym = self.inverse(x & M16)
        f, cf = self.forward(sym)
        return sym, (f * (x >> 16) + (x & M16) - cf) & M32

    def encode_step(self, x: int, sym: int, out_rev: bytearray) -> int:
        """Encoder: renormalise then C(s, x). Bytes go to out_rev in emit order."""
        f, cf = self.forward(sym)
        if f == 0:
            raise ValueError(f"symbol {sym} has zero frequency in {self}")
        x_max = f << 16                            # ((L >> 16) << 8) * f
        while x >= x_max:
            out_rev.append(x & 0xFF)
            x >>= 8
        return ((x // f) << 16) + (x % f) + cf


def model_for(alphabet_bits: int, z: int, q: int) -> Model:
    """Model selected by a fragment header (eab_clamped in FIG. 15d)."""
    eab = (alphabet_bits - z) & 0xF
    eab = eab if eab < 10 else 9
    if eab > 8:
        raise ValueError("eab 9 indexes past the [9] tables (Verilog X)")
    return Model(eab, q & 0xF)


# ---------------------------------------------------------------------------
# Fragment header (FIG. 14, FIG. 15d/15i)
# ---------------------------------------------------------------------------

@dataclass
class FragmentHeader:
    z: int          # hdr0[7:6] symbol width reduction
    hdr3: bool      # hdr0[5]   3-byte header (16-bit length)
    flush: bool     # hdr0[4]   zero rANS state when THIS fragment ends
    q: int          # hdr0[3:0] static model
    length: int     # payload bytes (input_remaining)

    def encode(self) -> bytes:
        n = self.length
        if n < 1 or n > 65535 + 257:
            raise ValueError("fragment length out of range 1..65792")
        hdr3 = n > 256
        b0 = (self.z & 3) << 6 | int(hdr3) << 5 | int(self.flush) << 4 | (self.q & 0xF)
        if not hdr3:
            return bytes((b0, n - 1))
        v = n - 257
        return bytes((b0, v & 0xFF, v >> 8))

    @staticmethod
    def parse(data: Sequence[int], pos: int) -> Tuple["FragmentHeader", int]:
        b0 = data[pos]
        hdr3 = bool(b0 >> 5 & 1)
        if hdr3:
            n = (data[pos + 1] | data[pos + 2] << 8) + 257
            used = 3
        else:
            n = data[pos + 1] + 1
            used = 2
        return FragmentHeader(b0 >> 6, hdr3, bool(b0 >> 4 & 1), b0 & 0xF, n), used


# ---------------------------------------------------------------------------
# Cycle-accurate lane (FIGS. 15a-15k / mdu_minirans.v module decoder)
# ---------------------------------------------------------------------------

class LaneCycleSim:
    """One `module decoder` instance, clocked one edge at a time.

    Drive it with `clock(din_valid, din, sym_ready, array_stall, go)`. Before
    clocking, `wires(...)` tells you `din_req` / `din_ready` / `sym_valid` for
    the current edge so a feeder or symvec model can react combinationally,
    exactly like the RTL handshakes.
    """

    def __init__(self, alphabet_bits: int, out_target: int):
        self.alphabet_bits = alphabet_bits & 0xF
        self.out_target = out_target
        # registers (reset values, FIG. 15g)
        self.ctrl = IDLE
        self.phase = 0
        self.p0 = 0
        self.p1 = 0
        self.input_remaining = 0
        self.output_remaining = 0
        self.sym = 0
        self.sym_buf_full = False
        self.input_buf_full = False
        self.input_buf = 0
        self.done = False
        self.error_overrun = False
        self.eab = 0
        self.q = 0
        self.model = Model(0, 0)       # stands for the *_sel2 table registers
        self.hdr3 = False
        self.flush_per_frag = False
        self.stall_p1 = False
        # bookkeeping (not RTL)
        self.cycle = 0
        self.symbols: List[int] = []
        self.byte_cycles: List[int] = []   # clock index at which each byte was accepted
        self.headers: List[FragmentHeader] = []
        self.overrun_seen = False
        self.end_state = 0                 # new_p0 at the latest symbol transfer
        self._hdr_bytes: List[int] = []

    # -- combinational logic -------------------------------------------------
    def wires(self, din_valid: bool = True, sym_ready: bool = True,
              array_stall: bool = False) -> Dict[str, object]:
        L = LOWER_LIMIT
        want = self.p0 < L and self.input_remaining > 0
        will = want and self.input_buf_full
        rswi = ((self.p0 & 0xFFFFFF) << 8 | self.input_buf) if will else self.p0
        new_in_rem = self.input_remaining - 1 if will else self.input_remaining
        need_ib_load = (not self.input_buf_full) and (
            self.ctrl in (HDR0, HDR1, HDR2)
            or (self.ctrl == PROCESSING and self.input_remaining > int(self.input_buf_full)))
        blocked_on_output = self.sym_buf_full and not sym_ready
        blocked_on_input = need_ib_load and not din_valid
        stall_p0 = array_stall or blocked_on_input or blocked_on_output
        din_req = need_ib_load and self.phase == 0 and not blocked_on_output
        din_ready = din_req and not array_stall
        sym_valid = self.sym_buf_full and self.phase == 0 and not stall_p0
        next_sbf = rswi > L and self.output_remaining > 0
        new_out_rem = self.output_remaining - 1 if (next_sbf and self.output_remaining > 0) \
            else self.output_remaining
        new_sym = self.model.inverse(rswi & M16)
        f, cf = self.model.forward(self.sym)
        new_p0 = (f * (self.p1 >> 16) + (self.p1 & M16) - cf) & M32
        return dict(will_feed=will, rswi=rswi, new_in_rem=new_in_rem,
                    need_ib_load=need_ib_load, stall_p0=stall_p0, din_req=din_req,
                    din_ready=din_ready, sym_valid=sym_valid, next_sbf=next_sbf,
                    new_out_rem=new_out_rem, new_sym=new_sym, new_p0=new_p0,
                    legal_end=new_p0 in LEGAL_END_STATES)

    # -- one clock edge -------------------------------------------------------
    def clock(self, din_valid: bool = False, din: int = 0, sym_ready: bool = True,
              array_stall: bool = False, go: bool = False) -> Dict[str, object]:
        w = self.wires(din_valid, sym_ready, array_stall)
        stall_p0 = w["stall_p0"]
        nxt_stall_p1 = stall_p0
        if self.phase == 0:
            self.error_overrun = bool(w["sym_valid"] and self.output_remaining == 0
                                      and not w["legal_end"])
            self.overrun_seen |= self.error_overrun
            if w["sym_valid"]:
                self.end_state = w["new_p0"]
            if not stall_p0:
                if self.ctrl in (PROCESSING, DRAINING):
                    self.p0 = w["new_p0"] if self.sym_buf_full else self.p1
                if w["din_ready"] and din_valid:
                    self.input_buf = din & 0xFF
                    self.input_buf_full = True
                    self.byte_cycles.append(self.cycle)
                if w["sym_valid"] and sym_ready:
                    self.sym_buf_full = False
        elif not self.stall_p1:
            self._phase1(w, go)
        self.stall_p1 = nxt_stall_p1
        self.phase ^= 1
        self.cycle += 1
        return w

    def _phase1(self, w: Dict[str, object], go: bool) -> None:
        L = LOWER_LIMIT
        c = self.ctrl
        if c == IDLE:
            if go:
                self.ctrl = HDR0
                self.output_remaining = self.out_target
                self.done = False
                self.p0 = self.p1 = 0
        elif c == HDR0:
            if self.input_buf_full:
                b = self.input_buf
                self.input_remaining = 0
                self.ctrl = HDR1
                self.q = b & 0xF
                self.hdr3 = bool(b >> 5 & 1)
                self.flush_per_frag = bool(b >> 4 & 1)
                eab_u = (self.alphabet_bits - (b >> 6)) & 0xF
                self.eab = eab_u if eab_u < 10 else 9
                self.input_buf_full = False
                self._hdr_bytes = [b]
        elif c == HDR1:
            if self.input_buf_full:
                self.input_buf_full = False
                self.model = Model(min(self.eab, 8), self.q)
                self._hdr_bytes.append(self.input_buf)
                if self.hdr3:
                    self.input_remaining = self.input_buf
                    self.ctrl = HDR2
                else:
                    self.input_remaining = self.input_buf + 1
                    self.ctrl = PROCESSING
                    self.headers.append(FragmentHeader.parse(self._hdr_bytes + [0], 0)[0])
        elif c == HDR2:
            if self.input_buf_full:
                self.input_remaining = ((self.input_buf << 8 | (self.input_remaining & 0xFF)) + 257) \
                    & ((1 << MAX_INPUT_LOG2) - 1)
                self.ctrl = PROCESSING
                self.input_buf_full = False
                self._hdr_bytes.append(self.input_buf)
                self.headers.append(FragmentHeader.parse(self._hdr_bytes, 0)[0])
        elif c == PROCESSING:
            self.p1 = w["rswi"]
            self.input_remaining = w["new_in_rem"]
            self.output_remaining = w["new_out_rem"]
            self.input_buf_full = self.input_buf_full and not w["will_feed"]
            self.sym = w["new_sym"]
            self.sym_buf_full = w["next_sbf"]
            if w["next_sbf"]:
                self.symbols.append(w["new_sym"])
            if w["new_in_rem"] == 0:
                if w["new_out_rem"] > 0:
                    if w["rswi"] == L:
                        self.ctrl = HDR0
                        if self.flush_per_frag:
                            self.p1 = 0
                    else:
                        self.ctrl = DRAINING
                else:
                    self.done = True
                    self.ctrl = IDLE
        elif c == DRAINING:
            self.p1 = w["rswi"]
            self.output_remaining = w["new_out_rem"]
            self.sym = w["new_sym"]
            self.sym_buf_full = w["next_sbf"]
            if w["next_sbf"]:
                self.symbols.append(w["new_sym"])
            if w["rswi"] <= L:
                if w["new_out_rem"] > 0:
                    self.ctrl = HDR0
                    if self.flush_per_frag:
                        self.p0 = self.p1 = 0
                else:
                    self.ctrl = IDLE
                    self.done = True
        else:
            self.ctrl = IDLE



def run_lane(data: Sequence[int], alphabet_bits: int, out_target: int,
             max_cycles: int = 10_000_000) -> LaneCycleSim:
    """Run one lane with an always-valid feeder and always-ready downstream."""
    lane = LaneCycleSim(alphabet_bits, out_target)
    lane.clock()                      # phase 0 idle edge
    lane.clock(go=True)               # phase 1: IDLE -> HDR0
    pos = 0
    while not lane.done and lane.cycle < max_cycles:
        valid = pos < len(data)
        w = lane.wires(din_valid=valid)
        took = w["din_ready"] and valid and lane.phase == 0 and not w["stall_p0"]
        lane.clock(din_valid=valid, din=data[pos] if valid else 0)
        if took:
            pos += 1
        if not valid and w["need_ib_load"] and lane.phase == 1:
            break                     # starved: lane wants more bytes than it was given
    lane.clock()                      # final phase-0 edge transfers the last symbol
    lane.bytes_used = pos             # type: ignore[attr-defined]
    return lane


# ---------------------------------------------------------------------------
# Functional lane decoder (same semantics, no clocks)
# ---------------------------------------------------------------------------

@dataclass
class LaneResult:
    symbols: List[int]
    bytes_used: int
    end_state: int            # state after the last symbol's forward update
    legal_end: bool
    done: bool                # reached out_target and consumed the whole budget
    headers: List[FragmentHeader] = field(default_factory=list)
    error: str = ""


def decode_lane(data: Sequence[int], alphabet_bits: int, out_target: int) -> LaneResult:
    """Functional model of one lane fed with its own bytes in order.

    Mirrors the phase-1/phase-0 iteration of the RTL: per iteration at most one
    byte is merged (when x < L), then a symbol is produced iff x > L, then the
    state is advanced by the symbol's forward update.
    """
    L = LOWER_LIMIT
    x = 0
    pos = 0
    out: List[int] = []
    headers: List[FragmentHeader] = []
    end_state = 0
    while True:
        if pos + 2 > len(data):
            return LaneResult(out, pos, end_state, end_state in LEGAL_END_STATES, False,
                              headers, "ran out of bytes before a fragment header")
        hdr, used = FragmentHeader.parse(data, pos)
        pos += used
        headers.append(hdr)
        model = model_for(alphabet_bits, hdr.z, hdr.q)
        rem = hdr.length
        draining = False
        while True:
            # phase 1: optional byte merge
            if not draining and x < L and rem > 0:
                if pos >= len(data):
                    return LaneResult(out, pos, end_state, False, False, headers,
                                      "fragment length exceeds data")
                x = ((x & 0xFFFFFF) << 8) | data[pos]
                pos += 1
                rem -= 1
            sym = None
            if x > L and len(out) < out_target:
                sym = model.inverse(x & M16)
                out.append(sym)
            # end-of-fragment / end-of-lane decisions (FIG. 15j)
            switch = False
            if not draining:
                if rem == 0:
                    if len(out) < out_target:
                        if x == L:
                            switch = True
                        else:
                            draining = True
                    else:
                        if sym is not None:
                            end_state = model_forward_state(model, x, sym)
                        return LaneResult(out, pos, end_state, end_state in LEGAL_END_STATES,
                                          True, headers)
                elif sym is None and x >= L:
                    return LaneResult(out, pos, x, False, False, headers,
                                      "deadlock: state == L with input remaining"
                                      if x == L else "deadlock: no symbols left but input remains")
            else:
                if x <= L:
                    if len(out) < out_target:
                        switch = True
                    else:
                        return LaneResult(out, pos, end_state, end_state in LEGAL_END_STATES,
                                          True, headers)
            # phase 0: forward update
            if sym is not None:
                x = model_forward_state(model, x, sym)
                end_state = x
            if switch:
                if hdr.flush:
                    x = 0
                break


def model_forward_state(model: Model, x: int, sym: Optional[int]) -> int:
    if sym is None:
        return x
    f, cf = model.forward(sym)
    return (f * (x >> 16) + (x & M16) - cf) & M32


# ---------------------------------------------------------------------------
# Synthetic encoder (inverse of the above; not described in the patent)
# ---------------------------------------------------------------------------

@dataclass
class FragmentSpec:
    symbols: Sequence[int]
    q: int
    z: int = 0
    flush: bool = True        # reset state when this fragment ends


def encode_lane(alphabet_bits: int, frags: Sequence[FragmentSpec], trim: bool = True) -> bytes:
    """Encode fragments into one lane byte stream (headers in-band).

    Fragments joined by flush=False share one rANS stream; a boundary inside such
    a stream is only representable where the next symbol needs a refill byte.
    """
    if not frags:
        raise ValueError("need at least one fragment")
    for f in frags:
        if not f.symbols:
            raise ValueError("empty fragment")
    # group fragments into streams (state retained across flush=False boundaries)
    groups: List[List[int]] = [[]]
    for i, f in enumerate(frags):
        groups[-1].append(i)
        if f.flush and i != len(frags) - 1:
            groups.append([])
    out = bytearray()
    for g in groups:
        out += _encode_stream(alphabet_bits, [frags[i] for i in g], trim)
    return bytes(out)


def _encode_stream(alphabet_bits: int, frags: Sequence[FragmentSpec], trim: bool) -> bytes:
    seq: List[Tuple[int, Model]] = []
    sym_frag: List[int] = []
    for fi, f in enumerate(frags):
        m = model_for(alphabet_bits, f.z, f.q)
        for s in f.symbols:
            if s >> m.bits:
                raise ValueError(f"symbol {s} does not fit {m.bits} bits")
            seq.append((s, m))
            sym_frag.append(fi)
    # encode in reverse; remember how many bytes were emitted before each symbol
    x = LOWER_LIMIT
    rev = bytearray()
    emitted_before: List[int] = [0] * len(seq)   # bytes emitted (rev order) before C(sym i)
    for i in range(len(seq) - 1, -1, -1):
        s, m = seq[i]
        f, _ = m.forward(s)
        x_max = f << 16
        while x >= x_max:
            rev.append(x & 0xFF)
            x >>= 8
        emitted_before[i] = len(rev)
        x = m.encode_step(x, s, bytearray())
    ntrim = 0
    if trim:
        # bytes emitted while x was still L or L>>8 are zeros and the last ones read
        while ntrim < len(rev) and ntrim < 2 and rev[ntrim] == 0 and \
                emitted_before[-1] > ntrim:
            ntrim += 1
    stream = bytes(x.to_bytes(4, "big")) + bytes(reversed(rev[ntrim:]))
    total = len(stream)
    # R_i = decode-order byte count consumed before symbol i can be produced
    R = [4 + (len(rev) - emitted_before[i]) for i in range(len(seq))]
    # fragment byte boundaries: end of fragment k = R of its last symbol ...
    bounds: List[int] = []
    for k in range(len(frags) - 1):
        last = max(i for i, fk in enumerate(sym_frag) if fk == k)
        if R[last + 1] <= R[last]:
            raise ValueError(f"cannot split retained stream after fragment {k}: "
                             "next symbol needs no refill byte")
        bounds.append(R[last])
    bounds.append(total)
    out = bytearray()
    start = 0
    for k, f in enumerate(frags):
        n = bounds[k] - start
        hdr = FragmentHeader(f.z, n > 256, f.flush, f.q, n)
        out += hdr.encode() + stream[start:bounds[k]]
        start = bounds[k]
    return bytes(out)


# ---------------------------------------------------------------------------
# Multi-lane array with a pluggable feeder (HYPOTHESIS, not from the patent)
# ---------------------------------------------------------------------------

FeederPolicy = Callable[[List[LaneCycleSim], List[Dict[str, object]], int], List[int]]


def feeder_lane_order(lanes: List[LaneCycleSim], wires: List[Dict[str, object]],
                      window: int = FEED_WINDOW) -> List[int]:
    """Grant up to `window` requesting lanes per phase-0, lowest lane index first."""
    return [i for i, w in enumerate(wires) if w["din_req"]][:window]


class ArraySim:
    """Lanes clocked together; the feeder hands out bytes from ONE shared stream.

    `policy(lanes, wires, window)` returns the lane indices that receive the
    next bytes of the shared stream on this phase-0 edge, in stream order.
    Downstream is always ready unless `symvec` is given (callable returning a
    per-lane sym_ready list).
    """

    def __init__(self, alphabet_bits: Sequence[int], out_targets: Sequence[int],
                 policy: FeederPolicy = feeder_lane_order, window: int = FEED_WINDOW):
        self.lanes = [LaneCycleSim(a, n) for a, n in zip(alphabet_bits, out_targets)]
        self.policy = policy
        self.window = window
        self.schedule: List[Tuple[int, int]] = []   # (cycle, lane) per shared byte

    def run(self, stream: Optional[Sequence[int]] = None,
            per_lane: Optional[Sequence[Sequence[int]]] = None,
            max_cycles: int = 5_000_000) -> int:
        """Feed from one shared `stream`, or from `per_lane` byte lists (to derive
        the interleave a policy would produce). Returns bytes handed out."""
        for ln in self.lanes:
            ln.clock()
            ln.clock(go=True)
        pos = 0
        cur = [0] * len(self.lanes)
        cyc = 0
        while not all(ln.done for ln in self.lanes) and cyc < max_cycles:
            if self.lanes[0].phase == 0:
                wires = [ln.wires(din_valid=True) for ln in self.lanes]
                grant = self.policy(self.lanes, wires, self.window)
                feed: Dict[int, int] = {}
                for i in grant:
                    if per_lane is not None:
                        if cur[i] < len(per_lane[i]):
                            feed[i] = per_lane[i][cur[i]]
                            cur[i] += 1
                            self.schedule.append((cyc, i))
                            pos += 1
                    elif stream is not None and pos < len(stream):
                        feed[i] = stream[pos]
                        self.schedule.append((cyc, i))
                        pos += 1
                for i, ln in enumerate(self.lanes):
                    # a lane not granted this edge sees din_valid = 0 (it stalls)
                    req = wires[i]["need_ib_load"]
                    ln.clock(din_valid=(i in feed) or not req, din=feed.get(i, 0))
            else:
                for ln in self.lanes:
                    ln.clock()
            cyc += 1
        return pos


# ---------------------------------------------------------------------------
# Self-tests (synthetic data only)
# ---------------------------------------------------------------------------

def _rand_symbols(rng: random.Random, m: Model, n: int) -> List[int]:
    """Draw symbols from the model's own distribution (skip zero-frequency ones)."""
    out = []
    top = 1 << m.bits
    while len(out) < n:
        s = m.inverse(rng.randrange(1, 1 << 16))
        if s < top and m.forward(s)[0] > 0:
            out.append(s)
    return out


def _check_lane(data: bytes, ab: int, syms: List[int]) -> None:
    r = decode_lane(data, ab, len(syms))
    assert r.error == "", r.error
    assert r.symbols == syms, "functional decode mismatch"
    assert r.bytes_used == len(data), (r.bytes_used, len(data))
    assert r.done and r.legal_end, hex(r.end_state)
    c = run_lane(data, ab, len(syms))
    assert c.done, "cycle model did not finish"
    assert c.symbols == syms, "cycle decode mismatch"
    assert c.bytes_used == len(data)  # type: ignore[attr-defined]
    assert not c.overrun_seen, "error_overrun raised"
    assert c.end_state == r.end_state


def selftest(seed: int = 1, n_cases: int = 300) -> None:
    rng = random.Random(seed)
    # 1. table sanity + exact reciprocal division for every slot of every model
    for eab in range(9):
        for q in range(16):
            m = Model(eab, q)
            assert sum(m.freq[k] * (1 if k == 0 else 1 << (k - 1))
                       for k in range(m.bits + 1)) == 1 << 16
            for cf in range(1, 1 << 16):
                s = m.inverse(cf)
                f, c = m.forward(s)
                assert c <= cf < c + f, (eab, q, cf, s, f, c)
    print("models: 144/144 exact (every cf in 1..65535 maps into its symbol's slot)")
    # 2. single-fragment round trips, every (eab, q)
    n = 0
    for eab in range(9):
        for q in range(16):
            m = Model(eab, q)
            syms = _rand_symbols(rng, m, rng.randrange(1, 400))
            data = encode_lane(eab, [FragmentSpec(syms, q)])
            _check_lane(data, eab, syms)
            n += 1
    print(f"single-fragment round trips: {n}/{n}")
    # 3. multi-fragment lanes: random z, q, flush, long (3-byte header) fragments
    ok = 0
    tried = 0
    for _ in range(n_cases):
        ab = rng.randrange(1, 9)
        frags = []
        for k in range(rng.randrange(1, 5)):
            z = rng.randrange(0, min(3, ab) + 1)
            q = rng.randrange(16)
            m = model_for(ab, z, q)
            cnt = rng.choice((1, 2, 5, 50, 700, 3000))
            frags.append(FragmentSpec(_rand_symbols(rng, m, cnt), q, z, rng.random() < 0.6))
        try:
            data = encode_lane(ab, frags)
        except ValueError as e:
            if "cannot split" in str(e):
                continue                       # unrepresentable retained boundary
            raise
        tried += 1
        syms = [s for f in frags for s in f.symbols]
        _check_lane(data, ab, syms)
        ok += 1
    print(f"multi-fragment round trips (functional == cycle model): {ok}/{tried}")
    # 4. corrupted byte budget must fail the convergence check
    m = Model(7, 11)
    syms = _rand_symbols(rng, m, 500)
    data = bytearray(encode_lane(7, [FragmentSpec(syms, 11)]))
    data[len(data) // 2] ^= 0x5A
    r = decode_lane(bytes(data), 7, len(syms))
    assert not (r.done and r.legal_end and r.symbols == syms)
    print("corruption detected: ok")
    # 5. two lanes behind one shared stream with the lane-order feeder hypothesis
    specs = [(7, [FragmentSpec(_rand_symbols(rng, Model(7, 12), 64), 12)]),
             (4, [FragmentSpec(_rand_symbols(rng, Model(4, 3), 64), 3)])]
    per_lane = [encode_lane(ab, fr) for ab, fr in specs]
    # derive the interleave the policy produces, then decode from the shared stream
    probe = ArraySim([s[0] for s in specs], [64, 64])
    probe.run(per_lane=per_lane)
    cur = [0, 0]
    shared = bytearray()
    for _, i in probe.schedule:
        shared.append(per_lane[i][cur[i]])
        cur[i] += 1
    arr = ArraySim([s[0] for s in specs], [64, 64])
    used = arr.run(bytes(shared))
    assert used == len(shared)
    for ln, (_, fr) in zip(arr.lanes, specs):
        assert ln.symbols == list(fr[0].symbols) and ln.done
    print("2-lane shared-stream round trip via lane-order feeder: ok")


if __name__ == "__main__":
    selftest()
