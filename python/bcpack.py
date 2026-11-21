"""Clean-room Xbox Series X|S MDU BCPack rANS entropy codec and BC1-BC7 stream decoder.

Runs under `python -I` using only the Python standard library.
Licensed under MIT OR Apache-2.0. Zero game data included.
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

RANS_M = 1 << 16
RANS_L = 1 << 24
LEGAL_END_STATES = (RANS_L, RANS_L >> 8, RANS_L >> 16)

MODEL_LAMBDAS: Tuple[float, ...] = (
    0.0,        # q = 0 (uniform)
    0.739926,   # q = 1
    1.496350,   # q = 2
    2.307231,   # q = 3
    3.221322,   # q = 4
    4.307485,   # q = 5
    5.676284,   # q = 6
    7.508501,   # q = 7
    10.112494,  # q = 8
    14.036894,  # q = 9
    20.348947,  # q = 10
    31.445855,  # q = 11
    53.681684,  # q = 12
    109.260000, # q = 13
    328.115610, # q = 14
    1.0e9,      # q = 15 (all non-zero symbols frequency 1)
)

EXACT_TABLES: Dict[Tuple[int, int], Tuple[int, ...]] = {
    (2, 1): (21168, 17592, 13388),
    (4, 1): (5665, 5407, 5048, 4398, 3347),
    (2, 2): (26354, 18130, 10526),
    (3, 2): (14406, 11948, 9065, 5263),
    (4, 2): (7539, 6867, 5975, 4531, 2632),
    (2, 3): (31900, 17918, 7859),
    (3, 4): (22624, 15126, 8437, 2728),
    (2, 5): (43800, 14920, 3408),
    (3, 6): (33414, 16436, 6031, 906),
    (4, 7): (24559, 15361, 7808, 2124, 188),
    (2, 8): (60308, 4812, 208),
    (2, 9): (63575, 1901, 30),
    (3, 10): (60388, 4744, 200, 1),
    (3, 11): (64250, 1262, 10, 1),
    (8, 11): (7575, 6701, 5582, 3892, 1918, 494, 34, 1, 1),
    (3, 12): (65456, 74, 1, 1),
    (4, 12): (63248, 2208, 34, 1, 1),
    (5, 12): (53291, 9957, 1104, 14, 1, 1),
    (5, 13): (63380, 2084, 22, 1, 1, 1),
    (6, 13): (53649, 9731, 1042, 4, 1, 1, 1),
    (6, 14): (65147, 327, 1, 1, 1, 1, 1),
}

BC1_LANE_AB: Tuple[int, ...] = (8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8)
BC4_LANE_AB: Tuple[int, ...] = (8, 4, 8, 8, 8, 8, 8, 8)
BC5_LANE_AB: Tuple[int, ...] = (8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8)


def is_legal_end_state(x: int) -> bool:
    return x in LEGAL_END_STATES


@dataclass(frozen=True)
class FragmentHeader:
    raw_mode: int
    z_adjust: int
    hdr3: bool
    flush_per_frag: bool
    model_q: int
    payload_bytes: int
    header_bytes: int

    @staticmethod
    def parse_mode_byte(b0: int) -> Tuple[int, bool, bool, int]:
        z = (b0 >> 6) & 0x03
        hdr3 = bool((b0 >> 5) & 0x01)
        flush = bool((b0 >> 4) & 0x01)
        q = b0 & 0x0F
        return z, hdr3, flush, q

    @classmethod
    def from_bytes(cls, b0: int, b1: int, b2: Optional[int] = None) -> "FragmentHeader":
        z, hdr3, flush, q = cls.parse_mode_byte(b0)
        if hdr3:
            if b2 is None:
                raise ValueError("hdr3 fragment requires 3rd header byte")
            length = (b1 | (b2 << 8)) + 257
            return cls(b0, z, True, flush, q, length, 3)
        return cls(b0, z, False, flush, q, b1 + 1, 2)

    @staticmethod
    def encode_header(z_adjust: int, flush_per_frag: bool, model_q: int, payload_bytes: int) -> bytes:
        if not (1 <= payload_bytes <= 65792):
            raise ValueError(f"invalid payload_bytes: {payload_bytes}")
        hdr3 = payload_bytes > 256
        b0 = ((z_adjust & 3) << 6) | ((1 if hdr3 else 0) << 5) | ((1 if flush_per_frag else 0) << 4) | (model_q & 0x0F)
        if hdr3:
            v = payload_bytes - 257
            return bytes((b0, v & 0xFF, (v >> 8) & 0xFF))
        return bytes((b0, payload_bytes - 1))


@dataclass(frozen=True)
class ProbabilityModel:
    eab: int
    q: int
    freqs: Tuple[int, ...]
    bases: Tuple[int, ...]

    @classmethod
    def create(cls, eab: int, q: int) -> "ProbabilityModel":
        eab = max(1, min(8, eab))
        q = q & 0x0F
        counts = [1] + [1 << (k - 1) for k in range(1, eab + 1)]
        if q == 0:
            f = RANS_M >> eab
            freqs = [f] * (eab + 1)
        elif (
            q == 15
            or (eab == 1 and q >= 11)
            or (eab == 2 and q >= 12)
            or (eab == 3 and q >= 13)
            or (eab in (4, 5) and q >= 14)
        ):
            freqs = [RANS_M - ((1 << eab) - 1)] + [1] * eab
        elif (eab, q) in EXACT_TABLES:
            freqs = list(EXACT_TABLES[(eab, q)])
        else:
            lam = MODEL_LAMBDAS[q]
            n_syms = float(1 << eab)
            denom = 1.0 - math.exp(-lam)
            raw_b = [0.0]
            for k in range(1, eab + 1):
                t = float(1 << (k - 1)) / n_syms
                cdf = (1.0 - math.exp(-lam * t)) / denom
                raw_b.append(cdf * 65536.0)
            act_top = 65536
            freqs = [0] * (eab + 1)
            for k in range(eab, 0, -1):
                c = counts[k]
                f = max(1, int(round((act_top - raw_b[k]) / c)))
                freqs[k] = f
                act_top -= f * c
            freqs[0] = max(2, act_top)

        bases = []
        acc = 0
        for f, c in zip(freqs, counts):
            bases.append(acc)
            acc += f * c
        assert acc == RANS_M, f"model sum mismatch: {acc}"
        return cls(eab, q, tuple(freqs), tuple(bases))

    def forward_params(self, sym: int) -> Tuple[int, int]:
        if sym == 0:
            return self.freqs[0] - 1, 1
        seg = sym.bit_length()
        step = sym - (1 << (seg - 1))
        fa = self.freqs[seg]
        return fa, self.bases[seg] + fa * step

    def inverse_lookup(self, cf_in: int) -> Tuple[int, int, int]:
        if cf_in <= 0:
            raise ValueError("invalid rANS slot 0")
        seg = 0
        for k in range(len(self.bases) - 1, -1, -1):
            if cf_in >= self.bases[k]:
                seg = k
                break
        if seg == 0:
            return 0, self.freqs[0] - 1, 1
        b = self.bases[seg]
        fa = self.freqs[seg]
        step = (cf_in - b) // fa
        sym = (1 << (seg - 1)) + step
        return sym, fa, b + step * fa

    def encode_fragment(self, symbols: Sequence[int]) -> bytes:
        x = RANS_L
        renorm_rev = bytearray()
        for sym in reversed(symbols):
            f, cf = self.forward_params(sym)
            max_x = ((RANS_L >> 16) << 8) * f
            while x >= max_x:
                renorm_rev.append(x & 0xFF)
                x >>= 8
            x = ((x // f) << 16) + (x % f) + cf
        renorm_rev.reverse()
        return struct.pack(">I", x) + bytes(renorm_rev)

    def decode_fragment(self, payload: bytes, max_symbols: int = 65536) -> Tuple[List[int], int]:
        if len(payload) < 4:
            raise ValueError("payload shorter than 4 bytes")
        x = struct.unpack_from(">I", payload, 0)[0]
        pos = 4
        out: List[int] = []
        while len(out) < max_symbols:
            while x < RANS_L and pos < len(payload):
                x = ((x & 0xFFFFFF) << 8) | payload[pos]
                pos += 1
            if pos == len(payload) and is_legal_end_state(x):
                return out, x
            if x < RANS_L:
                raise ValueError("payload underflow before legal end state")
            cf_in = x & 0xFFFF
            sym, fwd_f, fwd_cf = self.inverse_lookup(cf_in)
            out.append(sym)
            x = fwd_f * (x >> 16) + cf_in - fwd_cf
        while x < RANS_L and pos < len(payload):
            x = ((x & 0xFFFFFF) << 8) | payload[pos]
            pos += 1
        if pos == len(payload) and is_legal_end_state(x):
            return out, x
        raise ValueError("fragment did not converge within max_symbols")


_MODEL_CACHE: Dict[Tuple[int, int], ProbabilityModel] = {
    (eab, q): ProbabilityModel.create(eab, q) for eab in range(1, 9) for q in range(16)
}


def get_model(eab: int, q: int) -> ProbabilityModel:
    return _MODEL_CACHE[(max(1, min(8, eab)), q & 0x0F)]


IDLE, HDR0, HDR1, HDR2, PROC, DRAIN = 0, 1, 2, 3, 4, 5


@dataclass
class RansLane:
    lane_index: int
    alphabet_bits: int
    output_remaining: int
    ctrl_state: int = IDLE
    input_buf: int = 0
    input_buf_full: bool = False
    sym_buf: int = 0
    sym_buf_full: bool = False
    input_remaining: int = 0
    rans_state_p0: int = 0
    rans_state_p1: int = 0
    hdr3: bool = False
    flush_per_frag: bool = False
    eab: int = 8
    current_q: int = 0
    model: ProbabilityModel = field(default_factory=lambda: get_model(8, 0))
    done: bool = False
    decoded_symbols: List[int] = field(default_factory=list)
    fragments: List[FragmentHeader] = field(default_factory=list)
    _hdr0_byte: int = 0
    _hdr1_byte: int = 0

    def launch(self) -> None:
        self.ctrl_state = HDR0
        self.rans_state_p0 = 0
        self.rans_state_p1 = 0
        self.done = False

    def need_ib_load(self) -> bool:
        if self.input_buf_full:
            return False
        if self.ctrl_state in (HDR0, HDR1, HDR2):
            return True
        if self.ctrl_state == PROC:
            return self.input_remaining > (1 if self.input_buf_full else 0)
        return False

    def step_phase0(self, stream: bytes, pos: int, sym_ready: bool) -> int:
        blocked = self.sym_buf_full and not sym_ready
        if not blocked:
            if self.ctrl_state in (PROC, DRAIN) or self.sym_buf_full:
                if self.sym_buf_full:
                    fwd_f, fwd_cf = self.model.forward_params(self.sym_buf)
                    fwd_p = self.rans_state_p1 & 0xFFFF
                    self.rans_state_p0 = (fwd_f * (self.rans_state_p1 >> 16) + fwd_p - fwd_cf) & 0xFFFFFFFF
                    if self.done:
                        self.rans_state_p1 = self.rans_state_p0
                else:
                    self.rans_state_p0 = self.rans_state_p1
            if self.sym_buf_full and sym_ready:
                self.decoded_symbols.append(self.sym_buf)
                self.sym_buf_full = False
        if self.need_ib_load() and pos < len(stream):
            self.input_buf = stream[pos]
            self.input_buf_full = True
            pos += 1
        return pos

    def step_phase1(self, stalled: bool) -> None:
        if stalled or self.ctrl_state == IDLE:
            return
        if self.ctrl_state == HDR0:
            if self.input_buf_full:
                b = self.input_buf
                self._hdr0_byte = b
                self.input_remaining = 0
                self.ctrl_state = HDR1
                z, hdr3, flush, q = FragmentHeader.parse_mode_byte(b)
                self.current_q = q
                self.hdr3 = hdr3
                self.flush_per_frag = flush
                self.eab = max(1, min(8, self.alphabet_bits - z))
                self.input_buf_full = False
        elif self.ctrl_state == HDR1:
            if self.input_buf_full:
                b = self.input_buf
                self._hdr1_byte = b
                self.input_buf_full = False
                self.model = get_model(self.eab, self.current_q)
                if self.hdr3:
                    self.input_remaining = b
                    self.ctrl_state = HDR2
                else:
                    self.input_remaining = b + 1
                    self.fragments.append(FragmentHeader.from_bytes(self._hdr0_byte, b, None))
                    self.ctrl_state = PROC
        elif self.ctrl_state == HDR2:
            if self.input_buf_full:
                b = self.input_buf
                self.input_buf_full = False
                self.input_remaining = ((b << 8) | (self.input_remaining & 0xFF)) + 257
                self.fragments.append(FragmentHeader.from_bytes(self._hdr0_byte, self._hdr1_byte, b))
                self.ctrl_state = PROC
        elif self.ctrl_state in (PROC, DRAIN):
            is_proc = self.ctrl_state == PROC
            want_to_feed = is_proc and (self.rans_state_p0 < RANS_L) and (self.input_remaining > 0)
            will_feed = want_to_feed and self.input_buf_full
            if will_feed:
                st_in = ((self.rans_state_p0 & 0xFFFFFF) << 8) | self.input_buf
                new_in_rem = self.input_remaining - 1
            else:
                st_in = self.rans_state_p0
                new_in_rem = self.input_remaining

            cf_in = st_in & 0xFFFF
            new_sym = self.model.inverse_lookup(cf_in)[0] if cf_in > 0 else 0
            next_sym_full = (st_in > RANS_L) and (self.output_remaining > 0)
            new_out_rem = self.output_remaining - 1 if next_sym_full else self.output_remaining

            self.rans_state_p1 = st_in
            self.output_remaining = new_out_rem
            self.sym_buf = new_sym
            self.sym_buf_full = next_sym_full

            if is_proc:
                self.input_remaining = new_in_rem
                if will_feed:
                    self.input_buf_full = False
                if new_in_rem == 0:
                    if new_out_rem > 0:
                        if st_in == RANS_L:
                            self.ctrl_state = HDR0
                            if self.flush_per_frag:
                                self.rans_state_p1 = 0
                        else:
                            self.ctrl_state = DRAIN
                    else:
                        self.done = True
                        self.ctrl_state = IDLE
            elif st_in <= RANS_L:
                if new_out_rem > 0:
                    self.ctrl_state = HDR0
                    if self.flush_per_frag:
                        self.rans_state_p0 = 0
                        self.rans_state_p1 = 0
                else:
                    self.ctrl_state = IDLE
                    self.done = True


def parse_main_8lane_header(stream: bytes) -> List[FragmentHeader]:
    if len(stream) < 48:
        raise ValueError("stream shorter than 48-byte main header")
    return [FragmentHeader.from_bytes(stream[i], stream[8 + i], stream[16 + i]) for i in range(8)]


def decode_multilane_stream(
    stream: bytes, alphabet_bits: Sequence[int], symbols_per_lane: int
) -> List[RansLane]:
    lanes = [RansLane(i, ab, symbols_per_lane) for i, ab in enumerate(alphabet_bits)]
    for l in lanes[:8]:
        l.launch()
    pos = 0
    max_rounds = symbols_per_lane * 6 + 128
    for rnd in range(max_rounds):
        if rnd == 6:
            for l in lanes[8:]:
                l.launch()
        all_ready = all(l.ctrl_state == IDLE or l.done or l.sym_buf_full for l in lanes)
        stalled = [l.sym_buf_full and not all_ready for l in lanes]
        for l in lanes:
            pos = l.step_phase0(stream, pos, all_ready)
        for l, st in zip(lanes, stalled):
            l.step_phase1(st)
        if all(l.done and not l.sym_buf_full for l in lanes):
            return lanes
    raise ValueError("multi-lane stream did not finish within max_rounds")


def encode_multilane_stream(lanes_spec: Sequence[Tuple[int, int, int, Sequence[int]]]) -> bytes:
    records: List[bytes] = []
    for ab, z, q, syms in lanes_spec:
        eab = max(1, min(8, ab - z))
        m = get_model(eab, q)
        payload = m.encode_fragment(syms)
        hdr = FragmentHeader.encode_header(z, True, q, len(payload))
        records.append(hdr + payload)

    n_lanes = len(lanes_spec)
    out_target = len(lanes_spec[0][3])
    lanes = [RansLane(i, ab, len(syms)) for i, (ab, _, _, syms) in enumerate(lanes_spec)]
    lane_pos = [0] * n_lanes
    out = bytearray()

    for l in lanes[:8]:
        l.launch()

    for rnd in range(out_target * 6 + 128):
        if rnd == 6:
            for l in lanes[8:]:
                l.launch()
        all_ready = all(l.ctrl_state == IDLE or l.done or l.sym_buf_full for l in lanes)
        stalled = [l.sym_buf_full and not all_ready for l in lanes]
        for i, l in enumerate(lanes):
            blocked = l.sym_buf_full and not all_ready
            if not blocked:
                if l.ctrl_state in (PROC, DRAIN) or l.sym_buf_full:
                    if l.sym_buf_full:
                        fwd_f, fwd_cf = l.model.forward_params(l.sym_buf)
                        fwd_p = l.rans_state_p1 & 0xFFFF
                        l.rans_state_p0 = (fwd_f * (l.rans_state_p1 >> 16) + fwd_p - fwd_cf) & 0xFFFFFFFF
                        if l.done:
                            l.rans_state_p1 = l.rans_state_p0
                    else:
                        l.rans_state_p0 = l.rans_state_p1
                if l.sym_buf_full and all_ready:
                    l.decoded_symbols.append(l.sym_buf)
                    l.sym_buf_full = False
            if l.need_ib_load() and lane_pos[i] < len(records[i]):
                b = records[i][lane_pos[i]]
                lane_pos[i] += 1
                out.append(b)
                l.input_buf = b
                l.input_buf_full = True
        for l, st in zip(lanes, stalled):
            l.step_phase1(st)
        if all(l.done and not l.sym_buf_full for l in lanes):
            break
    return bytes(out)


def assemble_bc1_blocks(lanes: Sequence[RansLane], num_blocks: int) -> bytes:
    out = bytearray(num_blocks * 8)
    r0 = g0 = b0 = r1 = g1 = b1 = 0
    for i in range(num_blocks):
        r0 = (r0 + lanes[2].decoded_symbols[i]) & 0x1F
        g0 = (g0 + lanes[3].decoded_symbols[i]) & 0x3F
        b0 = (b0 + lanes[4].decoded_symbols[i]) & 0x1F
        r1 = (r1 + lanes[5].decoded_symbols[i]) & 0x1F
        g1 = (g1 + lanes[6].decoded_symbols[i]) & 0x3F
        b1 = (b1 + lanes[7].decoded_symbols[i]) & 0x1F
        c0 = (r0 << 11) | (g0 << 5) | b0
        c1 = (r1 << 11) | (g1 << 5) | b1
        mode_bit = lanes[0].decoded_symbols[i] & 1
        if (mode_bit == 0 and c0 < c1) or (mode_bit == 1 and c0 > c1):
            c0, c1 = c1, c0
        struct.pack_into(
            "<HHBBBB",
            out,
            i * 8,
            c0,
            c1,
            lanes[8].decoded_symbols[i] & 0xFF,
            lanes[9].decoded_symbols[i] & 0xFF,
            lanes[10].decoded_symbols[i] & 0xFF,
            lanes[11].decoded_symbols[i] & 0xFF,
        )
    return bytes(out)


class TestBCPack(unittest.TestCase):
    def test_all_models_sum_to_65536(self) -> None:
        for eab in range(1, 9):
            for q in range(16):
                m = get_model(eab, q)
                counts = [1] + [1 << (k - 1) for k in range(1, eab + 1)]
                self.assertEqual(sum(f * c for f, c in zip(m.freqs, counts)), RANS_M)

    def test_patent_table_exact(self) -> None:
        m = get_model(8, 11)
        self.assertEqual(m.freqs, (7575, 6701, 5582, 3892, 1918, 494, 34, 1, 1))
        self.assertEqual(m.bases, (0, 7575, 14276, 25440, 41008, 56352, 64256, 65344, 65408))

    def test_canonical_constant_states(self) -> None:
        for eab, q, n, exp_st in (
            (2, 12, 1024, 0x01107ED4),
            (2, 12, 1536, 0x01192334),
            (2, 12, 4096, 0x0148A474),
            (3, 13, 1536, 0x0134B898),
            (5, 14, 1536, 0x021D88C0),
            (5, 14, 4096, 0x07618900),
            (1, 11, 8192, 0x0148B64C),
        ):
            m = get_model(eab, q)
            p = m.encode_fragment([0] * n)
            self.assertEqual(struct.unpack(">I", p)[0], exp_st)
            dec, end_x = m.decode_fragment(p, n)
            self.assertEqual(len(dec), n)
            self.assertEqual(end_x, RANS_L)

    def test_multilane_bc1_roundtrip(self) -> None:
        n = 32
        specs = [
            (8, 0, 8, [i % 4 for i in range(n)]),
            (8, 3, 14, [0] * n),
            (5, 3, 12, [0] * n),
            (6, 1, 10, [i % 5 for i in range(n)]),
            (5, 3, 12, [0] * n),
            (5, 2, 11, [i % 3 for i in range(n)]),
            (6, 3, 13, [0] * n),
            (5, 3, 12, [0] * n),
            (8, 0, 2, [(i * 13) & 0xFF for i in range(n)]),
            (8, 0, 2, [(i * 29) & 0xFF for i in range(n)]),
            (8, 0, 2, [(i * 53) & 0xFF for i in range(n)]),
            (8, 0, 2, [(i * 97) & 0xFF for i in range(n)]),
        ]
        stream = encode_multilane_stream(specs)
        lanes = decode_multilane_stream(stream, BC1_LANE_AB, n)
        for i, l in enumerate(lanes):
            self.assertEqual(l.decoded_symbols, specs[i][3])
            self.assertTrue(is_legal_end_state(l.rans_state_p1))
        bc1 = assemble_bc1_blocks(lanes, n)
        self.assertEqual(len(bc1), n * 8)


def main() -> int:
    ap = argparse.ArgumentParser(description="Clean-room Xbox Series BCPack rANS decoder")
    ap.add_argument("--selftest", action="store_true", help="Run synthetic unit tests")
    a = ap.parse_args()
    if a.selftest:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestBCPack)
        res = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if res.wasSuccessful() else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
