//! Clean-room Xbox Series X|S Media Decompression Unit (MDU) **BCPack** rANS entropy
//! codec and block-compressed texture (`BC1`..`BC7`) stream parser/decoder.
//!
//! Licensed under `MIT OR Apache-2.0`. Zero game data or proprietary assets included.

/// rANS probability precision $M = 2^{16} = 65536$.
pub const RANS_M: u32 = 1 << 16;

/// rANS lower state normalization limit $L = 2^{24} = 0x0100_0000$.
pub const RANS_L: u32 = 1 << 24;

/// Early-termination / trimmed rANS end states permitted by the MDU hardware decoder:
/// `L` (`0x0100_0000`), `L >> 8` (`0x0001_0000`), and `L >> 16` (`0x0000_0100`).
#[inline]
pub fn is_legal_end_state(x: u32) -> bool {
    x == RANS_L || x == (RANS_L >> 8) || x == (RANS_L >> 16)
}

/// BCPack chunk codec identifier stored in TTPF chunk table bits 24..29 (`BC_num * 4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BcFormat {
    Bc1 = 0x04,
    Bc2 = 0x08,
    Bc3 = 0x0C,
    Bc4 = 0x10,
    Bc5 = 0x14,
    Bc6h = 0x18,
    Bc7 = 0x1C,
}

impl BcFormat {
    pub fn from_codec_id(id: u8) -> Option<Self> {
        match id {
            0x04 => Some(Self::Bc1),
            0x08 => Some(Self::Bc2),
            0x0C => Some(Self::Bc3),
            0x10 => Some(Self::Bc4),
            0x14 => Some(Self::Bc5),
            0x18 => Some(Self::Bc6h),
            0x1C => Some(Self::Bc7),
            _ => None,
        }
    }

    /// Bytes per 4x4 compressed BCn block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Bc1 | Self::Bc4 => 8,
            Self::Bc2 | Self::Bc3 | Self::Bc5 | Self::Bc6h | Self::Bc7 => 16,
        }
    }

    /// Default per-lane symbol widths (`alphabet_bits`) for the 8 main lanes and index lanes.
    pub fn lane_alphabet_bits(self) -> &'static [u8] {
        match self {
            // BC1: L0 (8b mode/ctrl), L1 (8b const), L2..L7 (R0=5, G0=6, B0=5, R1=5, G1=6, B1=5), L8..L11 (4x 8b index rows)
            Self::Bc1 => &[8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8],
            // BC4: L0..L7 (endpoints + 3-bit index groups)
            Self::Bc4 => &[8, 4, 8, 8, 8, 8, 8, 8],
            // BC5: two BC4 channels across 12 lanes (L0..L3 endpoints/ctrl + 8 index lanes)
            Self::Bc5 => &[8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            // BC3 / BC2 / BC6H / BC7 default 8 main lanes + 4 secondary lanes
            Self::Bc2 | Self::Bc3 | Self::Bc6h | Self::Bc7 => &[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        }
    }
}

/// Decoded 1-byte fragment mode header (`DSTATE_HDR0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragmentHeader {
    /// Raw mode byte (`hdr0`).
    pub raw_mode: u8,
    /// Symbol width reduction `z = hdr0[7:6]` (`0..=3`), giving `eab = alphabet_bits - z`.
    pub z_adjust: u8,
    /// Extra length byte flag `hdr3 = hdr0[5]`:
    /// - `false` (2-byte header): `payload_bytes = b1 + 1` (`1..=256`).
    /// - `true`  (3-byte header): `payload_bytes = u16::from_le_bytes([b1, b2]) + 257` (`257..=65792`).
    pub hdr3: bool,
    /// State flush flag `flush_per_frag = hdr0[4]`: if true, flush/reload 32-bit state on fragment transition.
    pub flush_per_frag: bool,
    /// Static probability model index `current_q = hdr0[3:0]` (`0..=15`).
    pub model_q: u8,
    /// Total encoded payload bytes for this fragment (including 4 initial state bytes when flushed).
    pub payload_bytes: usize,
    /// Header length in bytes (`2` if `!hdr3`, `3` if `hdr3`).
    pub header_bytes: usize,
}

impl FragmentHeader {
    pub fn parse_mode_byte(b0: u8) -> (u8, bool, bool, u8) {
        let z_adjust = (b0 >> 6) & 0x03;
        let hdr3 = ((b0 >> 5) & 0x01) != 0;
        let flush_per_frag = ((b0 >> 4) & 0x01) != 0;
        let model_q = b0 & 0x0F;
        (z_adjust, hdr3, flush_per_frag, model_q)
    }

    pub fn from_bytes(b0: u8, b1: u8, b2: Option<u8>) -> Option<Self> {
        let (z_adjust, hdr3, flush_per_frag, model_q) = Self::parse_mode_byte(b0);
        if hdr3 {
            let hi = b2?;
            let len = (u16::from_le_bytes([b1, hi]) as usize) + 257;
            Some(Self {
                raw_mode: b0,
                z_adjust,
                hdr3,
                flush_per_frag,
                model_q,
                payload_bytes: len,
                header_bytes: 3,
            })
        } else {
            Some(Self {
                raw_mode: b0,
                z_adjust,
                hdr3,
                flush_per_frag,
                model_q,
                payload_bytes: (b1 as usize) + 1,
                header_bytes: 2,
            })
        }
    }

    pub fn encode_header(z_adjust: u8, flush_per_frag: bool, model_q: u8, payload_bytes: usize) -> Vec<u8> {
        assert!((1..=65792).contains(&payload_bytes), "invalid fragment payload length");
        let hdr3 = payload_bytes > 256;
        let b0 = ((z_adjust & 0x03) << 6)
            | ((hdr3 as u8) << 5)
            | ((flush_per_frag as u8) << 4)
            | (model_q & 0x0F);
        if hdr3 {
            let v = (payload_bytes - 257) as u16;
            let [lo, hi] = v.to_le_bytes();
            vec![b0, lo, hi]
        } else {
            vec![b0, (payload_bytes - 1) as u8]
        }
    }
}

/// Power-of-two segmented probability model (`base_table[eab][q]` and `freq_table[eab][q]`).
///
/// For effective alphabet width `eab` (`1..=8`, alphabet size $2^{\text{eab}}$), symbols are grouped
/// into `eab + 1` power-of-two segments:
/// - Segment 0: symbol `0` (count = 1)
/// - Segment 1: symbol `1` (count = 1)
/// - Segment $k$ ($k \ge 1$): symbols $[2^{k-1}, 2^k - 1]$ (count = $2^{k-1}$)
///
/// Within each segment $k$, every symbol shares frequency `freqs[k]`.
/// For symbol `0` (`fwd_cfa == 0`), the hardware reserves slot `0` so `fwd_cf = 1` and
/// `fwd_f = freqs[0] - 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbabilityModel {
    pub eab: u8,
    pub q: u8,
    pub freqs: Vec<u16>,
    pub bases: Vec<u16>,
}

/// Recovered exponential decay parameters $\lambda(q)$ across the 16 static probability models `q = 0..15`.
pub const MODEL_LAMBDAS: [f64; 16] = [
    0.0,        // q = 0 (uniform)
    0.739926,   // q = 1
    1.496350,   // q = 2
    2.307231,   // q = 3
    3.221322,   // q = 4
    4.307485,   // q = 5
    5.676284,   // q = 6
    7.508501,   // q = 7
    10.112494,  // q = 8
    14.036894,  // q = 9
    20.348947,  // q = 10
    31.445855,  // q = 11
    53.681684,  // q = 12
    109.260000, // q = 13
    328.115610, // q = 14
    1.0e9,      // q = 15 (all non-zero symbols frequency 1)
];

impl ProbabilityModel {
    /// Construct the static probability model for `(eab, q)` (`eab in 1..=8`, `q in 0..=15`).
    pub fn new(eab: u8, q: u8) -> Self {
        let eab = eab.clamp(1, 8);
        let q = q & 0x0F;
        let n_segs = (eab as usize) + 1;
        let mut counts = Vec::with_capacity(n_segs);
        counts.push(1u32);
        for k in 1..=eab {
            counts.push(1u32 << (k - 1));
        }

        let freqs_u16: Vec<u16> = if q == 0 {
            let f = (RANS_M >> eab) as u16;
            vec![f; n_segs]
        } else if q == 15
            || (eab == 1 && q >= 11)
            || (eab == 2 && q >= 12)
            || (eab == 3 && q >= 13)
            || ((eab == 4 || eab == 5) && q >= 14)
        {
            let mut v = vec![1u16; n_segs];
            v[0] = (RANS_M - ((1u32 << eab) - 1)) as u16;
            v
        } else if let Some(exact) = Self::exact_override(eab, q) {
            exact.to_vec()
        } else {
            let lam = MODEL_LAMBDAS[q as usize];
            let n_syms = (1u32 << eab) as f64;
            let denom = 1.0 - (-lam).exp();
            let mut raw_b = vec![0.0f64; n_segs];
            for k in 1..n_segs {
                let t = ((1u32 << (k - 1)) as f64) / n_syms;
                let cdf = (1.0 - (-lam * t).exp()) / denom;
                raw_b[k] = cdf * 65536.0;
            }
            let mut act_top = 65536i32;
            let mut fs = vec![0u16; n_segs];
            for k in (1..n_segs).rev() {
                let c = counts[k] as i32;
                let val = ((act_top as f64) - raw_b[k]) / (c as f64);
                let f = (val.round() as i32).max(1);
                fs[k] = f as u16;
                act_top -= f * c;
            }
            fs[0] = act_top.max(2) as u16;
            fs
        };

        let mut bases = Vec::with_capacity(n_segs);
        let mut acc = 0u32;
        for (f, c) in freqs_u16.iter().zip(counts.iter()) {
            bases.push(acc as u16);
            acc += (*f as u32) * (*c);
        }
        debug_assert_eq!(acc, RANS_M, "probability model must sum to 65536");

        Self {
            eab,
            q,
            freqs: freqs_u16,
            bases,
        }
    }

    fn exact_override(eab: u8, q: u8) -> Option<&'static [u16]> {
        Some(match (eab, q) {
            (2, 1) => &[21168, 17592, 13388],
            (4, 1) => &[5665, 5407, 5048, 4398, 3347],
            (2, 2) => &[26354, 18130, 10526],
            (3, 2) => &[14406, 11948, 9065, 5263],
            (4, 2) => &[7539, 6867, 5975, 4531, 2632],
            (2, 3) => &[31900, 17918, 7859],
            (3, 4) => &[22624, 15126, 8437, 2728],
            (2, 5) => &[43800, 14920, 3408],
            (3, 6) => &[33414, 16436, 6031, 906],
            (4, 7) => &[24559, 15361, 7808, 2124, 188],
            (2, 8) => &[60308, 4812, 208],
            (2, 9) => &[63575, 1901, 30],
            (3, 10) => &[60388, 4744, 200, 1],
            (3, 11) => &[64250, 1262, 10, 1],
            (8, 11) => &[7575, 6701, 5582, 3892, 1918, 494, 34, 1, 1],
            (3, 12) => &[65456, 74, 1, 1],
            (4, 12) => &[63248, 2208, 34, 1, 1],
            (5, 12) => &[53291, 9957, 1104, 14, 1, 1],
            (5, 13) => &[63380, 2084, 22, 1, 1, 1],
            (6, 13) => &[53649, 9731, 1042, 4, 1, 1, 1],
            (6, 14) => &[65147, 327, 1, 1, 1, 1, 1],
            _ => return None,
        })
    }

    /// Forward symbol lookup (`PHASE 0`): given symbol `sym`, returns `(fwd_f, fwd_cf)`.
    #[inline]
    pub fn forward_params(&self, sym: u16) -> (u32, u32) {
        if sym == 0 {
            let fa = self.freqs[0] as u32;
            (fa - 1, 1)
        } else {
            let seg = (16 - sym.leading_zeros()) as usize;
            let seg_start = 1u16 << (seg - 1);
            let steps = (sym - seg_start) as u32;
            let base = self.bases[seg] as u32;
            let fa = self.freqs[seg] as u32;
            (fa, base + fa * steps)
        }
    }

    /// Inverse state lookup (`PHASE 1`): given `cf_in = state & 0xFFFF`, returns `(sym, fwd_f, fwd_cf)`.
    #[inline]
    pub fn inverse_lookup(&self, cf_in: u16) -> Option<(u16, u32, u32)> {
        if cf_in == 0 {
            return None;
        }
        let mut seg = 0usize;
        for (k, &b) in self.bases.iter().enumerate().rev() {
            if cf_in >= b {
                seg = k;
                break;
            }
        }
        if seg == 0 {
            let fa = self.freqs[0] as u32;
            Some((0, fa - 1, 1))
        } else {
            let b = self.bases[seg] as u32;
            let fa = self.freqs[seg] as u32;
            let step = ((cf_in as u32) - b) / fa;
            let sym = (1u16 << (seg - 1)) + (step as u16);
            Some((sym, fa, b + step * fa))
        }
    }

    /// Single-step rANS decode: returns `(sym, new_state)`.
    #[inline]
    pub fn decode_step(&self, state: u32) -> Option<(u16, u32)> {
        let cf_in = (state & 0xFFFF) as u16;
        let (sym, fwd_f, fwd_cf) = self.inverse_lookup(cf_in)?;
        let new_state = fwd_f * (state >> 16) + (cf_in as u32) - fwd_cf;
        Some((sym, new_state))
    }

    /// Encode a slice of symbols into a standalone fragment `(initial_state_be4_plus_renorm_bytes)`.
    /// Symbols are encoded in reverse order so forward decoding recovers `symbols[0..N]`.
    pub fn encode_fragment(&self, symbols: &[u16]) -> Vec<u8> {
        let mut x: u32 = RANS_L;
        let mut renorm_rev: Vec<u8> = Vec::new();
        for &sym in symbols.iter().rev() {
            let (f, cf) = self.forward_params(sym);
            let max_x = ((RANS_L >> 16) << 8) * f;
            while x >= max_x {
                renorm_rev.push((x & 0xFF) as u8);
                x >>= 8;
            }
            x = ((x / f) << 16) + (x % f) + cf;
        }
        let mut out = Vec::with_capacity(4 + renorm_rev.len());
        out.extend_from_slice(&x.to_be_bytes());
        renorm_rev.reverse();
        out.extend_from_slice(&renorm_rev);
        out
    }

    /// Decode a standalone fragment payload (`4-byte big-endian state + renorm bytes`)
    /// until all payload bytes are consumed and `x` reaches a legal end state (`L`, `L>>8`, or `L>>16`),
    /// or up to `max_symbols`.
    pub fn decode_fragment(&self, payload: &[u8], max_symbols: usize) -> Result<(Vec<u16>, u32), &'static str> {
        if payload.len() < 4 {
            return Err("fragment payload shorter than 4-byte initial state");
        }
        let mut x = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
        let mut pos = 4usize;
        let mut out = Vec::new();

        while out.len() < max_symbols {
            while x < RANS_L && pos < payload.len() {
                x = (x << 8) | (payload[pos] as u32);
                pos += 1;
            }
            if pos == payload.len() && is_legal_end_state(x) {
                return Ok((out, x));
            }
            if x < RANS_L {
                return Err("fragment payload underflow before legal end state");
            }
            let (sym, next_x) = self.decode_step(x).ok_or("invalid rANS slot 0 encountered")?;
            out.push(sym);
            x = next_x;
        }
        while x < RANS_L && pos < payload.len() {
            x = (x << 8) | (payload[pos] as u32);
            pos += 1;
        }
        if pos == payload.len() && is_legal_end_state(x) {
            Ok((out, x))
        } else {
            Err("fragment did not converge to legal end state within max_symbols")
        }
    }
}

/// Hardware control state of a single MDU rANS lane (`ctrl_state` in `US11234023B2` Fig. 15c).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneCtrlState {
    Idle,
    Hdr0,
    Hdr1,
    Hdr2,
    Processing,
    Draining,
}

/// A single MDU rANS decoder lane with 1-byte prefetch buffer (`input_buf`), 1-symbol output
/// buffer (`sym_buf`), and multi-fragment state machine (`DSTATE_IDLE`..`DSTATE_DRAINING`).
#[derive(Debug, Clone)]
pub struct RansLane {
    pub lane_index: usize,
    pub alphabet_bits: u8,
    pub ctrl_state: LaneCtrlState,
    pub input_buf: u8,
    pub input_buf_full: bool,
    pub sym_buf: u16,
    pub sym_buf_full: bool,
    pub input_remaining: usize,
    pub output_remaining: usize,
    pub rans_state_p0: u32,
    pub rans_state_p1: u32,
    pub hdr3: bool,
    pub flush_per_frag: bool,
    pub eab: u8,
    pub current_q: u8,
    pub model: ProbabilityModel,
    pub done: bool,
    pub decoded_symbols: Vec<u16>,
    pub fragments: Vec<FragmentHeader>,
    /// Bytes this lane pulled from the shared stream (bench instrumentation).
    pub bytes_read: usize,
    hdr0_byte: u8,
    hdr1_byte: u8,
}

impl RansLane {
    pub fn new(lane_index: usize, alphabet_bits: u8, out_target: usize) -> Self {
        Self {
            lane_index,
            alphabet_bits,
            ctrl_state: LaneCtrlState::Idle,
            input_buf: 0,
            input_buf_full: false,
            sym_buf: 0,
            sym_buf_full: false,
            input_remaining: 0,
            output_remaining: out_target,
            rans_state_p0: 0,
            rans_state_p1: 0,
            hdr3: false,
            flush_per_frag: false,
            eab: alphabet_bits,
            current_q: 0,
            model: ProbabilityModel::new(alphabet_bits, 0),
            done: false,
            decoded_symbols: Vec::with_capacity(out_target),
            fragments: Vec::new(),
            bytes_read: 0,
            hdr0_byte: 0,
            hdr1_byte: 0,
        }
    }

    pub fn launch(&mut self) {
        self.ctrl_state = LaneCtrlState::Hdr0;
        self.rans_state_p0 = 0;
        self.rans_state_p1 = 0;
        self.done = false;
    }

    /// Verilog `need_ib_load` condition (`Fig. 15c`).
    #[inline]
    pub fn need_ib_load(&self) -> bool {
        if self.input_buf_full {
            return false;
        }
        match self.ctrl_state {
            LaneCtrlState::Hdr0 | LaneCtrlState::Hdr1 | LaneCtrlState::Hdr2 => true,
            LaneCtrlState::Processing => self.input_remaining > (self.input_buf_full as usize),
            LaneCtrlState::Idle | LaneCtrlState::Draining => false,
        }
    }

    /// Execute `PHASE 0` (forward state update, input buffer load, output transfer).
    pub fn step_phase0(&mut self, stream: &[u8], pos: &mut usize, sym_ready: bool) {
        let blocked_on_output = self.sym_buf_full && !sym_ready;
        if !blocked_on_output {
            if matches!(self.ctrl_state, LaneCtrlState::Processing | LaneCtrlState::Draining)
                || self.sym_buf_full
            {
                if self.sym_buf_full {
                    let (fwd_f, fwd_cf) = self.model.forward_params(self.sym_buf);
                    let fwd_p = self.rans_state_p1 & 0xFFFF;
                    self.rans_state_p0 = fwd_f
                        .wrapping_mul(self.rans_state_p1 >> 16)
                        .wrapping_add(fwd_p)
                        .wrapping_sub(fwd_cf);
                    if self.done {
                        self.rans_state_p1 = self.rans_state_p0;
                    }
                } else {
                    self.rans_state_p0 = self.rans_state_p1;
                }
            }
            if self.sym_buf_full && sym_ready {
                self.decoded_symbols.push(self.sym_buf);
                self.sym_buf_full = false;
            }
        }
        if self.need_ib_load() && *pos < stream.len() {
            self.input_buf = stream[*pos];
            self.input_buf_full = true;
            self.bytes_read += 1;
            *pos += 1;
        }
    }

    /// Execute `PHASE 1` (header parsing, byte merge into rANS state, inverse symbol decode).
    pub fn step_phase1(&mut self, stalled: bool) {
        if stalled {
            return;
        }
        match self.ctrl_state {
            LaneCtrlState::Idle => {}
            LaneCtrlState::Hdr0 => {
                if self.input_buf_full {
                    let b = self.input_buf;
                    self.hdr0_byte = b;
                    self.input_remaining = 0;
                    self.ctrl_state = LaneCtrlState::Hdr1;
                    let (z, hdr3, flush, q) = FragmentHeader::parse_mode_byte(b);
                    self.current_q = q;
                    self.hdr3 = hdr3;
                    self.flush_per_frag = flush;
                    self.eab = self.alphabet_bits.saturating_sub(z).clamp(1, 8);
                    self.input_buf_full = false;
                }
            }
            LaneCtrlState::Hdr1 => {
                if self.input_buf_full {
                    let b = self.input_buf;
                    self.hdr1_byte = b;
                    self.input_buf_full = false;
                    self.model = ProbabilityModel::new(self.eab, self.current_q);
                    if self.hdr3 {
                        self.input_remaining = b as usize;
                        self.ctrl_state = LaneCtrlState::Hdr2;
                    } else {
                        self.input_remaining = (b as usize) + 1;
                        if let Some(fh) = FragmentHeader::from_bytes(self.hdr0_byte, b, None) {
                            self.fragments.push(fh);
                        }
                        self.ctrl_state = LaneCtrlState::Processing;
                    }
                }
            }
            LaneCtrlState::Hdr2 => {
                if self.input_buf_full {
                    let b = self.input_buf;
                    self.input_buf_full = false;
                    self.input_remaining = (((b as usize) << 8) | (self.input_remaining & 0xFF)) + 257;
                    if let Some(fh) = FragmentHeader::from_bytes(self.hdr0_byte, self.hdr1_byte, Some(b)) {
                        self.fragments.push(fh);
                    }
                    self.ctrl_state = LaneCtrlState::Processing;
                }
            }
            LaneCtrlState::Processing | LaneCtrlState::Draining => {
                let is_proc = self.ctrl_state == LaneCtrlState::Processing;
                let want_to_feed = is_proc && (self.rans_state_p0 < RANS_L) && (self.input_remaining > 0);
                let will_feed = want_to_feed && self.input_buf_full;
                let (st_in, new_in_rem) = if will_feed {
                    (
                        ((self.rans_state_p0 & 0x00FF_FFFF) << 8) | (self.input_buf as u32),
                        self.input_remaining - 1,
                    )
                } else {
                    (self.rans_state_p0, self.input_remaining)
                };

                let cf_in = (st_in & 0xFFFF) as u16;
                let new_sym = self
                    .model
                    .inverse_lookup(cf_in)
                    .map(|(s, _, _)| s)
                    .unwrap_or(0);
                let next_sym_full = (st_in > RANS_L) && (self.output_remaining > 0);
                let new_out_rem = if next_sym_full {
                    self.output_remaining - 1
                } else {
                    self.output_remaining
                };

                self.rans_state_p1 = st_in;
                self.output_remaining = new_out_rem;
                self.sym_buf = new_sym;
                self.sym_buf_full = next_sym_full;

                if is_proc {
                    self.input_remaining = new_in_rem;
                    if will_feed {
                        self.input_buf_full = false;
                    }
                    if new_in_rem == 0 {
                        if new_out_rem > 0 {
                            if st_in == RANS_L {
                                self.ctrl_state = LaneCtrlState::Hdr0;
                                if self.flush_per_frag {
                                    self.rans_state_p1 = 0;
                                }
                            } else {
                                self.ctrl_state = LaneCtrlState::Draining;
                            }
                        } else {
                            self.done = true;
                            self.ctrl_state = LaneCtrlState::Idle;
                        }
                    }
                } else if st_in <= RANS_L {
                    if new_out_rem > 0 {
                        self.ctrl_state = LaneCtrlState::Hdr0;
                        if self.flush_per_frag {
                            self.rans_state_p0 = 0;
                            self.rans_state_p1 = 0;
                        }
                    } else {
                        self.ctrl_state = LaneCtrlState::Idle;
                        self.done = true;
                    }
                }
            }
        }
    }
}

/// De-interleave the 48-byte 8-lane main header (6 rounds of 8 bytes) and parse the initial
/// fragment header (and initial 32-bit state for 2-byte headers) of lanes `0..8`.
pub fn parse_main_8lane_header(stream: &[u8]) -> Option<[FragmentHeader; 8]> {
    if stream.len() < 48 {
        return None;
    }
    let mut out = [FragmentHeader {
        raw_mode: 0,
        z_adjust: 0,
        hdr3: false,
        flush_per_frag: false,
        model_q: 0,
        payload_bytes: 0,
        header_bytes: 2,
    }; 8];
    for lane in 0..8 {
        let b0 = stream[lane];
        let b1 = stream[8 + lane];
        let b2 = stream[16 + lane];
        out[lane] = FragmentHeader::from_bytes(b0, b1, Some(b2))?;
    }
    Some(out)
}

/// Extract the 6-byte column-interleaved record for `lane` (`0..8`) from the 48-byte main header.
pub fn extract_main_lane_record(stream: &[u8], lane: usize) -> Option<[u8; 6]> {
    if stream.len() < 48 || lane >= 8 {
        return None;
    }
    let mut rec = [0u8; 6];
    for r in 0..6 {
        rec[r] = stream[r * 8 + lane];
    }
    Some(rec)
}

/// Encode a multi-lane BCPack chunk in round-robin hardware read order:
/// - Main lanes `0..8` launch at Round 0.
/// - Secondary/index lanes `8..K` launch at Round 6.
///
/// Each lane entry is `(alphabet_bits, z_adjust, model_q, symbols)`.
pub fn encode_multilane_stream(lanes_spec: &[(u8, u8, u8, Vec<u16>)]) -> Vec<u8> {
    assert!(lanes_spec.len() >= 8, "at least 8 main lanes required");
    // Encode each lane into its complete record bytes (header + state + renorm bytes).
    let mut lane_records: Vec<Vec<u8>> = Vec::with_capacity(lanes_spec.len());
    let mut lane_hdr3: Vec<bool> = Vec::with_capacity(lanes_spec.len());
    for &(ab, z, q, ref syms) in lanes_spec {
        let eab = ab.saturating_sub(z).clamp(1, 8);
        let model = ProbabilityModel::new(eab, q);
        let payload = model.encode_fragment(syms);
        let hdr = FragmentHeader::encode_header(z, true, q, payload.len());
        lane_hdr3.push(hdr.len() == 3);
        let mut full = hdr;
        full.extend_from_slice(&payload);
        lane_records.push(full);
    }

    // Drive an exact mirror of the multi-lane decoder to emit bytes in the exact round-robin order
    // that the feeder reads them.
    let n_lanes = lanes_spec.len();
    let out_target = lanes_spec[0].3.len();
    let mut dec_lanes: Vec<RansLane> = lanes_spec
        .iter()
        .enumerate()
        .map(|(i, &(ab, _, _, ref syms))| RansLane::new(i, ab, syms.len()))
        .collect();
    let mut lane_pos = vec![0usize; n_lanes];
    let mut out_stream = Vec::new();

    for l in dec_lanes.iter_mut().take(8) {
        l.launch();
    }

    for rnd in 0..(out_target * 4 + 64) {
        if rnd == 6 {
            for l in dec_lanes.iter_mut().skip(8) {
                l.launch();
            }
        }
        let all_syms_ready = dec_lanes
            .iter()
            .all(|l| l.ctrl_state == LaneCtrlState::Idle || l.done || l.sym_buf_full);

        // PHASE 0: feed 1 byte from lane_records[i] to each lane that requests one
        for i in 0..n_lanes {
            let blocked_on_output = dec_lanes[i].sym_buf_full && !all_syms_ready;
            if !blocked_on_output {
                if matches!(
                    dec_lanes[i].ctrl_state,
                    LaneCtrlState::Processing | LaneCtrlState::Draining
                ) || dec_lanes[i].sym_buf_full
                {
                    if dec_lanes[i].sym_buf_full {
                        let (fwd_f, fwd_cf) = dec_lanes[i].model.forward_params(dec_lanes[i].sym_buf);
                        let fwd_p = dec_lanes[i].rans_state_p1 & 0xFFFF;
                        dec_lanes[i].rans_state_p0 = fwd_f
                            .wrapping_mul(dec_lanes[i].rans_state_p1 >> 16)
                            .wrapping_add(fwd_p)
                            .wrapping_sub(fwd_cf);
                        if dec_lanes[i].done {
                            dec_lanes[i].rans_state_p1 = dec_lanes[i].rans_state_p0;
                        }
                    } else {
                        dec_lanes[i].rans_state_p0 = dec_lanes[i].rans_state_p1;
                    }
                }
                if dec_lanes[i].sym_buf_full && all_syms_ready {
                    let s = dec_lanes[i].sym_buf;
                    dec_lanes[i].decoded_symbols.push(s);
                    dec_lanes[i].sym_buf_full = false;
                }
            }
            if dec_lanes[i].need_ib_load() && lane_pos[i] < lane_records[i].len() {
                let b = lane_records[i][lane_pos[i]];
                lane_pos[i] += 1;
                out_stream.push(b);
                dec_lanes[i].input_buf = b;
                dec_lanes[i].input_buf_full = true;
            }
        }

        // PHASE 1
        for i in 0..n_lanes {
            let stalled = dec_lanes[i].sym_buf_full && !all_syms_ready;
            dec_lanes[i].step_phase1(stalled);
        }

        if dec_lanes.iter().all(|l| l.done && !l.sym_buf_full) {
            break;
        }
    }
    out_stream
}

/// Decode a multi-lane BCPack stream using the 2-phase column-interleaved feeder and lockstep
/// symbol vector assembly (`symvec`).
pub fn decode_multilane_stream(
    stream: &[u8],
    alphabet_bits: &[u8],
    symbols_per_lane: usize,
) -> Result<Vec<RansLane>, &'static str> {
    match decode_multilane_stream_traced(stream, alphabet_bits, symbols_per_lane)? {
        (lanes, true) => Ok(lanes),
        _ => Err("multi-lane stream did not finish within max_rounds"),
    }
}

/// Same as [`decode_multilane_stream`] but also returns the lanes when the decode did
/// not finish (`false`), so a test bench can report per-lane partial traces.
pub fn decode_multilane_stream_traced(
    stream: &[u8],
    alphabet_bits: &[u8],
    symbols_per_lane: usize,
) -> Result<(Vec<RansLane>, bool), &'static str> {
    if alphabet_bits.len() < 8 {
        return Err("at least 8 lanes required");
    }
    let n_lanes = alphabet_bits.len();
    let mut lanes: Vec<RansLane> = alphabet_bits
        .iter()
        .enumerate()
        .map(|(i, &ab)| RansLane::new(i, ab, symbols_per_lane))
        .collect();

    for l in lanes.iter_mut().take(8) {
        l.launch();
    }

    let mut pos = 0usize;
    let max_rounds = symbols_per_lane * 6 + 128;
    for rnd in 0..max_rounds {
        if rnd == 6 {
            for l in lanes.iter_mut().skip(8) {
                l.launch();
            }
        }
        let all_syms_ready = lanes
            .iter()
            .all(|l| l.ctrl_state == LaneCtrlState::Idle || l.done || l.sym_buf_full);
        let mut stalled_flags = vec![false; n_lanes];
        for (i, l) in lanes.iter_mut().enumerate() {
            stalled_flags[i] = l.sym_buf_full && !all_syms_ready;
            l.step_phase0(stream, &mut pos, all_syms_ready);
        }
        for (i, l) in lanes.iter_mut().enumerate() {
            l.step_phase1(stalled_flags[i]);
        }
        if lanes.iter().all(|l| l.done && !l.sym_buf_full) {
            for l in &mut lanes {
                if !l.sym_buf_full && l.decoded_symbols.len() < symbols_per_lane && l.output_remaining == 0 {
                    l.decoded_symbols.push(l.sym_buf);
                }
            }
            return Ok((lanes, true));
        }
    }
    // Flush final buffered symbol if all lanes completed output_remaining == 0
    if lanes.iter().all(|l| l.output_remaining == 0) {
        for l in &mut lanes {
            if l.sym_buf_full {
                l.decoded_symbols.push(l.sym_buf);
                l.sym_buf_full = false;
            }
        }
        return Ok((lanes, true));
    }
    Ok((lanes, false))
}

/// Assemble 8-byte BC1 blocks from the 12 decoded BC1 lanes:
/// - `L2..L4`: `(R0, G0, B0)` (5-6-5 color endpoint 0, delta-accumulated mod 32/64/32)
/// - `L5..L7`: `(R1, G1, B1)` (5-6-5 color endpoint 1, delta-accumulated mod 32/64/32)
/// - `L8..L11`: 4 rows of 8-bit index bytes (4x 2-bit indices per row)
pub fn assemble_bc1_blocks(lanes: &[RansLane], num_blocks: usize) -> Result<Vec<u8>, &'static str> {
    if lanes.len() < 12 {
        return Err("BC1 assembly requires 12 lanes");
    }
    for l in lanes.iter().take(12) {
        if l.decoded_symbols.len() < num_blocks {
            return Err("lane has fewer decoded symbols than num_blocks");
        }
    }
    let mut out = Vec::with_capacity(num_blocks * 8);
    let (mut r0, mut g0, mut b0) = (0u16, 0u16, 0u16);
    let (mut r1, mut g1, mut b1) = (0u16, 0u16, 0u16);

    for i in 0..num_blocks {
        r0 = (r0 + lanes[2].decoded_symbols[i]) & 0x1F;
        g0 = (g0 + lanes[3].decoded_symbols[i]) & 0x3F;
        b0 = (b0 + lanes[4].decoded_symbols[i]) & 0x1F;
        r1 = (r1 + lanes[5].decoded_symbols[i]) & 0x1F;
        g1 = (g1 + lanes[6].decoded_symbols[i]) & 0x3F;
        b1 = (b1 + lanes[7].decoded_symbols[i]) & 0x1F;

        let mut c0 = (r0 << 11) | (g0 << 5) | b0;
        let mut c1 = (r1 << 11) | (g1 << 5) | b1;
        // Lane 0 bit 0 can signal 3-color mode swap if c0 <= c1 in 4-color mode
        let mode_bit = lanes[0].decoded_symbols[i] & 1;
        if (mode_bit == 0 && c0 < c1) || (mode_bit == 1 && c0 > c1) {
            core::mem::swap(&mut c0, &mut c1);
        }

        out.extend_from_slice(&c0.to_le_bytes());
        out.extend_from_slice(&c1.to_le_bytes());
        out.push((lanes[8].decoded_symbols[i] & 0xFF) as u8);
        out.push((lanes[9].decoded_symbols[i] & 0xFF) as u8);
        out.push((lanes[10].decoded_symbols[i] & 0xFF) as u8);
        out.push((lanes[11].decoded_symbols[i] & 0xFF) as u8);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_128_probability_models_sum_to_65536() {
        for eab in 1..=8 {
            for q in 0..=15 {
                let m = ProbabilityModel::new(eab, q);
                let mut sum = 0u32;
                for (k, &f) in m.freqs.iter().enumerate() {
                    let count = if k == 0 { 1 } else { 1u32 << (k - 1) };
                    assert!(f >= 1, "frequency must be >= 1 for eab={eab} q={q}");
                    assert_eq!(m.bases[k] as u32, sum);
                    sum += (f as u32) * count;
                }
                assert_eq!(sum, RANS_M, "eab={eab} q={q} failed sum==65536");
            }
        }
    }

    #[test]
    fn test_patent_us11234023_base_and_freq_table_exact() {
        // US11234023B2 col. 19 lines 25-48 reference values for eab=8, q=11 (labeled [8][12] in 1-based example)
        let m = ProbabilityModel::new(8, 11);
        assert_eq!(m.freqs, vec![7575, 6701, 5582, 3892, 1918, 494, 34, 1, 1]);
        assert_eq!(
            m.bases,
            vec![0, 7575, 14276, 25440, 41008, 56352, 64256, 65344, 65408]
        );
    }

    #[test]
    fn test_confirmed_constant_lane_states_exact() {
        // Verify confirmed canonical constant-lane states across N = 1024, 1536, 2048, 4096, 8192:
        // - mode 0xdc (eab=2, q=12, K=4)
        // - mode 0xdd (eab=3, q=13, K=8)
        // - mode 0xde (eab=5, q=14, K=32)
        // - mode 0xdb (eab=1, q=11, K=2)
        let cases: &[(u8, u8, usize, u32)] = &[
            (2, 12, 1024, 0x0110_7ed4),
            (2, 12, 1536, 0x0119_2334),
            (2, 12, 4096, 0x0148_a474),
            (3, 13, 1024, 0x0122_09c0),
            (3, 13, 1536, 0x0134_b898),
            (3, 13, 4096, 0x01a5_d798),
            (5, 14, 1024, 0x01a5_d380),
            (5, 14, 1536, 0x021d_88c0),
            (5, 14, 2048, 0x02b7_4040),
            (5, 14, 4096, 0x0761_8900),
            (5, 14, 8192, 0x368d_cea0),
            (1, 11, 8192, 0x0148_b64c),
        ];

        for &(eab, q, n, expected_state) in cases {
            let model = ProbabilityModel::new(eab, q);
            let syms = vec![0u16; n];
            let payload = model.encode_fragment(&syms);
            assert_eq!(payload.len(), 4, "constant lane should need 0 renorm bytes");
            let st = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
            assert_eq!(st, expected_state, "mismatch for eab={eab} q={q} N={n}");

            let (decoded, end_state) = model.decode_fragment(&payload, n).unwrap();
            assert_eq!(decoded.len(), n);
            assert!(decoded.iter().all(|&s| s == 0));
            assert_eq!(end_state, RANS_L);
        }
    }

    #[test]
    fn test_single_symbol_uniform_0x10_prefix_fragments() {
        // Mode 0x10 (z=0, q=0, len=4): 1 uniform symbol on 5-bit (0x2000xx00) and 6-bit (0x4000xx00) planes
        let m5 = ProbabilityModel::new(5, 0);
        for sym in 1..32u16 {
            let p = m5.encode_fragment(&[sym]);
            assert_eq!(p.len(), 4);
            let st = u32::from_be_bytes([p[0], p[1], p[2], p[3]]);
            assert_eq!(st, 0x2000_0000 | ((sym as u32) << 11));
            let (dec, end_x) = m5.decode_fragment(&p, 1).unwrap();
            assert_eq!(dec, vec![sym]);
            assert_eq!(end_x, RANS_L);
        }

        let m6 = ProbabilityModel::new(6, 0);
        for sym in 1..64u16 {
            let p = m6.encode_fragment(&[sym]);
            assert_eq!(p.len(), 4);
            let st = u32::from_be_bytes([p[0], p[1], p[2], p[3]]);
            assert_eq!(st, 0x4000_0000 | ((sym as u32) << 10));
            let (dec, end_x) = m6.decode_fragment(&p, 1).unwrap();
            assert_eq!(dec, vec![sym]);
            assert_eq!(end_x, RANS_L);
        }
    }

    #[test]
    fn test_rans_fragment_roundtrip_all_models_and_widths() {
        for eab in 1..=8u8 {
            let max_sym = (1u16 << eab) - 1;
            for q in 0..=15u8 {
                let model = ProbabilityModel::new(eab, q);
                let syms: Vec<u16> = (0..128)
                    .map(|i| {
                        if q >= 12 {
                            if i % 17 == 0 {
                                (i as u16) & max_sym
                            } else {
                                0
                            }
                        } else {
                            ((i * 7 + (q as usize) * 3) as u16) & max_sym
                        }
                    })
                    .collect();
                let payload = model.encode_fragment(&syms);
                let (decoded, end_state) = model.decode_fragment(&payload, syms.len()).unwrap();
                assert_eq!(decoded, syms, "roundtrip failed for eab={eab} q={q}");
                assert_eq!(end_state, RANS_L, "end state != L for eab={eab} q={q}");
            }
        }
    }

    #[test]
    fn test_fragment_header_2byte_and_3byte_plus_257_encoding() {
        // 2-byte header (hdr3 = 0): lengths 1..=256
        for len in [1usize, 4, 18, 256] {
            let h = FragmentHeader::encode_header(2, true, 12, len);
            assert_eq!(h.len(), 2);
            let parsed = FragmentHeader::from_bytes(h[0], h[1], None).unwrap();
            assert_eq!(parsed.z_adjust, 2);
            assert!(!parsed.hdr3);
            assert!(parsed.flush_per_frag);
            assert_eq!(parsed.model_q, 12);
            assert_eq!(parsed.payload_bytes, len);
        }
        // 3-byte header (hdr3 = 1): lengths 257..=65792 use u16_le + 257
        for len in [257usize, 512, 2052, 4100, 6148, 65792] {
            let h = FragmentHeader::encode_header(1, true, 10, len);
            assert_eq!(h.len(), 3);
            let parsed = FragmentHeader::from_bytes(h[0], h[1], Some(h[2])).unwrap();
            assert_eq!(parsed.z_adjust, 1);
            assert!(parsed.hdr3);
            assert!(parsed.flush_per_frag);
            assert_eq!(parsed.model_q, 10);
            assert_eq!(parsed.payload_bytes, len);
        }
    }

    #[test]
    fn test_multilane_bc1_interleaved_roundtrip_and_block_assembly() {
        let n_blocks = 64usize;
        let abs_bc1 = BcFormat::Bc1.lane_alphabet_bits();
        let specs: Vec<(u8, u8, u8, Vec<u16>)> = vec![
            (abs_bc1[0], 0, 8, (0..n_blocks).map(|i| (i % 4) as u16).collect()),
            (abs_bc1[1], 3, 14, vec![0u16; n_blocks]),
            (abs_bc1[2], 3, 12, vec![0u16; n_blocks]),
            (abs_bc1[3], 1, 10, (0..n_blocks).map(|i| (i % 5) as u16).collect()),
            (abs_bc1[4], 3, 12, vec![0u16; n_blocks]),
            (abs_bc1[5], 2, 11, (0..n_blocks).map(|i| (i % 3) as u16).collect()),
            (abs_bc1[6], 3, 13, vec![0u16; n_blocks]),
            (abs_bc1[7], 3, 12, vec![0u16; n_blocks]),
            (abs_bc1[8], 0, 2, (0..n_blocks).map(|i| ((i * 13) & 0xFF) as u16).collect()),
            (abs_bc1[9], 0, 2, (0..n_blocks).map(|i| ((i * 29) & 0xFF) as u16).collect()),
            (abs_bc1[10], 0, 2, (0..n_blocks).map(|i| ((i * 53) & 0xFF) as u16).collect()),
            (abs_bc1[11], 0, 2, (0..n_blocks).map(|i| ((i * 97) & 0xFF) as u16).collect()),
        ];

        let stream = encode_multilane_stream(&specs);
        let hdrs = parse_main_8lane_header(&stream).unwrap();
        assert_eq!(hdrs[1].raw_mode, 0xDE);
        assert_eq!(hdrs[1].payload_bytes, 4);

        let decoded_lanes = decode_multilane_stream(&stream, abs_bc1, n_blocks).unwrap();
        for (i, lane) in decoded_lanes.iter().enumerate() {
            assert_eq!(
                lane.decoded_symbols, specs[i].3,
                "lane {i} symbols mismatch"
            );
            assert!(
                is_legal_end_state(lane.rans_state_p1),
                "lane {i} did not finish on legal end state: {:#x}",
                lane.rans_state_p1
            );
        }

        let bc1_bytes = assemble_bc1_blocks(&decoded_lanes, n_blocks).unwrap();
        assert_eq!(bc1_bytes.len(), n_blocks * 8);
    }
}
