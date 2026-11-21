# Xbox Series X|S MDU BCPack Format & rANS Architecture Specification

*Clean-room specification derived from public Microsoft hardware architecture disclosures (`US11234023B2`) and mathematical rANS state convergence verification (`x == L = 0x01000000`). Licensed under `MIT OR Apache-2.0`.*

---

## 1. Overview & Pipeline Position

**BCPack** is the fixed-function hardware entropy compressor/decompressor inside the Xbox Series X|S **Media Decompression Unit (MDU)** for DirectX block-compressed (`BC1`–`BC7`) GPU textures. In `RDS5` (`ttpf`) texture stores, chunk table entry bits `24..29` encode the codec ID as `BC_number * 4`:

| Codec ID | Format | Bytes / Block | Main Lanes (`R0..R5`) | Secondary / Index Lanes (`R6+`) |
|---|---|---|---|---|
| `0x04` | `BC1` (`DXGI 71/72`) | 8 B | 8 (`L0..L7`: mode, const, `R0 G0 B0 R1 G1 B1`) | 4 (`L8..L11`: 4 rows of 4x2-bit indices) |
| `0x0C` | `BC3` (`DXGI 77/78`) | 16 B | 8 (`L0..L7`: alpha/color endpoints & modes) | 4+ (alpha & color index lanes) |
| `0x10` | `BC4` (`DXGI 80/81`) | 8 B | 8 (`L0..L3` endpoints/ctrl + `L4..L7` 48-bit index slices) | — |
| `0x14` | `BC5` (`DXGI 83/84`) | 16 B | 8 (`L0..L3` dual-channel endpoints/ctrl + `L4..L7` ch0 indices) | 4 (`L8..L11`: ch1 48-bit index slices) |
| `0x18` | `BC6H` (`DXGI 95/96`) | 16 B | 8 (`L0..L7` mode/partition/endpoints) | index lanes |
| `0x1C` | `BC7` (`DXGI 98/99`) | 16 B | 8 (`L0..L7` mode/partition/rotation/endpoints) | index lanes |

---

## 2. Fragment Header Layout (`DSTATE_HDR0`..`DSTATE_HDR2`)

Every rANS lane is a sequence of $g \ge 1$ **variable-size fragments** (`US11234023B2` Fig. 14 and Fig. 15d/15i). A lane does **not** have a single fixed model for an entire chunk; when a fragment exhausts its byte budget (`input_remaining == 0`) and its rANS state drains to $\le L$ while `output_remaining > 0`, the lane transitions back to `DSTATE_HDR0` (`ctrl_state <= DSTATE_HDR0`) and reads the next fragment header in-band.

### 2.1 Mode Byte (`hdr0`) Bitfields

```
  Bit 7   Bit 6 | Bit 5  |     Bit 4      | Bit 3 .. Bit 0
+---------------+--------+----------------+----------------+
| hdr0_z_field  |  hdr3  | flush_per_frag |  hdr0_q_field  |
+---------------+--------+----------------+----------------+
```

- **`hdr0_z_field = hdr0[7:6]` (2 bits, `0..=3`)**: Symbol width reduction $z$. Subtracted from the lane's default `alphabet_bits` ($d$) to produce the effective alphabet bit width for the fragment:
  $$\text{eab} = \min(9,\ d - z)$$
  so the fragment alphabet covers $2^{\text{eab}}$ values ($0 \dots 2^{\text{eab}}-1$).
- **`hdr3 = hdr0[5]` (1 bit)**: Extra length byte flag (`DSTATE_HDR1` / `DSTATE_HDR2` in Fig. 15i):
  - **`hdr3 == 0` (2-byte header `[hdr0, b1]`)**:
    $$\text{input\_remaining} = b_1 + 1 \quad (1 \dots 256\text{ bytes})$$
    This explains why `0x1c`, `0x5c`, `0x9c`, `0xdc`, `0xdd`, `0xde` (all with bit 5 clear) use a 1-byte length field `n - 1`.
  - **`hdr3 == 1` (3-byte header `[hdr0, b1, b2]`)**:
    $$\text{input\_remaining} = (b_1 \mid (b_2 \ll 8)) + 257 \quad (257 \dots 65792\text{ bytes})$$
    This explains why `0x30`–`0x3f`, `0x70`–`0x7f`, `0xb0`–`0xbf` (all with bit 5 set) use a 16-bit little-endian length offset by **`+ 257`**.
- **`flush_per_frag = hdr0[4]` (1 bit)**: If `1`, the 32-bit rANS state register is zeroed at the end of the previous fragment and reloaded from the first 4 payload bytes (big-endian) of the new fragment. If `0`, the terminal state from the previous fragment carries over.
- **`hdr0_q_field = hdr0[3:0]` (4 bits, `0..=15`)**: Selects one of **16 static probability models** (`current_q`).

### 2.2 Resolution of `0x10` "Extension Lanes"

Records of the form `10 03 20 00 xx 00` (on 5-bit R/B planes) and `10 03 40 00 xx 00` (on 6-bit G planes) are **not** separate extension lanes or non-rANS parameter records:
- `0x10` has `z = 0, hdr3 = 0, flush_per_frag = 1, q = 0` (uniform distribution) and `b1 = 0x03` (`input_remaining = 4` bytes = 0 renorm bytes).
- Encoding **1 symbol** `sym` with uniform probability $f = 65536 / 2^{\text{eab}}$ starting from $x = L = \text{0x01000000}$ produces:
  - 5-bit plane ($f = 2048$, $\text{cf} = 2048 \cdot \text{sym}$): $x = (2^{24} / 2048) \cdot 2^{16} + 2048 \cdot \text{sym} = \text{0x20000000} \mid (\text{sym} \ll 11)$ (`20 00 (sym<<3) 00`).
  - 6-bit plane ($f = 1024$, $\text{cf} = 1024 \cdot \text{sym}$): $x = (2^{24} / 1024) \cdot 2^{16} + 1024 \cdot \text{sym} = \text{0x40000000} \mid (\text{sym} \ll 10)$ (`40 00 (sym<<2) 00`).
- Decoding that 1 initial base endpoint symbol in Round 6 returns $x$ to `0x01000000 == L` with `input_remaining == 0` and `output_remaining = N - 1 > 0`.
- Per Fig. 15j (`DSTATE_PROCESSING` / `DSTATE_DRAINING`), the lane immediately transitions to `DSTATE_HDR0` and reads its **Fragment 1 header** (e.g. `9c len ...` coding the remaining $N - 1$ delta symbols) in Round 7 in its normal lane slot.

---

## 3. Power-of-Two Segmented Static Probability Models (`q = 0..15`)

Instead of adaptive tables or per-chunk frequency headers, the MDU rANS decoder uses pre-defined lookup tables `base_table[eab][q][seg]` and `freq_table[eab][q][seg]` (`US11234023B2` Fig. 15b, 15d–15g).

### 3.1 Power-of-Two Segment Partition (`seg = 0..eab`)

For effective alphabet width $\text{eab} \in \{1 \dots 8\}$ ($2^{\text{eab}}$ symbols):
- **Segment 0**: symbol `0` (`seg_start = 0`, count = $1$).
- **Segment $k$ ($1 \le k \le \text{eab}$)**: symbols $[2^{k-1},\ 2^k - 1]$ (`seg_start = 1 << (k - 1)`, count = $2^{k-1}$).

All $2^{k-1}$ symbols in segment $k$ share a uniform step frequency $f_k = \text{freq\_table}[\text{eab}][q][k]$, so cumulative frequencies are piecewise linear across power-of-two buckets:
$$\text{fwd\_cfa}(\text{sym}) = \text{base\_table}[\text{eab}][q][k] + f_k \cdot (\text{sym} - 2^{k-1})$$

### 3.2 Slot-0 Reservation (`fwd_cfa == 0`)

In Fig. 15g (`1507`), symbol `0` (`fwd_cfa == 0`) is adjusted by 1 slot so that `cf = 0` is never assigned to a valid symbol:
```verilog
wire [15:0] fwd_cfa = fwd_base + (fwd_fa * fwd_steps);
wire [15:0] fwd_f   = (fwd_cfa == 0) ? fwd_fa - 1 : fwd_fa;
wire [15:0] fwd_cf  = (fwd_cfa == 0) ? 1 : fwd_cfa;
```
Consequently, when all $2^{\text{eab}} - 1$ non-zero symbols hit the minimum frequency floor $f_k = 1$, symbol `0` has `fwd_fa = 65536 - (2^eab - 1)` and **`fwd_f = 65536 - 2^eab` ($M - K$ with $K = 2^{\text{eab}}$)** starting at **`fwd_cf = 1`**. This directly produces all confirmed constant-lane parameters:
- `0xdb` ($\text{eab} = 1, q = 11$): $K = 2^1 = 2$, $f = 65534$
- `0xdc` ($\text{eab} = 2, q = 12$): $K = 2^2 = 4$, $f = 65532$
- `0xdd` ($\text{eab} = 3, q = 13$): $K = 2^3 = 8$, $f = 65528$
- `0x5e` ($\text{eab} = 4, q = 14$): $K = 2^4 = 16$, $f = 65520$
- `0xde` / `0x1e` ($\text{eab} = 5, q = 14$): $K = 2^5 = 32$, $f = 65504$
- `0x1f` on 6-bit ($\text{eab} = 6, q = 15$): $K = 2^6 = 64$, $f = 65472$
- `0x1f` on 8-bit ($\text{eab} = 8, q = 15$): $K = 2^8 = 256$, $f = 65280$

### 3.3 Recovered Model Decay Schedule $\lambda(q)$ (`q = 0..15`)

Each model $q \in \{0 \dots 15\}$ is a truncated exponential CDF on $[0, 1]$, $\text{CDF}(t) = \frac{1 - e^{-\lambda_q t}}{1 - e^{-\lambda_q}}$, quantized backwards from $k = \text{eab}$ down to $0$ with minimum frequency floor $1$:

| Model $q$ | $\lambda_q$ | Exact Solved Example `freq_table[eab][q]` |
|---|---|---|
| `0` | `0.000000` | Uniform: $f_k = 65536 / 2^{\text{eab}}$ |
| `1` | `0.739926` | `eab=2`: `[21168, 17592, 13388]` · `eab=4`: `[5665, 5407, 5048, 4398, 3347]` |
| `2` | `1.496350` | `eab=2`: `[26354, 18130, 10526]` · `eab=3`: `[14406, 11948, 9065, 5263]` · `eab=4`: `[7539, 6867, 5975, 4531, 2632]` |
| `3` | `2.307231` | `eab=2`: `[31900, 17918, 7859]` |
| `4` | `3.221322` | `eab=3`: `[22624, 15126, 8437, 2728]` |
| `5` | `4.307485` | `eab=2`: `[43800, 14920, 3408]` |
| `6` | `5.676284` | `eab=3`: `[33414, 16436, 6031, 906]` |
| `7` | `7.508501` | `eab=4`: `[24559, 15361, 7808, 2124, 188]` |
| `8` | `10.112494` | `eab=2`: `[60308, 4812, 208]` |
| `9` | `14.036894` | `eab=2`: `[63575, 1901, 30]` |
| `10` | `20.348947` | `eab=3`: `[60388, 4744, 200, 1]` |
| `11` | `31.445855` | `eab=3`: `[64250, 1262, 10, 1]` · `eab=8`: `[7575, 6701, 5582, 3892, 1918, 494, 34, 1, 1]` |
| `12` | `53.681684` | `eab=4`: `[63248, 2208, 34, 1, 1]` · `eab=5`: `[53291, 9957, 1104, 14, 1, 1]` |
| `13` | `109.260000` | `eab=5`: `[63380, 2084, 22, 1, 1, 1]` · `eab=6`: `[53649, 9731, 1042, 4, 1, 1, 1]` |
| `14` | `328.115610` | `eab=6`: `[65147, 327, 1, 1, 1, 1, 1]` · `eab<=5`: constant ($K = 2^{\text{eab}}$) |
| `15` | $\infty$ | All $\text{eab} \in \{1 \dots 8\}$: $[65536 - (2^{\text{eab}} - 1),\ 1,\ 1,\ \dots,\ 1]$ |

### 3.4 Early-Trimming Legal End States (`legal_end_state`)

In Fig. 15g (`1507`), the hardware decoder accepts three terminal states at the end of a fragment (`// we can terminate slightly early due to trimming in the compressor`):
$$\text{legal\_end\_state} \in \{L,\ L \gg 8,\ L \gg 16\} = \{\text{0x01000000},\ \text{0x00010000},\ \text{0x00000100}\}$$
When an encoder begins a fragment with a single $f = 1$ exception symbol starting from the trimmed initial state $x = L \gg 16 = \text{0x100}$, no renorm bytes are emitted ($0x100 < ((L \gg 16) \ll 8) \cdot 1 = 0x10000$) and the state transitions directly to $x = \text{0x01000000} + \text{cf}$.

---

## 4. Two-Phase Multi-Lane Feeder & Lockstep `symvec` Schedule

1. **Rounds 0..5 (48-byte main header)**: Main lanes `L0..L7` launch at Round 0 (`DSTATE_HDR0`) and read 1 byte per lane per round for 6 rounds (48 bytes total, column-interleaved).
   - Lanes with `hdr3 == 0` read 2 header bytes (`R0..R1`) + 4 initial state bytes (`R2..R5`).
   - Lanes with `hdr3 == 1` read 3 header bytes (`R0..R2`) + 3 initial state bytes (`R3..R5`).
2. **Round 6 (index lane launch & 1-symbol base drain)**:
   - Any `L0..L7` lane with `hdr3 == 1` reads its 4th initial state byte; any `L0..L7` lane with `hdr3 == 0` and `payload_bytes > 4` reads its 1-byte prefetch (`input_buf`).
   - Any `L2..L7` lane whose Fragment 0 is a 1-symbol uniform base value (`0x10 / 0x50 / 0x90 / 0xd0`, `payload_bytes == 4`) drains that 1 symbol to $x = L$ and transitions to `DSTATE_HDR0` at the end of Round 6.
   - Index lanes `L8..L11` launch at Round 6 (`DSTATE_HDR0`) and read their `HDR0` byte.
3. **Rounds 7..12 (header/state alignment barrier)**:
   - `L2..L7` Fragment 1 (`hdr3 == 0`: 2 header bytes in `R7..R8` + 4 state bytes in `R9..R12`) and `L8..L11` Fragment 0 (`hdr3 == 1`: `HDR1` in `R7`, `HDR2` in `R8` + 4 state bytes in `R9..R12`) finish loading their 4th initial state byte on the **exact same cycle (Round 12 `PHASE 1`)** and load their prefetch byte in **Round 13 (`PHASE 0`)**.
   - While any active lane is still in `HDR0..HDR2` or loading its initial 4 state bytes, lanes that already have a valid symbol in `sym_buf` stall on output (`blocked_on_output = sym_buf_full && !sym_ready`).

---

## 5. Empirical Validation on Real Corpus

Tested across **3,674 real BC1 chunks** (`7,162` self-contained `len == 4` fragments spanning `q = 0..15` and `eab = 1..8`) in `ttdb_graphics.ttpf`, `ttdb_interiors_0_3.ttpf`, and `ttdb_maps_0_7.ttpf`, plus **100,020 BC1 chunks** scanned across all 46 `ttdb` stores:
- **`len == 4` fragment convergence (`x in {L, L>>8, L>>16}`)**: **`7,162 / 7,162` (`100.00%`)** across all observed models (`q = 0`: `981/981`, `q = 1`: `2/2`, `q = 9`: `1/1`, `q = 10`: `1/1`, `q = 12`: `3139/3139`, `q = 13`: `574/574`, `q = 14`: `2416/2416`, `q = 15`: `48/48`).
- **Byte-budget conservation on multi-fragment 4,096-block BC1 chunks** (e.g., `ttdb_interiors_1_3.ttpf` chunk 2574, `clen = 19,512` bytes): sum of parsed headers and fragment byte budgets across `L0..L11` (including `L4` and `L7` Fragment 0 `0x10` + Fragment 1 `0x9c`/`0x9b` transitions) equals **`19,512 / 19,512` bytes** to the exact byte.
