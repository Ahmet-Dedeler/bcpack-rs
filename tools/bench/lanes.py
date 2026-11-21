"""BCPack lane-record parsing and the published constant-lane model.

Header field layout and legal end states follow Microsoft's reference decoder
(``microsoft_silicon-rans/mdu_minirans.v``, MIT), the authority for this file:

* hdr0[7:6] = z (alphabet reduction: eab = alphabet_bits - z, 4-bit, clamped to 9 if >= 10)
* hdr0[5]   = hdr3: 0 -> 2-byte header, input_remaining = b1 + 1
                    1 -> 3-byte header, input_remaining = (b2 << 8 | b1) + 257
* hdr0[4]   = flush_per_frag (state zeroed at the end of a fragment)
* hdr0[3:0] = q (model index)
* legal end state: L, L >> 8 or L >> 16 with L = 1 << 24
* ``input_remaining`` counts every payload byte of the fragment, including the
  initial state bytes (the state starts at 0 and is fed while it is < L).

Main-lane records sit in the first 48 bytes of a chunk, column-interleaved
(8 lanes x 6 rounds: byte r*8+lane), per SixNG formats/06-bcpack.md.

Constant-lane model (SixNG §6.1, confirmed byte-for-byte by SixNG and by
MarlomMedeiros on BC7): M = 2^16, L = 2^24, encoder starts at x = L, the repeated
symbol sits at cumulative start 1 with frequency M - K.  A lane is
"provably constant" when re-encoding N copies reproduces BOTH the record's byte
budget (4 + renorm bytes) and its 32-bit initial state exactly; chance of a
false match is ~1e-6 per record or less (32-bit state, budget fixed).
"""
from __future__ import annotations

import array
import bisect
import mmap
import os
from typing import Dict, List, Optional, Tuple

RANS_L = 1 << 24
RANS_M = 1 << 16
LEGAL_END_STATES = (RANS_L, RANS_L >> 8, RANS_L >> 16)

CONST_KS = (2, 4, 8, 16, 32, 64, 128, 256)
CONST_NMAX = 1 << 21          # symbols per lane covered by the constant tables
TABLE_DIR = os.environ.get("BCPACK_CONST_TABLES", "/tmp/bcpack-shared/const_tables")


# Constant-lane modes and the escape-slot counts K seen with them (z = 3 family).
# K = 2^(lane alphabet bits - 3): dc on 5-bit planes, dd on 6-bit planes, de on
# 8-bit lanes (BC1 lanes 0/1, BC7 lanes 2-6) and K=16 on BC7 lane 7, db K=2 on BC7
# lanes 0/1.  A (budget, state) hit at any other K/mode is reported separately
# because the false-positive rate per record is ~1e-5..1e-4 (see findings).
MODE_K = {0xdb: (2,), 0xdc: (4,), 0xdd: (8,), 0xde: (32, 16)}


def is_legal_end_state(x: int) -> bool:
    return x in LEGAL_END_STATES


def parse_record(rec: bytes) -> Dict:
    """Parse one 6-byte main-lane record (fragment 0 of that lane)."""
    mode = rec[0]
    z, hdr3, flush, q = mode >> 6, (mode >> 5) & 1, (mode >> 4) & 1, mode & 15
    r = {"mode": mode, "z": z, "hdr3": hdr3, "flush": flush, "q": q}
    if hdr3:
        len16 = rec[1] | (rec[2] << 8)
        r.update(hdr_bytes=3, budget=len16 + 257, budget_alt_plus1=len16 + 1,
                 state_prefix=rec[3:6].hex(), state=None)
    else:
        r.update(hdr_bytes=2, budget=rec[1] + 1, state_prefix=rec[2:6].hex(),
                 state=int.from_bytes(rec[2:6], "big"))
    return r


def main_records(chunk: bytes) -> List[bytes]:
    return [bytes(chunk[r * 8 + lane] for r in range(6)) for lane in range(8)]


# ---------------------------------------------------------------------------
# Constant-lane tables: for each K, state[N] and renorm[N] after encoding N
# repeated symbols from x = L.  Built once, stored as raw uint32 files, mmapped.
# ---------------------------------------------------------------------------
def _build_table(K: int, nmax: int, path: str):
    f = RANS_M - K
    bound = ((RANS_L >> 16) << 8) * f
    st = array.array("I", [0]) * (nmax + 1)
    rn = array.array("I", [0]) * (nmax + 1)
    x = RANS_L
    r = 0
    st[0] = x
    for n in range(1, nmax + 1):
        while x >= bound:
            x >>= 8
            r += 1
        x = ((x // f) << 16) + (x % f) + 1
        st[n] = x
        rn[n] = r
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "wb") as fh:
        st.tofile(fh)
        rn.tofile(fh)
    os.replace(tmp, path)


_TABLES: Dict[int, Tuple[memoryview, memoryview]] = {}


def const_table(K: int):
    if K not in _TABLES:
        os.makedirs(TABLE_DIR, exist_ok=True)
        path = os.path.join(TABLE_DIR, f"const_K{K}_N{CONST_NMAX}.u32")
        if not os.path.exists(path):
            _build_table(K, CONST_NMAX, path)
        fh = open(path, "rb")
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        mv = memoryview(mm).cast("I")
        n1 = CONST_NMAX + 1
        _TABLES[K] = (mv[:n1], mv[n1:2 * n1])
    return _TABLES[K]


def _pool_safe(K):  # picklable return for Pool.map
    const_table(K)
    return K


def const_match(budget: int, state: int, Ks=CONST_KS) -> List[Tuple[int, int]]:
    """All (K, N) such that N repeated symbols under the constant model with K
    escape slots give exactly `budget` payload bytes (4 + renorm) and `state`."""
    if state is None or state < RANS_L or budget < 4:
        return []
    r = budget - 4
    out = []
    for K in Ks:
        st, rn = const_table(K)
        lo = bisect.bisect_left(rn, r, 1)
        hi = bisect.bisect_right(rn, r, lo)
        if lo >= hi:
            continue
        # states strictly increase between renorms -> bisect inside [lo, hi)
        j = bisect.bisect_left(st, state, lo, hi)
        if j < hi and st[j] == state:
            out.append((K, j))
    return out


def const_renorm_bytes(K: int, N: int) -> bytes:
    """Renorm bytes the constant-model decoder consumes, in decode order."""
    f = RANS_M - K
    bound = ((RANS_L >> 16) << 8) * f
    x = RANS_L
    emitted = []
    for _ in range(N):
        while x >= bound:
            emitted.append(x & 255)
            x >>= 8
        x = ((x // f) << 16) + (x % f) + 1
    return bytes(reversed(emitted))
