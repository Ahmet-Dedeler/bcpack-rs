"""Compare bcpack-rs fitted (eab,q) model bases with Microsoft's silicon-rans base_table.

usage: python3 -I cmp_tables.py <mdu_minirans.v> <bcpack-rs/python dir>
No game data involved. Verilog: github.com/microsoft/silicon-rans (MIT).
"""
import re, sys
src = open(sys.argv[1]).read()
sys.path.insert(0, sys.argv[2])
import bcpack as b

def table(name):
    m = re.search(r'assign %s = (.*?)\}; // end %s' % (name, name), src, re.S)
    rows = [[int(x) for x in r.split(',')] for r in re.findall(r"'\{([0-9,\s]+)\}", m.group(1)) if r.count(',') == 9]
    assert len(rows) == 144, len(rows)
    return [[rows[e * 16 + q] for q in range(16)] for e in range(9)]  # [eab-1][q][seg]

base = table('base_table')
bad = []
for eab in range(1, 9):
    for q in range(16):
        v = base[eab - 1][q][:eab + 1]
        r = list(b.get_model(eab, q).bases)
        if v != r:
            bad.append((eab, q, max(abs(x - y) for x, y in zip(v, r))))
print("bcpack-rs (eab 1..8, q 0..15) models whose segment bases differ from Verilog: %d of 128" % len(bad))
print("max abs base difference over those:", max(d for _, _, d in bad))
print("small (<=2) diffs: %d ; large (>2): %d" % (sum(d <= 2 for *_, d in bad), sum(d > 2 for *_, d in bad)))
print("mismatching (eab,q,maxdiff):", bad)
