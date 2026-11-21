"""Print stream byte -> lane ownership for the first R rows of 8 bytes.
python3 -I bytemap7.py STORE CHUNK CONFIG [ROWS] [START]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scan7, mdusim
rel, ch, cfg = sys.argv[1], int(sys.argv[2]), scan7.parse_config(sys.argv[3])
rows = int(sys.argv[4]) if len(sys.argv) > 4 else 12
start = int(sys.argv[5]) if len(sys.argv) > 5 else 0
data, blocks = scan7.read_chunk(rel, ch)
orig = mdusim.finish
def fin(lanes, *a):
    r = orig(lanes, *a); r['bytes'] = [l.bytes for l in lanes]; return r
mdusim.finish = fin
r = mdusim.simulate(data, [w-1 for w, _ in cfg], [scan7.count_of(t, blocks) for _, t in cfg], stream_len=len(data))
own = {}
for i, b in enumerate(r['bytes']):
    for p in b: own[p] = i
for row in range(start // 8, start // 8 + rows):
    print("%5d" % (row*8), ' '.join('%02x:%-2s' % (data[p], own.get(p, '?')) for p in range(row*8, min(len(data), row*8+8))))
