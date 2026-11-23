#!/usr/bin/env python3
# bcpack-rs by Ahmet Dedeler (https://github.com/Ahmet-Dedeler, https://ahmetdedeler.com)
# Source: https://github.com/Ahmet-Dedeler/bcpack-rs
# SPDX-License-Identifier: MIT OR Apache-2.0
# If you copy, vendor or port this file, keep this header and credit bcpack-rs in your README.
"""Refresh the decoder's two fitted data tables from the research outputs.

usage: python3 -I tools/import_tables.py [--window WINDOW.json] [--quad QUAD.json]

At least one of the two options is required; the other table is left as is.

  data/bc1_window.json      256-slot preset endpoint window (see docs/BCPACK_SPEC.md 5.3)
  data/bc1_quad_table.json  256-entry index-lane codebook (docs/BCPACK_SPEC.md 5.4)

Inputs are the research agents' fit outputs (window_table.json / window_fit.json from the
endpoint-window fit, quad_table.json from the index fit). Only the fitted values and
summary confidence numbers are copied; no per-texture data and no game bytes.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def arg(name, default):
    if name not in sys.argv:
        return default
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv):
        sys.exit(f"missing value for {name}")
    return sys.argv[idx + 1]


def dump(doc, path, list_key):
    """JSON with one list element per line (diff friendly)."""
    lines = ["{"]
    keys = [k for k in doc if k != list_key] + [list_key]
    for n, k in enumerate(keys):
        comma = "," if n < len(keys) - 1 else ""
        if k == list_key:
            items = [json.dumps(x, separators=(", ", ": ")) for x in doc[k]]
            lines.append('  "%s": [' % k)
            lines += ["    " + it + ("," if j < len(items) - 1 else "") for j, it in enumerate(items)]
            lines.append("  ]" + comma)
        else:
            lines.append("  %s: %s%s" % (json.dumps(k), json.dumps(doc[k]), comma))
    lines.append("}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _validate_endpoint_value(slot, val):
    limits = (31, 63, 31, 31, 63, 31)
    if not isinstance(val, (list, tuple)) or len(val) != 6:
        raise ValueError(f"window slot {slot}: expected 6 endpoint fields")
    out = [int(x) for x in val]
    for i, (x, hi) in enumerate(zip(out, limits)):
        if not (0 <= x <= hi):
            raise ValueError(f"window slot {slot} field {i}: {x} not in 0..={hi}")
    return out


def import_window(path):
    d = json.load(open(path))
    if "slots" in d and isinstance(d["slots"], list):   # already in our format
        if len(d["slots"]) != 256:
            raise ValueError(f"window slots must have 256 entries, got {len(d['slots'])}")
        slots = []
        for idx, entry in enumerate(d["slots"]):
            if entry is None:
                slots.append(None)
            elif isinstance(entry, dict) and "value" in entry:
                e2 = dict(entry)
                e2["value"] = _validate_endpoint_value(idx, entry["value"])
                slots.append(e2)
            else:
                raise ValueError(f"window slot {idx}: bad entry {entry!r}")
        meta = {k: v for k, v in d.items() if k != "slots"}
    else:
        src = d["slots"] if isinstance(d.get("slots"), dict) else d
        slots = [None] * 256
        meta = {}
        for k, v in src.items():
            if not (isinstance(k, str) and k.isascii() and k.isdigit()):
                continue
            idx = int(k)
            if not (0 <= idx < 256) or not isinstance(v, dict):
                continue
            val = v.get("value") or v.get("joint")
            if val is None:
                continue
            ent = {"value": _validate_endpoint_value(idx, val)}
            for key in ("textures", "near", "blocks", "rms", "halves_agree"):
                if key in v and isinstance(v[key], (int, float, bool)):
                    ent[key] = v[key]
            if "per" in v and "textures" not in ent:
                ent["textures"] = len(v["per"])
            slots[idx] = ent
    out = {
        "format": "BCPack BC1 preset endpoint window: slot s = endpoint fields "
                  "[R0, G0, B0, R1, G1, B1] read by a back reference that points before block 0 "
                  "(slot = (i - d) mod 256). null = not fitted yet (decoded as zeros and reported).",
        "source": "fitted against mip-1 consistency on retail texture pools (endpoint-window fit); "
                  "values only",
        "fitted": sum(1 for s in slots if s is not None),
        "slots": slots,
    }
    out.update({k: v for k, v in meta.items() if k not in out})
    dump(out, os.path.join(ROOT, "data", "bc1_window.json"), "slots")
    print("window: %d / 256 slots fitted" % out["fitted"])


def import_quad(path):
    d = json.load(open(path))
    ents = sorted(d["entries"], key=lambda e: e["sym"])
    assert [e["sym"] for e in ents] == list(range(256))
    for e in ents:
        codes = e["codes"]
        assert len(codes) == 4 and all(isinstance(c, int) and 0 <= c <= 3 for c in codes), (
            f"bad quad codes for sym {e['sym']}: {codes}"
        )
    pats = set(tuple(e["codes"]) for e in ents)
    assert len(pats) == 256, "not a bijection"
    out = {
        "format": d["format"],
        "source": d["source"],
        "bijection": True,
        "entries": [{k: e[k] for k in ("sym", "codes", "stable_halves", "stable_edgeonly", "margin", "share") if k in e}
                    for e in ents],
    }
    dump(out, os.path.join(ROOT, "data", "bc1_quad_table.json"), "entries")
    print("quad table: 256 entries, %d stable across halves" % sum(e.get("stable_halves", False) for e in ents))


if __name__ == "__main__":
    w, q = arg("--window", None), arg("--quad", None)
    if w is None and q is None:
        sys.exit(__doc__)
    if w:
        import_window(w)
    if q:
        import_quad(q)
