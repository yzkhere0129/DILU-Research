"""Select 50 representative matrices from dense_track_dump_500K 384-step pool.

Strategy:
  pre_melt    (t < 3e-7  ): 10 matrices, evenly spaced
  melt        (3e-7 ~ 6e-7): 15 matrices  (most physics active)
  evap_early  (6e-7 ~ 9e-7): 10 matrices
  evap_late   (t > 9e-7  ): 15 matrices

Writes: <out>/manifest.json with the 50 selected timesteps + phase labels.
Reads: validate_dense/sane_pool.txt (any timestep is OK; pool entries already
sanity-checked).

Usage (dev):
  PYTHONPATH=. python dilu/amgx/bench/suite/select_50_matrices.py \\
      --pool validate_dense/sane_pool.txt \\
      --out  dilu/amgx/bench/suite/manifest.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


PHASE_QUOTAS = [
    ("pre_melt",   (0.0,    3.0e-7), 10),
    ("melt",       (3.0e-7, 6.0e-7), 15),
    ("evap_early", (6.0e-7, 9.0e-7), 10),
    ("evap_late",  (9.0e-7, 2.0e-6), 15),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="sane_pool.txt path")
    ap.add_argument("--out",  required=True, help="manifest.json output")
    args = ap.parse_args()

    pool = Path(args.pool).expanduser()
    out  = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    rels = []
    for line in pool.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        ts = line.split("/")[0]
        try:
            t_s = float(ts)
        except ValueError:
            continue
        rels.append((t_s, ts))
    rels.sort()
    print(f"# {len(rels)} sane matrices in pool")

    selected = []
    for phase, (lo, hi), quota in PHASE_QUOTAS:
        in_band = [(t, ts) for t, ts in rels if lo <= t < hi]
        if len(in_band) == 0:
            print(f"  WARN: no matrices in {phase} band [{lo}, {hi})")
            continue
        if len(in_band) <= quota:
            picks = in_band
            print(f"  {phase}: took all {len(in_band)} matrices in band")
        else:
            # Evenly spaced indices
            idx = np.linspace(0, len(in_band)-1, quota, dtype=int)
            picks = [in_band[i] for i in idx]
            print(f"  {phase}: picked {len(picks)} from {len(in_band)}")
        for t, ts in picks:
            selected.append({"t_s": t, "ts": ts, "phase": phase})

    print(f"\n# Total selected: {len(selected)}")
    manifest = {
        "schema_version": "1.0",
        "n_matrices": len(selected),
        "source_pool": str(pool),
        "phases": [p[0] for p in PHASE_QUOTAS],
        "matrices": selected,
    }
    out.write_text(json.dumps(manifest, indent=2))
    print(f"→ {out}")

    # Phase breakdown
    from collections import Counter
    cnt = Counter(m["phase"] for m in selected)
    print("\nPhase breakdown:")
    for p, n in cnt.items():
        print(f"  {p:<12} {n}")


if __name__ == "__main__":
    main()
