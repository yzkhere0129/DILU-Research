"""Verify lab machine has correct senior data — before running LU on it.

Prints fingerprint (file count, size, first-row hash) for melting +
evaporation datasets so we can confirm lab matches what user uploaded.

Usage on lab:
    cd ~/DILU-Research && git pull origin main
    python3 -m dilu.amgx.bench.lab_data_fingerprint
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def fingerprint_dataset(name: str, base: Path):
    if not base.exists():
        print(f"\n[{name}]  MISSING: {base}")
        return
    print(f"\n[{name}]  {base}")
    pre = base / "Pre_Solving"
    after = base / "After_Solving"
    for sub_label, sub in [("Pre_Solving", pre), ("After_Solving", after)]:
        if not sub.exists():
            print(f"  {sub_label}: MISSING"); continue
        files = [f for f in sub.iterdir() if not f.name.endswith("Zone.Identifier")]
        sizes = sum(f.stat().st_size for f in files)
        print(f"  {sub_label}:  {len(files)} files,  {sizes/1e9:.2f} GB total")
        # Print bucket counts by prefix
        buckets = {}
        for f in files:
            prefix = f.name.split("_")[0]
            buckets[prefix] = buckets.get(prefix, 0) + 1
        for k, v in sorted(buckets.items()):
            print(f"    {k}_*: {v}")
    # Hash of one specific file's first 4096 bytes (signature)
    pivot = pre / "matrix_pd_70_1.csv"
    if pivot.exists():
        with pivot.open("rb") as f:
            h = hashlib.sha256(f.read(4096)).hexdigest()[:16]
        print(f"  matrix_pd_70_1.csv first-4KB sha256-prefix: {h}")
        print(f"    (size: {pivot.stat().st_size/1e6:.1f} MB)")


def main():
    base = Path(__file__).resolve().parents[2] / "benchmark"
    print(f"Repo benchmark dir: {base}")
    fingerprint_dataset("Melting",     base / "Melting"     / "Melting")
    fingerprint_dataset("Evaporation", base / "Evaporation" / "Evaporation")


if __name__ == "__main__":
    main()
