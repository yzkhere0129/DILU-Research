#!/usr/bin/env python3
"""Apply matrixDumper integration to a fresh v2412 laserMeltFoam clone.

Run this on lab Xeon AFTER `git pull` and AFTER `git clone laserMeltFoam`:

    cd ~/DILU-Research && git pull origin main
    python3 dilu/openfoam_patches/laserMeltFoam/apply_patches.py \\
        --dest ~/src/laserMeltFoam/laserMeltFoam

What it does:
  1. Copies matrixDumper.H into <dest>/
  2. Copies TEqn.H into <dest>/      (overwrites — diff verified to be hook-only)
  3. Copies pEqn.H into <dest>/      (overwrites — diff verified to be hook-only)
  4. Surgical-patches <dest>/laserMeltFoam.C with two string-level inserts
     (NEVER overwrites the file — preserves all v2412 logic)

Idempotent — if patches already applied (matrixDumper occurrences already
present), it skips and reports.

After running: `cd ~/src/laserMeltFoam && ./Allwmake -j8`
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def patch_main_C(dest: Path) -> int:
    """Insert matrixDumper include + instance into laserMeltFoam.C. Idempotent.

    Returns: 0 = applied, 1 = already applied, 2 = anchor not found (FATAL).
    """
    text = dest.read_text()
    n_existing = text.count("matrixDumper")
    if n_existing >= 2:
        print(f"  [skip] laserMeltFoam.C already has {n_existing} matrixDumper "
              f"references — assumed already patched.")
        return 1

    # ---- Patch 1: insert matrixDumper.H include after interpolationTable.H ----
    anchor1 = '#include "interpolationTable.H"\n'
    payload1 = (
        '#include "interpolationTable.H"\n'
        '\n'
        '// DILU-Research crosscheck matrix dumper\n'
        '#include "matrixDumper.H"\n'
    )
    if anchor1 not in text:
        # Fallback anchor: any solver should have fvCFD.H
        anchor1 = '#include "fvCFD.H"\n'
        payload1 = (
            '#include "fvCFD.H"\n'
            '\n'
            '// DILU-Research crosscheck matrix dumper\n'
            '#include "matrixDumper.H"\n'
        )
        if anchor1 not in text:
            print(f"  [FAIL] Cannot find anchor for include patch in laserMeltFoam.C")
            print(f"         Searched for both interpolationTable.H and fvCFD.H — "
                  f"neither found.")
            return 2
    text = text.replace(anchor1, payload1, 1)

    # ---- Patch 2: insert matrixDumper instance after createFields.H ----
    # Anchor with 4-space indent (inside main() body)
    anchor2 = '    #include "createFields.H"\n'
    payload2 = (
        '    #include "createFields.H"\n'
        '\n'
        '    // DILU-Research matrixDumper: reads system/matrixDumperDict if present.\n'
        '    Foam::matrixDumper matrixDumper_(runTime);\n'
    )
    if anchor2 not in text:
        print(f"  [FAIL] Cannot find anchor for instance patch in laserMeltFoam.C")
        print(f"         Looked for indented '#include \"createFields.H\"' inside main()")
        return 2
    text = text.replace(anchor2, payload2, 1)

    dest.write_text(text)
    n_after = text.count("matrixDumper")
    print(f"  [ok]   patched laserMeltFoam.C: {n_existing} → {n_after} "
          f"matrixDumper references")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True,
                     help="Path to fresh laserMeltFoam solver dir, e.g. "
                          "~/src/laserMeltFoam/laserMeltFoam")
    ap.add_argument("--no-backup", action="store_true",
                     help="Skip making .v2412orig backup")
    args = ap.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    if not dest.exists():
        print(f"ERROR: dest dir does not exist: {dest}")
        sys.exit(1)

    expected_files = ["laserMeltFoam.C", "TEqn.H", "pEqn.H"]
    for f in expected_files:
        if not (dest / f).exists():
            print(f"ERROR: {dest / f} not found — is this really the laserMeltFoam "
                  f"solver dir?")
            sys.exit(1)

    print(f"Source dir : {THIS_DIR}")
    print(f"Dest dir   : {dest}")
    print()

    # ---- Step 1: backup originals ----
    if not args.no_backup:
        for f in expected_files:
            orig = dest / f
            backup = dest / (f + ".v2412orig")
            if not backup.exists():
                shutil.copy2(orig, backup)
                print(f"  [backup] {f} → {f}.v2412orig")
            else:
                print(f"  [skip ] {f}.v2412orig already exists")
        print()

    # ---- Step 2: copy matrixDumper.H (new file) ----
    src_md = THIS_DIR / "matrixDumper.H"
    dst_md = dest / "matrixDumper.H"
    shutil.copy2(src_md, dst_md)
    print(f"  [cp]   matrixDumper.H → {dst_md.name}")

    # ---- Step 3: copy TEqn.H + pEqn.H (overwrite — diff was hook-only) ----
    for f in ["TEqn.H", "pEqn.H"]:
        shutil.copy2(THIS_DIR / f, dest / f)
        print(f"  [cp]   {f} (overwrites v2412 fresh — hook-only diff verified)")

    # ---- Step 4: surgical-patch laserMeltFoam.C ----
    print(f"  [patch] laserMeltFoam.C ...")
    rc = patch_main_C(dest / "laserMeltFoam.C")
    if rc == 2:
        print(f"\nFATAL: laserMeltFoam.C patch failed.  Check anchors above.")
        sys.exit(2)

    # ---- Step 5: verify ----
    print()
    print("=== Verification ===")
    for f in ["laserMeltFoam.C", "TEqn.H", "pEqn.H"]:
        path = dest / f
        n = path.read_text().count("matrixDumper")
        print(f"  {f}: {n} matrixDumper references")
    print()
    print("Done.  Now rebuild:")
    print("  source /usr/lib/openfoam/openfoam2412/etc/bashrc")
    print("  cd ~/src/laserMeltFoam && ./Allwmake -j8")


if __name__ == "__main__":
    main()
