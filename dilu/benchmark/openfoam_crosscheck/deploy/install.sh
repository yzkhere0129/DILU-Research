#!/bin/bash
# install.sh — deploy DILU-Research matrixDumper into LaserbeamFoam solver
#
# Usage:
#   ./install.sh <path_to_laserMeltFoam_src_dir>
# e.g.:
#   ./install.sh ~/LaserbeamFoam/applications/solvers/laserMeltFoam
#
# What it does:
#   1. Backup original laserMeltFoam.C, TEqn.H, pEqn.H → <file>.orig
#   2. Apply 3 unified diffs to insert matrixDumper calls
#   3. Drop matrixDumper.H into the solver dir
#   4. Run wmake to recompile

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path_to_laserMeltFoam_src>"
    echo "Example: $0 ~/LaserbeamFoam/applications/solvers/laserMeltFoam"
    exit 1
fi

SRC="$(realpath "$1")"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$SRC" ]; then
    echo "ERROR: $SRC does not exist"; exit 1
fi
if [ ! -f "$SRC/laserMeltFoam.C" ]; then
    echo "ERROR: $SRC/laserMeltFoam.C not found — is this the right dir?"
    exit 1
fi

# --- OpenFOAM sanity ---
if [ -z "${WM_PROJECT_VERSION:-}" ]; then
    echo "ERROR: OpenFOAM env not sourced. Run:"
    echo "  source /usr/lib/openfoam/openfoam2506/etc/bashrc   # or your path"
    exit 1
fi
echo "[install] OpenFOAM version: ${WM_PROJECT_VERSION}"
if [ "${WM_PROJECT_VERSION}" != "v2506" ] && [ "${WM_PROJECT_VERSION}" != "2506" ]; then
    echo "[install] WARNING: tested against v2506; you have ${WM_PROJECT_VERSION}."
    echo "[install] Will try anyway — inspect wmake output for errors."
fi

# --- Backup and apply ---
for f in laserMeltFoam.C TEqn.H pEqn.H; do
    if [ ! -f "$SRC/$f" ]; then
        echo "ERROR: $SRC/$f not found"; exit 1
    fi
    if [ -f "$SRC/$f.orig" ]; then
        echo "[install] $SRC/$f.orig already exists — restoring before re-patch"
        cp "$SRC/$f.orig" "$SRC/$f"
    else
        cp "$SRC/$f" "$SRC/$f.orig"
        echo "[install] backed up $f → $f.orig"
    fi
done

# Apply diffs with context-based matching (patch will adjust line offsets)
for f in laserMeltFoam.C TEqn.H pEqn.H; do
    echo "[install] applying patch for $f"
    if ! patch -p0 --no-backup-if-mismatch -d "$SRC" < "$THIS_DIR/$f.diff"; then
        echo ""
        echo "ERROR: patch for $f failed."
        echo "  This means your $f differs from the LaserbeamFoam upstream"
        echo "  against which the diff was prepared."
        echo "  Options:"
        echo "    (a) inspect $THIS_DIR/$f.diff and apply manually"
        echo "    (b) copy $THIS_DIR/$f.patched wholesale (if your $f is unmodified)"
        echo ""
        exit 1
    fi
done

cp "$THIS_DIR/matrixDumper.H" "$SRC/matrixDumper.H"
echo "[install] copied matrixDumper.H"

# --- Build ---
echo "[install] running wmake in $SRC"
cd "$SRC"
wmake 2>&1 | tail -20
if [ $? -ne 0 ]; then
    echo "ERROR: wmake failed"; exit 1
fi

echo ""
echo "[install] SUCCESS"
echo "  New binary: ${FOAM_USER_APPBIN}/laserMeltFoam"
echo "  Original sources saved as *.orig in $SRC"
echo ""
echo "Next: apply case patch with patch_case.sh"
