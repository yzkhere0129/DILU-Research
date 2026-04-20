#!/usr/bin/env bash
# Wrap the bench harness under Nsight Systems for kernel + launch attribution.
# Optional: skip gracefully if nsys is not installed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${SCRIPT_DIR}/baseline_rtx3050.qdrep"

if ! command -v nsys >/dev/null; then
  echo "nsys not on PATH; skipping capture (Phase 1 allowed-to-skip)" >&2
  exit 0
fi

nsys profile \
  -t cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  -o "${OUT}" \
  python3 "${SCRIPT_DIR}/bench_ffi_dispatch.py"

echo "Capture written to ${OUT}"
