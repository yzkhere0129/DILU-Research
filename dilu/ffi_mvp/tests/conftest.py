"""Test harness — ensures jax_enable_x64 and VRAM safety rails before any jax import.

The kernel and reference are float64 end-to-end; x64 must be on during tracing.
VRAM rails are HARD constraints on this 4 GB laptop box (CLAUDE-brief §safety):
XLA preallocation OFF, capped at 0.5 (~1.5 GB), platform allocator so buffers
are released on free rather than pooled.
"""
import os
import sys

# MUST precede any `import jax` in this process. conftest is imported first by
# pytest and by the `import conftest` at the head of each test file.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)

# Make dilu/ffi_mvp/python importable as a plain package without installing.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
