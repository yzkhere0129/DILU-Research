"""Phase 4 AMGx test harness — inherits Phase 2's VRAM rails and float64 default.

One test per subprocess is the documented convention (see Phase 2 report §2).
Do NOT run pytest in parallel on this 4 GB GPU.
"""
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
