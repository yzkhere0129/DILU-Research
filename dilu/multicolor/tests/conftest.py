"""Phase 3 test harness — inherits Phase 1/2 VRAM rails + jax_enable_x64.

Hardware safety (brief §HARDWARE SAFETY):
  - XLA env vars MUST be set before `import jax`.
  - One test per subprocess (-n 0). The platform allocator frees on process
    exit, keeping the 4 GB VRAM rail honest.
"""
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from jax import config as _jax_config  # noqa: E402
_jax_config.update("jax_enable_x64", True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# Also expose tests/ dir so `from _harness import ...` works like Phase 2.
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
