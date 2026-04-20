"""Phase 2 test harness — inherits Phase 1's VRAM rails and jax_enable_x64.

One test per subprocess (the platform allocator frees on exit — see Phase 1
report §2). Do NOT run pytest in parallel (-n>0) on this machine.
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
