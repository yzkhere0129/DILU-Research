"""Canonical config JSON verbatim test (REPRODUCE.md §5).

The 9 config strings in `python/config.py` are reproduced character-for-
character in REPRODUCE.md. A bit-exact reproducer of this module MUST emit
the same characters; any reformatting that changes a numeric literal, key
name, ordering, or whitespace WILL change AMGx's observable iteration
counts (because the configs are parsed by hash-based `AMGX_config_create`
and AMG hierarchy construction depends on every key).

This test:
  1. Verifies each canonical config parses as valid JSON.
  2. Asserts the key invariants (outer solver, preconditioner algorithm,
     smoother, scaling) that REPRODUCE.md §5 promises.
  3. SHA-256-hashes each config string; if a hash drifts, regenerate the
     stored hash deliberately (and update REPRODUCE.md §5 to match).

The hash table is generated on first run if missing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dilu.amgx.python import config as cfg


GOLDEN_HASH_FILE = Path(__file__).parent / "fixtures" / "config_hashes.json"

CANONICAL = {
    "CLASSICAL_V_CYCLE":                cfg.CLASSICAL_V_CYCLE,
    "AGGRESSIVE_COARSENING":            cfg.AGGRESSIVE_COARSENING,
    "CLASSICAL_V_DIAGSCALED":           cfg.CLASSICAL_V_DIAGSCALED,
    "CLASSICAL_V_DIAGSCALED_TIGHT":     cfg.CLASSICAL_V_DIAGSCALED_TIGHT,
    "CLASSICAL_V_DIAGSCALED_BICGSTAB":  cfg.CLASSICAL_V_DIAGSCALED_BICGSTAB,
    "CLASSICAL_GS_PCG":                 cfg.CLASSICAL_GS_PCG,
    "CLASSICAL_GS_BICGSTAB":            cfg.CLASSICAL_GS_BICGSTAB,
    "AGGREGATION_PCG":                  cfg.AGGREGATION_PCG,
    "MINI_AMG_TEST":                    cfg.MINI_AMG_TEST,
}


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_or_seed_golden():
    if GOLDEN_HASH_FILE.exists():
        return json.loads(GOLDEN_HASH_FILE.read_text())
    GOLDEN_HASH_FILE.parent.mkdir(exist_ok=True)
    golden = {name: _h(s) for name, s in CANONICAL.items()}
    GOLDEN_HASH_FILE.write_text(json.dumps(golden, indent=2, sort_keys=True))
    print(f"  seeded golden config hashes at {GOLDEN_HASH_FILE}")
    return golden


@pytest.mark.parametrize("name,raw", list(CANONICAL.items()))
def test_config_parses_as_json(name, raw):
    """Every canonical config must be valid JSON."""
    parsed = json.loads(raw)
    assert "solver" in parsed, f"{name}: missing top-level 'solver'"
    assert parsed.get("config_version") == 2, (
        f"{name}: config_version must be 2"
    )


def test_pd_solver_is_pcg_and_T_solver_is_bicgstab():
    """The headline routing invariant: pd → PCG, T → BICGSTAB."""
    pd = json.loads(cfg.CLASSICAL_V_DIAGSCALED)["solver"]
    assert pd["solver"] == "PCG", f"pd outer must be PCG, got {pd['solver']!r}"
    assert pd["scaling"] == "DIAGONAL_SYMMETRIC"
    T = json.loads(cfg.CLASSICAL_V_DIAGSCALED_BICGSTAB)["solver"]
    assert T["solver"] == "BICGSTAB", (
        f"T outer must be BICGSTAB, got {T['solver']!r}"
    )
    assert T["scaling"] == "DIAGONAL_SYMMETRIC"


def test_classical_amg_uses_d2_interpolator_block_jacobi_smoother():
    """Production preconditioner invariants."""
    pd = json.loads(cfg.CLASSICAL_V_DIAGSCALED)["solver"]["preconditioner"]
    assert pd["algorithm"] == "CLASSICAL"
    assert pd["interpolator"] == "D2"
    assert pd["smoother"]["solver"] == "BLOCK_JACOBI"
    assert pd["presweeps"] == 1 and pd["postsweeps"] == 1


def test_config_hashes_match_golden():
    """SHA-256 of each canonical string must match the stored golden hash."""
    golden = _load_or_seed_golden()
    drift = []
    for name, raw in CANONICAL.items():
        observed = _h(raw)
        if name not in golden:
            drift.append(f"{name}: not in golden table — was it just added?")
        elif golden[name] != observed:
            drift.append(
                f"{name}: hash drift\n"
                f"    golden:   {golden[name]}\n"
                f"    observed: {observed}\n"
                f"  If this drift is intentional, regenerate "
                f"{GOLDEN_HASH_FILE.name} AND update REPRODUCE.md §5."
            )
    assert not drift, "Canonical config strings have drifted:\n" + "\n".join(drift)


def test_with_tolerance_preserves_structure():
    """with_tolerance() must only change tolerance and max_iters, not anything else."""
    base = json.loads(cfg.CLASSICAL_V_DIAGSCALED)
    patched = json.loads(cfg.with_tolerance(cfg.CLASSICAL_V_DIAGSCALED, 1e-12,
                                            max_iters=999))
    assert patched["solver"]["tolerance"] == 1e-12
    assert patched["solver"]["max_iters"] == 999
    # Every other key in solver must be identical
    for k, v in base["solver"].items():
        if k in ("tolerance", "max_iters"):
            continue
        assert patched["solver"][k] == v, (
            f"with_tolerance changed solver.{k}: {v!r} -> "
            f"{patched['solver'][k]!r}"
        )
