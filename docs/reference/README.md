# reference/ — External Papers and Third-Party Material

Read-only directory. Stores external papers, standards, baseline data, and
third-party method documents.

**Do not modify original content.** Editorial notes may be added with explicit
`[Editor's note]` markers, but the original text must not be altered.

## Files

| File | Content | Source |
|------|---------|--------|
| `Barkhudarov_Lagrangian_VOF.md` | Barkhudarov (2004) "Lagrangian VOF Advection Method for FLOW-3D" — full-text Markdown conversion | External paper |
| `Matrix_Output_Guide_external.md` | PCG-internal-hook method for extracting OpenFOAM pressure-equation matrices via patching `src/OpenFOAM/.../PCG.C`. Original from collaborator (Mei Yang). | External method |
| `PICT.md` | Franz, Wei, Guastoni, Thuerey (TUM) "PICT — Differentiable PISO Solver for Simulation-Coupled Learning". Full preprint (~2.7 MB, embedded figures). | External paper |
| `PICT_code_analysis.md` | Our **annotation** of PICT's source — what their solver actually does, file:line references, comparison vs NVIDIA `pbicgstab.cpp` and our DILU/AMGx stack. Includes empirical test of PICT's rank-deficient correction on our LPBF pd matrix. | Editorial (ours) |

## When to use

- **Comparing approaches**: when authoring a `specs/*.md` document, cite the
  reference here as the conceptual source. Our `specs/` describes what *we*
  built; `reference/` describes what others did.
- **Validating against published baselines**: papers in `reference/` are the
  ground truth for V-conservation, shape-error metrics, etc.

## See also

- `/reference/Flow3Dmanual/theory/` contains the complete FLOW-3D v2022R1
  Theory Manual in Markdown (~100 files), covering Navier-Stokes, VOF,
  turbulence, solidification, etc.
- For our own (DILU-Research) methodology specs, see `../specs/`. Specifically
  `MATRIX_EXTRACTION_SPEC.md` is our solver-level-hook variant of the
  extraction technique documented here in `Matrix_Output_Guide_external.md`.

## Authority

External material in this directory is **immutable external truth**. When our
specs (`specs/*.md`) reproduce something documented here, the reference is the
authoritative source for the original technique; our spec is the authoritative
description of our implementation.
