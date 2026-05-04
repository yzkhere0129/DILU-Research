# Guide: Outputting the DICPCG Matrix from OpenFOAM

**Purpose:** This document describes how to modify the PCG solver source code in OpenFOAM-v2412 to output the linear system (matrix A, source vector b, solution vector x) solved by DICPCG at the first time step.

**Target case:** LPBF\_tutorial with laserMeltFoam (serial run)

\---

## 1\. Background

DICPCG consists of two components:

* **DIC** (Diagonal Incomplete Cholesky) -- the preconditioner
* **PCG** (Preconditioned Conjugate Gradient) -- the solver

In the LPBF\_tutorial case, DICPCG is used for the pressure equations only:

* `pd` -- dynamic pressure correction (PIMPLE corrections 1 and 2)
* `pdFinal` -- final pressure correction (PIMPLE correction 3, the last one)

The `pcorr.\*` entry exists in `fvSolution` but the laserMeltFoam solver never solves a `pcorr` field, so it has no effect.

Other equations use different solvers:

* Temperature (`T`) -- PBiCG + DILU
* Momentum (`U`) -- PBiCGStab + DILU (momentum predictor is disabled, so U is not solved directly)

The pressure matrix is assembled in `pEqn.H` as:

```cpp
fvScalarMatrix pdEqn(fvm::laplacian(rUAf, pd) == fvc::div(phi));
```

This produces a **symmetric** matrix, meaning `upper()` and `lower()` are the same data.

### How many times is the pressure equation solved per time step?

The PIMPLE loop in `laserMeltFoam.C` calls `pEqn.H` inside `while (pimple.correct())`. With `nOuterCorrectors 1` and `nCorrectors 3`, the pressure equation is solved **3 times per time step**:

|Solve|Field name in PCG|fvSolution entry used|
|-|-|-|
|1st PIMPLE correction|`"pd"`|`pd`|
|2nd PIMPLE correction|`"pd"`|`pd`|
|3rd (final) PIMPLE correction|`"pdFinal"`|`pdFinal`|

The field name switches to `"pdFinal"` on the last correction because `pEqn.H` calls:

```cpp
pdEqn.solve(mesh.solutionDict(pd.select(pimple.finalInnerIter())));
```

which selects `pdFinal` when `pimple.finalInnerIter()` is true.

\---

## 2\. OpenFOAM's LDU Matrix Storage

OpenFOAM stores the matrix in LDU (Lower-Diagonal-Upper) format, which is tied to mesh face connectivity:

* `diag()` -- diagonal coefficients, size = nCells
* `upper()` -- upper off-diagonal coefficients, size = nFaces
* `lower()` -- lower off-diagonal coefficients, size = nFaces (same as upper for symmetric matrices)
* `lduAddr().lowerAddr()` -- owner cell index for each face
* `lduAddr().upperAddr()` -- neighbour cell index for each face

Each internal mesh face connects an owner cell and a neighbour cell, producing one upper and one lower off-diagonal entry.

\---

## 3\. Output Format: COO (Coordinate)

The output uses COO sparse format, which stores each non-zero entry as a `(row, col, value)` triplet. The mapping from LDU to COO is:

* Diagonal: `diag\[i]` --> `(i, i, diag\[i])`
* Upper triangle: `upper\[face]` --> `(lowerAddr\[face], upperAddr\[face], upper\[face])`
* Lower triangle: `lower\[face]` --> `(upperAddr\[face], lowerAddr\[face], lower\[face])`

For symmetric matrices, `upper\[face]` and `lower\[face]` are the same value.

\---

## 4\. File to Modify

**File path:**

```
OpenFOAM-v2412/src/OpenFOAM/matrices/lduMatrix/solvers/PCG/PCG.C
```

\---

## 5\. Modification Steps

### Step 1: Add Header

At the top of `PCG.C`, after line 30:

```cpp
#include "PrecisionAdaptor.H"
```

Add:

```cpp
#include "OFstream.H"
#include "Time.H"
```

### Step 2: Add Dump Code

After line 100:

```cpp
solveScalarField rA(source - wA);
```

Insert the following block:

```cpp
    // --- Dump matrix, source, and solution at first time step
    if (matrix().mesh().thisDb().time().timeIndex() == 1)
    {
        static int pcgDumpCount = 0;
        pcgDumpCount++;

        const labelUList\& owner = matrix\_.lduAddr().lowerAddr();
        const labelUList\& neighbour = matrix\_.lduAddr().upperAddr();
        const scalarField\& diagCoeffs = matrix\_.diag();
        const scalarField\& upperCoeffs = matrix\_.upper();

        label nFaces = owner.size();

        // Write matrix in COO format
        OFstream matrixFile
        (
            "matrix\_" + fieldName\_ + "\_" + Foam::name(pcgDumpCount) + ".txt"
        );

        // Header: nRows nCols nNonZeros
        matrixFile << nCells << " " << nCells << " "
                   << (nCells + 2\*nFaces) << nl;

        // Diagonal entries (row, col, value)
        for (label i = 0; i < nCells; i++)
        {
            matrixFile << i << " " << i << " " << diagCoeffs\[i] << nl;
        }

        // Upper triangle entries (owner -> neighbour)
        for (label face = 0; face < nFaces; face++)
        {
            matrixFile << owner\[face] << " " << neighbour\[face]
                       << " " << upperCoeffs\[face] << nl;
        }

        // Lower triangle entries (neighbour -> owner, same values for symmetric)
        for (label face = 0; face < nFaces; face++)
        {
            matrixFile << neighbour\[face] << " " << owner\[face]
                       << " " << upperCoeffs\[face] << nl;
        }

        // Write source vector (b in Ax = b)
        OFstream sourceFile
        (
            "source\_" + fieldName\_ + "\_" + Foam::name(pcgDumpCount) + ".txt"
        );
        for (label i = 0; i < nCells; i++)
        {
            sourceFile << i << " " << source\[i] << nl;
        }

        // Write solution vector (x in Ax = b)
        OFstream solutionFile
        (
            "solution\_" + fieldName\_ + "\_" + Foam::name(pcgDumpCount) + ".txt"
        );
        for (label i = 0; i < nCells; i++)
        {
            solutionFile << i << " " << psi\[i] << nl;
        }
    }
```

### Step 3: Recompile

```bash
cd $WM\_PROJECT\_DIR/src/OpenFOAM
wmake
```

### Step 4: Run the Case (Serial)

```bash
cd /path/to/LPBF\_tutorial\_1
laserMeltFoam
```

\---

## 6\. Output Files

After the first time step completes, the following 9 files appear in the case run directory:

|File|Contents|PIMPLE Correction|
|-|-|-|
|`matrix\_pd\_1.txt`|Pressure matrix A (COO format)|1st correction|
|`source\_pd\_1.txt`|Right-hand side vector b|1st correction|
|`solution\_pd\_1.txt`|Solution vector x (initial guess)|1st correction|
|`matrix\_pd\_2.txt`|Pressure matrix A (COO format)|2nd correction|
|`source\_pd\_2.txt`|Right-hand side vector b|2nd correction|
|`solution\_pd\_2.txt`|Solution vector x|2nd correction|
|`matrix\_pdFinal\_3.txt`|Pressure matrix A (COO format)|3rd (final) correction|
|`source\_pdFinal\_3.txt`|Right-hand side vector b|3rd (final) correction|
|`solution\_pdFinal\_3.txt`|Solution vector x|3rd (final) correction|

The matrix and source vector change between corrections because:

* The `rUAf` coefficient (from the momentum equation) is updated between corrections
* The `phi` flux (right-hand side) is corrected after each pressure solve
* The solution vector `psi` carries the result from the previous correction as the initial guess for the next one

### File format

Matrix files have one header line followed by COO triplets:

```
1024000 1024000 7000000
0 0 1.234e+06
1 1 1.234e+06
...
0 1 -5.678e+05
1 2 -5.678e+05
...
1 0 -5.678e+05
2 1 -5.678e+05
...
```

Source and solution files have one entry per line:

```
0 3.456e-08
1 2.789e-08
...
```

\---

## 7\. Notes

* The pressure matrix is **symmetric**, so the upper and lower triangle values are identical. Both are written using `upperCoeffs` in the code above.
* `fieldName\_` is an inherited member variable of the solver class. It will be `"pd"` for corrections 1-2 and `"pdFinal"` for correction 3.
* `pcgDumpCount` is a `static` variable, so it persists across all calls to `scalarSolve()` and increments each time PCG is invoked. This ensures each pressure solve gets a unique file.
* `nCells` is already defined at line 83 of the original `PCG.C`.
* `nl` is OpenFOAM's newline constant, equivalent to `"\\n"`.
* The output files are written to the **case run directory** (the directory from which you execute the solver).
* For a 1,024,000-cell mesh, each matrix file will contain approximately 7 million lines and may be several hundred MB in size. With 3 pressure solves, expect roughly 1-2 GB total output.
* The time step check `timeIndex() == 1` ensures the dump only happens at the first time step. Change this value to dump at a different time step.
* The `pcorr.\*` entry in `fvSolution` is unused by laserMeltFoam and will not produce any output.

