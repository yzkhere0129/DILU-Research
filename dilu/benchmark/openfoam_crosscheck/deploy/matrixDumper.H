/*---------------------------------------------------------------------------*\
  matrixDumper.H
  ---------------
  Per-solve (A, b, x0, x_final) extractor for DILU-Research OpenFOAM
  cross-check. Triggered per-timestep whitelist from system/matrixDumperDict.

  Dumps the EFFECTIVE linear system that the underlying lduMatrix::solver
  sees AT SOLVE ENTRY — i.e.:

      diag_effective  = matrix.diag() + Σ_patches internalCoeffs     (scalarField)
      upper, lower    = matrix.upper(), matrix.lower()               (per-face)
      source_effective= matrix.source() + Σ_patches boundaryCoeffs   (scalarField)

  so that A_effective @ x_openfoam_final ≈ b_effective holds to OpenFOAM tol,
  and A_effective, b_effective are byte-for-byte what cuSPARSE / AMGx will
  consume on the Python side.

  Output layout per dumped (time, equation, corrector):
      postProcessing/matrices/<timeName>/<eqName>_corr<k>/
          ├── A.mm            (MatrixMarket coordinate, real, general)
          ├── b.mm            (MatrixMarket array, real, general, N×1)
          ├── x0.mm           (ditto, initial guess, N×1)
          ├── x_final.mm      (ditto, post-solve, N×1)
          └── metadata.json

  Scalar fvMatrix only (pd, T). Vector/tensor extension deferred.

  Part of: DILU-Research / openfoam_crosscheck
\*---------------------------------------------------------------------------*/

#ifndef matrixDumper_H
#define matrixDumper_H

#include "Time.H"
#include "fvMatrix.H"
#include "fvMatrices.H"
#include "lduAddressing.H"
#include "IOdictionary.H"
#include "fileName.H"
#include "OSspecific.H"
#include "SolverPerformance.H"
#include "GeometricField.H"
#include "volMesh.H"
#include "volFields.H"
#include "fvPatchField.H"
#include "Switch.H"

#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <unordered_set>
#include <utility>
#include <chrono>

namespace Foam
{

class matrixDumper
{
    // -- State
    const Time& runTime_;
    bool        enabled_;
    bool        binaryMM_;        // reserved; ASCII only in v1.0
    fileName    outputDir_;       // relative to case dir
    std::unordered_set<label> dumpStepsSet_;
    wordList    equationFilter_;  // empty = all

    // Per-equation max corrector count to dump (missing key = unlimited)
    std::map<word, label> maxCorrPerEq_;

    // -- Auto corrector-index tracker (resets each new time step)
    mutable label                           lastTimeIdx_;
    mutable std::map<word, label>           eqCounters_;
    mutable std::set<std::pair<word,label>> dumpedCorrs_;  // (eq, idx) actually written

    inline void resetCountersIfNewStep_() const
    {
        if (runTime_.timeIndex() != lastTimeIdx_)
        {
            eqCounters_.clear();
            dumpedCorrs_.clear();
            lastTimeIdx_ = runTime_.timeIndex();
        }
    }

    inline label nextCorrIdx_(const word& eqName) const
    {
        resetCountersIfNewStep_();
        return eqCounters_[eqName]++;
    }

    inline label peekCorrIdx_(const word& eqName) const
    {
        resetCountersIfNewStep_();
        auto it = eqCounters_.find(eqName);
        return (it == eqCounters_.end()) ? 0 : (it->second - 1);
    }

    inline bool withinMaxCorr_(const word& eqName, label idx) const
    {
        auto it = maxCorrPerEq_.find(eqName);
        return (it == maxCorrPerEq_.end()) ? true : (idx < it->second);
    }

    // First timeIndex this dumper ever saw. Set lazily on first call.
    // Used to compute a RELATIVE step index so dumpTimeSteps=(1..N) means
    // "first N time steps AFTER the dumper was live", regardless of whether
    // the run started from startTime=0 or resumed from latestTime.
    mutable label firstTimeIdx_ = -1;

    // -- Internal helpers
    inline bool stepMatches_() const
    {
        if (firstTimeIdx_ < 0) firstTimeIdx_ = runTime_.timeIndex();
        const label relIdx = runTime_.timeIndex() - firstTimeIdx_ + 1;
        return dumpStepsSet_.count(relIdx) != 0;
    }

    inline fileName matDir_(const word& eqName, const label k) const
    {
        return runTime_.path() / outputDir_
             / runTime_.timeName()
             / (eqName + "_corr" + Foam::name(k));
    }

    // -- Write a MatrixMarket coordinate real general file (COO)
    template<class CooTriple>
    static void writeMMCoo_(
        const fileName& path,
        label nRows, label nCols, label nnz,
        const CooTriple& cooEmitter,
        const word& comment
    );

    // -- Write a MatrixMarket array real general (N x 1 vector)
    static void writeMMArray_(
        const fileName& path,
        const scalarField& v,
        const word& comment
    );

    // -- Write metadata.json (scalar-only)
    void writeMeta_(
        const fileName& dir,
        const fvScalarMatrix& matrix,
        const word& eqName,
        label correctorIdx,
        const SolverPerformance<scalar>* perf,   // nullable (pre-solve)
        label nRows, label nnz
    ) const;

public:
    // Constructor: reads system/matrixDumperDict if present; else disabled.
    matrixDumper(const Time& runTime);

    inline bool enabled() const { return enabled_; }
    inline bool shouldDumpNow() const { return enabled_ && stepMatches_(); }

    inline bool shouldDumpEq(const word& eqName) const
    {
        if (!shouldDumpNow()) return false;
        if (equationFilter_.empty()) return true;
        forAll(equationFilter_, i)
            if (equationFilter_[i] == eqName) return true;
        return false;
    }

    // === Pre-solve: dump A, b, x0, metadata (scalar-only v1.0)
    //     correctorIdx auto-assigned (resets per time step).
    void dumpPreSolve(
        const fvScalarMatrix& matrix,
        const word& eqName,
        const volScalarField& x0
    );

    // === Post-solve: dump x_final + update metadata (solver perf)
    //     Pairs with the most recent pre-solve of the same eqName.
    void dumpPostSolve(
        const word& eqName,
        const volScalarField& xFinal,
        const SolverPerformance<scalar>& perf,
        const fvScalarMatrix& matrix
    );
};

// =========================================================================
// Implementation (header-only for solver-local utility)
// =========================================================================

inline matrixDumper::matrixDumper(const Time& runTime)
:
    runTime_(runTime),
    enabled_(false),
    binaryMM_(false),
    outputDir_("postProcessing/matrices"),
    dumpStepsSet_(),
    equationFilter_(),
    maxCorrPerEq_(),
    lastTimeIdx_(-1),
    eqCounters_(),
    dumpedCorrs_()
{
    const fileName dictPath = runTime.system() / "matrixDumperDict";
    const fileName absDictPath = runTime.path() / dictPath;

    if (!isFile(absDictPath))
    {
        Info<< "[matrixDumper] No system/matrixDumperDict — DISABLED." << endl;
        return;
    }

    IOdictionary dict(
        IOobject(
            "matrixDumperDict",
            runTime.system(),
            runTime,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false   // registerObject = false
        )
    );

    enabled_   = dict.getOrDefault<bool>("enabled", false);
    binaryMM_  = dict.getOrDefault<bool>("binaryMM", false);
    outputDir_ = dict.getOrDefault<fileName>("outputDir", "postProcessing/matrices");

    labelList steps(dict.getOrDefault<labelList>("dumpTimeSteps", labelList()));
    for (const label s : steps) dumpStepsSet_.insert(s);

    equationFilter_ = dict.getOrDefault<wordList>("equations", wordList());

    // Read maxCorrectorsPerEq subdict (optional):  { pd  3;  T  2; }
    if (dict.found("maxCorrectorsPerEq"))
    {
        const dictionary& sub = dict.subDict("maxCorrectorsPerEq");
        for (const auto& e : sub)
        {
            const word& key = e.keyword();
            maxCorrPerEq_[key] = readLabel(sub.lookup(key));
        }
    }

    if (binaryMM_)
    {
        WarningInFunction
            << "binaryMM=true requested but not implemented in v1.0; "
            << "falling back to ASCII." << endl;
        binaryMM_ = false;
    }

    Info<< "[matrixDumper] ENABLED. outputDir=" << outputDir_
        << "  dumpTimeSteps=" << steps
        << "  equations=" << (equationFilter_.empty() ? wordList({"*"}) : equationFilter_)
        << endl;
}

template<class CooTriple>
void matrixDumper::writeMMCoo_
(
    const fileName& path,
    label nRows, label nCols, label nnz,
    const CooTriple& cooEmitter,
    const word& comment
)
{
    std::ofstream os(path.c_str());
    if (!os)
    {
        FatalErrorInFunction << "Cannot open " << path << " for write"
            << exit(FatalError);
    }
    os << "%%MatrixMarket matrix coordinate real general\n";
    os << "% " << comment << "\n";
    os << "% generated by DILU-Research matrixDumper v1.0\n";
    os << nRows << ' ' << nCols << ' ' << nnz << '\n';
    os << std::scientific << std::setprecision(17);
    cooEmitter(os);   // 1-based indices per MM spec
    os.close();
}

inline void matrixDumper::writeMMArray_
(
    const fileName& path,
    const scalarField& v,
    const word& comment
)
{
    std::ofstream os(path.c_str());
    if (!os)
    {
        FatalErrorInFunction << "Cannot open " << path << " for write"
            << exit(FatalError);
    }
    os << "%%MatrixMarket matrix array real general\n";
    os << "% " << comment << "\n";
    os << "% generated by DILU-Research matrixDumper v1.0\n";
    os << v.size() << ' ' << 1 << '\n';
    os << std::scientific << std::setprecision(17);
    forAll(v, i) os << v[i] << '\n';
    os.close();
}

inline void matrixDumper::writeMeta_
(
    const fileName& dir,
    const fvScalarMatrix& matrix,
    const word& eqName,
    label correctorIdx,
    const SolverPerformance<scalar>* perf,
    label nRows, label nnz
) const
{
    std::ofstream os((dir / "metadata.json").c_str());
    if (!os)
    {
        FatalErrorInFunction << "Cannot open metadata.json in " << dir
            << exit(FatalError);
    }

    os << std::scientific << std::setprecision(17);
    os << "{\n";
    os << "  \"schema_version\": \"1.0\",\n";
    os << "  \"case\": \"" << runTime_.path().name() << "\",\n";
    os << "  \"openfoam_version\": \"v2506\",\n";
    os << "  \"time\": {\n";
    os << "    \"step_index\": " << runTime_.timeIndex() << ",\n";
    os << "    \"time_value\": " << runTime_.value() << ",\n";
    os << "    \"delta_t\":    " << runTime_.deltaTValue() << "\n";
    os << "  },\n";
    os << "  \"equation\": {\n";
    os << "    \"name\": \"" << eqName << "\",\n";
    os << "    \"corrector_index\": " << correctorIdx << "\n";
    os << "  },\n";
    os << "  \"matrix\": {\n";
    os << "    \"n_rows\": " << nRows << ",\n";
    os << "    \"n_cols\": " << nRows << ",\n";
    os << "    \"nnz\":    " << nnz << ",\n";
    os << "    \"has_upper\": " << (matrix.hasUpper() ? "true" : "false") << ",\n";
    os << "    \"has_lower\": " << (matrix.hasLower() ? "true" : "false") << ",\n";
    os << "    \"symmetric\": "
       << (matrix.symmetric() ? "true" : "false") << "\n";
    os << "  }";
    if (perf)
    {
        os << ",\n";
        os << "  \"solver_openfoam\": {\n";
        os << "    \"initial_residual\": " << perf->initialResidual() << ",\n";
        os << "    \"final_residual\":   " << perf->finalResidual()   << ",\n";
        os << "    \"iterations\":       " << perf->nIterations()     << ",\n";
        os << "    \"converged\":        " << (perf->converged() ? "true" : "false") << "\n";
        os << "  }\n";
    }
    else
    {
        os << "\n";
    }
    os << "}\n";
    os.close();
}

inline void matrixDumper::dumpPreSolve
(
    const fvScalarMatrix& matrix,
    const word& eqName,
    const volScalarField& x0
)
{
    if (!shouldDumpEq(eqName)) return;

    const label correctorIdx = nextCorrIdx_(eqName);
    if (!withinMaxCorr_(eqName, correctorIdx))
    {
        // Over cap — do not emit files; post-solve will also skip via dumpedCorrs_
        return;
    }

    const fileName dir = matDir_(eqName, correctorIdx);
    mkDir(dir);
    dumpedCorrs_.insert({eqName, correctorIdx});

    // ---- Effective diagonal (public D() = diag + boundary internalCoeffs) ----
    //   fvMatrix::D() internally calls the protected addBoundaryDiag,
    //   returning the exact diag seen by lduMatrix::solver.
    scalarField diagFull(matrix.D()());

    // ---- Off-diagonal entries ----
    // For symmetric matrices (pd with DIC), matrix.lower() aliases upper.
    // For asymmetric (T with DILU), they are distinct.
    const scalarField& upper = matrix.upper();
    const scalarField& lower = matrix.lower();

    // ---- Effective source (manually fold boundaryCoeffs; serial-only) ----
    //   Mimics protected addBoundarySource(source, couples=false).
    //   couples=false => only non-coupled patches contribute, matching the
    //   path used inside lduMatrix::solver for the initial source.
    scalarField sourceFull(matrix.source());
    const auto& psiField = matrix.psi();
    const auto& bCoeffs  = matrix.boundaryCoeffs();
    forAll(psiField.boundaryField(), patchi)
    {
        const auto& ptf = psiField.boundaryField()[patchi];
        if (ptf.coupled())
        {
            // Serial-only guard. For parallel runs extend this block.
            continue;
        }
        const scalarField& pbc = bCoeffs[patchi];
        const labelUList&  pa  = matrix.lduAddr().patchAddr(patchi);
        forAll(pa, facei)
        {
            sourceFull[pa[facei]] += pbc[facei];
        }
    }

    // ---- LDU addressing ----
    const lduAddressing& addr  = matrix.lduAddr();
    const labelUList&    uAddr = addr.upperAddr();  // = neighbour of face
    const labelUList&    lAddr = addr.lowerAddr();  // = owner of face

    const label N      = diagFull.size();
    const label nFaces = upper.size();
    const label nnz    = N + 2*nFaces;

    // ---- A.mm (COO 1-based) ----
    auto cooWriter = [&](std::ofstream& os) {
        for (label c = 0; c < N; ++c)
            os << (c+1) << ' ' << (c+1) << ' ' << diagFull[c] << '\n';
        for (label f = 0; f < nFaces; ++f)
        {
            const label own = lAddr[f] + 1;
            const label nei = uAddr[f] + 1;
            os << own << ' ' << nei << ' ' << upper[f] << '\n';
            os << nei << ' ' << own << ' ' << lower[f] << '\n';
        }
    };
    writeMMCoo_(
        dir / "A.mm",
        N, N, nnz,
        cooWriter,
        "A_effective = diag(+boundary_internalCoeffs) + off-diag(upper,lower)"
    );

    writeMMArray_(dir / "b.mm", sourceFull,
        "b_effective = source + boundary_coeffs");

    writeMMArray_(dir / "x0.mm", x0.primitiveField(),
        "initial guess (psi at solve entry)");

    // ---- normFactor (lduMatrixSolver::normFactor; see audit §10.3) ----
    // Mimic the EXACT solver code path:
    //   1. addBoundarySource(source)            — already done, sourceFull above
    //   2. addBoundaryDiag(matrix.diag(), 0)    — temp-swap diag in place
    //   3. wA = lduMatrix::Amul(psi)            — using post-fold diag
    //   4. sumA, *= gAverage(psi), gSum( |wA-tmp| + |source-tmp| ) + small_
    //   5. restore diag
    solveScalar normFactorOF = 0.0;
    {
        const auto bCoeffsScalar = bCoeffs.component(0);
        const auto& interfaces = matrix.psi().boundaryField().scalarInterfaces();

        // Step 2: temp-swap fvMatrix's diag with diagFull (= D = raw + intCoeffs)
        fvScalarMatrix& matrixNC = const_cast<fvScalarMatrix&>(matrix);
        scalarField saveDiagBak(matrixNC.diag());
        matrixNC.diag() = diagFull;

        // Step 3: lduMatrix::Amul (the SAME call PBiCG.C line 97 makes)
        solveScalarField psiSS(x0.primitiveField().size());
        forAll(psiSS, i) psiSS[i] = x0.primitiveField()[i];
        solveScalarField wA(N, Foam::Zero);
        matrix.Amul(wA, psiSS, bCoeffsScalar, interfaces, 0);

        // Step 4a: sumA (also uses post-fold diag now)
        solveScalarField sumAField(N, Foam::Zero);
        matrix.sumA(sumAField, bCoeffsScalar, interfaces);

        const solveScalar psibar = Foam::gAverage(psiSS, matrix.mesh().comm());
        scalarField sumARaw(sumAField);  // for separate dump
        sumAField *= psibar;

        solveScalarField srcSS(sourceFull.size());
        forAll(srcSS, i) srcSS[i] = sourceFull[i];

        normFactorOF = Foam::gSum(
            (Foam::mag(wA - sumAField) + Foam::mag(srcSS - sumAField))(),
            matrix.mesh().comm()
        ) + 1e-20;

        // Step 5: restore raw diag
        matrixNC.diag() = saveDiagBak;

        writeMMArray_(dir / "sumA.mm", sumARaw,
            "sumA[c] = OpenFOAM lduMatrix::sumA() with post-fold diag (matches PBiCG view)");
    }

    {
        std::ofstream nf((dir / "normFactor.txt").c_str());
        nf << std::scientific << std::setprecision(17) << normFactorOF << "\n";
    }

    writeMeta_(dir, matrix, eqName, correctorIdx, nullptr, N, nnz);

    Info<< "[matrixDumper] pre-solve " << eqName << "_corr" << correctorIdx
        << "  N=" << N << "  nnz=" << nnz << endl;
}

inline void matrixDumper::dumpPostSolve
(
    const word& eqName,
    const volScalarField& xFinal,
    const SolverPerformance<scalar>& perf,
    const fvScalarMatrix& matrix
)
{
    if (!shouldDumpEq(eqName)) return;

    const label correctorIdx = peekCorrIdx_(eqName);
    // Skip if pre-solve was gated by maxCorr cap
    if (dumpedCorrs_.find({eqName, correctorIdx}) == dumpedCorrs_.end())
        return;

    const fileName dir = matDir_(eqName, correctorIdx);

    writeMMArray_(dir / "x_final.mm", xFinal.primitiveField(),
        "x after OpenFOAM solve");

    const label N   = matrix.diag().size();
    const label nnz = N + 2*matrix.upper().size();
    writeMeta_(dir, matrix, eqName, correctorIdx, &perf, N, nnz);

    Info<< "[matrixDumper] post-solve " << eqName << "_corr" << correctorIdx
        << "  finalRes=" << perf.finalResidual()
        << "  iters="    << perf.nIterations() << endl;
}

} // namespace Foam

#endif
