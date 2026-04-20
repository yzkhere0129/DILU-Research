// FFI handler: cusparse_dilu_analyze.
//
// Contract (arch doc §4.2):
//   Inputs:  row_ptr int32[n+1], col_idx int32[nnz], values f64[nnz]
//   Outputs: token uint64[1]   (opaque — indexes the plan cache)
//
// Work:
//   1. Get/create singleton cuSPARSE handle for this device.
//   2. Allocate a working_values device buffer (nnz doubles). Seed it with A's
//      values — cuSPARSE analysis phase reads values even though only the
//      pattern is load-bearing, and an uninitialized diagonal can make the
//      analysis reject the matrix as structurally singular on some versions.
//   3. Create cusparseSpMatDescr_t mat_L (fill=LOWER, diag=NON_UNIT) and
//      mat_U (fill=UPPER, diag=NON_UNIT), both pointing at working_values.
//   4. Create SpSVDescr for both L and U. Run bufferSize + analysis on each.
//   5. Create 4 DnVec descriptors (r, y_mid, y_scaled, z) bound to dummy
//      owned buffers; apply will rebind via cusparseDnVecSetValues.
//   6. Insert PlanEntry into cache, return token.
//
// Sharp edge (arch doc §4.2): cuSPARSE analysis requires the SpMat to
// already carry fill_mode and diag_type attributes BEFORE analysis runs. We
// set them via cusparseSpMatSetAttribute right after CreateCsr.

#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdint.h>

#include <cstddef>
#include <cstring>
#include <string>

#include "plan_registry.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using dilu::cusparse::PlanEntry;
using dilu::cusparse::PatternFingerprint;
using dilu::cusparse::get_handle;
using dilu::cusparse::plan_cache_insert;
using dilu::cusparse::destroy_plan_entry;
using dilu::cusparse::cusparse_status_str;

#define CHECK_CUDA(expr)                                                  \
  do {                                                                     \
    cudaError_t _e = (expr);                                               \
    if (_e != cudaSuccess) {                                               \
      destroy_plan_entry(entry);                                           \
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,                       \
                        std::string("cuda: ") + cudaGetErrorString(_e));  \
    }                                                                      \
  } while (0)

#define CHECK_CUSPARSE(expr)                                              \
  do {                                                                     \
    cusparseStatus_t _s = (expr);                                          \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                   \
      destroy_plan_entry(entry);                                           \
      return ffi::Error(XLA_FFI_Error_Code_INTERNAL,                       \
                        std::string("cusparse: ") +                        \
                            cusparse_status_str(_s));                      \
    }                                                                      \
  } while (0)

static ffi::Error CusparseDiluAnalyzeImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::S32> row_ptr,
    ffi::Buffer<ffi::DataType::S32> col_idx,
    ffi::Buffer<ffi::DataType::F64> values,
    ffi::Buffer<ffi::DataType::S32> diag_offset,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> token_out) {
  const size_t n_plus_1 = row_ptr.element_count();
  if (n_plus_1 < 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "analyze: row_ptr empty");
  }
  const int64_t n = static_cast<int64_t>(n_plus_1 - 1);
  const int64_t nnz = static_cast<int64_t>(values.element_count());
  if (col_idx.element_count() != static_cast<size_t>(nnz)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "analyze: col_idx/values length mismatch");
  }
  if (diag_offset.element_count() != static_cast<size_t>(n)) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "analyze: diag_offset must have length n");
  }
  if (n > INT32_MAX || nnz > INT32_MAX) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "analyze: n or nnz exceeds int32");
  }
  if (token_out->element_count() != 1) {
    return ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                      "analyze: token output must be uint64[1]");
  }

  cusparseHandle_t handle = nullptr;
  cusparseStatus_t hs = get_handle(&handle);
  if (hs != CUSPARSE_STATUS_SUCCESS) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("analyze: cusparseCreate: ") +
                          cusparse_status_str(hs));
  }
  cusparseStatus_t sss = cusparseSetStream(handle, stream);
  if (sss != CUSPARSE_STATUS_SUCCESS) {
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                      std::string("analyze: setStream: ") +
                          cusparse_status_str(sss));
  }

  PlanEntry* entry = new PlanEntry();
  entry->fingerprint = PatternFingerprint{
      static_cast<int32_t>(n), static_cast<int32_t>(nnz)};

  // Own CSR pattern copies — XLA may rebind input buffers across jit
  // boundaries, but cusparseCreateCsr captures a pointer and expects it to
  // stay valid. Copying into plan-owned buffers decouples us from JAX buffer
  // lifecycles.
  CHECK_CUDA(cudaMallocAsync((void**)&entry->row_ptr,
                             (n + 1) * sizeof(int32_t), stream));
  CHECK_CUDA(cudaMemcpyAsync(entry->row_ptr, row_ptr.typed_data(),
                             (n + 1) * sizeof(int32_t),
                             cudaMemcpyDeviceToDevice, stream));
  CHECK_CUDA(cudaMallocAsync((void**)&entry->col_idx,
                             nnz * sizeof(int32_t), stream));
  CHECK_CUDA(cudaMemcpyAsync(entry->col_idx, col_idx.typed_data(),
                             nnz * sizeof(int32_t),
                             cudaMemcpyDeviceToDevice, stream));
  CHECK_CUDA(cudaMallocAsync((void**)&entry->diag_offset,
                             n * sizeof(int32_t), stream));
  CHECK_CUDA(cudaMemcpyAsync(entry->diag_offset, diag_offset.typed_data(),
                             n * sizeof(int32_t),
                             cudaMemcpyDeviceToDevice, stream));

  // Working_values: seed with A's values so analysis sees a matrix whose
  // diagonal is nonzero (cuSPARSE rejects structurally-singular inputs).
  CHECK_CUDA(cudaMallocAsync((void**)&entry->working_values,
                             nnz * sizeof(double), stream));
  CHECK_CUDA(cudaMemcpyAsync(entry->working_values,
                             values.typed_data(),
                             nnz * sizeof(double),
                             cudaMemcpyDeviceToDevice, stream));
  CHECK_CUDA(cudaMallocAsync((void**)&entry->y_mid,
                             n * sizeof(double), stream));
  CHECK_CUDA(cudaMallocAsync((void**)&entry->y_scaled,
                             n * sizeof(double), stream));

  // Two SpMat descriptors (LOWER / UPPER), both pointing at plan-owned
  // row_ptr/col_idx/working_values. Both non-unit diag (d_i sits on diagonal).
  CHECK_CUSPARSE(cusparseCreateCsr(
      &entry->mat_L, n, n, nnz,
      entry->row_ptr, entry->col_idx, entry->working_values,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  cusparseFillMode_t fill_L = CUSPARSE_FILL_MODE_LOWER;
  cusparseDiagType_t diag_ty = CUSPARSE_DIAG_TYPE_NON_UNIT;
  CHECK_CUSPARSE(cusparseSpMatSetAttribute(
      entry->mat_L, CUSPARSE_SPMAT_FILL_MODE, &fill_L, sizeof(fill_L)));
  CHECK_CUSPARSE(cusparseSpMatSetAttribute(
      entry->mat_L, CUSPARSE_SPMAT_DIAG_TYPE, &diag_ty, sizeof(diag_ty)));

  CHECK_CUSPARSE(cusparseCreateCsr(
      &entry->mat_U, n, n, nnz,
      entry->row_ptr, entry->col_idx, entry->working_values,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  cusparseFillMode_t fill_U = CUSPARSE_FILL_MODE_UPPER;
  CHECK_CUSPARSE(cusparseSpMatSetAttribute(
      entry->mat_U, CUSPARSE_SPMAT_FILL_MODE, &fill_U, sizeof(fill_U)));
  CHECK_CUSPARSE(cusparseSpMatSetAttribute(
      entry->mat_U, CUSPARSE_SPMAT_DIAG_TYPE, &diag_ty, sizeof(diag_ty)));

  // Create DnVec descriptors for the solver inputs/outputs. Pointers get
  // rebound per apply via cusparseDnVecSetValues. Bind initial dummies.
  CHECK_CUSPARSE(cusparseCreateDnVec(&entry->vec_r,    n, entry->y_mid,    CUDA_R_64F));
  CHECK_CUSPARSE(cusparseCreateDnVec(&entry->vec_y_mid, n, entry->y_mid,   CUDA_R_64F));
  CHECK_CUSPARSE(cusparseCreateDnVec(&entry->vec_y_scaled, n, entry->y_scaled, CUDA_R_64F));
  CHECK_CUSPARSE(cusparseCreateDnVec(&entry->vec_z,    n, entry->y_mid,    CUDA_R_64F));

  // SpSV analysis.
  CHECK_CUSPARSE(cusparseSpSV_createDescr(&entry->spsv_L));
  CHECK_CUSPARSE(cusparseSpSV_createDescr(&entry->spsv_U));

  const double alpha = 1.0;
  CHECK_CUSPARSE(cusparseSpSV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      entry->mat_L, entry->vec_r, entry->vec_y_mid,
      CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, entry->spsv_L,
      &entry->buf_L_sz));
  if (entry->buf_L_sz > 0) {
    CHECK_CUDA(cudaMallocAsync(&entry->buf_L, entry->buf_L_sz, stream));
  }
  CHECK_CUSPARSE(cusparseSpSV_analysis(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      entry->mat_L, entry->vec_r, entry->vec_y_mid,
      CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, entry->spsv_L,
      entry->buf_L));

  CHECK_CUSPARSE(cusparseSpSV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      entry->mat_U, entry->vec_y_scaled, entry->vec_z,
      CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, entry->spsv_U,
      &entry->buf_U_sz));
  if (entry->buf_U_sz > 0) {
    CHECK_CUDA(cudaMallocAsync(&entry->buf_U, entry->buf_U_sz, stream));
  }
  CHECK_CUSPARSE(cusparseSpSV_analysis(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
      entry->mat_U, entry->vec_y_scaled, entry->vec_z,
      CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, entry->spsv_U,
      entry->buf_U));

  // Insert into cache and write token to output buffer.
  uint64_t token = plan_cache_insert(entry);
  // NOTE: we write the uint64 token to a device buffer. Use cudaMemcpyAsync
  // from host. This is entirely owned by the XLA stream.
  CHECK_CUDA(cudaMemcpyAsync(token_out->typed_data(), &token, sizeof(uint64_t),
                             cudaMemcpyHostToDevice, stream));

  return ffi::Error::Success();
}

#undef CHECK_CUDA
#undef CHECK_CUSPARSE

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CusparseDiluAnalyze, CusparseDiluAnalyzeImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // row_ptr
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // col_idx
        .Arg<ffi::Buffer<ffi::DataType::F64>>()   // values
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // diag_offset
        .Ret<ffi::Buffer<ffi::DataType::U64>>()); // token[1]
