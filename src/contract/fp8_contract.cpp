// The triangle multiplication's e4m3 contraction out[c] = a[c] b[c]^T, both along their last axis, through cuBLASLt's
// FP8 GEMM (sm_89 on); the cuBLASLt jax already loaded is opened at run time, so the library still links libc only.
#if !defined(__HIP_PLATFORM_AMD__)
#include <cublasLt.h>
#include <dlfcn.h>

#include <map>
#include <mutex>
#include <tuple>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

struct Lt {
    decltype(&cublasLtCreate) create = nullptr;
    decltype(&cublasLtMatmulDescCreate) desc_create = nullptr;
    decltype(&cublasLtMatmulDescSetAttribute) desc_set = nullptr;
    decltype(&cublasLtMatrixLayoutCreate) layout_create = nullptr;
    decltype(&cublasLtMatrixLayoutSetAttribute) layout_set = nullptr;
    decltype(&cublasLtMatmulPreferenceCreate) pref_create = nullptr;
    decltype(&cublasLtMatmulPreferenceSetAttribute) pref_set = nullptr;
    decltype(&cublasLtMatmulPreferenceDestroy) pref_destroy = nullptr;
    decltype(&cublasLtMatmulAlgoGetHeuristic) heuristic = nullptr;
    decltype(&cublasLtMatmul) matmul = nullptr;
    cublasLtHandle_t handle = nullptr;
};

// the cuBLASLt already in the process (XLA's), else the first one the loader finds; null when there is none
const Lt* lt() {
    static const Lt* loaded = [] () -> const Lt* {
        static Lt l;
        void* so = nullptr;
        for (const char* name : {"libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"}) {
            so = dlopen(name, RTLD_NOW | RTLD_NOLOAD);
            if (so == nullptr) so = dlopen(name, RTLD_NOW | RTLD_LOCAL);
            if (so != nullptr) break;
        }
        if (so == nullptr) return nullptr;
#define CCK_SYM(field, name) \
    if ((l.field = reinterpret_cast<decltype(l.field)>(dlsym(so, #name))) == nullptr) return nullptr;
        CCK_SYM(create, cublasLtCreate)
        CCK_SYM(desc_create, cublasLtMatmulDescCreate)
        CCK_SYM(desc_set, cublasLtMatmulDescSetAttribute)
        CCK_SYM(layout_create, cublasLtMatrixLayoutCreate)
        CCK_SYM(layout_set, cublasLtMatrixLayoutSetAttribute)
        CCK_SYM(pref_create, cublasLtMatmulPreferenceCreate)
        CCK_SYM(pref_set, cublasLtMatmulPreferenceSetAttribute)
        CCK_SYM(pref_destroy, cublasLtMatmulPreferenceDestroy)
        CCK_SYM(heuristic, cublasLtMatmulAlgoGetHeuristic)
        CCK_SYM(matmul, cublasLtMatmul)
#undef CCK_SYM
        if (l.create(&l.handle) != CUBLAS_STATUS_SUCCESS) return nullptr;
        return &l;
    }();
    return loaded;
}

struct Plan {
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulAlgo_t algo;
    size_t workspace;
};

// one plan a shape and workspace size, kept for the process: shapes are fixed per compiled fold
const Plan* plan(const Lt& l, int c, int i, int j, int k, size_t workspace) {
    static std::mutex mu;
    static std::map<std::tuple<int, int, int, int, size_t>, Plan> plans;
    std::lock_guard<std::mutex> lock(mu);
    const auto key = std::make_tuple(c, i, j, k, workspace);
    if (auto it = plans.find(key); it != plans.end()) return &it->second;
    Plan p{};
    const cublasOperation_t t = CUBLAS_OP_T, n = CUBLAS_OP_N;
    // column-major, out^T [J, I] = b [J, K] . a^T [K, I]: both operands K-contiguous, the TN form FP8 takes
    if (l.desc_create(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS ||
        l.desc_set(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t)) != CUBLAS_STATUS_SUCCESS ||
        l.desc_set(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &n, sizeof(n)) != CUBLAS_STATUS_SUCCESS) return nullptr;
    auto layout = [&](cublasLtMatrixLayout_t* out, cudaDataType type, int rows, int cols, long long stride) {
        return l.layout_create(out, type, rows, cols, rows) == CUBLAS_STATUS_SUCCESS &&
               l.layout_set(*out, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &c, sizeof(c)) == CUBLAS_STATUS_SUCCESS &&
               l.layout_set(*out, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)) ==
                   CUBLAS_STATUS_SUCCESS;
    };
    if (!layout(&p.la, CUDA_R_8F_E4M3, k, j, (long long)j * k) ||
        !layout(&p.lb, CUDA_R_8F_E4M3, k, i, (long long)i * k) ||
        !layout(&p.lc, CUDA_R_16BF, j, i, (long long)i * j)) return nullptr;
    cublasLtMatmulPreference_t pref;
    if (l.pref_create(&pref) != CUBLAS_STATUS_SUCCESS) return nullptr;
    l.pref_set(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace));
    cublasLtMatmulHeuristicResult_t res;
    int found = 0;
    const bool ok = l.heuristic(l.handle, p.op, p.la, p.lb, p.lc, p.lc, pref, 1, &res, &found) == CUBLAS_STATUS_SUCCESS;
    l.pref_destroy(pref);
    if (!ok || found == 0) return nullptr;
    p.algo = res.algo;
    p.workspace = res.workspaceSize;
    return &plans.emplace(key, p).first->second;
}

ffi::Error fp8_contract(cudaStream_t stream, ffi::AnyBuffer a, ffi::AnyBuffer b, ffi::Result<ffi::AnyBuffer> out,
                        ffi::Result<ffi::AnyBuffer> ws) {
    const auto da = a.dimensions(), db = b.dimensions(), dd = out->dimensions();
    if (a.element_type() != ffi::DataType::F8E4M3FN || b.element_type() != ffi::DataType::F8E4M3FN ||
        out->element_type() != ffi::DataType::BF16 || da.size() != 3 || db.size() != 3 || dd.size() != 3) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                          "CckFp8Contract: e4m3 a [C, I, K], b [C, J, K] to bf16 [C, I, J]");
    }
    const int c = int(da[0]), i = int(da[1]), j = int(db[1]), k = int(da[2]);
    if (db[0] != c || db[2] != k || dd[0] != c || dd[1] != i || dd[2] != j) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "CckFp8Contract: mismatched shapes");
    }
    // cuBLASLt's FP8 GEMM wants 16-byte leading dimensions: K a multiple of 16, J of 8
    if (k % 16 != 0 || j % 8 != 0) {
        return ffi::Error(ffi::ErrorCode::kInvalidArgument, "CckFp8Contract: K must be a multiple of 16 and J of 8");
    }
    const Lt* l = lt();
    if (l == nullptr) return ffi::Error(ffi::ErrorCode::kUnavailable, "CckFp8Contract: no cuBLASLt in this process");
    const Plan* p = plan(*l, c, i, j, k, ws->size_bytes());
    if (p == nullptr) {
        return ffi::Error(ffi::ErrorCode::kUnimplemented, "CckFp8Contract: cuBLASLt has no FP8 GEMM here");
    }
    const float one = 1.0f, zero = 0.0f;
    const cublasStatus_t s = l->matmul(l->handle, p->op, &one, b.untyped_data(), p->la, a.untyped_data(), p->lb, &zero,
                                       out->untyped_data(), p->lc, out->untyped_data(), p->lc, &p->algo,
                                       ws->untyped_data(), p->workspace, stream);
    if (s != CUBLAS_STATUS_SUCCESS) {
        return ffi::Error(ffi::ErrorCode::kInternal, "CckFp8Contract: cublasLtMatmul failed");
    }
    return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(CckFp8Contract, fp8_contract,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Arg<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Ret<ffi::AnyBuffer>());
#endif
