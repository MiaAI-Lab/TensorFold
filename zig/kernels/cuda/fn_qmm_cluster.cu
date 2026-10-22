// Decode plan D5 (work/research/R1-decode.md 3.5): the MTP head's bf16 matrices in 4-bit for the drafts. Their shapes
// need K slices (qmm.split_k: a 2560 x 2560 matrix takes 4, the hyper-connections' down projection 8), which
// TensorFold's qmm_kernel (fn_qmm.cu, a copy of cuda/kernels/qmm.cu by Ash Hart) sums inside a thread-block cluster
// on sm_90 and later (qmm_cuda's dispatch<32, F32, true>); fn_qmm.cu holds only the one-slice forms, so the cluster
// forms are instantiated here from the same source. Only the drafts read them: the target's output never changes.
#include "fn_qmm.cu"

#define TF_QMMC(BM, F32) template __global__ void tf_fn_qmm::qmm_kernel<32, BM, 64, 1, 4, 4, F32, true, false>( \
    const __nv_bfloat16*, const float*, const uint32_t*, const __nv_bfloat16*, const __nv_bfloat16*, void*, float*, \
    int, int, int, int, int, int, int);
TF_QMMC(16, false)
TF_QMMC(32, false)
TF_QMMC(64, false)
TF_QMMC(16, true)
TF_QMMC(32, true)
TF_QMMC(64, true)
