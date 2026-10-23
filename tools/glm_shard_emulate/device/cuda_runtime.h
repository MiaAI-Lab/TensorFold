// A syntax-only stand-in for the CUDA runtime header: clang's CUDA front end (no SDK; no NVPTX code generation needed)
// checks glm_shard.cu's device code as nvcc would see it:
//   clang++ -x cuda -nocudainc -nocudalib --cuda-device-only --cuda-gpu-arch=sm_90 -fms-extensions \
//     -I tools/glm_shard_emulate/device -fsyntax-only zig/kernels/cuda/glm_shard.cu
#pragma once
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
struct uint4 { unsigned x, y, z, w; };
struct __cuda_builtin_threadIdx_t { __declspec(property(get = __fetch_builtin_x)) unsigned x; static __device__ unsigned __fetch_builtin_x() { return __nvvm_read_ptx_sreg_tid_x(); } };
struct __cuda_builtin_blockIdx_t { __declspec(property(get = __fetch_builtin_x)) unsigned x; static __device__ unsigned __fetch_builtin_x() { return __nvvm_read_ptx_sreg_ctaid_x(); } };
struct __cuda_builtin_blockDim_t { __declspec(property(get = __fetch_builtin_x)) unsigned x; static __device__ unsigned __fetch_builtin_x() { return __nvvm_read_ptx_sreg_ntid_x(); } };
extern const __device__ __cuda_builtin_threadIdx_t threadIdx; extern const __device__ __cuda_builtin_blockIdx_t blockIdx; extern const __device__ __cuda_builtin_blockDim_t blockDim;
