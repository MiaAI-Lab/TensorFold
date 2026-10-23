// Stand-ins for the CUDA names glm_shard.cu uses, so its kernels compile as host C++ (emulate.cpp).
#pragma once
#define __global__
#define __launch_bounds__(x)
struct uint4 { unsigned x, y, z, w; };
struct dim3 { unsigned x, y, z; };
inline dim3 blockIdx, threadIdx, blockDim, gridDim;
