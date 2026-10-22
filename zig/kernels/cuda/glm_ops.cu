// TensorFold glm5_next/cuda/{multi_tune,dflash2,dflash2_multi}.py torch replacements.
// Ash Hart and TensorFold contributors. Plain driver-launch ABI, like fn_ops.cu.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

// Sort selected values descending; equal values retain compaction order. The host
// lexsorts by token id. ATen sorted tie permutations still need GPU qualification.
extern "C" __global__ void glm_sort(float* vals, int64_t* ids, uint32_t k) {
    uint64_t base = uint64_t(blockIdx.x)*k;
    for (uint32_t phase=0; phase<k; ++phase) {
        for (uint32_t i=2*threadIdx.x+(phase&1); i+1<k; i+=2*blockDim.x) {
            float a=vals[base+i], b=vals[base+i+1];
            if ((!isnan(a) && isnan(b)) || a<b) {
                vals[base+i]=b; vals[base+i+1]=a;
                int64_t id=ids[base+i]; ids[base+i]=ids[base+i+1]; ids[base+i+1]=id;
            }
        }
        __syncthreads();
    }
}
extern "C" __global__ void glm_pack(const float* v, const int64_t* ids, float* out,
                                    uint32_t k, int64_t offset) {
    uint64_t r=blockIdx.x;
    for(uint32_t j=threadIdx.x;j<k;j+=blockDim.x) {
        out[r*2*k+j]=v[r*k+j];
        out[r*2*k+k+j]=__int_as_float(int32_t(ids[r*k+j]+offset));
    }
}
// index_copy_(dim): unique indices, contiguous [outer, destination, inner] elements.
// Source is [outer, rows, inner]; byte addressing supports bf16/int32/int64 equally.
extern "C" __global__ void glm_index_copy(const char* src, char* dst, const int64_t* idx,
    uint64_t rows, uint64_t dest_rows, uint64_t inner_bytes, uint64_t outer, uint32_t* invalid) {
    uint64_t r=blockIdx.x; int64_t to=idx[r];
    if(to<0 || uint64_t(to)>=dest_rows) { if(threadIdx.x==0) atomicExch(invalid,1u); return; }
    for(uint64_t o=blockIdx.y;o<outer;o+=gridDim.y)
        for(uint64_t b=threadIdx.x;b<inner_bytes;b+=blockDim.x)
            dst[(o*dest_rows+to)*inner_bytes+b]=src[(o*rows+r)*inner_bytes+b];
}
extern "C" __global__ void glm_rotary(const int64_t* pos, const float* inv, float* c, float* s,
                                      uint64_t rows, uint64_t half) {
    for(uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;j<rows*half;j+=uint64_t(gridDim.x)*blockDim.x) {
        float angle=float(pos[j/half])*inv[j%half];
        c[j]=cosf(angle); s[j]=sinf(angle);
    }
}
extern "C" __global__ void glm_ring(const int64_t* pos, const int64_t* slots, int64_t* out,
                                    uint64_t rows, int64_t cap) {
    for(uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;j<rows;j+=uint64_t(gridDim.x)*blockDim.x) {
        int64_t rem=pos[j]%cap; if(rem<0) rem+=cap;
        out[j]=slots[j]*cap+rem;
    }
}
// pos_dev + offset, int64 position additions; modular two's-complement arithmetic.
extern "C" __global__ void glm_int_add(const void* src, void* dst, uint64_t count, int64_t value, uint32_t bytes) {
    for(uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;j<count;j+=uint64_t(gridDim.x)*blockDim.x) {
        if(bytes==8) reinterpret_cast<uint64_t*>(dst)[j]=reinterpret_cast<const uint64_t*>(src)[j]+uint64_t(value);
        else reinterpret_cast<uint32_t*>(dst)[j]=reinterpret_cast<const uint32_t*>(src)[j]+uint32_t(value);
    }
}
extern "C" __global__ void glm_rank_add(const float* ranks, __nv_bfloat16* out, uint64_t count, uint32_t world) {
    for(uint64_t j=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;j<count;j+=uint64_t(gridDim.x)*blockDim.x) {
        float acc=ranks[j]; for(uint32_t r=1;r<world;++r) acc=acc+ranks[uint64_t(r)*count+j];
        out[j]=__float2bfloat16_rn(acc);
    }
}
extern "C" __global__ void glm_scaled_max(const __nv_bfloat16* logits, double* out,
                                         uint64_t columns, double temp) {
    // Deterministic reduction: each row independent, no accumulation rounding.
    __shared__ double best[256]; double v=-INFINITY;
    for(uint64_t j=threadIdx.x;j<columns;j+=256) v=fmax(v,double(__bfloat162float(logits[uint64_t(blockIdx.x)*columns+j]))/temp);
    best[threadIdx.x]=v; __syncthreads();
    for(int d=128;d;d/=2) { if(threadIdx.x<d) best[threadIdx.x]=fmax(best[threadIdx.x],best[threadIdx.x+d]); __syncthreads(); }
    if(threadIdx.x==0) out[blockIdx.x]=best[0];
}
extern "C" __global__ void glm_mass_pack(const float* vals,const int64_t* cols,const int64_t* mass,
    const int64_t* sums, int64_t* out, uint64_t width, uint32_t selected, uint32_t count, int64_t offset, double temp) {
    uint64_t r=blockIdx.x, at=r*(uint64_t(3)*count+2);
    for(uint32_t j=threadIdx.x;j<count;j+=blockDim.x) {
        bool real=j<selected; int64_t col=real?cols[r*selected+j]:0;
        out[at+j]=__double_as_longlong(real?double(vals[r*selected+j])/temp:-INFINITY);
        out[at+count+j]=real?col+offset:-1;
        out[at+2*count+j]=real?mass[r*width+col]:0;
    }
    if(threadIdx.x==0) { out[at+3*count]=sums[r]; out[at+3*count+1]=width; }
}

// Greedy sampling's top-1 as one pass: per row the largest bf16 logit and its column, packed [value | int32 id] for the
// gather. It replaces to_f32 + radix top-k(1) + sort + pack and returns what they returned:
//  - order = tf_topk_f32 selection_key: a NaN (any payload) ranks above everything, +0 above -0, -inf lowest;
//  - ties (equal key) keep the lowest column (the radix compaction fills the first tied slot in column order);
//  - the value written is the input at that column, widened exactly (a NaN keeps its payload bits).
// Rows are `columns` apart. id = column + offset as int32 bits, as glm_pack writes them.
__device__ __forceinline__ uint32_t glm_greedy_key(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return UINT32_MAX;
    return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}
// Stage 1: grid (GREEDY_SPLIT, rows). Block b of row r scans its slice of the columns and leaves its (key, column) in
// part[r * GREEDY_SPLIT + b] as two words; a block holding no element leaves key 0 / column 0xffffffff.
#define GREEDY_SPLIT 16
extern "C" __global__ void glm_greedy_part(const __nv_bfloat16* logits, uint32_t* part, uint64_t columns) {
    __shared__ uint32_t wkey[32];
    __shared__ uint32_t wcol[32];
    const __nv_bfloat16* row = logits + uint64_t(blockIdx.y) * columns;
    uint32_t best_key = 0, best_col = 0xffffffffu;
    bool have = false;
    const bool vec = columns % 8 == 0 && (reinterpret_cast<uintptr_t>(row) & 15) == 0;
    if (vec) {
        const uint64_t chunks = columns / 8, per = (chunks + GREEDY_SPLIT - 1) / GREEDY_SPLIT;
        const uint64_t lo = blockIdx.x * per, hi = lo + per < chunks ? lo + per : chunks;
        const uint4* v = reinterpret_cast<const uint4*>(row);
        for (uint64_t c = lo + threadIdx.x; c < hi; c += blockDim.x) {
            const uint4 w = v[c];
            const uint32_t words[4] = {w.x, w.y, w.z, w.w};
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t key = glm_greedy_key(__uint_as_float((i & 1 ? words[i >> 1] & 0xffff0000u : words[i >> 1] << 16)));
                if (!have || key > best_key) { have = true; best_key = key; best_col = uint32_t(c * 8 + i); }
            }
        }
    } else {
        const uint64_t per = (columns + GREEDY_SPLIT - 1) / GREEDY_SPLIT;
        const uint64_t lo = blockIdx.x * per, hi = lo + per < columns ? lo + per : columns;
        for (uint64_t c = lo + threadIdx.x; c < hi; c += blockDim.x) {
            const uint32_t key = glm_greedy_key(__bfloat162float(row[c]));
            if (!have || key > best_key) { have = true; best_key = key; best_col = uint32_t(c); }
        }
    }
    // (key, lowest column) over the block. A thread without an element holds key 0 and column 0xffffffff; no real
    // element has key 0 (a NaN maps to UINT32_MAX, every other float to a nonzero key), so it never wins.
    for (int d = 16; d; d >>= 1) {
        const uint32_t k = __shfl_down_sync(0xffffffffu, best_key, d), c = __shfl_down_sync(0xffffffffu, best_col, d);
        if (k > best_key || (k == best_key && c < best_col)) { best_key = k; best_col = c; }
    }
    if ((threadIdx.x & 31) == 0) { wkey[threadIdx.x >> 5] = best_key; wcol[threadIdx.x >> 5] = best_col; }
    __syncthreads();
    if (threadIdx.x < 32) {
        const uint32_t warps = blockDim.x >> 5;
        best_key = threadIdx.x < warps ? wkey[threadIdx.x] : 0;
        best_col = threadIdx.x < warps ? wcol[threadIdx.x] : 0xffffffffu;
        for (int d = 16; d; d >>= 1) {
            const uint32_t k = __shfl_down_sync(0xffffffffu, best_key, d), c = __shfl_down_sync(0xffffffffu, best_col, d);
            if (k > best_key || (k == best_key && c < best_col)) { best_key = k; best_col = c; }
        }
        if (threadIdx.x == 0) {
            part[(uint64_t(blockIdx.y) * GREEDY_SPLIT + blockIdx.x) * 2] = best_key;
            part[(uint64_t(blockIdx.y) * GREEDY_SPLIT + blockIdx.x) * 2 + 1] = best_col;
        }
    }
}
// Stage 2: grid (rows), one warp: the best of a row's GREEDY_SPLIT partials, packed [value | int32 id] into out.
extern "C" __global__ void glm_greedy_pick(const __nv_bfloat16* logits, const uint32_t* part, float* out, uint64_t columns, int64_t offset) {
    uint32_t best_key = 0, best_col = 0xffffffffu;
    if (threadIdx.x < GREEDY_SPLIT) {
        best_key = part[(uint64_t(blockIdx.x) * GREEDY_SPLIT + threadIdx.x) * 2];
        best_col = part[(uint64_t(blockIdx.x) * GREEDY_SPLIT + threadIdx.x) * 2 + 1];
    }
    for (int d = 16; d; d >>= 1) {
        const uint32_t k = __shfl_down_sync(0xffffffffu, best_key, d), c = __shfl_down_sync(0xffffffffu, best_col, d);
        if (k > best_key || (k == best_key && c < best_col)) { best_key = k; best_col = c; }
    }
    if (threadIdx.x == 0) {
        out[uint64_t(blockIdx.x) * 2] = __bfloat162float(logits[uint64_t(blockIdx.x) * columns + best_col]);
        out[uint64_t(blockIdx.x) * 2 + 1] = __int_as_float(int32_t(int64_t(best_col) + offset));
    }
}

// L2 prefetch of weight ranges in decode windows (glm5_next/cuda/l2pf.cu; Zig side: cuda_l2pf.zig). Read-only: no result
// can change. TABLE [n, 2] int64 (address, bytes) pieces, 16-byte aligned, a few hundred KiB at most.
// Adapted (by l2pf.cu) from jayleaton/glm53-tensorfold-spark patches/0460 (Apache-2.0, Copyright 2026 Jay Leaton).
// MODE 0 bulk: one cp.async.bulk.prefetch.L2 a piece; 1 lines: prefetch.global.L2::evict_last per 128-byte line, a warp
// a piece; 2 touch: ld.global.cg of every line's first 16 bytes.
extern "C" __global__ void glm_l2pf(const long long* __restrict__ table, int n, int mode, unsigned* __restrict__ sink) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int warps = (gridDim.x * blockDim.x) >> 5;
    if (mode == 0) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
            const unsigned long long a = (unsigned long long)table[2 * i];
            const unsigned bytes = (unsigned)table[2 * i + 1];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(a), "r"(bytes) : "memory");
#else
            for (unsigned o = 0; o < bytes; o += 128)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(a + o));
#endif
        }
        return;
    }
    unsigned acc = 0;
    for (int i = warp; i < n; i += warps) {
        const unsigned long long a = (unsigned long long)table[2 * i];
        const unsigned bytes = (unsigned)table[2 * i + 1];
        for (unsigned o = lane * 128u; o < bytes; o += 32u * 128u) {
            if (mode == 1) {
                asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"(a + o));
            } else {
                unsigned x, y, z, w;
                asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                             : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(a + o));
                acc ^= x ^ y ^ z ^ w;
            }
        }
    }
    if (mode == 2 && acc == 0x9e3779b9u && sink != nullptr) sink[0] = acc;
}

// Thin host launchers for tools/zig/check_glm_p7_ops.py. Zig launches the same
// entry points directly through the driver and does not link these launchers.
extern "C" cudaError_t glm_sort_launch(float* vals, int64_t* ids, uint64_t rows, uint32_t k, cudaStream_t stream) {
    glm_sort<<<unsigned(rows),256,0,stream>>>(vals,ids,k); return cudaGetLastError();
}
extern "C" cudaError_t glm_rotary_launch(const int64_t* pos,const float* inv,float* c,float* s,uint64_t rows,uint64_t half,cudaStream_t stream) {
    glm_rotary<<<unsigned((rows*half+255)/256),256,0,stream>>>(pos,inv,c,s,rows,half); return cudaGetLastError();
}
extern "C" cudaError_t glm_index_copy_launch(const char* src,char* dst,const int64_t* idx,uint64_t rows,uint64_t dest_rows,uint64_t inner,uint64_t outer,uint32_t* invalid,cudaStream_t stream) {
    glm_index_copy<<<dim3(unsigned(rows),unsigned(outer)),256,0,stream>>>(src,dst,idx,rows,dest_rows,inner,outer,invalid); return cudaGetLastError();
}
extern "C" cudaError_t glm_ring_launch(const int64_t* pos,const int64_t* slots,int64_t* out,uint64_t rows,int64_t cap,cudaStream_t stream) {
    glm_ring<<<unsigned((rows+255)/256),256,0,stream>>>(pos,slots,out,rows,cap); return cudaGetLastError();
}
extern "C" cudaError_t glm_rank_add_launch(const float* ranks,__nv_bfloat16* out,uint64_t count,uint32_t world,cudaStream_t stream) {
    glm_rank_add<<<unsigned((count+255)/256),256,0,stream>>>(ranks,out,count,world); return cudaGetLastError();
}
extern "C" cudaError_t glm_mass_pack_launch(const float* vals,const int64_t* cols,const int64_t* mass,const int64_t* sums,int64_t* out,uint64_t rows,uint64_t width,uint32_t selected,uint32_t count,int64_t offset,double temp,cudaStream_t stream) {
    glm_mass_pack<<<unsigned(rows),256,0,stream>>>(vals,cols,mass,sums,out,width,selected,count,offset,temp); return cudaGetLastError();
}

extern "C" cudaError_t glm_greedy_pick_launch(const __nv_bfloat16* logits, uint32_t* part, float* out, uint64_t rows, uint64_t columns, int64_t offset, cudaStream_t stream) {
    glm_greedy_part<<<dim3(GREEDY_SPLIT,unsigned(rows)),256,0,stream>>>(logits,part,columns);
    cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) return e;
    glm_greedy_pick<<<unsigned(rows),32,0,stream>>>(logits,part,out,columns,offset); return cudaGetLastError();
}

// A prompt chunk's picture rows (glm5_next vision: x.index_copy_(0, rows, features.repeat(1, streams))): feature row i
// replaces prompt row rows[i] (absolute) on every hyper-connection stream; pos is the chunk's first row.
extern "C" __global__ void glm_media_rows(const uint32_t* rows, const uint16_t* feats, uint16_t* x,
                                          uint32_t pos, uint32_t width, uint32_t streams) {
    uint64_t r = blockIdx.x;
    uint64_t row = uint64_t(rows[r]) - pos;
    for (uint32_t j = threadIdx.x; j < width * streams; j += blockDim.x)
        x[row * width * streams + j] = feats[r * width + (j % width)];
}
