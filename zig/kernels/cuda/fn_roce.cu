// The GPU half of the one-shot RoCE all-gather between 2 to 4 DGX Sparks (decode plan D1, work/research/R1-decode.md 3.1):
// b12x's "RoCEnante" protocol (https://github.com/local-inference-lab/b12x, b12x/comm/roce, by the b12x contributors
// at Local Inference Lab, Apache License 2.0), written anew in CUDA C++ for TensorFold's Zig runtime (the host half is
// zig/src/cuda/roce_proxy.c, the driver zig/src/cuda/roce.zig; the region layout is described there).
//
// One launch is one all-gather of `n` units (16 or 4 bytes) a rank into out[world][n], rank 0's first, as
// ncclAllGather lays it out: stage this rank's shard into the pinned send slot, ring the proxy (the last block, after
// a system fence), wait for every peer's flag on every stripe (a ring's relayed ones arrive the same way), copy the
// local shard and the peers' from their receive slots into place, in rank order, and the last block publishes the sequence as the new epoch. The sequence lives on the device, so a CUDA
// graph replays the launch as often as it likes. The bytes are moved, never computed on: exact by construction.
// Every block waits for the peer, so the grid must be resident at once (the driver keeps it at 16 blocks or fewer).
//
// Decode plan D2 (L2 prefetch during the gather): blocks past the first `gblocks` do no gathering; they issue L2
// prefetches over [pf, pf + pf_bytes) (the next kernels' weights, read-only) and exit, so DRAM fills the L2 while the
// gather waits on the network. `prefetch` below is the same for a gather that NCCL makes.
#include <stdint.h>

namespace tf_fn_roce {

constexpr int SLOTS = 2, FLAG_WORDS = 32;  // 128-byte flags, `n_flag` stripes (flags) from each peer

struct State {
  unsigned epoch, stage, tail, pad;
};

__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_sys(unsigned* p, unsigned v) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint4 ld_sys(const uint4* p) {
  uint4 r;
  // no "memory" clobber: the loads of one copy loop may be in flight together (the flag's acquire and the block
  // barrier already order them after the peer's write)
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}
__device__ __forceinline__ unsigned ld_sys(const unsigned* p) {
  unsigned r;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// T: uint4 (16-byte units, both pointers 16-aligned) or unsigned (4-byte units). `bytes` is the shard rounded up to
// 16 (what the proxy writes); region pointers are device addresses of the mapped host region.
__device__ __forceinline__ void prefetch_lines(const char* pf, unsigned long long bytes, unsigned long long first,
                                               unsigned long long step) {
  for (unsigned long long off = first * 128; off < bytes; off += step * 128)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(pf + off));
}

__global__ void __launch_bounds__(512) prefetch(const char* pf, unsigned long long pf_bytes) {
  prefetch_lines(pf, pf_bytes, (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x,
                 (unsigned long long)gridDim.x * blockDim.x);
}

template <typename T>
__global__ void __launch_bounds__(512) gather(const T* in, T* out, int n, unsigned bytes,
                                              T* send, const T* recv, const unsigned* flags, unsigned* ctrl,
                                              unsigned long long slot_bytes, State* st, int rank, int world, int n_flag,
                                              unsigned long long timeout_ns, int gblocks, const char* pf,
                                              unsigned long long pf_bytes) {
  if ((int)blockIdx.x >= gblocks) {
    prefetch_lines(pf, pf_bytes, (unsigned long long)(blockIdx.x - gblocks) * blockDim.x + threadIdx.x,
                   (unsigned long long)(gridDim.x - gblocks) * blockDim.x);
    return;
  }
  const unsigned seq = *reinterpret_cast<volatile unsigned*>(&st->epoch) + 1u;
  const unsigned slot = seq % SLOTS;
  const size_t slot_units = slot_bytes / sizeof(T);
  const int index = blockIdx.x * blockDim.x + threadIdx.x, stride = gblocks * blockDim.x;
  // 1. this rank's shard into its send slot
  T* mine = send + slot * slot_units;
  for (int i = index; i < n; i += stride) mine[i] = in[i];
  __syncthreads();
  // 2. the last block to finish staging rings the doorbell (a system fence per block orders its stores first)
  if (threadIdx.x == 0) {
    __threadfence_system();
    if (atomicAdd(&st->stage, 1u) == (unsigned)gblocks - 1) {
      atomicExch(&st->stage, 0u);
      st_relaxed_sys(ctrl + 4 + slot, bytes);
      __threadfence_system();
      st_relaxed_sys(ctrl, seq);
    }
  }
  // 3. every peer's shard: its flag on every stripe (each written after that stripe's payload on the same queue pair)
  if ((int)threadIdx.x < world * n_flag) {
    const int peer = (int)threadIdx.x / n_flag, stripe = (int)threadIdx.x % n_flag;
    if (peer != rank) {
      const unsigned* f = flags + (((size_t)peer * SLOTS + slot) * n_flag + stripe) * FLAG_WORDS;
      const unsigned long long t0 = globaltimer();
      while (ld_acquire_sys(f) != seq) {
        if (globaltimer() - t0 > timeout_ns) {
          // the peer is gone or its proxy failed: say which, then trap rather than read a stale slot (a fatal error
          // for the context: the process restarts; roce.zig timedOut reads the sequence and the peer)
          st_relaxed_sys(ctrl + 3, (unsigned)peer);
          st_relaxed_sys(ctrl + 6, (unsigned)stripe);
          st_relaxed_sys(ctrl + 2, seq);
          __threadfence_system();
          __trap();
        }
      }
    }
  }
  __syncthreads();
  // 4. every shard into place, rank order
  for (int source = 0; source < world; ++source) {
    T* out_src = out + (size_t)source * n;
    if (source == rank) {
      for (int i = index; i < n; i += stride) out_src[i] = in[i];
      continue;
    }
    const T* theirs = recv + (size_t)(source * SLOTS + slot) * slot_units;
    // four loads from the receive slot in flight a thread, then their stores
    for (int i0 = index; i0 < n; i0 += 4 * stride) {
      T v[4];
#pragma unroll
      for (int u = 0; u < 4; ++u)
        if (i0 + u * stride < n) v[u] = ld_sys(theirs + i0 + u * stride);
#pragma unroll
      for (int u = 0; u < 4; ++u)
        if (i0 + u * stride < n) out_src[i0 + u * stride] = v[u];
    }
  }
  // 5. the last block to finish publishes the sequence (every block read the old epoch before it got here)
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(&st->tail, 1u) == (unsigned)gblocks - 1) {
    atomicExch(&st->tail, 0u);
    __threadfence();
    *reinterpret_cast<volatile unsigned*>(&st->epoch) = seq;
  }
}

}  // namespace tf_fn_roce

template __global__ void tf_fn_roce::gather<uint4>(const uint4*, uint4*, int, unsigned, uint4*, const uint4*,
                                                   const unsigned*, unsigned*, unsigned long long,
                                                   tf_fn_roce::State*, int, int, int, unsigned long long, int,
                                                   const char*, unsigned long long);
template __global__ void tf_fn_roce::gather<unsigned>(const unsigned*, unsigned*, int, unsigned, unsigned*,
                                                      const unsigned*, const unsigned*, unsigned*,
                                                      unsigned long long, tf_fn_roce::State*, int, int, int,
                                                      unsigned long long, int, const char*, unsigned long long);
