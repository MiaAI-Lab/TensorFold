// GLM CUDA foundations ABI probe; layout constants from TensorFold glm5_next/cuda/kv8.py and segments.py.
// Ash Hart and TensorFold contributors. This probe performs no model arithmetic.
extern "C" __global__ void glm_foundations_abi(unsigned int* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 528; out[1] = 144; out[2] = 2048; out[3] = 2051;
    }
}
