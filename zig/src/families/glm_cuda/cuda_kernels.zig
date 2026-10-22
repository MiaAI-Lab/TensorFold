//! Image plumbing for glm5_next/cuda/{kda,kda_chunk,hc,exl3}.cu and cuda/kernels/qmm*.cu.
//! Actual math/launch dispatch belongs to P2/P3/P4/P6; experts planning reuses TensorFold's experts image.
const std = @import("std");
const cuda = @import("cuda");
pub const Image = enum { foundations, kda, kda_chunk, hc, exl3, qmm, qmm_prefill, experts, vision };
pub fn bytes(image: Image) []const u8 {
    return switch (image) {
        .foundations => cuda.kernels.glm_foundations,
        .kda => cuda.kernels.glm_kda,
        .kda_chunk => cuda.kernels.glm_kda_chunk,
        .hc => cuda.kernels.glm_hc,
        .vision => cuda.kernels.glm_vision,
        .exl3 => cuda.kernels.glm_exl3,
        .qmm => cuda.kernels.glm_qmm,
        .qmm_prefill => cuda.kernels.glm_qmm_prefill,
        .experts => cuda.kernels.experts,
    };
}
pub fn load(d: *const cuda.Driver, image: Image) !cuda.Module {
    const blob = bytes(image);
    if (blob.len == 0) return error.MissingGlmKernelImage;
    return cuda.Module.load(d, blob);
}
test "host-only builds have no GLM kernel images" {
    if (!cuda.kernels.available) try std.testing.expectEqual(@as(usize, 0), bytes(.foundations).len);
}
