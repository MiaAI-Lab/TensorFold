//! P4 boundary: glm5_next/cuda/kda.py chain_segments / kda.cpp chain_wide_segments.
//! No replacement recurrence math: P4 installs chain_segments after loading all functions before capture.
const cuda = @import("cuda");
pub const Scratch = struct { out: u64, k: u64, v: u64, g: u64, b: u64, q_tmp: u64, y_tmp: u64 };
pub const Request = struct {
    seg: u64,
    nseg: usize,
    p: u64,
    p_stride: usize,
    b_off: usize,
    a: u64,
    a_stride: usize,
    g: u64,
    g_stride: usize,
    conv: u64,
    conv_slot: usize,
    conv_w: u64,
    rec: u64,
    rec_slot: usize,
    rec_parity: usize,
    a_log: u64,
    dt_bias: u64,
    norm_w: u64,
    eps: f32 = 1e-5,
    lower: f32 = -5,
    rows: usize,
    heads: usize,
    scratch: Scratch,
};
pub const Interface = struct {
    ctx: ?*anyopaque = null,
    chain_segments: *const fn (?*anyopaque, cuda.Stream, Request) anyerror!void = missing,
    fn missing(_: ?*anyopaque, _: cuda.Stream, _: Request) !void {
        return error.MissingKdaPackage;
    }
    pub fn run(k: Interface, stream: cuda.Stream, r: Request) !void {
        try k.chain_segments(k.ctx, stream, r);
    }
};
test "uninstalled KDA fails explicitly without touching CUDA" {
    const std = @import("std");
    try std.testing.expectError(error.MissingKdaPackage, (Interface{}).run(undefined, undefined));
}
