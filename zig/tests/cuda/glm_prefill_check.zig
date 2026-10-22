//! tools/zig/glm_prefill_dump.py version 2 byte comparison. CPU-only; no CUDA driver is opened.
const std = @import("std");
const glm = @import("glm_cuda");
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) return error.ExpectedReferenceAndActualDirectories;
    try glm.prefill_check.compareDirectories(init.gpa, init.io, args[1], args[2]);
    std.debug.print("PASS: all 45 prompt-chunk layers match byte for byte\n", .{});
}
