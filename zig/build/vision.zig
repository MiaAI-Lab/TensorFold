//! The server's view of the vision module (zig/src/vision): `json` is the server's own JSON value type, shared with it.
const std = @import("std");

pub fn attach(b: *std.Build, m: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    const json = b.createModule(.{ .root_source_file = b.path("zig/src/server/json.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const vision = b.createModule(.{ .root_source_file = b.path("zig/src/vision/vision.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "json", .module = json }} });
    m.addImport("json", json);
    m.addImport("vision", vision);
}
