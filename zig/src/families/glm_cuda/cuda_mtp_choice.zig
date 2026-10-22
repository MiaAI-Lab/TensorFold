//! Port of glm5_next/cuda/engine.py:mtp_head and the recipe scripts/config.sh drafter contract.
//! Authored by Ash Hart and TensorFold contributors; Zig CUDA serving path authored by jschmied.
const std = @import("std");
pub const Choice = enum { dflash2, mtp };
pub fn parse(value: []const u8) !Choice {
    if (std.mem.eql(u8, value, "dflash2")) return .dflash2;
    if (std.mem.eql(u8, value, "mtp")) return .mtp;
    return error.BadDrafter;
}
pub fn environment() !Choice {
    return parse(if (std.c.getenv("DRAFTER")) |v| std.mem.span(v) else "dflash2");
}
pub fn load(value: []const u8, choice: Choice, drafts: bool, layers: usize, streams: usize) !bool {
    const v = std.mem.trim(u8, value, " \t\r\n");
    const auto = v.len == 0 or std.ascii.eqlIgnoreCase(v, "auto");
    if (!auto and !std.mem.eql(u8, v, "0") and !std.mem.eql(u8, v, "1")) return error.BadMtpSetting;
    if (drafts and choice == .mtp and streams != 1) return error.MtpRequiresSingleStream;
    const on = layers != 0 and streams == 1 and (std.mem.eql(u8, v, "1") or (auto and choice == .mtp and drafts));
    if (drafts and choice == .mtp and !on) return error.MtpHeadUnavailable;
    return on;
}
pub fn setting() []const u8 {
    return if (std.c.getenv("TF_GLM_MTP")) |v| std.mem.span(v) else "auto";
}
test "recipe defaults to DFlash2 and auto excludes MTP; explicit single-stream MTP loads" {
    try std.testing.expectEqual(Choice.dflash2, try parse("dflash2"));
    for ([_]bool{ false, true }) |drafts| try std.testing.expect(!try load("auto", .dflash2, drafts, 1, 1));
    try std.testing.expect(try load(" AUTO ", .mtp, true, 1, 1));
    try std.testing.expect(try load("1", .dflash2, false, 1, 1));
    try std.testing.expect(!try load("1", .dflash2, true, 1, 8));
    try std.testing.expectError(error.MtpHeadUnavailable, load("0", .mtp, true, 1, 1));
    try std.testing.expectError(error.MtpHeadUnavailable, load("auto", .mtp, true, 0, 1));
    try std.testing.expectError(error.MtpRequiresSingleStream, load("auto", .mtp, true, 1, 2));
    try std.testing.expectError(error.BadMtpSetting, load("yes", .dflash2, true, 1, 1));
    try std.testing.expectError(error.BadDrafter, parse("other"));
}
