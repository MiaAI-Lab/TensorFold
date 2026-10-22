//! Layer byte oracle for glm5_next/cuda/forward.py layer_forward; tools/zig/glm_layer_dump.py writes v1 dumps.
const std = @import("std");
const cuda = @import("cuda");
pub const Dump = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    manifest: std.json.Parsed(std.json.Value),
    rows: usize,
    rank: usize,
    world: usize,
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Dump {
        const path = try std.fs.path.join(gpa, &.{ dir, "manifest.json" });
        defer gpa.free(path);
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(raw);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, raw, .{});
        errdefer parsed.deinit();
        if (parsed.value != .object) return error.BadLayerDump;
        const obj = parsed.value.object;
        for ([_]struct { []const u8, i64 }{ .{ "version", 1 }, .{ "layers", 45 }, .{ "hidden", 4096 }, .{ "streams", 4 } }) |field| {
            const val = obj.get(field[0]) orelse return error.BadLayerDump;
            if (val != .integer or val.integer != field[1]) return error.BadLayerDump;
        }
        for ([_]struct { []const u8, []const u8 }{ .{ "dtype", "bf16" }, .{ "endian", "little" } }) |field| {
            const val = obj.get(field[0]) orelse return error.BadLayerDump;
            if (val != .string or !std.mem.eql(u8, val.string, field[1])) return error.BadLayerDump;
        }
        const rows = try integer(obj, "rows");
        const rank = try integer(obj, "rank");
        const world = try integer(obj, "world");
        if (rows < 1 or rows > 64 or world < 1 or world > 4 or rank >= world) return error.BadLayerDump;
        return .{ .gpa = gpa, .io = io, .dir = dir, .manifest = parsed, .rows = rows, .rank = rank, .world = world };
    }
    fn integer(obj: std.json.ObjectMap, name: []const u8) !usize {
        const val = obj.get(name) orelse return error.BadLayerDump;
        if (val != .integer or val.integer < 0) return error.BadLayerDump;
        return @intCast(val.integer);
    }
    pub fn deinit(d: *Dump) void {
        d.manifest.deinit();
    }
    pub fn layer(d: Dump, index: usize) ![]u8 {
        if (index >= 45) return error.BadLayer;
        var name: [32]u8 = undefined;
        const path = try std.fs.path.join(d.gpa, &.{ d.dir, try std.fmt.bufPrint(&name, "layer-{d:0>2}.bf16", .{index}) });
        defer d.gpa.free(path);
        const raw = try std.Io.Dir.cwd().readFileAlloc(d.io, path, d.gpa, .limited(d.rows * 16384 * 2 + 1));
        errdefer d.gpa.free(raw);
        if (raw.len != d.rows * 16384 * 2) return error.BadLayerBytes;
        return raw;
    }
};
pub fn compare(index: usize, expected: []const u8, actual: []const u8) !void {
    if (expected.len != actual.len) return error.BadLayerBytes;
    for (expected, actual, 0..) |a, b, byte| if (a != b) {
        const elem = byte / 2;
        std.log.err("layer {d}: byte {d}, row {d}, stream {d}, column {d}: expected 0x{x:0>2}, got 0x{x:0>2}", .{ index, byte, elem / 16384, elem % 16384 / 4096, elem % 4096, a, b });
        return error.LayerMismatch;
    };
}
/// Attach call to Forward.observer for an eager device forward (capture rejects observers).
pub const Observer = struct {
    dump: *const Dump,
    stream: cuda.Stream,
    checked: usize = 0,
    pub fn call(ctx: *anyopaque, index: usize, ptr: u64, bytes: usize) !void {
        const o: *Observer = @ptrCast(@alignCast(ctx));
        if (index != o.checked or bytes != o.dump.rows * 16384 * 2) return error.BadLayerSequence;
        try o.stream.synchronize();
        const expected = try o.dump.layer(index);
        defer o.dump.gpa.free(expected);
        const actual = try o.dump.gpa.alloc(u8, bytes);
        defer o.dump.gpa.free(actual);
        const view: cuda.DeviceBuffer = .{ .d = o.stream.d, .ptr = ptr, .len = bytes };
        try view.download(0, actual);
        try compare(index, expected, actual);
        o.checked += 1;
    }
    pub fn finish(o: Observer) !void {
        if (o.checked != 45) return error.IncompleteForward;
    }
};
pub fn compareDirectories(gpa: std.mem.Allocator, io: std.Io, expected_dir: []const u8, actual_dir: []const u8) !void {
    var expected = try Dump.open(gpa, io, expected_dir);
    defer expected.deinit();
    var actual = try Dump.open(gpa, io, actual_dir);
    defer actual.deinit();
    if (expected.rows != actual.rows or expected.world != actual.world or expected.rank != actual.rank) return error.DumpGeometryMismatch;
    // The input tokens, positions, segments and initial-state fingerprints identify a forward, not just its shape.
    for ([_][]const u8{ "tokens", "segments", "initial_state_sha256" }) |key| {
        const a = expected.manifest.value.object.get(key) orelse return error.BadLayerDump;
        const b = actual.manifest.value.object.get(key) orelse return error.BadLayerDump;
        const ta = try std.json.Stringify.valueAlloc(gpa, a, .{});
        defer gpa.free(ta);
        const tb = try std.json.Stringify.valueAlloc(gpa, b, .{});
        defer gpa.free(tb);
        if (!std.mem.eql(u8, ta, tb)) return error.DumpInputMismatch;
    }
    for (0..45) |i| {
        const a = try expected.layer(i);
        defer gpa.free(a);
        const b = try actual.layer(i);
        defer gpa.free(b);
        try compare(i, a, b);
    }
}
test "layer bytes compare exactly including BF16 low byte" {
    try compare(0, &.{ 0, 0x3f, 1, 0x40 }, &.{ 0, 0x3f, 1, 0x40 });
    try std.testing.expectError(error.BadLayerBytes, compare(0, &.{0}, &.{}));
    try std.testing.expectError(error.IncompleteForward, (Observer{ .dump = undefined, .stream = undefined }).finish());
}
test "compile device observer and dump reader" {
    if (std.c.getenv("TF_GLM_P6_COMPILE_PROBE") != null) {
        try Observer.call(undefined, 0, 0, 0);
        try compareDirectories(std.testing.allocator, undefined, "", "");
    }
}
