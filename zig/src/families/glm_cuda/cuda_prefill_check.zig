//! Prompt-chunk byte oracle. tools/zig/glm_prefill_dump.py writes version 2. Version 1 is the decode window oracle.
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
    pos: usize,
    grid: usize,
    chunk: usize,
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Dump {
        const path = try std.fs.path.join(gpa, &.{ dir, "manifest.json" });
        defer gpa.free(path);
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(raw);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, raw, .{});
        errdefer parsed.deinit();
        if (parsed.value != .object) return error.BadLayerDump;
        const obj = parsed.value.object;
        for ([_]struct { []const u8, i64 }{ .{ "version", 2 }, .{ "layers", 45 }, .{ "hidden", 4096 }, .{ "streams", 4 } }) |field| {
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
        const pos = try integer(obj, "pos");
        const prompt_grid = try integer(obj, "grid");
        const chunk = try integer(obj, "chunk");
        if (rows < 1 or rows > 2048 or world < 1 or world > 4 or rank >= world) return error.BadLayerDump;
        return .{ .gpa = gpa, .io = io, .dir = dir, .manifest = parsed, .rows = rows, .rank = rank, .world = world, .pos = pos, .grid = prompt_grid, .chunk = chunk };
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
        std.log.err("prompt layer {d}: byte {d}, row {d}, stream {d}, column {d}: expected 0x{x:0>2}, got 0x{x:0>2}", .{ index, byte, elem / 16384, elem % 16384 / 4096, elem % 4096, a, b });
        return error.LayerMismatch;
    };
}

/// Attach call to Engine.observer. It synchronizes at the layer boundary, so it stays off a captured graph.
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
    if (expected.rows != actual.rows or expected.world != actual.world or expected.rank != actual.rank or expected.pos != actual.pos or expected.grid != actual.grid or expected.chunk != actual.chunk) return error.DumpGeometryMismatch;
    for ([_][]const u8{ "tokens", "segments", "initial_state_sha256", "pos", "grid", "chunk" }) |key| {
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

const manifest_v2 =
    \\{"version":2,"layers":45,"hidden":4096,"streams":4,"rows":8,"rank":0,"world":1,"pos":2048,"grid":64,"chunk":2048,"dtype":"bf16","endian":"little","tokens":[1,2],"segments":[{"slot":0,"pos":2048,"rows":8}],"initial_state_sha256":"aa"}
;

fn writeDump(dir: std.Io.Dir, io: std.Io, body: []const u8) !void {
    try dir.writeFile(io, .{ .sub_path = "manifest.json", .data = body });
}

test "version 2 rejects a decode dump and rows outside 1..2048" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    try writeDump(tmp.dir, io, "{\"version\":1,\"layers\":45,\"hidden\":4096,\"streams\":4,\"rows\":8,\"rank\":0,\"world\":1,\"dtype\":\"bf16\",\"endian\":\"little\"}");
    const v1 = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer std.testing.allocator.free(v1);
    try std.testing.expectError(error.BadLayerDump, Dump.open(std.testing.allocator, io, v1));
    try writeDump(tmp.dir, io, "{\"version\":2,\"layers\":45,\"hidden\":4096,\"streams\":4,\"rows\":0,\"rank\":0,\"world\":1,\"pos\":0,\"grid\":64,\"chunk\":2048,\"dtype\":\"bf16\",\"endian\":\"little\"}");
    try std.testing.expectError(error.BadLayerDump, Dump.open(std.testing.allocator, io, v1));
    try writeDump(tmp.dir, io, "{\"version\":2,\"layers\":45,\"hidden\":4096,\"streams\":4,\"rows\":2049,\"rank\":0,\"world\":1,\"pos\":0,\"grid\":64,\"chunk\":2048,\"dtype\":\"bf16\",\"endian\":\"little\"}");
    try std.testing.expectError(error.BadLayerDump, Dump.open(std.testing.allocator, io, v1));
    try writeDump(tmp.dir, io, manifest_v2);
    var dump = try Dump.open(std.testing.allocator, io, v1);
    defer dump.deinit();
    try std.testing.expectEqual(@as(usize, 2048), dump.pos);
    try std.testing.expectEqual(@as(usize, 64), dump.grid);
    try std.testing.expectEqual(@as(usize, 2048), dump.chunk);
}

test "prompt manifests must agree on position, grid and chunk before any layer byte" {
    var a = std.testing.tmpDir(.{});
    defer a.cleanup();
    var b = std.testing.tmpDir(.{});
    defer b.cleanup();
    const io = std.testing.io;
    try writeDump(a.dir, io, manifest_v2);
    try writeDump(b.dir, io, "{\"version\":2,\"layers\":45,\"hidden\":4096,\"streams\":4,\"rows\":8,\"rank\":0,\"world\":1,\"pos\":0,\"grid\":64,\"chunk\":2048,\"dtype\":\"bf16\",\"endian\":\"little\",\"tokens\":[1,2],\"segments\":[{\"slot\":0,\"pos\":2048,\"rows\":8}],\"initial_state_sha256\":\"aa\"}");
    const da = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{a.sub_path});
    defer std.testing.allocator.free(da);
    const db = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{b.sub_path});
    defer std.testing.allocator.free(db);
    try std.testing.expectError(error.DumpGeometryMismatch, compareDirectories(std.testing.allocator, io, da, db));
    try compare(3, &.{ 0, 0x3f }, &.{ 0, 0x3f });
    try std.testing.expectError(error.BadLayerBytes, compare(3, &.{0}, &.{}));
    try std.testing.expectError(error.IncompleteForward, (Observer{ .dump = undefined, .stream = undefined }).finish());
}

test "compile the prompt observer without a GPU" {
    if (std.c.getenv("TF_GLM_P8_COMPILE_PROBE") != null) {
        try Observer.call(undefined, 0, 0, 0);
        try compareDirectories(std.testing.allocator, undefined, "", "");
    }
}
