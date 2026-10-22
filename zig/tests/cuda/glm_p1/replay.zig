//! Replay real glm5_next/cuda Triton launches against eager Python CPU pointer-span fixtures.
//! TensorFold/Ash Hart reference; preserves aliases, view offsets and original specialization.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const aot = cuda.aot;
const PtrDiff = struct { to: []const u8, from: []const u8, elem: u64 };
const ArgJson = struct { name: []const u8, type: []const u8, buffer: ?usize = null, offset: u64 = 0, int: ?i64 = null, f32: ?u32 = null, ptrdiff: ?PtrDiff = null };

fn addressOf(sample: Sample, buffers: []const cuda.DeviceBuffer, name: []const u8) !u64 {
    for (sample.args) |a| if (std.mem.eql(u8, a.name, name)) return buffers[a.buffer orelse return error.PtrDiffNotPointer].ptr + a.offset;
    for (sample.extras) |a| if (std.mem.eql(u8, a.name, name)) return buffers[a.buffer orelse return error.PtrDiffNotPointer].ptr + a.offset;
    return error.PtrDiffUnknownArgument;
}
const ConstJson = struct { int: ?i64 = null, f32: ?u32 = null };
const Sample = struct { @"fn": []const u8, hash: []const u8, phase: []const u8, grid: [3]usize, args: []ArgJson, extras: []ArgJson = &.{}, consts: std.json.ArrayHashMap(ConstJson), folder: []const u8, buffers: []usize };
const Fixtures = struct { samples: []Sample };
fn read(gpa: std.mem.Allocator, io: std.Io, root: []const u8, folder: []const u8, index: usize, suffix: []const u8) ![]u8 {
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}/{d}.{s}.bin", .{ root, folder, index, suffix });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(2 << 30));
}
pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len != 3) return error.ExpectedAotDirectoryAndCaptureRankDirectory;
    const gpa = init.gpa;
    const io = init.io;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var set = try aot.Set.load(gpa, io, &driver, ctx.device, argv[1]);
    defer set.deinit();
    var memo: glm.triton.Memo = .{ .gpa = gpa };
    defer memo.deinit();
    const tri: glm.triton.Tri = .{ .set = &set, .s = stream, .memo = &memo };
    const path = try std.fs.path.join(gpa, &.{ argv[2], "replay.json" });
    defer gpa.free(path);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(raw);
    const parsed = try std.json.parseFromSlice(Fixtures, gpa, raw, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    if (parsed.value.samples.len == 0) return error.EmptyFixtureSet;
    for (parsed.value.samples) |sample| {
        const buffers = try gpa.alloc(cuda.DeviceBuffer, sample.buffers.len);
        defer gpa.free(buffers);
        var loaded: usize = 0;
        defer for (buffers[0..loaded]) |*b| b.free();
        for (sample.buffers, 0..) |size, i| {
            const bytes = try read(gpa, io, argv[2], sample.folder, i, "in");
            defer gpa.free(bytes);
            if (bytes.len != size) return error.FixtureSizeMismatch;
            buffers[i] = try cuda.DeviceBuffer.alloc(&driver, size);
            loaded += 1;
            try buffers[i].upload(0, bytes);
        }
        var args: [64]aot.Arg = undefined;
        if (sample.args.len > args.len) return error.TooManyArguments;
        for (sample.args, 0..) |arg, i| {
            if (arg.ptrdiff) |d| {
                // Address-dependent scalar (hc_post_pair's RS): recomputed from this replay's own buffer addresses.
                const to: i64 = @bitCast(try addressOf(sample, buffers, d.to));
                const from: i64 = @bitCast(try addressOf(sample, buffers, d.from));
                const v = @divFloor(to - from, @as(i64, @intCast(d.elem)));
                // Triton types an integer argument by magnitude: i32 when it fits, else i64.
                args[i] = if (std.math.cast(i32, v)) |small| aot.int(arg.name, small) else aot.word(arg.name, @bitCast(v));
                continue;
            }
            args[i] = if (arg.buffer) |b| aot.ptr(arg.name, arg.type, buffers[b].ptr + arg.offset) else if (arg.f32) |v| aot.float(arg.name, @bitCast(v)) else if (std.mem.eql(u8, arg.type, "i32")) aot.int(arg.name, @intCast(arg.int.?)) else aot.word(arg.name, @bitCast(arg.int.?));
        }
        var consts: [64]aot.Const = undefined;
        const map = sample.consts.map;
        if (map.count() > consts.len) return error.TooManyConstants;
        for (map.keys(), map.values(), 0..) |name, value, i| {
            consts[i] = .{ .name = name, .int = value.int, .f32 = if (value.f32) |v| @as(f32, @bitCast(v)) else null };
        }
        try tri.run(sample.@"fn", sample.grid, args[0..sample.args.len], consts[0..map.count()]);
        try stream.synchronize();
        for (buffers, 0..) |buffer, i| {
            const want = try read(gpa, io, argv[2], sample.folder, i, "out");
            defer gpa.free(want);
            const got = try gpa.alloc(u8, want.len);
            defer gpa.free(got);
            try buffer.download(0, got);
            if (!std.mem.eql(u8, want, got)) {
                for (want, got, 0..) |w, g, at| if (w != g) {
                    std.log.err("{s} {s} buffer {d} byte {d}: Python {x}, Zig {x}", .{ sample.@"fn", sample.phase, i, at, w, g });
                    break;
                };
                return error.TritonByteMismatch;
            }
        }
        std.log.info("PASS {s} {s} ({s})", .{ sample.@"fn", sample.phase, sample.hash });
    }
    std.log.info("all {d} real Python Triton fixtures byte-exact", .{parsed.value.samples.len});
}
