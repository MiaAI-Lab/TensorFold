//! Upload cuda_weights.Plan: glm5_next/cuda/split.py RankReader.get and weights.py on_device.
//! Raw checkpoint bits only; prepared Q4/F8 projections are a separate load-time transformation.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
pub const DeviceWeights = struct {
    buffer: cuda.DeviceBuffer,
    plan: *const weights.Plan,
    pub fn deinit(w: *DeviceWeights) void {
        w.buffer.free();
    }
    pub fn tensor(w: DeviceWeights, name: []const u8) !u64 {
        for (w.plan.items.items) |item| if (std.mem.eql(u8, item.src.name, name)) return w.buffer.ptr + item.device_offset;
        return error.MissingWeight;
    }
    /// At most 8 MiB staging. Column/dim1 slices seek each contiguous rank span rather than reading a whole shard.
    /// Names and metadata remain owned by Plan/Catalog; both must outlive this value.
    pub fn upload(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, plan: *const weights.Plan, model_dir: []const u8, draft_dir: ?[]const u8) !DeviceWeights {
        var buf = try cuda.DeviceBuffer.alloc(d, @intCast(plan.device_bytes));
        errdefer buf.free();
        try buf.fill8(0, null);
        const staging = try gpa.alloc(u8, 8 << 20);
        defer gpa.free(staging);
        const window = try gpa.alloc(u8, 32 << 20);
        defer gpa.free(window);
        for (plan.items.items) |item| {
            const is_draft = item.src.source == .dflash2;
            const dir = if (is_draft) draft_dir orelse return error.MissingDraftDirectory else model_dir;
            const file_name = if (is_draft) item.src.file["dflash2:".len..] else item.src.file;
            const path = try std.fs.path.join(gpa, &.{ dir, file_name });
            defer gpa.free(path);
            const file = try std.Io.Dir.cwd().openFile(io, path, .{});
            defer file.close(io);
            const spans = try SliceSpans.init(item);
            var i: u64 = 0;
            if (spans.bytes > staging.len) {
                // large spans: the staging buffer cycles through each one
                while (i < spans.count) : (i += 1) {
                    var copied: u64 = 0;
                    while (copied < spans.bytes) {
                        const n: usize = @intCast(@min(staging.len, spans.bytes - copied));
                        if (try file.readPositionalAll(io, staging[0..n], spans.file_begin + i * spans.file_stride + copied) != n) return error.TruncatedWeight;
                        try buf.upload(@intCast(item.device_offset + i * spans.bytes + copied), staging[0..n]);
                        copied += n;
                    }
                }
            }
            while (i < spans.count) {
                const n = try spans.readRun(io, file, i, staging, window);
                try buf.upload(@intCast(item.device_offset + i * spans.bytes), staging[0 .. n * spans.bytes]);
                i += n;
            }
        }
        return .{ .buffer = buf, .plan = plan };
    }
};
/// A row/column/dim1 rank slice decomposed into contiguous file spans (all offsets in bytes).
pub const SliceSpans = struct {
    file_begin: u64,
    file_stride: u64,
    bytes: u64,
    count: u64,
    /// Reads spans `first..` into `out` (contiguously, `bytes` each), as many as fit: a strided slice is read as
    /// whole file windows (one sequential pread covers many spans) instead of one small read per span.
    /// Returns the number of spans read.
    pub fn readRun(sp: SliceSpans, io: std.Io, file: std.Io.File, first: u64, out: []u8, window: []u8) !u64 {
        const fit: u64 = @max(1, @as(u64, out.len) / sp.bytes);
        var n = @min(fit, sp.count - first);
        const at = sp.file_begin + first * sp.file_stride;
        if (n > 1 or sp.bytes <= out.len) {
            if (sp.file_stride > sp.bytes and n * sp.file_stride <= window.len + sp.file_stride - sp.bytes) {
                // a strided run inside one window
                n = @min(n, (@as(u64, window.len) - sp.bytes) / sp.file_stride + 1);
                const len: usize = @intCast((n - 1) * sp.file_stride + sp.bytes);
                if (try file.readPositionalAll(io, window[0..len], at) != len) return error.TruncatedWeight;
                for (0..@as(usize, @intCast(n))) |k| @memcpy(out[k * @as(usize, @intCast(sp.bytes)) ..][0..@intCast(sp.bytes)], window[k * @as(usize, @intCast(sp.file_stride)) ..][0..@intCast(sp.bytes)]);
                return n;
            }
            if (sp.file_stride == sp.bytes) {
                const len: usize = @intCast(n * sp.bytes);
                if (try file.readPositionalAll(io, out[0..len], at) != len) return error.TruncatedWeight;
                return n;
            }
        }
        if (sp.bytes > out.len) {
            // one span larger than the staging: the caller's staging cannot hold it; read its first part
            return error.SpanLargerThanStaging;
        }
        const len: usize = @intCast(sp.bytes);
        if (try file.readPositionalAll(io, out[0..len], at) != len) return error.TruncatedWeight;
        return 1;
    }
    pub fn init(item: weights.Item) !SliceSpans {
        const src = item.src;
        const sl = item.slice;
        var inner: u64 = src.dtype_size;
        const axis: usize = sl.axis orelse return .{ .file_begin = src.header_base + src.data_begin, .file_stride = sl.device_bytes, .bytes = sl.device_bytes, .count = 1 };
        if (axis >= src.rank or sl.stop > src.shape[axis] or sl.start > sl.stop) return error.BadWeightSlice;
        for (src.shape[axis + 1 .. src.rank]) |n| inner *= n;
        var outer: u64 = 1;
        for (src.shape[0..axis]) |n| outer *= n;
        const bytes = (sl.stop - sl.start) * inner;
        if (outer * bytes != sl.device_bytes) return error.UnsupportedWeightConversion;
        return .{ .file_begin = src.header_base + src.data_begin + sl.start * inner, .file_stride = src.shape[axis] * inner, .bytes = bytes, .count = outer };
    }
};
test "noncontiguous trellis dim1 and dense column slices preserve raw bits" {
    var item: weights.Item = undefined;
    item.src = .{ .name = "trellis", .file = "x", .dtype = "I16", .dtype_size = 2, .rank = 3, .shape = .{ 256, 128, 64, 1, 1, 1, 1, 1 }, .header_base = 100, .data_begin = 20, .data_end = 0 };
    item.slice = .{ .kind = .col, .axis = 1, .start = 48, .stop = 88, .device_dtype = "I32", .device_bytes = 256 * 40 * 64 * 2 };
    const spans = try SliceSpans.init(item);
    try std.testing.expectEqual(@as(u64, 120 + 48 * 128), spans.file_begin);
    try std.testing.expectEqual(@as(u64, 256), spans.count);
    try std.testing.expectEqual(@as(u64, 128 * 128), spans.file_stride);
    try std.testing.expectEqual(@as(u64, 40 * 128), spans.bytes);
}
test "analyze upload without loading CUDA" {
    if (std.c.getenv("TF_GLM_P6_COMPILE_PROBE") != null) {
        var w = try DeviceWeights.upload(std.testing.allocator, undefined, undefined, undefined, "", null);
        defer w.deinit();
    }
}

test "readRun reads strided and contiguous spans like one read per span" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    var data: [4096]u8 = undefined;
    for (&data, 0..) |*b, i| b.* = @truncate(i *% 31 +% (i >> 8));
    try tmp.dir.writeFile(io, .{ .sub_path = "f", .data = &data });
    const file = try tmp.dir.openFile(io, "f", .{});
    defer file.close(io);
    const cases = [_]SliceSpans{
        .{ .file_begin = 100, .file_stride = 200, .bytes = 40, .count = 17 },
        .{ .file_begin = 7, .file_stride = 64, .bytes = 64, .count = 30 },
        .{ .file_begin = 0, .file_stride = 1000, .bytes = 300, .count = 4 },
    };
    for (cases) |sp| {
        for ([_]usize{ 50, 128, 700, 4000 }) |staging_len| {
            for ([_]usize{ 300, 1000, 4096 }) |window_len| {
                var staging: [4096]u8 = undefined;
                var window: [4096]u8 = undefined;
                if (staging_len < sp.bytes) continue;
                var i: u64 = 0;
                while (i < sp.count) {
                    const n = try sp.readRun(io, file, i, staging[0..staging_len], window[0..window_len]);
                    try std.testing.expect(n >= 1);
                    for (0..@as(usize, @intCast(n))) |k| {
                        const want = data[@intCast(sp.file_begin + (i + k) * sp.file_stride)..][0..@intCast(sp.bytes)];
                        try std.testing.expectEqualSlices(u8, want, staging[k * @as(usize, @intCast(sp.bytes)) ..][0..@intCast(sp.bytes)]);
                    }
                    i += n;
                }
            }
        }
    }
}
