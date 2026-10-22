//! glm5_next/cuda/verify.py BatchedVerify: graphs keyed only by R, device positions/segments/parities.
//! Ash Hart and TensorFold contributors; graph capture pattern from TensorFold Flash Next.
const std = @import("std");
const cuda = @import("cuda");
const Forward = @import("cuda_forward.zig").Forward;
const State = @import("cuda_state.zig").State;
const segments = @import("cuda_segments.zig");
const Stager = @import("cuda_stage.zig").Stager;
pub const Segment = struct { st: *const State, tokens: []const i32 };
pub const Verify = struct {
    f: *Forward,
    tables: *segments.SegRows,
    kseg: cuda.DeviceBuffer,
    graphs: [65]?cuda.graph.Exec = @splat(null),
    host: [8][6]i32 = @splat(@splat(0)),
    used: usize = 0,
    /// Pinned, stream-ordered staging of the per-round metadata (ids, row/segment tables, KDA table). Unset: the
    /// synchronous path (drain the stream, blocking uploads).
    stager: Stager = .{},
    pub fn init(f: *Forward, tables: *segments.SegRows) !Verify {
        if (f.b.rows > 64 or tables.max_segs > 8 or tables.rows != f.b.rows or tables.dev == null) return error.BadWindow;
        const kseg = try cuda.DeviceBuffer.alloc(f.tri.s.d, tables.max_segs * 6 * 4);
        f.tables = tables;
        f.kseg = kseg.ptr;
        var stager: Stager = .{};
        if (std.c.getenv("TF_GLM_SYNC_STAGE") == null) stager = try Stager.init(f.tri.s.d, f.tri.s, 64 << 10);
        return .{ .f = f, .tables = tables, .kseg = kseg, .stager = stager };
    }
    pub fn deinit(v: *Verify) void {
        v.stager.deinit();
        for (&v.graphs) |*g| if (g.*) |*e| e.deinit();
        v.kseg.free();
        v.f.kseg = 0;
    }
    /// No committed state changes. Validate the entire window before replacing device tables.
    pub fn stage(v: *Verify, spans: []const Segment) !usize {
        var descriptions: [8]segments.Segment = undefined;
        var table: [8][6]i32 = @splat(@splat(0));
        if (spans.len == 0 or spans.len > v.tables.max_segs) return error.BadSegments;
        var total: usize = 0;
        for (spans, 0..) |s, si| {
            if (s.tokens.len == 0 or s.st.slot >= v.f.slots.count or s.st.pos > s.st.capacity or s.tokens.len > s.st.capacity - s.st.pos or s.tokens.len > v.f.b.rows - total) return error.BadSegment;
            for (spans[0..si]) |old| if (old.st.slot == s.st.slot) return error.DuplicateSlot;
            for (s.st.cur) |parity| if (parity != s.st.cur[0]) return error.MixedKdaParity;
            for (s.tokens) |token| if (token < 0 or token >= 154880) return error.BadToken;
            descriptions[si] = .{ .base = s.st.base, .pos = s.st.pos, .rows = s.tokens.len, .ring_base = s.st.slot * v.f.caches.ring };
            table[si] = .{ @intCast(total), @intCast(s.tokens.len), @intCast(s.st.slot), s.st.cur[0], @intCast(s.st.slot), @intCast(s.tokens.len) };
            total += s.tokens.len;
        }
        for (table[spans.len..v.tables.max_segs]) |*row| row.* = .{ @intCast(total), 0, 0, 0, 0, 0 };
        _ = try v.tables.fill(descriptions[0..spans.len], 2051);
        if (v.stager.ready()) {
            // Stream-ordered: the copies queue behind the previous round's graph, which may still read the tables.
            var ids: [64]i32 = undefined;
            var n: usize = 0;
            for (spans) |s| {
                @memcpy(ids[n..][0..s.tokens.len], s.tokens);
                n += s.tokens.len;
            }
            try v.stager.put(v.f.b.ids, std.mem.sliceAsBytes(ids[0..n]));
            try v.tables.stageAsync(&v.stager);
            try v.stager.put(v.kseg.ptr, std.mem.asBytes(&table)[0 .. v.tables.max_segs * 24]);
        } else {
            // A graph may still be reading the previous round's tables. Synchronous staging needs no pinned lifetime.
            try v.f.tri.s.synchronize();
            var row: usize = 0;
            for (spans) |s| {
                const view: cuda.DeviceBuffer = .{ .d = v.f.tri.s.d, .ptr = v.f.b.ids, .len = v.f.b.rows * 4 };
                try view.upload(row * 4, std.mem.sliceAsBytes(s.tokens));
                row += s.tokens.len;
            }
            try v.tables.stage();
            try v.kseg.upload(0, std.mem.asBytes(&table)[0 .. v.tables.max_segs * 24]);
        }
        v.host = table;
        v.used = total;
        return total;
    }
    /// Caller stages a valid dummy window first. Warm-up resolves AOT variants before entering capture.
    /// Capture before stream admission; clear slot 0 after capturing every R as Python does.
    pub fn capture(v: *Verify) !void {
        const r = v.used;
        if (r == 0 or r > 64) return error.BadRows;
        if (v.graphs[r] != null) return error.AlreadyCaptured;
        if (v.f.observer != null) return error.ObserverDuringCapture;
        try v.f.compute(r);
        try v.f.compute(r);
        try v.f.tri.s.synchronize();
        try cuda.graph.beginCapture(v.f.tri.s, .global);
        v.f.compute(r) catch |e| {
            if (cuda.graph.endCapture(v.f.tri.s)) |g0| {
                var g = g0;
                g.deinit();
            } else |_| {}
            return e;
        };
        var graph = try cuda.graph.endCapture(v.f.tri.s);
        defer graph.deinit();
        var executable = try graph.instantiate();
        errdefer executable.deinit();
        try executable.upload(v.f.tri.s);
        v.graphs[r] = executable;
    }
    /// BatchedVerify.capture's full 1..window graph table, before any slot is admitted.
    pub fn captureAll(v: *Verify, home: *const State) !void {
        if (home.slot != 0 or home.base != 0 or home.pos != 0) return error.CaptureAfterAdmission;
        for (home.cur) |parity| if (parity != 0) return error.CaptureAfterAdmission;
        var tokens: [64]i32 = @splat(0);
        for (1..v.f.b.rows + 1) |rows| {
            _ = try v.stage(&.{.{ .st = home, .tokens = tokens[0..rows] }});
            try v.capture();
        }
        try v.f.tri.s.synchronize();
        const sizes = @import("cuda_state.zig").Slots.layout(v.f.g, 1, v.f.slots.rows);
        const rec: cuda.DeviceBuffer = .{ .d = v.f.tri.s.d, .ptr = v.f.slots.rec, .len = sizes[0] };
        const conv: cuda.DeviceBuffer = .{ .d = v.f.tri.s.d, .ptr = v.f.slots.conv, .len = sizes[1] };
        try rec.fill8(0, null);
        try conv.fill8(0, null);
        v.used = 0;
    }
    pub fn forward(v: *Verify, spans: []const Segment) !void {
        const r = try v.stage(spans);
        if (v.graphs[r]) |graph| try graph.launchOn(v.f.tri.s) else try v.f.compute(r);
    }
};
test "compile BatchedVerify graph and staging entry points" {
    std.testing.refAllDecls(Verify);
    if (std.c.getenv("TF_GLM_P6_COMPILE_PROBE") != null) {
        var v = try Verify.init(undefined, undefined);
        defer v.deinit();
        try v.capture();
        try v.captureAll(undefined);
        try v.forward(&.{});
    }
}

test "invalid verify windows cannot reach staging or mutate tables" {
    var slots: @import("cuda_state.zig").Slots = undefined;
    slots.count = 2;
    var caches: @import("cuda_state.zig").Caches = undefined;
    caches.ring = 64;
    var f: Forward = undefined;
    f.slots = &slots;
    f.caches = &caches;
    f.b.rows = 4;
    var tables = try segments.SegRows.init(std.testing.allocator, 4, 4, 64);
    defer tables.deinit();
    var v: Verify = .{ .f = &f, .tables = &tables, .kseg = undefined };
    var st: State = undefined;
    st.slot = 0;
    st.base = 0;
    st.pos = 5;
    st.capacity = 8;
    st.cur = @splat(0);
    try std.testing.expectError(error.BadSegments, v.stage(&.{}));
    try std.testing.expectError(error.BadSegment, v.stage(&.{.{ .st = &st, .tokens = &.{ 1, 2, 3, 4 } }}));
    try std.testing.expectError(error.BadToken, v.stage(&.{.{ .st = &st, .tokens = &.{154880} }}));
    try std.testing.expectError(error.DuplicateSlot, v.stage(&.{ .{ .st = &st, .tokens = &.{1} }, .{ .st = &st, .tokens = &.{2} } }));
    st.cur[33] = 1;
    try std.testing.expectError(error.MixedKdaParity, v.stage(&.{.{ .st = &st, .tokens = &.{1} }}));
    try std.testing.expectEqual(@as(usize, 0), tables.used);
    for (tables.host) |word| try std.testing.expectEqual(@as(i32, 0), word);
}
