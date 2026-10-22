//! Port of glm5_next/cuda/mtp.py:mtp_stage/mtp_compute and decode.py:absorb/draft.
//! Authored by Ash Hart and TensorFold contributors; jschmied authored the Zig CUDA serving path.
//! Replay the Python Triton arithmetic; missing variants fail closed, never synthesize cubins.
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;
const state = @import("cuda_state.zig");
const fw = @import("cuda_forward.zig");
const pf = @import("cuda_prefill.zig");
const mm = @import("cuda_matmuls.zig");
const seg = @import("cuda_segments.zig");
const copy_mod = @import("cuda_copy_drafts.zig");
const Tri = @import("cuda_triton.zig").Tri;

/// Only chained rows are speculative. Accepted target rows always replace them on absorb.
pub const Cursor = struct {
    len: usize = 0,
    drafted: usize = 0,
    pub fn rollback(c: *Cursor) !void {
        if (c.drafted > c.len) return error.BadMtpCursor;
        c.len -= c.drafted;
        c.drafted = 0;
    }
    pub fn advance(c: *Cursor, rows: usize, speculative: bool) void {
        c.len += rows;
        if (speculative) c.drafted += rows;
    }
};
pub fn residual(t: Tri, x: u64, g: u64, rows: usize, world: usize) !void {
    try t.run("_residual_add", .{ rows, 4, 1 }, &.{ aot.ptr("X", "*bf16", x), aot.ptr("XOUT", "*bf16", x), aot.ptr("G", "*fp32", g), aot.int("RS", @intCast(rows * 4096)) }, &.{ aot.ci("D", 4096), aot.ci("WORLD", @intCast(world)), aot.ci("BLOCK", 1024) });
}
/// Strided halves of [embedding norm | hidden norm], with SUMS=false and the Python pointer ABI.
pub fn inputNorm(t: Tri, embedding: u64, hidden: u64, cat: u64, w: fw.MtpW, rows: usize) !void {
    for ([_]u64{ embedding, hidden }, [_]u64{ w.enorm, w.hnorm }, 0..) |x, norm, i| {
        const out = cat + i * 8192;
        try t.run("_rmsnorm", .{ rows, 1, 1 }, &.{ aot.ptr("X", "*bf16", x), aot.int("x_stride", 4096), aot.ptr("W", "*bf16", norm), aot.ptr("OUT", "*bf16", out), aot.int("o_stride", 8192), aot.ptr("XS", "*bf16", out), aot.float("eps", 1e-5) }, &.{ aot.ci("D", 4096), aot.ci("BLOCK", 4096), aot.ci("SUMS", 0) });
    }
}

pub const Head = struct {
    cache: state.Caches,
    b: state.Buffers,
    tables: seg.SegRows,
    arena: state.Arena,
    me: u64,
    hin: u64,
    cat: u64,
    mx: u64,
    pending: u64,
    cursor: Cursor = .{},
    copies: ?copy_mod.Copy = null,
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, g: state.Geometry, capacity: usize) !Head {
        var cache = try state.Caches.initMtp(d, capacity, 2048);
        errdefer cache.deinit();
        var b = try state.Buffers.init(d, g, 64, false);
        errdefer b.deinit();
        var tables = try seg.SegRows.init(gpa, 64, 4, cache.ring);
        errdefer tables.deinit();
        try tables.initDevice(d);
        const sizes = [_]usize{ 2048 * 8192, 2048 * 8192, 2048 * 16384, 2048 * 8192, 8192 };
        var arena = try state.Arena.init(d, &sizes);
        errdefer arena.deinit();
        const me = arena.take(sizes[0]);
        const hin = arena.take(sizes[1]);
        const cat = arena.take(sizes[2]);
        const mx = arena.take(sizes[3]);
        const pending = arena.take(sizes[4]);
        return .{ .cache = cache, .b = b, .tables = tables, .arena = arena, .me = me, .hin = hin, .cat = cat, .mx = mx, .pending = pending };
    }
    pub fn reset(h: *Head, gpa: std.mem.Allocator, prompt_ids: []const u32) !void {
        h.cursor = .{};
        if (h.copies) |*c| c.deinit();
        h.copies = null;
        if (std.c.getenv("TF_GLM_COPY_DRAFTS")) |v| if (std.mem.eql(u8, std.mem.span(v), "0")) return;
        h.copies = try copy_mod.Copy.init(gpa, prompt_ids, .{});
        h.copies.?.prompt = prompt_ids.len;
    }
    pub fn deinit(h: *Head) void {
        if (h.copies) |*c| c.deinit();
        h.arena.deinit();
        h.tables.deinit();
        h.b.deinit();
        h.cache.deinit();
    }
    fn stage(h: *Head, e: anytype, tokens: []const u32, hidden: u64, ids: u64, max_rows: usize) !void {
        if (tokens.len == 0 or tokens.len > max_rows or h.cursor.len + tokens.len > h.cache.rows) return error.BadMtpRows;
        var host: [2048]i32 = undefined;
        for (tokens, 0..) |v, i| {
            if (v >= 154880) return error.BadToken;
            host[i] = @intCast(v);
        }
        try e.stream.synchronize();
        try (cuda.DeviceBuffer{ .d = e.stream.d, .ptr = ids, .len = max_rows * 4 }).upload(0, std.mem.sliceAsBytes(host[0..tokens.len]));
        try e.torch.on(e.stream).base.copy(h.hin, hidden, tokens.len * 8192);
    }
    fn inputs(h: *Head, e: anytype, ids: u64, rows: usize) !void {
        const w = e.prepared.weights.mtp.?;
        try e.fwd.tri.embed(ids, e.prepared.weights.embed, h.me, rows, 1);
        if (h.cursor.len == 0) try e.stream.d.check(e.stream.d.api.cuMemsetD8Async(h.me, 0, 8192, e.stream.handle), "MTP zero position-0 embedding");
        try inputNorm(e.fwd.tri, h.me, h.hin, h.cat, w, rows);
    }
    fn post(h: *Head, e: anytype, part: u64, gathered: u64, rows: usize) !void {
        if (e.comm) |cm| try cm.allGather(part, gathered, rows * 4096, .f32, e.stream.handle);
        try residual(e.fwd.tri, h.mx, if (e.g.world == 1) part else gathered, rows, e.g.world);
    }
    /// Prompt arithmetic: use the prefill projections and DSA/MoE kernels, with the MTP cache and origin.
    /// The target's last hidden row is saved separately; its logits are retained for first-token sampling.
    pub fn prompt(h: *Head, e: anytype, next: []const u32, hidden: u64) !void {
        if (next.len == 0) return;
        var p = e.pe;
        try h.stage(e, next, hidden, p.b.ids, p.b.rows);
        const rows = next.len;
        const w = e.prepared.weights.mtp.?;
        try h.inputs(e, p.b.ids, rows);
        p.cache = try h.cache.view(0, h.cache.rows, 0);
        p.pos_dev = e.bound.?.st.fixed.mtp_pos_dev;
        try p.putI32(p.pos_dev, @intCast(h.cursor.len));
        const blocks = (rows + 511) / 512;
        for (0..blocks) |i| try p.putI32(p.s.origins + i * 16, @intCast(h.cursor.len + i * 512));
        try p.mat(h.cat, 8192, w.eh, h.mx, rows, false);
        try p.norm(h.mx, 4096, w.layer.attn.norm, p.b.normed, 4096, rows);
        try p.dsaLayer(w.layer.mixer.dsa, 0, rows, h.cursor.len, null);
        try h.post(e, p.b.part, p.b.gathered, rows);
        try p.norm(h.mx, 4096, w.layer.ffn.norm, p.b.normed, 4096, rows);
        try p.ffn(w.layer, rows, null);
        try h.post(e, p.b.part, p.b.gathered, rows);
        h.cursor.advance(rows, false);
    }
    pub fn step(h: *Head, e: anytype, next: []const u32, hidden: u64, speculative: bool) !u64 {
        try h.stage(e, next, hidden, h.b.ids, h.b.rows);
        const rows = next.len;
        const w = e.prepared.weights.mtp.?;
        try h.inputs(e, h.b.ids, rows);
        _ = try h.tables.fill(&.{.{ .base = 0, .pos = h.cursor.len, .rows = rows, .ring_base = 0 }}, 2051);
        try h.tables.stage();
        var f = e.fwd;
        f.b = h.b;
        f.tables = &h.tables;
        f.caches = &h.cache;
        try e.stream.d.check(e.stream.d.api.cuMemsetD32Async(e.bound.?.st.fixed.mtp_pos_dev, @intCast(h.cursor.len), 1, e.stream.handle), "MTP origin");
        try f.mat(h.cat, 8192, try f.sums(h.cat, 8192, 8192, rows), w.eh, h.mx, rows, false);
        try f.norm(h.mx, 4096, w.layer.attn.norm, f.b.normed, 4096, f.b.xs, rows);
        try f.dsaMtp(w.layer.mixer.dsa, 0, 0, rows, e.bound.?.st.fixed.mtp_pos_dev);
        try h.post(e, f.b.part, f.b.gathered, rows);
        try f.norm(h.mx, 4096, w.layer.ffn.norm, f.b.normed, 4096, f.b.xs, rows);
        try f.ffn(w.layer, rows);
        try h.post(e, f.b.part, f.b.gathered, rows);
        try f.norm(h.mx + (rows - 1) * 8192, 4096, w.norm, f.b.normed, 4096, f.b.xs, 1);
        try f.mat(f.b.normed, 4096, f.b.xs, w.head, f.b.logits, 1, false);
        h.cursor.advance(rows, speculative);
        return f.b.logits;
    }
};

/// Adapter for the existing P7 E.DraftReq contract. Absorb committed target rows, then chain the normalized MTP row.
pub fn draftMany(h: anytype, e: anytype, reqs: anytype) !void {
    if (reqs.len != 1) return error.MtpRequiresSingleStream;
    const r = &reqs[0];
    _ = try e.draftSlot(r.seq);
    e.bound = r.seq;
    if (r.next.len == 0 or r.next.len > 64 or r.count > r.out.len or !std.math.isFinite(r.confidence)) return error.BadMtpRequest;
    const hidden = switch (r.source) {
        .prefill_last => blk: {
            if (r.next.len != 1) return error.BadMtpRequest;
            break :blk h.pending;
        },
        .window => |i| blk: {
            if (i >= e.round_n or !e.committed[i] or r.next.len > e.round_len[i]) return error.BadMtpRequest;
            break :blk e.fwd.b.normed + e.round_row0[i] * 8192;
        },
    };
    try h.cursor.rollback();
    var logits = try h.step(e, r.next, hidden, false);
    if (h.copies) |*c| {
        try c.extend(r.next);
        const proposed = c.propose(r.out[0..r.count]);
        if (proposed.len != 0) {
            r.drafted = proposed.len;
            return;
        }
    }
    const confidence = @abs(r.confidence);
    var chain: f64 = 1;
    r.drafted = 0;
    for (0..r.count) |j| {
        var token: [1]u32 = undefined;
        var prob: [1]f64 = undefined;
        const position = [_]u64{@intCast(r.seq.st.pos + 1 + j)};
        try e.sampler.samplePacked(&.{.{ .logits = logits, .positions = &position, .sampling = r.sampling, .out = &token, .probs = if (confidence > 0) &prob else null }});
        if (confidence > 0 and j > 0 and chain * prob[0] < confidence) break;
        r.out[j] = token[0];
        r.drafted += 1;
        if (confidence > 0) {
            chain *= prob[0];
            if (chain < confidence) break;
        }
        if (j + 1 < r.count) logits = try h.step(e, &token, h.b.normed, true);
    }
}
test "MTP rollback retains absorbed rows and replaces every chained row after rejection" {
    var c: Cursor = .{ .len = 5 };
    c.advance(2, false);
    c.advance(3, true);
    try c.rollback();
    try std.testing.expectEqual(@as(usize, 7), c.len);
    try std.testing.expectEqual(@as(usize, 0), c.drafted);
    c.advance(1, false);
    try c.rollback();
    try std.testing.expectEqual(@as(usize, 8), c.len);
    c.drafted = 9;
    try std.testing.expectError(error.BadMtpCursor, c.rollback());
}
test "MTP input and residual recorder preserve concat stride, BF16 intermediate, rank-order ABI" {
    var rec: @import("cuda_triton.zig").Recorder = .{};
    const t: Tri = .{ .rec = &rec };
    var w: fw.MtpW = undefined;
    w.enorm = 0x3000;
    w.hnorm = 0x4000;
    try inputNorm(t, 0x1000, 0x2000, 0x5000, w, 2);
    try residual(t, 0x9000, 0xa000, 2, 2);
    try std.testing.expectEqual(@as(usize, 3), rec.n);
    try std.testing.expectEqual(@as(i32, 8192), rec.launches[0].args[4].value.i32);
    try std.testing.expectEqual(@as(u64, 0x7000), rec.launches[1].args[3].value.ptr.addr);
    try std.testing.expectEqualStrings("*bf16", rec.launches[0].args[5].value.ptr.ty);
    try std.testing.expectEqualStrings("_residual_add", rec.launches[2].name);
    try std.testing.expectEqual(@as(i32, 8192), rec.launches[2].args[3].value.i32);
}

test "fake MTP forward chains its normalized output, rolls back rejected rows, and P7 emits serial tokens" {
    const rounds = @import("cuda_rounds.zig");
    const FakeHead = struct {
        cursor: Cursor = .{},
        pending: u64 = 100,
        b: struct { normed: u64 = 200 } = .{},
        copies: ?copy_mod.Copy = null,
        steps: usize = 0,
        rollbacks_seen: usize = 0,
        fn step(h: *@This(), _: anytype, next: []const u32, hidden: u64, speculative: bool) !u64 {
            if (speculative) try std.testing.expectEqual(@as(u64, 200), hidden) else {
                try std.testing.expectEqual(@as(usize, 0), h.cursor.drafted);
                h.rollbacks_seen += 1;
            }
            h.cursor.advance(next.len, speculative);
            h.steps += 1;
            return if (speculative) 999 else next[next.len - 1] + 1;
        }
    };
    const FakeSampler = struct {
        fn samplePacked(_: @This(), parts: []const @import("cuda_sampler.zig").Part) !void {
            for (parts) |p| {
                p.out[0] = @intCast(p.logits);
                if (p.probs) |q| q[0] = 0.5;
            }
        }
    };
    const E = struct {
        const Self = @This();
        const Seq = struct { st: state.State, slot: usize = 0 };
        pub const DraftSource = union(enum) { prefill_last, window: usize };
        pub const DraftReq = struct { seq: *Seq, source: DraftSource, next: []const u32, count: usize, out: []u32, sampling: ?@import("lanes").Sampling = null, confidence: f64 = 0, drafted: usize = 0 };
        head: FakeHead = .{},
        sampler: FakeSampler = .{},
        bound: ?*Seq = null,
        round_n: usize = 1,
        committed: [1]bool = .{true},
        round_len: [1]usize = .{16},
        round_row0: [1]usize = .{0},
        fwd: struct { b: struct { normed: u64 = 300 } = .{} } = .{},
        pos: usize = 0,
        replays: usize = 0,
        pub fn draftSlot(_: *Self, s: *Seq) !usize {
            return s.slot;
        }
        pub fn draftMany(e: *Self, reqs: []DraftReq) !void {
            try @import("cuda_mtp.zig").draftMany(&e.head, e, reqs);
        }
        pub fn forward(e: *Self, windows: []const rounds.Window, _: []const i32) !void {
            e.pos = windows[0].state.pos;
        }
        pub fn sample(e: *Self, _: []const rounds.Window, out: []u32) !void {
            for (out, 0..) |*t, i| t.* = @intCast(e.pos + i + 2);
        }
        pub fn stageCommit(_: *Self, _: []const i32) !void {}
        pub fn replay(e: *Self) !void {
            e.replays += 1;
        }
        pub fn convShift(_: *Self) !void {}
        pub fn setPosition(_: *Self, _: *state.State, _: usize) !void {}
        pub fn commitDrafts(_: *Self, _: []const rounds.Window, _: []const usize) !void {}
    };
    for ([_]usize{ 0, 1, 2, 7, 20 }) |limit| {
        var a: E.Seq = .{ .st = .{ .slot = 0, .base = 0, .capacity = 2048, .fixed = undefined, .cache = undefined } };
        var b = a;
        var serial: E = .{};
        var drafted: E = .{};
        const generate = @import("cuda_generate.zig");
        const x = try generate.run(E, &serial, std.testing.allocator, &a, 1, limit, &.{8}, false);
        defer std.testing.allocator.free(x);
        const y = try generate.run(E, &drafted, std.testing.allocator, &b, 1, limit, &.{8}, true);
        defer std.testing.allocator.free(y);
        try std.testing.expectEqualSlices(u32, x, y);
        try std.testing.expectEqual(a.st.pos, b.st.pos);
        if (limit == 20) {
            try std.testing.expect(drafted.replays > 0);
            try std.testing.expect(drafted.head.rollbacks_seen > 1);
        }
    }
    var e: E = .{};
    var seq: E.Seq = .{ .st = .{ .slot = 0, .base = 0, .capacity = 2048, .fixed = undefined, .cache = undefined } };
    var proposals: [3]u32 = undefined;
    var req = [_]E.DraftReq{.{ .seq = &seq, .source = .prefill_last, .next = &.{1}, .count = 3, .confidence = 0.3, .out = &proposals }};
    try e.draftMany(&req);
    try std.testing.expectEqual(@as(usize, 1), req[0].drafted);
    try std.testing.expectEqual(@as(usize, 1), e.head.cursor.drafted);
    req[0].confidence = 0.9;
    try e.draftMany(&req);
    try std.testing.expectEqual(@as(usize, 1), req[0].drafted);
    try std.testing.expectEqual(@as(usize, 0), e.head.cursor.drafted);
    e = .{};
    e.head.copies = try copy_mod.Copy.init(std.testing.allocator, &.{ 1, 2, 3, 4, 1 }, .{ .match = 2, .most = 2, .reply_match = 0, .pad = &.{} });
    defer e.head.copies.?.deinit();
    req[0].next = &.{2};
    req[0].count = 2;
    try e.draftMany(&req);
    try std.testing.expectEqualSlices(u32, &.{ 3, 4 }, proposals[0..req[0].drafted]);
    try std.testing.expectEqual(@as(usize, 1), e.head.cursor.len);
    try std.testing.expectEqual(@as(usize, 0), e.head.cursor.drafted);
}
