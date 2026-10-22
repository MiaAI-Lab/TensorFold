//! TensorFold glm5_next/cuda/dflash2_multi.py:_block_compute/_taps_compute/_layer.
//! Ash Hart and TensorFold contributors; jschmied CUDA graph/launch patterns.
//! Target embedding is read-only; all writable buffers and KV belong to the drafter.
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("cuda_matmuls.zig");
const weights = @import("cuda_dflash_weights.zig");
const config = @import("cuda_dflash_config.zig");
const Tri = @import("cuda_triton.zig").Tri;
const Torch = @import("cuda_torch_ops.zig").Torch;
const Coll = @import("flashnext").mirror.Coll;
const base = @import("cuda_torch_ops.zig").shared;
const stage_mod = @import("cuda_stage.zig");
const Stager = stage_mod.Stager;
/// Pinned readback layout: the tap graph's invalid flag, the block graph's, the candidates, the projection.
const pin_tap = 0;
const pin_block = 16;
const pin_cand = 64;
const pin_proj = pin_cand + 4 * 8 * 7 * 32 * 4;
const pin_bytes = pin_proj + 8 * 7 * 256 * 4;
pub const Forward = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    cfg: config.Config,
    rank: config.Rank,
    streams: usize,
    w: weights.Weights,
    embed: u64,
    tri: Tri,
    th: Torch,
    entries: *const mm.Entries,
    comm: ?Coll,
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
    selector: cuda.cublaslt.Linear,
    selector_ws: u64,
    selector_ws_len: usize,
    inv: u64,
    ids: u64,
    rowpos: u64,
    rowslots: u64,
    indices: u64,
    bpos: u64,
    bslots: u64,
    x: u64,
    normed: u64,
    dyn: u64,
    conv: u64,
    prod: u64,
    q: u64,
    k: u64,
    v: u64,
    attn: u64,
    gu: u64,
    act: u64,
    sums: u64,
    part: u64,
    rowpart: u64,
    gathered: u64,
    rowout: u64,
    tap_in: u64,
    cos: u64,
    sin: u64,
    h: u64,
    hn: u64,
    logits: u64,
    logitsf: u64,
    vals: u64,
    local: u64,
    top_scratch: u64,
    pack_buf: u64,
    candidates: u64,
    projbf: u64,
    projection: u64,
    invalid: u64,
    kc: [5]u64,
    vc: [5]u64,
    blocks: [8]?cuda.graph.Exec = @splat(null),
    taps: [8]?cuda.graph.Exec = @splat(null),
    /// Stream-ordered pinned staging and readback (unset with TF_GLM_SYNC_STAGE=1: the blocking calls).
    stager: Stager = .{},
    pin: ?cuda.HostBuffer = null,
    tap_pending: bool = false,
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, cfg: config.Config, rank: config.Rank, streams: usize, w: weights.Weights, embed: u64, tri: Tri, th: Torch, entries: *const mm.Entries, comm: ?Coll, lt: *const cuda.cublaslt.Library) !Forward {
        if (streams < 1 or streams > 8 or (rank.world > 1 and comm == null)) return error.BadDraftStreams;
        if (comm) |c| if (c.world != rank.world or c.rank != rank.rank) return error.BadDraftRank;
        var f: Forward = undefined;
        f.gpa = gpa;
        f.d = d;
        f.stream = stream;
        f.cfg = cfg;
        f.rank = rank;
        f.streams = streams;
        f.w = w;
        f.embed = embed;
        f.tri = tri;
        f.th = th;
        f.entries = entries;
        f.comm = comm;
        f.owned = .empty;
        f.blocks = @splat(null);
        f.taps = @splat(null);
        f.stager = .{};
        f.pin = null;
        f.tap_pending = false;
        errdefer {
            for (f.owned.items) |*b| b.free();
            f.owned.deinit(gpa);
        }
        const r: usize = 64;
        const c = streams * config.Config.ring + 64;
        var inv: [64]f32 = undefined;
        for (&inv, 0..) |*v, i| v.* = 1 / std.math.pow(f32, cfg.theta, @as(f32, @floatFromInt(2 * i)) / 128);
        f.inv = try f.upload(std.mem.sliceAsBytes(&inv));
        inline for (.{ "ids", "rowpos", "rowslots", "indices", "bpos", "bslots" }, .{ r * 4, r * 8, r * 8, r * 8, 8 * 8, 8 * 8 }) |field, n| @field(f, field) = try f.alloc(n);
        inline for (.{ "x", "normed", "conv", "rowout", "h", "hn" }) |field| @field(f, field) = try f.alloc(r * 4096 * 2);
        f.dyn = try f.alloc(r * 1024 * 2);
        f.q = try f.alloc(r * rank.heads * 128 * 2);
        f.k = try f.alloc(r * rank.kv * 128 * 2);
        f.v = try f.alloc(r * rank.kv * 128 * 2);
        f.attn = try f.alloc(r * rank.heads * 128 * 2);
        f.gu = try f.alloc(r * 2 * rank.inter * 2);
        f.act = try f.alloc(r * rank.inter * 2);
        // qkv width may exceed D (TP=1: 6144).
        f.prod = try f.alloc(r * @max(4096, (rank.heads + 2 * rank.kv) * 128) * 2);
        f.sums = try f.alloc(r * 20480 / 64 * 4);
        var partial: usize = 0;
        for (w.layers) |l| for ([_]mm.Q4{ l.a_kp, l.m_kp, l.qkv, l.kv, l.o, l.gu, l.down }) |q| {
            if (mm.splitK(q.n, q.k) > 1) partial = @max(partial, mm.splitK(q.n, q.k) * q.n);
        };
        if (mm.splitK(w.fc.n, w.fc.k) > 1) partial = @max(partial, mm.splitK(w.fc.n, w.fc.k) * w.fc.n);
        if (mm.splitK(w.head.n, w.head.k) > 1) partial = @max(partial, mm.splitK(w.head.n, w.head.k) * w.head.n);
        f.part = try f.alloc(r * partial * 4);
        f.rowpart = try f.alloc(r * 4096 * 4);
        f.gathered = try f.alloc(rank.world * r * 4096 * 4);
        f.tap_in = try f.alloc(r * 5 * 4096 * 2);
        f.cos = try f.alloc(r * 64 * 4);
        f.sin = try f.alloc(r * 64 * 4);
        f.logits = try f.alloc(r * rank.vocab * 2);
        f.logitsf = try f.alloc(r * rank.vocab * 4);
        f.vals = try f.alloc(r * 16 * 4);
        f.local = try f.alloc(r * 16 * 8);
        f.top_scratch = try f.alloc(base.topkScratchBytes(7, rank.vocab));
        f.pack_buf = try f.alloc(r * 32 * 4);
        f.candidates = try f.alloc(rank.world * r * 32 * 4);
        f.projbf = try f.alloc(r * 256 * 2);
        f.projection = try f.alloc(r * 256 * 4);
        f.invalid = try f.alloc(4);
        for (0..5) |i| {
            f.kc[i] = try f.alloc(rank.kv * c * 128 * 2);
            f.vc[i] = try f.alloc(rank.kv * c * 128 * 2);
        }
        f.selector = try cuda.cublaslt.Linear.init(lt, 7, 256, 4096, .bf16, 4 << 20);
        errdefer f.selector.deinit();
        try deterministicSelector(&f.selector);
        f.selector_ws_len = f.selector.workspace_need;
        f.selector_ws = try f.alloc(f.selector_ws_len);
        if (std.c.getenv("TF_GLM_SYNC_STAGE") == null) {
            f.stager = try Stager.init(d, stream, 16 << 10);
            errdefer f.stager.deinit();
            f.pin = try cuda.HostBuffer.alloc(d, pin_bytes);
        }
        return f;
    }
    fn alloc(f: *Forward, n: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(f.d, n);
        errdefer b.free();
        try b.fill8(0, null);
        try f.owned.append(f.gpa, b);
        return b.ptr;
    }
    fn upload(f: *Forward, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.fromHost(f.d, bytes);
        errdefer b.free();
        try f.owned.append(f.gpa, b);
        return b.ptr;
    }
    pub fn stage(f: *Forward, ptr: u64, bytes: []const u8) !void {
        if (f.stager.ready()) return f.stager.put(ptr, bytes); // queued behind whatever the stream holds: no drain
        try f.stream.synchronize(); // pageable staging outside capture; no lifetime race with replay
        try (cuda.DeviceBuffer{ .d = f.d, .ptr = ptr, .len = bytes.len }).upload(0, bytes);
    }
    pub fn deinit(f: *Forward) void {
        f.stager.deinit();
        if (f.pin) |*h| h.free();
        for (&f.blocks) |*g| if (g.*) |*exec| exec.deinit();
        for (&f.taps) |*g| if (g.*) |*exec| exec.deinit();
        f.selector.deinit();
        for (f.owned.items) |*b| b.free();
        f.owned.deinit(f.gpa);
    }
    fn mat(f: *Forward, x: u64, q: mm.Q4, out: u64, rows: usize, sums: ?u64, f32_out: bool) !void {
        const xs = sums orelse f.sums;
        if (sums == null) try f.tri.groupSums(x, xs, rows, q.k);
        try mm.matmul(f.entries, f.stream, x, q.k, xs, q, out, f.part, rows, f32_out, false, false, .{});
    }
    fn norm(f: *Forward, x: u64, w: u64, out: u64, rows: usize) !void {
        try f.tri.draftRmsnorm(x, 4096, w, out, f.sums, rows, 4096, f.cfg.eps);
    }
    fn row(f: *Forward, x: u64, q: mm.Q4, rows: usize, sums: ?u64) !void {
        try f.mat(x, q, f.rowpart, rows, sums, true);
        if (f.comm) |c| try c.allGather(f.rowpart, f.gathered, rows * 4096, .f32, f.stream.handle) else try f.th.base.copy(f.gathered, f.rowpart, rows * 4096 * 4);
        try f.th.rankAdd(f.gathered, f.rowout, rows * 4096, f.rank.world);
    }
    fn prep(f: *Forward, l: weights.Layer, rows: usize, heads: usize) !void {
        try f.tri.draftPrep(f.prod, l.q_norm, l.k_norm, f.cos, f.sin, f.q, f.k, f.v, rows, heads, f.rank.kv, f.cfg.eps);
    }
    fn cache(f: *Forward, i: usize, rows: usize) !void {
        const cap = f.streams * config.Config.ring + 64;
        try f.th.indexCopy(f.k, f.kc[i], f.indices, rows, cap, 256, f.rank.kv, f.invalid);
        try f.th.indexCopy(f.v, f.vc[i], f.indices, rows, cap, 256, f.rank.kv, f.invalid);
    }
    pub fn blockCompute(f: *Forward, streams: usize) !void {
        if (streams < 1 or streams > f.streams) return error.BadDraftStreams;
        const r = streams * 8;
        try f.th.base.fill32(f.invalid, 0, 1);
        try f.th.rotary(f.rowpos, f.inv, f.cos, f.sin, r, 64);
        try f.th.ring(f.rowpos, f.rowslots, f.indices, r, config.Config.ring);
        try f.tri.embed(f.ids, f.embed, f.x, r, 1);
        for (f.w.layers, 0..) |l, i| {
            try f.norm(f.x, l.in_norm, f.normed, r);
            try f.mat(f.normed, l.a_kp, f.dyn, r, f.sums, false);
            try f.tri.draftConv(f.normed, f.dyn, l.a_base, null, f.conv, r, 0, 8);
            try f.mat(f.conv, l.qkv, f.prod, r, null, false);
            try f.prep(l, r, f.rank.heads);
            try f.cache(i, r);
            try f.tri.draftAttention(f.q, f.kc[i], f.vc[i], f.attn, f.bpos, f.bslots, streams, f.rank.heads, f.rank.kv, f.streams * config.Config.ring + 64);
            try f.row(f.attn, l.o, r, null);
            try f.tri.draftConv(f.rowout, f.dyn, l.a_base, f.x, f.conv, r, 1, 8);
            try f.th.base.copy(f.x, f.conv, r * 4096 * 2);
            try f.norm(f.x, l.post_norm, f.normed, r);
            try f.mat(f.normed, l.m_kp, f.dyn, r, f.sums, false);
            try f.tri.draftConv(f.normed, f.dyn, l.m_base, null, f.conv, r, 0, 8);
            try f.mat(f.conv, l.gu, f.gu, r, null, false);
            try f.tri.draftSwiglu(f.gu, f.act, f.sums, r, f.rank.inter);
            try f.row(f.act, l.down, r, f.sums);
            try f.tri.draftConv(f.rowout, f.dyn, l.m_base, f.x, f.conv, r, 1, 8);
            try f.th.base.copy(f.x, f.conv, r * 4096 * 2);
        }
        for (0..streams) |s| try f.th.base.copy(f.h + s * 7 * 8192, f.x + (s * 8 + 1) * 8192, 7 * 8192);
        const rows = streams * 7;
        try f.norm(f.h, f.w.norm, f.hn, rows);
        try f.mat(f.hn, f.w.head, f.logits, rows, f.sums, false);
        try f.th.base.toF32(f.logits, f.logitsf, rows * f.rank.vocab);
        for (0..streams) |s| {
            try f.th.topkSorted(f.logitsf + s * 7 * f.rank.vocab * 4, f.rank.vocab, 7, 16, f.vals, f.local, f.top_scratch);
            try f.th.pack(f.vals, f.local, f.pack_buf + s * 7 * 32 * 4, 7, 16, @intCast(f.rank.vocab_start));
            try f.selector.run(f.hn + s * 7 * 8192, f.w.hproj, f.projbf + s * 7 * 256 * 2, f.selector_ws, f.selector_ws_len, f.stream.handle);
        }
        try f.th.base.toF32(f.projbf, f.projection, rows * 256);
        if (f.comm) |c| try c.allGather(f.pack_buf, f.candidates, rows * 32, .f32, f.stream.handle) else try f.th.base.copy(f.candidates, f.pack_buf, rows * 32 * 4);
    }
    pub fn tapsCompute(f: *Forward, rows: usize) !void {
        if (rows < 8 or rows > 64 or rows % 8 != 0) return error.BadTapRows;
        try f.th.base.fill32(f.invalid, 0, 1);
        try f.mat(f.tap_in, f.w.fc, f.prod, rows, null, false);
        try f.norm(f.prod, f.w.hidden_norm, f.normed, rows);
        try f.th.rotary(f.rowpos, f.inv, f.cos, f.sin, rows, 64);
        try f.th.ring(f.rowpos, f.rowslots, f.indices, rows, config.Config.ring);
        for (f.w.layers, 0..) |l, i| {
            try f.mat(f.normed, l.kv, f.prod, rows, f.sums, false);
            // mat's group sums are still the context sums used by every KV projection.
            try f.prep(l, rows, 0);
            try f.cache(i, rows);
        }
    }
    /// Warm-up resolves Tri memo specializations and cuBLAS handles before capture.
    /// All capture destinations use the trash slot, leaving real stream contexts intact.
    pub fn capture(f: *Forward) !void {
        var ids: [64]u32 = @splat(f.cfg.mask);
        var pos: [64]i64 = undefined;
        var slots: [64]i64 = @splat(@intCast(f.streams));
        for (&pos, 0..) |*p, i| p.* = @intCast(i % 8);
        try f.stage(f.ids, std.mem.sliceAsBytes(&ids));
        try f.stage(f.rowpos, std.mem.sliceAsBytes(&pos));
        try f.stage(f.rowslots, std.mem.sliceAsBytes(&slots));
        const ends: [8]i64 = @splat(0);
        const bs: [8]i64 = @splat(@intCast(f.streams));
        try f.stage(f.bpos, std.mem.sliceAsBytes(&ends));
        try f.stage(f.bslots, std.mem.sliceAsBytes(&bs));
        for (0..8) |i| {
            try f.tapsCompute((i + 1) * 8);
            try f.stream.synchronize();
            if (f.taps[i] != null) continue;
            try cuda.graph.beginCapture(f.stream, .global);
            f.tapsCompute((i + 1) * 8) catch |err| {
                var aborted = cuda.graph.endCapture(f.stream) catch return err;
                aborted.deinit();
                return err;
            };
            var g = try cuda.graph.endCapture(f.stream);
            defer g.deinit();
            f.taps[i] = try g.instantiate();
        }
        for (0..f.streams) |i| {
            try f.blockCompute(i + 1);
            try f.stream.synchronize();
            if (f.blocks[i] != null) continue;
            try cuda.graph.beginCapture(f.stream, .global);
            f.blockCompute(i + 1) catch |err| {
                var aborted = cuda.graph.endCapture(f.stream) catch return err;
                aborted.deinit();
                return err;
            };
            var g = try cuda.graph.endCapture(f.stream);
            defer g.deinit();
            f.blocks[i] = try g.instantiate();
        }
    }
    pub fn block(f: *Forward, n: usize) !void {
        if (n < 1 or n > f.streams) return error.BadDraftStreams;
        if (f.blocks[n - 1]) |g| try g.launchOn(f.stream) else try f.blockCompute(n);
    }
    pub fn tapsPass(f: *Forward, n: usize) !void {
        if (n < 8 or n > 64 or n % 8 != 0) return error.BadTapRows;
        if (f.taps[n / 8 - 1]) |g| try g.launchOn(f.stream) else try f.tapsCompute(n);
    }
    pub fn read(f: *Forward, ptr: u64, bytes: []u8) !void {
        try f.stream.synchronize();
        try (cuda.DeviceBuffer{ .d = f.d, .ptr = ptr, .len = bytes.len }).download(0, bytes);
    }
    /// Pinned readback and staging in use (not the blocking calls).
    pub fn asyncOk(f: *const Forward) bool {
        return f.pin != null and !stage_mod.sync_mode;
    }
    fn pinned(f: *Forward, at: usize, n: usize) []u8 {
        return f.pin.?.bytes[at..][0..n];
    }
    /// After a tap graph: queue the invalid flag's copy and carry on; `settle` reads it once the stream has been
    /// synchronized (the next proposal's readback). Without pinned staging this is the blocking `check`.
    pub fn tapsDone(f: *Forward) !void {
        if (!f.asyncOk()) return f.check();
        const view: cuda.DeviceBuffer = .{ .d = f.d, .ptr = f.invalid, .len = 4 };
        try view.downloadAsync(0, f.pinned(pin_tap, 4), f.stream.handle);
        f.tap_pending = true;
    }
    /// After block(n): queue the invalid flag, this rank's candidates and the projection for one synchronization.
    pub fn queueProposal(f: *Forward, n: usize) !void {
        const copies = [_]struct { ptr: u64, at: usize, bytes: usize }{
            .{ .ptr = f.invalid, .at = pin_block, .bytes = 4 },
            .{ .ptr = f.candidates, .at = pin_cand, .bytes = f.rank.world * n * 7 * 32 * 4 },
            .{ .ptr = f.projection, .at = pin_proj, .bytes = n * 7 * 256 * 4 },
        };
        for (copies) |c| {
            const view: cuda.DeviceBuffer = .{ .d = f.d, .ptr = c.ptr, .len = c.bytes };
            try view.downloadAsync(0, f.pinned(c.at, c.bytes), f.stream.handle);
        }
    }
    /// The stream has been synchronized since queueProposal / tapsDone: check the flags they queued.
    pub fn settle(f: *Forward, block_too: bool) !void {
        if (f.tap_pending) {
            f.tap_pending = false;
            if (std.mem.readInt(u32, f.pinned(pin_tap, 4)[0..4], .little) != 0) return error.BadDraftIndex;
        }
        if (block_too and std.mem.readInt(u32, f.pinned(pin_block, 4)[0..4], .little) != 0) return error.BadDraftIndex;
    }
    pub fn candidatesHost(f: *Forward, n: usize) []const f32 {
        return @alignCast(std.mem.bytesAsSlice(f32, f.pinned(pin_cand, f.rank.world * n * 7 * 32 * 4)));
    }
    pub fn projectionHost(f: *Forward, n: usize) []const f32 {
        return @alignCast(std.mem.bytesAsSlice(f32, f.pinned(pin_proj, n * 7 * 256 * 4)));
    }
    pub fn check(f: *Forward) !void {
        var invalid: [4]u8 = undefined;
        try f.read(f.invalid, &invalid);
        if (std.mem.readInt(u32, &invalid, .little) != 0) return error.BadDraftIndex;
    }
};

/// CUDA's installed cublasLt.h: ALGO_CONFIG_SPLITK_NUM=2, REDUCTION_SCHEME=3.
/// Pick a fixed nonsplit algorithm; parallel reduction schemes may change proposal bits.
fn deterministicSelector(linear: *cuda.cublaslt.Linear) !void {
    const Get = *const fn (*const cuda.cublaslt.Algo, c_int, *anyopaque, usize, *usize) callconv(.c) c_int;
    var library = linear.lt.lib;
    const get = library.lookup(Get, "cublasLtMatmulAlgoConfigGetAttribute") orelse return error.MissingSymbol;
    var found: [32]cuda.cublaslt.Heuristic = undefined;
    var count: c_int = 0;
    const api = linear.lt.api;
    try linear.lt.check(api.cublasLtMatmulAlgoGetHeuristic(linear.handle, linear.desc, linear.a, linear.b, linear.c, linear.c, linear.pref, 32, &found, &count), "DFlash selector heuristics");
    for (found[0..@intCast(count)]) |candidate| {
        if (candidate.state != 0) continue;
        var split: i32 = 0;
        var scheme: u32 = 0;
        var written: usize = 0;
        try linear.lt.check(get(&candidate.algo, 2, &split, 4, &written), "DFlash selector split K");
        try linear.lt.check(get(&candidate.algo, 3, &scheme, 4, &written), "DFlash selector reduction scheme");
        if (split <= 1 and scheme == 0) {
            linear.algo = candidate.algo;
            linear.workspace_need = candidate.workspace_size;
            return;
        }
    }
    return error.NoDeterministicSelectorAlgorithm;
}

test "pinned readback regions fit the largest proposal and do not overlap" {
    try std.testing.expect(pin_block + 4 <= pin_cand);
    try std.testing.expect(pin_tap + 4 <= pin_block);
    try std.testing.expectEqual(@as(usize, 4 * 8 * 7 * 32 * 4), pin_proj - pin_cand);
    try std.testing.expectEqual(@as(usize, 8 * 7 * 256 * 4), pin_bytes - pin_proj);
}
