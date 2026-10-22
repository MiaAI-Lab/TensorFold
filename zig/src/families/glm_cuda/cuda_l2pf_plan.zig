//! The L2 prefetch sites' tables for GLM's decode weights: glm5_next/cuda/l2pf.py Prefetch.__init__ and its _ffn / _next /
//! _out plans, over this engine's weight records (cuda_forward.zig Weights). Sizes of the loaded tensors come from the
//! device allocations the weight loader made (one allocation a tensor).
const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("cuda_forward.zig");
const mm = @import("cuda_matmuls.zig");
const hc = @import("cuda_hc.zig");
const l2 = @import("cuda_l2pf.zig");

const Sizes = std.AutoHashMapUnmanaged(u64, u64);

const Ctx = struct {
    sizes: *const Sizes,
    arena: std.mem.Allocator,
    fn one(c: Ctx, ptr: u64) ?l2.Tensor {
        const n = c.sizes.get(ptr) orelse return null;
        return .{ .ptr = ptr, .bytes = n };
    }
    fn plain(c: Ctx, list: []const u64) !l2.Group {
        var ts: std.ArrayList(l2.Tensor) = .empty;
        for (list) |p| if (c.one(p)) |t| try ts.append(c.arena, t);
        return .{ .tensors = ts.items };
    }
    fn glue(c: Ctx, w: hc.Weights) !l2.Group {
        return c.plain(&.{ w.fn_w, w.base, w.scale, w.norm });
    }
    fn q4(c: Ctx, q: mm.Q4) !l2.Group {
        const w = c.one(q.w) orelse return .{ .tensors = &.{} };
        var ts: std.ArrayList(l2.Tensor) = .empty;
        try ts.append(c.arena, w);
        for ([_]u64{ q.s, q.b }) |p| if (c.one(p)) |t| try ts.append(c.arena, t);
        return .{ .tensors = ts.items, .weight = 0, .q4 = .{ .n = q.n, .k = q.k, .split_k = mm.splitK(q.n, q.k) } };
    }
};

/// What site a reads: the FFN's glue weights, then the router and shared expert (a dense MLP's gate/up and down).
fn ffnGroups(c: Ctx, out: *std.ArrayList(l2.Group), layer: fwd.Layer) !void {
    try out.append(c.arena, try c.glue(layer.ffn));
    switch (layer.mlp) {
        .dense => |w| {
            try out.append(c.arena, try c.q4(w.gu));
            try out.append(c.arena, try c.q4(w.down));
        },
        .moe => |w| {
            try out.append(c.arena, try c.plain(&.{w.router}));
            try out.append(c.arena, try c.plain(&.{w.bias}));
            try out.append(c.arena, try c.q4(w.shared.gu));
            try out.append(c.arena, try c.q4(w.shared.down));
        },
    }
}

/// Site f: the next layer's glue weights and input projections; after the last layer the final norm and the head.
fn nextGroups(c: Ctx, out: *std.ArrayList(l2.Group), w: *const fwd.Weights, next: ?fwd.Layer) !void {
    const n = next orelse {
        try out.append(c.arena, try c.plain(&.{w.norm}));
        try out.append(c.arena, try c.plain(&.{ w.head_scale, w.head }));
        return;
    };
    try out.append(c.arena, try c.glue(n.attn));
    switch (n.mixer) {
        .kda => |k| {
            try out.append(c.arena, try c.q4(k.fb));
            try out.append(c.arena, try c.q4(k.gb));
            try out.append(c.arena, try c.plain(&.{ k.conv, k.a_log, k.dt_bias }));
            try out.append(c.arena, try c.q4(k.proj));
        },
        .dsa => |a| {
            try out.append(c.arena, try c.plain(&.{ a.q_norm, a.kv_norm }));
            try out.append(c.arena, try c.q4(a.proj));
            try out.append(c.arena, try c.q4(a.qb));
            try out.append(c.arena, try c.q4(a.kw));
            try out.append(c.arena, try c.plain(&.{a.gate}));
        },
    }
}

/// Site o: a KDA layer's output projection; a DSA layer's value half of kv_b (what expand_v reads), then its output
/// projection.
fn outGroups(c: Ctx, out: *std.ArrayList(l2.Group), layer: fwd.Layer) !void {
    switch (layer.mixer) {
        .kda => |k| try out.append(c.arena, try c.q4(k.o)),
        .dsa => |a| {
            try out.append(c.arena, try c.plain(&.{a.wv}));
            try out.append(c.arena, try c.q4(a.o));
        },
    }
}

/// null when TF_GLM_L2PF is off. `buffers` are the weight loader's device allocations (pointer and length).
pub fn build(gpa: std.mem.Allocator, d: *const cuda.Driver, f: cuda.Function, w: *const fwd.Weights, buffers: []const cuda.DeviceBuffer) !?*l2.Prefetch {
    const s = l2.Settings.fromEnv() catch |e| {
        std.log.warn("TF_GLM_L2PF settings refused ({t}); L2 prefetch off", .{e});
        return null;
    };
    if (s.mode == null) return null;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var sizes: Sizes = .empty;
    try sizes.ensureTotalCapacity(arena, @intCast(buffers.len));
    for (buffers) |b| sizes.putAssumeCapacity(b.ptr, b.len);
    const c: Ctx = .{ .sizes = &sizes, .arena = arena };
    var b = l2.Builder.init(gpa, s);
    defer b.deinit();
    for (w.layers, 0..) |layer, i| {
        var g: std.ArrayList(l2.Group) = .empty;
        try ffnGroups(c, &g, layer);
        try b.add(i, .a, g.items);
        g.clearRetainingCapacity();
        try nextGroups(c, &g, w, if (i + 1 < w.layers.len) w.layers[i + 1] else null);
        try b.add(i, .f, g.items);
        g.clearRetainingCapacity();
        try outGroups(c, &g, layer);
        try b.add(i, .o, g.items);
    }
    const p = try gpa.create(l2.Prefetch);
    errdefer gpa.destroy(p);
    p.* = try l2.Prefetch.init(d, f, &b);
    var per = [3]struct { n: usize = 0, bytes: u64 = 0 }{ .{}, .{}, .{} };
    for (b.sites) |row| for (row, 0..) |site, k| if (site) |x| {
        per[k].n += 1;
        per[k].bytes += x.bytes;
    };
    const names = [_][]const u8{ "bulk", "lines", "touch" };
    var mib: [3]f64 = undefined;
    for (&mib, per) |*m, x| m.* = @as(f64, @floatFromInt(x.bytes)) / @as(f64, @floatFromInt(@max(1, x.n))) / (1 << 20);
    std.log.info("L2 prefetch in decode windows ({s}, up to {d:.1} MiB a site): a {d} sites {d:.1} MiB, f {d} sites {d:.1} MiB, o {d} sites {d:.1} MiB", .{ names[s.mode.?], s.mb, per[0].n, mib[0], per[1].n, mib[1], per[2].n, mib[2] });
    return p;
}

test "the plan builder compiles against the weight records" {
    _ = build;
}
