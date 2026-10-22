//! CPU oracles from TensorFold dflash2.py:Drafter.chain and copy_drafts.py.
const std = @import("std");
const policy = @import("cuda_draft_policy.zig");
const Copy = @import("cuda_copy_drafts.zig").Copy;
fn number(v: std.json.Value) f64 {
    return switch (v) {
        .float => v.float,
        .integer => @floatFromInt(v.integer),
        else => unreachable,
    };
}
fn floats(v: std.json.Value, out: []f32) void {
    for (v.array.items, out) |x, *y| y.* = @floatCast(number(x));
}
fn ids(v: std.json.Value, out: []u32) void {
    for (v.array.items, out) |x, *y| y.* = @intFromFloat(number(x));
}
fn get(v: std.json.Value, key: []const u8) std.json.Value {
    return v.object.get(key).?;
}
test "32 Python fnc chains, keyed noise, selector edges and cost cutoffs" {
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, @embedFile("fixtures_p9.json"), .{});
    defer parsed.deinit();
    const root = parsed.value;
    var pred: [32 * 3]f32 = undefined;
    var succ: [32 * 3]f32 = undefined;
    floats(get(root, "pred"), &pred);
    floats(get(root, "succ"), &succ);
    for (get(root, "chains").array.items) |c| {
        var cs: [7 * 16]policy.Candidate = undefined;
        for (&cs, get(c, "ids").array.items, get(c, "values").array.items) |*cand, id, val| cand.* = .{ .id = @intFromFloat(number(id)), .value = number(val) };
        var proj: [7 * 3]f32 = undefined;
        floats(get(c, "projection"), &proj);
        var costs: [8]f64 = undefined;
        const cost = get(c, "costs");
        if (cost != .null) for (cost.array.items, &costs) |x, *y| {
            y.* = number(x);
        };
        var out: [7]u32 = undefined;
        const n = try policy.chain(&cs, &proj, .{ .pred = &pred, .succ = &succ, .rank = 3 }, @intFromFloat(number(get(c, "anchor"))), @intFromFloat(number(get(c, "first"))), .{ .seed = @intFromFloat(number(get(c, "seed"))), .temperature = number(get(c, "temperature")) }, .{ .threshold = number(get(c, "threshold")) }, &out, if (cost == .null) null else &costs);
        const want = get(c, "want");
        var expected: [7]u32 = undefined;
        ids(want, expected[0..want.array.items.len]);
        try std.testing.expectEqualSlices(u32, expected[0..want.array.items.len], out[0..n]);
    }
}
test "32 Python prompt lookup transcripts including partial keeps and reply match" {
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, @embedFile("fixtures_p9.json"), .{});
    defer parsed.deinit();
    for (get(parsed.value, "copies").array.items) |case| {
        var initial: [32]u32 = undefined;
        ids(get(case, "context"), &initial);
        var c = try Copy.init(std.testing.allocator, &initial, .{ .reply_match = @intFromFloat(number(get(case, "reply"))), .miss_most = @intFromFloat(number(get(case, "miss"))) });
        defer c.deinit();
        var out: [15]u32 = undefined;
        for (get(case, "actions").array.items) |act| {
            const got = c.propose(out[0..@intFromFloat(number(get(act, "room")))]);
            const want = get(act, "want");
            var expected: [15]u32 = undefined;
            ids(want, expected[0..want.array.items.len]);
            try std.testing.expectEqualSlices(u32, expected[0..want.array.items.len], got);
            const draws = get(act, "draws");
            var tokens: [16]u32 = undefined;
            ids(draws, tokens[0..draws.array.items.len]);
            try c.extend(tokens[0..draws.array.items.len]);
        }
    }
}
test "arbitrary proposed drafts preserve the serial target draws through P7 acceptance" {
    const rounds = @import("cuda_rounds.zig");
    var rng = std.Random.DefaultPrng.init(77);
    const random = rng.random();
    for (0..100) |_| {
        var pos: usize = 0;
        while (pos < 100) {
            var input: [16]u32 = undefined;
            var draws: [16]u32 = undefined;
            const rows = @min(100 - pos, 16);
            input[0] = @intCast(pos);
            for (0..rows) |i| {
                draws[i] = @intCast(pos + i + 1);
                if (i > 0) input[i] = if (random.boolean()) @intCast(pos + i) else 999;
            }
            const keep = try rounds.accept(input[0..rows], draws[0..rows], &.{});
            for (draws[0..keep], 0..) |draw, i| try std.testing.expectEqual(@as(u32, @intCast(pos + i + 1)), draw);
            pos += keep;
        }
    }
}
test "Tri DFlash2 reference signatures cover KV-group ranks and tap buckets" {
    const tr = @import("cuda_triton.zig");
    const Rank = @import("cuda_dflash_config.zig").Rank;
    for (1..5) |world| for (0..world) |rank| {
        const g = try Rank.init(world, rank);
        var rec: tr.Recorder = .{};
        const t: tr.Tri = .{ .rec = &rec };
        try t.draftConv(16, 32, 48, null, 64, 8, 0, 8);
        try t.draftConv(16, 32, 48, 80, 64, 8, 1, 8);
        try t.draftPrep(16, 32, 48, 64, 80, 96, 112, 128, 8, g.heads, g.kv, 1e-5);
        try t.draftPrep(16, 32, 48, 64, 80, 96, 112, 128, 64, 0, g.kv, 1e-5);
        try t.draftAttention(16, 32, 48, 64, 80, 96, 3, g.heads, g.kv, 3 * 2176 + 64);
        const attn = rec.launches[4];
        try std.testing.expectEqual(@as([3]u32, .{ 8, @intCast(g.heads + 2 * g.kv), 1 }), rec.launches[2].grid);
        try std.testing.expectEqual(@as([3]u32, .{ @intCast(g.kv), 3, 1 }), attn.grid);
        try std.testing.expectEqualStrings("L", attn.args[8].name);
        try std.testing.expectEqual(@as(i32, 24), attn.args[8].value.i32);
        try std.testing.expectEqual(@as(?i64, 2176), attn.consts[4].int);
        try std.testing.expectEqual(@as(?i64, @intCast(g.heads / g.kv)), attn.consts[1].int);
        try std.testing.expectEqual(@as(?i64, 1), rec.launches[1].consts[4].int);
        try std.testing.expectEqual(@as(i32, 64), rec.launches[3].args[8].value.i32);
        rec.n = 0;
    };
}
test "DFlash checkpoint config accepts shipped geometry and rejects causal change" {
    const cfg = @import("cuda_dflash_config.zig").Config;
    const json = @embedFile("fixtures_dflash_config.json");
    const c = try cfg.parse(std.testing.allocator, json);
    try std.testing.expectEqual(@as(u32, 154856), c.mask);
    const changed = try std.mem.replaceOwned(u8, std.testing.allocator, json, "\"is_causal\": false", "\"is_causal\": true");
    defer std.testing.allocator.free(changed);
    try std.testing.expectError(error.UnsupportedDraftConfig, cfg.parse(std.testing.allocator, changed));
}
test "lane adapter compiles E.DraftReq and validates requests before device work" {
    const df = @import("cuda_dflash.zig");
    const Toy = struct {
        pub const Req = struct { seq: usize, next: []const u32, count: usize, sampling: ?policy.Sampling = null, out: []u32, drafted: usize = 0 };
        pub fn draftSlot(_: *@This(), seq: usize) !usize {
            return seq;
        }
        fn adapt(m: *df.Multi, e: *@This(), qs: []Req) !void {
            try df.draftMany(m, e, qs);
        }
    };
    std.mem.doNotOptimizeAway(&Toy.adapt);
    var engine: Toy = .{};
    var m: df.Multi = undefined;
    var empty: [0]Toy.Req = .{};
    try std.testing.expectError(error.BadDraftStreams, Toy.adapt(&m, &engine, &empty));
    var out: [7]u32 = undefined;
    var req = [_]Toy.Req{.{ .seq = 0, .next = &.{}, .count = 7, .out = &out }};
    try std.testing.expectError(error.BadDraftRequest, Toy.adapt(&m, &engine, &req));
}
