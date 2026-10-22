//! Compare the Zig EXL3 weight plan against tools/zig/glm_weight_plan.py's JSON fixture (the Python
//! engine's split rules, computed from safetensors JSON headers only).
const std = @import("std");
const cfgs = @import("cuda_config.zig");
const lay = @import("cuda_layouts.zig");
const W = @import("cuda_weights.zig");

const fixture_text = @embedFile("fixtures_weight_plan.json");
const io = std.testing.io;

fn asInt(v: std.json.Value) i64 {
    return switch (v) {
        .integer => |i| i,
        .float => |f| @intFromFloat(f),
        else => 0,
    };
}

fn asU32(v: std.json.Value) u32 {
    return @intCast(asInt(v));
}

fn asU64(v: std.json.Value) u64 {
    return @intCast(asInt(v));
}

test "fixture shares match Layout.split_sizes and draft_parts" {
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, fixture_text, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const cfg_o = root.get("config").?.object;
    var why: cfgs.Why = .{};
    var cfg = try cfgs.parse(std.testing.allocator, @embedFile("fixtures_config.json"), @embedFile("fixtures_generation_config.json"), .{ .assistant_end = 154828 }, &why);
    defer cfg.deinit();
    try std.testing.expectEqual(asU32(cfg_o.get("hidden").?), cfg.hidden);
    try std.testing.expectEqual(asU32(cfg_o.get("layers").?), cfg.layers);
    try std.testing.expectEqual(asU32(cfg_o.get("vocab").?), cfg.vocab);
    try std.testing.expectEqual(asU32(cfg_o.get("heads").?), cfg.heads);
    try std.testing.expectEqual(asU32(cfg_o.get("moe_width").?), cfg.moe_width);
    try std.testing.expectEqual(asU32(cfg_o.get("experts").?), cfg.experts);
    try std.testing.expectEqual(cfgs.Quant.exl3, cfg.quant);

    const shares = root.get("shares").?.object;
    inline for (.{ 1, 2, 3, 4 }) |world| {
        const wobj = shares.get(std.fmt.comptimePrint("{d}", .{world})).?.object;
        const layout = try lay.Layout.fromConfig(&cfg, 0, world);
        var buf: [8]u32 = undefined;
        inline for (.{ .heads, .lin, .dense, .moe, .shared, .vocab }) |fam| {
            const got = try layout.sizes(fam, &buf);
            const want = wobj.get(@tagName(fam)).?.array.items;
            try std.testing.expectEqual(want.len, got.len);
            for (want, got) |wv, gv| try std.testing.expectEqual(asU32(wv), gv);
        }
        if (wobj.get("dflash_q")) |q| {
            var qs: [8]u32 = undefined;
            var kvs: [8]u32 = undefined;
            var mlps: [8]u32 = undefined;
            try lay.draftParts(32, 8, 12288, world, &qs, &kvs, &mlps);
            for (q.array.items, 0..) |wv, i| try std.testing.expectEqual(asU32(wv), qs[i]);
            for (wobj.get("dflash_kv").?.array.items, 0..) |wv, i| try std.testing.expectEqual(asU32(wv), kvs[i]);
            for (wobj.get("dflash_mlp").?.array.items, 0..) |wv, i| try std.testing.expectEqual(asU32(wv), mlps[i]);
        }
    }
}

test "Zig plan matches the Python fixture against the real checkpoint headers" {
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, fixture_text, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const ckpt = root.get("checkpoint").?.string;
    std.Io.Dir.cwd().access(io, ckpt, .{}) catch return error.SkipZigTest;
    const dflash_v = root.get("dflash2").?;
    const dflash: ?[]const u8 = if (dflash_v == .string) dflash_v.string else null;

    var why: cfgs.Why = .{};
    var cfg = try cfgs.Config.read(std.testing.allocator, io, ckpt, .{}, &why);
    defer cfg.deinit();
    try cfg.check(&why);

    var cat = try W.loadCatalog(std.testing.allocator, io, ckpt, dflash);
    defer cat.deinit();

    const ranks = root.get("ranks").?.object;
    inline for (.{ 1, 2, 3, 4 }) |world| {
        const wkey = std.fmt.comptimePrint("{d}", .{world});
        const want_ranks = ranks.get(wkey).?.array.items;
        for (want_ranks, 0..) |wr, r| {
            const wo = wr.object;
            var p = try W.plan(std.testing.allocator, &cfg, &cat, .{ .rank = @intCast(r), .world = world, .dflash2_dir = dflash });
            defer p.deinit();
            try std.testing.expectEqual(asU32(wo.get("count").?), @as(u32, @intCast(p.items.items.len)));
            try std.testing.expectEqual(asU32(wo.get("dropped").?), p.dropped);
            try std.testing.expectEqual(asU64(wo.get("file_bytes").?), p.file_bytes);
            try std.testing.expectEqual(asU64(wo.get("device_bytes").?), p.device_bytes);
            var hex: [64]u8 = undefined;
            try std.testing.expectEqualStrings(wo.get("sha256").?.string, p.hex(&hex));
        }
    }

    const sample = root.get("sample").?.array.items;
    var last_world: u32 = 0;
    var last_rank: u32 = 0;
    var plan_hold: ?W.Plan = null;
    defer if (plan_hold) |*p| p.deinit();
    for (sample) |sv| {
        const so = sv.object;
        const world = asU32(so.get("world").?);
        const rank = asU32(so.get("rank").?);
        if (plan_hold == null or world != last_world or rank != last_rank) {
            if (plan_hold) |*p| p.deinit();
            plan_hold = try W.plan(std.testing.allocator, &cfg, &cat, .{ .rank = rank, .world = world, .dflash2_dir = dflash });
            last_world = world;
            last_rank = rank;
        }
        const name = so.get("name").?.string;
        const it = W.find(&plan_hold.?, name) orelse {
            std.debug.print("missing sample {s} tp={d} rank={d}\n", .{ name, world, rank });
            return error.MissingSample;
        };
        try std.testing.expectEqualStrings(so.get("file").?.string, it.src.file);
        try std.testing.expectEqual(asU64(so.get("file_begin").?), it.slice.file_begin);
        try std.testing.expectEqual(asU64(so.get("file_end").?), it.slice.file_end);
        try std.testing.expectEqual(asU64(so.get("device_offset").?), it.device_offset);
        try std.testing.expectEqualStrings(so.get("dtype").?.string, it.src.dtype);
        try std.testing.expectEqualStrings(so.get("kind").?.string, it.slice.kind.text());
        var slbuf: [64]u8 = undefined;
        try std.testing.expectEqualStrings(so.get("slice").?.string, it.slice.text(&slbuf));
        try std.testing.expectEqualStrings(so.get("device_dtype").?.string, it.slice.device_dtype);
        const want_shape = so.get("shape").?.array.items;
        try std.testing.expectEqual(want_shape.len, it.slice.rank);
        for (want_shape, 0..) |d, i| try std.testing.expectEqual(asU32(d), it.slice.shape[i]);
        const want_ds = so.get("device_shape").?.array.items;
        try std.testing.expectEqual(want_ds.len, it.slice.device_rank);
        for (want_ds, 0..) |d, i| try std.testing.expectEqual(asU32(d), it.slice.device_shape[i]);
    }
}
