//! The TP=4 recipe's prompt speed features that sit on the row-split prompt chunk: TF_GLM_PROMPT_OWN_FRONT (patch 0102,
//! glm5_next/cuda/own_front.py) and TF_GLM_PROMPT_MICROBATCH (patch 0104, microbatch.py). Pure functions and no CUDA:
//! settings with Python's validation, the half-chunk cut (a multiple of lcm(64, world)), when a chunk runs as two
//! halves, the halves' row layout, and the step orders (one layer of a microbatched chunk, an own-row front's shares and
//! waits) that the CPU tests check with fakes. cuda_prefill.zig (`microLayers`, `dsaOwnFront`, `selectOwn`) launches them
//! on the split chunk (cuda_prefill_split.zig); `cut`, `applies` and the settings are used there as they are.
//! Ash Hart and TensorFold contributors wrote the Python engine; the two features' design is Mia's AI Lab's
//! (Apache-2.0), after drowzeys' (github.com/drowzeys/TensorFold, glm53-tp4-2026-10-05) sequence-parallel prompt; no code of theirs.
const std = @import("std");

pub const grid_rows: usize = 64; // KDA's chunked recurrence keeps prompt rows on a 64-row grid

pub const Settings = struct {
    split: bool = false, // TF_GLM_HC_SPLIT
    own_front: bool = false, // TF_GLM_PROMPT_OWN_FRONT
    microbatch: bool = false, // TF_GLM_PROMPT_MICROBATCH
    micro_min: usize = 2048, // TF_GLM_PROMPT_MICROBATCH_MIN_ROWS: halves of fewer rows run the chunk whole
    min_rows: usize = 64, // TF_GLM_HC_SPLIT_MIN_ROWS: shorter chunks run unsplit

    /// Either feature asked for: both live on the row split.
    pub fn wanted(s: Settings) bool {
        return s.own_front or s.microbatch;
    }
};

fn flag(v: ?[]const u8) !bool {
    const t = std.mem.trim(u8, v orelse "", " \t\r\n");
    if (t.len == 0 or std.mem.eql(u8, t, "0")) return false;
    if (std.mem.eql(u8, t, "1")) return true;
    return error.BadPromptSetting;
}

fn int(v: ?[]const u8, default: usize, lo: usize, hi: usize) !usize {
    const t = std.mem.trim(u8, v orelse "", " \t\r\n");
    if (t.len == 0) return default;
    const n = std.fmt.parseInt(usize, t, 10) catch return error.BadPromptSetting;
    if (n < lo or n > hi) return error.BadPromptSetting;
    return n;
}

/// `env.get(name)` returns ?[]const u8 (the process environment, or a fake).
pub fn parse(env: anytype) !Settings {
    const s: Settings = .{
        .split = try flag(env.get("TF_GLM_HC_SPLIT")),
        .own_front = try flag(env.get("TF_GLM_PROMPT_OWN_FRONT")),
        .microbatch = try flag(env.get("TF_GLM_PROMPT_MICROBATCH")),
        .micro_min = try int(env.get("TF_GLM_PROMPT_MICROBATCH_MIN_ROWS"), 2048, 64, 16384),
        .min_rows = try int(env.get("TF_GLM_HC_SPLIT_MIN_ROWS"), 64, 2, 16384),
    };
    if (s.own_front and !s.split) return error.OwnFrontNeedsSplit; // hcsplit.py SplitSettings.__post_init__
    if (s.microbatch and !s.split) return error.MicrobatchNeedsSplit;
    return s;
}

const ProcessEnv = struct {
    fn get(_: ProcessEnv, name: [:0]const u8) ?[]const u8 {
        return if (std.c.getenv(name.ptr)) |p| std.mem.span(p) else null;
    }
};
pub fn fromEnv() !Settings {
    return parse(ProcessEnv{});
}

/// Rows of a chunk of R rounded up to whole shares a rank (HcSplit.padded).
pub fn padded(R: usize, world: usize) usize {
    return (R + world - 1) / world * world;
}

/// Half A's rows of a chunk of R: about half, a multiple of 64 and of the world size (microbatch.cut).
pub fn cut(R: usize, world: usize) usize {
    const unit = grid_rows * world / std.math.gcd(grid_rows, world);
    return @max(unit, (R / 2) / unit * unit);
}

/// Whether a row-split chunk of R rows runs as two halves (microbatch.applies): the feature on, one stream's prompt,
/// both halves at least the minimum rows (and of what the split takes).
pub fn applies(s: Settings, world: usize, R: usize, one_stream: bool) bool {
    if (!(s.microbatch and one_stream)) return false;
    const h = cut(R, world);
    const least = @max(s.micro_min, s.min_rows);
    return h >= least and R - h >= least;
}

/// The two halves of a chunk of R rows: A = rows [0, h) at position p, B = rows [h, R) at p + h, B padded to whole
/// shares. B's partial staging starts after A's (`world * h * width` elements).
pub const Halves = struct {
    h: usize,
    nb: usize, // B's rows with its pad rows
    rb: usize, // B's real rows
    gath_b: usize, // offset of B's staging in elements, for rows of `width`

    pub fn pos(hv: Halves, half: Half, p: usize) usize {
        return if (half == .a) p else p + hv.h;
    }
    pub fn rows(hv: Halves, half: Half) usize {
        return if (half == .a) hv.h else hv.rb;
    }
};
pub const Half = enum { a, b };

pub fn halves(R: usize, world: usize, alloc_rows: usize, width: usize) !Halves {
    const h = cut(R, world);
    if (R <= h) return error.BadRows;
    const nb = padded(R - h, world);
    if (h + nb > alloc_rows) return error.HalvesPastBuffers;
    return .{ .h = h, .nb = nb, .rb = R - h, .gath_b = world * h * width };
}

pub const Block = enum { mixer, ffn };
pub const Step = union(enum) {
    /// The main stream waits for the half's queued chain.
    join: Half,
    /// A half's block on the main stream. `own_front`: a DSA block that runs its own-row front.
    block: struct { half: Half, kind: Block, own_front: bool = false },
    /// The half's chain (partials' exchange, glue on its own rows, glued rows' swap) queued on the side stream.
    chain: struct { half: Half, layer_end: bool, share: bool = true },
};

/// One layer of a microbatched chunk (microbatch.layer_steps): the main stream runs A's attention, B's, A's MoE or
/// MLP, B's; each half's chain is queued as soon as its block has filled its partials; a half is joined only right
/// before its next block. `dsa_own`: this layer's block takes the own-row front; `next_dsa_own`: the next layer's does
/// (the layer-end glue then keeps the normed rows on their owner: share = false). `last`: no next layer.
pub fn layerSteps(out: *std.ArrayList(Step), gpa: std.mem.Allocator, own_front: bool, dsa_own: bool, next_dsa_own: bool) !void {
    const hs = [_]Half{ .a, .b };
    for (hs) |h| {
        try out.append(gpa, .{ .join = h });
        try out.append(gpa, .{ .block = .{ .half = h, .kind = .mixer, .own_front = own_front and dsa_own } });
        try out.append(gpa, .{ .chain = .{ .half = h, .layer_end = false } });
    }
    for (hs) |h| {
        try out.append(gpa, .{ .join = h });
        try out.append(gpa, .{ .block = .{ .half = h, .kind = .ffn } });
        try out.append(gpa, .{ .chain = .{ .half = h, .layer_end = true, .share = !(own_front and next_dsa_own) } });
    }
}

/// The row ranges of [0, R) outside this rank's [lo, hi) (own_front._others).
pub fn others(R: usize, lo: usize, hi: usize) [2]?[2]usize {
    var r: [2]?[2]usize = .{ null, null };
    const a_end = @min(lo, R);
    if (a_end > 0) r[0] = .{ 0, a_end };
    if (R > hi) r[1] = .{ hi, R };
    return r;
}

/// A rank's own rows of a split chunk of Rp (padded) rows: [rank H, rank H + H).
pub fn ownRows(Rp: usize, world: usize, rank: usize) [2]usize {
    const hh = Rp / world;
    return .{ rank * hh, rank * hh + hh };
}

pub const Front = enum {
    own_projections, // q_a / kv_a and norms, q_b for own heads, indexer kw / gate / query projection on own rows
    share_keys, // index keys and gates (slot 0)
    share_rows, // the norms' rows qr / lat (slot 1)
    wait_keys,
    index_update, // every row's index key, gate, pooled key
    select_own, // this rank's rows' pools over those keys
    share_pools, // slot 2
    wait_rows,
    q_b_others, // q_b on the other ranks' rows
    latent_write,
    absorb,
    dense_attention,
    wait_pools,
    pool_tokens,
    sparse_attention,
    expand,
    out_proj,
};

/// An own-row DSA front's order (own_front.front + rest): `indexed` = the layer has the index cache, `sparse_rows` =
/// some row selects (position + R - 1 past the dense limit), `all_sparse` = every row does (no dense attention).
pub fn frontSteps(buf: *[20]Front, indexed: bool, sparse_rows: bool, all_sparse: bool) []const Front {
    var n: usize = 0;
    const add = struct {
        fn f(b: *[20]Front, k: *usize, s: Front) void {
            b[k.*] = s;
            k.* += 1;
        }
    }.f;
    add(buf, &n, .own_projections);
    if (indexed) add(buf, &n, .share_keys);
    add(buf, &n, .share_rows);
    if (indexed) {
        add(buf, &n, .wait_keys);
        add(buf, &n, .index_update);
        if (sparse_rows) {
            add(buf, &n, .select_own);
            add(buf, &n, .share_pools);
        }
    }
    add(buf, &n, .wait_rows);
    add(buf, &n, .q_b_others);
    add(buf, &n, .latent_write);
    add(buf, &n, .absorb);
    if (!all_sparse) add(buf, &n, .dense_attention);
    if (sparse_rows) {
        add(buf, &n, .wait_pools);
        add(buf, &n, .pool_tokens);
        add(buf, &n, .sparse_attention);
    }
    add(buf, &n, .expand);
    add(buf, &n, .out_proj);
    return buf[0..n];
}

// ---------------------------------------------------------------- tests

const FakeEnv = struct {
    pairs: []const [2][]const u8,
    fn get(e: FakeEnv, name: []const u8) ?[]const u8 {
        for (e.pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "the recipe's TP=4 settings parse and the features are off when unset" {
    const off = try parse(FakeEnv{ .pairs = &.{} });
    try std.testing.expect(!off.wanted() and !off.split);
    const tp4 = try parse(FakeEnv{ .pairs = &.{ .{ "TF_GLM_HC_SPLIT", "1" }, .{ "TF_GLM_PROMPT_OWN_FRONT", "1" }, .{ "TF_GLM_PROMPT_MICROBATCH", "1" } } });
    try std.testing.expect(tp4.own_front and tp4.microbatch and tp4.split);
    try std.testing.expectEqual(@as(usize, 2048), tp4.micro_min);
    const tp3 = try parse(FakeEnv{ .pairs = &.{ .{ "TF_GLM_HC_SPLIT", "1" }, .{ "TF_GLM_PROMPT_MICROBATCH", "1" }, .{ "TF_GLM_PROMPT_MICROBATCH_MIN_ROWS", "1920" } } });
    try std.testing.expectEqual(@as(usize, 1920), tp3.micro_min);
}

test "settings are refused as Python refuses them" {
    try std.testing.expectError(error.OwnFrontNeedsSplit, parse(FakeEnv{ .pairs = &.{.{ "TF_GLM_PROMPT_OWN_FRONT", "1" }} }));
    try std.testing.expectError(error.MicrobatchNeedsSplit, parse(FakeEnv{ .pairs = &.{.{ "TF_GLM_PROMPT_MICROBATCH", "1" }} }));
    try std.testing.expectError(error.BadPromptSetting, parse(FakeEnv{ .pairs = &.{.{ "TF_GLM_PROMPT_MICROBATCH", "2" }} }));
    try std.testing.expectError(error.BadPromptSetting, parse(FakeEnv{ .pairs = &.{.{ "TF_GLM_PROMPT_MICROBATCH_MIN_ROWS", "63" }} }));
    try std.testing.expectError(error.BadPromptSetting, parse(FakeEnv{ .pairs = &.{.{ "TF_GLM_PROMPT_MICROBATCH_MIN_ROWS", "x" }} }));
}

test "the cut is a multiple of lcm(64, world): 4096 rows split 2048 + 2048 at 2 and 4, 1920 + 2176 at 3" {
    try std.testing.expectEqual(@as(usize, 2048), cut(4096, 2));
    try std.testing.expectEqual(@as(usize, 2048), cut(4096, 4));
    try std.testing.expectEqual(@as(usize, 1920), cut(4096, 3)); // lcm 192: 2048 / 192 = 10.67 -> 10 * 192
    try std.testing.expectEqual(@as(usize, 2048), cut(4160, 2)); // 2080 -> 2048
    // every cut is a multiple of the unit and of 64 and of the world, and at least one unit
    for ([_]usize{ 2, 3, 4 }) |w| {
        const unit = grid_rows * w / std.math.gcd(grid_rows, w);
        var r: usize = 64;
        while (r <= 8192) : (r += 64) {
            const h = cut(r, w);
            try std.testing.expect(h % unit == 0 and h % 64 == 0 and h % w == 0 and h >= unit);
            try std.testing.expect(h <= r / 2 or h == unit);
        }
    }
}

test "a chunk runs in halves only when both halves reach the minimum" {
    const s: Settings = .{ .split = true, .microbatch = true };
    try std.testing.expect(applies(s, 4, 4096, true));
    try std.testing.expect(!applies(s, 4, 2048, true)); // 1024 + 1024 < 2048
    try std.testing.expect(!applies(s, 4, 4096, false)); // several streams' prompts in one chunk
    try std.testing.expect(!applies(.{ .split = true }, 4, 4096, true)); // feature off
    // 3 ranks: 1920 + 2176 needs the minimum lowered to 1920
    try std.testing.expect(!applies(s, 3, 4096, true));
    try std.testing.expect(applies(.{ .split = true, .microbatch = true, .micro_min = 1920 }, 3, 4096, true));
    // a prompt's short tail runs whole
    try std.testing.expect(!applies(s, 4, 2112, true));
}

test "halves: positions, rows, pad rows and B's staging offset" {
    const hv = try halves(4096, 3, 4096 + 2, 4096);
    try std.testing.expectEqual(@as(usize, 1920), hv.h);
    try std.testing.expectEqual(@as(usize, 2176), hv.rb);
    try std.testing.expectEqual(@as(usize, 2178), hv.nb); // 2176 padded to a multiple of 3
    try std.testing.expectEqual(@as(usize, 3 * 1920 * 4096), hv.gath_b);
    try std.testing.expectEqual(@as(usize, 1000), hv.pos(.a, 1000));
    try std.testing.expectEqual(@as(usize, 1000 + 1920), hv.pos(.b, 1000));
    try std.testing.expectEqual(@as(usize, 1920), hv.rows(.a));
    try std.testing.expectEqual(@as(usize, 2176), hv.rows(.b));
    try std.testing.expectError(error.HalvesPastBuffers, halves(4096, 3, 4096, 4096));
    const four = try halves(4096, 4, 4096, 4096);
    try std.testing.expectEqual(@as(usize, 2048), four.nb);
}

test "A's and B's rows tile the chunk exactly" {
    for ([_]usize{ 2, 3, 4 }) |w| {
        const hv = try halves(4096, w, 4096 + w, 4096);
        try std.testing.expectEqual(@as(usize, 4096), hv.rows(.a) + hv.rows(.b));
        try std.testing.expectEqual(hv.rows(.a), hv.pos(.b, 0));
    }
}

fn indexOf(steps: []const Step, comptime tag: std.meta.Tag(Step), half: Half, kind: ?Block, from: usize) ?usize {
    for (steps[from..], from..) |s, i| switch (s) {
        .join => |h| if (tag == .join and h == half) return i,
        .block => |b| if (tag == .block and b.half == half and b.kind == kind.?) return i,
        .chain => |c| if (tag == .chain and c.half == half and (kind.? == .ffn) == c.layer_end) return i,
    };
    return null;
}

test "layer order: A's block before B's, a chain right after its block, a half joined before its next block" {
    var list: std.ArrayList(Step) = .empty;
    defer list.deinit(std.testing.allocator);
    try layerSteps(&list, std.testing.allocator, true, true, false);
    const st = list.items;
    try std.testing.expectEqual(@as(usize, 12), st.len);
    // main-stream block order: A mixer, B mixer, A ffn, B ffn
    const a_mix = indexOf(st, .block, .a, .mixer, 0).?;
    const b_mix = indexOf(st, .block, .b, .mixer, 0).?;
    const a_ffn = indexOf(st, .block, .a, .ffn, 0).?;
    const b_ffn = indexOf(st, .block, .b, .ffn, 0).?;
    try std.testing.expect(a_mix < b_mix and b_mix < a_ffn and a_ffn < b_ffn);
    // each block is preceded by its half's join and followed by its half's chain
    for ([_]struct { h: Half, k: Block }{ .{ .h = .a, .k = .mixer }, .{ .h = .b, .k = .mixer }, .{ .h = .a, .k = .ffn }, .{ .h = .b, .k = .ffn } }) |x| {
        const bi = indexOf(st, .block, x.h, x.k, 0).?;
        try std.testing.expectEqual(Step{ .join = x.h }, st[bi - 1]);
        try std.testing.expect(std.meta.activeTag(st[bi + 1]) == .chain);
        try std.testing.expect(st[bi + 1].chain.half == x.h and st[bi + 1].chain.layer_end == (x.k == .ffn));
    }
    // the own front is on A's and B's mixer only; the layer-end chains share their rows unless the next layer owns
    try std.testing.expect(st[1].block.own_front and st[4].block.own_front and !st[7].block.own_front);
    try std.testing.expect(st[8].chain.share and st[11].chain.share);
    var next: std.ArrayList(Step) = .empty;
    defer next.deinit(std.testing.allocator);
    try layerSteps(&next, std.testing.allocator, true, false, true);
    try std.testing.expect(!next.items[1].block.own_front and !next.items[8].chain.share and !next.items[11].chain.share);
    var off: std.ArrayList(Step) = .empty;
    defer off.deinit(std.testing.allocator);
    try layerSteps(&off, std.testing.allocator, false, true, true);
    try std.testing.expect(!off.items[1].block.own_front and off.items[8].chain.share);
}

test "own front: shares are issued in one order (keys, rows, pools) and each consumer waits for its share" {
    var buf: [20]Front = undefined;
    const full = frontSteps(&buf, true, true, false);
    const pos = struct {
        fn of(s: []const Front, x: Front) ?usize {
            for (s, 0..) |y, i| if (y == x) return i;
            return null;
        }
    }.of;
    try std.testing.expect(pos(full, .share_keys).? < pos(full, .share_rows).?);
    try std.testing.expect(pos(full, .share_rows).? < pos(full, .share_pools).?);
    try std.testing.expect(pos(full, .share_keys).? < pos(full, .wait_keys).?);
    try std.testing.expect(pos(full, .wait_keys).? < pos(full, .index_update).?);
    try std.testing.expect(pos(full, .index_update).? < pos(full, .select_own).?);
    try std.testing.expect(pos(full, .select_own).? < pos(full, .share_pools).?);
    // q_b on other ranks' rows, the latent write and absorb wait for the rows; sparse tokens wait for the pools
    try std.testing.expect(pos(full, .wait_rows).? < pos(full, .q_b_others).?);
    try std.testing.expect(pos(full, .q_b_others).? < pos(full, .latent_write).?);
    try std.testing.expect(pos(full, .share_pools).? < pos(full, .wait_pools).?);
    try std.testing.expect(pos(full, .dense_attention).? < pos(full, .wait_pools).?); // dense runs while the pools are on the wire
    try std.testing.expect(pos(full, .wait_pools).? < pos(full, .pool_tokens).?);
    try std.testing.expect(pos(full, .pool_tokens).? < pos(full, .sparse_attention).?);
    try std.testing.expect(pos(full, .expand).? < pos(full, .out_proj).?);
    // short context: no selection, no pools to share or wait for
    const dense = frontSteps(&buf, true, false, false);
    try std.testing.expect(pos(dense, .select_own) == null and pos(dense, .share_pools) == null and pos(dense, .wait_pools) == null);
    try std.testing.expect(pos(dense, .dense_attention) != null);
    // past the dense limit entirely: no dense attention
    const sparse_only = frontSteps(&buf, true, true, true);
    try std.testing.expect(pos(sparse_only, .dense_attention) == null and pos(sparse_only, .sparse_attention) != null);
    // no index cache: no key shares
    const noindex = frontSteps(&buf, false, false, false);
    try std.testing.expect(pos(noindex, .share_keys) == null and pos(noindex, .index_update) == null);
}

test "other ranks' rows are the two ranges around this rank's own" {
    // 4 ranks, 4096 rows: rank 1 owns [1024, 2048)
    try std.testing.expectEqual([2]usize{ 1024, 2048 }, ownRows(4096, 4, 1));
    const o = others(4096, 1024, 2048);
    try std.testing.expectEqual([2]usize{ 0, 1024 }, o[0].?);
    try std.testing.expectEqual([2]usize{ 2048, 4096 }, o[1].?);
    const first = others(4096, 0, 1024);
    try std.testing.expect(first[0] == null);
    try std.testing.expectEqual([2]usize{ 1024, 4096 }, first[1].?);
    const last = others(4096, 3072, 4096);
    try std.testing.expectEqual([2]usize{ 0, 3072 }, last[0].?);
    try std.testing.expect(last[1] == null);
    // a short chunk: the own range past the rows is cut to them
    const tail = others(100, 96, 128);
    try std.testing.expectEqual([2]usize{ 0, 96 }, tail[0].?);
    try std.testing.expect(tail[1] == null);
}
