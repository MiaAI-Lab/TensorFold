//! L2 prefetch of the next kernels' weights in decode windows: glm5_next/cuda/l2pf.py (the sites, ranges and pieces) and
//! l2pf.cu (the kernel, zig/kernels/cuda/glm_ops.cu glm_l2pf). A small kernel on a side stream, forked at a site of the
//! main stream, asks for the weights the main stream reads next to be brought into the L2 while it waits on an
//! all-gather or runs latency-bound glue. It only reads, so every model result keeps its bits.
//! Sites: a (at the attention block's all-gather: the FFN's glue weights, router, shared expert), f (at the FFN's
//! all-gather: the next layer's glue weights and input projections; after the last layer the final norm and the head),
//! o (after the attention block's input projections: its output projection).
//! l2pf.py adapts jayleaton/glm53-tensorfold-spark patches/0460 (Apache-2.0, Copyright 2026 Jay Leaton): the sites, the
//! bulk / lines / touch mechanisms, `q4_heads`, the side stream forked at a site and joined before a forward returns.
//! Deviation: a range's end is rounded down to 16 bytes, not up, so no prefetch reads past an allocation.
const std = @import("std");
const cuda = @import("cuda");

pub const align_bytes: u64 = 16;
pub const default_piece: u64 = 32 << 10;
pub const one_wave: usize = 192;

pub const Which = enum(u2) { a, f, o };
pub const Tensor = struct { ptr: u64, bytes: u64 };
/// A 4-bit matrix (cuda_matmuls Q4): `weight` is the packed words (npad * k / 2 bytes).
pub const Q4Info = struct { n: usize, k: usize, split_k: usize };
/// One weight object: its tensors; `weight` indexes the big matrix of a 4-bit one.
pub const Group = struct { tensors: []const Tensor, weight: ?usize = null, q4: ?Q4Info = null };
pub const Range = struct { addr: u64, bytes: u64 };

pub const Settings = struct {
    mode: ?u32 = 0, // null: off; 0 bulk, 1 lines, 2 touch
    mb: f64 = 8.0,
    sites: [3]bool = .{ true, true, true },
    rows: usize = 64,
    blocks: usize = 0,
    threads: u32 = 128,
    piece: u64 = default_piece,

    pub const ParseError = error{BadL2pfSetting};
    /// TF_GLM_L2PF (default on = bulk; 0/off; 1|bulk|lines|touch), TF_GLM_L2PF_{MB,SITES,ROWS,BLOCKS,THREADS,KB}.
    pub fn parse(get: *const fn ([:0]const u8) ?[]const u8) ParseError!Settings {
        var s: Settings = .{};
        if (get("TF_GLM_L2PF")) |v| {
            if (std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off")) {
                s.mode = null;
            } else if (std.mem.eql(u8, v, "1") or std.ascii.eqlIgnoreCase(v, "bulk")) {
                s.mode = 0;
            } else if (std.ascii.eqlIgnoreCase(v, "lines")) {
                s.mode = 1;
            } else if (std.ascii.eqlIgnoreCase(v, "touch")) {
                s.mode = 2;
            } else return error.BadL2pfSetting;
        }
        if (get("TF_GLM_L2PF_MB")) |v| s.mb = std.fmt.parseFloat(f64, v) catch return error.BadL2pfSetting;
        if (!(s.mb > 0 and s.mb <= 64)) return error.BadL2pfSetting;
        if (get("TF_GLM_L2PF_SITES")) |v| {
            s.sites = .{ false, false, false };
            if (v.len == 0) return error.BadL2pfSetting;
            for (v) |ch| switch (std.ascii.toLower(ch)) {
                'a' => s.sites[0] = true,
                'f' => s.sites[1] = true,
                'o' => s.sites[2] = true,
                else => return error.BadL2pfSetting,
            };
        }
        if (get("TF_GLM_L2PF_ROWS")) |v| s.rows = std.fmt.parseInt(usize, v, 10) catch return error.BadL2pfSetting;
        if (get("TF_GLM_L2PF_BLOCKS")) |v| s.blocks = std.fmt.parseInt(usize, v, 10) catch return error.BadL2pfSetting;
        if (get("TF_GLM_L2PF_THREADS")) |v| s.threads = std.fmt.parseInt(u32, v, 10) catch return error.BadL2pfSetting;
        if (get("TF_GLM_L2PF_KB")) |v| s.piece = (std.fmt.parseInt(u64, v, 10) catch return error.BadL2pfSetting) << 10;
        if (s.threads < 32 or s.threads % 32 != 0 or s.threads > 1024) return error.BadL2pfSetting;
        if (s.piece < (1 << 10) or s.piece > (1 << 20)) return error.BadL2pfSetting;
        return s;
    }
    pub fn fromEnv() ParseError!Settings {
        const Env = struct {
            fn get(name: [:0]const u8) ?[]const u8 {
                return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
            }
        };
        return parse(&Env.get);
    }
    pub fn budget(s: Settings) u64 {
        return @intFromFloat(s.mb * (1 << 20));
    }
    /// Blocks for a site of `n` pieces: TF_GLM_L2PF_BLOCKS, or a thread a piece (bulk) / a warp a piece.
    pub fn grid(s: Settings, n: usize) u32 {
        if (s.blocks != 0) return @intCast(s.blocks);
        const per: usize = if (s.mode.? == 0) s.threads else s.threads / 32;
        return @intCast(@max(1, @min(48, (n + per - 1) / per)));
    }
};

/// `t`'s bytes from its start (16-byte aligned downward), at most `left`, the end rounded down to 16 bytes.
fn span(t: Tensor, left: u64) ?Range {
    const lo = t.ptr - t.ptr % align_bytes;
    var hi = @min(t.ptr + t.bytes, lo + left);
    hi -= hi % align_bytes;
    return if (hi > lo) .{ .addr = lo, .bytes = hi - lo } else null;
}

/// l2pf.py q4_heads: the first `left / chunks` bytes of EVERY (64-column tile, K slice) chunk of a one-wave launch.
pub fn q4Heads(gpa: std.mem.Allocator, out: *std.ArrayList(Range), weight: Tensor, q: Q4Info, left: u64) !bool {
    const tiles = (q.n + 63) / 64;
    const chunks: u64 = tiles * q.split_k;
    if (chunks == 0 or chunks > one_wave or weight.bytes % chunks != 0) return false;
    const size = weight.bytes / chunks;
    const per = @min(size, left / chunks) / align_bytes * align_bytes;
    if (per < 512 or weight.ptr % align_bytes != 0 or size % align_bytes != 0) return false;
    for (0..chunks) |c| try out.append(gpa, .{ .addr = weight.ptr + c * size, .bytes = per });
    return true;
}

fn bytesLess(_: void, a: Tensor, b: Tensor) bool {
    return a.bytes < b.bytes;
}

/// l2pf.py ranges: `groups` in the order the main stream reads them, up to `budget` bytes. Within a group the small
/// tensors first (scales, biases, norms), then a 4-bit matrix's chunk heads when its decode launch is one wave, else
/// the large tensors' leading bytes.
pub fn ranges(gpa: std.mem.Allocator, out: *std.ArrayList(Range), groups: []const Group, budget: u64) !void {
    var left = budget;
    for (groups) |g| {
        var order: [8]Tensor = undefined;
        if (g.tensors.len > order.len) return error.TooManyTensors;
        @memcpy(order[0..g.tensors.len], g.tensors);
        const ts: []Tensor = order[0..g.tensors.len];
        std.mem.sort(Tensor, ts, {}, bytesLess);
        var heads: std.ArrayList(Range) = .empty;
        defer heads.deinit(gpa);
        var skip: ?u64 = null;
        if (g.q4) |q| if (g.weight) |wi| {
            const w = g.tensors[wi];
            var small: u64 = 0;
            for (g.tensors, 0..) |t, i| if (i != wi) {
                small += t.bytes;
            };
            if (left > small and try q4Heads(gpa, &heads, w, q, left - small)) skip = w.ptr;
        };
        for (ts) |t| {
            if (left < align_bytes) return;
            if (skip != null and t.ptr == skip.?) continue;
            if (span(t, left)) |sp| {
                try out.append(gpa, sp);
                left -= sp.bytes;
            }
        }
        var heads_total: u64 = 0;
        for (heads.items) |h| heads_total += h.bytes;
        if (heads.items.len > 0 and left >= heads_total) {
            try out.appendSlice(gpa, heads.items);
            left -= heads_total;
        }
    }
}

/// Cut ranges into pieces of at most `piece` bytes.
pub fn pieces(gpa: std.mem.Allocator, out: *std.ArrayList(Range), spans: []const Range, piece: u64) !void {
    for (spans) |r| {
        var o: u64 = 0;
        while (o < r.bytes) : (o += piece) try out.append(gpa, .{ .addr = r.addr + o, .bytes = @min(piece, r.bytes - o) });
    }
}

pub const Site = struct { first: u32, count: u32, grid: u32, bytes: u64 };

/// The device table (address, bytes) rows of every site plus where each layer's sites are.
pub const Builder = struct {
    gpa: std.mem.Allocator,
    s: Settings,
    rows: std.ArrayList(Range) = .empty,
    sites: [64][3]?Site = @splat(@splat(null)),
    pub fn init(gpa: std.mem.Allocator, s: Settings) Builder {
        return .{ .gpa = gpa, .s = s };
    }
    pub fn deinit(b: *Builder) void {
        b.rows.deinit(b.gpa);
    }
    pub fn add(b: *Builder, layer: usize, which: Which, groups: []const Group) !void {
        if (layer >= b.sites.len or !b.s.sites[@backingInt(which)] or groups.len == 0) return;
        var spans: std.ArrayList(Range) = .empty;
        defer spans.deinit(b.gpa);
        try ranges(b.gpa, &spans, groups, b.s.budget());
        var total: u64 = 0;
        for (spans.items) |r| total += r.bytes;
        const first = b.rows.items.len;
        try pieces(b.gpa, &b.rows, spans.items, b.s.piece);
        const n = b.rows.items.len - first;
        if (n == 0) return;
        b.sites[layer][@backingInt(which)] = .{ .first = @intCast(first), .count = @intCast(n), .grid = b.s.grid(n), .bytes = total };
    }
};

pub const Prefetch = struct {
    f: cuda.Function,
    table: cuda.DeviceBuffer,
    sink: cuda.DeviceBuffer,
    side: cuda.Stream,
    fork: cuda.Event,
    done: cuda.Event,
    sites: [64][3]?Site,
    mode: u32,
    threads: u32,
    rows: usize,
    forked: bool = false,

    pub fn init(d: *const cuda.Driver, f: cuda.Function, b: *const Builder) !Prefetch {
        const none = [_]Range{.{ .addr = 0, .bytes = 0 }};
        const rows: []const Range = if (b.rows.items.len == 0) &none else b.rows.items;
        var table = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(rows));
        errdefer table.free();
        var sink = try cuda.DeviceBuffer.alloc(d, 4);
        errdefer sink.free();
        try sink.fill8(0, null);
        var side = try cuda.Stream.init(d, true);
        errdefer side.deinit();
        var fork = try cuda.Event.init(d, false);
        errdefer fork.deinit();
        const done = try cuda.Event.init(d, false);
        return .{ .f = f, .table = table, .sink = sink, .side = side, .fork = fork, .done = done, .sites = b.sites, .mode = b.s.mode.?, .threads = b.s.threads, .rows = b.s.rows };
    }
    pub fn deinit(p: *Prefetch) void {
        p.side.synchronize() catch {};
        p.done.deinit();
        p.fork.deinit();
        p.side.deinit();
        p.sink.free();
        p.table.free();
    }
    /// Fork the side stream at this point of `main` and prefetch the site's ranges there (a decode window of more than
    /// TF_GLM_L2PF_ROWS rows skips every site).
    pub fn site(p: *Prefetch, main: cuda.Stream, layer: usize, which: Which, rows: usize) !void {
        if (rows > p.rows or layer >= p.sites.len) return;
        const st = p.sites[layer][@backingInt(which)] orelse return;
        try p.fork.record(main);
        try p.side.wait(p.fork);
        var a: cuda.Args = .{};
        a.add(p.table.ptr + @as(u64, st.first) * @sizeOf(Range));
        a.add(@as(i32, @intCast(st.count)));
        a.add(@as(i32, @intCast(p.mode)));
        a.add(p.sink.ptr);
        try cuda.launch.launch(p.f, .{ .grid = .{ .x = st.grid }, .block = .{ .x = p.threads } }, p.side, &a);
        p.forked = true;
    }
    /// The main stream waits for the side stream (a captured graph's branches rejoin before its end).
    pub fn join(p: *Prefetch, main: cuda.Stream) !void {
        if (!p.forked) return;
        try p.done.record(p.side);
        try main.wait(p.done);
        p.forked = false;
    }
};

test "ranges: small tensors first, then the big matrix's leading bytes, within the budget" {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(Range) = .empty;
    defer out.deinit(gpa);
    const big = Tensor{ .ptr = 0x10000, .bytes = 1 << 20 };
    const small = Tensor{ .ptr = 0x200000, .bytes = 1000 };
    try ranges(gpa, &out, &.{.{ .tensors = &.{ big, small } }}, 4096);
    try std.testing.expectEqual(@as(usize, 2), out.items.len);
    try std.testing.expectEqual(@as(u64, 0x200000), out.items[0].addr);
    try std.testing.expectEqual(@as(u64, 992), out.items[0].bytes); // 1000 rounded down to 16
    try std.testing.expectEqual(@as(u64, 0x10000), out.items[1].addr);
    try std.testing.expectEqual(@as(u64, 4096 - 992), out.items[1].bytes);
}

test "ranges: a one-wave 4-bit matrix contributes the head of every chunk, a bigger one its leading bytes" {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(Range) = .empty;
    defer out.deinit(gpa);
    // n = 2048, k = 4096, split 4: 32 tiles * 4 = 128 chunks of (2048 * 4096 / 2) / 128 = 32768 bytes
    const w = Tensor{ .ptr = 0x1000000, .bytes = 2048 * 4096 / 2 };
    const sc = Tensor{ .ptr = 0x2000000, .bytes = 2048 * 64 * 2 };
    const bi = Tensor{ .ptr = 0x3000000, .bytes = 2048 * 64 * 2 };
    const g: Group = .{ .tensors = &.{ w, sc, bi }, .weight = 0, .q4 = .{ .n = 2048, .k = 4096, .split_k = 4 } };
    try ranges(gpa, &out, &.{g}, 8 << 20);
    try std.testing.expectEqual(@as(usize, 2 + 128), out.items.len);
    try std.testing.expectEqual(@as(u64, 0x1000000 + 32768), out.items[3].addr);
    try std.testing.expectEqual(@as(u64, 32768), out.items[3].bytes);
    out.clearRetainingCapacity();
    const wide: Group = .{ .tensors = &.{ w, sc, bi }, .weight = 0, .q4 = .{ .n = 12576, .k = 4096, .split_k = 4 } };
    try ranges(gpa, &out, &.{wide}, 1 << 20);
    try std.testing.expectEqual(@as(usize, 3), out.items.len);
    try std.testing.expectEqual(@as(u64, 0x1000000), out.items[2].addr);
    var total: u64 = 0;
    for (out.items) |r| total += r.bytes;
    try std.testing.expect(total <= 1 << 20);
}

test "builder cuts pieces, honors the site letters and sizes the grid" {
    const gpa = std.testing.allocator;
    var s: Settings = .{};
    s.sites = .{ true, false, true };
    var b = Builder.init(gpa, s);
    defer b.deinit();
    const t = Tensor{ .ptr = 0x4000, .bytes = 100 << 10 };
    try b.add(3, .a, &.{.{ .tensors = &.{t} }});
    try b.add(3, .f, &.{.{ .tensors = &.{t} }});
    try std.testing.expect(b.sites[3][1] == null);
    const a = b.sites[3][0].?;
    try std.testing.expectEqual(@as(u32, 4), a.count); // 100 KiB in 32 KiB pieces
    try std.testing.expectEqual(@as(u32, 1), a.grid);
    try std.testing.expectEqual(@as(u64, 32 << 10), b.rows.items[1].bytes);
    try std.testing.expectEqual(@as(u64, 4 << 10), b.rows.items[3].bytes);
}

test "settings parse the environment's spellings" {
    const Env = struct {
        var pairs: []const [2][]const u8 = &.{};
        fn get(name: [:0]const u8) ?[]const u8 {
            for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
            return null;
        }
    };
    Env.pairs = &.{};
    try std.testing.expectEqual(@as(?u32, 0), (try Settings.parse(&Env.get)).mode);
    Env.pairs = &.{ .{ "TF_GLM_L2PF", "off" }, .{ "TF_GLM_L2PF_SITES", "fo" } };
    const off = try Settings.parse(&Env.get);
    try std.testing.expectEqual(@as(?u32, null), off.mode);
    try std.testing.expect(!off.sites[0] and off.sites[1] and off.sites[2]);
    Env.pairs = &.{.{ "TF_GLM_L2PF", "lines" }};
    try std.testing.expectEqual(@as(?u32, 1), (try Settings.parse(&Env.get)).mode);
    Env.pairs = &.{.{ "TF_GLM_L2PF", "yes" }};
    try std.testing.expectError(error.BadL2pfSetting, Settings.parse(&Env.get));
    Env.pairs = &.{.{ "TF_GLM_L2PF_MB", "100" }};
    try std.testing.expectError(error.BadL2pfSetting, Settings.parse(&Env.get));
    Env.pairs = &.{.{ "TF_GLM_L2PF_THREADS", "100" }};
    try std.testing.expectError(error.BadL2pfSetting, Settings.parse(&Env.get));
}
