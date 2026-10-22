//! TF_GLM_HC_SPLIT=1 (recipe patch 0010, glm5_next/cuda/hcsplit.py) for the Zig prompt chunk, two ranks: the chunk's
//! hyper-connection glue is split by rows. Rank r owns rows [r H, r H + H), H = padded(R) / 2 (an odd chunk gets one zero
//! pad row). At each of a layer's two sites a rank sends the peer its fp32 partial of the peer's rows, receives the
//! peer's partial of its own, sums them rank 0 first (the same `_hc_post` kernel on a [2, H, D] block), runs the next
//! hc_pre on its own rows and swaps its normed rows (bf16) for the peer's. Every glue kernel is row-independent and the
//! partial sum order is the unsplit one, so every bit is the unsplit path's. Past two ranks the engine runs unsplit.
//! Pure row arithmetic here; the launches are in cuda_prefill.zig.
//! Ash Hart and TensorFold contributors wrote the Python engine; this port is by Mia's AI Lab.
const std = @import("std");
const cuda = @import("cuda");

pub const Settings = struct {
    on: bool = false, // TF_GLM_HC_SPLIT
    min_rows: usize = 64, // TF_GLM_HC_SPLIT_MIN_ROWS: shorter chunks run unsplit
    overlap: u8 = 0, // TF_GLM_PREFILL_OVERLAP: 0, 1 (side stream, row pieces), 2 (and the next block's front on own rows)
    pieces: usize = 2, // TF_GLM_OVERLAP_PIECES
    own_front: bool = false, // TF_GLM_PROMPT_OWN_FRONT: DSA's replicated front on own rows, its outputs shared
};

pub fn padded(rows: usize, world: usize) usize {
    return (rows + world - 1) / world * world;
}

/// Whether a chunk of `rows` rows runs split: the feature on, two ranks, long enough, pad rows fit the buffers.
pub fn applies(s: Settings, world: usize, rows: usize, alloc_rows: usize) bool {
    return s.on and world == 2 and rows >= s.min_rows and padded(rows, world) <= alloc_rows;
}

pub const Piece = struct { off: usize, n: usize };

/// A rank's rows in pieces (hcsplit.py begin): n pieces of ceil(H / n) rows, whole 64-row tiles where possible.
pub fn pieces(h: usize, count: usize, out: []Piece) []Piece {
    std.debug.assert(count <= max_pieces);
    var size = (h + count - 1) / count;
    if (size > 64) size = @min(h, (size + 63) / 64 * 64);
    var k: usize = 0;
    var off: usize = 0;
    while (off < h) : (off += size) {
        out[k] = .{ .off = off, .n = @min(size, h - off) };
        k += 1;
    }
    return out[0..k];
}

/// TF_GLM_PREFILL_OVERLAP=1: the second stream and the events that order it with the main one (hcsplit.py HcSplit).
pub const max_pieces = 8;
pub const Lane = struct {
    stream: cuda.Stream,
    fill: [max_pieces]cuda.Event, // main -> side: piece j of the partial is written
    part: [max_pieces]cuda.Event, // side -> main: piece j's partial exchange is done
    glue: [max_pieces]cuda.Event, // main -> side: piece j's glued rows are ready to swap
    own: cuda.Event, // main -> side at the start of a chunk
    done: cuda.Event, // side -> main: the swaps are done
    hf: [2]cuda.Event, // micro-batching: main -> side, half's partial is written
    hd: [2]cuda.Event, // micro-batching: side -> main, half's chain is done
    share_in: cuda.Event, // main -> side: an own-row share's rows are written
    share_out: cuda.Event, // side -> main: the share has arrived
    pub fn init(d: *const cuda.Driver) !Lane {
        var l: Lane = undefined;
        l.stream = try cuda.Stream.init(d, true);
        for (0..max_pieces) |i| {
            l.fill[i] = try cuda.Event.init(d, false);
            l.part[i] = try cuda.Event.init(d, false);
            l.glue[i] = try cuda.Event.init(d, false);
        }
        l.own = try cuda.Event.init(d, false);
        l.done = try cuda.Event.init(d, false);
        for (0..2) |i| {
            l.hf[i] = try cuda.Event.init(d, false);
            l.hd[i] = try cuda.Event.init(d, false);
        }
        l.share_in = try cuda.Event.init(d, false);
        l.share_out = try cuda.Event.init(d, false);
        return l;
    }
    pub fn deinit(l: *Lane) void {
        for (0..max_pieces) |i| {
            l.fill[i].deinit();
            l.part[i].deinit();
            l.glue[i].deinit();
        }
        l.own.deinit();
        l.done.deinit();
        for (0..2) |i| {
            l.hf[i].deinit();
            l.hd[i].deinit();
        }
        l.share_in.deinit();
        l.share_out.deinit();
        l.stream.deinit();
    }
};

/// One split chunk's row arithmetic.
pub const Run = struct {
    rows: usize,
    h: usize,
    mine: usize,
    theirs: usize,
    np: usize,
    pieces: [max_pieces]Piece,
    lane: bool,

    pub fn init(rows: usize, rank: usize, s: Settings) Run {
        var r: Run = undefined;
        r.rows = rows;
        r.h = padded(rows, 2) / 2;
        r.mine = rank * r.h;
        r.theirs = (1 - rank) * r.h;
        r.lane = s.overlap != 0;
        const count: usize = if (r.lane) s.pieces else 1;
        const ps = pieces(r.h, count, &r.pieces);
        r.np = ps.len;
        return r;
    }
};

pub fn flagEnv(env: anytype, name: []const u8) !bool {
    const t = std.mem.trim(u8, env.get(name) orelse "", " \t\r\n");
    if (t.len == 0 or std.mem.eql(u8, t, "0")) return false;
    if (std.mem.eql(u8, t, "1")) return true;
    return error.BadPromptSetting;
}

fn intEnv(env: anytype, name: []const u8, default: usize, lo: usize, hi: usize) !usize {
    const t = std.mem.trim(u8, env.get(name) orelse "", " \t\r\n");
    if (t.len == 0) return default;
    const n = std.fmt.parseInt(usize, t, 10) catch return error.BadPromptSetting;
    if (n < lo or n > hi) return error.BadPromptSetting;
    return n;
}

pub fn parse(env: anytype) !Settings {
    const on = try flagEnv(env, "TF_GLM_HC_SPLIT");
    const lvl = std.mem.trim(u8, env.get("TF_GLM_PREFILL_OVERLAP") orelse "", " \t\r\n");
    const overlap: u8 = if (lvl.len == 0 or std.mem.eql(u8, lvl, "0")) 0 else if (std.mem.eql(u8, lvl, "1")) 1 else if (std.mem.eql(u8, lvl, "2")) 2 else return error.BadPromptSetting;
    if (overlap != 0 and !on) return error.OverlapNeedsSplit;
    const own = try flagEnv(env, "TF_GLM_PROMPT_OWN_FRONT");
    if (own and !on) return error.OwnFrontNeedsSplit;
    return .{
        .own_front = own,
        .on = on,
        .min_rows = try intEnv(env, "TF_GLM_HC_SPLIT_MIN_ROWS", 64, 2, 16384),
        .overlap = overlap,
        .pieces = try intEnv(env, "TF_GLM_OVERLAP_PIECES", 2, 1, 8),
    };
}

const ProcessEnv = struct {
    fn get(_: ProcessEnv, name: []const u8) ?[]const u8 {
        var buf: [96:0]u8 = undefined;
        @memcpy(buf[0..name.len], name);
        buf[name.len] = 0;
        return if (std.c.getenv(&buf)) |p| std.mem.span(p) else null;
    }
};
/// TF_GLM_PREFILL_ROWS: 64 to 16384, a multiple of 64 (the KDA grid); unset is 2048.
pub fn prefillRows(env: anytype) !usize {
    const n = try intEnv(env, "TF_GLM_PREFILL_ROWS", 2048, 64, 16384);
    if (n % 64 != 0) return error.BadPromptSetting;
    return n;
}
pub fn rowsFromEnv() !usize {
    return prefillRows(ProcessEnv{});
}

pub fn fromEnv() !Settings {
    return parse(ProcessEnv{});
}

const Fake = struct {
    pairs: []const [2][]const u8,
    fn get(e: Fake, name: []const u8) ?[]const u8 {
        for (e.pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "recipe split settings: SPLIT=1 with OVERLAP=2, default pieces" {
    const s = try parse(Fake{ .pairs = &.{ .{ "TF_GLM_HC_SPLIT", "1" }, .{ "TF_GLM_PREFILL_OVERLAP", "2" } } });
    try std.testing.expect(s.on and s.overlap == 2 and s.pieces == 2 and s.min_rows == 64);
    try std.testing.expect(!(try parse(Fake{ .pairs = &.{} })).on);
    try std.testing.expectError(error.OverlapNeedsSplit, parse(Fake{ .pairs = &.{.{ "TF_GLM_PREFILL_OVERLAP", "1" }} }));
    try std.testing.expectError(error.BadPromptSetting, parse(Fake{ .pairs = &.{ .{ "TF_GLM_HC_SPLIT", "1" }, .{ "TF_GLM_PREFILL_OVERLAP", "3" } } }));
}

test "applies: two ranks, long enough, pad rows fit" {
    const s: Settings = .{ .on = true };
    try std.testing.expect(applies(s, 2, 2048, 2048));
    try std.testing.expect(applies(s, 2, 925, 2048));
    try std.testing.expect(!applies(s, 2, 63, 2048));
    try std.testing.expect(!applies(s, 2, 2047, 2047)); // pad row would not fit
    try std.testing.expect(!applies(s, 3, 2048, 2048)); // past two ranks: unsplit for now
    try std.testing.expect(!applies(.{}, 2, 2048, 2048));
}

test "pieces tile a rank's rows, whole 64-row tiles where possible" {
    var buf: [8]Piece = undefined;
    const p = pieces(1024, 2, &buf);
    try std.testing.expectEqual(@as(usize, 2), p.len);
    try std.testing.expectEqual(Piece{ .off = 0, .n = 512 }, p[0]);
    try std.testing.expectEqual(Piece{ .off = 512, .n = 512 }, p[1]);
    var buf2: [8]Piece = undefined;
    const t = pieces(463, 2, &buf2); // a 925-row tail: 232 -> 256-row tiles
    try std.testing.expectEqual(@as(usize, 2), t.len);
    try std.testing.expectEqual(Piece{ .off = 256, .n = 207 }, t[1]);
    var buf3: [8]Piece = undefined;
    const one = pieces(32, 2, &buf3);
    try std.testing.expectEqual(@as(usize, 2), one.len);
    var sum: usize = 0;
    for (p) |x| sum += x.n;
    try std.testing.expectEqual(@as(usize, 1024), sum);
    try std.testing.expectEqual(@as(usize, 926), padded(925, 2));
}

test "TF_GLM_PREFILL_ROWS: default 2048, multiples of 64 up to 16384" {
    try std.testing.expectEqual(@as(usize, 2048), try prefillRows(Fake{ .pairs = &.{} }));
    try std.testing.expectEqual(@as(usize, 4096), try prefillRows(Fake{ .pairs = &.{.{ "TF_GLM_PREFILL_ROWS", "4096" }} }));
    try std.testing.expectError(error.BadPromptSetting, prefillRows(Fake{ .pairs = &.{.{ "TF_GLM_PREFILL_ROWS", "4100" }} }));
    try std.testing.expectError(error.BadPromptSetting, prefillRows(Fake{ .pairs = &.{.{ "TF_GLM_PREFILL_ROWS", "32768" }} }));
}
