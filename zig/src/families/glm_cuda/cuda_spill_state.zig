//! Spill v2's kept states of the GLM CUDA engine: what a spilled kept state (cuda_snapshot.zig) holds on one rank's
//! disk and the copies between its device buffer and the core/spill store's staging. Port of wojo's GLM 0110 (PR
//! #124; BadAd84's block tier: block_hashes' shared blocks, spill.encode's small state, the tail captured at keep,
//! `release`), in the shape of the DeepSeek port's dsv41/spill_state.zig.
//!
//! A kept state is one buffer (cuda_snapshot.sections): the 11 DSA layers' latent rows [0, at) and pooled index keys
//! [0, at/4 + 1), the index rings, the 34 KDA recurrences (each layer's current parity), the conv windows, and
//! DFlash2's rings when a drafter runs. On disk it is
//!   - blocks: the pool planes of tokens [k * 2048, k * 2048 + 2048), layer by layer (its latent rows, then its pooled
//!     rows), named by core/spill's blockNames (the id chain salted with the lineage), so a later turn or a fork names
//!     the blocks its earlier turns wrote and writes only its own. GLM keeps on a 64-token grid, so the block a state's
//!     end cuts is a tail of its own tokens, captured at keep (the store's Queued.tail);
//!   - sections, per point: `rings`, `rec`, `conv` and, with a drafter, `draft`, each one copy of the buffer's own.
//! Pooled row at/4 (the snapshot's + 1) is never state: `at` is a multiple of 64, so that row's four tokens all lie at
//! or past `at`; _pool_keys writes a pool only once its four tokens are in a window (P 0038), and the selection reads
//! the row masked until the resumed prefill writes it whole. A block holds Python's ceil(e/4) = e/4 pooled rows; an
//! unpacked state reads zeros there. Arithmetic and copies only: the tests run on host memory (`Copier`).
const std = @import("std");
const core = @import("core");
const snapshot = @import("cuda_snapshot.zig");
const state = @import("cuda_state.zig");
const spill = core.spill;
const format = spill.format;
const G = state.Geometry;

/// The GLM kept-state format (bumped with any change to the block order, the sections or the digest's input).
pub const format_version: u32 = 1;
/// Tokens a block: the pool's extent (Python's ALIGN).
pub const block_tokens: usize = G.extent;
/// Kept states end on multiples of this (cuda_engine takeSnap); lookups pass it as Query.grid.
pub const grid: usize = 64;
/// Pool-plane bytes a token on a rank: 11 layers of a 528-byte latent row and a quarter of a 144-byte pooled row.
pub const token_bytes: usize = G.dsa_layers * (G.latent_row + G.pool_row / G.pool);

/// DFlash2's rings of a slot: `kv` heads on this rank, `ring` rows (cuda_snapshot.Draft).
pub const Drafter = struct {
    kv: usize,
    ring: usize,

    /// cuda_snapshot.Draft.bytes: K and V of 5 layers, 128 bf16 a head's row.
    pub fn bytes(d: Drafter) usize {
        return 10 * d.kv * d.ring * 128 * 2;
    }
};

/// A point's sections, in the buffer's order (their names on disk).
pub const Section = enum(u8) {
    rings = 0,
    rec = 1,
    conv = 2,
    draft = 3,

    pub fn name(s: Section) []const u8 {
        return @tagName(s);
    }
};

/// Tokens [first, first + tokens) of a point: one block.
pub const Span = struct { first: usize, tokens: usize };

/// One rank's kept-state layout: the buffer (cuda_snapshot.sections), a block's bytes and the sections as the store
/// lays them out (each on a 256-byte offset: a small slot's bytes).
pub const Layout = struct {
    g: G,
    /// Index ring rows (Caches.ring: 2,112 for a pool of several streams, a window of 2,048; a lone stream's follows
    /// TF_GLM_PREFILL_ROWS).
    ring: usize,
    drafter: ?Drafter = null,
    /// A whole block's bytes: 12,705,792 (3,102 pages, so a block slot has no padding).
    block_bytes: usize,
    small_bytes: usize,
    n: usize,
    order: [4]Section,
    lens: [4]usize,
    specs_buf: [4]spill.store.SectionSpec,

    pub fn init(g: G, ring: usize, drafter: ?Drafter) !Layout {
        // spill v2 is ported for the recipe's two ranks (its captured kernels, the pair's agreement); TP 1, 3 and 4
        // are refused until a run checks them
        if (g.world != 2) return error.SpillNeedsTwoRanks;
        if (ring == 0 or ring % 64 != 0) return error.BadLayout;
        if (drafter) |d| if (d.kv == 0 or d.ring == 0) return error.BadLayout;
        var l: Layout = .{ .g = g, .ring = ring, .drafter = drafter, .block_bytes = spanBytes(block_tokens), .small_bytes = 0, .n = 0, .order = undefined, .lens = undefined, .specs_buf = undefined };
        const all = [_]struct { Section, usize }{
            .{ .rings, G.dsa_layers * 2 * ring * 128 * 2 },
            .{ .rec, G.kda_layers * snapshot.recLayerBytes(g) },
            .{ .conv, G.kda_layers * snapshot.convLayerBytes(g) },
            .{ .draft, if (drafter) |d| d.bytes() else 0 },
        };
        var off: usize = 0;
        for (all) |s| {
            if (s[0] == .draft and drafter == null) continue;
            l.order[l.n] = s[0];
            l.lens[l.n] = s[1];
            l.specs_buf[l.n] = .{ .name = s[0].name(), .len = s[1] };
            l.n += 1;
            off = std.mem.alignForward(usize, off, 256) + s[1]; // (store.keep's layout)
        }
        l.small_bytes = off;
        return l;
    }

    /// The kept-state buffer of a state of `at` tokens (cuda_engine sectionsFor): what a restore allocates.
    pub fn sections(l: *const Layout, at: usize) snapshot.Sections {
        return snapshot.sections(l.g, at, l.ring, if (l.drafter) |d| d.bytes() else 0);
    }

    /// error.OffGrid unless `sec` is a state on the grid; error.LayoutMismatch unless it is this layout's buffer.
    pub fn check(l: *const Layout, sec: snapshot.Sections) !void {
        if (sec.at == 0 or sec.at % grid != 0) return error.OffGrid;
        if (!std.meta.eql(sec, l.sections(sec.at))) return error.LayoutMismatch;
    }

    /// A point's blocks: ceil(at / 2048), the last a tail unless `at` ends a block.
    pub fn blocks(_: *const Layout, at: usize) !usize {
        if (at == 0 or at % grid != 0) return error.OffGrid;
        return std.math.divCeil(usize, at, block_tokens) catch unreachable;
    }

    /// Block `k` of a point at `at`.
    pub fn span(_: *const Layout, at: usize, k: usize) Span {
        const first = k * block_tokens;
        return .{ .first = first, .tokens = @min(block_tokens, at - first) };
    }

    /// The bytes of a block of `tokens` (a multiple of 4: whole pools).
    pub fn spanBytes(tokens: usize) usize {
        return tokens * token_bytes;
    }

    /// The tail's bytes of a point at `at` (the store's Keep.tail_bytes; 0 when `at` ends a block).
    pub fn tailBytes(_: *const Layout, at: usize) usize {
        return spanBytes(at % block_tokens);
    }

    /// A point's bytes on a rank: its blocks (shared or not) and its sections. error.Overflow for an `at` no context
    /// holds (a parsed manifest's).
    pub fn pointBytes(l: *const Layout, at: usize) !usize {
        const whole = (try l.blocks(at)) - @intFromBool(at % block_tokens != 0);
        return std.math.add(usize, try std.math.add(usize, try std.math.mul(usize, whole, l.block_bytes), l.tailBytes(at)), l.small_bytes);
    }

    /// The sections a keep names (Keep.sections), in the buffer's order.
    pub fn specs(l: *const Layout) []const spill.store.SectionSpec {
        return l.specs_buf[0..l.n];
    }

    /// The store's bounds for a context of `max_tokens`.
    pub fn limits(l: *const Layout, max_tokens: u32) format.Limits {
        return .{ .max_tokens = max_tokens, .block_tokens = @intCast(block_tokens), .block_bytes = l.block_bytes, .max_small_bytes = l.small_bytes };
    }

    /// The bytes encode() writes.
    pub const encoded_bytes = 16 + 4 + 4 + 16 * 4 + 6 * 8 + 4 * 16;

    /// The layout's canonical bytes (the digest's input), little-endian: magic, format version, owner layout
    /// (replicated: KV sharding's owner-sharded store is round 3's), TP, the block and the grid, the planes' shape
    /// (layers, row bytes, pool, index width), the ring, the KDA heads and width, the drafter's heads, rows and
    /// layers, the section count, then token, block, small, rec, conv and draft bytes, and every section's name and
    /// length (an absent draft as zeros). The same on both ranks.
    pub fn encode(l: *const Layout) [encoded_bytes]u8 {
        var e: Enc = .{};
        e.bytes("tf-glm-state\x00\x00\x00\x00");
        e.int(u32, format_version);
        e.int(u32, @backingInt(format.OwnerLayout.replicated));
        const d: Drafter = l.drafter orelse .{ .kv = 0, .ring = 0 };
        const dl: usize = if (l.drafter != null) 5 else 0;
        for ([_]usize{ l.g.world, block_tokens, grid, G.dsa_layers, G.kda_layers, G.latent_row, G.pool_row, G.pool, G.index_dim, l.ring, l.g.lin_heads, G.lin_dim, d.kv, d.ring, dl, l.n }) |v| e.int(u32, @intCast(v));
        for ([_]usize{ token_bytes, l.block_bytes, l.small_bytes, snapshot.recLayerBytes(l.g), snapshot.convLayerBytes(l.g), d.bytes() }) |v| e.int(u64, v);
        for (0..4) |i| {
            var nm: [8]u8 = @splat(0);
            if (i < l.n) @memcpy(nm[0..l.order[i].name().len], l.order[i].name());
            e.bytes(&nm);
            e.int(u64, if (i < l.n) l.lens[i] else 0);
        }
        std.debug.assert(e.at == encoded_bytes);
        return e.buf;
    }

    /// SHA-256 of encode(): the layout's part of the compatibility fingerprint.
    pub fn digest(l: *const Layout) format.Digest {
        var out: format.Digest = undefined;
        std.crypto.hash.sha2.Sha256.hash(&l.encode(), &out, .{});
        return out;
    }

    const Enc = struct {
        buf: [encoded_bytes]u8 = undefined,
        at: usize = 0,

        fn bytes(e: *Enc, b: []const u8) void {
            @memcpy(e.buf[e.at..][0..b.len], b);
            e.at += b.len;
        }

        fn int(e: *Enc, comptime T: type, v: T) void {
            std.mem.writeInt(T, e.buf[e.at..][0..@sizeOf(T)], v, .little);
            e.at += @sizeOf(T);
        }
    };
};

/// How the bytes move, every call in one stream's order: device to host staging (pack), host staging to device
/// (unpack), a device fill. On the GPU cuMemcpyDtoHAsync, cuMemcpyHtoDAsync and cuMemsetD8Async on the engine's stream
/// with pinned staging: after any return (an error included) the caller records its event and waits for it before
/// the staging goes on or back. Host memory in the tests.
pub const Copier = struct {
    ptr: *anyopaque,
    d2h: *const fn (ptr: *anyopaque, dst: []u8, src: u64) anyerror!void,
    h2d: *const fn (ptr: *anyopaque, dst: u64, src: []const u8) anyerror!void,
    fill: *const fn (ptr: *anyopaque, dst: u64, byte: u8, n: usize) anyerror!void,
};

/// A kept state's device buffer and its sections (cuda_engine Snap: `buf.?.ptr`, `sec`).
pub const Kept = struct { buf: u64, sec: snapshot.Sections };

/// The store's source of a kept state's rows: its Snap id (never reused). A kept state's bytes never change, so one
/// generation; what ends them is its free (Guard).
pub fn sourceOf(snap_id: u64) spill.Source {
    return .{ .id = snap_id, .generation = 0 };
}

/// Tokens [first, first + tokens) of kept state `k`, layer by layer (its latent rows, then its pooled rows): from the
/// buffer into `out`, or from `in` into the buffer. 22 copies; a pooled plane's stride is its own at/4 + 1 rows.
fn blockCopies(c: Copier, k: Kept, first: usize, tokens: usize, out: ?[]u8, in: ?[]const u8) !void {
    const s = k.sec;
    const lat = tokens * G.latent_row;
    const pool = tokens / G.pool * G.pool_row;
    var at: usize = 0;
    for (0..G.dsa_layers) |i| {
        const l = k.buf + s.latent + i * s.latentBytes() + first * G.latent_row;
        const p = k.buf + s.pooled + i * s.pooledBytes() + first / G.pool * G.pool_row;
        if (out) |o| {
            try c.d2h(c.ptr, o[at..][0..lat], l);
            try c.d2h(c.ptr, o[at + lat ..][0..pool], p);
        } else {
            try c.h2d(c.ptr, l, in.?[at..][0..lat]);
            try c.h2d(c.ptr, p, in.?[at + lat ..][0..pool]);
        }
        at += lat + pool;
    }
}

/// Block [first, first + tokens) of a state of `sec` staged in `len` bytes: refused before any copy unless it is one
/// of the point's blocks (`span`) and `len` its bytes.
fn checkBlock(l: *const Layout, sec: snapshot.Sections, first: usize, tokens: usize, len: usize) !void {
    try l.check(sec);
    if (first % block_tokens != 0 or first >= sec.at or tokens != @min(block_tokens, sec.at - first)) return error.BadBlock;
    if (len != Layout.spanBytes(tokens)) return error.BadStaging;
}

/// Where each of the layout's sections sits in a small state of `small` bytes the store laid out (`secs`, matched by
/// name), in the layout's order. error.BadSections for a missing, extra, resized or out-of-range one.
fn offsets(l: *const Layout, secs: []const format.Section, small: usize) ![4]usize {
    if (secs.len != l.n) return error.BadSections;
    var out: [4]usize = undefined;
    for (l.order[0..l.n], l.lens[0..l.n], 0..) |s, len, i| {
        const sec = for (secs) |*x| {
            if (std.mem.eql(u8, x.name.slice(), s.name())) break x;
        } else return error.BadSections;
        if (sec.len != len or sec.offset > small or sec.len > small - sec.offset) return error.BadSections;
        out[i] = @intCast(sec.offset);
    }
    return out;
}

fn sectionAt(sec: snapshot.Sections, s: Section) usize {
    return switch (s) {
        .rings => sec.rings,
        .rec => sec.rec,
        .conv => sec.conv,
        .draft => sec.draft,
    };
}

/// Copies out of kept states: a keep's sections and tail (Queued.small, Queued.tail) and the blocks the store asks
/// for between rounds (StageReq). Each call checks everything before its first copy.
pub const Packer = struct {
    layout: *const Layout,
    c: Copier,

    /// Kept state `k`'s tokens [first, first + tokens) (a StageReq's, or the tail's at a keep) into `dst`.
    pub fn packBlock(p: *const Packer, k: Kept, first: usize, tokens: usize, dst: []u8) !void {
        try checkBlock(p.layout, k.sec, first, tokens, dst.len);
        try blockCopies(p.c, k, first, tokens, dst, null);
    }

    /// Kept state `k`'s sections into the small state `dst` at the offsets the store gave them (Queued.sections).
    pub fn packSmall(p: *const Packer, k: Kept, dst: []u8, secs: []const format.Section) !void {
        const l = p.layout;
        try l.check(k.sec);
        const off = try offsets(l, secs, dst.len);
        for (l.order[0..l.n], l.lens[0..l.n], off[0..l.n]) |s, len, o| try p.c.d2h(p.c.ptr, dst[o..][0..len], k.buf + sectionAt(k.sec, s));
    }

    /// Guard.release's `.stage`: kept state `k`'s write (source `src`) gets the rest of its blocks staged now
    /// (Store.guard from token 0, into free write slots, packed here before the buffer goes), or is cancelled when
    /// they do not all fit. `oks[i]`: whether `reqs[i]`'s copies were made. The caller waits for its stream, reports
    /// each (Store.staged(reqs[i].ticket, oks[i])), and only then lets the buffer go.
    pub fn stageRest(p: *const Packer, s: *spill.Store, src: spill.Source, k: Kept, reqs: []spill.StageReq, oks: []bool) spill.store.GuardResult {
        std.debug.assert(oks.len >= reqs.len);
        const r = s.guard(src, 0, reqs);
        for (reqs[0..r.staged], oks[0..r.staged]) |q, *ok| {
            ok.* = true;
            p.packBlock(k, q.first, q.tokens, q.dst) catch {
                ok.* = false;
            };
        }
        return r;
    }
};

/// A restore into a new kept-state buffer (allocated for `layout.sections(at)`, as cuda_engine spillIn's): the
/// blocks in order as the store hands them (readReady), the sections (readSmallView), then `finish`.
pub const Unpacker = struct {
    layout: *const Layout,
    c: Copier,
    kept: Kept,
    next: usize = 0,
    small: bool = false,

    pub fn init(layout: *const Layout, c: Copier, kept: Kept) !Unpacker {
        try layout.check(kept.sec);
        return .{ .layout = layout, .c = c, .kept = kept };
    }

    /// The point's next block (ReadyBlock: `first`, `tokens`, `data`). error.BlockOrder for any other.
    pub fn block(u: *Unpacker, first: usize, tokens: usize, src: []const u8) !void {
        if (first != u.next * block_tokens) return error.BlockOrder;
        try checkBlock(u.layout, u.kept.sec, first, tokens, src.len);
        try blockCopies(u.c, u.kept, first, tokens, null, src);
        u.next += 1;
    }

    /// The point's sections (SmallView: `bytes`, `sections`).
    pub fn smallState(u: *Unpacker, src: []const u8, secs: []const format.Section) !void {
        const l = u.layout;
        const off = try offsets(l, secs, src.len);
        for (l.order[0..l.n], l.lens[0..l.n], off[0..l.n]) |s, len, o| try u.c.h2d(u.c.ptr, u.kept.buf + sectionAt(u.kept.sec, s), src[o..][0..len]);
        u.small = true;
    }

    /// Every block and the sections in: then each layer's pooled row at/4 (no token of the state's) is zeroed.
    pub fn finish(u: *Unpacker) !void {
        const s = u.kept.sec;
        if (u.next != try u.layout.blocks(s.at) or !u.small) return error.Incomplete;
        for (0..G.dsa_layers) |i| try u.c.fill(u.c.ptr, u.kept.buf + s.pooled + i * s.pooledBytes() + s.at / G.pool * G.pool_row, 0, G.pool_row);
    }
};

/// What Guard.release decided for a kept state the prompt cache let go (Tier.release, before the family frees it).
pub const Release = enum {
    /// nothing of its write reads it: freed as usual
    free,
    /// one reference more, until its last block is staged and copied (Guard.settle)
    held,
    /// past the guard's limit: its rest is staged now (Packer.stageRest), or its write cancelled
    stage,
};

/// The release guard (GLM 0110's `release`: rows about to go while their write waits are staged first). A kept state
/// is its own buffer, so where Python copied a released state's remaining blocks to the host at once, the buffer is
/// held instead: one reference more while its write still reads it, so the cache's free drops only the cache's
/// reference and the blocks go out between rounds as before. The held buffers' bytes stay within `limit`; past it, a
/// state's rest is staged at its release. `Snap` has `refs` (std.atomic.Value(u32)) and `sec`, as cuda_engine's.
pub fn Guard(comptime Snap: type) type {
    return struct {
        const Self = @This();
        limit: u64,
        bytes: u64 = 0,
        held: std.ArrayList(*Snap) = .empty,

        /// `snap` leaves the prompt cache; `reading`: its write still reads its buffer (Store.pins from token 0, or
        /// its capture or a block's copy not yet complete).
        pub fn release(g: *Self, gpa: std.mem.Allocator, snap: *Snap, reading: bool) Release {
            if (!reading) return .free;
            if (g.holds(snap)) return .held;
            const n: u64 = snap.sec.total;
            if (g.bytes + n > g.limit) return .stage;
            g.held.append(gpa, snap) catch return .stage;
            _ = snap.refs.fetchAdd(1, .acq_rel);
            g.bytes += n;
            return .held;
        }

        pub fn holds(g: *const Self, snap: *const Snap) bool {
            for (g.held.items) |h| if (h == snap) return true;
            return false;
        }

        /// Between rounds, after the stage: every held state nothing reads any more (`idle`) let go (`drop`: the
        /// family's free, which frees the buffer with its last reference). How many went.
        pub fn settle(g: *Self, ctx: *anyopaque, idle: *const fn (ctx: *anyopaque, snap: *Snap) bool, drop: *const fn (ctx: *anyopaque, snap: *Snap) void) usize {
            var n: usize = 0;
            var i: usize = 0;
            while (i < g.held.items.len) {
                const s = g.held.items[i];
                if (!idle(ctx, s)) {
                    i += 1;
                    continue;
                }
                _ = g.held.orderedRemove(i);
                g.bytes -= s.sec.total;
                drop(ctx, s);
                n += 1;
            }
            return n;
        }

        /// At a stop, after the tier's flush: every held state let go.
        pub fn deinit(g: *Self, gpa: std.mem.Allocator, ctx: *anyopaque, drop: *const fn (ctx: *anyopaque, snap: *Snap) void) void {
            for (g.held.items) |s| drop(ctx, s);
            g.held.deinit(gpa);
            g.bytes = 0;
        }
    };
}

// -- tests: host memory stands in for the device ---------------------------------------------------------------------

const testing = std.testing;

const HostCopy = struct {
    copies: usize = 0,
    fills: usize = 0,

    fn d2h(ptr: *anyopaque, dst: []u8, src: u64) anyerror!void {
        const h: *HostCopy = @ptrCast(@alignCast(ptr));
        h.copies += 1;
        const s: [*]const u8 = @ptrFromInt(src);
        @memcpy(dst, s[0..dst.len]);
    }
    fn h2d(ptr: *anyopaque, dst: u64, src: []const u8) anyerror!void {
        const h: *HostCopy = @ptrCast(@alignCast(ptr));
        h.copies += 1;
        const d: [*]u8 = @ptrFromInt(dst);
        @memcpy(d[0..src.len], src);
    }
    fn fill(ptr: *anyopaque, dst: u64, byte: u8, n: usize) anyerror!void {
        const h: *HostCopy = @ptrCast(@alignCast(ptr));
        h.fills += 1;
        const d: [*]u8 = @ptrFromInt(dst);
        @memset(d[0..n], byte);
    }
    fn copier(h: *HostCopy) Copier {
        return .{ .ptr = h, .d2h = d2h, .h2d = h2d, .fill = fill };
    }
    // cuda_snapshot's save and restore over host memory
    fn snapCopy(_: *anyopaque, dst: u64, src: u64, n: usize) anyerror!void {
        const d: [*]u8 = @ptrFromInt(dst);
        const s: [*]const u8 = @ptrFromInt(src);
        @memcpy(d[0..n], s[0..n]);
    }
    fn snapCopier() snapshot.Copier {
        return .{ .ptr = undefined, .copy = snapCopy };
    }
};

/// Bytes that depend only on `seed` and their position `at` (a multiple of 8, as `out.len`): the same computation
/// writes the same bytes at the same rows.
fn pattern(out: []u8, seed: u64, at: usize) void {
    std.debug.assert(out.len % 8 == 0 and at % 8 == 0);
    for (std.mem.bytesAsSlice(u64, out), 0..) |*w, j| {
        var x = seed *% 0x9e3779b97f4a7c15 +% @as(u64, at / 8 + j) *% 0xbf58476d1ce4e5b9;
        x ^= x >> 31;
        x *%= 0x94d049bb133111eb;
        w.* = x ^ (x >> 29);
    }
}

/// One sequence's live state on host memory (cuda_state's views over plain arrays, every recurrence at parity 0).
/// `run(seed, from, to)` is a prefill of rows [from, to) by computation `seed`: their latent rows and complete pools,
/// stale bytes in the pooled row past them (no pool there yet), and the fixed state as it stands at `to`.
const Live = struct {
    gpa: std.mem.Allocator,
    g: G,
    rec: []u8,
    conv: []u8,
    latent: [11][]u8,
    pooled: [11][]u8,
    keys: [11][]u8,
    gates: [11][]u8,
    kc: [5][]u8,
    vc: [5][]u8,
    drafter: ?Drafter,
    cap: usize,
    st: state.State,

    fn init(gpa: std.mem.Allocator, rows: usize, ring: usize, drafter: ?Drafter) !Live {
        var f: Live = undefined;
        f.gpa = gpa;
        f.g = try G.init(2, 0);
        f.drafter = drafter;
        f.rec = try gpa.alloc(u8, 34 * snapshot.recLayerBytes(f.g));
        f.conv = try gpa.alloc(u8, 34 * snapshot.convLayerBytes(f.g));
        for (0..11) |i| {
            f.latent[i] = try gpa.alloc(u8, rows * G.latent_row);
            f.pooled[i] = try gpa.alloc(u8, (rows / 4 + 2) * G.pool_row);
            f.keys[i] = try gpa.alloc(u8, ring * 256);
            f.gates[i] = try gpa.alloc(u8, ring * 256);
            @memset(f.pooled[i], 0);
        }
        const d = drafter orelse Drafter{ .kv = 0, .ring = 0 };
        f.cap = d.ring + 64;
        for (0..5) |i| {
            f.kc[i] = try gpa.alloc(u8, d.kv * f.cap * 256);
            f.vc[i] = try gpa.alloc(u8, d.kv * f.cap * 256);
        }
        var cache: state.CacheView = .{ .latent = undefined, .pooled = undefined, .keys = undefined, .gates = undefined, .ring_base = 0, .capacity = rows };
        for (0..11) |i| {
            cache.latent[i] = @intFromPtr(f.latent[i].ptr);
            cache.pooled[i] = @intFromPtr(f.pooled[i].ptr);
            cache.keys[i] = @intFromPtr(f.keys[i].ptr);
            cache.gates[i] = @intFromPtr(f.gates[i].ptr);
        }
        f.st = .{ .slot = 0, .base = 0, .capacity = rows, .fixed = undefined, .cache = cache };
        f.st.fixed.rec = @intFromPtr(f.rec.ptr);
        f.st.fixed.conv = @intFromPtr(f.conv.ptr);
        return f;
    }

    fn deinit(f: *Live) void {
        f.gpa.free(f.rec);
        f.gpa.free(f.conv);
        for (0..11) |i| {
            f.gpa.free(f.latent[i]);
            f.gpa.free(f.pooled[i]);
            f.gpa.free(f.keys[i]);
            f.gpa.free(f.gates[i]);
        }
        for (0..5) |i| {
            f.gpa.free(f.kc[i]);
            f.gpa.free(f.vc[i]);
        }
    }

    fn draft(f: *const Live) ?snapshot.Draft {
        const d = f.drafter orelse return null;
        var x: snapshot.Draft = .{ .kc = undefined, .vc = undefined, .kv = d.kv, .ring = d.ring, .cap = f.cap, .slot = 0 };
        for (0..5) |i| {
            x.kc[i] = @intFromPtr(f.kc[i].ptr);
            x.vc[i] = @intFromPtr(f.vc[i].ptr);
        }
        return x;
    }

    fn run(f: *Live, seed: u64, from: usize, to: usize) void {
        const t: u64 = to;
        for (0..11) |i| {
            const s = seed *% 31 +% @as(u64, i);
            pattern(f.latent[i][from * G.latent_row .. to * G.latent_row], s, from * G.latent_row);
            pattern(f.pooled[i][from / 4 * G.pool_row .. to / 4 * G.pool_row], s ^ 0x55, from / 4 * G.pool_row);
            pattern(f.pooled[i][to / 4 * G.pool_row ..][0..G.pool_row], s ^ 0xbad ^ t, 0);
            pattern(f.keys[i], s ^ t, 0);
            pattern(f.gates[i], s ^ t ^ 1, 0);
        }
        pattern(f.rec, seed ^ t, 0);
        pattern(f.conv, seed ^ t ^ 2, 0);
        for (0..5) |i| {
            pattern(f.kc[i], seed ^ t ^ @as(u64, 3 + i), 0);
            pattern(f.vc[i], seed ^ t ^ @as(u64, 9 + i), 0);
        }
        f.st.pos = to;
    }

    /// The state as it stands, saved into a new buffer by cuda_snapshot (takeSnap's copy).
    fn keep(f: *Live, l: *const Layout) !TKept {
        const sec = l.sections(f.st.pos);
        const buf = try f.gpa.alloc(u8, sec.total);
        @memset(buf, 0xee);
        try snapshot.save(HostCopy.snapCopier(), f.g, sec, @intFromPtr(buf.ptr), &f.st, f.draft());
        return .{ .buf = buf, .k = .{ .buf = @intFromPtr(buf.ptr), .sec = sec } };
    }

    /// Block [first, first + tokens) straight from the live planes (a replica of the order, not the packer).
    fn expectBlock(f: *const Live, gpa: std.mem.Allocator, first: usize, tokens: usize) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        for (0..11) |i| {
            try out.appendSlice(gpa, f.latent[i][first * G.latent_row ..][0 .. tokens * G.latent_row]);
            try out.appendSlice(gpa, f.pooled[i][first / 4 * G.pool_row ..][0 .. tokens / 4 * G.pool_row]);
        }
        return out.toOwnedSlice(gpa);
    }
};

const TKept = struct {
    buf: []u8,
    k: Kept,

    fn deinit(t: *TKept, gpa: std.mem.Allocator) void {
        gpa.free(t.buf);
    }
};

/// The sections as store.keep lays them out (each on a 256-byte offset).
fn storeSections(l: *const Layout, out: []format.Section) []format.Section {
    var off: u64 = 0;
    for (l.specs(), out[0..l.n]) |s, *x| {
        off = std.mem.alignForward(u64, off, 256);
        x.* = .{ .name = format.Name.of(s.name) catch unreachable, .offset = off, .len = s.len, .crc = 0 };
        off += s.len;
    }
    return out[0..l.n];
}

/// `got` (an unpacked state) holds `want`'s state: its latent rows, pooled rows [0, at/4) of each layer, zeros in
/// pooled row at/4 (where `want` holds whatever the pool held there), and every section.
fn expectState(l: *const Layout, want: Kept, got: Kept) !void {
    try testing.expect(std.meta.eql(want.sec, got.sec));
    const s = want.sec;
    const a: [*]const u8 = @ptrFromInt(want.buf);
    const b: [*]const u8 = @ptrFromInt(got.buf);
    try testing.expectEqualSlices(u8, a[s.latent..][0 .. G.dsa_layers * s.latentBytes()], b[s.latent..][0 .. G.dsa_layers * s.latentBytes()]);
    for (0..G.dsa_layers) |i| {
        const at = s.pooled + i * s.pooledBytes();
        const rows = s.at / 4 * G.pool_row;
        try testing.expectEqualSlices(u8, a[at..][0..rows], b[at..][0..rows]);
        try testing.expect(std.mem.allEqual(u8, b[at + rows ..][0..G.pool_row], 0));
        try testing.expect(!std.mem.allEqual(u8, a[at + rows ..][0..G.pool_row], 0)); // (the live stale bytes)
    }
    for (l.order[0..l.n], l.lens[0..l.n]) |x, len| try testing.expectEqualSlices(u8, a[sectionAt(s, x)..][0..len], b[sectionAt(s, x)..][0..len]);
}

test "the TP=2 layout: 6,204 B a token, 12,705,792 B a block, the sections, tails on the 64-token grid" {
    const g = try G.init(2, 0);
    const l = try Layout.init(g, 2112, .{ .kv = 4, .ring = 2176 });
    try testing.expectEqual(@as(usize, 6204), token_bytes);
    try testing.expectEqual(@as(usize, 12_705_792), l.block_bytes);
    try testing.expectEqual(@as(usize, 0), l.block_bytes % 4096); // 3,102 pages: a block slot has no padding
    try testing.expectEqual(@as(usize, 11 * (2048 * 528 + 512 * 144)), l.block_bytes);
    // the sections, in the buffer's order: 22 rings of 2,112 rows, 34 recurrences of 32 heads, 34 conv windows,
    // DFlash2's 4 kv heads a rank
    try testing.expectEqual(@as(usize, 4), l.n);
    const want = [_]struct { []const u8, usize }{ .{ "rings", 11 * 2 * 2112 * 256 }, .{ "rec", 34 * 32 * 128 * 128 * 4 }, .{ "conv", 34 * 3 * 3 * 32 * 128 * 2 }, .{ "draft", 10 * 4 * 2176 * 256 } };
    for (l.specs(), want) |s, w| {
        try testing.expectEqualStrings(w[0], s.name);
        try testing.expectEqual(@as(u64, w[1]), s.len);
    }
    try testing.expectEqual(@as(usize, 107_986_944), l.small_bytes); // ~103 MiB a point: Python's ~91 MB (96,092,160 B) and the rings
    const d: snapshot.Draft = .{ .kc = undefined, .vc = undefined, .kv = 4, .ring = 2176, .cap = 0, .slot = 0 };
    try testing.expectEqual(d.bytes(), l.drafter.?.bytes());
    // the buffer is cuda_snapshot's: its sections hold exactly these lengths
    const sec = l.sections(4096 + 64);
    try l.check(sec);
    for (l.order[0..l.n], l.lens[0..l.n]) |s, len| try testing.expect(sectionAt(sec, s) + len <= sec.total);
    try testing.expectEqual(sec.total - sec.draft, std.mem.alignForward(usize, l.lens[3], 256));
    // blocks and tails: every grid point but 1 in 32 ends inside a block
    try testing.expectEqual(@as(usize, 1), try l.blocks(64));
    try testing.expectEqual(@as(usize, 397_056), l.tailBytes(64));
    try testing.expectEqual(@as(usize, 12_308_736), l.tailBytes(1984));
    try testing.expectEqual(@as(usize, 0), l.tailBytes(4096));
    try testing.expectEqual(@as(usize, 3), try l.blocks(4096 + 64));
    try testing.expectEqual(Span{ .first = 4096, .tokens = 64 }, l.span(4096 + 64, 2));
    try testing.expectEqual(Span{ .first = 2048, .tokens = 2048 }, l.span(4096 + 64, 1));
    try testing.expectEqual(@as(usize, 2 * 12_705_792 + 397_056 + 107_986_944), try l.pointBytes(4096 + 64));
    try testing.expectEqual(@as(usize, 64 * 12_705_792 + 107_986_944), try l.pointBytes(131072));
    try testing.expectError(error.OffGrid, l.blocks(0));
    try testing.expectError(error.OffGrid, l.blocks(100));
    try testing.expectError(error.OffGrid, l.pointBytes(4096 + 4));
    // a state off the grid, or of another ring or drafter, is not this layout's buffer
    try testing.expectError(error.OffGrid, l.check(snapshot.sections(g, 100, 2112, d.bytes())));
    try testing.expectError(error.LayoutMismatch, l.check(snapshot.sections(g, 4096, 2176, d.bytes())));
    try testing.expectError(error.LayoutMismatch, l.check(snapshot.sections(g, 4096, 2112, 0)));
    // the store's bounds, and the small state as it lays the sections out
    const lim = l.limits(131072);
    try testing.expectEqual(@as(u64, 12_705_792), lim.block_bytes);
    try testing.expectEqual(@as(u32, 2048), lim.block_tokens);
    try testing.expectEqual(@as(u64, l.small_bytes), lim.max_small_bytes);
    var secs: [4]format.Section = undefined;
    const laid = storeSections(&l, &secs);
    try testing.expectEqual(@as(u64, l.small_bytes), laid[3].offset + laid[3].len);
    // without a drafter: three sections, ~81 MiB
    const n = try Layout.init(g, 2112, null);
    try testing.expectEqual(@as(usize, 3), n.n);
    try testing.expectEqual(@as(usize, 11_894_784 + 71_303_168 + 2_506_752), n.small_bytes);
    try n.check(snapshot.sections(g, 4096, 2112, 0));
}

test "spill state refuses TP other than 2, and malformed rings and drafters" {
    for ([_][2]usize{ .{ 1, 0 }, .{ 3, 1 }, .{ 4, 3 } }) |wr| try testing.expectError(error.SpillNeedsTwoRanks, Layout.init(try G.init(wr[0], wr[1]), 2112, null));
    const g = try G.init(2, 1);
    try testing.expectError(error.BadLayout, Layout.init(g, 0, null));
    try testing.expectError(error.BadLayout, Layout.init(g, 2100, null));
    try testing.expectError(error.BadLayout, Layout.init(g, 2112, .{ .kv = 0, .ring = 2176 }));
    _ = try Layout.init(g, 2112, .{ .kv = 4, .ring = 2176 });
}

test "the layout digest is the same on both ranks and covers the ring, the drafter and the format" {
    const a = try Layout.init(try G.init(2, 0), 2112, .{ .kv = 4, .ring = 2176 });
    const b = try Layout.init(try G.init(2, 1), 2112, .{ .kv = 4, .ring = 2176 });
    try testing.expectEqualSlices(u8, &a.digest(), &b.digest());
    const enc = a.encode();
    try testing.expect(std.mem.startsWith(u8, &enc, "tf-glm-state"));
    try testing.expectEqual(format_version, std.mem.readInt(u32, enc[16..20], .little));
    const others = [_]Layout{
        try Layout.init(try G.init(2, 0), 2176, .{ .kv = 4, .ring = 2176 }), // a lone stream's TF_GLM_PREFILL_ROWS: another ring
        try Layout.init(try G.init(2, 0), 2112, null), // no drafter
        try Layout.init(try G.init(2, 0), 2112, .{ .kv = 4, .ring = 2112 }), // a drafter of another ring
        try Layout.init(try G.init(2, 0), 2112, .{ .kv = 8, .ring = 2176 }), // and of other heads
    };
    for (others) |o| try testing.expect(!std.mem.eql(u8, &a.digest(), &o.digest()));
    // and so are the sizes
    var x = a;
    x.small_bytes += 1;
    try testing.expect(!std.mem.eql(u8, &a.digest(), &x.digest()));
}

test "pack then unpack gives the kept state back, whole blocks, a tail and a one-block point; 22 copies a block" {
    const gpa = testing.allocator;
    const l = try Layout.init(try G.init(2, 0), 128, .{ .kv = 2, .ring = 16 });
    var live = try Live.init(gpa, 4096 + 256, 128, l.drafter);
    defer live.deinit();
    var hc: HostCopy = .{};
    const p: Packer = .{ .layout = &l, .c = hc.copier() };
    for ([_]usize{ 4096 + 192, 4096, 64 }, 0..) |at, seed| {
        live.run(seed + 1, 0, at);
        var kept = try live.keep(&l);
        defer kept.deinit(gpa);
        const nb = try l.blocks(at);
        const packed_blocks = try gpa.alloc([]u8, nb);
        defer gpa.free(packed_blocks);
        for (0..nb) |k| {
            const sp = l.span(at, k);
            packed_blocks[k] = try gpa.alloc(u8, Layout.spanBytes(sp.tokens));
            const before = hc.copies;
            try p.packBlock(kept.k, sp.first, sp.tokens, packed_blocks[k]);
            try testing.expectEqual(@as(usize, 22), hc.copies - before);
            const want = try live.expectBlock(gpa, sp.first, sp.tokens);
            defer gpa.free(want);
            try testing.expectEqualSlices(u8, want, packed_blocks[k]);
        }
        defer for (packed_blocks) |b| gpa.free(b);
        try testing.expectEqual(if (at % 2048 == 0) l.block_bytes else l.tailBytes(at), packed_blocks[nb - 1].len);
        var secs: [4]format.Section = undefined;
        const laid = storeSections(&l, &secs);
        const small = try gpa.alloc(u8, l.small_bytes);
        defer gpa.free(small);
        try p.packSmall(kept.k, small, laid);
        // a new buffer, as a restore allocates it
        const nbuf = try gpa.alloc(u8, l.sections(at).total);
        defer gpa.free(nbuf);
        @memset(nbuf, 0x77);
        var u = try Unpacker.init(&l, hc.copier(), .{ .buf = @intFromPtr(nbuf.ptr), .sec = l.sections(at) });
        for (0..nb) |k| {
            const sp = l.span(at, k);
            try u.block(sp.first, sp.tokens, packed_blocks[k]);
        }
        try testing.expectError(error.Incomplete, u.finish());
        try u.smallState(small, laid);
        try u.finish();
        try expectState(&l, kept.k, u.kept);
        // and cuda_snapshot's restore of it gives a sequence the state's rows, rings, recurrences and drafter rings
        var other = try Live.init(gpa, 4096 + 256, 128, l.drafter);
        defer other.deinit();
        other.run(99, 0, 4096 + 256);
        try snapshot.restore(HostCopy.snapCopier(), live.g, u.kept.sec, u.kept.buf, &other.st, other.draft());
        for (0..11) |i| {
            try testing.expectEqualSlices(u8, live.latent[i][0 .. at * G.latent_row], other.latent[i][0 .. at * G.latent_row]);
            try testing.expectEqualSlices(u8, live.pooled[i][0 .. at / 4 * G.pool_row], other.pooled[i][0 .. at / 4 * G.pool_row]);
            try testing.expectEqualSlices(u8, live.keys[i], other.keys[i]);
            try testing.expectEqualSlices(u8, live.gates[i], other.gates[i]);
        }
        try testing.expectEqualSlices(u8, live.rec, other.rec);
        try testing.expectEqualSlices(u8, live.conv, other.conv);
        for (0..5) |i| for (0..2) |h| {
            try testing.expectEqualSlices(u8, live.kc[i][h * live.cap * 256 ..][0 .. 16 * 256], other.kc[i][h * other.cap * 256 ..][0 .. 16 * 256]);
            try testing.expectEqualSlices(u8, live.vc[i][h * live.cap * 256 ..][0 .. 16 * 256], other.vc[i][h * other.cap * 256 ..][0 .. 16 * 256]);
        };
    }
}

test "packing and unpacking refuse a wrong block, staging, order, section or layout before any copy" {
    const gpa = testing.allocator;
    const l = try Layout.init(try G.init(2, 0), 128, null);
    var live = try Live.init(gpa, 4096, 128, null);
    defer live.deinit();
    live.run(5, 0, 2048 + 128);
    var kept = try live.keep(&l);
    defer kept.deinit(gpa);
    var hc: HostCopy = .{};
    const p: Packer = .{ .layout = &l, .c = hc.copier() };
    const big = try gpa.alloc(u8, l.block_bytes);
    defer gpa.free(big);
    try testing.expectError(error.BadBlock, p.packBlock(kept.k, 64, 2048, big)); // not a block's start
    try testing.expectError(error.BadBlock, p.packBlock(kept.k, 2048, 2048, big)); // the tail is 128 tokens
    try testing.expectError(error.BadBlock, p.packBlock(kept.k, 4096, 128, big)); // past the point
    try testing.expectError(error.BadStaging, p.packBlock(kept.k, 0, 2048, big[1..]));
    try testing.expectError(error.BadStaging, p.packBlock(kept.k, 2048, 128, big));
    // a buffer of another layout (a drafter's sections) or off the grid
    const ld = try Layout.init(try G.init(2, 0), 128, .{ .kv = 2, .ring = 16 });
    const pd: Packer = .{ .layout = &ld, .c = hc.copier() };
    try testing.expectError(error.LayoutMismatch, pd.packBlock(kept.k, 0, 2048, big));
    var off = kept.k;
    off.sec.at = 2048 + 100;
    try testing.expectError(error.OffGrid, p.packBlock(off, 0, 2048, big));
    // sections: missing, resized, renamed, out of the slot
    var secs: [4]format.Section = undefined;
    const laid = storeSections(&l, &secs);
    const small = try gpa.alloc(u8, l.small_bytes);
    defer gpa.free(small);
    try testing.expectError(error.BadSections, p.packSmall(kept.k, small, laid[0..2]));
    var bad: [3]format.Section = laid[0..3].*;
    bad[1].len -= 1;
    try testing.expectError(error.BadSections, p.packSmall(kept.k, small, &bad));
    bad = laid[0..3].*;
    bad[2].name = try format.Name.of("rings");
    try testing.expectError(error.BadSections, p.packSmall(kept.k, small, &bad));
    try testing.expectError(error.BadSections, p.packSmall(kept.k, small[0 .. small.len - 1], laid));
    try testing.expectEqual(@as(usize, 0), hc.copies);
    // unpacking: in order only, every block before finish
    const nbuf = try gpa.alloc(u8, kept.k.sec.total);
    defer gpa.free(nbuf);
    var u = try Unpacker.init(&l, hc.copier(), .{ .buf = @intFromPtr(nbuf.ptr), .sec = kept.k.sec });
    const tail = try gpa.alloc(u8, l.tailBytes(2048 + 128));
    defer gpa.free(tail);
    try testing.expectError(error.BlockOrder, u.block(2048, 128, tail));
    try testing.expectError(error.BadStaging, u.block(0, 2048, tail));
    try testing.expectEqual(@as(usize, 0), hc.copies);
    try p.packBlock(kept.k, 0, 2048, big);
    try p.packBlock(kept.k, 2048, 128, tail);
    try p.packSmall(kept.k, small, laid);
    try u.block(0, 2048, big);
    try testing.expectError(error.BlockOrder, u.block(0, 2048, big));
    try u.block(2048, 128, tail);
    try testing.expectError(error.Incomplete, u.finish());
    try testing.expectError(error.BadSections, u.smallState(small, laid[1..]));
    try u.smallState(small, laid);
    try u.finish();
    try testing.expectEqual(@as(usize, 11), hc.fills);
    try expectState(&l, kept.k, u.kept);
    try testing.expectError(error.OffGrid, Unpacker.init(&l, hc.copier(), off));
}

test "two turns of a conversation and a fork share their earlier blocks: equal names, equal bytes" {
    const gpa = testing.allocator;
    const l = try Layout.init(try G.init(2, 0), 128, null);
    var hc: HostCopy = .{};
    const p: Packer = .{ .layout = &l, .c = hc.copier() };
    var live = try Live.init(gpa, 4608 + 64, 128, null);
    defer live.deinit();
    // turn 1 at 3,072 (a whole block and a 1,024-token tail); turn 2 goes on to 4,608 in the same computation
    live.run(1, 0, 3072);
    var t1 = try live.keep(&l);
    defer t1.deinit(gpa);
    live.run(1, 3072, 4608);
    var t2 = try live.keep(&l);
    defer t2.deinit(gpa);
    // a fork: turn 1 resumed (its kept state restored) and another reply computed from 3,072 to 4,096
    try snapshot.restore(HostCopy.snapCopier(), live.g, t1.k.sec, t1.k.buf, &live.st, null);
    live.run(2, 3072, 4096);
    var fk = try live.keep(&l);
    defer fk.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(42);
    var ids: [4608]u32 = undefined;
    for (&ids) |*x| x.* = prng.random().intRangeLessThan(u32, 0, 154880);
    var ids_f: [4096]u32 = ids[0..4096].*;
    for (ids_f[3072..]) |*x| x.* = prng.random().intRangeLessThan(u32, 0, 154880);
    const l1 = format.Lineage.fresh(@splat(7), 1);
    var lf = try l1.fork(gpa, @splat(9), 3072);
    defer lf.deinit(gpa);
    const names = struct {
        fn of(a: std.mem.Allocator, lin: format.Lineage, x: []const u32, out: []format.Span) !void {
            var ch: std.ArrayList(format.Digest) = .empty;
            defer ch.deinit(a);
            try format.chain(a, x, block_tokens, &ch);
            format.blockNames(lin, ch.items, x, block_tokens, out);
        }
    };
    var n1: [2]format.Span = undefined;
    var n2: [3]format.Span = undefined;
    var nf: [2]format.Span = undefined;
    var nsame: [2]format.Span = undefined;
    try names.of(gpa, l1, ids[0..3072], &n1);
    try names.of(gpa, l1, &ids, &n2);
    try names.of(gpa, lf, &ids_f, &nf);
    try names.of(gpa, lf, ids[0..4096], &nsame);
    try testing.expectEqualSlices(u8, &n1[0].name, &n2[0].name);
    try testing.expectEqualSlices(u8, &n1[0].name, &nf[0].name);
    try testing.expect(!std.mem.eql(u8, &n1[1].name, &n2[1].name)); // turn 1's tail is its own
    try testing.expect(!std.mem.eql(u8, &n2[1].name, &nf[1].name));
    try testing.expect(!std.mem.eql(u8, &n2[1].name, &nsame[1].name)); // the fork's own computation, even of the same ids
    // what the names promise: block 0 the same bytes from every one of them
    const b = try gpa.alloc(u8, 3 * l.block_bytes);
    defer gpa.free(b);
    const b1 = b[0..l.block_bytes];
    const b2 = b[l.block_bytes..][0..l.block_bytes];
    const bf = b[2 * l.block_bytes ..][0..l.block_bytes];
    try p.packBlock(t1.k, 0, 2048, b1);
    try p.packBlock(t2.k, 0, 2048, b2);
    try p.packBlock(fk.k, 0, 2048, bf);
    try testing.expectEqualSlices(u8, b1, b2);
    try testing.expectEqualSlices(u8, b1, bf);
    // block 1: turn 2's whole block begins with turn 1's tail, layer by layer; the fork's differs past 3,072
    const tail = try gpa.alloc(u8, l.tailBytes(3072));
    defer gpa.free(tail);
    try p.packBlock(t1.k, 2048, 1024, tail);
    try p.packBlock(t2.k, 2048, 2048, b2);
    try p.packBlock(fk.k, 2048, 2048, bf);
    for (0..11) |i| {
        const tl = tail[i * 1024 * 564 ..];
        const wl = b2[i * 2048 * 564 ..];
        const fl = bf[i * 2048 * 564 ..];
        try testing.expectEqualSlices(u8, tl[0 .. 1024 * 528], wl[0 .. 1024 * 528]);
        try testing.expectEqualSlices(u8, tl[1024 * 528 ..][0 .. 256 * 144], wl[2048 * 528 ..][0 .. 256 * 144]);
        try testing.expectEqualSlices(u8, wl[0 .. 1024 * 528], fl[0 .. 1024 * 528]);
        try testing.expect(!std.mem.eql(u8, wl[1024 * 528 .. 2048 * 528], fl[1024 * 528 .. 2048 * 528]));
    }
}

// -- the real store over a scratch directory (TF_SPILL_TEST_DIR, else /tmp) -------------------------------------------

/// A store sized for this layout over a fresh directory, the engine's side of it played by the test: keeps fill the
/// small state and tail from the kept buffer, `pump` packs every requested block from its source.
const TStore = struct {
    gpa: std.mem.Allocator,
    path: [256]u8 = undefined,
    root: [:0]const u8 = undefined,
    bufs: [4]spill.fsio.Buffer = undefined,
    lists: [4][1]spill.fsio.Buffer = undefined,
    s: *spill.Store = undefined,
    l: *const Layout,
    p: Packer,
    sources: std.AutoHashMapUnmanaged(u64, Kept) = .empty,

    fn init(gpa: std.mem.Allocator, l: *const Layout, hc: *HostCopy, tag: []const u8, write_slots: usize) !*TStore {
        const t = try gpa.create(TStore);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .l = l, .p = .{ .layout = l, .c = hc.copier() } };
        t.root = try spill.fsio.testRoot(&t.path, tag);
        _ = std.c.close(try spill.fsio.freshDir(gpa, t.root));
        const block = std.mem.alignForward(usize, l.block_bytes, 4096);
        const small = std.mem.alignForward(usize, l.small_bytes, 4096);
        const sizes = [4]usize{ write_slots * block, 3 * block, small, small };
        var made: usize = 0;
        errdefer for (t.bufs[0..made]) |b| gpa.free(b);
        for (sizes, 0..) |n, i| {
            t.bufs[i] = try gpa.alignedAlloc(u8, .fromByteUnits(4096), n);
            t.lists[i] = .{t.bufs[i]};
            made += 1;
        }
        t.s = try spill.Store.open(gpa, testing.io, .{
            .root = t.root,
            .prefix = "glm-spill-",
            .rank = 0,
            .compat = format.compatDigest(l.digest(), .replicated, @intCast(block_tokens), l.block_bytes),
            .limits = l.limits(8192),
            .cap_bytes = 4 << 30,
            .floor_bytes = 0,
            .min_tokens = 64,
            .backlog_bytes = 1 << 30,
            .staging = .{ .write = &t.lists[0], .read = &t.lists[1], .small = &t.lists[2], .restore = &t.lists[3] },
            .readers = 2,
            .max_reads = 1,
            .direct = false,
        });
        return t;
    }

    fn deinit(t: *TStore) void {
        t.s.close();
        spill.fsio.dropDir(t.gpa, t.root);
        for (t.bufs) |b| t.gpa.free(b);
        t.sources.deinit(t.gpa);
        t.gpa.destroy(t);
    }

    /// A keep of kept state `k` (source `id`) under `lin`: the sections and tail captured, reported. Its job, or null
    /// when the store wrote nothing.
    fn keep(t: *TStore, ids: []const u32, lin: format.Lineage, id: u64, k: Kept, capture: bool) !?spill.JobId {
        try t.sources.put(t.gpa, id, k);
        const at = k.sec.at;
        const r = t.s.keep(.{ .keyed = ids[0..at], .lineage = lin, .sections = t.l.specs(), .tail_bytes = t.l.tailBytes(at), .source = sourceOf(id) });
        if (r != .queued) return null;
        const q = r.queued;
        try t.p.packSmall(k, q.small, q.sections);
        if (q.tail) |tail| {
            const sp = t.l.span(at, (try t.l.blocks(at)) - 1);
            try t.p.packBlock(k, sp.first, sp.tokens, tail);
        }
        if (capture) t.s.captured(q.job, true);
        return q.job;
    }

    /// Between rounds: every block the store asks for packed from its source and reported.
    fn pump(t: *TStore) !void {
        var reqs: [4]spill.StageReq = undefined;
        while (true) {
            const n = t.s.stage(&reqs);
            if (n == 0) return;
            for (reqs[0..n]) |q| {
                const k = t.sources.get(q.source.id) orelse return error.TestUnexpectedResult;
                t.p.packBlock(k, q.first, q.tokens, q.dst) catch {
                    t.s.staged(q.ticket, false);
                    continue;
                };
                t.s.staged(q.ticket, true);
            }
        }
    }

    /// Pumps until source `id`'s write reads none of its rows (write slots come back as the writer writes).
    fn stageAll(t: *TStore, id: u64) !void {
        var i: usize = 0;
        while (i < 5000 and t.s.pins(sourceOf(id), 0)) : (i += 1) {
            try t.pump();
            std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        try testing.expect(!t.s.pins(sourceOf(id), 0));
    }

    /// Waits until the writer has written `n` blocks in all (their slots back).
    fn waitWritten(t: *TStore, n: u64) !void {
        var i: usize = 0;
        while (i < 5000 and t.s.counts().written_blocks < n) : (i += 1) std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
        try testing.expectEqual(n, t.s.counts().written_blocks);
    }

    fn settle(t: *TStore) !void {
        var i: usize = 0;
        while (i < 5000 and t.s.pendingStage() > 0) : (i += 1) {
            try t.pump();
            std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        try testing.expect(t.s.drain(10 * std.time.ns_per_s).done);
    }

    /// Point `ids[0..at]` read back into a new buffer (gpa-owned).
    fn restore(t: *TStore, hc: *HostCopy, ids: []const u32, at: usize) ![]u8 {
        const key = format.pointKey(ids[0..at], @intCast(block_tokens), &.{});
        const id = try t.s.beginRead(key);
        const sec = t.l.sections(at);
        const buf = try t.gpa.alloc(u8, sec.total);
        errdefer t.gpa.free(buf);
        @memset(buf, 0x33);
        var u = try Unpacker.init(t.l, hc.copier(), .{ .buf = @intFromPtr(buf.ptr), .sec = sec });
        const total = try t.l.blocks(at);
        while (u.next < total) {
            try testing.expect(t.s.readAwait(id, @intCast(u.next), 10 * std.time.ns_per_s, .{}));
            var rb: [2]spill.ReadyBlock = undefined;
            const got = t.s.readReady(id, null, &rb);
            try testing.expect(got.status != .failed);
            for (rb[0..got.n]) |b| {
                try u.block(b.first, b.tokens, b.data);
                t.s.readRelease(b.slot);
            }
        }
        var view: ?spill.store.SmallView = null;
        for (0..5000) |_| {
            view = t.s.readSmallView(id);
            if (view != null) break;
            std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        try u.smallState(view.?.bytes, view.?.sections);
        var done = (try t.s.endRead(id, .commit)).?;
        done.lineage.deinit(t.gpa);
        try u.finish();
        return buf;
    }
};

test "the store: two turns and a fork written, their shared blocks once, each read back into the state it kept" {
    const gpa = testing.allocator;
    const l = try Layout.init(try G.init(2, 0), 128, null);
    var hc: HostCopy = .{};
    var t = try TStore.init(gpa, &l, &hc, "glm-state-turns", 3);
    defer t.deinit();
    var live = try Live.init(gpa, 4608 + 64, 128, null);
    defer live.deinit();
    var prng = std.Random.DefaultPrng.init(3);
    var ids: [4608]u32 = undefined;
    for (&ids) |*x| x.* = prng.random().intRangeLessThan(u32, 0, 154880);
    var ids_f: [4096]u32 = ids[0..4096].*;
    for (ids_f[3072..]) |*x| x.* = prng.random().intRangeLessThan(u32, 0, 154880);
    const l1 = format.Lineage.fresh(@splat(1), 1);
    var lf = try l1.fork(gpa, @splat(2), 3072);
    defer lf.deinit(gpa);
    live.run(1, 0, 3072);
    var t1 = try live.keep(&l);
    defer t1.deinit(gpa);
    live.run(1, 3072, 4608);
    var t2 = try live.keep(&l);
    defer t2.deinit(gpa);
    try snapshot.restore(HostCopy.snapCopier(), live.g, t1.k.sec, t1.k.buf, &live.st, null);
    live.run(2, 3072, 4096);
    var fk = try live.keep(&l);
    defer fk.deinit(gpa);
    // turn 1: block 0 staged, its tail captured at keep
    _ = (try t.keep(&ids, l1, 1, t1.k, true)).?;
    try t.settle();
    try testing.expectEqual(@as(u64, 1), t.s.counts().staged_blocks);
    try testing.expectEqual(@as(u64, 2), t.s.counts().written_blocks);
    // turn 2 names turn 1's block 0: it stages only block 1, its tail at keep
    _ = (try t.keep(&ids, l1, 2, t2.k, true)).?;
    try t.settle();
    try testing.expectEqual(@as(u64, 2), t.s.counts().staged_blocks);
    try testing.expectEqual(@as(u64, 4), t.s.counts().written_blocks);
    // the fork names block 0 too; its block 1 is its own
    _ = (try t.keep(&ids_f, lf, 3, fk.k, true)).?;
    try t.settle();
    try testing.expectEqual(@as(u64, 3), t.s.counts().staged_blocks);
    try testing.expectEqual(@as(u64, 5), t.s.counts().written_blocks);
    try testing.expectEqual(@as(u64, 5), t.s.counts().blocks);
    try testing.expectEqual(@as(u64, 3), t.s.counts().points);
    // a keep of turn 2 again writes nothing
    try testing.expectEqual(@as(?spill.JobId, null), try t.keep(&ids, l1, 4, t2.k, true));
    // each read back is the state it kept (pooled row at/4 zeroed)
    for ([_]struct { []const u32, Kept }{ .{ &ids, t2.k }, .{ &ids_f, fk.k }, .{ &ids, t1.k } }) |c| {
        const buf = try t.restore(&hc, c[0], c[1].sec.at);
        defer gpa.free(buf);
        try expectState(&l, c[1], .{ .buf = @intFromPtr(buf.ptr), .sec = c[1].sec });
    }
}

/// A kept state as the guard sees it (cuda_engine Snap's `refs` and `sec`); `freed` when its last reference went.
const TSnap = struct {
    refs: std.atomic.Value(u32) = .init(1),
    sec: snapshot.Sections,
    id: u64,
    freed: bool = false,

    fn drop(_: *anyopaque, s: *TSnap) void {
        if (s.refs.fetchSub(1, .acq_rel) == 1) s.freed = true;
    }
    fn idle(ctx: *anyopaque, s: *TSnap) bool {
        const st: *spill.Store = @ptrCast(@alignCast(ctx));
        return !st.pins(sourceOf(s.id), 0);
    }
};

test "the release guard holds a released state while its write reads it, and past its limit stages the rest or cancels" {
    const gpa = testing.allocator;
    const l = try Layout.init(try G.init(2, 0), 128, null);
    var hc: HostCopy = .{};
    var t = try TStore.init(gpa, &l, &hc, "glm-state-guard", 3);
    defer t.deinit();
    var live = try Live.init(gpa, 6208 + 64, 128, null);
    defer live.deinit();
    var prng = std.Random.DefaultPrng.init(4);
    var ids: [6208]u32 = undefined;
    for (&ids) |*x| x.* = prng.random().intRangeLessThan(u32, 0, 154880);
    const ctx: *anyopaque = t.s;
    // held: the cache's free leaves the buffer to the guard, which lets it go once its blocks are staged
    {
        live.run(1, 0, 4608);
        var a = try live.keep(&l);
        defer a.deinit(gpa);
        var sa: TSnap = .{ .sec = a.k.sec, .id = 11 };
        var g: Guard(TSnap) = .{ .limit = 1 << 30 };
        defer g.deinit(gpa, ctx, TSnap.drop);
        _ = (try t.keep(&ids, format.Lineage.fresh(@splat(3), 1), sa.id, a.k, true)).?;
        try testing.expect(t.s.pins(sourceOf(sa.id), 0));
        try testing.expectEqual(Release.held, g.release(gpa, &sa, t.s.pins(sourceOf(sa.id), 0)));
        try testing.expectEqual(Release.held, g.release(gpa, &sa, true)); // (once)
        try testing.expectEqual(@as(u32, 2), sa.refs.load(.acquire));
        try testing.expectEqual(@as(u64, a.k.sec.total), g.bytes);
        TSnap.drop(ctx, &sa); // the cache's free
        try testing.expect(!sa.freed);
        try testing.expectEqual(@as(usize, 0), g.settle(ctx, TSnap.idle, TSnap.drop)); // still read
        try t.stageAll(sa.id);
        try testing.expectEqual(@as(usize, 1), g.settle(ctx, TSnap.idle, TSnap.drop));
        try testing.expect(sa.freed and g.bytes == 0 and !g.holds(&sa));
        try t.settle();
        const buf = try t.restore(&hc, &ids, 4608);
        defer gpa.free(buf);
        try expectState(&l, a.k, .{ .buf = @intFromPtr(buf.ptr), .sec = a.k.sec });
        // a published state is free at once
        var sb: TSnap = .{ .sec = a.k.sec, .id = 11 };
        try testing.expectEqual(Release.free, g.release(gpa, &sb, t.s.pins(sourceOf(sb.id), 0)));
    }
    // past the limit: the rest staged at the release, then the buffer may go
    {
        live.run(2, 4608, 6208);
        var c = try live.keep(&l); // 3 whole blocks (block 0 and 1 another lineage's: written again) and a tail
        defer c.deinit(gpa);
        var sc: TSnap = .{ .sec = c.k.sec, .id = 12 };
        var g: Guard(TSnap) = .{ .limit = c.k.sec.total - 1 };
        defer g.deinit(gpa, ctx, TSnap.drop);
        const w = t.s.counts().written_blocks;
        _ = (try t.keep(&ids, format.Lineage.fresh(@splat(4), 1), sc.id, c.k, true)).?;
        try t.waitWritten(w + 1); // (its tail: every write slot free again)
        try testing.expectEqual(Release.stage, g.release(gpa, &sc, t.s.pins(sourceOf(sc.id), 0)));
        var reqs: [4]spill.StageReq = undefined;
        var oks: [4]bool = undefined;
        const r = t.p.stageRest(t.s, sourceOf(sc.id), c.k, &reqs, &oks);
        try testing.expectEqual(@as(usize, 3), r.staged);
        for (reqs[0..r.staged], oks[0..r.staged]) |q, ok| t.s.staged(q.ticket, ok);
        try testing.expect(!t.s.pins(sourceOf(sc.id), 0));
        TSnap.drop(ctx, &sc);
        try testing.expect(sc.freed);
        try t.settle();
        const buf = try t.restore(&hc, &ids, 6208);
        defer gpa.free(buf);
        try expectState(&l, c.k, .{ .buf = @intFromPtr(buf.ptr), .sec = c.k.sec });
    }
    // past the limit with too few write slots (its tail holds one until its capture is reported): cancelled
    {
        var other: [6208]u32 = undefined;
        for (&other) |*x| x.* = prng.random().intRangeLessThan(u32, 0, 154880);
        var d = try live.keep(&l);
        defer d.deinit(gpa);
        var sd: TSnap = .{ .sec = d.k.sec, .id = 13 };
        var g: Guard(TSnap) = .{ .limit = 0 };
        defer g.deinit(gpa, ctx, TSnap.drop);
        const job = (try t.keep(&other, format.Lineage.fresh(@splat(5), 1), sd.id, d.k, false)).?;
        try testing.expectEqual(Release.stage, g.release(gpa, &sd, t.s.pins(sourceOf(sd.id), 0)));
        var reqs: [4]spill.StageReq = undefined;
        var oks: [4]bool = undefined;
        const r = t.p.stageRest(t.s, sourceOf(sd.id), d.k, &reqs, &oks);
        try testing.expectEqual(@as(usize, 0), r.staged);
        try testing.expectEqual(@as(usize, 1), r.cancelled);
        t.s.captured(job, true);
        var cs: [4]spill.Completion = undefined;
        var cancelled = false;
        for (0..5000) |_| {
            for (cs[0..t.s.poll(&cs)]) |x| if (x == .cancelled and x.cancelled.job == job) {
                cancelled = true;
            };
            if (cancelled) break;
            std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        try testing.expect(cancelled);
        try testing.expect(!t.s.presence(format.pointKey(&other, @intCast(block_tokens), &.{})));
        try testing.expect(!t.s.pins(sourceOf(sd.id), 0));
    }
}
