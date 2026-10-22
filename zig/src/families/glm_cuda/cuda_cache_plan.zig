//! The recipe's kept-prompt settings and memory plan for the GLM CUDA engine: glm5_next/cuda/engine.py
//! (`kept_entries`, `shared_prefix`, TF_GLM_CACHE_ENTRIES, TF_GLM_CACHE_GIB, TF_GLM_CACHE_SHARE_PCT) and multi.py
//! (`keep_per_chat`, `kept_bytes_cap`), the launch scripts' MEMORY_RESERVE_GIB. Pure parsing and arithmetic; the
//! environment is read by the caller and handed in, so tests need no process state.
//!
//! Credits (the recipe's authors of each rule): the entry count as a share of the pool, diagnosed by jdecker76
//! (https://github.com/jdecker76, issue #84); the kept-state byte and per-conversation caps, by Thomas Wade
//! (https://github.com/ThomasWadeZ, PR #65); the cap's recency order for shared system-prompt states, diagnosed and
//! fixed by meleesciony (https://github.com/meleesciony, issue #75); the pool compaction before eviction (patch 0074),
//! by ezoushen (https://github.com/ezoushen); the spill tier (patch 0088), by Robert Wojciechowski
//! (https://github.com/wojo). TensorFold's prompt cache core is Ash Hart's and its contributors'.
//!
//! The values mean here what they meant on the Python engine, with the one difference the engines have: the Python
//! pool holds a kept state's attention rows in place and counts only its fixed state against the entry reserve; the
//! Zig engine copies a state's rows and rings into a buffer of its own, so the budget and the byte cap bound the
//! whole of it. No compaction before eviction (patch 0074) is needed: a state frees exactly what it holds.
const std = @import("std");

pub const gib: f64 = 1 << 30;

/// The recipe's kept-state settings, as the environment gives them (null: unset).
pub const Env = struct {
    entries: ?[]const u8 = null, // TF_GLM_CACHE_ENTRIES (8; the launch scripts set 32)
    share_pct: ?[]const u8 = null, // TF_GLM_CACHE_SHARE_PCT (0: off)
    kept_bytes_gib: ?[]const u8 = null, // TF_GLM_KEPT_BYTES_GIB (0: no cap)
    per_chat: ?[]const u8 = null, // TF_GLM_KEEP_PER_CHAT (0: no limit)
    shared_prefix: ?[]const u8 = null, // TF_GLM_SHARED_PREFIX (0 off; 1 or on: 256; or 16 and more)
    cache_gib: ?[]const u8 = null, // TF_GLM_CACHE_GIB (3; the launch scripts' KV_POOL_GIB)
    reserve_gib: ?[]const u8 = null, // TENSORFOLD_MEMORY_RESERVE_GIB (MEMORY_RESERVE_GIB)
    headroom_gib: ?[]const u8 = null, // TF_GLM_KEEP_HEADROOM_GIB (2): what the budget leaves free above the reserve, for the prompts' own buffers
};

pub const shared_default: u32 = 256;
pub const shared_min: u32 = 16;
/// Kept states never take more than this share of the budget by their count (engine.kept_entries).
pub const ceiling_pct: f64 = 75.0;

pub const Error = error{ BadEntries, BadSharePct, BadKeptBytes, BadPerChat, BadSharedPrefix, BadCacheGib, BadReserve, BadHeadroom };

fn trimmed(v: ?[]const u8) []const u8 {
    return std.mem.trim(u8, v orelse "", " \t\r\n");
}

fn decimal(text: []const u8) ?u32 {
    if (text.len == 0) return null;
    for (text) |c| if (c < '0' or c > '9') return null;
    return std.fmt.parseInt(u32, text, 10) catch null;
}

/// TF_GLM_SHARED_PREFIX: 0 (off, the default), or the fewest tokens past a request's resume point a shared-prefix state
/// adds (1 or on: 256; else 16 or more).
pub fn sharedPrefix(v: ?[]const u8) Error!u32 {
    const raw = trimmed(v);
    if (raw.len > 16) return error.BadSharedPrefix;
    var buf: [16]u8 = undefined;
    const t = std.ascii.lowerString(&buf, raw);
    if (t.len == 0 or std.mem.eql(u8, t, "0") or std.mem.eql(u8, t, "off")) return 0;
    if (std.mem.eql(u8, t, "1") or std.mem.eql(u8, t, "on")) return shared_default;
    const n = decimal(t) orelse return error.BadSharedPrefix;
    if (n < shared_min) return error.BadSharedPrefix;
    return n;
}

/// TF_GLM_KEEP_PER_CHAT: 0 (no limit, the default) or a count.
pub fn keepPerChat(v: ?[]const u8) Error!u32 {
    const t = trimmed(v);
    if (t.len == 0) return 0;
    return decimal(t) orelse error.BadPerChat;
}

/// TF_GLM_KEPT_BYTES_GIB in bytes: 0 (the default) or a non-positive value means no cap.
pub fn keptBytesCap(v: ?[]const u8) Error!u64 {
    const t = trimmed(v);
    if (t.len == 0) return 0;
    const g = std.fmt.parseFloat(f64, t) catch return error.BadKeptBytes;
    if (!std.math.isFinite(g)) return error.BadKeptBytes;
    return if (g <= 0) 0 else @intFromFloat(g * gib);
}

/// TF_GLM_KEEP_HEADROOM_GIB in bytes: what the kept states' budget leaves free above the memory reserve, for the prompts'
/// own buffers as they grow (2 GiB by default; 0: none). The budget stops that far above the reserve; the saves stop at
/// the reserve itself, and a keep that would cross it is refused on every rank before the ranks agree to it (cuda_lanes
/// pass), never a failed prefill.
pub fn keepHeadroom(v: ?[]const u8) Error!u64 {
    const t = trimmed(v);
    if (t.len == 0) return 2 << 30;
    const g = std.fmt.parseFloat(f64, t) catch return error.BadHeadroom;
    if (!std.math.isFinite(g) or g < 0) return error.BadHeadroom;
    return @intFromFloat(@min(g * gib, 1e18));
}

/// TF_GLM_CACHE_ENTRIES (at least 1; default 8).
pub fn cacheEntries(v: ?[]const u8) Error!u32 {
    const t = trimmed(v);
    if (t.len == 0) return 8;
    return decimal(t) orelse error.BadEntries;
}

fn sharePct(v: ?[]const u8) Error!f64 {
    const t = trimmed(v);
    if (t.len == 0) return 0;
    const g = std.fmt.parseFloat(f64, t) catch return error.BadSharePct;
    if (!std.math.isFinite(g) or g < 0) return error.BadSharePct;
    return g;
}

pub const Entries = struct { count: u32, capped: bool, share: bool };

/// engine.kept_entries: how many kept states the budget holds. `share_pct` (0: off) makes the count at least that share of
/// the budget in states of `state` bytes each; past three quarters of the budget the count is cut.
pub fn keptEntries(configured: u32, share_pct: f64, budget: u64, state: u64) Entries {
    const st: f64 = @floatFromInt(@max(1, state));
    const b: f64 = @floatFromInt(budget);
    var n: u32 = @max(1, configured);
    var share = false;
    if (share_pct > 0) {
        const wanted: u32 = @intFromFloat(@min(b * @min(share_pct, ceiling_pct) / 100.0 / st, 4e9));
        n = @max(n, wanted);
        share = true;
    }
    const ceiling: u32 = @max(1, @as(u32, @intFromFloat(@min(b * 0.75 / st, 4e9))));
    if (n > ceiling) return .{ .count = ceiling, .capped = true, .share = share };
    return .{ .count = n, .capped = false, .share = share };
}

/// What memory the kept states may take: TF_GLM_CACHE_GIB (or --prompt-cache-gib), less what the machine leaves after
/// the reserve (MemAvailable now minus MEMORY_RESERVE_GIB, less the keep headroom: `plan` passes reserve + headroom).
pub fn budgetBytes(asked_gib: ?f64, env_gib: ?[]const u8, available: u64, reserve: u64) Error!u64 {
    const wanted: f64 = if (asked_gib) |g| g else blk: {
        const t = trimmed(env_gib);
        if (t.len == 0) break :blk 3.0;
        const g = std.fmt.parseFloat(f64, t) catch return error.BadCacheGib;
        if (!std.math.isFinite(g)) return error.BadCacheGib;
        break :blk g;
    };
    if (wanted <= 0) return 0;
    const bytes: u64 = @intFromFloat(@min(wanted * gib, 1e18));
    return @min(bytes, available -| reserve);
}

/// TENSORFOLD_MEMORY_RESERVE_GIB in bytes, else a tenth of the machine's memory but at least 4 GiB (unified memory is the
/// OS's too: native/cuda_memory.zig reserveBytes).
pub fn reserveBytes(v: ?[]const u8, total: u64) Error!u64 {
    const t = trimmed(v);
    if (t.len == 0) return @max(@as(u64, 4 << 30), total / 10);
    const g = std.fmt.parseFloat(f64, t) catch return error.BadReserve;
    if (!(g >= 2) or !std.math.isFinite(g)) return error.BadReserve;
    const bytes: u64 = @intFromFloat(@min(g * gib, 1e18));
    return if (bytes <= total) bytes else error.BadReserve;
}

/// /proc/meminfo's MemTotal and MemAvailable in bytes.
pub const MemInfo = struct { total: u64, available: u64 };

pub fn meminfo(text: []const u8) ?MemInfo {
    var total: ?u64 = null;
    var available: ?u64 = null;
    var rows = std.mem.tokenizeScalar(u8, text, '\n');
    while (rows.next()) |row| {
        var words = std.mem.tokenizeAny(u8, row, ": \t");
        const key = words.next() orelse continue;
        const is_total = std.mem.eql(u8, key, "MemTotal");
        if (!is_total and !std.mem.eql(u8, key, "MemAvailable")) continue;
        const kib = std.fmt.parseInt(u64, words.next() orelse return null, 10) catch return null;
        const bytes = std.math.mul(u64, kib, 1024) catch return null;
        if (is_total) total = bytes else available = bytes;
    }
    return .{ .total = total orelse return null, .available = available orelse return null };
}

/// MemTotal and MemAvailable now (null where /proc/meminfo cannot be read).
pub fn memNow() ?MemInfo {
    const fd = std.c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var buf: [4096]u8 = undefined;
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return null;
    return meminfo(buf[0..@intCast(n)]);
}

/// The whole plan.
pub const Plan = struct {
    budget: u64,
    entries: u32,
    bytes_cap: u64,
    per_chat: u32,
    shared_least: u32,
    reserve: u64,
    /// the keep headroom (keepHeadroom): the budget stops at reserve + headroom; the keep floor stays the reserve
    headroom: u64,
    entries_capped: bool,
    share_pct: f64,
};

/// `state`: one kept state's fixed bytes (recurrent states, conv windows, rings, the drafter's window).
pub fn plan(env: Env, asked_gib: ?f64, mem: MemInfo, state: u64) Error!Plan {
    const reserve = try reserveBytes(env.reserve_gib, mem.total);
    const headroom = try keepHeadroom(env.headroom_gib);
    const budget = try budgetBytes(asked_gib, env.cache_gib, mem.available, reserve +| headroom);
    const pct = try sharePct(env.share_pct);
    const n = keptEntries(try cacheEntries(env.entries), pct, budget, state);
    return .{
        .budget = budget,
        .entries = n.count,
        .bytes_cap = try keptBytesCap(env.kept_bytes_gib),
        .per_chat = try keepPerChat(env.per_chat),
        .shared_least = try sharedPrefix(env.shared_prefix),
        .reserve = reserve,
        .headroom = headroom,
        .entries_capped = n.capped,
        .share_pct = pct,
    };
}

/// A save of `bytes` is refused when the machine would be left under the reserve (MEMORY_RESERVE_GIB holds while it runs).
pub fn fits(available: u64, reserve: u64, bytes: u64) bool {
    return available >= reserve +| bytes;
}

test "kept_entries: the configured count, a share of the budget, and the three-quarters ceiling" {
    const g: u64 = 1 << 30;
    // 12 GiB, ~45 MiB each
    const state: u64 = 45 << 20;
    try std.testing.expectEqual(@as(u32, 32), keptEntries(32, 0, 12 * g, state).count);
    const shared = keptEntries(8, 50, 12 * g, state);
    try std.testing.expectEqual(@as(u32, @intFromFloat(@as(f64, 12 * 1024) * 0.5 / 45.0)), shared.count);
    try std.testing.expect(shared.share and !shared.capped);
    // 100 % asks 75 %, and the ceiling (75 %) holds it
    const all = keptEntries(8, 100, 12 * g, state);
    try std.testing.expectEqual(@as(u32, @intFromFloat(@as(f64, 12 * 1024) * 0.75 / 45.0)), all.count);
    // 64 entries of 600 MiB past three quarters of 12 GiB: cut
    const cut = keptEntries(64, 0, 12 * g, 600 << 20);
    try std.testing.expect(cut.capped);
    try std.testing.expectEqual(@as(u32, 15), cut.count);
    try std.testing.expectEqual(@as(u32, 1), keptEntries(0, 0, g, 10 * g).count);
}

test "settings parse as the Python engine does" {
    try std.testing.expectEqual(@as(u32, 0), try sharedPrefix(null));
    try std.testing.expectEqual(@as(u32, 0), try sharedPrefix("off"));
    try std.testing.expectEqual(@as(u32, 256), try sharedPrefix("1"));
    try std.testing.expectEqual(@as(u32, 256), try sharedPrefix(" ON "));
    try std.testing.expectEqual(@as(u32, 64), try sharedPrefix("64"));
    try std.testing.expectError(error.BadSharedPrefix, sharedPrefix("8"));
    try std.testing.expectError(error.BadSharedPrefix, sharedPrefix("many"));
    try std.testing.expectEqual(@as(u32, 0), try keepPerChat(""));
    try std.testing.expectEqual(@as(u32, 3), try keepPerChat("3"));
    try std.testing.expectError(error.BadPerChat, keepPerChat("-1"));
    try std.testing.expectEqual(@as(u64, 0), try keptBytesCap("0"));
    try std.testing.expectEqual(@as(u64, 2 << 30), try keptBytesCap("2"));
    try std.testing.expectEqual(@as(u64, 1 << 29), try keptBytesCap("0.5"));
    try std.testing.expectEqual(@as(u32, 8), try cacheEntries(null));
    try std.testing.expectEqual(@as(u32, 32), try cacheEntries("32"));
    try std.testing.expectError(error.BadEntries, cacheEntries("x"));
}

test "the memory plan: the pool size asked, less what the reserve leaves of the machine" {
    const g: u64 = 1 << 30;
    // 12.5 GiB asked, 100 GiB free, 14.5 reserved: asked wins
    try std.testing.expectEqual(@as(u64, 12 * g + g / 2), try budgetBytes(null, "12.5", 100 * g, 14 * g + g / 2));
    // 20 GiB free less 14.5 reserved leaves 5.5
    try std.testing.expectEqual(5 * g + g / 2, try budgetBytes(null, "12.5", 20 * g, 14 * g + g / 2));
    // under the reserve: nothing
    try std.testing.expectEqual(@as(u64, 0), try budgetBytes(null, "12.5", 10 * g, 14 * g + g / 2));
    // the flag wins over the environment; 0 is off; unset is 3 GiB
    try std.testing.expectEqual(@as(u64, 4 * g), try budgetBytes(4, "12.5", 100 * g, g));
    try std.testing.expectEqual(@as(u64, 0), try budgetBytes(0, "12.5", 100 * g, g));
    try std.testing.expectEqual(@as(u64, 3 * g), try budgetBytes(null, null, 100 * g, g));
    try std.testing.expectError(error.BadCacheGib, budgetBytes(null, "lots", 100 * g, g));
    try std.testing.expectEqual(@as(u64, 14 * g + g / 2), try reserveBytes("14.5", 121 * g));
    try std.testing.expectEqual((121 * g + 1000) / 10, try reserveBytes(null, 121 * g + 1000)); // a tenth
    try std.testing.expectEqual(@as(u64, 4 * g), try reserveBytes(null, 24 * g));
    try std.testing.expectError(error.BadReserve, reserveBytes("1", 121 * g));
    try std.testing.expect(fits(30 * g, 14 * g, g));
    try std.testing.expect(!fits(15 * g, 14 * g + g / 2, g));
}

test "plan: the recipe's launch defaults" {
    const g: u64 = 1 << 30;
    const mem = meminfo("MemTotal:  126000000 kB\nMemFree: 1 kB\nMemAvailable: 100000000 kB\n").?;
    const p = try plan(.{ .entries = "32", .cache_gib = "12.5", .reserve_gib = "14.5", .shared_prefix = "1" }, null, mem, 100 << 20);
    try std.testing.expectEqual(@as(u32, 32), p.entries);
    try std.testing.expectEqual(@as(u64, 12 * g + g / 2), p.budget);
    try std.testing.expectEqual(@as(u32, 256), p.shared_least);
    try std.testing.expectEqual(@as(u64, 0), p.bytes_cap);
    try std.testing.expect(!p.entries_capped);
    try std.testing.expect(meminfo("MemTotal: 1 kB\n") == null);
}

test "the keep headroom: the budget stops that far above the reserve" {
    const g: u64 = 1 << 30;
    try std.testing.expectEqual(@as(u64, 2 * g), try keepHeadroom(null));
    try std.testing.expectEqual(@as(u64, 0), try keepHeadroom("0"));
    try std.testing.expectEqual(g / 2, try keepHeadroom("0.5"));
    try std.testing.expectError(error.BadHeadroom, keepHeadroom("-1"));
    try std.testing.expectError(error.BadHeadroom, keepHeadroom("some"));
    try std.testing.expectError(error.BadHeadroom, keepHeadroom("inf"));
    try std.testing.expectError(error.BadHeadroom, keepHeadroom("nan"));
    try std.testing.expectEqual(@as(u64, 1_000_000_000_000_000_000), try keepHeadroom("1e12")); // capped, no overflow
    // the floor: a state that leaves exactly the floor fits, one byte more does not
    try std.testing.expect(fits(100, 60, 40));
    try std.testing.expect(!fits(99, 60, 40));
    try std.testing.expect(fits(100, 0, 100));
    // ~22.9 GiB available, 19.6 reserved: 8 GiB asked gives ~1.3 GiB with the 2 GiB headroom, ~3.3 without it
    const mem = meminfo("MemTotal:  126000000 kB\nMemFree: 1 kB\nMemAvailable: 24000000 kB\n").?;
    const p = try plan(.{ .reserve_gib = "19.6" }, 8, mem, 100 << 20);
    try std.testing.expectEqual(@as(u64, 2 * g), p.headroom);
    try std.testing.expect(p.budget > g and p.budget < 2 * g);
    const p0 = try plan(.{ .reserve_gib = "19.6", .headroom_gib = "0" }, 8, mem, 100 << 20);
    try std.testing.expectEqual(@as(u64, 0), p0.headroom);
    try std.testing.expect(p0.budget > 3 * g and p0.budget < 4 * g);
    // the launch defaults keep their budget: 100 GiB free leaves room for both
    const big = meminfo("MemTotal:  126000000 kB\nMemFree: 1 kB\nMemAvailable: 100000000 kB\n").?;
    try std.testing.expectEqual(@as(u64, 12 * g + g / 2), (try plan(.{ .cache_gib = "12.5", .reserve_gib = "14.5" }, null, big, 100 << 20)).budget);
}
