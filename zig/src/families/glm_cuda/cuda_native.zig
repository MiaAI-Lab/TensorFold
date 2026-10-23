//! GLM CUDA registration for the native server, modeled on flashnext/cuda_native.zig.
//! Rank 0 serves. Ranks 1..tp-1 (tp = 2, 3 or 4) follow over the control links and the same NCCL communicator: rank 0
//! accepts every follower on its `--master` address and port (cuda_link.zig Server.acceptRanks), and each batch of
//! its calls reaches all of them, as the Python engine's TP=N runs rank 0's calls on every other rank.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const link = @import("cuda_link.zig");
const comm = @import("cuda_comm.zig");
const mirror = @import("cuda_mirror.zig");
const lane_backend = @import("cuda_lanes.zig");
const serving = @import("cuda_serving.zig");
const engine = @import("cuda_engine.zig");
const cache_plan = @import("cuda_cache_plan.zig");
const spill_mod = @import("cuda_spill.zig");
const choice_mod = @import("cuda_mtp_choice.zig");

pub const model_type = "glm5_next";
pub const formats: []const []const u8 = &.{"exl3"};
pub const default_context: i64 = 131072;
pub const prefill_step: u32 = 0;
pub const prompt_rows: u32 = 2048;
/// Engine.init already allocates all requested slot/cache storage.
pub const preallocated_streams = true;
/// native/cuda.zig openOwn opens it (captured kernel set required, two to four ranks, own prompt-state hooks).
pub const own_memory = true;

pub const Options = struct {
    context: usize,
    drafts: bool,
    parallel: u32 = 1,
    prompt_cache_gib: ?f64 = null,
    /// the spill tier (--snapshot-dir, --spill-*; the recipe's start.sh passes these; SPILL_GIB / SPILL_DIR in the environment
    /// serve when gib is 0, for scripts that start the engine without them)
    spill_dir: ?[]const u8 = null,
    spill_gib: f64 = 0,
    spill_highwater: f64 = 0.70,
    spill_min_tokens: u32 = 8192,
    spill_min_free_gib: f64 = 50,
    spill_flush_s: ?f64 = null,
    tp: u32 = 1,
    rank: u32 = 0,
    master: []const u8 = "",
    master_port: u16 = 29551,
    eager: bool = false,
    draft_dir: ?[]const u8 = null,
    vision: bool = false,
    /// rank 0 loads the vision tower (--vision-backend zig)
    tower: bool = false,
};

/// The engine takes picture and clip rows (Request.media).
pub const media = true;

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    follow: ?*const fn (*anyopaque) anyerror!void = null,
    cache: ?lane_backend.CacheHooks = null,
    read_status: ?*const fn (*anyopaque, *?u64, *?u64) void = null,
};

const link_timeout_ms: u32 = 10 * 60 * 1000;

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    if (!validTp(o.tp)) return error.UnsupportedTp;
    if (o.rank >= o.tp) return error.BadRank;
    const draft_dir = o.draft_dir orelse if (std.c.getenv("TF_GLM_DFLASH2_DIR")) |v| std.mem.span(v) else null;
    const choice = try choice_mod.environment();
    if (o.drafts and choice == .mtp and o.parallel != 1) return error.MtpRequiresSingleStream;
    if (o.drafts and choice == .dflash2 and draft_dir == null) return error.MissingDraftDirectory;
    if (o.parallel == 0 or o.parallel > 8) return error.BadParallel;
    const own = try gpa.create(Own);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .rank = o.rank, .world = o.tp };
    errdefer own.closeLink();
    if (o.tp > 1) {
        installStop();
        const address = std.Io.net.IpAddress.parse(o.master, o.master_port) catch {
            std.log.warn("--master {s}: not an IP address", .{o.master});
            return error.BadMaster;
        };
        if (o.rank == 0) {
            const exact = if (std.c.getenv("TF_MASTER_BIND")) |v| std.mem.eql(u8, std.mem.span(v), "exact") else false;
            own.server = try link.Server.open(link.bindAddress(address, o.tp, exact));
            std.log.info("rank 0: waiting for {d} other rank(s) on {s}:{d}", .{ o.tp - 1, o.master, o.master_port });
            own.link = try own.server.?.acceptRanks(gpa, o.tp, @intCast(link_timeout_ms));
        } else {
            std.log.info("rank {d}: joining rank 0 at {s}:{d}", .{ o.rank, o.master, o.master_port });
            own.link = try link.Link.join(io, address, link_timeout_ms);
            try own.link.?.hello(o.rank);
        }
        const text = try settings(gpa, io, dir, kernels, o);
        defer gpa.free(text);
        own.link.?.agreeRanks(gpa, o.rank, text) catch |e| {
            if (e == error.RanksDisagree) std.log.err("rank {d} loads with \"{s}\"; the other rank's settings differ", .{ o.rank, text });
            return e;
        };
        own.comm = try comm.Comm.init(gpa, own.link.?, o.rank, o.tp);
        std.log.info("rank {d}: settings agreed, NCCL joined", .{o.rank});
        own.watch = .{ .io = io, .fd = own.link.?.fd, .more = own.link.?.more, .rank = o.rank, .comm = mirror.Coll.of(comm.Comm, &own.comm.?) };
        try own.watch.?.start();
    }
    const streams: usize = @max(@as(usize, 1), @min(@as(usize, o.parallel), lane_backend.max_streams));
    own.engine = try engine.Engine.init(gpa, io, ctx, dir, kernels, .{
        .context = o.context,
        .drafts = o.drafts,
        .streams = streams,
        .rank = o.rank,
        .world = o.tp,
        .comm = if (own.comm) |*c| c else null,
        .eager = o.eager or envText("TF_GLM_EAGER") != null,
        .draft_dir = draft_dir,
        .draft_choice = choice,
        .vision = o.vision,
        .tower = o.tower,
    });
    errdefer own.engine.?.deinit();
    const window: usize = 16;
    own.serving = try gpa.create(serving.Serving(engine.Engine));
    errdefer gpa.destroy(own.serving.?);
    if (o.tp > 1) {
        own.bcast = try mirror.Broadcast.init(ctx.d, mirror.Coll.of(comm.Comm, &own.comm.?), own.engine.?.stream.handle);
        errdefer if (own.bcast) |*b| b.deinit();
        own.agreement = try mirror.Agreement.init(ctx.d, mirror.Coll.of(comm.Comm, &own.comm.?), own.engine.?.stream.handle, &own.watch.?);
        errdefer if (own.agreement) |*a| a.deinit();
        own.records = .{ .gpa = gpa, .wire = own.bcast.?.wire(), .link = mirrorLink(own.link.?), .watch = &own.watch.? };
    }
    try own.serving.?.init(gpa, own.engine.?, o.context, o.drafts, streams, window, @as(usize, o.rank), if (own.records) |*r| r.channel() else null);
    if (own.agreement) |*a| own.serving.?.setAgreement(a.gate(), if (own.watch) |*w| w else null);
    const build_key = buildKey(gpa, io, dir, kernels, o) catch 0;
    return .{
        .backend = own.serving.?.laneBackend(),
        .facts = own.serving.?.facts(),
        .rows = @intCast(window),
        .stream_bytes = own.engine.?.seqBytes(o.context + window),
        .ctx = own,
        .deinit = Own.release,
        .follow = if (o.tp > 1 and o.rank >= 1) Own.follow else null,
        .cache = cacheFor(own, o, build_key),
        .read_status = Own.readStatus,
    };
}

fn envText(name: [:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name.ptr) orelse return null;
    return std.mem.span(v);
}

/// What a stored state depends on beyond its tokens: the weights' snapshot (and the drafter's), the captured kernel set, the
/// world and whether drafts run (their window rides in the state). Not the context or the stream count.
fn buildKey(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, kernels: []const u8, o: Options) !u64 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_][]const u8{ "config.json", "model.safetensors.index.json", "tokenizer.json" }) |name| try hashFile(gpa, io, &h, dir, name);
    if (o.drafts) {
        const dd = o.draft_dir orelse if (std.c.getenv("TF_GLM_DFLASH2_DIR")) |v| std.mem.span(v) else return error.MissingDraftDirectory;
        h.update(std.fs.path.basename(std.mem.trimEnd(u8, dd, "/")));
        try hashFile(gpa, io, &h, dd, "config.json");
        try hashFile(gpa, io, &h, dd, "model.safetensors.index.json");
    }
    try hashFile(gpa, io, &h, kernels, "aot.json");
    h.update(if (o.drafts) "drafts" else "plain");
    var tp: [4]u8 = undefined;
    std.mem.writeInt(u32, &tp, o.tp, .little);
    h.update(&tp);
    h.update("tfspill-state-v1");
    const d = h.finalResult();
    return std.hash.Wyhash.hash(0, &d);
}

const SpillSettings = struct { highwater: f64, flush_s: f64 };

/// The spill tier (SPILL_GIB, SPILL_DIR, SPILL_HIGHWATER, SPILL_MIN_TOKENS, SPILL_MIN_FREE_GIB, SPILL_FLUSH_S, the launch
/// scripts' names): off unless SPILL_GIB is above 0. Every rank opens its own folder; they agree through the settings text.
fn openSpill(own: *Own, o: Options, build_key: u64) ?SpillSettings {
    const num = struct {
        fn get(name: [:0]const u8, default: f64) f64 {
            const t = envText(name) orelse return default;
            return std.fmt.parseFloat(f64, std.mem.trim(u8, t, " ")) catch default;
        }
    };
    var gib = o.spill_gib;
    var root: []const u8 = o.spill_dir orelse "";
    var highwater = o.spill_highwater;
    var min_tokens: f64 = @floatFromInt(o.spill_min_tokens);
    var min_free = o.spill_min_free_gib;
    if (!(gib > 0)) { // not given as flags: the launch scripts' variable names
        gib = num.get("SPILL_GIB", 0);
        highwater = num.get("SPILL_HIGHWATER", 0.70);
        min_tokens = num.get("SPILL_MIN_TOKENS", 8192);
        min_free = num.get("SPILL_MIN_FREE_GIB", 50);
        root = envText("SPILL_DIR") orelse "";
    }
    if (!(gib > 0)) return null;
    if (root.len == 0) {
        const home = envText("HOME") orelse return null;
        root = std.fmt.allocPrint(own.gpa, "{s}/.cache/tensorfold-spill", .{home}) catch return null;
    }
    const flush_s = o.spill_flush_s orelse num.get("TF_SPILL_FLUSH_S", num.get("SPILL_FLUSH_S", 60));
    const e = own.engine.?;
    const tier = own.gpa.create(spill_mod.Tier) catch return null;
    tier.* = spill_mod.Tier.open(own.gpa, .{
        .root = root,
        .build = build_key,
        .rank = o.rank,
        .cap = @intFromFloat(gib * cache_plan.gib),
        .min_tokens = @intFromFloat(min_tokens),
        .min_free = @intFromFloat(min_free * cache_plan.gib),
    }) catch |err| {
        std.log.err("spill: cannot open {s} ({s}); no spill tier", .{ root, @errorName(err) });
        own.gpa.destroy(tier);
        return null;
    };
    e.spill = tier;
    e.spill_wait_ns = @intFromFloat(@max(0, flush_s) * 1e9);
    const st = tier.snapshot();
    std.log.info("spill tier: {s}, up to {d:.0} GiB, {d} states ({d} MiB) found; written past {d:.0}% of the cache, from {d} tokens, keeping {d:.0} GiB of disk free", .{ tier.dir, gib, st.entries, st.bytes >> 20, highwater * 100, tier.cfg.min_tokens, @as(f64, @floatFromInt(tier.cfg.min_free)) / cache_plan.gib });
    return .{ .highwater = highwater, .flush_s = flush_s };
}

/// The recipe's prompt cache for this process (rank 0 holds the store; both ranks hold the states and the reserve):
/// budget and caps from the environment as the Python engine reads them, the machine's memory less the reserve.
fn cacheFor(own: *Own, o: Options, build_key: u64) ?lane_backend.CacheHooks {
    // MTP drafts: the head's taps pair a continuation's rows with the prompt's tokens from its start (promptTaps,
    // prompt_at), so a prefill that goes on from a kept or live state would misplace them; kept states wait for that
    if (own.engine.?.mtp != null) {
        if (o.rank == 0) std.log.info("prompt cache: off (MTP drafts do not resume from kept states yet)", .{});
        return null;
    }
    const env: cache_plan.Env = .{
        .entries = envText("TF_GLM_CACHE_ENTRIES"),
        .share_pct = envText("TF_GLM_CACHE_SHARE_PCT"),
        .kept_bytes_gib = envText("TF_GLM_KEPT_BYTES_GIB"),
        .per_chat = envText("TF_GLM_KEEP_PER_CHAT"),
        .shared_prefix = envText("TF_GLM_SHARED_PREFIX"),
        .cache_gib = envText("TF_GLM_CACHE_GIB"),
        .reserve_gib = envText("TENSORFOLD_MEMORY_RESERVE_GIB"),
        .headroom_gib = envText("TF_GLM_KEEP_HEADROOM_GIB"),
    };
    const read = cache_plan.memNow() orelse {
        std.log.warn("prompt cache: /proc/meminfo is unreadable; no prompt cache", .{});
        return null;
    };
    // MemAvailable first, the per-CPU page lists right after (as the KV pool's reading), on a unified GPU only (as the
    // pool's): the budget counts them, the keep floor's live reads (keepFits, the saves) do not
    const e = own.engine.?;
    const unified = if (e.ctx) |c| (c.attribute(.integrated) catch 0) != 0 else false;
    const mem = if (unified) cache_plan.counting(read, cache_plan.parkedNow(own.gpa)) else read;
    const p = cache_plan.plan(env, o.prompt_cache_gib, mem, e.snapBytes(0)) catch |err| {
        std.log.err("prompt cache: bad setting ({s}); no prompt cache", .{@errorName(err)});
        return null;
    };
    if (p.budget > 0 and mem.available - read.available >= 1 << 28) std.log.info("rank {d}: the kept prompt states' budget counts the {d:.2} GiB free on the per-CPU page lists beyond their resting level (neither MemFree nor MemAvailable does)", .{ o.rank, @as(f64, @floatFromInt(mem.available - read.available)) / cache_plan.gib });
    // the keep floor is the reserve: a save (or a load from the spill tier) that would leave the machine under it is
    // refused (cuda_lanes pass agrees on the room first); the headroom only shapes the budget, so the kept states stay
    // that far above the floor and the prompts' own buffers have it
    e.keep_reserve = p.reserve;
    // every rank writes and reads its own copies (rank 0 decides, rank 1 follows its notes)
    const spill = if (p.budget > 0) openSpill(own, o, build_key) else null;
    if (o.rank != 0) return null;
    if (p.budget == 0) {
        std.log.info("prompt cache: off (budget 0)", .{});
        return null;
    }
    std.log.info("prompt cache: {d:.1} GiB for kept states, at most {d} (fixed {d} MiB each){s}; per chat {d}, bytes cap {d} MiB, shared prefix {d}; {d:.1} GiB reserved, {d:.1} GiB headroom for the prompts' buffers", .{
        @as(f64, @floatFromInt(p.budget)) / cache_plan.gib,                                        p.entries,                                     e.snapBytes(0) >> 20,
        if (p.entries_capped) " (cut to three quarters of the budget)" else "",                    p.per_chat,                                    p.bytes_cap >> 20,
        p.shared_least,                                                                            @as(f64, @floatFromInt(p.reserve)) / cache_plan.gib,                              @as(f64, @floatFromInt(p.headroom)) / cache_plan.gib,
    });
    return own.serving.?.local.cacheHooks(p.budget, .{ .grid = 64, .min_gap = 256, .min_prompt = 64, .entries = p.entries, .bytes_cap = p.bytes_cap, .per_chat = p.per_chat, .shared_least = p.shared_least }, if (spill) |sp| sp.highwater else 0, if (spill) |sp| sp.flush_s else 60);
}

const Own = struct {
    gpa: std.mem.Allocator,
    rank: u32,
    world: u32,
    engine: ?*engine.Engine = null,
    serving: ?*serving.Serving(engine.Engine) = null,
    server: ?link.Server = null,
    link: ?link.Link = null,
    comm: ?comm.Comm = null,
    bcast: ?mirror.Broadcast = null,
    agreement: ?mirror.Agreement = null,
    records: ?mirror.Records = null,
    watch: ?mirror.Watch = null,

    fn readStatus(p: *anyopaque, tokens: *?u64, free: *?u64) void {
        const own: *Own = @ptrCast(@alignCast(p));
        if (own.engine) |e| {
            const counts = e.poolStatus();
            tokens.* = counts.pool_tokens;
            free.* = counts.pool_free_tokens;
        }
    }

    fn closeLink(own: *Own) void {
        if (own.watch) |*w| {
            w.quiet();
            w.deinit();
            own.watch = null;
        }
        if (own.comm) |*c| c.deinit();
        own.comm = null;
        if (own.link) |l| l.close();
        own.link = null;
        if (own.server) |s| s.close();
        own.server = null;
    }

    fn release(p: *anyopaque) void {
        const own: *Own = @ptrCast(@alignCast(p));
        if (own.watch) |*w| w.quiet();
        if (own.serving) |s| {
            s.deinit();
            own.gpa.destroy(s);
            own.serving = null;
        }
        if (own.records) |*r| r.deinit();
        if (own.bcast) |*b| b.deinit();
        if (own.agreement) |*a| a.deinit();
        if (own.watch) |*w| w.deinit();
        own.watch = null;
        if (own.engine) |e| e.deinit();
        own.engine = null;
        own.closeLink();
        own.gpa.destroy(own);
    }

    fn follow(p: *anyopaque) !void {
        const own: *Own = @ptrCast(@alignCast(p));
        const s = own.serving orelse return error.NotFollower;
        const wait: u32 = if (own.engine) |e| @intCast(@min(e.spill_wait_ns / std.time.ns_per_s, 3600)) else 60;
        installFollowStop(wait + 15);
        try s.follow();
    }
};

/// The tensor-parallel sizes the engine runs: one GPU, or the Python engine's two, three and four.
pub fn validTp(tp: u32) bool {
    return tp >= 1 and tp <= comm.max_world;
}

/// Flash Next's Records stores its own Link type. Both links are one socket fd, so the record borrows that fd.
fn mirrorLink(l: link.Link) @import("flashnext").link.Link {
    const Flash = @import("flashnext").link.Link;
    comptime {
        if (@sizeOf(link.Link) != @sizeOf(Flash)) @compileError("GLM Link and Flash Next Link differ in size");
        if (@offsetOf(link.Link, "fd") != @offsetOf(Flash, "fd")) @compileError("Link.fd");
        if (@offsetOf(link.Link, "more") != @offsetOf(Flash, "more")) @compileError("Link.more");
    }
    return .{ .fd = l.fd, .more = l.more };
}

fn installStop() void {
    const posix = std.posix;
    const stop: posix.Sigaction = .{ .handler = .{ .handler = onStop }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(.TERM, &stop, null);
    posix.sigaction(.INT, &stop, null);
}

extern "c" fn alarm(seconds: c_uint) c_uint;

/// How long a following rank outlives a stop signal: rank 0 closes the link once it has stored its kept states (SPILL_FLUSH_S),
/// and this rank writes its own copies meanwhile; SIGALRM ends it if rank 0 never does.
var follow_grace_s: std.atomic.Value(u32) = .init(75);

/// A following rank (rank 1 and up) on SIGTERM / SIGINT: it keeps running so the notes rank 0 sends while it stores its kept
/// states (a clean stop, stop.sh) reach it; rank 0's stop frame or the link closing ends it, else the alarm does.
fn onStopFollowing(_: std.posix.SIG) callconv(.c) void {
    const line = "tensorfold: stop signal: this rank follows rank 0 to its end (it stores its kept states first)\n";
    _ = std.c.write(2, line, line.len);
    _ = alarm(follow_grace_s.load(.monotonic));
}

fn installFollowStop(grace_s: u32) void {
    follow_grace_s.store(grace_s, .monotonic);
    const posix = std.posix;
    const stop: posix.Sigaction = .{ .handler = .{ .handler = onStopFollowing }, .mask = posix.sigemptyset(), .flags = std.c.SA.RESTART };
    posix.sigaction(.TERM, &stop, null);
    posix.sigaction(.INT, &stop, null);
}

fn onStop(_: std.posix.SIG) callconv(.c) void {
    const line = "tensorfold: stop signal: this rank exits, the other follows\n";
    _ = std.c.write(2, line, line.len);
    std.c._exit(0);
}

fn settings(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, kernels: []const u8, o: Options) ![]u8 {
    var snapshot = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_][]const u8{ "config.json", "model.safetensors.index.json", "tokenizer.json" }) |name| try hashFile(gpa, io, &snapshot, dir, name);
    const choice = try choice_mod.environment();
    snapshot.update(@tagName(choice));
    snapshot.update(choice_mod.setting());
    if (std.c.getenv("TF_GLM_DRAFT_QUANT")) |v| snapshot.update(std.mem.span(v));
    if (o.drafts and choice == .dflash2) {
        const dd = o.draft_dir orelse if (std.c.getenv("TF_GLM_DFLASH2_DIR")) |v| std.mem.span(v) else return error.MissingDraftDirectory;
        snapshot.update(std.fs.path.basename(std.mem.trimEnd(u8, dd, "/")));
        try hashFile(gpa, io, &snapshot, dd, "config.json");
        try hashFile(gpa, io, &snapshot, dd, "model.safetensors.index.json");
    }
    var kernel_set = std.crypto.hash.sha2.Sha256.init(.{});
    try hashFile(gpa, io, &kernel_set, kernels, "aot.json");
    const display_cfg = try @import("cuda_display_kv.zig").Config.read();
    const memory = @import("cuda_display_kv.zig");
    return std.fmt.allocPrint(gpa, "glm5_next display={d}:{s} cache={s} reserve={s} tp={d} checkpoint={s} snapshot={x} context={d} parallel={d} drafts={} eager={} vision={} tower={} kernels={x}", .{
        display_cfg.bytes, @tagName(display_cfg.kind), memory.env("TF_GLM_CACHE_GIB") orelse memory.env("KV_POOL_GIB") orelse "3", memory.env("TENSORFOLD_MEMORY_RESERVE_GIB") orelse memory.env("MEMORY_RESERVE_GIB") orelse "auto", o.tp, std.fs.path.basename(std.mem.trimEnd(u8, dir, "/")), snapshot.finalResult(), o.context, o.parallel, o.drafts, o.eager, o.vision, o.tower, kernel_set.finalResult(),
    });
}

fn hashFile(gpa: std.mem.Allocator, io: std.Io, h: *std.crypto.hash.sha2.Sha256, dir: []const u8, name: []const u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30)) catch |e| switch (e) {
        error.FileNotFound => {
            h.update(name);
            h.update(":absent;");
            return;
        },
        else => return e,
    };
    defer gpa.free(bytes);
    h.update(name);
    h.update(":");
    h.update(bytes);
    h.update(";");
}

test "GLM metadata advertises EXL3" {
    try std.testing.expectEqualStrings("glm5_next", model_type);
    try std.testing.expectEqualStrings("exl3", formats[0]);
}

test "open rejects a world beyond 4 and a rank outside it before any CUDA call" {
    const o = Options{ .context = 8, .drafts = false };
    try std.testing.expectError(error.UnsupportedTp, open(std.testing.allocator, std.testing.io, undefined, "", "", .{ .context = o.context, .drafts = false, .tp = 5 }));
    try std.testing.expectError(error.BadRank, open(std.testing.allocator, std.testing.io, undefined, "", "", .{ .context = o.context, .drafts = false, .tp = 4, .rank = 4 }));
    try std.testing.expectError(error.BadMaster, open(std.testing.allocator, std.testing.io, undefined, "", "", .{ .context = o.context, .drafts = false, .tp = 3, .rank = 1, .master = "not-an-ip" }));
    try std.testing.expect(validTp(1) and validTp(2) and validTp(3) and validTp(4) and !validTp(0) and !validTp(5));
    try std.testing.expectError(error.BadRank, open(std.testing.allocator, std.testing.io, undefined, "", "", .{ .context = o.context, .drafts = false, .tp = 2, .rank = 2 }));
    try std.testing.expectError(error.BadMaster, open(std.testing.allocator, std.testing.io, undefined, "", "", .{ .context = o.context, .drafts = false, .tp = 2, .rank = 0, .master = "not-an-ip" }));
    try std.testing.expectError(error.BadRank, open(std.testing.allocator, std.testing.io, undefined, "", "", .{ .context = o.context, .drafts = false, .tp = 1, .rank = 1 }));
}
/// Greedy (or keyed-sampling) generation without the serving core: both ranks run this same sequence of calls, so every
/// NCCL gathers line up. P7 target acceptance commits P9 proposals; serial mode uses one-row windows.
pub fn draftStats() @TypeOf(@import("cuda_generate.zig").stats) {
    return @import("cuda_generate.zig").stats;
}

pub fn generate(ctx: *anyopaque, io: std.Io, gpa: std.mem.Allocator, prompt: []const u32, max_tokens: usize, stop: []const u32, timing: *[2]u64) ![]u32 {
    const own: *Own = @ptrCast(@alignCast(ctx));
    const e = own.engine orelse return error.NotLoaded;
    if (prompt.len > e.context or max_tokens > e.context - prompt.len) return error.ContextFull;
    const seq = try e.newSeq(prompt.len + max_tokens);
    defer e.freeSeq(seq);
    e.bind(seq);
    const t0 = std.Io.Timestamp.now(io, .awake);
    if (max_tokens == 0) {
        timing.* = .{ 0, 0 };
        return gpa.alloc(u32, 0);
    }
    abSwitch(e);
    const first = try e.prefillWith(prompt, null, .{});
    try e.stream.synchronize();
    const t1 = std.Io.Timestamp.now(io, .awake);
    const prof = std.c.getenv("TF_GLM_PROFILE_API") != null and prompt.len < 100;
    // nsys --capture-range=cudaProfilerApi hooks the runtime's cudaProfilerStart; the driver's cuProfilerStart, called
    // through the dlopened libcuda, did not start a capture. The runtime library is opened only for this.
    if (prof) profileWait();
    var rt: ?std.DynLib = null;
    var rt_start: ?*const fn () callconv(.c) c_int = null;
    var rt_stop: ?*const fn () callconv(.c) c_int = null;
    if (prof) {
        rt = std.DynLib.open("libcudart.so.13") catch std.DynLib.open("libcudart.so") catch null;
        if (rt) |*lib| {
            rt_start = lib.lookup(*const fn () callconv(.c) c_int, "cudaProfilerStart");
            rt_stop = lib.lookup(*const fn () callconv(.c) c_int, "cudaProfilerStop");
        }
        if (rt_start) |f| _ = f() else _ = e.stream.d.api.cuProfilerStart();
    }
    const out = try @import("cuda_generate.zig").run(engine.Engine, e, gpa, seq, first.first, max_tokens, stop, e.drafts);
    errdefer gpa.free(out);
    try e.stream.synchronize();
    if (prof) {
        if (rt_stop) |f| _ = f() else _ = e.stream.d.api.cuProfilerStop();
        if (rt) |*lib| lib.close();
    }
    const t2 = std.Io.Timestamp.now(io, .awake);
    timing[0] = @intCast(@divTrunc(t0.durationTo(t1).nanoseconds, 1_000_000));
    timing[1] = @intCast(@divTrunc(t1.durationTo(t2).nanoseconds, 1_000_000));
    return out;
}

/// One unsplit prompt chunk. The context is a prefill Engine. Binding lives in cuda_prefill so this file stays a call hook.
pub fn prefillChunk(ctx: *anyopaque, rows: usize, pos: usize, parity: *[34]u1, head: bool) !void {
    const prefill = @import("cuda_prefill.zig");
    const fw = @import("cuda_forward.zig");
    prefill.bindForwardHook();
    const e: *prefill.Engine = @ptrCast(@alignCast(ctx));
    try fw.enterPrompt(@ptrCast(e), rows, pos, parity, head);
}

test "serving refuses more than eight lanes before any CUDA call" {
    for ([_]u32{ 0, 9, 32 }) |parallel| try std.testing.expectError(error.BadParallel, open(std.testing.allocator, std.testing.io, undefined, "", "", .{ .context = 131072, .drafts = false, .parallel = parallel }));
    try std.testing.expectEqual(@as(i64, 131072), default_context);
    try std.testing.expectEqual(@as(u32, 0), prefill_step);
}

/// TF_GLM_AB=stage,greedy: measurement aid. Odd generations (1st, 3rd, ...) run the new paths, even ones the old: the
/// blocking metadata staging and readbacks (`stage`) and/or the radix top-k greedy (`greedy`). Both ranks count the
/// same generations, so they switch together; one process then carries an interleaved A/B with no session drift.
var ab_calls: usize = 0;
fn abSwitch(e: *engine.Engine) void {
    const v = std.c.getenv("TF_GLM_AB") orelse return;
    const spec = std.mem.span(v);
    ab_calls += 1;
    const old = ab_calls % 2 == 0;
    @import("cuda_stage.zig").sync_mode = old and std.mem.indexOf(u8, spec, "stage") != null;
    e.sampler.legacy_greedy = old and std.mem.indexOf(u8, spec, "greedy") != null;
    std.log.info("A/B generation {d}: {s}", .{ ab_calls, if (old) "old paths" else "new paths" });
}

/// TF_GLM_PROFILE_AT=<seconds since this process started>: the profiled decode waits until then, so an nsys time window
/// (`--delay`, `--duration`) can enclose it whatever the weight-load time was (nsys did not honour the capture range
/// the driver and runtime profiler calls set).
fn profileWait() void {
    const v = std.c.getenv("TF_GLM_PROFILE_AT") orelse return;
    const at = std.fmt.parseFloat(f64, std.mem.span(v)) catch return;
    var buf: [1024]u8 = undefined;
    const fd = std.c.open("/proc/self/stat", .{}, @as(std.c.mode_t, 0));
    if (fd < 0) return;
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return;
    const text = buf[0..@intCast(n)];
    // fields after the ")" that ends the command name; starttime is field 22 overall = the 20th after it
    const close = std.mem.lastIndexOfScalar(u8, text, ')') orelse return;
    var it = std.mem.tokenizeScalar(u8, text[close + 1 ..], ' ');
    var i: usize = 0;
    var ticks: f64 = 0;
    while (it.next()) |tok| : (i += 1) if (i == 19) {
        ticks = std.fmt.parseFloat(f64, tok) catch return;
        break;
    };
    var now: std.c.timespec = undefined;
    if (std.c.clock_gettime(.BOOTTIME, &now) != 0) return;
    const up = @as(f64, @floatFromInt(now.sec)) + @as(f64, @floatFromInt(now.nsec)) / 1e9 - ticks / 100.0;
    if (up >= at) return;
    const wait = at - up;
    std.log.info("profile: waiting {d:.1} s (process age {d:.1} s)", .{ wait, up });
    var req: std.c.timespec = .{ .sec = @intFromFloat(wait), .nsec = @intFromFloat((wait - @floor(wait)) * 1e9) };
    _ = std.c.nanosleep(&req, null);
}
