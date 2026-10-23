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
const shard = @import("cuda_shard.zig");
const identity = @import("cuda_spill_identity.zig");
const spill_state = @import("cuda_spill_state.zig");
const spill_wire = @import("cuda_spill_wire.zig");
const spill_tier = @import("cuda_spill_tier.zig");
const format = @import("core").spill.format;

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
    /// spill v2's knobs (the server's cli.spill: TF_GLM_SPILL_QUOTA, _TURNS, _READ_BLOCKS, _STAGE_MIB, and the flush
    /// bound), used with TF_GLM_SPILL_V2=1 only
    spill_quota: f64 = 0,
    spill_turns: u32 = 4,
    spill_read_blocks: u32 = 16,
    spill_stage_mib: u32 = 256,
    spill_flush_ms: u32 = 60_000,
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
    /// TF_GLM_KV_SHARD: the parts' rows and counters for /health and /metrics (null: off)
    kv_parts: ?lanes.kv_parts.Source = null,
    /// the window a request may use when the engine had to shorten --context (TF_GLM_KV_SHARD: a part's; null: as asked)
    context: ?usize = null,
    /// TF_GLM_SPILL_V2=1 on rank 0: the prompt cache's tier (native/cuda.zig sets it on the store; null: off)
    tier: ?*SpillLeader = null,
};

/// Spill v2's tier on rank 0 and on the other ranks (cuda_spill_tier).
pub const SpillLeader = spill_tier.Leader(engine.Engine);
pub const SpillFollower = spill_tier.Follower(engine.Engine);

const link_timeout_ms: u32 = 10 * 60 * 1000;

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    if (!validTp(o.tp)) return error.UnsupportedTp;
    if (o.rank >= o.tp) return error.BadRank;
    const draft_dir = o.draft_dir orelse if (std.c.getenv("TF_GLM_DFLASH2_DIR")) |v| std.mem.span(v) else null;
    const choice = try choice_mod.environment();
    if (o.drafts and choice == .mtp and o.parallel != 1) return error.MtpRequiresSingleStream;
    if (o.drafts and choice == .dflash2 and draft_dir == null) return error.MissingDraftDirectory;
    if (o.parallel == 0 or o.parallel > 8) return error.BadParallel;
    // TF_GLM_KV_SHARD: every rank reads its own; a rank that cannot fails with the others, after the agreement
    const kv = shard.Switches.read();
    var kv_switches: shard.Switches = .{};
    // TF_GLM_SPILL_V2 and the recipe's identity with it: the same (each rank's own, refused after the agreement)
    const v2 = spillV2(o, choice);
    var spill_env: ?identity.Env = null;
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
        const text = try settings(gpa, io, dir, kernels, o, kv, v2);
        defer gpa.free(text);
        own.link.?.agreeRanks(gpa, o.rank, text) catch |e| {
            if (e == error.RanksDisagree) std.log.err("rank {d} loads with \"{s}\"; the other rank's settings differ", .{ o.rank, text });
            // a rank that could not read its TF_GLM_KV_SHARD* or spill v2 identity (its text said `unreadable`) says why
            _ = kv catch |bad| return kvRefused(bad);
            _ = v2 catch |bad| return kvRefused(bad);
            return e;
        };
        kv_switches = kvAdmit(kv, o, choice) catch |e| return kvRefused(e);
        spill_env = spillAdmit(v2, o, choice) catch |e| return kvRefused(e);
        own.comm = try comm.Comm.init(gpa, own.link.?, o.rank, o.tp);
        std.log.info("rank {d}: settings agreed, NCCL joined", .{o.rank});
        own.watch = .{ .io = io, .fd = own.link.?.fd, .more = own.link.?.more, .rank = o.rank, .comm = mirror.Coll.of(comm.Comm, &own.comm.?) };
        try own.watch.?.start();
    } else {
        kv_switches = kvAdmit(kv, o, choice) catch |e| return kvRefused(e);
        spill_env = spillAdmit(v2, o, choice) catch |e| return kvRefused(e);
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
        .kv_shard = if (kv_switches.on) kv_switches else null,
        .spill_v2 = if (spill_env != null) tierSettings(o, 0) else null,
    });
    errdefer own.engine.?.deinit();
    // spill v2: every rank's fingerprint, agreed over the control link before any store opens (TP=2: spillAdmit)
    const agreement: ?identity.Agreement = if (spill_env) |env| try spillAgree(gpa, io, own, ctx, dir, kernels, o, choice, draft_dir, env, kv_switches, streams) else null;
    // TF_GLM_KV_SHARD may shorten the window a request uses to a part's (cuda_shard.fit); as asked otherwise
    const context = own.engine.?.context;
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
    try own.serving.?.init(gpa, own.engine.?, context, o.drafts, streams, window, @as(usize, o.rank), if (own.records) |*r| r.channel() else null);
    if (own.agreement) |*a| own.serving.?.setAgreement(a.gate(), if (own.watch) |*w| w else null);
    const build_key = if (agreement == null) buildKey(gpa, io, dir, kernels, o) catch 0 else 0;
    var budget: u64 = 0;
    const cache = cacheFor(own, o, build_key, agreement != null, &budget);
    if (agreement) |a| openTier(own, gpa, io, o, a, budget) catch |e| return kvRefused(e);
    return .{
        .backend = own.serving.?.laneBackend(),
        .facts = own.serving.?.facts(),
        .rows = @intCast(window),
        .stream_bytes = own.engine.?.seqBytes(context + window),
        .ctx = own,
        .deinit = Own.release,
        .follow = if (o.tp > 1 and o.rank >= 1) Own.follow else null,
        .cache = cache,
        .read_status = Own.readStatus,
        .kv_parts = if (own.engine.?.shard != null) .{ .ctx = own.engine.?, .read = engine.Engine.readKvParts } else null,
        .context = if (context != o.context) context else null,
        .tier = own.tier,
    };
}

/// TF_GLM_KV_SHARD once the ranks agreed on it (each input is in the settings text, so every rank answers alike):
/// this rank's own read error, a setting sharding cannot serve, or any `on` in a build without its device paths or
/// glm_shard.cu's kernels.
fn kvAdmit(read: anyerror!shard.Switches, o: Options, choice: choice_mod.Choice) !shard.Switches {
    const sw = try read;
    try shard.refuse(sw, o.tp, o.parallel, o.drafts and choice == .mtp);
    if (sw.on and !shard.gpu_paths) return error.KvShardNotBuilt;
    if (sw.on) try shard.kernelsIn(cuda.kernels.available, cuda.kernels.glm_shard);
    return sw;
}

/// The refusal in the start-up's log, in words.
fn kvRefused(err: anyerror) anyerror {
    std.log.err("{s}", .{refusal(err) orelse @errorName(err)});
    return err;
}

/// The words for a refusal this family explains (native/cuda.zig shows them in place of the error's name).
pub fn refusal(err: anyerror) ?[]const u8 {
    return shard.why(err) orelse identity.why(err) orelse switch (err) {
        error.BadSpillV2 => "TF_GLM_SPILL_V2: 1 (spill v2) or 0 (v1, the rollback), nothing else",
        error.SpillV2NeedsTwoRanks => "TF_GLM_SPILL_V2=1: spill v2 runs on the recipe's two ranks only for now (--tp 2)",
        error.SpillV2WithMtp => "TF_GLM_SPILL_V2=1: the prompt cache is off with MTP drafts, so spill v2 would keep nothing (DFlash2 drafts or none)",
        error.SpillV2NoPromptCache => "TF_GLM_SPILL_V2=1 needs the prompt cache, and a rank's kept-state budget is 0 (--prompt-cache-gib; with PROMPT_CACHE_GIB a lower KV_POOL_GIB)",
        error.SpillV2StoreFailed => "TF_GLM_SPILL_V2=1: a rank's spill v2 store did not open (that rank's log says why)",
        error.StagingTooSmall => "TF_GLM_SPILL_STAGE_MIB: spill v2's write ring holds two blocks at least (25 MiB), TF_GLM_SPILL_READ_BLOCKS one",
        else => null,
    };
}

/// TF_GLM_SPILL_V2=1 with the spill on (--spill-gib above 0 and --snapshot-dir): spill v2 instead of v1, and this rank's
/// identity from the recipe (cuda_spill_identity.Env). Null: off (unset, blank or 0; or the spill off, and then nothing is
/// read). Any other value is this rank's error, as a missing identity is.
fn spillV2(o: Options, choice: choice_mod.Choice) anyerror!?identity.Env {
    if (!(o.spill_gib > 0) or o.spill_dir == null) return null;
    const v = std.mem.trim(u8, envText("TF_GLM_SPILL_V2") orelse return null, " ");
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return null;
    if (!std.mem.eql(u8, v, "1")) return error.BadSpillV2;
    return try identity.Env.read(envText, o.drafts and choice == .dflash2);
}

/// Spill v2 once the ranks agreed on it (its switch and knobs are in the settings text, so every rank answers alike):
/// this rank's own read error, a world it does not run on yet (the recipe's two ranks; the identity agreement and the
/// rank protocol are the pair's), MTP drafts (the prompt cache is off with them), or a staging too small for its
/// blocks. With TF_GLM_KV_SHARD=1 the tier is owner-sharded (Engine.spillLayout).
fn spillAdmit(read: anyerror!?identity.Env, o: Options, choice: choice_mod.Choice) !?identity.Env {
    const env = (try read) orelse return null;
    if (o.tp != 2) return error.SpillV2NeedsTwoRanks;
    if (o.drafts and choice == .mtp) return error.SpillV2WithMtp;
    try spill_tier.check(tierSettings(o, 0));
    return env;
}

/// Spill v2's settings from the server's (`hold`: the release guard's limit, a quarter of the kept-state budget: held
/// buffers sit outside it until their blocks are out).
fn tierSettings(o: Options, budget: u64) spill_tier.Settings {
    return .{ .dir = o.spill_dir orelse "", .gib = o.spill_gib, .min_free_gib = o.spill_min_free_gib, .min_tokens = o.spill_min_tokens, .quota = o.spill_quota, .turns = o.spill_turns, .read_blocks = o.spill_read_blocks, .stage_mib = o.spill_stage_mib, .hold_bytes = budget / 4 };
}

/// Spill v2's tier on this rank, once the ranks agreed on their identities and on one kept-state budget: this rank's
/// store under its own compat (Agreement.storeCommon, and the layout's owner: TF_GLM_KV_SHARD's owner-sharded store is
/// another namespace) over the staging the engine took before its pool, every rank's open agreed (one that failed
/// fails the start on all), rank 0's tier for the prompt cache, the other ranks' on their lane backends.
fn openTier(own: *Own, gpa: std.mem.Allocator, io: std.Io, o: Options, a: identity.Agreement, budget: u64) !void {
    if (budget == 0) return error.SpillV2NoPromptCache;
    const e = own.engine.?;
    e.current();
    const layout = try e.spillLayout();
    const s = tierSettings(o, budget);
    const compat = layout.compat(a.storeCommon(@intCast(o.rank)));
    const opened = SpillLeader.S.open(gpa, io, e, layout, s, o.rank, compat, @intCast(e.context), e.tierStaging(), .{});
    const ok = if (opened) |_| true else |err| blk: {
        std.log.err("rank {d}: spill v2's store under {s} did not open ({s})", .{ o.rank, s.dir, @errorName(err) });
        break :blk false;
    };
    const all = e.smallestOverRanks(@intFromBool(ok)) catch 0;
    const side = try opened;
    if (all == 0) {
        side.close();
        return error.SpillV2StoreFailed;
    }
    const handed = spill_tier.giveFolders(side.store);
    const c = side.store.counts();
    std.log.info("rank {d}: spill v2 tier: {s}, up to {d:.0} GiB, {d} points ({d} MiB) found; each kept state written at keep from {d} tokens, keeping {d:.0} GiB of disk free{s}{s}", .{ o.rank, s.dir, s.gib, c.points, c.stored_bytes >> 20, s.min_tokens, s.min_free_gib, if (layout.owner == .owner_sharded) "; owner-sharded (TF_GLM_KV_SHARD: a point's latent rows on its home part's rank only)" else "", if (handed) "; its folders given to the spill directory's owner" else "" });
    // a following rank outlives a stop signal this long: rank 0's flush has it drain its own writes first
    e.spill_wait_ns = @as(u64, o.spill_flush_ms) * std.time.ns_per_ms;
    if (o.rank == 0) {
        var boot: [16]u8 = undefined;
        io.random(&boot);
        own.tier = SpillLeader.create(gpa, side, &own.serving.?.local, s, boot) catch |err| {
            side.close();
            return err;
        };
    } else {
        own.follower_tier = SpillFollower.create(gpa, side, &own.serving.?.local) catch |err| {
            side.close();
            return err;
        };
        own.serving.?.local.spill_side = own.follower_tier.?.spillSide();
    }
}

/// Spill v2's identity of this rank (cuda_spill_identity) from the opened engine, agreed with every other rank over the
/// control link: the pair's Agreement (each rank's store opens under its `storeCommon`), or the start fails on every
/// rank, the rank and field named in each one's log.
fn spillAgree(gpa: std.mem.Allocator, io: std.Io, own: *Own, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options, choice: choice_mod.Choice, draft_dir: ?[]const u8, env: identity.Env, kv: shard.Switches, streams: usize) !identity.Agreement {
    const e = own.engine.?;
    const layout = try e.spillLayout();
    var fbuf: [256]u8 = undefined;
    var obuf: [48]u8 = undefined;
    const r: identity.Rank = .{
        .dir = dir,
        .kernels = kernels,
        .drafter = if (!o.drafts) .off else if (choice == .dflash2) .dflash2 else .mtp,
        .draft_dir = draft_dir,
        .draft_quant = envText("TF_GLM_DRAFT_QUANT") orelse "minmax",
        .world = o.tp,
        .rank = o.rank,
        .ring = e.caches.ring,
        .layout = layout.digest(),
        .owner_layout = layout.ownerText(&obuf),
        .capability = try ctx.capability(),
        .driver = try ctx.d.version(),
        .state_format = spill_state.format_version,
        .wire = spill_wire.version,
        .features = try identity.featureString(&fbuf, .{ .spill_min_tokens = o.spill_min_tokens, .kv_shard = kv.on, .drafts = o.drafts, .streams = @intCast(streams), .prefill_rows = @intCast(e.prefill_rows) }),
        .env = env,
    };
    const fp = try identity.build(gpa, io, r);
    var m: identity.Mismatch = .{};
    return identity.agree(own.link.?, gpa, o.rank, o.tp, &fp, &m) catch |err| {
        var buf: [192]u8 = undefined;
        if (err == error.SpillIdentityMismatch) std.log.err("rank {d}: {s}", .{ o.rank, identity.mismatchLine(&buf, m) });
        return err;
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
    if (own.engine.?.shard != null) {
        // TF_GLM_KV_SHARD: a kept state's latent rows are on its home part's owner only, and v1's stored state does not
        // say which part that was (spill v2's owner-sharded tier does); every rank leaves the tier off alike
        if (o.rank == 0) std.log.warn("spill: off with TF_GLM_KV_SHARD=1 (v1 does not record the part a stored state's latent rows lie in; TF_GLM_SPILL_V2=1 does)", .{});
        return null;
    }
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
/// `v2` (TF_GLM_SPILL_V2=1): every rank meets one gather for the smallest budget (a rank's kept states and restores
/// follow rank 0's decisions), `agreed`, and v1's tier stays shut.
fn cacheFor(own: *Own, o: Options, build_key: u64, v2: bool, agreed: *u64) ?lane_backend.CacheHooks {
    agreed.* = 0;
    // MTP drafts: the head's taps pair a continuation's rows with the prompt's tokens from its start (promptTaps,
    // prompt_at), so a prefill that goes on from a kept or live state would misplace them; kept states wait for that
    if (own.engine.?.mtp != null) {
        if (o.rank == 0) std.log.info("prompt cache: off (MTP drafts do not resume from kept states yet)", .{});
        return null;
    }
    // (spill v2: a rank that returns before its plan still meets the gather, with 0)
    var met = false;
    defer if (v2 and !met) {
        _ = agreeBudget(own, 0);
    };
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
    met = true;
    const budget = if (v2) agreeBudget(own, p.budget) else p.budget;
    agreed.* = budget;
    // every rank writes and reads its own copies (rank 0 decides, rank 1 follows its notes)
    const spill = if (budget > 0 and !v2) openSpill(own, o, build_key) else null;
    if (o.rank != 0) return null;
    if (budget == 0) {
        std.log.info("prompt cache: off (budget 0)", .{});
        return null;
    }
    std.log.info("prompt cache: {d:.1} GiB for kept states, at most {d} (fixed {d} MiB each){s}; per chat {d}, bytes cap {d} MiB, shared prefix {d}; {d:.1} GiB reserved, {d:.1} GiB headroom for the prompts' buffers", .{
        @as(f64, @floatFromInt(budget)) / cache_plan.gib,                                          p.entries,                                     e.snapBytes(0) >> 20,
        if (p.entries_capped) " (cut to three quarters of the budget)" else "",                    p.per_chat,                                    p.bytes_cap >> 20,
        p.shared_least,                                                                            @as(f64, @floatFromInt(p.reserve)) / cache_plan.gib,                              @as(f64, @floatFromInt(p.headroom)) / cache_plan.gib,
    });
    return own.serving.?.local.cacheHooks(budget, .{ .grid = 64, .min_gap = 256, .min_prompt = 64, .entries = p.entries, .bytes_cap = p.bytes_cap, .per_chat = p.per_chat, .shared_least = p.shared_least }, if (spill) |sp| sp.highwater else 0, if (spill) |sp| sp.flush_s else 60);
}

/// Spill v2: the smallest kept-state budget over the ranks (one gather; 0 when it cannot be agreed: v2 is then refused).
fn agreeBudget(own: *Own, mine: u64) u64 {
    const e = own.engine.?;
    const all = e.smallestOverRanks(mine) catch |err| {
        std.log.err("rank {d}: spill v2's kept-state budget could not be agreed ({s})", .{ own.rank, @errorName(err) });
        return 0;
    };
    if (all < mine) std.log.info("rank {d}: spill v2: the kept-state budget is {d:.1} GiB, another rank's (this rank's {d:.1} GiB)", .{ own.rank, @as(f64, @floatFromInt(all)) / cache_plan.gib, @as(f64, @floatFromInt(mine)) / cache_plan.gib });
    return all;
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
    /// spill v2's tier: rank 0's (the prompt cache's) or this rank's lane backend's
    tier: ?*SpillLeader = null,
    follower_tier: ?*SpillFollower = null,

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
        // spill v2's tier before the kept states and the engine it holds buffers of (the store's own release already
        // ran: native/cuda.zig deinitializes the prompt cache first)
        if (own.engine) |e| if (own.tier != null or own.follower_tier != null) e.current();
        if (own.tier) |t| t.destroy();
        own.tier = null;
        if (own.follower_tier) |t| {
            if (own.serving) |s| s.local.spill_side = null;
            t.destroy();
        }
        own.follower_tier = null;
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

/// What both ranks must load alike, as text (its sha256 crosses the link). TF_GLM_KV_SHARD's words close it when on (or
/// unreadable), then spill v2's; off, the text is as before.
fn settings(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, kernels: []const u8, o: Options, kv: anyerror!shard.Switches, v2: anyerror!?identity.Env) ![]u8 {
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
    var kv_buf: [80]u8 = undefined;
    const kv_words = if (kv) |sw| sw.text(&kv_buf) else |_| shard.unreadable;
    // spill v2: on (its identity read) with the knobs every rank's store and tier must share, or a rank that cannot read
    // either says so (a word no rank sends when it can)
    var v2_buf: [192]u8 = undefined;
    const v2_words: []const u8 = if (v2) |env| (if (env != null) try spillWords(&v2_buf, o) else "") else |_| " spill_v2=unreadable";
    return std.fmt.allocPrint(gpa, "glm5_next display={d}:{s} cache={s} reserve={s} tp={d} checkpoint={s} snapshot={x} context={d} parallel={d} drafts={} eager={} vision={} tower={} kernels={x}{s}{s}", .{
        display_cfg.bytes, @tagName(display_cfg.kind), memory.env("TF_GLM_CACHE_GIB") orelse memory.env("KV_POOL_GIB") orelse "3", memory.env("TENSORFOLD_MEMORY_RESERVE_GIB") orelse memory.env("MEMORY_RESERVE_GIB") orelse "auto", o.tp, std.fs.path.basename(std.mem.trimEnd(u8, dir, "/")), snapshot.finalResult(), o.context, o.parallel, o.drafts, o.eager, o.vision, o.tower, kernel_set.finalResult(), kv_words, v2_words,
    });
}

/// Spill v2's words in the settings text: on, and what each rank's store and tier must share (its cap and floor, the
/// quota, the turns kept, the staging, the flush bound).
fn spillWords(buf: []u8, o: Options) ![]const u8 {
    return std.fmt.bufPrint(buf, " spill_v2=1 gib={d} free={d} quota={d} turns={d} reads={d} stage={d} flush={d}", .{ o.spill_gib, o.spill_min_free_gib, o.spill_quota, o.spill_turns, o.spill_read_blocks, o.spill_stage_mib, o.spill_flush_ms });
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

test "TF_GLM_KV_SHARD: the settings text carries it only when on, a rank that cannot read it a word no rank sends, and every rank refuses alike after the agreement" {
    const gpa = std.testing.allocator;
    const o: Options = .{ .context = 8192, .drafts = false, .parallel = 4, .tp = 2 };
    const off = try settings(gpa, std.testing.io, "/nonexistent-glm", "/nonexistent-kernels", o, shard.Switches{}, null);
    defer gpa.free(off);
    try std.testing.expect(std.mem.indexOf(u8, off, "kv_shard") == null);
    const on = try settings(gpa, std.testing.io, "/nonexistent-glm", "/nonexistent-kernels", o, shard.Switches{ .on = true }, null);
    defer gpa.free(on);
    try std.testing.expect(std.mem.endsWith(u8, on, " kv_shard=1 carry=1 twins=1 check=1024 lone=0"));
    try std.testing.expectEqualStrings(off, on[0..off.len]); // the same text, the words appended
    const every = try settings(gpa, std.testing.io, "/nonexistent-glm", "/nonexistent-kernels", o, shard.Switches{ .on = true, .check = 64 }, null);
    defer gpa.free(every);
    try std.testing.expect(std.mem.endsWith(u8, every, " check=64 lone=0")); // another digest count: the ranks disagree
    const bad = try settings(gpa, std.testing.io, "/nonexistent-glm", "/nonexistent-kernels", o, error.BadKvShard, null);
    defer gpa.free(bad);
    try std.testing.expect(std.mem.endsWith(u8, bad, shard.unreadable));
    try std.testing.expect(!std.mem.eql(u8, bad, on) and !std.mem.eql(u8, bad, off));
    // after the agreement: off passes; this rank's read error; what sharding cannot serve; on, with this build's GPU paths
    try std.testing.expectEqual(shard.Switches{}, try kvAdmit(shard.Switches{}, .{ .context = 8, .drafts = false }, .dflash2));
    try std.testing.expectError(error.BadKvShardCheck, kvAdmit(error.BadKvShardCheck, o, .dflash2));
    try std.testing.expectError(error.KvShardNeedsTwoRanks, kvAdmit(shard.Switches{ .on = true }, .{ .context = 8, .drafts = false, .parallel = 4 }, .dflash2));
    try std.testing.expectError(error.KvShardNeedsParallel, kvAdmit(shard.Switches{ .on = true }, .{ .context = 8, .drafts = false, .tp = 2 }, .dflash2));
    try std.testing.expectError(error.KvShardWithMtp, kvAdmit(shard.Switches{ .on = true }, .{ .context = 8, .drafts = true, .tp = 2, .parallel = 2 }, .mtp));
    try std.testing.expectError(error.KvShardWithLone, kvAdmit(shard.Switches{ .on = true, .lone = true }, o, .dflash2));
    try std.testing.expectEqual(shard.Switches{ .on = true }, try kvAdmit(shard.Switches{ .on = true }, o, .dflash2));
    try std.testing.expectEqualStrings("TF_GLM_KV_SHARD=1: KV sharding's GPU paths are not in this build yet", refusal(error.KvShardNotBuilt).?);
    try std.testing.expectEqualStrings("TF_GLM_KV_SHARD=1: this build has no glm_shard kernels (its fatbins came without glm_shard.fatbin)", refusal(error.NoKvShardKernels).?);
    try std.testing.expect(refusal(error.OutOfMemory) == null);
}

test "TF_GLM_SPILL_V2: the settings text carries it only when on and after KV sharding's words, a rank that cannot read it or its identity a word no rank sends; two ranks only; every refusal in words" {
    const gpa = std.testing.allocator;
    const o: Options = .{ .context = 8192, .drafts = false, .parallel = 4, .tp = 2, .spill_gib = 64, .spill_dir = "/spill" };
    const env: identity.Env = .{ .engine = "a1b2c3d", .patches = "sha256:0014", .model_revision = "4f2e9a1" };
    const off = try settings(gpa, std.testing.io, "/nonexistent-glm", "/nonexistent-kernels", o, shard.Switches{}, null);
    defer gpa.free(off);
    try std.testing.expect(std.mem.indexOf(u8, off, "spill_v2") == null);
    const on = try settings(gpa, std.testing.io, "/nonexistent-glm", "/nonexistent-kernels", o, shard.Switches{}, @as(?identity.Env, env));
    defer gpa.free(on);
    try std.testing.expect(std.mem.endsWith(u8, on, " spill_v2=1 gib=64 free=50 quota=0 turns=4 reads=16 stage=256 flush=60000"));
    try std.testing.expectEqualStrings(off, on[0..off.len]); // the same text, the words appended
    var other = o;
    other.spill_turns = 0;
    const turns = try settings(gpa, std.testing.io, "/nonexistent-glm", "/nonexistent-kernels", other, shard.Switches{}, @as(?identity.Env, env));
    defer gpa.free(turns);
    try std.testing.expect(!std.mem.eql(u8, turns, on)); // a rank with other knobs: the ranks disagree
    const both = try settings(gpa, std.testing.io, "/nonexistent-glm", "/nonexistent-kernels", o, shard.Switches{ .on = true }, @as(?identity.Env, env));
    defer gpa.free(both);
    try std.testing.expect(std.mem.indexOf(u8, both, " kv_shard=1 carry=1 twins=1 check=1024 lone=0 spill_v2=1 ") != null);
    const bad = try settings(gpa, std.testing.io, "/nonexistent-glm", "/nonexistent-kernels", o, shard.Switches{}, error.NoEngineId);
    defer gpa.free(bad);
    try std.testing.expect(std.mem.endsWith(u8, bad, " spill_v2=unreadable"));
    try std.testing.expect(!std.mem.eql(u8, bad, on) and !std.mem.eql(u8, bad, off));
    // after the agreement: off passes; this rank's own error; a world other than two ranks; MTP drafts; a staging too
    // small for two blocks; on (KV sharding or not: Engine.spillLayout makes the tier owner-sharded)
    try std.testing.expectEqual(@as(?identity.Env, null), try spillAdmit(null, o, .dflash2));
    try std.testing.expectError(error.NoModelRevision, spillAdmit(error.NoModelRevision, o, .dflash2));
    for ([_]u32{ 1, 3, 4 }) |tp| {
        var x = o;
        x.tp = tp;
        try std.testing.expectError(error.SpillV2NeedsTwoRanks, spillAdmit(env, x, .dflash2));
    }
    var mtp = o;
    mtp.drafts = true;
    try std.testing.expectError(error.SpillV2WithMtp, spillAdmit(env, mtp, .mtp));
    try std.testing.expectEqualStrings("a1b2c3d", (try spillAdmit(env, mtp, .dflash2)).?.engine);
    var small = o;
    small.spill_stage_mib = 24;
    try std.testing.expectError(error.StagingTooSmall, spillAdmit(env, small, .dflash2));
    small.spill_stage_mib = 25;
    try std.testing.expectEqualStrings("a1b2c3d", (try spillAdmit(env, small, .dflash2)).?.engine);
    try std.testing.expectEqualStrings("a1b2c3d", (try spillAdmit(env, o, .dflash2)).?.engine);
    // the tier's settings from the server's (the guard's limit a quarter of the agreed budget)
    const ts = tierSettings(.{ .context = 8, .drafts = false, .spill_gib = 64, .spill_dir = "/spill", .spill_quota = 0.25, .spill_turns = 2, .spill_read_blocks = 8, .spill_stage_mib = 128, .spill_min_tokens = 4096, .spill_min_free_gib = 10 }, 8 << 30);
    try std.testing.expectEqualStrings("/spill", ts.dir);
    try std.testing.expectEqual(@as(f64, 64), ts.gib);
    try std.testing.expectEqual(@as(f64, 0.25), ts.quota);
    try std.testing.expectEqual(@as(u32, 2), ts.turns);
    try std.testing.expectEqual(@as(u32, 8), ts.read_blocks);
    try std.testing.expectEqual(@as(u32, 128), ts.stage_mib);
    try std.testing.expectEqual(@as(u32, 4096), ts.min_tokens);
    try std.testing.expectEqual(@as(f64, 10), ts.min_free_gib);
    try std.testing.expectEqual(@as(u64, 2 << 30), ts.hold_bytes);
    // with the spill off (no --spill-gib, or no --snapshot-dir) nothing is read: off whatever the environment holds
    try std.testing.expectEqual(@as(?identity.Env, null), try spillV2(.{ .context = 8, .drafts = false }, .dflash2));
    try std.testing.expectEqual(@as(?identity.Env, null), try spillV2(.{ .context = 8, .drafts = false, .spill_gib = 64 }, .dflash2));
    for ([_]anyerror{ error.BadSpillV2, error.SpillV2NeedsTwoRanks, error.SpillV2WithMtp, error.SpillV2NoPromptCache, error.SpillV2StoreFailed, error.StagingTooSmall, error.NoEngineId, error.NoRecipePatches, error.NoModelRevision, error.NoDrafterRevision, error.SpillIdentityMismatch }) |e|
        try std.testing.expect(std.mem.startsWith(u8, refusal(e).?, "TF_GLM_SPILL_"));
    try std.testing.expectEqualStrings("TF_GLM_KV_SHARD=1: KV sharding's GPU paths are not in this build yet", refusal(error.KvShardNotBuilt).?);
}

test "TF_GLM_KV_SHARD=1 is admitted whatever TF_GLM_SPILL_V2 holds, and spill v2 with it (an owner-sharded tier)" {
    const C = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    const gpa = std.testing.allocator;
    const names = [_][:0]const u8{ "TF_GLM_KV_SHARD", "TF_GLM_SPILL_V2" };
    var was: [names.len]?[:0]u8 = undefined;
    for (names, &was) |n, *w| w.* = if (std.c.getenv(n.ptr)) |v| try gpa.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer for (names, was) |n, w| if (w) |v| {
        _ = C.setenv(n.ptr, v.ptr, 1);
        gpa.free(v);
    } else {
        _ = C.unsetenv(n.ptr);
    };
    _ = C.setenv("TF_GLM_KV_SHARD", "1", 1);
    // no spill tier asked for (no --spill-gib): nothing would spill, so nothing is refused
    const o: Options = .{ .context = 8192, .drafts = false, .parallel = 4, .tp = 2 };
    for ([_][:0]const u8{ "1", " 1 ", "0", "yes" }) |v| {
        _ = C.setenv("TF_GLM_SPILL_V2", v.ptr, 1);
        try std.testing.expect((try kvAdmit(shard.Switches.read(), o, .dflash2)).on);
    }
    // with the spill tier on as well, spill v2 is admitted beside it (round 3: no refusal of the pair)
    const env: identity.Env = .{ .engine = "a1b2c3d", .patches = "sha256:0014", .model_revision = "4f2e9a1" };
    var spilled = o;
    spilled.spill_gib = 64;
    spilled.spill_dir = "/spill";
    try std.testing.expect((try kvAdmit(shard.Switches.read(), spilled, .dflash2)).on);
    try std.testing.expectEqualStrings("a1b2c3d", (try spillAdmit(env, spilled, .dflash2)).?.engine);
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
