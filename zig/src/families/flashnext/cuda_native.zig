//! Flash Next for the native server on CUDA: the engine, its MTP head and the lane backend. native/cuda.zig drives
//! what `open` returns; nothing here knows the server.
//!
//! Two ranks (`tp` 2, work/PLAN.md "Two ranks in serving"): before either loads, rank 0 listens on
//! `master`:`master_port` and rank 1 joins it (cuda_link.zig); both agree on their settings (checkpoint, snapshot,
//! window, drafts, YaRN, kernel set), join the NCCL communicator (cuda_comm.zig) and load their halves. Rank 0's
//! backend is then the mirror (cuda_mirror.zig): each lane call goes to rank 1 before it runs, as a record
//! broadcast over NCCL on the engine's stream (a long prompt on the link); rank 1 runs `follow` instead of serving
//! and makes the same calls on its own engine until rank 0 stops.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const core = @import("core");
const link = @import("cuda_link.zig");
const comm = @import("cuda_comm.zig");
const rope = @import("cuda_rope.zig");
const mirror = @import("cuda_mirror.zig");
const lane_backend = @import("cuda_lanes.zig");

pub const model_type = "qwen4_exp";
/// Weight formats as tensorfold.native.contract.weight_format names them: NVIDIA's ModelOpt NVFP4 export (top-10;
/// its quant_algo MIXED_PRECISION, NVFP4 experts) and
/// INT4-AutoRound's GPTQ int4 (azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound, top-5); the engine reads either
/// from the checkpoint's quantization_config (cuda_config.zig: quant_method "modelopt" or "gptq").
pub const formats: []const []const u8 = &.{ "modelopt-nvfp4", "modelopt-mixed-precision", "gptq-b4" };
/// The checkpoint's native window; 1,048,576 with YaRN (work/PLAN.md, "1M context").
pub const default_context: i64 = 262144;
pub const prefill_step: u32 = 2048;
/// Sizes its own memory, keeps its own prompt states and may run on two ranks: native/cuda.zig openOwn opens it.
pub const own_memory = true;

/// `tp` 2: rank 0 serves and leads, rank 1 follows it over the link to `master`:`master_port` (cuda_link.zig).
/// `parallel`: streams one shared forward holds (--parallel; the engine's Options.streams, the lanes' max_streams).
/// `prompt_cache_gib`: the kept prompt states' budget (null: a quarter of the sequences' memory, 16 GiB at most;
/// 0: no prompt reuse).
pub const Options = struct { context: usize, drafts: bool, parallel: u32 = 1, prompt_cache_gib: ?f64 = null, tp: u32 = 1, rank: u32 = 0, master: []const u8 = "", master_port: u16 = 29551 };

/// The prompt cache's rules for this engine (core/prompt_cache.zig Rules): a kept state depends on its tokens
/// alone (a resume absorbs the next prompt's own token into the MTP head), prompt chunks are chunk-invariant.
pub const cache_rules = .{ .lookahead = 0, .planned = false };

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    /// Full-window sequence storage per admitted stream, for native CUDA memory admission.
    stream_bytes: usize,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    /// rank 1's loop: rank 0's requests in step until it stops
    follow: ?*const fn (*anyopaque) anyerror!void = null,
    /// the prompt cache's family calls and budget (rank 0 or one rank; native/cuda.zig makes the Store)
    cache: ?lane_backend.CacheHooks = null,
};

/// How long either rank waits for the other at the link (the other may still be starting its container).
const link_timeout_ms: u32 = 10 * 60 * 1000;

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    return openOn(@import("cuda_engine.zig"), gpa, io, ctx, dir, kernels, o);
}

/// `open` over the engine module `M` (cuda_engine.zig: M.Engine, M.Options as PLAN.md's Engine API contract).
pub fn openOn(comptime M: type, gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    const Own = Owned(M);
    if (o.tp != 1 and o.tp != 2) return error.UnsupportedTp;
    if (o.rank >= o.tp) return error.BadRank;
    const own = try gpa.create(Own);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .rank = o.rank, .world = o.tp };
    errdefer own.closeLink();
    const yarn = try yarnFor(gpa, io, dir, o.context);
    // the window: 15 drafts (16 rows: deeper MTP chains and copy windows; decode D4, TP=2 served edits +37%, code and
    // prose unchanged at confidence 0.5; aot/fnaot3 holds every width to 16) unless TF_FLASHNEXT_DEPTH says (Python's 6)
    const depth: usize = if (getenv("TF_FLASHNEXT_DEPTH")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadDepth else 15;
    if (depth < 1 or depth > lane_backend.max_drafts) return error.BadDepth;
    if (o.tp == 2) {
        // a rank stops on SIGTERM / SIGINT while it loads and, rank 1, while it follows (in a container it is PID 1,
        // where a signal's default action is ignored); rank 0's server takes them over once it serves
        installStop();
        const address = std.Io.net.IpAddress.parse(o.master, o.master_port) catch {
            std.log.err("--master {s}: not an IP address", .{o.master});
            return error.BadMaster;
        };
        if (o.rank == 0) {
            own.server = try link.Server.open(address);
            std.log.info("rank 0: waiting for rank 1 on {s}:{d}", .{ o.master, o.master_port });
            own.link = try own.server.?.accept(@intCast(link_timeout_ms));
        } else {
            std.log.info("rank 1: joining rank 0 at {s}:{d}", .{ o.master, o.master_port });
            own.link = try link.Link.join(io, address, link_timeout_ms);
        }
        const text = try settings(gpa, io, dir, kernels, o, depth, yarn);
        defer gpa.free(text);
        own.link.?.agree(gpa, o.rank, text) catch |e| {
            if (e == error.RanksDisagree) std.log.err("rank {d} loads with \"{s}\"; the other rank's settings differ (see its log)", .{ o.rank, text });
            return e;
        };
        own.comm = try comm.Comm.init(gpa, own.link.?, o.rank, 2);
        std.log.info("rank {d}: settings agreed, NCCL joined", .{o.rank});
        // from here a lost rank (dead, out of memory, link down) ends this one too, also while it loads
        own.watch = .{ .io = io, .fd = own.link.?.fd, .rank = o.rank, .comm = mirror.Coll.of(comm.Comm, &own.comm.?) };
        try own.watch.?.start();
    }
    const streams: usize = @max(1, @min(o.parallel, lane_backend.max_streams));
    own.e = try M.Engine.init(gpa, io, ctx, dir, kernels, .{
        .context = o.context,
        .depth = depth,
        .streams = streams,
        .mtp = o.drafts,
        .rank = o.rank,
        .world = o.tp,
        .comm = if (own.comm) |*c| c else null,
        .yarn = yarn,
    });
    errdefer own.e.deinit();
    own.lanes = lane_backend.Lanes(M.Engine).init(gpa, own.e, o.drafts, o.context, depth + 1);
    own.lanes.streams = streams;
    own.lanes.confidence = @import("cuda_engine.zig").confidenceSetting();
    own.lanes.product_streams = @import("cuda_engine.zig").productStreams();
    own.lanes.wide_confidence = @import("cuda_engine.zig").wide_confidence;
    own.lanes.overcommit = if (getenv("TF_FLASHNEXT_OVERCOMMIT")) |v| std.mem.eql(u8, v, "1") else false;
    // prompts fill in layer slices between rounds (the engine's fillBegin / fillStep): opt-in until the sliced
    // fill is verified exact on GPUs; both ranks agree on it
    own.lanes.sliced = if (getenv("TF_FLASHNEXT_FILL")) |v| std.mem.eql(u8, v, "1") else false;
    if (getenv("TF_FLASHNEXT_FILL_LAYERS")) |v| own.lanes.fill_layers = @max(1, std.fmt.parseInt(usize, v, 10) catch 8);
    errdefer own.lanes.deinit();
    if (o.drafts) {
        const ids = try costText(gpa, io, dir);
        defer gpa.free(ids);
        try own.lanes.measure(io, ids);
    }
    var backend = own.lanes.backend();
    own.lanes.follower = o.tp == 2 and o.rank == 1;
    if (o.tp == 2) {
        own.bcast = try mirror.Broadcast.init(ctx.d, mirror.Coll.of(comm.Comm, &own.comm.?), streamOf(own.e));
        own.agreement = try mirror.Agreement.init(ctx.d, mirror.Coll.of(comm.Comm, &own.comm.?), streamOf(own.e), &own.watch.?);
        own.lanes.gate = own.agreement.?.gate();
        own.records = .{ .gpa = gpa, .wire = own.bcast.?.wire(), .link = own.link.?, .watch = &own.watch.? };
        if (o.rank == 0) {
            own.leader = mirror.Leader.init(gpa, backend, own.records.?.channel());
            own.leader.?.watch = &own.watch.?;
            own.lanes.notes = own.leader.?.notes();
            backend = own.leader.?.backend();
        }
    }
    return .{
        .backend = backend,
        .facts = own.lanes.facts(),
        .stream_bytes = if (@hasDecl(M.Engine, "seqBytes")) own.e.seqBytes(o.context + depth + 1) else 0,
        .rows = if (o.drafts) @intCast(depth + 1) else 1,
        .ctx = own,
        .deinit = Own.release,
        .follow = if (o.tp == 2 and o.rank == 1) Own.follow else null,
        .cache = if (o.rank == 0) own.lanes.cacheHooks(cacheBudget(o, &own.lanes)) else null,
    };
}

fn Owned(comptime M: type) type {
    return struct {
        const Self = @This();
        gpa: std.mem.Allocator,
        rank: u32,
        world: u32,
        e: *M.Engine = undefined,
        lanes: lane_backend.Lanes(M.Engine) = undefined,
        server: ?link.Server = null,
        link: ?link.Link = null,
        comm: ?comm.Comm = null,
        bcast: ?mirror.Broadcast = null,
        agreement: ?mirror.Agreement = null,
        records: ?mirror.Records = null,
        leader: ?mirror.Leader = null,
        watch: ?mirror.Watch = null,

        fn freeWire(own: *Self) void {
            if (own.bcast) |*b| b.deinit();
            own.bcast = null;
            if (own.agreement) |*g| g.deinit();
            own.agreement = null;
        }

        fn closeLink(own: *Self) void {
            own.freeWire();
            if (own.watch) |*w| {
                w.quiet();
                w.deinit();
            }
            own.watch = null;
            if (own.comm) |*c| c.deinit();
            own.comm = null;
            if (own.link) |l| l.close();
            own.link = null;
            if (own.server) |s| s.close();
            own.server = null;
        }

        fn release(p: *anyopaque) void {
            const own: *Self = @ptrCast(@alignCast(p));
            if (own.watch) |*w| w.quiet(); // a clean stop: the other rank closing its link is no loss now
            if (own.leader) |*l| l.deinit(); // rank 1 stops
            if (own.records) |*r| r.deinit();
            own.freeWire();
            if (own.watch) |*w| w.deinit();
            own.lanes.deinit();
            own.e.deinit();
            own.closeLink();
            own.gpa.destroy(own);
        }

        fn noteFn(p: *anyopaque, bytes: []const u8) anyerror!void {
            const l: *lane_backend.Lanes(M.Engine) = @ptrCast(@alignCast(p));
            return l.applyNote(bytes);
        }

        fn savedFn(p: *anyopaque, key: u64) ?*anyopaque {
            const l: *lane_backend.Lanes(M.Engine) = @ptrCast(@alignCast(p));
            return l.savedFor(key);
        }

        fn follow(p: *anyopaque) anyerror!void {
            const own: *Self = @ptrCast(@alignCast(p));
            const records = if (own.records) |*r| r else return error.NoLink;
            var f = mirror.Follower.init(own.gpa, own.lanes.backend(), records.channel());
            f.watch = records.watch;
            f.family = .{ .ptr = &own.lanes, .note = noteFn, .saved = savedFn };
            defer f.deinit();
            try f.run();
            std.log.info("rank 1: rank 0 stopped after {d} calls", .{f.calls});
        }
    };
}

/// SIGTERM / SIGINT end the process at once (exit 0): the other rank sees the link close and exits too. Only
/// async-signal-safe calls (write, _exit): the main thread may be waiting in a collective or on the link.
fn installStop() void {
    const posix = std.posix;
    const stop: posix.Sigaction = .{ .handler = .{ .handler = onStop }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(.TERM, &stop, null);
    posix.sigaction(.INT, &stop, null);
}

fn onStop(_: std.posix.SIG) callconv(.c) void {
    const line = "tensorfold: stop signal: this rank exits, the other follows\n";
    _ = std.c.write(2, line, line.len);
    std.c._exit(0);
}

/// The kept prompt states' budget: --prompt-cache-gib, else a quarter of the sequences' memory (16 GiB at most).
fn cacheBudget(o: Options, l: anytype) u64 {
    if (o.prompt_cache_gib) |g| return if (g > 0) @intFromFloat(g * (1 << 30)) else 0;
    const seqs: u64 = @min(l.budget(), std.math.maxInt(u64) / 2);
    return @min(seqs / 4, 16 << 30);
}

/// The engine's stream (a cuda.Stream or its handle): the mirror's records run in its order.
fn streamOf(e: anytype) cuda.abi.Stream {
    const s = e.stream;
    return if (@TypeOf(s) == cuda.Stream) s.handle else @ptrCast(s);
}

/// The text both ranks' costs are timed over (the lane rounds' own words: prose and code).
const cost_text = "The river had been rising for three days, and by the time the ferry stopped running the town had moved its " ++
    "market up the hill. Children carried baskets of apples past the church while their parents argued about " ++
    "whether the old bridge would hold. In the workshop behind the bakery, a carpenter measured each plank " ++
    "twice, wrote the numbers on the wall, and cut slowly.\n\ndef mean(values):\n    total = 0\n    for v in " ++
    "values:\n        total += v\n    return total / len(values)\n";

fn costText(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![]u32 {
    const path = try std.fs.path.join(gpa, &.{ dir, "tokenizer.json" });
    defer gpa.free(path);
    var tok = try core.tokenizer.loadTokenizer(io, gpa, path);
    defer tok.deinit();
    return tok.encode(gpa, cost_text);
}

/// The window this checkpoint serves (native/cuda.zig's admission): YaRN's factor x original when the config or
/// TF_FLASHNEXT_YARN turns it on, else max_position_embeddings; 0 when config.json cannot be read.
pub fn modelWindow(a: std.mem.Allocator, io: std.Io, dir: []const u8) i64 {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return 0;
    defer a.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24)) catch return 0;
    defer a.free(text);
    return windowOf(a, text, getenv(rope.env_factor), getenv(rope.env_ramp)) catch 0;
}

fn windowOf(a: std.mem.Allocator, config_text: []const u8, factor_env: ?[]const u8, ramp_env: ?[]const u8) !i64 {
    if (try yarnOf(a, config_text, 0, factor_env, ramp_env)) |y| return @intCast(y.window());
    const doc = try std.json.parseFromSlice(std.json.Value, a, config_text, .{});
    defer doc.deinit();
    if (doc.value != .object) return 0;
    const t = if (doc.value.object.get("text_config")) |x| (if (x == .object) x.object else doc.value.object) else doc.value.object;
    const v = t.get("max_position_embeddings") orelse return 0;
    return if (v == .integer and v.integer > 0) v.integer else 0;
}

/// YaRN for this run: the checkpoint's rope_parameters or TF_FLASHNEXT_YARN (cuda_rope.fromConfig), else past the
/// native window the factor that reaches `context`; null below it.
fn yarnFor(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, context: usize) !?rope.Yarn {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    return yarnOf(gpa, text, context, getenv(rope.env_factor), getenv(rope.env_ramp));
}

fn getenv(name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

fn yarnOf(gpa: std.mem.Allocator, config_text: []const u8, context: usize, factor_env: ?[]const u8, ramp_env: ?[]const u8) !?rope.Yarn {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, config_text, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadConfig;
    const text_cfg = if (doc.value.object.get("text_config")) |t| (if (t == .object) t.object else doc.value.object) else doc.value.object;
    const native: u64 = if (text_cfg.get("max_position_embeddings")) |v| (if (v == .integer and v.integer > 0) @intCast(v.integer) else 262144) else 262144;
    const empty: std.json.ObjectMap = .empty;
    const params = if (text_cfg.get("rope_parameters") orelse text_cfg.get("rope_scaling")) |r| (if (r == .object) r.object else empty) else empty;
    if (try rope.fromConfig(params, native, factor_env, ramp_env)) |y| return y;
    // TF_FLASHNEXT_YARN=off keeps the default rope: a window past the native one is then the engine's refusal
    const off = if (factor_env) |v| std.mem.trim(u8, v, " \t\r\n").len > 0 else false;
    if (off or context <= native) return null;
    return .{ .factor = @as(f64, @floatFromInt(context)) / @as(f64, @floatFromInt(native)), .original = native };
}

/// The canonical settings text both ranks must load with (its digest is compared over the link).
fn settings(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, kernels: []const u8, o: Options, depth: usize, yarn: ?rope.Yarn) ![]u8 {
    var snapshot = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_][]const u8{ "config.json", "model.safetensors.index.json", "tokenizer.json" }) |name| try hashFile(gpa, io, &snapshot, dir, name);
    var kernel_set = std.crypto.hash.sha2.Sha256.init(.{});
    try hashFile(gpa, io, &kernel_set, kernels, "aot.json");
    return settingsText(gpa, std.fs.path.basename(std.mem.trimEnd(u8, dir, "/")), snapshot.finalResult(), kernel_set.finalResult(), o, depth, yarn);
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

fn settingsText(gpa: std.mem.Allocator, base: []const u8, snapshot: [32]u8, kernel_set: [32]u8, o: Options, depth: usize, yarn: ?rope.Yarn) ![]u8 {
    var y: [96]u8 = undefined;
    const yt = if (yarn) |v| try std.fmt.bufPrint(&y, "{d}x{d}/ramp{d}/att{d}", .{ v.factor, v.original, v.ramp_positions, v.attention() }) else "off";
    const fill = getenv("TF_FLASHNEXT_FILL") orelse "0";
    const fill_layers = getenv("TF_FLASHNEXT_FILL_LAYERS") orelse "8";
    return std.fmt.allocPrint(gpa, "flashnext tp={d} checkpoint={s} snapshot={x} context={d} parallel={d} drafts={} depth={d} confidence={d}/{d} yarn={s} fill={s}/{s} kernels={x}", .{
        o.tp, base, snapshot, o.context, o.parallel, o.drafts, depth, @import("cuda_engine.zig").confidenceSetting(), @import("cuda_engine.zig").productStreams(), yt, fill, fill_layers, kernel_set,
    });
}

test {
    _ = mirror;
    _ = lane_backend;
}

test "open compiles over the engine" {
    std.testing.refAllDecls(@This());
    _ = &openOn;
    const f = openOn;
    _ = f;
    if (false) _ = try open(undefined, undefined, undefined, "", "", .{ .context = 1, .drafts = true });
}

test "YaRN comes from the checkpoint, the environment or a window past the native one" {
    const gpa = std.testing.allocator;
    const cfg = @embedFile("fixtures_cuda_config.json");
    try std.testing.expect((try yarnOf(gpa, cfg, 262144, null, null)) == null);
    const forced = (try yarnOf(gpa, cfg, 262144, "4.0", null)).?;
    try std.testing.expectEqual(@as(f64, 4), forced.factor);
    try std.testing.expectEqual(@as(u64, 262144), forced.original);
    try std.testing.expectEqual(@as(u64, 1 << 20), forced.window());
    try std.testing.expect((try yarnOf(gpa, cfg, 1 << 20, "off", null)) == null);
    const past = (try yarnOf(gpa, cfg, 1 << 20, null, null)).?;
    try std.testing.expectEqual(@as(u64, 1 << 20), past.window());
}

test "the served window is YaRN's with TF_FLASHNEXT_YARN, else the config's" {
    const gpa = std.testing.allocator;
    const cfg = @embedFile("fixtures_cuda_config.json");
    try std.testing.expectEqual(@as(i64, 262144), try windowOf(gpa, cfg, null, null));
    try std.testing.expectEqual(@as(i64, 1 << 20), try windowOf(gpa, cfg, "4.0", null));
    try std.testing.expectEqual(@as(i64, 262144), try windowOf(gpa, cfg, "off", null));
}

test "the settings text names everything both ranks must share" {
    const gpa = std.testing.allocator;
    const a = try settingsText(gpa, "fc694b54", @splat(1), @splat(2), .{ .context = 32768, .drafts = true, .tp = 2, .rank = 0 }, 6, null);
    defer gpa.free(a);
    // rank 1's text differs only where its settings do: the rank itself is not part of it
    const b = try settingsText(gpa, "fc694b54", @splat(1), @splat(2), .{ .context = 32768, .drafts = true, .tp = 2, .rank = 1, .master = "10.0.34.1" }, 6, null);
    defer gpa.free(b);
    try std.testing.expectEqualStrings(a, b);
    const c = try settingsText(gpa, "fc694b54", @splat(1), @splat(2), .{ .context = 65536, .drafts = true, .tp = 2 }, 6, .{ .factor = 4, .original = 262144 });
    defer gpa.free(c);
    try std.testing.expect(!std.mem.eql(u8, a, c));
    try std.testing.expect(std.mem.indexOf(u8, a, "context=32768") != null);
    try std.testing.expect(std.mem.indexOf(u8, a, "yarn=off") != null);
    try std.testing.expect(std.mem.indexOf(u8, c, "yarn=4x262144") != null);
}

test "open on the toy engine: one rank serves its lanes, the records and follower compile for two" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = @embedFile("fixtures_cuda_config.json") });
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const M = struct {
        pub const Engine = lane_backend.Toy;
    };
    const ctx: *const cuda.Context = undefined;
    const loaded = try openOn(M, gpa, io, ctx, dir, "kernels", .{ .context = 4096, .drafts = false });
    defer loaded.deinit(loaded.ctx);
    try std.testing.expect(loaded.follow == null);
    try std.testing.expectEqual(@as(u32, 1), loaded.rows);
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &.{ 1, 2, 3 }, .max_new = 8 });
    defer s.deinit(gpa);
    const first = try loaded.backend.opening(gpa, &s);
    try std.testing.expectEqual(lane_backend.Toy.next(&.{ 1, 2, 3 }, &.{}, null, 3), first);
    loaded.backend.release(&s);
    try std.testing.expectError(error.BadRank, openOn(M, gpa, io, ctx, dir, "kernels", .{ .context = 4096, .drafts = false, .tp = 2, .rank = 2 }));
    std.testing.refAllDecls(mirror.Records);
    std.testing.refAllDecls(mirror.Broadcast);
    std.testing.refAllDecls(mirror.Agreement);
}
