//! `tensorfold` for Flash Next (qwen4_exp) on CUDA: token-id runs and the exactness checks against the Python
//! engine's capture (zig/tests/cuda/flashnext/capture.py: weights.json, teacher.json, prefill/ and teacher/ dumps,
//! prompts.json, results.json). cuda_main.zig hands a checkpoint whose config.json says qwen4_exp here.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const flashnext = @import("flashnext");

const Engine = flashnext.Engine;
const decode = flashnext.decode;
const fwd = flashnext.forward;
const Allocator = std.mem.Allocator;

const usage =
    \\usage (Flash Next, qwen4_exp):
    \\  tensorfold run MODEL --tokens ID,ID,... [--tokens-file PATH] [--max-tokens N] [--no-drafts] [--report PATH]
    \\           [--temperature T --top-k K --top-p P --min-p M --seed S] [--ignore-eos] [--context N] [--kernels DIR]
    \\  tensorfold teacher MODEL TEACHER.json [--dump CAPTURE/teacher]   (48 teacher-forced steps: tokens, logits,
    \\           MTP draws; with --dump every step's per-layer digests)
    \\  tensorfold prefill MODEL PROMPTS.json NAME [--dump CAPTURE/prefill/NAME] [--teacher TEACHER.json]
    \\  tensorfold check-weights MODEL WEIGHTS.json   (every device buffer's sha256 against the oracle's)
    \\  tensorfold gate-many MODEL CAPTURE_DIR --streams N [--max-tokens N]   (N streams in shared rounds mixing the
    \\           captured prompts and rules: all drafted, all serial, then mixed; each stream = its solo run)
    \\           [--arrive K --fill-layers L]   (odd streams arrive at rounds K, 2K, ... and fill their prompts L
    \\           layers a round between the others' rounds) [--burst]   (the streams present at the start prefill
    \\           together, one prompt pass over each group that fits)
    \\  tensorfold reuse MODEL CAPTURE_DIR [--only NAME,...]   (kept prompt states: a resumed next turn and a copied
    \\           prefix in another sequence decode as a fresh prefill does)
    \\  tensorfold bench-many MODEL --streams MAX --kind code,prose,chat,structured,dash-prose,dash-code,dash-structured [--rows N,N,...] [--reps K] [--max-tokens N]
    \\           [--ignore-eos] [--no-drafts] [--depth D] [--served]   (N streams of a kind's prompts in shared rounds, for each N in --rows:
    \\           aggregate tok/s; code greedy, prose and chat sampled; TF_FLASHNEXT_PROFILE=1 times rounds by part)
    \\  tensorfold pool MODEL --streams N   (N sequences grown in turn until the memory budget refuses: the tokens
    \\           the rank's caches hold at N streams)
    \\  tensorfold bench MODEL --tokens-file PATH[,PATH...] [--reps N] [--max-tokens N] [--no-drafts]   (each
    \\           prompt N times in one load: prefill seconds and tok/s, the reply's first token and sha)
    \\  tensorfold mm-check MODEL [--rows N,N,...]   (no weights: the prompt matmuls' in-program K
    \\           slices against Python's split kernels + _reduce, bytes compared, then their speed)
    \\  tensorfold qsa-check MODEL   (no weights: the prompt indexer's _scores_rows and tiled select against Python's)
    \\  tensorfold fp4-check MODEL [--rows N,N,...]   (no weights: the shared expert's single-slice NVFP4 matmul on
    \\           fn_ops' K-serial kernel against Triton's _fp4mm, bytes compared, then both timed)
    \\  tensorfold experts-check MODEL [--rows N,N,...]   (no weights: the prompt expert kernel on larger items
    \\           against Python's on 16-pair items, random experts and skewed routing, bytes compared, timed)
    \\  tensorfold fp8-check MODEL DIR   (no weights: the block-FP8 lane matmul (cuda_fp8.zig, qmmf FP8G) against
    \\           Python's Fp8BlockLinear on tools/zig/check_fp8block.py's cases in DIR, bytes compared, timed)
    \\  tensorfold int4-check MODEL   (no weights: the GPTQ int4 expert and head kernels (cuda_int4.zig) against a host
    \\           reference from the GPTQ bytes, row/tile invariance, plan == dense, then their speed)
    \\  every command: [--tp 2 --rank R --master ADDR --master-port PORT] (both ranks run the same command)
    \\  tensorfold gate MODEL CAPTURE_DIR [--max-tokens N] [--only NAME,...] [--report PATH]   (every captured
    \\           prompt serial and drafted, greedy and sampled, against results.json; drafted == serial)
    \\
;

/// Whether `dir`'s config.json names a Flash Next model type (cuda_config.model_types).
pub fn wants(gpa: Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(gpa, &.{ dir, "config.json" }) catch return false;
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24)) catch return false;
    defer gpa.free(text);
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch return false;
    defer parsed.deinit();
    const mt = parsed.value.object.get("model_type") orelse return false;
    if (mt != .string) return false;
    for (flashnext.config.model_types) |m| if (std.mem.eql(u8, m, mt.string)) return true;
    return false;
}

const Options = struct {
    model: []const u8,
    tokens: []u32 = &.{},
    max_tokens: usize = 256,
    sampling: lanes.Sampling = .{ .seed = 0, .temperature = 0 },
    drafts: bool = true,
    report: ?[]const u8 = null,
    kernels: ?[]const u8 = null,
    device: u32 = 0,
    dump: ?[]const u8 = null,
    teacher: ?[]const u8 = null,
    only: ?[]const u8 = null,
    context: usize = 262144,
    stop_eos: bool = true,
    graphs: bool = true,
    tp: u32 = 1,
    streams: usize = 1,
    kv_gib: ?f64 = null,
    reserve_gib: f64 = 12,
    repeat: usize = 1,
    against_solo: bool = false,
    arrive: usize = 0,
    fill_layers: usize = 0,
    burst: bool = false,
    rank: u32 = 0,
    master: []const u8 = "",
    master_port: u16 = 29551,
    reps: usize = 1,
    rows: []const u8 = "",
    kind: []const u8 = "code",
    depth: usize = 6,
    /// bench-many: the served confidence rule (confidenceSetting) instead of Python's 0.70
    served: bool = false,
};

fn parseIds(gpa: Allocator, text: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n[]");
    while (it.next()) |t| try out.append(gpa, try std.fmt.parseInt(u32, t, 10));
    return out.toOwnedSlice(gpa);
}

pub fn main(init: std.process.Init, args: []const []const u8) !u8 {
    const gpa = init.gpa;
    if (args.len < 3) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var o: Options = .{ .model = args[2] };
    defer gpa.free(o.tokens);
    var positional: std.ArrayList([]const u8) = .empty;
    defer positional.deinit(gpa);
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        const v = if (i + 1 < args.len) args[i + 1] else "";
        if (std.mem.eql(u8, a, "--tokens")) {
            gpa.free(o.tokens);
            o.tokens = try parseIds(gpa, v);
            i += 1;
        } else if (std.mem.eql(u8, a, "--tokens-file") and std.mem.eql(u8, args[1], "bench")) {
            try positional.append(gpa, v);
            i += 1;
        } else if (std.mem.eql(u8, a, "--tokens-file")) {
            const text = try std.Io.Dir.cwd().readFileAlloc(init.io, v, gpa, .limited(1 << 26));
            defer gpa.free(text);
            gpa.free(o.tokens);
            o.tokens = try parseIds(gpa, text);
            i += 1;
        } else if (std.mem.eql(u8, a, "--max-tokens")) {
            o.max_tokens = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--temperature")) {
            o.sampling.temperature = try std.fmt.parseFloat(f64, v);
            i += 1;
        } else if (std.mem.eql(u8, a, "--top-k")) {
            o.sampling.top_k = try std.fmt.parseInt(u32, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--top-p")) {
            o.sampling.top_p = try std.fmt.parseFloat(f64, v);
            i += 1;
        } else if (std.mem.eql(u8, a, "--min-p")) {
            o.sampling.min_p = try std.fmt.parseFloat(f64, v);
            i += 1;
        } else if (std.mem.eql(u8, a, "--seed")) {
            o.sampling.seed = try std.fmt.parseInt(u64, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--report")) {
            o.report = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--kernels")) {
            o.kernels = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--device")) {
            o.device = try std.fmt.parseInt(u32, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--dump")) {
            o.dump = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--teacher")) {
            o.teacher = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--only")) {
            o.only = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--context")) {
            o.context = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--kv-gib")) {
            o.kv_gib = try std.fmt.parseFloat(f64, v);
            i += 1;
        } else if (std.mem.eql(u8, a, "--repeat")) {
            o.repeat = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--arrive")) {
            o.arrive = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--fill-layers")) {
            o.fill_layers = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--burst")) {
            o.burst = true;
        } else if (std.mem.eql(u8, a, "--against-solo")) {
            o.against_solo = true;
        } else if (std.mem.eql(u8, a, "--streams")) {
            o.streams = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--tp")) {
            o.tp = try std.fmt.parseInt(u32, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--rank")) {
            o.rank = try std.fmt.parseInt(u32, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--master")) {
            o.master = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--master-port")) {
            o.master_port = try std.fmt.parseInt(u16, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--reps")) {
            o.reps = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--depth")) {
            o.depth = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--served")) {
            o.served = true;
        } else if (std.mem.eql(u8, a, "--kind")) {
            o.kind = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--rows")) {
            o.rows = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--no-drafts")) {
            o.drafts = false;
        } else if (std.mem.eql(u8, a, "--ignore-eos")) {
            o.stop_eos = false;
        } else if (std.mem.eql(u8, a, "--eager")) {
            o.graphs = false;
        } else if (std.mem.startsWith(u8, a, "--")) {
            std.debug.print("unknown option {s}\n{s}", .{ a, usage });
            return 2;
        } else try positional.append(gpa, a);
    }
    const kernels = o.kernels orelse init.environ_map.get("TENSORFOLD_CUDA_KERNELS") orelse {
        std.debug.print("the Triton kernel set: --kernels DIR or TENSORFOLD_CUDA_KERNELS (aot.json + cubins/)\n", .{});
        return 2;
    };
    const cmd = args[1];
    const rest = positional.items;
    if (init.environ_map.get("TENSORFOLD_MEMORY_RESERVE_GIB")) |v| o.reserve_gib = try std.fmt.parseFloat(f64, v);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, @intCast(o.device));
    defer ctx.deinit();
    if (std.mem.eql(u8, cmd, "mm-check")) return mmCheck(gpa, init.io, &ctx, kernels, o);
    if (std.mem.eql(u8, cmd, "experts-check")) return expertsCheck(gpa, &ctx, o);
    if (std.mem.eql(u8, cmd, "fp8-check") and rest.len == 1) {
        const ok = try flashnext.fp8.check(gpa, init.io, &ctx, rest[0]);
        std.debug.print("{s} fp8-check: the block-FP8 lane matmul {s} Python's Fp8BlockLinear\n", .{ if (ok) "PASS" else "FAIL", if (ok) "byte-equal to" else "DIFFERS from" });
        return if (ok) 0 else 1;
    }
    if (std.mem.eql(u8, cmd, "int4-check")) {
        var k = try flashnext.kernels.Kernels.load(&ctx);
        defer k.deinit();
        var stream = try cuda.Stream.init(ctx.d, true);
        defer stream.deinit();
        const ok = try flashnext.int4_check.check(gpa, &ctx, .{ .k = &k, .s = stream });
        std.debug.print("{s} int4-check: the GPTQ int4 kernels against the host order, row and tile invariance, plan == dense\n", .{if (ok) "PASS" else "FAIL"});
        return if (ok) 0 else 1;
    }
    if (std.mem.eql(u8, cmd, "qsa-check")) return qsaCheck(gpa, init.io, &ctx, kernels);
    if (std.mem.eql(u8, cmd, "fp4-check")) return fp4Check(gpa, init.io, &ctx, kernels, o);
    const mtp = !std.mem.eql(u8, cmd, "check-weights-nomtp");
    // two ranks: both run the same command (every engine call in the same order), rank 0 listening on --master
    var server: ?flashnext.link.Server = null;
    defer if (server) |sv| sv.close();
    var lk: ?flashnext.link.Link = null;
    defer if (lk) |l| l.close();
    var comm: ?flashnext.comm.Comm = null;
    defer if (comm) |*c| c.deinit();
    if (o.tp == 2) {
        const address = try std.Io.net.IpAddress.parse(o.master, o.master_port);
        if (o.rank == 0) {
            server = try flashnext.link.Server.open(address);
            std.debug.print("rank 0: waiting for rank 1 on {s}:{d}\n", .{ o.master, o.master_port });
            lk = try server.?.accept(600_000);
        } else {
            lk = try flashnext.link.Link.join(init.io, address, 600_000);
        }
        var sb: [256]u8 = undefined;
        try lk.?.agree(gpa, o.rank, try std.fmt.bufPrint(&sb, "{s} {s} context {d} mtp {}", .{ cmd, std.fs.path.basename(o.model), o.context, mtp }));
        comm = try flashnext.comm.Comm.init(gpa, lk.?, o.rank, 2);
        std.debug.print("rank {d} of 2: NCCL joined\n", .{o.rank});
    } else if (o.tp != 1) return error.BadTp;
    std.debug.print("loading {s} (kernels {s})\n", .{ o.model, kernels });
    const yarn = try yarnFor(gpa, init.io, o.model, init.environ_map.get(flashnext.rope.env_factor), init.environ_map.get(flashnext.rope.env_ramp));
    if (yarn) |y| std.debug.print("YaRN: factor {d} over {d} positions\n", .{ y.factor, y.original });
    const e = try Engine.init(gpa, init.io, &ctx, o.model, kernels, .{ .context = o.context, .mtp = mtp, .graphs = o.graphs, .rank = o.rank, .world = o.tp, .comm = if (comm) |*c| c else null, .yarn = yarn, .streams = o.streams, .depth = o.depth, .reserve_gib = o.reserve_gib, .kv_budget = if (o.kv_gib) |g| @as(usize, @intFromFloat(g * (1 << 30))) else null });
    if (o.graphs and (std.mem.eql(u8, cmd, "gate") or std.mem.eql(u8, cmd, "run"))) {
        const t0 = now(init.io);
        const n = try e.warmGraphs();
        std.debug.print("{d} decode graphs captured in {d:.1}s\n", .{ n, since(init.io, t0) });
    }
    std.debug.print("sequence memory {d:.2} GiB, {d:.2} GiB a {d}-row sequence\n", .{ @as(f64, @floatFromInt(e.budget.limit)) / (1 << 30), @as(f64, @floatFromInt(e.seqBytes(e.max_len))) / (1 << 30), e.max_len });
    defer e.deinit();
    std.debug.print("loaded in {d:.1}s: {d} weight buffers, {d:.2} GiB; cache rows {d}\n", .{ e.load_seconds, e.w.named.items.len, @as(f64, @floatFromInt(e.w.bytes)) / (1 << 30), e.max_len });
    if (std.mem.eql(u8, cmd, "run")) return run(gpa, init.io, e, o);
    if (std.mem.eql(u8, cmd, "bench")) return bench(gpa, init.io, e, o, positional.items);
    if (std.mem.eql(u8, cmd, "teacher") and rest.len == 1) return teacher(gpa, init.io, e, rest[0], o.dump, o.drafts);
    if (std.mem.eql(u8, cmd, "prefill") and rest.len == 2) return prefillCheck(gpa, init.io, e, rest[0], rest[1], o.dump, o.teacher);
    if (std.mem.eql(u8, cmd, "check-weights") and rest.len == 1) return checkWeights(gpa, init.io, e, rest[0]);
    if (std.mem.eql(u8, cmd, "gate") and rest.len == 1) return gate(gpa, init.io, e, rest[0], o);
    // --served: the server's hybrid stop rule in generateMany too (TF_FLASHNEXT_PRODUCT_STREAMS)
    if (o.served) e.product_streams = flashnext.engine.productStreams();
    if (std.mem.eql(u8, cmd, "gate-many") and rest.len == 1) return gateMany(gpa, init.io, e, rest[0], o);
    if (std.mem.eql(u8, cmd, "bench-many")) return benchMany(gpa, init.io, e, o);
    if (std.mem.eql(u8, cmd, "reuse") and rest.len == 1) return reuse(gpa, init.io, e, rest[0], o);
    if (std.mem.eql(u8, cmd, "pool")) return pool(gpa, e, o);
    std.debug.print("{s}", .{usage});
    return 2;
}

/// YaRN from the config's rope parameters or TF_FLASHNEXT_YARN (cuda_rope.fromConfig, the patched Python's rule).
fn yarnFor(gpa: Allocator, io: std.Io, dir: []const u8, factor_env: ?[]const u8, ramp_env: ?[]const u8) !?flashnext.rope.Yarn {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const doc = try readJson(gpa, io, path);
    defer doc.deinit();
    const top = doc.value.object;
    const text_cfg = if (top.get("text_config")) |t| (if (t == .object) t.object else top) else top;
    const native: u64 = if (text_cfg.get("max_position_embeddings")) |v| (if (v == .integer and v.integer > 0) @intCast(v.integer) else 262144) else 262144;
    const empty: std.json.ObjectMap = .empty;
    const params = if (text_cfg.get("rope_parameters") orelse text_cfg.get("rope_scaling")) |r| (if (r == .object) r.object else empty) else empty;
    return flashnext.rope.fromConfig(params, native, factor_env, ramp_env);
}

fn readJson(gpa: Allocator, io: std.Io, path: []const u8) !std.json.Parsed(std.json.Value) {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(text);
    return std.json.parseFromSlice(std.json.Value, gpa, text, .{});
}

fn shaHex(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

fn deviceSha(gpa: Allocator, e: *Engine, ptr: u64, len: usize) ![64]u8 {
    const host = try gpa.alloc(u8, len);
    defer gpa.free(host);
    try e.stream.synchronize();
    const buf: cuda.DeviceBuffer = .{ .d = e.ctx.d, .ptr = ptr, .len = len };
    try buf.download(0, host);
    return shaHex(host);
}

fn now(io: std.Io) std.Io.Timestamp {
    return std.Io.Clock.awake.now(io);
}

fn since(io: std.Io, t: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(now(io).toNanoseconds() - t.toNanoseconds())) / 1e9;
}

pub const Generated = struct {
    tokens: []u32,
    prefill_s: f64,
    decode_s: f64,
    rounds: usize,
    drafted: usize,
    accepted: usize,
};

/// Python Engine.generate on one GPU: `_serial` (draft false: a prefill without the MTP head, one token a round)
/// or `_decode` (a prefill absorbing every chunk, then MTP-drafted rounds), the reply capped by the window.
pub fn generate(gpa: Allocator, io: std.Io, e: *Engine, prompt: []const u32, max_tokens: usize, sampling: ?lanes.Sampling, drafts: bool, stop_eos: bool) !Generated {
    const room = @as(i64, @intCast(e.max_len)) - @as(i64, @intCast(prompt.len)) - @as(i64, @intCast(e.depth)) - 1;
    if (room < 1) return error.PromptPastWindow;
    const count = @max(1, @min(max_tokens, @as(usize, @intCast(room))));
    e.absorbing = drafts;
    const t0 = now(io);
    const first = try e.prefill(prompt, sampling);
    try e.stream.synchronize();
    const prefill_s = since(io, t0);
    if ((stop_eos and e.isEos(first)) or count <= 1) {
        const one = try gpa.alloc(u32, 1);
        one[0] = first;
        return .{ .tokens = one, .prefill_s = prefill_s, .decode_s = 0, .rounds = 0, .drafted = 0, .accepted = 0 };
    }
    const t1 = now(io);
    const opts: decode.Options = .{ .count = count, .depth = e.depth, .confidence = flashnext.engine.confidenceSetting(), .stop_eos = stop_eos };
    var r = if (drafts and e.depth > 0) try decode.drafted(Engine, e, gpa, first, opts) else try decode.serial(Engine, e, gpa, first, opts);
    errdefer r.deinit(gpa);
    try e.stream.synchronize();
    const decode_s = since(io, t1);
    return .{ .tokens = try r.tokens.toOwnedSlice(gpa), .prefill_s = prefill_s, .decode_s = decode_s, .rounds = r.rounds, .drafted = r.drafted, .accepted = r.accepted };
}

fn sha12(gpa: Allocator, tokens: []const u32) ![12]u8 {
    const text = try core.ids_json.write(gpa, tokens);
    defer gpa.free(text);
    const hex = shaHex(text);
    return hex[0..12].*;
}

fn run(gpa: Allocator, io: std.Io, e: *Engine, o: Options) !u8 {
    if (o.tokens.len == 0) {
        std.debug.print("run: --tokens or --tokens-file\n", .{});
        return 2;
    }
    const s: ?lanes.Sampling = if (o.sampling.temperature > 0) o.sampling else null;
    const g = try generate(gpa, io, e, o.tokens, o.max_tokens, s, o.drafts, o.stop_eos);
    defer gpa.free(g.tokens);
    const sha = try sha12(gpa, g.tokens);
    const steps = @max(1, g.tokens.len - 1);
    const ms = g.decode_s * 1e3 / @as(f64, @floatFromInt(steps));
    std.debug.print("tokens {d} sha {s} prefill {d:.4}s decode {d:.4}s {d:.3} ms/token rounds {d} drafted {d} accepted {d}\n", .{ g.tokens.len, sha, g.prefill_s, g.decode_s, ms, g.rounds, g.drafted, g.accepted });
    if (o.report) |path| {
        const report = .{ .engine = "zig-cuda-flashnext", .prompt_tokens = o.tokens, .tokens = g.tokens, .sha = sha, .prefill_seconds = g.prefill_s, .decode_seconds = g.decode_s, .ms_per_token = ms, .rounds = g.rounds, .drafted = g.drafted, .accepted = g.accepted, .drafts = o.drafts, .sampling = s, .load_seconds = e.load_seconds, .max_len = e.max_len };
        const json = try std.json.Stringify.valueAlloc(gpa, report, .{});
        defer gpa.free(json);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json });
    }
    return 0;
}

/// A step's dump against `dir`/digests.json (the capture's per-layer digests), or none without a dir.
const StepDump = struct {
    parsed: ?std.json.Parsed(std.json.Value) = null,
    dump: fwd.Dump,

    fn open(gpa: Allocator, io: std.Io, dir: ?[]const u8) !StepDump {
        var s: StepDump = .{ .dump = .{ .gpa = gpa, .io = io } };
        const d = dir orelse return s;
        const path = try std.fs.path.join(gpa, &.{ d, "digests.json" });
        defer gpa.free(path);
        s.parsed = readJson(gpa, io, path) catch |err| {
            std.debug.print("  no digests at {s} ({s})\n", .{ path, @errorName(err) });
            return s;
        };
        return s;
    }

    fn arm(s: *StepDump, e: *Engine) void {
        if (s.parsed) |*p| s.dump.want = &p.value.object;
        e.f.dump = &s.dump;
    }

    fn close(s: *StepDump, e: *Engine, label: []const u8) usize {
        e.f.dump = null;
        if (s.parsed != null) std.debug.print("  {s}: dumps {d} equal, {d} differ, {d} not in the oracle\n", .{ label, s.dump.equal, s.dump.differ, s.dump.missing });
        const bad = s.dump.differ;
        s.dump.deinit();
        if (s.parsed) |*p| p.deinit();
        return bad;
    }
};

/// The capture's teacher steps: from an empty state, step i forwards tokens[i] at pos i, draws greedily, then the
/// MTP head absorbs (row i's streams, tokens[i + 1]) at mtp pos i, then commit(1, 1).
fn teacher(gpa: Allocator, io: std.Io, e: *Engine, path: []const u8, dump_root: ?[]const u8, with_mtp: bool) !u8 {
    const t = try readJson(gpa, io, path);
    defer t.deinit();
    const ob = t.value.object;
    const tokens = ob.get("tokens").?.array.items;
    const sampled = ob.get("sampled").?.array.items;
    const logits = ob.get("logits_sha256").?.array.items;
    const mtp_next = ob.get("mtp_next").?.array.items;
    const mtp_sha = ob.get("mtp_logits_sha256").?.array.items;
    e.setSampling(null);
    try e.reset();
    var bad: usize = 0;
    var bad_mtp: usize = 0;
    var bad_dumps: usize = 0;
    const V = e.g.head_n;
    for (tokens, 0..) |tok, step| {
        var dir_buf: [512]u8 = undefined;
        const dir: ?[]const u8 = if (dump_root) |r| try std.fmt.bufPrint(&dir_buf, "{s}/step{d:0>3}", .{ r, step }) else null;
        var sd = try StepDump.open(gpa, io, dir);
        sd.arm(e);
        try e.forward(&.{@intCast(tok.integer)});
        var got: [1]u32 = undefined;
        try e.sample(1, e.pos() + 1, &got);
        const lsha = try deviceSha(gpa, e, e.buf.b.logits, V * 2);
        // the oracle's full tensors (dump steps): how close (two ranks have no bit-exact oracle)
        if (dir) |dd| {
            try closeness(gpa, io, e, dd, "final_mixed", e.buf.b.mixed, 0, e.g.hidden);
            try closeness(gpa, io, e, dd, "logits", e.buf.b.logits, @intCast(e.w.vocab_offset), V);
        }
        const ok = got[0] == sampled[step].integer and std.mem.eql(u8, &lsha, logits[step].string);
        var mtp_ok = true;
        var mtp_tok: u32 = 0;
        var msha_ok = true;
        if (with_mtp and step + 1 < tokens.len and e.w.mtp != null) {
            const lg = try e.teacherMtp(@intCast(tokens[step + 1].integer));
            const n = e.draws.?.columns;
            const msha = try deviceSha(gpa, e, lg, n * 2);
            mtp_tok = try e.drawDraft(e.pos() + 2);
            msha_ok = std.mem.eql(u8, &msha, mtp_sha[step].string);
            mtp_ok = mtp_tok == mtp_next[step].integer and msha_ok;
        }
        try e.commit(1, 1);
        var label: [32]u8 = undefined;
        bad_dumps += sd.close(e, try std.fmt.bufPrint(&label, "step {d}", .{step}));
        if (!ok) bad += 1;
        if (!mtp_ok) bad_mtp += 1;
        if (!ok or !mtp_ok) std.debug.print("step {d}: token {d} (oracle {d}) logits {s}; MTP draw {d} (oracle {d}) logits {s}\n", .{ step, got[0], sampled[step].integer, if (std.mem.eql(u8, &lsha, logits[step].string)) "equal" else "DIFFER", mtp_tok, if (step < mtp_next.len) mtp_next[step].integer else -1, if (msha_ok) "equal" else "DIFFER" });
    }
    const pass = bad == 0 and bad_mtp == 0 and bad_dumps == 0 and with_mtp;
    std.debug.print("{s} teacher: {d} of {d} steps' tokens and logits bit-equal, {d} of {d} MTP draws and logits, {d} dump mismatches\n", .{ if (pass) "PASS" else "FAIL", tokens.len - bad, tokens.len, mtp_next.len - bad_mtp, mtp_next.len, bad_dumps });
    return if (pass) 0 else 1;
}

fn bf16f(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// `n` bf16 values at `ptr` against `dir`/`name`.bin's from element `offset` (absent: nothing printed): max |diff|,
/// the reference's max |value|, values bit-equal, and whether the argmax agrees.
fn closeness(gpa: Allocator, io: std.Io, e: *Engine, dir: []const u8, name: []const u8, ptr: u64, offset: usize, n: usize) !void {
    var pb: [600]u8 = undefined;
    const path = try std.fmt.bufPrint(&pb, "{s}/{s}.bin", .{ dir, name });
    const ref = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26)) catch return;
    defer gpa.free(ref);
    if (ref.len < (offset + n) * 2) return;
    const host = try gpa.alloc(u16, n);
    defer gpa.free(host);
    try e.stream.synchronize();
    const buf: cuda.DeviceBuffer = .{ .d = e.ctx.d, .ptr = ptr, .len = n * 2 };
    try buf.download(0, std.mem.sliceAsBytes(host));
    var diff: f32 = 0;
    var amax: f32 = 0;
    var same: usize = 0;
    var best_a: usize = 0;
    var best_b: usize = 0;
    for (host, 0..) |h, j| {
        const r = std.mem.readInt(u16, ref[(offset + j) * 2 ..][0..2], .little);
        const a = bf16f(h);
        const b = bf16f(r);
        diff = @max(diff, @abs(a - b));
        amax = @max(amax, @abs(b));
        same += @intFromBool(h == r);
        if (a > bf16f(host[best_a])) best_a = j;
        if (b > bf16f(std.mem.readInt(u16, ref[(offset + best_b) * 2 ..][0..2], .little))) best_b = j;
    }
    std.debug.print("  {s} vs the oracle: max |diff| {d:.5} (|ref| max {d:.3}), {d} of {d} bit-equal, argmax {d} vs {d}\n", .{ name, diff, amax, same, n, best_a + offset, best_b + offset });
}

fn idsOf(gpa: Allocator, v: std.json.Value) ![]u32 {
    const items = v.array.items;
    const ids = try gpa.alloc(u32, items.len);
    for (items, ids) |x, *y| y.* = @intCast(x.integer);
    return ids;
}

/// One captured prompt's prefill (the oracle's: chunks of 2048 absorbing into the MTP head): its first token, the
/// last row's streams, pos and mtp_len against teacher.json's "prefill", every chunk's dumps against the digests.
fn prefillCheck(gpa: Allocator, io: std.Io, e: *Engine, prompts_path: []const u8, name: []const u8, dump: ?[]const u8, teacher_path: ?[]const u8) !u8 {
    const p = try readJson(gpa, io, prompts_path);
    defer p.deinit();
    const ids = try idsOf(gpa, p.value.object.get(name) orelse return error.NoSuchPrompt);
    defer gpa.free(ids);
    var sd = try StepDump.open(gpa, io, dump);
    sd.arm(e);
    e.absorbing = true;
    const t0 = now(io);
    const first = try e.prefill(ids, null);
    try e.stream.synchronize();
    const secs = since(io, t0);
    const ls = try deviceSha(gpa, e, e.lastStreams(), e.g.wide() * 2);
    const bad_dumps = sd.close(e, name);
    std.debug.print("RESULT prefill {s}: {d} tokens in {d:.3}s, first token {d}, pos {d}, mtp_len {d}, last streams {s}\n", .{ name, ids.len, secs, first, e.bound.st.pos, e.bound.st.mtp_len, ls[0..12] });
    var pass = bad_dumps == 0;
    if (teacher_path) |tp| {
        const t = try readJson(gpa, io, tp);
        defer t.deinit();
        const want = t.value.object.get("prefill").?.object.get(name) orelse return error.NoSuchPrompt;
        const w = want.object;
        const same = first == w.get("first").?.integer and e.bound.st.pos == w.get("pos").?.integer and e.bound.st.mtp_len == w.get("mtp_len").?.integer and std.mem.eql(u8, &ls, w.get("last_streams").?.string);
        pass = pass and same;
        std.debug.print("{s} prefill {s} against the oracle: first {d} (oracle {d}), last streams {s}\n", .{ if (same) "PASS" else "FAIL", name, first, w.get("first").?.integer, if (std.mem.eql(u8, &ls, w.get("last_streams").?.string)) "equal" else "DIFFER" });
    }
    return if (pass) 0 else 1;
}

/// Every device buffer's sha256 against the oracle's weights.json (W4's digests), by Python's dotted name.
fn checkWeights(gpa: Allocator, io: std.Io, e: *Engine, path: []const u8) !u8 {
    const want = try readJson(gpa, io, path);
    defer want.deinit();
    const ds = try flashnext.weights.digests(gpa, &e.w);
    defer flashnext.weights.freeDigests(gpa, ds);
    var same: usize = 0;
    var bad: usize = 0;
    var ours: usize = 0;
    for (ds) |d| {
        if (!d.oracle) {
            ours += 1;
            continue;
        }
        const expect = want.value.object.get(d.name) orelse {
            std.debug.print("MISSING {s}: the oracle has no tensor of this name\n", .{d.name});
            bad += 1;
            continue;
        };
        if (std.mem.eql(u8, &d.sha256, expect.string)) same += 1 else {
            std.debug.print("DIFFER {s}: {d} bytes {s}\n", .{ d.name, d.len, d.shape });
            bad += 1;
        }
    }
    var names: usize = 0;
    var it = want.value.object.iterator();
    while (it.next()) |kv| names += @intFromBool(!std.mem.endsWith(u8, kv.key_ptr.*, ".shape"));
    const pass = bad == 0 and same == names;
    std.debug.print("{s} weights: {d} equal, {d} wrong, {d} ours alone, {d} in the oracle\n", .{ if (pass) "PASS" else "FAIL", same, bad, ours, names });
    return if (pass) 0 else 1;
}

fn listed(only: ?[]const u8, name: []const u8) bool {
    const l = only orelse return true;
    var it = std.mem.tokenizeAny(u8, l, ",");
    while (it.next()) |x| if (std.mem.eql(u8, x, name)) return true;
    return false;
}

/// The run gate (zig/tests/cuda/nemotron/gate.py's rule on one load): every captured prompt, serial and drafted,
/// greedy and with the capture's sampling (seed 1234, temperature 1, top_k 20, top_p 0.95), token for token
/// against the oracle's results.json; drafted must equal serial.
fn gate(gpa: Allocator, io: std.Io, e: *Engine, capture: []const u8, o: Options) !u8 {
    var pb: [512]u8 = undefined;
    const prompts = try readJson(gpa, io, try std.fmt.bufPrint(&pb, "{s}/prompts.json", .{capture}));
    defer prompts.deinit();
    const results = try readJson(gpa, io, try std.fmt.bufPrint(&pb, "{s}/results.json", .{capture}));
    defer results.deinit();
    const res = results.value.object.get("results") orelse {
        std.debug.print("{s}/results.json has no results yet (a partial capture)\n", .{capture});
        return 2;
    };
    const samplings = [_]struct { name: []const u8, s: ?lanes.Sampling }{ .{ .name = "greedy", .s = null }, .{ .name = "sampled", .s = .{ .seed = 1234, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.0 } } };
    var bad: usize = 0;
    var total: usize = 0;
    var report: std.ArrayList(u8) = .empty;
    defer report.deinit(gpa);
    try report.appendSlice(gpa, "{");
    for (samplings) |smp| {
        var it = prompts.value.object.iterator();
        while (it.next()) |kv| {
            const name = kv.key_ptr.*;
            if (!listed(o.only, name)) continue;
            const ids = try idsOf(gpa, kv.value_ptr.*);
            defer gpa.free(ids);
            var serial_tokens: ?[]u32 = null;
            defer if (serial_tokens) |s| gpa.free(s);
            for ([_]bool{ false, true }) |drafts| {
                if (drafts and !o.drafts) continue; // gate --no-drafts: the serial runs alone
                const label = if (drafts) "drafted" else "serial";
                var kb: [128]u8 = undefined;
                const key = try std.fmt.bufPrint(&kb, "{s}-{s}/{s}", .{ label, smp.name, name });
                const g = try generate(gpa, io, e, ids, o.max_tokens, smp.s, drafts, true);
                var keep_tokens = false;
                defer if (!keep_tokens) gpa.free(g.tokens);
                const want = res.object.get(key);
                var equal = false;
                var oracle_n: usize = 0;
                if (want) |w| {
                    const wt = w.object.get("tokens").?.array.items;
                    oracle_n = wt.len;
                    // a shorter --max-tokens compares the oracle's first tokens
                    equal = wt.len == g.tokens.len or (g.tokens.len == o.max_tokens and wt.len > g.tokens.len);
                    if (equal) for (wt[0..g.tokens.len], g.tokens) |a, b| {
                        if (a.integer != b) {
                            equal = false;
                            break;
                        }
                    };
                }
                var db: [64]u8 = undefined;
                var first_diff: usize = 0;
                if (!equal and want != null) {
                    const wt = want.?.object.get("tokens").?.array.items;
                    while (first_diff < @min(wt.len, g.tokens.len) and wt[first_diff].integer == g.tokens[first_diff]) first_diff += 1;
                }
                var same_serial = true;
                if (drafts) {
                    if (serial_tokens) |s| same_serial = std.mem.eql(u32, s, g.tokens);
                } else {
                    serial_tokens = g.tokens;
                    keep_tokens = true;
                }
                total += 1;
                // two ranks have no oracle (Python's TP=2 refuses NVFP4), nor does a checkpoint Python never ran (an
                // empty results.json: INT4-AutoRound): drafted == serial is the gate there
                const no_oracle = e.world > 1 or res.object.count() == 0;
                const ok = (equal or no_oracle) and same_serial;
                const verdict = if (no_oracle) (if (!same_serial) "DIFFER" else if (equal) "TP1-SAME" else if (e.world > 1) "TP1-OTHER" else "SELF") else if (ok) "EQUAL" else "DIFFER";
                if (!ok) bad += 1;
                const sha = try sha12(gpa, g.tokens);
                const steps = @max(1, g.tokens.len - 1);
                std.debug.print("{s} {s}: {d} tokens sha {s} (oracle {d}{s}{s}) prefill {d:.3}s decode {d:.2} ms/token rounds {d} accepted {d}{s}\n", .{ verdict, key, g.tokens.len, sha, oracle_n, if (want == null) ", missing" else "", if (!equal and want != null) try std.fmt.bufPrint(&db, ", first difference at {d}", .{first_diff}) else "", g.prefill_s, g.decode_s * 1e3 / @as(f64, @floatFromInt(steps)), g.rounds, g.accepted, if (drafts and !same_serial) "; drafted != serial" else "" });
                if (report.items.len > 1) try report.appendSlice(gpa, ",");
                const entry = .{ .tokens = g.tokens, .sha = sha, .equal = equal, .drafted_equals_serial = same_serial, .prefill_s = g.prefill_s, .decode_s = g.decode_s, .rounds = g.rounds, .drafted = g.drafted, .accepted = g.accepted };
                const json = try std.json.Stringify.valueAlloc(gpa, entry, .{});
                defer gpa.free(json);
                try report.print(gpa, "\"{s}\":{s}", .{ key, json });
            }
        }
    }
    try report.appendSlice(gpa, "}");
    if (o.report) |path| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = report.items });
    std.debug.print("{s} gate: {d} of {d} runs equal the oracle's tokens (drafted == serial)\n", .{ if (bad == 0) "PASS" else "FAIL", total - bad, total });
    return if (bad == 0) 0 else 1;
}

/// The shared-round gate: N streams (the captured prompts in turn, greedy and sampled alternating) decoded
/// together in shared rounds, three passes (every stream drafted, every stream serial, then mixed); each stream's
/// tokens against the oracle's solo run (results.json); at two ranks, against this engine's own solo runs.
fn gateMany(gpa: Allocator, io: std.Io, e: *Engine, capture: []const u8, o: Options) !u8 {
    var pb: [512]u8 = undefined;
    const prompts = try readJson(gpa, io, try std.fmt.bufPrint(&pb, "{s}/prompts.json", .{capture}));
    defer prompts.deinit();
    const results = try readJson(gpa, io, try std.fmt.bufPrint(&pb, "{s}/results.json", .{capture}));
    defer results.deinit();
    const res = results.value.object.get("results") orelse return error.PartialCapture;
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(gpa);
    var it = prompts.value.object.iterator();
    while (it.next()) |kv| if (listed(o.only, kv.key_ptr.*)) try names.append(gpa, kv.key_ptr.*);
    const sampled: lanes.Sampling = .{ .seed = 1234, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.0 };
    const N = o.streams;
    const ids = try gpa.alloc([]u32, names.items.len);
    defer {
        for (ids) |x| gpa.free(x);
        gpa.free(ids);
    }
    for (names.items, ids) |nm, *x| {
        // --repeat K: the prompt's ids K times (long contexts for the memory checks; no oracle: --against-solo)
        const one = try idsOf(gpa, prompts.value.object.get(nm).?);
        defer gpa.free(one);
        x.* = try gpa.alloc(u32, one.len * o.repeat);
        for (0..o.repeat) |k| @memcpy(x.*[k * one.len ..][0..one.len], one);
    }
    var bad: usize = 0;
    var total: usize = 0;
    for ([_][]const u8{ "drafted", "serial", "mixed" }) |pass| {
        const reqs = try gpa.alloc(Engine.Request, N);
        defer {
            for (reqs) |*r| r.tokens.deinit(gpa);
            gpa.free(reqs);
        }
        for (reqs, 0..) |*r, i| {
            const greedy = (i + i / names.items.len) % 2 == 0;
            const drafts = if (std.mem.eql(u8, pass, "drafted")) true else if (std.mem.eql(u8, pass, "serial")) false else i % 2 == 1;
            // --arrive K --fill-layers L: odd streams arrive at rounds K, 2K, ... and fill L layers between rounds
            const arrive = if (o.fill_layers > 0 and i % 2 == 1) o.arrive * (1 + i / 2) else 0;
            // --served: the server's stop rule (the running product) instead of Python's 0.70
            const conf = if (o.served) flashnext.engine.confidenceSetting() else flashnext.engine.default_confidence;
            r.* = .{ .prompt = ids[i % names.items.len], .max_tokens = o.max_tokens, .sampling = if (greedy) null else sampled, .drafts = drafts, .arrive = arrive, .confidence = conf };
        }
        const t0 = now(io);
        try e.generateManyWith(gpa, reqs, .{ .fill_layers = o.fill_layers, .burst = o.burst });
        const secs = since(io, t0);
        const pass_times = e.times;
        var produced: usize = 0;
        for (reqs, 0..) |r, i| {
            produced += r.tokens.items.len;
            const name = names.items[i % names.items.len];
            var kb: [128]u8 = undefined;
            const key = try std.fmt.bufPrint(&kb, "{s}-{s}/{s}", .{ if (r.drafts) "drafted" else "serial", if (r.sampling == null) "greedy" else "sampled", name });
            var equal = false;
            var first_diff: usize = 0;
            if (o.against_solo) {
                // the same request alone in this engine
                var solo = [1]Engine.Request{.{ .prompt = r.prompt, .max_tokens = r.max_tokens, .sampling = r.sampling, .drafts = r.drafts, .confidence = r.confidence }};
                defer solo[0].tokens.deinit(gpa);
                try e.generateMany(gpa, &solo);
                const st = solo[0].tokens.items;
                while (first_diff < @min(st.len, r.tokens.items.len) and st[first_diff] == r.tokens.items[first_diff]) first_diff += 1;
                equal = st.len == r.tokens.items.len and first_diff == st.len;
            } else if (res.object.get(key)) |w| {
                const wt = w.object.get("tokens").?.array.items;
                const n = @min(wt.len, r.tokens.items.len);
                while (first_diff < n and wt[first_diff].integer == r.tokens.items[first_diff]) first_diff += 1;
                equal = first_diff == n and (wt.len == r.tokens.items.len or r.tokens.items.len == o.max_tokens);
            }
            total += 1;
            if (!equal and (e.world == 1 or o.against_solo)) bad += 1;
            const sha = try sha12(gpa, r.tokens.items);
            std.debug.print("{s} {s} stream {d} {s}: {d} tokens sha {s}{s} rounds {d} accepted {d}\n", .{ if (equal) "EQUAL" else if (e.world > 1) "TP2" else "DIFFER", pass, i, key, r.tokens.items.len, sha, if (equal) "" else try std.fmt.bufPrint(&pb, " (first difference at {d})", .{first_diff}), r.rounds, r.accepted });
        }
        const tm = pass_times;
        const dec = tm.verify + tm.sample + tm.draft;
        std.debug.print("RESULT {s} x{d}: {d} tokens in {d:.2}s ({d:.1} tokens/s together); prefill {d:.2}s, then {d:.1} tokens/s decoding over {d} rounds: verify {d:.1} ms (enqueued in {d:.1}), draws+commits {d:.1} ms, drafting {d:.1} ms a round\n", .{ pass, N, produced, secs, @as(f64, @floatFromInt(produced)) / secs, tm.prefill, @as(f64, @floatFromInt(produced - N)) / dec, tm.rounds, tm.verify * 1e3 / @as(f64, @floatFromInt(@max(1, tm.rounds))), tm.verify_host * 1e3 / @as(f64, @floatFromInt(@max(1, tm.rounds))), tm.sample * 1e3 / @as(f64, @floatFromInt(@max(1, tm.rounds))), tm.draft * 1e3 / @as(f64, @floatFromInt(@max(1, tm.rounds))) });
    }
    std.debug.print("{s} gate-many x{d}: {d} of {d} streams equal their solo runs\n", .{ if (bad == 0) "PASS" else "FAIL", N, total - bad, total });
    return if (bad == 0) 0 else 1;
}

/// Decode `count` tokens (drafted) on the bound sequence after `first`.
fn decodeBound(gpa: Allocator, e: *Engine, first: u32, count: usize) ![]u32 {
    var r = try decode.drafted(Engine, e, gpa, first, .{ .count = count, .depth = e.depth, .confidence = flashnext.engine.confidenceSetting(), .stop_eos = false });
    return r.tokens.toOwnedSlice(gpa);
}

/// Prompt reuse against fresh prefills: P2 = P ++ (16 reply tokens). Fresh: P2 prefilled from empty. Resumed: P
/// prefilled keeping its state at len - 1, a 32-token reply (rows past the point overwritten), then P2 resumed from
/// the kept state. Forked: a second sequence takes the kept prefix (copyPrefix) and resumes P2. 48 drafted tokens
/// each, greedy and sampled; all three must agree.
fn reuse(gpa: Allocator, io: std.Io, e: *Engine, capture: []const u8, o: Options) !u8 {
    var pb: [512]u8 = undefined;
    const prompts = try readJson(gpa, io, try std.fmt.bufPrint(&pb, "{s}/prompts.json", .{capture}));
    defer prompts.deinit();
    const sampled: lanes.Sampling = .{ .seed = 1234, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.0 };
    var bad: usize = 0;
    var total: usize = 0;
    e.absorbing = true;
    var it = prompts.value.object.iterator();
    while (it.next()) |kv| {
        if (!listed(o.only, kv.key_ptr.*)) continue;
        const p = try idsOf(gpa, kv.value_ptr.*);
        defer gpa.free(p);
        for ([_]?lanes.Sampling{ null, sampled }) |smp| {
            // the reply P would get, so P2 is a realistic next turn
            const own = e.bound;
            const f0 = try e.prefill(p, smp);
            const reply = try decodeBound(gpa, e, f0, 17);
            defer gpa.free(reply);
            const p2 = try gpa.alloc(u32, p.len + 16);
            defer gpa.free(p2);
            @memcpy(p2[0..p.len], p);
            @memcpy(p2[p.len..], reply[1..17]);
            // fresh
            const fresh = try decodeBound(gpa, e, try e.prefill(p2, smp), 48);
            defer gpa.free(fresh);
            // resumed: keep at len - 1, a reply that overwrites the rows past it, then the next turn
            const kept = try e.prefillWith(p, smp, .{ .keep_at = p.len - 1 });
            const snap = kept.kept.?;
            defer e.freeSnapshot(snap);
            const other = try decodeBound(gpa, e, kept.first, 32);
            gpa.free(other);
            const resumed = try decodeBound(gpa, e, (try e.prefillWith(p2, smp, .{ .resume_from = snap })).first, 48);
            defer gpa.free(resumed);
            // forked into another sequence by its prefix
            const b = try e.newSeq(e.max_len);
            defer e.freeSeq(b);
            try e.copyPrefix(b, own, snap.pos, snap.mtp_len);
            e.bind(b);
            const forked = try decodeBound(gpa, e, (try e.prefillWith(p2, smp, .{ .resume_from = snap })).first, 48);
            defer gpa.free(forked);
            e.bind(own);
            const ok_r = std.mem.eql(u32, fresh, resumed);
            const ok_f = std.mem.eql(u32, fresh, forked);
            total += 2;
            bad += @as(usize, @intFromBool(!ok_r)) + @intFromBool(!ok_f);
            std.debug.print("{s} reuse {s} {s}: {d}-token next turn resumed at {d}: resumed {s}, forked {s} (fresh sha {s})\n", .{ if (ok_r and ok_f) "EQUAL" else "DIFFER", kv.key_ptr.*, if (smp == null) "greedy" else "sampled", p2.len, snap.pos, if (ok_r) "equal" else "DIFFERS", if (ok_f) "equal" else "DIFFERS", try sha12(gpa, fresh) });
        }
    }
    std.debug.print("{s} reuse: {d} of {d} resumed runs equal their fresh prefills\n", .{ if (bad == 0) "PASS" else "FAIL", total - bad, total });
    return if (bad == 0) 0 else 1;
}

/// The tokens the sequences' budget holds at N streams: N sequences grown a step at a time in turn until every
/// one is refused (error.NoRoom) or at its window.
fn pool(gpa: Allocator, e: *Engine, o: Options) !u8 {
    const N = o.streams;
    const seqs = try gpa.alloc(*flashnext.forward.Seq, N);
    defer gpa.free(seqs);
    var made: usize = 0;
    defer for (seqs[0..made]) |q| e.freeSeq(q);
    for (seqs) |*q| {
        q.* = e.newSeq(e.max_len) catch |err| {
            std.debug.print("RESULT pool x{d}: only {d} sequences fit ({s})\n", .{ N, made, @errorName(err) });
            return 1;
        };
        made += 1;
    }
    var growing = N;
    const done = try gpa.alloc(bool, N);
    defer gpa.free(done);
    @memset(done, false);
    while (growing > 0) {
        for (seqs, done) |q, *d| {
            if (d.*) continue;
            const next = q.st.capacity + flashnext.state.grow_step;
            if (next > q.st.limit) {
                d.* = true;
                growing -= 1;
                continue;
            }
            e.f.grow(q, next) catch |err| switch (err) {
                error.NoRoom => {
                    d.* = true;
                    growing -= 1;
                },
                else => return err,
            };
        }
    }
    var rows: usize = 0;
    for (seqs) |q| rows += q.st.capacity;
    std.debug.print("RESULT pool x{d}: {d} cache rows ({d} a stream), {d:.2} GiB of {d:.2} GiB charged, {d:.0} bytes a row overall\n", .{ N, rows, rows / N, @as(f64, @floatFromInt(e.budget.used)) / (1 << 30), @as(f64, @floatFromInt(e.budget.limit)) / (1 << 30), @as(f64, @floatFromInt(e.budget.used)) / @as(f64, @floatFromInt(@max(rows, 1))) });
    return 0;
}

/// `bench`: every prompt file (ids) `repeat` times in one load; prefill seconds, tok/s and the reply's first tokens.
fn bench(gpa: Allocator, io: std.Io, e: *Engine, o: Options, files: []const []const u8) !u8 {
    var prompts: std.ArrayList([]u32) = .empty;
    defer {
        for (prompts.items) |p| gpa.free(p);
        prompts.deinit(gpa);
    }
    for (files) |list| {
        var it = std.mem.tokenizeScalar(u8, list, ',');
        while (it.next()) |path| {
            const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
            defer gpa.free(text);
            try prompts.append(gpa, try parseIds(gpa, text));
        }
    }
    if (prompts.items.len == 0) {
        std.debug.print("bench: --tokens-file PATH[,PATH...]\n", .{});
        return 2;
    }
    const s: ?lanes.Sampling = if (o.sampling.temperature > 0) o.sampling else null;
    for (0..o.reps) |rep| {
        for (prompts.items) |p| {
            const g = try generate(gpa, io, e, p, o.max_tokens, s, o.drafts, false);
            defer gpa.free(g.tokens);
            const sha = try sha12(gpa, g.tokens);
            std.debug.print("bench rep {d} prompt {d} tokens: prefill {d:.4}s {d:.1} tok/s; reply {d} tokens sha {s} first {d}", .{ rep, p.len, g.prefill_s, @as(f64, @floatFromInt(p.len)) / g.prefill_s, g.tokens.len, sha, g.tokens[0] });
            if (g.tokens.len > 1) std.debug.print(" decode {d:.2} ms/token", .{g.decode_s * 1e3 / @as(f64, @floatFromInt(g.tokens.len - 1))});
            std.debug.print("\n", .{});
        }
    }
    return 0;
}

/// `mm-check`: no checkpoint; the kernel set's in-program K-slice matmuls against the split kernels + _reduce.
fn mmCheck(gpa: Allocator, io: std.Io, ctx: *const cuda.Context, kernels: []const u8, o: Options) !u8 {
    const d = ctx.d;
    var set = try cuda.aot.Set.load(gpa, io, d, ctx.device, kernels);
    defer set.deinit();
    if (!flashnext.prompt.available(&set)) {
        std.debug.print("the kernel set {s} has no _b16mm_ks / _fp4mm_ks\n", .{kernels});
        return 1;
    }
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    const t: flashnext.triton.Tri = .{ .set = &set, .s = stream };
    var rows: std.ArrayList(usize) = .empty;
    defer rows.deinit(gpa);
    if (o.rows.len > 0) {
        var it = std.mem.tokenizeScalar(u8, o.rows, ',');
        while (it.next()) |r| try rows.append(gpa, try std.fmt.parseInt(usize, r, 10));
    } else try rows.appendSlice(gpa, &.{ 129, 200, 974, 2048, 2049, 4096 });
    const tp: u32 = 3; // TP=1 and TP=2 shapes
    const ok = try flashnext.prompt.check(gpa, d, t, tp, rows.items);
    for (flashnext.prompt.cases) |c| if (c.tp & tp != 0) {
        try flashnext.prompt.bench(gpa, d, t, c, 2048, 10);
        try flashnext.prompt.bench(gpa, d, t, c, 4096, 10);
    };
    std.debug.print("{s} mm-check: in-program K slices {s} Python's split kernels + _reduce\n", .{ if (ok) "PASS" else "FAIL", if (ok) "byte-equal to" else "DIFFER from" });
    return if (ok) 0 else 1;
}

/// `experts-check`: no checkpoint; fn_experts_prompt.cu against nvfp4_expert_kernel (TP=2 and TP=1 widths).
fn expertsCheck(gpa: Allocator, ctx: *const cuda.Context, o: Options) !u8 {
    var k = try flashnext.kernels.Kernels.load(ctx);
    defer k.deinit();
    var p = try flashnext.moe_prompt.Prompt4.init(ctx);
    defer p.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    const ops: flashnext.kernels.Ops = .{ .k = &k, .s = stream };
    var rows: std.ArrayList(usize) = .empty;
    defer rows.deinit(gpa);
    if (o.rows.len > 0) {
        var it = std.mem.tokenizeScalar(u8, o.rows, ',');
        while (it.next()) |r| try rows.append(gpa, try std.fmt.parseInt(usize, r, 10));
    } else try rows.appendSlice(gpa, &.{ 129, 974, 2048, 4096 });
    var ok = true;
    // the plan's item size and row tiles a warp: every choice must give the same bytes; the first is the default
    const cfgs = [_][3]usize{ .{ p.tile, p.t_gu, p.t_down }, .{ 64, 2, 4 }, .{ 48, 3, 2 }, .{ 16, 2, 2 } };
    for (cfgs, 0..) |c, ci| {
        p.tile = c[0];
        p.t_gu = c[1];
        p.t_down = c[2];
        ok = (try flashnext.moe_prompt.check(gpa, ctx, ops, &p, 320, rows.items, false)) and ok;
        if (ci == 0) {
            ok = (try flashnext.moe_prompt.check(gpa, ctx, ops, &p, 320, rows.items, true)) and ok;
            ok = (try flashnext.moe_prompt.check(gpa, ctx, ops, &p, 640, rows.items, false)) and ok;
        }
    }
    std.debug.print("{s} experts-check: the prompt expert kernel {s} Python's on 16-pair items\n", .{ if (ok) "PASS" else "FAIL", if (ok) "byte-equal to" else "DIFFER from" });
    return if (ok) 0 else 1;
}

/// `qsa-check`: no checkpoint; the prompt indexer's row-tiled scoring and tiled selection against Python's kernels.
fn qsaCheck(gpa: Allocator, io: std.Io, ctx: *const cuda.Context, kernels: []const u8) !u8 {
    var set = try cuda.aot.Set.load(gpa, io, ctx.d, ctx.device, kernels);
    defer set.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    const ok = try flashnext.prompt.qsaCheck(gpa, ctx.d, .{ .set = &set, .s = stream });
    std.debug.print("{s} qsa-check\n", .{if (ok) "PASS" else "FAIL"});
    return if (ok) 0 else 1;
}

/// `fp4-check`: no checkpoint; fn_ops' K-serial NVFP4 kernel against Triton's `_fp4mm` at one K slice (the shared
/// expert's gate/up), bytes compared at decode-sized rows, then both timed.
fn fp4Check(gpa: Allocator, io: std.Io, ctx: *const cuda.Context, kernels: []const u8, o: Options) !u8 {
    var set = try cuda.aot.Set.load(gpa, io, ctx.d, ctx.device, kernels);
    defer set.deinit();
    var ops = try flashnext.torch_ops.Ops.load(ctx);
    defer ops.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    const t: flashnext.triton.Tri = .{ .set = &set, .s = stream };
    const th = ops.on(stream);
    var rows: std.ArrayList(usize) = .empty;
    defer rows.deinit(gpa);
    if (o.rows.len > 0) {
        var it = std.mem.tokenizeScalar(u8, o.rows, ',');
        while (it.next()) |r| try rows.append(gpa, try std.fmt.parseInt(usize, r, 10));
    } else try rows.appendSlice(gpa, &flashnext.fp4_serial.default_rows);
    const ok = try flashnext.fp4_serial.check(gpa, ctx.d, t, th, rows.items);
    try flashnext.fp4_serial.bench(gpa, ctx.d, t, th, &.{ 1, 8, 36, 48, 64, 113, 128, 224 }, 50);
    std.debug.print("{s} fp4-check: the K-serial kernel {s} _fp4mm at one K slice\n", .{ if (ok) "PASS" else "FAIL", if (ok) "byte-equal to" else "DIFFERS from" });
    return if (ok) 0 else 1;
}

/// bench-many's prompts: 32 a kind, stream i takes prompt i % 32 (user turns, the chat template around them).
const bench_code = [_][]const u8{
    "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "Write a C function that reverses a singly linked list in place, with comments.",
    "Implement an LRU cache class in Python with get and put in O(1).",
    "Write a Rust function that merges two sorted vectors into one sorted vector.",
    "Write a Go HTTP handler that returns the current server time as JSON.",
    "Implement binary search over a sorted array of integers in Java, iteratively and recursively.",
    "Write a Python script that counts word frequencies in a text file and prints the top 20.",
    "Write a TypeScript function that debounces another function, with a usage example.",
    "Implement a min-heap in C++ with push, pop and top, without using the standard library heap.",
    "Write a SQL query that finds the second highest salary in each department, and explain it.",
    "Write a Bash script that backs up a directory into a dated tar.gz and keeps the last seven.",
    "Implement Dijkstra's shortest path algorithm in Python using heapq.",
    "Write a Python class for a bank account with deposit, withdraw and a transaction history.",
    "Write a JavaScript function that deep-clones an object, handling arrays, dates and cycles.",
    "Implement a trie in Python with insert, search and starts_with.",
    "Write a C program that reads integers from stdin and prints their mean and standard deviation.",
    "Write a Python generator that yields the prime numbers forever using an incremental sieve.",
    "Implement a thread-safe bounded queue in Java with put and take.",
    "Write a Rust program that reads a CSV file and sums the third column.",
    "Write a Python function that validates a Sudoku board, with unit tests.",
    "Implement quicksort in Haskell and explain its complexity.",
    "Write a Python decorator that retries a function up to three times with exponential backoff.",
    "Write a Go program that runs ten workers over a channel of jobs and collects their results.",
    "Implement matrix multiplication in NumPy and in pure Python, and compare them.",
    "Write a Python function that converts Roman numerals to integers and back.",
    "Write a C++ class for a 2D vector with operator overloading for +, - and dot product.",
    "Implement a simple tokenizer for arithmetic expressions in Python and evaluate them.",
    "Write a Kotlin data class for a user and a function that sorts users by age then name.",
    "Write a Python async function that fetches several URLs concurrently with aiohttp.",
    "Implement a union-find structure with path compression and union by rank in C.",
    "Write a Python function that flattens an arbitrarily nested list, with tests.",
    "Write a shell one-liner and an explanation that finds the ten largest files under a directory.",
};
const bench_prose = [_][]const u8{
    "Write a short story of about 300 words about a lighthouse keeper who finds a message in a bottle.",
    "Describe a walk through an autumn forest at dusk in about 300 words.",
    "Write a 300-word letter from a sailor to his sister about his first voyage.",
    "Write a short story about a robot who learns to bake bread.",
    "Describe a busy night market in a coastal city, in vivid detail.",
    "Write a fable about a fox and a crow who must share a winter shelter.",
    "Write a diary entry from a girl moving to a new town on the first day of school.",
    "Describe a thunderstorm rolling over a wheat field, from the farmer's porch.",
    "Write a short story about two strangers stuck in a train station overnight.",
    "Write a 300-word scene where an old musician plays one last concert.",
    "Describe the inside of a library that has been closed for fifty years.",
    "Write a short myth that explains why the moon changes shape.",
    "Write a story about a child who finds a door at the back of a wardrobe that leads to a garden.",
    "Describe a morning in a mountain village as the first snow falls.",
    "Write a letter from an astronaut to her younger self.",
    "Write a short story about a cat who runs a small bookshop.",
    "Describe a desert crossing by camel caravan at night under the stars.",
    "Write a scene where a detective realises the culprit is her oldest friend.",
    "Write a story about a gardener who grows a plant that sings.",
    "Describe a harbour at dawn as the fishing boats return.",
    "Write a short story about a clockmaker who can pause time for one minute a day.",
    "Write a monologue of a tree that has stood in a city square for three hundred years.",
    "Describe a family dinner during a power cut, lit only by candles.",
    "Write a short story about a lost dog finding its way home across the city.",
    "Write a poem in prose about the sea in winter.",
    "Describe the last day of summer camp from a counsellor's point of view.",
    "Write a story about a baker who receives a mysterious order every Tuesday.",
    "Write a short story set on a spaceship where the gardener is the captain.",
    "Describe a rainy afternoon in a small cafe in Paris.",
    "Write a story about an old map that changes every time it is unfolded.",
    "Write a scene where two rivals must climb a mountain together.",
    "Describe the sounds of a city waking up, street by street.",
};
const bench_chat = [_][]const u8{
    "What are three practical tips for sleeping better? Answer in a few sentences each.",
    "Explain why the sky is blue to a ten-year-old.",
    "What should I pack for a weekend hiking trip in the mountains?",
    "How do I make a good first impression at a job interview?",
    "What is the difference between a virus and a bacterium?",
    "Can you suggest a simple weekly exercise plan for a beginner?",
    "How does compound interest work? Give a small example.",
    "What are some good ways to learn a new language as an adult?",
    "Why do leaves change colour in autumn?",
    "How can I reduce food waste at home?",
    "What is a black hole, in simple terms?",
    "Give me a recipe for a quick vegetarian dinner.",
    "How do vaccines train the immune system?",
    "What are the pros and cons of working from home?",
    "How do I start a small vegetable garden on a balcony?",
    "Why is the ocean salty?",
    "What are some tips for staying focused while studying?",
    "How do airplanes stay in the air?",
    "What should I consider when adopting a dog?",
    "Explain what inflation is and why it happens.",
    "How can I improve my public speaking?",
    "What causes the seasons on Earth?",
    "How do I write a good cover letter?",
    "What is the difference between weather and climate?",
    "How can I save money on groceries?",
    "Why do we dream?",
    "What are good habits for keeping my computer secure?",
    "How do rainbows form?",
    "What is the best way to prepare for a marathon?",
    "How does the stock market work, briefly?",
    "What are some easy houseplants for beginners?",
    "How do bees make honey?",
};

const bench_structured = [_][]const u8{
    "Return a JSON array of 10 European capital cities with fields name, country, population and founded_year.",
    "Convert this list into a JSON object keyed by id: 1 apple red, 2 banana yellow, 3 grape purple, 4 lime green.",
    "Write a JSON schema for a blog post with title, author, tags, published date and a list of comments.",
    "Produce a YAML configuration for a web service with a database, a cache, three workers and logging settings.",
    "Return a markdown table of the planets of the solar system with columns name, diameter in km, moons and orbit days.",
    "Give a JSON list of 8 recipes, each with name, cuisine, minutes, difficulty and a list of ingredients.",
    "Write a CSV with a header and 15 rows of fictional employees: id, name, department, salary, start_date.",
    "Return a JSON object describing a library: name, address, opening hours per weekday and 5 sections with shelves.",
    "Extract the entities from this sentence as JSON: Maria flew from Lisbon to Tokyo on 3 May 2024 with Air Canada.",
    "Produce an OpenAPI 3 YAML snippet for a REST endpoint that creates, reads, updates and deletes a todo item.",
    "Return a JSON array of 12 months with fields name, days, season in the northern hemisphere and a holiday.",
    "Write a markdown table comparing five programming languages by typing, compilation, memory management and year.",
    "Give a JSON object for a product catalog with 6 products, each with sku, name, price, stock and categories.",
    "Return the periodic table's first 20 elements as JSON with symbol, name, atomic number and atomic mass.",
    "Write a GitHub Actions workflow YAML that tests a Python package on three Python versions and two operating systems.",
    "Produce a JSON timetable for a school week: five days, six periods a day, subject and teacher for each.",
    "Return a JSON array of 10 books with title, author, year, genre and a one-sentence summary.",
    "Write a Kubernetes deployment and service YAML for a container serving on port 8080 with three replicas.",
    "Give a markdown table of 10 countries with capital, currency, official language and continent.",
    "Return a JSON object that models a chess game in progress: players, clock times, moves so far and board state.",
    "Produce a JSON list of 15 HTTP status codes with code, name, category and a short description.",
    "Write an XML document describing a music album with artist, year, label and twelve tracks with durations.",
    "Return a JSON object for a travel itinerary: 5 days, each with city, hotel, activities and meals.",
    "Give a CSV of 20 weather observations with date, city, high, low, precipitation and conditions.",
    "Return a JSON array describing 8 dog breeds with name, size, life_expectancy, temperament and origin.",
    "Produce a TOML configuration for a command line tool with profiles, output formats, retries and logging.",
    "Write a JSON object for an invoice with seller, buyer, ten line items, taxes and totals.",
    "Return a markdown table of the first 15 US presidents with term start, term end and party.",
    "Give a JSON representation of a directory tree with folders src, tests and docs and the files in each.",
    "Return a JSON array of 10 airports with IATA code, city, country, latitude and longitude.",
    "Produce a JSON object for a football league table of 10 teams with played, won, drawn, lost, goals and points.",
    "Write a JSON list of 12 workouts with name, muscle group, sets, reps and rest seconds.",
};

/// The owner's sparkDash decode bench (its prompt catalog, src/shared/llmPrompts.js; greedy, thinking off): one
/// prose prompt and one structured prompt, each stream's copy suffixed " (stream i/N)" past one stream, and one code
/// task a stream.
const dash_prose = "Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough.";
const dash_structured = "Count from 1 to 200. Output only the numbers, separated by spaces. No other text.";
const dash_code_tail = "Output only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code.";
const dash_code = [_][]const u8{
    "binary_search\ndef binary_search(nums, target) -> int: index of target in a sorted list, or -1.\n" ++ dash_code_tail,
    "merge_sort\ndef merge_sort(nums) -> list: stable sort of a list of ints, returning a new list.\n" ++ dash_code_tail,
    "lru_cache\nclass LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used.\n" ++ dash_code_tail,
    "token_bucket\nclass TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False.\n" ++ dash_code_tail,
    "ring_buffer\nclass RingBuffer: push and pop over a fixed-capacity array, raising on overflow and underflow.\n" ++ dash_code_tail,
    "dijkstra\ndef dijkstra(graph, src) -> dict: shortest path weights from src on a non-negative weighted graph.\n" ++ dash_code_tail,
    "edit_distance\ndef edit_distance(a, b) -> int: Levenshtein distance between two strings.\n" ++ dash_code_tail,
    "semver_cmp\ndef semver_cmp(a, b) -> int: compare dotted numeric versions, negative if a < b.\n" ++ dash_code_tail,
    "url_parse\ndef url_parse(url) -> dict: scheme, host, port, path, and query pairs. No extra libraries.\n" ++ dash_code_tail,
    "json_pointer\ndef json_pointer(doc, pointer) -> object: follow an RFC 6901 pointer, or None if missing.\n" ++ dash_code_tail,
    "glob_match\ndef glob_match(pattern, text) -> bool: * and ? wildcards, no character classes.\n" ++ dash_code_tail,
    "csv_parse\ndef csv_parse(text) -> list: rows of fields, honoring double-quoted commas and escaped quotes.\n" ++ dash_code_tail,
    "rle\ndef rle_encode(s) -> str and rle_decode(s) -> str: run-length encoding of single-byte runs.\n" ++ dash_code_tail,
    "top_k\ndef top_k(nums, k) -> list: the k largest ints, unordered, using a bounded heap.\n" ++ dash_code_tail,
    "interval_merge\ndef merge_intervals(spans) -> list: merge overlapping [start, end] pairs.\n" ++ dash_code_tail,
    "topo_sort\ndef topo_sort(nodes, edges) -> list: a valid order, or None if the graph has a cycle.\n" ++ dash_code_tail,
    "bloom_filter\nclass BloomFilter: add(item) and might_contain(item) with two hash functions over a bit array.\n" ++ dash_code_tail,
    "moving_average\nclass MovingAverage: next(x) returns the mean of the last window values.\n" ++ dash_code_tail,
    "retry_backoff\ndef backoff_delays(attempts, base_ms, cap_ms) -> list: exponential delays clipped at the cap.\n" ++ dash_code_tail,
    "base64_encode\ndef b64_encode(data) -> str and b64_decode(text) -> bytes: standard base64, no libraries.\n" ++ dash_code_tail,
    "expr_eval\ndef eval_expr(text) -> int: evaluate non-negative ints with + - * / and parentheses.\n" ++ dash_code_tail,
    "histogram_percentile\ndef percentile(samples, p) -> float: nearest-rank percentile of a list of numbers.\n" ++ dash_code_tail,
    "redact_secrets\ndef redact(text) -> str: replace AWS-looking keys and password= values with ***. Keep the rest.\n" ++ dash_code_tail,
    "chunk_text\ndef chunk_text(text, size) -> list: split into chunks of at most size chars, breaking on spaces when possible.\n" ++ dash_code_tail,
    "route_match\ndef route_match(pattern, path) -> dict or None: /users/:id style params.\n" ++ dash_code_tail,
    "crc32\ndef crc32(data) -> int: IEEE CRC-32 of a bytes object.\n" ++ dash_code_tail,
    "fixed_window\nclass FixedWindow: allow() is True up to limit events per window_s, else False.\n" ++ dash_code_tail,
    "diff_lines\ndef diff_lines(a, b) -> list: line diff as equal/delete/insert ops using a simple LCS.\n" ++ dash_code_tail,
    "infix_postfix\ndef infix_to_postfix(tokens) -> list: shunting-yard for + - * / and parentheses.\n" ++ dash_code_tail,
    "consistent_hash\nclass ConsistentHash: add_node, remove_node, and get_node(key) on a ring of virtual nodes.\n" ++ dash_code_tail,
    "utf8_decode\ndef utf8_decode(data) -> str: decode UTF-8 bytes, replacing invalid sequences with U+FFFD.\n" ++ dash_code_tail,
    "dependency_closure\ndef closure(root, deps) -> list: packages reachable from root, each name once, in visit order.\n" ++ dash_code_tail,
};

/// bench-many: for each kind (--kind code,prose,chat,structured,dash-prose,dash-code,dash-structured) and stream count (--rows N,N,...; default --streams), the
/// kind's prompts decoded together in shared rounds, --reps times: aggregate tok/s and each round's parts
/// (Engine.Times; TF_FLASHNEXT_PROFILE adds the GPU parts, eagerly). Code greedy; prose and chat sampled T 1,
/// top_k 20, top_p 0.95 (dbench's rules), seeds 1000 + stream.
fn benchMany(gpa: Allocator, io: std.Io, e: *Engine, o: Options) !u8 {
    const path = try std.fs.path.join(gpa, &.{ o.model, "tokenizer.json" });
    defer gpa.free(path);
    var tok = try core.tokenizer.loadTokenizer(io, gpa, path);
    defer tok.deinit();
    // <|im_start|>user\n TEXT <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n (the captured prompts' frame)
    const head = [_]u32{ 248045, 846, 198 };
    const tail = [_]u32{ 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };
    var counts: std.ArrayList(usize) = .empty;
    defer counts.deinit(gpa);
    if (o.rows.len > 0) {
        var ct = std.mem.tokenizeScalar(u8, o.rows, ',');
        while (ct.next()) |c| try counts.append(gpa, try std.fmt.parseInt(usize, c, 10));
    } else try counts.append(gpa, o.streams);
    var kinds = std.mem.tokenizeScalar(u8, o.kind, ',');
    while (kinds.next()) |kind| {
        // dash-*: the owner's sparkDash prompts (built a stream count below), greedy
        const dash = std.mem.startsWith(u8, kind, "dash-");
        const texts: []const []const u8 = if (std.mem.eql(u8, kind, "code")) &bench_code else if (std.mem.eql(u8, kind, "prose")) &bench_prose else if (std.mem.eql(u8, kind, "chat")) &bench_chat else if (std.mem.eql(u8, kind, "structured")) &bench_structured else if (std.mem.eql(u8, kind, "dash-code")) &dash_code else if (std.mem.eql(u8, kind, "dash-prose")) &.{dash_prose} else if (std.mem.eql(u8, kind, "dash-structured")) &.{dash_structured} else return error.UnknownKind;
        // code and structured greedy; prose and chat sampled (dbench's rules); --temperature T samples every kind
        const greedy = o.sampling.temperature <= 0 and (dash or std.mem.eql(u8, kind, "code") or std.mem.eql(u8, kind, "structured"));
        const prompts = try gpa.alloc([]u32, texts.len);
        defer {
            for (prompts) |x| gpa.free(x);
            gpa.free(prompts);
        }
        for (texts, prompts) |t, *x| {
            const body = try tok.encode(gpa, t);
            defer gpa.free(body);
            x.* = try std.mem.concat(gpa, u32, &.{ &head, body, &tail });
        }
        for (counts.items) |N| for (0..o.reps) |rep| {
            const reqs = try gpa.alloc(Engine.Request, N);
            defer {
                for (reqs) |*r| r.tokens.deinit(gpa);
                gpa.free(reqs);
            }
            // sparkDash's one-prompt kinds: the prompt itself alone, else " (stream i/N)" on each stream's copy
            const own = try gpa.alloc([]u32, N);
            defer {
                for (own) |x| if (x.len > 0) gpa.free(x);
                gpa.free(own);
            }
            for (own, 0..) |*x, i| {
                x.* = &.{};
                if (!dash or texts.len > 1 or N == 1) continue;
                var tb: [512]u8 = undefined;
                const t = try std.fmt.bufPrint(&tb, "{s} (stream {d}/{d})", .{ texts[0], i + 1, N });
                const bodyt = try tok.encode(gpa, t);
                defer gpa.free(bodyt);
                x.* = try std.mem.concat(gpa, u32, &.{ &head, bodyt, &tail });
            }
            for (reqs, 0..) |*r, i| {
                var smp: lanes.Sampling = .{ .seed = 1000 + i, .temperature = 1.0, .top_k = 20, .top_p = 0.95, .min_p = 0.0 };
                if (o.sampling.temperature > 0) {
                    smp = o.sampling;
                    smp.seed = o.sampling.seed + i;
                }
                r.* = .{ .prompt = if (own[i].len > 0) own[i] else prompts[i % prompts.len], .max_tokens = o.max_tokens, .sampling = if (greedy) null else smp, .drafts = o.drafts, .confidence = if (o.served) flashnext.engine.confidenceSetting() else flashnext.engine.default_confidence, .stop_eos = o.stop_eos };
            }
            const t0 = now(io);
            try e.generateMany(gpa, reqs);
            const secs = since(io, t0);
            const tm = e.times;
            var produced: usize = 0;
            var rounds: usize = 0;
            for (reqs) |r| {
                produced += r.tokens.items.len;
                rounds += r.rounds;
            }
            const nr: f64 = @floatFromInt(@max(1, tm.rounds));
            const dec = tm.verify + tm.sample + tm.draft;
            std.debug.print("RESULT bench-many {s} {s} x{d} rep {d}: {d} tokens, prefill {d:.2}s, {d:.1} tokens/s decoding ({d:.1} with the prefills) over {d} rounds, {d:.2} tokens a stream-round; a round: verify {d:.1} ms (enqueued in {d:.1}), draws+commits {d:.1} ms, drafting {d:.1} ms\n", .{ kind, if (greedy) "greedy" else "sampled", N, rep, produced, tm.prefill, @as(f64, @floatFromInt(produced - N)) / dec, @as(f64, @floatFromInt(produced)) / secs, tm.rounds, @as(f64, @floatFromInt(produced - N)) / @as(f64, @floatFromInt(@max(1, rounds))), tm.verify * 1e3 / nr, tm.verify_host * 1e3 / nr, tm.sample * 1e3 / nr, tm.draft * 1e3 / nr });
        };
    }
    return 0;
}
