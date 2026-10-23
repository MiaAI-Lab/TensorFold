//! ``tensorfold serve`` flags: one table for parsing them and for ``capabilities --json``.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const prepare_gate = @import("prepare_gate.zig");
const Allocator = std.mem.Allocator;

pub const Kind = enum {
    value, // --flag VALUE or --flag=VALUE
    store_true, // --flag
    append, // repeatable --flag VALUE
};

pub const Flag = struct {
    name: []const u8,
    kind: Kind = .value,
    /// argparse's choices; empty: any value of the flag's type.
    choices: []const []const u8 = &.{},
    /// This binary honours the flag (and lists it in ``capabilities``); the rest are refused as unsupported.
    native: bool = false,
    /// The values it honours when only some of the choices.
    native_values: ?[]const []const u8 = null,
    /// A flag the GLM recipe passes that this engine does not serve yet: parsed, then refused with this reason.
    later: ?[]const u8 = null,
};

const vision_later = "GLM vision (pictures and clips) is not built into the Zig engine yet (package H); serve with VISION=0 until it lands";

const backend_values: []const []const u8 = if (builtin.os.tag == .macos) &.{ "auto", "mlx" } else if (served("--backend=rocm")) &.{ "auto", "rocm" } else &.{ "auto", "cuda" };

/// The flags past the common ones the binary's backend serves, from its root's `serves` (test roots: none).
const native_serves: []const []const u8 = if (@hasDecl(@import("root"), "serves")) @import("root").serves else &.{};

fn served(comptime name: []const u8) bool {
    return servedBy(native_serves, name);
}

fn servedBy(list: []const []const u8, name: []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, x, name)) return true;
    return false;
}

/// The CUDA build's own flags and variables (the Metal build refuses them).
const cuda_build = builtin.os.tag == .linux;

/// Every Python serve flag (``cli_args.build_parser``), the ones this binary serves marked native.
pub const flags = [_]Flag{
    .{ .name = "--host", .native = true },
    .{ .name = "--port", .native = true },
    .{ .name = "--name", .native = true },
    .{ .name = "--chat-template", .native = true },
    .{ .name = "--alias", .kind = .append, .native = true },
    .{ .name = "--api-key", .kind = .append, .native = true },
    .{ .name = "--api-key-file", .native = true },
    .{ .name = "--metrics-open", .kind = .store_true, .native = true },
    .{ .name = "--dashboard", .kind = .store_true, .native = true },
    .{ .name = "--vision", .kind = .store_true, .native = cuda_build },
    .{ .name = "--vision-urls", .kind = .store_true, .native = cuda_build },
    .{ .name = "--vision-backend", .choices = &.{ "zig", "python" }, .native = cuda_build },
    .{ .name = "--vision-offload", .kind = .store_true },
    .{ .name = "--vision-max-images", .native = cuda_build },
    .{ .name = "--vision-image-tokens", .native = cuda_build },
    .{ .name = "--context", .native = true },
    .{ .name = "--speed-up", .native = true },
    .{ .name = "--max-tokens", .native = true },
    .{ .name = "--temperature", .native = true },
    .{ .name = "--top-p", .native = true },
    .{ .name = "--top-k", .native = true },
    .{ .name = "--min-p", .native = true },
    .{ .name = "--thinking", .kind = .store_true, .native = true },
    .{ .name = "--no-thinking", .kind = .store_true, .native = true },
    .{ .name = "--reasoning-effort", .choices = &.{ "low", "medium", "high", "xhigh" }, .native = true },
    .{ .name = "--thinking-budget", .native = true },
    .{ .name = "--loop-guard", .kind = .store_true, .native = true },
    .{ .name = "--no-drafts", .kind = .store_true, .native = true },
    .{ .name = "--keep-warm", .native = true },
    .{ .name = "--compact-at", .native = true },
    .{ .name = "--compact-keep", .native = true },
    .{ .name = "--compact-memory", .native = true },
    .{ .name = "--slide", .kind = .store_true, .native = true },
    .{ .name = "--slide-graph", .native = true },
    .{ .name = "--drafter", .native = true },
    .{ .name = "--drafter-bits", .choices = &.{ "0", "4" }, .native = true },
    .{ .name = "--mtp-drafts", .native = served("--mtp-drafts") },
    .{ .name = "--mtp-confidence", .native = served("--mtp-confidence") },
    .{ .name = "--lane-kernels", .choices = &.{ "auto", "on", "off" } },
    .{ .name = "--prompt-cache-gib", .native = true },
    .{ .name = "--prompt-cache-over-cap", .kind = .store_true, .native = true },
    .{ .name = "--checkpoint-slots", .native = served("--checkpoint-slots") },
    .{ .name = "--spill-gib", .native = cuda_build },
    .{ .name = "--spill-highwater", .native = cuda_build },
    .{ .name = "--spill-min-tokens", .native = cuda_build },
    .{ .name = "--spill-min-free-gib", .native = cuda_build },
    .{ .name = "--spill-flush-s", .native = cuda_build },
    .{ .name = "--snapshot-dir", .native = true }, // the spill tier's folder (the recipe passes /spill with --spill-gib)
    .{ .name = "--max-snapshots", .native = true, .native_values = &.{"0"} },
    .{ .name = "--parallel", .native = true },
    .{ .name = "--decode-share" },
    .{ .name = "--prefill-pass" },
    .{ .name = "--pass-cache-gib" },
    .{ .name = "--mlx-cache-gib" },
    .{ .name = "--ssd-experts" },
    .{ .name = "--ple-on-ssd", .kind = .store_true },
    .{ .name = "--no-update-check", .kind = .store_true, .native = true },
    .{ .name = "--backend", .choices = &.{ "auto", "mlx", "cuda", "rocm" }, .native = true, .native_values = backend_values },
    .{ .name = "--tp", .choices = &.{ "1", "2", "3", "4", "8" }, .native = served("--tp") },
    .{ .name = "--rank", .native = served("--tp") },
    .{ .name = "--master", .native = served("--tp") },
    .{ .name = "--master-port", .native = served("--tp") },
    // the GPU engine's policy switches as key=value pairs, repeatable
    .{ .name = "--policy", .kind = .append, .native = served("--policy") },
    .{ .name = "--kv-dtype", .choices = &.{ "bf16", "int8", "int4" } },
    .{ .name = "--prefill-fp8", .kind = .store_true },
    .{ .name = "--no-prefill-fp8", .kind = .store_true },
    .{ .name = "--precision", .choices = &.{ "checkpoint", "full" } },
    // the CUDA CLI's own: the GPU ordinal, and whole prompt chunks a call runs as staggered segments
    .{ .name = "--device", .native = cuda_build },
    .{ .name = "--segments", .native = cuda_build },
    // the native server's own: shared prompt states (a harness) kept on disk for later sessions and servers
    .{ .name = "--learn", .kind = .store_true, .native = true },
    .{ .name = "--learn-dir", .native = true },
    .{ .name = "--learn-gib", .native = true },
    .{ .name = "--learn-min-free-gib", .native = true },
};

/// The variables this binary honours as the Python engine does, then the CUDA build's.
pub const env = [_][]const u8{ "TENSORFOLD_API_KEY", "TENSORFOLD_NO_LIVE", "TENSORFOLD_SEED_SALT", "TENSORFOLD_REQUEST_LOG", "TENSORFOLD_NO_UPDATE_CHECK", "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE" } ++
    (if (cuda_build) [_][]const u8{ "TF_CUDA_DEVICE", "TF_CUDA_SEGMENTS", "TENSORFOLD_CUDA_KERNELS", "TENSORFOLD_MEMORY_RESERVE_GIB", "TENSORFOLD_CUDA_MEMORY_LIMIT_GB" } else [_][]const u8{}) ++
    // the GLM recipe's server switches (recipe.zig; scripts/config.sh exports them)
    (if (cuda_build) [_][]const u8{ "TF_GLM_EFFORT_TAIL", "TF_GLM_CLEAR_THINKING", "TF_GLM_STREAM_SMOOTH", "TF_GLM_STREAM_SMOOTH_MS", "TF_GLM_LOOP_GUARD", "TF_GLM_MAX_QUEUED", "TF_GLM_KEEP_REASONING", "TF_GLM_KEEP_REASONING_MB", "TF_GLM_ASSISTANT_ENDS", "TF_GLM_PROMPT_STOP" } else [_][]const u8{}) ++
    (if (cuda_build) spill_env else [_][]const u8{});

/// Spill v2's variables past its flags (api.Spill, `spill`): the Python fork's names, then the DeepSeek port's.
pub const spill_env = [_][]const u8{ "TF_GLM_SPILL_QUOTA", "TF_GLM_SPILL_TURNS", "TF_GLM_SPILL_READ_BLOCKS", "TF_GLM_SPILL_STAGE_MIB", "TF_GLM_SPILL_LOAD_MS", "TF_SPILL_LOAD_WAIT_MS", "TF_SPILL_DEFER_MS", "TF_SPILL_FLUSH_S" };

pub const Args = struct {
    model: []const u8 = "",
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    name: []const u8 = "",
    chat_template: ?[]const u8 = null,
    alias: []const []const u8 = &.{},
    api_key: []const []const u8 = &.{},
    api_key_file: ?[]const u8 = null,
    metrics_open: bool = false,
    dashboard: bool = false,
    context: ?i64 = null,
    speed_up: ?[]const u8 = null, // speed-up mode: this Mac's settings for the two-Mac link (Flash Next)
    prompt_cache_gib: ?f64 = null, // kept prompt states' budget (0: none; null: what 70% of RAM leaves past the loaded server)
    prompt_cache_over_cap: bool = false, // a --prompt-cache-gib past that is kept, not refused
    learn: bool = false, // keep shared prompt states on disk (--learn-dir: where; it implies --learn)
    learn_dir: ?[]const u8 = null,
    learn_min_free_gib: f64 = 4,
    learn_gib: f64 = 32, // disk for learned states, every model and build together
    snapshot_dir: ?[]const u8 = null, // the spill tier's folder
    spill_gib: f64 = 0, // the spill tier's disk cap a rank (0: off)
    spill_highwater: f64 = 0.70,
    spill_highwater_given: bool = false, // (v1's early-write mark; the v2 tier ignores it and says so)
    spill_min_tokens: u32 = 8192,
    spill_min_free_gib: f64 = 50,
    spill_flush_s: ?f64 = null, // a clean stop's budget (null: TF_SPILL_FLUSH_S, else 60)
    spill: api.Spill = .{}, // spill v2's settings, the flags' and the variables' (main sets it: `spill`)
    max_tokens: i64 = 4096,
    temperature: ?f64 = null,
    top_p: ?f64 = null,
    top_k: ?i64 = null,
    min_p: ?f64 = null,
    thinking: bool = true,
    reasoning_effort: ?[]const u8 = null,
    thinking_budget: i64 = 0,
    loop_guard: bool = false,
    no_drafts: bool = false,
    vision: bool = false, // picture and clip input (the CUDA engine hosts the vision tower)
    vision_urls: bool = false, // --vision: also fetch public https URLs
    vision_backend: []const u8 = "python", // zig: the engine hosts the tower; python: Python TensorFold's, in a helper process
    vision_max_images: ?i64 = null,
    vision_image_tokens: ?i64 = null,
    keep_warm: i64 = 900, // seconds the idle keepalive runs after the last request ends (0: off)
    compact_auto: bool = false,
    compact_fraction: ?f64 = null,
    compact_keep: ?u32 = null,
    compact_memory: ?[]const u8 = null,
    drafter: ?[]const u8 = null,
    drafter_bits: u8 = 4,
    slide: bool = false, // Sliding Weights learns what /v1/slide/learn is told into the served weights, live
    slide_graph: ?[]const u8 = null, // the Sliding Weights fact graph's file (null: memory only)
    parallel: []const u8 = "auto",
    backend: []const u8 = "auto",
    device: ?u32 = null,
    segments: ?u32 = null,
    mtp_drafts: ?u32 = null,
    mtp_confidence: ?f64 = null,
    checkpoint_slots: ?i64 = null,
    tp: u32 = 1,
    rank: u32 = 0,
    master: []const u8 = "",
    master_port: u16 = 29551,
    policy: []const u8 = "", // every --policy joined by commas
};

/// A usage error's message (argparse's ``error:`` line); the caller exits 2.
pub const Usage = struct { message: []const u8 = "" };

fn find(name: []const u8) ?Flag {
    for (flags) |f| if (std.mem.eql(u8, f.name, name)) return f;
    return null;
}

fn fail(u: *Usage, a: Allocator, comptime fmt: []const u8, args: anytype) error{ Usage, OutOfMemory } {
    u.message = try std.fmt.allocPrint(a, fmt, args);
    return error.Usage;
}

/// ``serve`` arguments as the switch passes them: the model and full flag names.
pub fn parse(a: Allocator, argv: []const []const u8, u: *Usage) error{ Usage, OutOfMemory }!Args {
    var out: Args = .{};
    var alias: std.ArrayList([]const u8) = .empty;
    var keys: std.ArrayList([]const u8) = .empty;
    var model: ?[]const u8 = null;
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const token = argv[i];
        if (!std.mem.startsWith(u8, token, "--")) {
            if (model != null) return fail(u, a, "unrecognized arguments: {s}", .{token});
            model = token;
            continue;
        }
        const eq = std.mem.indexOfScalar(u8, token, '=');
        const name = token[0 .. eq orelse token.len];
        const flag = find(name) orelse return fail(u, a, "unrecognized arguments: {s}", .{token});
        var value: ?[]const u8 = null;
        if (flag.kind != .store_true) {
            if (eq) |e| value = token[e + 1 ..] else {
                i += 1;
                if (i >= argv.len) return fail(u, a, "argument {s}: expected one argument", .{name});
                value = argv[i];
            }
            if (flag.choices.len > 0) for (flag.choices) |c| {
                if (std.mem.eql(u8, c, value.?)) break;
            } else return fail(u, a, "argument {s}: invalid choice: '{s}'", .{ name, value.? });
        } else if (eq != null) return fail(u, a, "argument {s}: ignored explicit argument '{s}'", .{ name, token[eq.? + 1 ..] });
        if (flag.later) |why| return fail(u, a, "{s}: {s}", .{ name, why });
        if (!flag.native) return fail(u, a, "{s} is not served by the native engine yet; the Python engine 0.6.6 serves it: python -m pip install git+https://github.com/ashhart/TensorFold.git@v0.6.6", .{name});
        if (flag.native_values) |allowed| if (value) |v| for (allowed) |x| {
            if (std.mem.eql(u8, x, v)) break;
        } else return fail(u, a, "{s} {s} is not served by the native engine; the Python engine 0.6.6 serves it: python -m pip install git+https://github.com/ashhart/TensorFold.git@v0.6.6", .{ name, v });
        try apply(a, &out, name, value, u, &alias, &keys);
    }
    out.model = model orelse return fail(u, a, "the following arguments are required: model", .{});
    try ranks(u, a, out);
    if (out.tp > 1 and out.master.len == 0) return fail(u, a, "--tp {d} needs --master, the address rank 0 listens on for the other ranks", .{out.tp});
    out.alias = alias.items;
    out.api_key = keys.items;
    return out;
}

fn int(u: *Usage, a: Allocator, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!i64 {
    return std.fmt.parseInt(i64, std.mem.trim(u8, v, " "), 10) catch fail(u, a, "argument {s}: invalid int value: '{s}'", .{ name, v });
}

fn float(u: *Usage, a: Allocator, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!f64 {
    return @import("fields.zig").pyFloat(v) orelse fail(u, a, "argument {s}: invalid float value: '{s}'", .{ name, v });
}

/// A finite GiB count from 0 (off) to below 2^34, whose bytes fit a u64.
fn gib(u: *Usage, a: Allocator, name: []const u8, v: []const u8) error{ Usage, OutOfMemory }!f64 {
    const g = try float(u, a, name, v);
    if (!std.math.isFinite(g) or g < 0 or g >= 1 << 34) return fail(u, a, "argument {s}: expected GiB from 0 (off) to below 2^34: '{s}'", .{ name, v });
    return g;
}

fn apply(a: Allocator, out: *Args, name: []const u8, value: ?[]const u8, u: *Usage, alias: *std.ArrayList([]const u8), keys: *std.ArrayList([]const u8)) error{ Usage, OutOfMemory }!void {
    const v = value orelse "";
    const is = struct {
        fn f(x: []const u8, y: []const u8) bool {
            return std.mem.eql(u8, x, y);
        }
    }.f;
    if (try cudaFlag(a, out, name, v, u) or try gpuFlag(a, out, name, v, u)) return;
    if (is(name, "--chat-template")) {
        out.chat_template = v;
        return;
    }
    if (is(name, "--host")) out.host = v else if (is(name, "--port")) {
        const p = try int(u, a, name, v);
        if (p < 0 or p > 65535) return fail(u, a, "argument --port: invalid port: '{s}'", .{v});
        out.port = @intCast(p);
    } else if (is(name, "--name")) out.name = v else if (is(name, "--alias")) try alias.append(a, v) else if (is(name, "--api-key")) try keys.append(a, v) else if (is(name, "--api-key-file")) out.api_key_file = v else if (is(name, "--metrics-open")) out.metrics_open = true else if (is(name, "--dashboard")) out.dashboard = true else if (is(name, "--context")) out.context = try int(u, a, name, v) else if (is(name, "--speed-up")) out.speed_up = v else if (is(name, "--prompt-cache-gib")) out.prompt_cache_gib = try gib(u, a, name, v) else if (is(name, "--prompt-cache-over-cap")) out.prompt_cache_over_cap = true else if (is(name, "--learn")) out.learn = true else if (is(name, "--learn-dir")) {
        out.learn = true;
        out.learn_dir = v;
    } else if (is(name, "--snapshot-dir")) {
        out.snapshot_dir = if (std.mem.eql(u8, v, "none")) null else v;
    } else if (is(name, "--spill-gib")) {
        out.spill_gib = try gib(u, a, name, v);
    } else if (is(name, "--spill-highwater")) {
        const f = try float(u, a, name, v);
        if (!(f > 0 and f <= 1)) return fail(u, a, "argument --spill-highwater: a fraction in (0, 1]: '{s}'", .{v});
        out.spill_highwater = f;
        out.spill_highwater_given = true;
    } else if (is(name, "--spill-min-tokens")) {
        out.spill_min_tokens = std.math.cast(u32, try int(u, a, name, v)) orelse return fail(u, a, "argument --spill-min-tokens: a token count from 0: '{s}'", .{v});
    } else if (is(name, "--spill-min-free-gib")) {
        out.spill_min_free_gib = try gib(u, a, name, v);
    } else if (is(name, "--spill-flush-s")) {
        out.spill_flush_s = try gib(u, a, name, v);
    } else if (is(name, "--learn-min-free-gib")) {
        out.learn_min_free_gib = try gib(u, a, name, v);
    } else if (is(name, "--learn-gib")) {
        out.learn = true;
        out.learn_gib = try gib(u, a, name, v);
    } else if (is(name, "--max-tokens")) out.max_tokens = try int(u, a, name, v) else if (is(name, "--temperature")) out.temperature = try float(u, a, name, v) else if (is(name, "--top-p")) out.top_p = try float(u, a, name, v) else if (is(name, "--top-k")) out.top_k = try int(u, a, name, v) else if (is(name, "--min-p")) out.min_p = try float(u, a, name, v) else if (is(name, "--thinking")) out.thinking = true else if (is(name, "--no-thinking")) out.thinking = false else if (is(name, "--reasoning-effort")) out.reasoning_effort = v else if (is(name, "--thinking-budget")) out.thinking_budget = try int(u, a, name, v) else if (is(name, "--loop-guard")) out.loop_guard = true else if (is(name, "--drafter")) out.drafter = v else if (is(name, "--drafter-bits")) out.drafter_bits = @intCast(try int(u, a, name, v)) else if (is(name, "--no-drafts")) out.no_drafts = true else if (is(name, "--keep-warm")) out.keep_warm = try int(u, a, name, v) else if (is(name, "--compact-at")) {
        if (std.mem.eql(u8, v, "auto")) out.compact_auto = true else {
            const f = try float(u, a, name, v);
            if (!(f > 0 and f <= 1)) return fail(u, a, "argument --compact-at: expected auto or a fraction in (0, 1]: '{s}'", .{v});
            out.compact_fraction = f;
        }
    } else if (is(name, "--compact-keep")) {
        const n = try int(u, a, name, v);
        if (n < 0) return fail(u, a, "argument --compact-keep: expected a token count from 0: '{s}'", .{v});
        out.compact_keep = @intCast(n);
    } else if (is(name, "--compact-memory")) out.compact_memory = v else if (is(name, "--slide")) out.slide = true else if (is(name, "--slide-graph")) out.slide_graph = v else if (is(name, "--parallel")) out.parallel = v else if (is(name, "--backend")) out.backend = v;
}

/// The CUDA build's --device and --segments; false for any other flag.
fn cudaFlag(a: Allocator, out: *Args, name: []const u8, v: []const u8, u: *Usage) error{ Usage, OutOfMemory }!bool {
    if (std.mem.eql(u8, name, "--device")) {
        out.device = std.math.cast(u32, try int(u, a, name, v)) orelse return fail(u, a, "argument --device: a GPU ordinal from 0: '{s}'", .{v});
    } else if (std.mem.eql(u8, name, "--vision")) {
        out.vision = true;
    } else if (std.mem.eql(u8, name, "--vision-urls")) {
        out.vision = true;
        out.vision_urls = true;
    } else if (std.mem.eql(u8, name, "--vision-backend")) {
        out.vision = true;
        out.vision_backend = v;
    } else if (std.mem.eql(u8, name, "--vision-max-images")) {
        out.vision_max_images = try int(u, a, name, v);
    } else if (std.mem.eql(u8, name, "--vision-image-tokens")) {
        out.vision_image_tokens = try int(u, a, name, v);
    } else if (std.mem.eql(u8, name, "--segments")) {
        const n = try int(u, a, name, v);
        out.segments = if (n >= 1) @intCast(@min(n, std.math.maxInt(u32))) else return fail(u, a, "argument --segments: a count from 1: '{s}'", .{v});
    } else return false;
    return true;
}

/// A rank names one of the `tp` processes.
fn ranks(u: *Usage, a: Allocator, out: Args) error{ Usage, OutOfMemory }!void {
    if (out.rank >= out.tp) return fail(u, a, "argument --rank: {d} is not below --tp {d}", .{ out.rank, out.tp });
}

/// The GPU engine's tensor-parallel, MTP, kept-entry and policy flags; false for any other flag.
fn gpuFlag(a: Allocator, out: *Args, name: []const u8, v: []const u8, u: *Usage) error{ Usage, OutOfMemory }!bool {
    const is = struct {
        fn f(x: []const u8, y: []const u8) bool {
            return std.mem.eql(u8, x, y);
        }
    }.f;
    if (is(name, "--tp")) out.tp = @intCast(try int(u, a, name, v)) else if (is(name, "--rank")) {
        out.rank = std.math.cast(u32, try int(u, a, name, v)) orelse return fail(u, a, "argument --rank: a rank from 0: '{s}'", .{v});
    } else if (is(name, "--master")) out.master = v else if (is(name, "--master-port")) {
        const p = try int(u, a, name, v);
        if (p < 0 or p > 65535) return fail(u, a, "argument --master-port: invalid port: '{s}'", .{v});
        out.master_port = @intCast(p);
    } else if (is(name, "--mtp-drafts")) {
        const n = try int(u, a, name, v);
        if (n < 0 or n > 64) return fail(u, a, "argument --mtp-drafts: a count from 0 to 64: '{s}'", .{v});
        out.mtp_drafts = @intCast(n);
    } else if (is(name, "--mtp-confidence")) {
        const c = try float(u, a, name, v);
        if (!(c >= 0 and c <= 1)) return fail(u, a, "argument --mtp-confidence: a probability in [0, 1]: '{s}'", .{v});
        out.mtp_confidence = c;
    } else if (is(name, "--checkpoint-slots")) {
        const n = try int(u, a, name, v);
        if (n < 0) return fail(u, a, "argument --checkpoint-slots: a count from 0: '{s}'", .{v});
        out.checkpoint_slots = n;
    } else if (is(name, "--policy")) {
        if (std.mem.indexOfScalar(u8, v, '=') == null) return fail(u, a, "argument --policy: expected key=value: '{s}'", .{v});
        out.policy = if (out.policy.len == 0) v else try std.fmt.allocPrint(a, "{s},{s}", .{ out.policy, v });
    } else return false;
    return true;
}

/// Spill v2's settings: the spill flags, then, when they ask for the disk tier, its variables from `env_in` (null:
/// none read; main passes a GLM's). Python 0110 read them only with the spill on (`_spill_store`) and refused a
/// malformed knob at startup: so does this, naming it in `problem`, and with the spill off a stray one is never read.
/// An unset or empty one keeps its default, and so does 0 for the restore budgets (a 0 ns budget never progresses a
/// restore). TF_SPILL_FLUSH_S, v1's variable too, keeps v1's reading: anything but a number of seconds is the
/// default, a negative one 0.
pub fn spill(a: Allocator, args: Args, env_in: ?*const std.process.Environ.Map, problem: *[]const u8) error{ Invalid, OutOfMemory }!api.Spill {
    var s: api.Spill = .{ .dir = args.snapshot_dir, .gib = args.spill_gib, .min_free_gib = args.spill_min_free_gib, .min_tokens = args.spill_min_tokens, .highwater = if (args.spill_highwater_given) args.spill_highwater else null };
    const env_map = if (s.enabled()) env_in else null;
    if (spillVar(env_map, "TF_GLM_SPILL_QUOTA")) |t| {
        const q = @import("fields.zig").pyFloat(t) orelse -1;
        if (!(q >= 0 and q <= 1)) {
            problem.* = try std.fmt.allocPrint(a, "TF_GLM_SPILL_QUOTA='{s}': a share of --spill-gib from 0 (none) to 1", .{t});
            return error.Invalid;
        }
        s.quota = q;
    }
    s.turns = try spillCount(a, env_map, "TF_GLM_SPILL_TURNS", 0, s.turns, problem);
    s.read_blocks = try spillCount(a, env_map, "TF_GLM_SPILL_READ_BLOCKS", 1, s.read_blocks, problem);
    s.stage_mib = try spillCount(a, env_map, "TF_GLM_SPILL_STAGE_MIB", 1, s.stage_mib, problem);
    if (spillVar(env_map, "TF_GLM_SPILL_LOAD_MS")) |t| { // milliseconds, a fraction too (Python read a float)
        const ms = @import("fields.zig").pyFloat(t) orelse -1;
        if (!(ms >= 0 and std.math.isFinite(ms))) {
            problem.* = try std.fmt.allocPrint(a, "TF_GLM_SPILL_LOAD_MS='{s}': a number of milliseconds from 0", .{t});
            return error.Invalid;
        }
        if (ms > 0) s.load_ms = ms;
    }
    const wait = try spillCount(a, env_map, "TF_SPILL_LOAD_WAIT_MS", 0, s.load_wait_ms, problem);
    if (wait > 0) s.load_wait_ms = wait;
    s.defer_ms = try spillCount(a, env_map, "TF_SPILL_DEFER_MS", 0, s.defer_ms, problem);
    const flush_s = args.spill_flush_s orelse if (spillVar(env_map, "TF_SPILL_FLUSH_S")) |t| (@import("fields.zig").pyFloat(t) orelse 60) else 60;
    if (std.math.isFinite(flush_s)) s.flush_ms = @intFromFloat(@min(@max(flush_s, 0) * 1000, std.math.maxInt(u32)));
    return s;
}

/// A variable's text, trimmed (null: unset or empty).
fn spillVar(env_map: ?*const std.process.Environ.Map, name: []const u8) ?[]const u8 {
    const m = env_map orelse return null;
    const t = std.mem.trim(u8, m.get(name) orelse return null, " \t\r\n");
    return if (t.len == 0) null else t;
}

/// A variable's whole number from `least` (`default` when unset), else error.Invalid with `problem` set.
fn spillCount(a: Allocator, env_map: ?*const std.process.Environ.Map, name: []const u8, least: u32, default: u32, problem: *[]const u8) error{ Invalid, OutOfMemory }!u32 {
    const t = spillVar(env_map, name) orelse return default;
    const n: ?u32 = std.fmt.parseInt(u32, t, 10) catch null;
    if (n != null and n.? >= least) return n.?;
    problem.* = try std.fmt.allocPrint(a, "{s}='{s}': a whole number from {d}", .{ name, t, least });
    return error.Invalid;
}

/// ``--parallel``: "auto" is up to 8 requests at once; a number caps it.
pub fn parallel(text: []const u8) ?u32 {
    const t = std.mem.trim(u8, text, " ");
    if (std.ascii.eqlIgnoreCase(t, "auto")) return 8;
    const n = std.fmt.parseInt(i64, t, 10) catch return null;
    return @intCast(@max(1, @min(n, 4096)));
}

/// ``--parallel`` named a number (an engine that fits fewer refuses), not "auto" (it serves what fits).
pub fn parallelFixed(text: []const u8) bool {
    return !std.ascii.eqlIgnoreCase(std.mem.trim(u8, text, " "), "auto") and parallel(text) != null;
}

/// What a server asks of its engine, from its flags.
pub fn request(a: Allocator, dir: []const u8, model_type: []const u8, args: Args) !api.Open {
    return .{
        .dir = dir,
        .model_type = model_type,
        .context = args.context,
        .lanes = parallel(args.parallel) orelse 8,
        .lanes_fixed = parallelFixed(args.parallel),
        .drafts = !args.no_drafts,
        .drafter = args.drafter,
        .drafter_bits = args.drafter_bits,
        .speed_up = args.speed_up,
        .prompt_cache_gib = args.prompt_cache_gib,
        .prompt_cache_over_cap = args.prompt_cache_over_cap,
        .spill_dir = args.snapshot_dir,
        .spill_gib = args.spill_gib,
        .spill_highwater = args.spill_highwater,
        .spill_min_tokens = args.spill_min_tokens,
        .spill_min_free_gib = args.spill_min_free_gib,
        .spill_flush_s = args.spill_flush_s,
        .spill = args.spill,
        .learn = if (args.learn) args.learn_dir orelse try api.prompt_imprint.defaultRoot(a) else null,
        .learn_gib = args.learn_gib,
        .learn_min_free_gib = args.learn_min_free_gib,
        .slide = args.slide,
        .device = args.device,
        .segments = args.segments,
        .mtp_drafts = args.mtp_drafts,
        .mtp_confidence = args.mtp_confidence,
        .keep = args.checkpoint_slots,
        .tp = args.tp,
        .rank = args.rank,
        .master = args.master,
        .master_port = args.master_port,
        .policy = args.policy,
        .vision = args.vision,
        .vision_zig = args.vision and std.mem.eql(u8, args.vision_backend, "zig"),
    };
}

/// What ``capabilities --json`` reports about the engine side: its version, chip, backends and families.
pub const Engines = struct {
    version: []const u8,
    chip: ?[]const u8 = null,
    backends: []const []const u8 = &.{},
    /// model_type to the weight formats it reads, as gate entries name them.
    families: []const api.Family = &.{},
};

/// The native capabilities document: the table's native flags and the honoured variables.
pub fn capabilities(w: *std.Io.Writer, e: Engines) !void {
    try w.print("{{\"schema\": 1, \"engine\": \"zig\", \"version\": \"{s}\", \"chip\": ", .{e.version});
    if (e.chip) |c| try w.print("\"{s}\"", .{c}) else try w.writeAll("null");
    try w.writeAll(", \"backends\": [");
    for (e.backends, 0..) |b, i| try w.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", b });
    try w.writeAll("], \"families\": {");
    for (e.families, 0..) |f, i| {
        try w.print("{s}\"{s}\": [", .{ if (i > 0) ", " else "", f.model_type });
        for (f.formats, 0..) |fmt, j| try w.print("{s}\"{s}\"", .{ if (j > 0) ", " else "", fmt });
        try w.writeAll("]");
    }
    try w.writeAll("}, \"serve\": {\"flags\": {");
    var first = true;
    for (flags) |f| {
        if (!f.native) continue;
        try w.print("{s}\"{s}\": {{", .{ if (first) "" else ", ", f.name });
        first = false;
        const values = f.native_values orelse f.choices;
        if (values.len > 0) {
            try w.writeAll("\"values\": [");
            for (values, 0..) |v, j| try w.print("{s}\"{s}\"", .{ if (j > 0) ", " else "", v });
            try w.writeAll("]");
        }
        try w.writeAll("}");
    }
    try w.writeAll("}}, \"env\": [");
    // (the prepare gate's floor is the server's own: prepare_gate.zig)
    for (env ++ prepare_gate.env, 0..) |name, i| try w.print("{s}\"{s}\"", .{ if (i > 0) ", " else "", name });
    try w.writeAll("]}\n");
}

test "parse and capabilities share the table" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    const args = try parse(a, &.{ "/models/x", "--port", "9000", "--api-key=k1", "--api-key", "k2", "--no-thinking", "--reasoning-effort", "low" }, &u);
    try std.testing.expectEqual(@as(u16, 9000), args.port);
    try std.testing.expectEqual(@as(usize, 2), args.api_key.len);
    try std.testing.expect(!args.thinking);
    try std.testing.expectEqualStrings("x", (try parse(a, &.{ "m", "--drafter", "x" }, &u)).drafter.?);
    try std.testing.expectEqual(@as(u8, 0), (try parse(a, &.{ "m", "--drafter-bits", "0" }, &u)).drafter_bits);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--drafter-bits", "8" }, &u));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--reasoning-effort", "max" }, &u));
    try std.testing.expectEqual(@as(?f64, 12.5), (try parse(a, &.{ "m", "--prompt-cache-gib", "12.5" }, &u)).prompt_cache_gib);
    try std.testing.expectEqual(@as(?f64, 0), (try parse(a, &.{ "m", "--prompt-cache-gib", "0" }, &u)).prompt_cache_gib);
    try std.testing.expect(!(try parse(a, &.{ "m", "--prompt-cache-gib", "16" }, &u)).prompt_cache_over_cap);
    try std.testing.expect((try parse(a, &.{ "m", "--prompt-cache-gib", "16", "--prompt-cache-over-cap" }, &u)).prompt_cache_over_cap);
    try std.testing.expect((try parse(a, &.{ "m", "--learn" }, &u)).learn);
    const dir = try parse(a, &.{ "m", "--learn-dir", "/x/y" }, &u);
    try std.testing.expect(dir.learn and std.mem.eql(u8, "/x/y", dir.learn_dir.?));
    const capped = try parse(a, &.{ "m", "--learn-gib", "8" }, &u);
    try std.testing.expect(capped.learn and capped.learn_gib == 8);
    for ([_][]const u8{ "1e300", "inf", "nan", "-1", "17179869184" }) |bad| try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--prompt-cache-gib", bad }, &u));
    var out: std.Io.Writer.Allocating = .init(a);
    try capabilities(&out.writer, .{ .version = "0.6.5" });
    const doc = out.written();
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"--no-thinking\": {}") != null);
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"--drafter\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, doc, "\"--compact-at\": {}") != null);
    const on = try parse(a, &.{ "m", "--compact-at", "auto", "--compact-keep", "100", "--compact-memory", "notes" }, &u);
    try std.testing.expect(on.compact_auto);
    try std.testing.expectEqual(@as(?u32, 100), on.compact_keep);
    try std.testing.expectEqualStrings("notes", on.compact_memory.?);
    try std.testing.expectEqualStrings("graph.json", (try parse(a, &.{ "m", "--slide-graph", "graph.json" }, &u)).slide_graph.?);
    try std.testing.expect((try parse(a, &.{ "m", "--slide" }, &u)).slide);
    try std.testing.expectEqual(@as(?f64, 0.5), (try parse(a, &.{ "m", "--compact-at", "0.5" }, &u)).compact_fraction);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--compact-at", "0" }, &u));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--compact-at", "2" }, &u));
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--compact-keep", "-1" }, &u));
}

test "--device and --segments: CUDA builds serve them, values checked" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    if (cuda_build) {
        const args = try parse(a, &.{ "m", "--device", "1", "--segments=2" }, &u);
        try std.testing.expectEqual(@as(?u32, 1), args.device);
        try std.testing.expectEqual(@as(?u32, 2), args.segments);
    } else try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--device", "1" }, &u));
    var out: Args = .{};
    try std.testing.expect(try cudaFlag(a, &out, "--device", "0", &u));
    try std.testing.expectEqual(@as(?u32, 0), out.device);
    try std.testing.expectError(error.Usage, cudaFlag(a, &out, "--device", "-1", &u));
    try std.testing.expectError(error.Usage, cudaFlag(a, &out, "--segments", "0", &u));
    try std.testing.expect(try cudaFlag(a, &out, "--segments", "4", &u));
    try std.testing.expectEqual(@as(?u32, 4), out.segments);
    try std.testing.expect(!try cudaFlag(a, &out, "--port", "1", &u));
    try std.testing.expect(parallelFixed("3") and !parallelFixed("auto") and !parallelFixed("x"));
}

test "learned disk floor parses once with a finite byte-safe default and is a native capability" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var u: Usage = .{};
    const normal = try parse(arena.allocator(), &.{"m"}, &u);
    try std.testing.expectEqual(@as(f64, 4), normal.learn_min_free_gib);
    const explicit = try parse(arena.allocator(), &.{ "m", "--learn-min-free-gib", "1.5" }, &u);
    try std.testing.expectEqual(@as(f64, 1.5), explicit.learn_min_free_gib);
    for ([_][]const u8{ "nan", "inf", "-1", "17179869184" }) |bad| {
        try std.testing.expectError(error.Usage, parse(arena.allocator(), &.{ "m", "--learn-min-free-gib", bad }, &u));
    }
}

test "the GPU engine's flags: refused and unlisted unless the backend serves them, values checked" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    // each flag, a value, and the `serves` entry that turns it on
    const gated = [_][3][]const u8{ .{ "--tp", "2", "--tp" }, .{ "--rank", "0", "--tp" }, .{ "--master", "node0", "--tp" }, .{ "--master-port", "29600", "--tp" }, .{ "--policy", "matrix=off", "--policy" }, .{ "--mtp-drafts", "2", "--mtp-drafts" }, .{ "--mtp-confidence", "0.5", "--mtp-confidence" }, .{ "--checkpoint-slots", "3", "--checkpoint-slots" } };
    var out: std.Io.Writer.Allocating = .init(a);
    try capabilities(&out.writer, .{ .version = "0" });
    for (gated) |g| {
        const listed = std.mem.indexOf(u8, out.written(), try std.fmt.allocPrint(a, "\"{s}\"", .{g[0]})) != null;
        try std.testing.expectEqual(servedBy(native_serves, g[2]), listed);
        if (!listed) try std.testing.expectError(error.Usage, parse(a, &.{ "m", g[0], g[1] }, &u));
    }
    try std.testing.expect(servedBy(&.{ "--tp", "--policy" }, "--tp") and !servedBy(&.{"--tp"}, "--policy") and !servedBy(&.{}, "--tp"));
    var args: Args = .{};
    for ([_][2][]const u8{ .{ "--tp", "4" }, .{ "--rank", "2" }, .{ "--master", "node0" }, .{ "--master-port", "29600" }, .{ "--mtp-drafts", "2" }, .{ "--mtp-confidence", "0.5" }, .{ "--checkpoint-slots", "3" }, .{ "--policy", "matrix=off" }, .{ "--policy", "kernels=reference" } }) |g| try std.testing.expect(try gpuFlag(a, &args, g[0], g[1], &u));
    try std.testing.expectEqual(@as(u32, 4), args.tp);
    try std.testing.expectEqual(@as(u32, 2), args.rank);
    try std.testing.expectEqualStrings("node0", args.master);
    try std.testing.expectEqual(@as(u16, 29600), args.master_port);
    try std.testing.expectEqual(@as(?u32, 2), args.mtp_drafts);
    try std.testing.expectEqual(@as(?f64, 0.5), args.mtp_confidence);
    try std.testing.expectEqual(@as(?i64, 3), args.checkpoint_slots);
    try std.testing.expectEqualStrings("matrix=off,kernels=reference", args.policy);
    try ranks(&u, a, args);
    args.rank = 4;
    try std.testing.expectError(error.Usage, ranks(&u, a, args));
    for ([_][2][]const u8{ .{ "--rank", "-1" }, .{ "--master-port", "70000" }, .{ "--mtp-drafts", "65" }, .{ "--mtp-confidence", "1.5" }, .{ "--checkpoint-slots", "-1" }, .{ "--policy", "matrix" } }) |g| try std.testing.expectError(error.Usage, gpuFlag(a, &args, g[0], g[1], &u));
    try std.testing.expect(!try gpuFlag(a, &args, "--port", "1", &u));
}

test "an engine is asked for what the flags say, and the defaults of the flags not given" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const plain = try request(a, "d", "t", .{});
    const want: api.Open = .{ .dir = "d", .model_type = "t", .learn = null };
    inline for (.{ "mtp_drafts", "mtp_confidence", "keep", "tp", "rank", "master_port", "lanes", "drafts", "device", "segments" }) |field| try std.testing.expectEqual(@field(want, field), @field(plain, field));
    try std.testing.expectEqualStrings("", plain.policy);
    try std.testing.expectEqualStrings("", plain.master);
    const gpu = try request(a, "d", "t", .{ .tp = 4, .rank = 1, .master = "node0", .master_port = 29600, .mtp_drafts = 2, .mtp_confidence = 0.5, .checkpoint_slots = 3, .policy = "matrix=off" });
    try std.testing.expectEqual(@as(u32, 4), gpu.tp);
    try std.testing.expectEqual(@as(u32, 1), gpu.rank);
    try std.testing.expectEqualStrings("node0", gpu.master);
    try std.testing.expectEqual(@as(u16, 29600), gpu.master_port);
    try std.testing.expectEqual(@as(?u32, 2), gpu.mtp_drafts);
    try std.testing.expectEqual(@as(?f64, 0.5), gpu.mtp_confidence);
    try std.testing.expectEqual(@as(?i64, 3), gpu.keep);
    try std.testing.expectEqualStrings("matrix=off", gpu.policy);
}

test "the recipe's start.sh flags: the spill flags are read and vision is accepted" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: Usage = .{};
    const got = try parse(a, &.{ "m", "--context", "131072", "--parallel", "8", "--max-tokens", "32768", "--no-thinking", "--spill-gib", "64", "--snapshot-dir", "/spill", "--spill-highwater", "0.70", "--spill-min-tokens", "8192", "--spill-min-free-gib", "50" }, &u);
    try std.testing.expectEqual(@as(f64, 64), got.spill_gib);
    try std.testing.expectEqualStrings("/spill", got.snapshot_dir.?);
    try std.testing.expectEqual(@as(f64, 0.70), got.spill_highwater);
    try std.testing.expectEqual(@as(u32, 8192), got.spill_min_tokens);
    try std.testing.expectEqual(@as(f64, 50), got.spill_min_free_gib);
    try std.testing.expectError(error.Usage, parse(a, &.{ "m", "--spill-highwater", "1.5" }, &u));
    const vis = try parse(a, &.{ "m", "--vision", "--vision-urls" }, &u);
    try std.testing.expect(vis.vision and vis.vision_urls);
}

test "spill v2's settings: the spill flags and its variables, defaults when unset, a malformed knob refused by name, none read with the spill off" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const off = try spill(a, .{}, null, &problem);
    try std.testing.expect(!off.enabled());
    try std.testing.expectEqual(api.Spill{}, off);
    var vars = std.process.Environ.Map.init(a);
    for ([_][2][]const u8{ .{ "TF_GLM_SPILL_QUOTA", "0.25" }, .{ "TF_GLM_SPILL_TURNS", "0" }, .{ "TF_GLM_SPILL_READ_BLOCKS", " 8 " }, .{ "TF_GLM_SPILL_STAGE_MIB", "512" }, .{ "TF_GLM_SPILL_LOAD_MS", "25" }, .{ "TF_SPILL_LOAD_WAIT_MS", "250" }, .{ "TF_SPILL_DEFER_MS", "0" }, .{ "TF_SPILL_FLUSH_S", "5" } }) |kv| try vars.put(kv[0], kv[1]);
    const args: Args = .{ .snapshot_dir = "/spill", .spill_gib = 64, .spill_min_tokens = 4096, .spill_min_free_gib = 20, .spill_flush_s = 2.5, .spill_highwater = 0.7, .spill_highwater_given = true };
    const on = try spill(a, args, &vars, &problem);
    try std.testing.expect(on.enabled());
    try std.testing.expectEqualStrings("/spill", on.dir.?);
    try std.testing.expectEqual(@as(f64, 64), on.gib);
    try std.testing.expectEqual(@as(f64, 20), on.min_free_gib);
    try std.testing.expectEqual(@as(u32, 4096), on.min_tokens);
    try std.testing.expectEqual(@as(f64, 0.25), on.quota);
    try std.testing.expectEqual(@as(u32, 0), on.turns);
    try std.testing.expectEqual(@as(u32, 8), on.read_blocks);
    try std.testing.expectEqual(@as(u32, 512), on.stage_mib);
    try std.testing.expectEqual(@as(f64, 25), on.load_ms);
    try std.testing.expectEqual(@as(u32, 250), on.load_wait_ms);
    try std.testing.expectEqual(@as(u32, 0), on.defer_ms); // 0: never deferred
    try std.testing.expectEqual(@as(u32, 2500), on.flush_ms); // the flag before the variable
    try std.testing.expectEqual(@as(?f64, 0.7), on.highwater);
    const tier: Args = .{ .snapshot_dir = "/spill", .spill_gib = 64 }; // (the variables are read with the spill on)
    try std.testing.expectEqual(@as(u32, 5000), (try spill(a, tier, &vars, &problem)).flush_ms);
    try std.testing.expectEqual(@as(?f64, null), (try spill(a, .{ .spill_highwater = 0.7 }, null, &problem)).highwater); // (the default, not given)
    // 0 for the restore budgets keeps their defaults; a malformed TF_SPILL_FLUSH_S keeps v1's reading (60 s)
    var zero = std.process.Environ.Map.init(a);
    try zero.put("TF_GLM_SPILL_LOAD_MS", "0");
    try zero.put("TF_SPILL_LOAD_WAIT_MS", "0");
    try zero.put("TF_SPILL_FLUSH_S", "soon");
    const kept = try spill(a, tier, &zero, &problem);
    try std.testing.expectEqual(@as(f64, 40), kept.load_ms);
    try std.testing.expectEqual(@as(u32, 30_000), kept.load_wait_ms);
    try std.testing.expectEqual(@as(u32, 60_000), kept.flush_ms);
    // milliseconds of a restore budget may be a fraction, as Python 0110 read them (the recipe checks a number)
    var frac = std.process.Environ.Map.init(a);
    try frac.put("TF_GLM_SPILL_LOAD_MS", "2.5");
    try std.testing.expectEqual(@as(f64, 2.5), (try spill(a, tier, &frac, &problem)).load_ms);
    const malformed = [_][2][]const u8{ .{ "TF_GLM_SPILL_QUOTA", "1.5" }, .{ "TF_GLM_SPILL_QUOTA", "a quarter" }, .{ "TF_GLM_SPILL_TURNS", "-1" }, .{ "TF_GLM_SPILL_TURNS", "all" }, .{ "TF_GLM_SPILL_READ_BLOCKS", "0" }, .{ "TF_GLM_SPILL_STAGE_MIB", "0" }, .{ "TF_GLM_SPILL_LOAD_MS", "-1" }, .{ "TF_GLM_SPILL_LOAD_MS", "inf" }, .{ "TF_GLM_SPILL_LOAD_MS", "nan" }, .{ "TF_GLM_SPILL_LOAD_MS", "soon" }, .{ "TF_SPILL_DEFER_MS", "x" }, .{ "TF_SPILL_DEFER_MS", "1.5" } };
    for (malformed) |kv| {
        var bad = std.process.Environ.Map.init(a);
        try bad.put(kv[0], kv[1]);
        problem = "";
        try std.testing.expectError(error.Invalid, spill(a, tier, &bad, &problem));
        try std.testing.expect(std.mem.indexOf(u8, problem, kv[0]) != null);
        // the spill off (no --spill-gib, or no --snapshot-dir): never read, never a reason to refuse the start
        try std.testing.expectEqual(api.Spill{}, try spill(a, .{}, &bad, &problem));
        try std.testing.expectEqual(api.Spill{ .gib = 64 }, try spill(a, .{ .spill_gib = 64 }, &bad, &problem));
    }
    var empty = std.process.Environ.Map.init(a);
    try empty.put("TF_GLM_SPILL_QUOTA", "");
    try std.testing.expectEqual(@as(f64, 0), (try spill(a, tier, &empty, &problem)).quota); // empty: unset
    // the engine is asked for them whole, beside v1's flat fields
    const o = try request(a, "d", "t", .{ .snapshot_dir = "/spill", .spill_gib = 64, .spill = on });
    try std.testing.expectEqual(on, o.spill);
    try std.testing.expectEqualStrings("/spill", o.spill_dir.?);
    var out: std.Io.Writer.Allocating = .init(a);
    try capabilities(&out.writer, .{ .version = "0" });
    for (spill_env) |name| try std.testing.expectEqual(cuda_build, std.mem.indexOf(u8, out.written(), name) != null);
    if (cuda_build) {
        var u: Usage = .{};
        try std.testing.expect((try parse(a, &.{ "m", "--spill-highwater", "0.5" }, &u)).spill_highwater_given);
        try std.testing.expect(!(try parse(a, &.{"m"}, &u)).spill_highwater_given);
    }
}
