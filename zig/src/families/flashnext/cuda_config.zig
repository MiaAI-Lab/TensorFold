//! Flash Next's dimensions for the CUDA engine: config.json (the root and its text_config) read into every field the
//! Python CUDA engine's Config.read takes (qwen4_exp/cuda/weight_types.py), its derived sizes, ModelOpt's
//! quantization_config, the MTP block, the end-of-reply ids (config.json's, then generation_config.json's), and the rope
//! options with a caller's YaRN override (cuda_rope.zig does the rope math). std only: the Metal family's config.zig
//! lives outside this module's import path.
const std = @import("std");
const ngram = @import("cuda_ngram.zig");

/// config.json model_type values this family serves (Python qwen4_exp MODEL_TYPES).
pub const model_types = [_][]const u8{ "qwen4_exp", "qwen3_8_flash_next" };
pub const max_layers = 64;
pub const max_ple = 8;
pub const max_eos = 8;
pub const max_mtp_layers = 4;

pub const LayerType = enum { linear, attention };
pub const Quant = enum { mlx, modelopt, gptq };

/// Why a check refused the checkpoint, kept for the caller's log line (tests read it).
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const RopeType = enum { default, yarn };

/// YaRN as HF / vLLM name it (rope_parameters with rope_type "yarn"); the caller's override for 1M contexts.
pub const Yarn = struct {
    factor: f64,
    original_max_position_embeddings: u64,
    beta_fast: f64 = 32,
    beta_slow: f64 = 1,
    /// null: HF's default 0.1 * ln(factor) + 1
    attention_factor: ?f64 = null,
    truncate: bool = true,
};

pub const Rope = struct {
    kind: RopeType = .default,
    theta: f64,
    /// partial_rotary_factor: rotary_dim = int(head_dim * partial)
    partial: f64,
    rotary_dim: u32,
    mrope_section: [3]u32,
    mrope_interleaved: bool,
    /// text_config.max_position_embeddings (the native window)
    max_position: u64,
    /// YaRN's fields (kind == .yarn), from the checkpoint or the caller
    yarn: ?Yarn = null,
};

pub const Options = struct {
    /// replaces the checkpoint's rope scaling with YaRN (work/PLAN.md "1M context")
    yarn: ?Yarn = null,
};

/// text_config.mtp and its siblings: the multi-token-prediction head.
pub const Mtp = struct {
    layers: u32 = 0,
    layer_types: [max_mtp_layers]LayerType = @splat(.attention),
    hybrid: bool = false,
    rope_theta: ?f64 = null,
    hidden_from_layer: ?i64 = null,
    dedicated_embeddings: bool = false,
};

/// ModelOpt's per-module algorithms in quantized_layers.
pub const Algo = enum { nvfp4, fp8, fp8_pb_wo, other };

pub const Spec = struct { bits: u32, kind: []const u8, group: ?u32, dynamic: bool };
pub const Group = struct { name: []const u8, targets: []const []const u8, weights: ?Spec, inputs: ?Spec };
pub const QuantizedLayer = struct { name: []const u8, algo: Algo, algo_name: []const u8, group: ?u32 };

/// quantization_config of a ModelOpt export (quant_method "modelopt").
pub const ModelOpt = struct {
    algo: []const u8,
    producer: []const u8 = "",
    version: []const u8 = "",
    groups: []const Group,
    layers: []const QuantizedLayer,
    ignore: []const []const u8,

    /// The quantized_layers entry of `module` (e.g. "model.language_model.layers.3.mlp.experts").
    pub fn layer(self: ModelOpt, module: []const u8) ?QuantizedLayer {
        for (self.layers) |l| if (std.mem.eql(u8, l.name, module)) return l;
        return null;
    }

    /// Whether an ignore glob matches `module` (fnmatch '*' and '?', as ModelOpt writes them).
    pub fn ignored(self: ModelOpt, module: []const u8) bool {
        for (self.ignore) |g| if (glob(g, module)) return true;
        return false;
    }
};

pub fn glob(pattern: []const u8, name: []const u8) bool {
    var p: usize = 0;
    var n: usize = 0;
    var star: ?usize = null;
    var mark: usize = 0;
    while (n < name.len) {
        if (p < pattern.len and (pattern[p] == '?' or pattern[p] == name[n])) {
            p += 1;
            n += 1;
        } else if (p < pattern.len and pattern[p] == '*') {
            star = p;
            mark = n;
            p += 1;
        } else if (star) |s| {
            p = s + 1;
            mark += 1;
            n = mark;
        } else return false;
    }
    while (p < pattern.len and pattern[p] == '*') p += 1;
    return p == pattern.len;
}

/// One `dynamic` rule of a GPTQModel / AutoRound quantization_config: "-:regex" leaves the modules it matches
/// unquantized, "+:regex" overrides their settings (re.match: anchored at the start).
pub const Rule = struct { skip: bool, pattern: []const u8, bits: ?u32 = null };

/// quantization_config of a GPTQ export (quant_method "gptq": AutoRound's and GPTQModel's).
pub const Gptq = struct {
    bits: u32,
    group: u32,
    sym: bool,
    desc_act: bool,
    lm_head: bool,
    format: []const u8 = "",
    dynamic: []const Rule = &.{},

    /// The bits `module` is quantized at (its last matching "+" rule's, else the base), null when unquantized.
    pub fn bitsOf(self: Gptq, module: []const u8) ?u32 {
        if (!self.quantized(module)) return null;
        var bits = self.bits;
        for (self.dynamic) |r| if (!r.skip and regexMatch(r.pattern, module)) if (r.bits) |b| {
            bits = b;
        };
        return bits;
    }

    /// Whether `module` (e.g. "model.language_model.layers.3.mlp.experts.7.gate_proj") is quantized: no "-" rule
    /// matches it (the lm_head only when `lm_head`).
    pub fn quantized(self: Gptq, module: []const u8) bool {
        for (self.dynamic) |r| if (r.skip and regexMatch(r.pattern, module)) return false;
        if (std.mem.eql(u8, module, "lm_head") and !self.lm_head) {
            for (self.dynamic) |r| if (!r.skip and regexMatch(r.pattern, module)) return true;
            return false;
        }
        return true;
    }
};

/// Python's re.match for the subset GPTQ dynamic rules use: literals, '.', '\x' escapes, '*' and '+' after an
/// atom, a final '$'. Anchored at the start; without '$' a prefix match is enough.
pub fn regexMatch(pattern: []const u8, text: []const u8) bool {
    return reAt(pattern, 0, text, 0);
}

fn reAtom(pattern: []const u8, p: usize) struct { len: usize, lit: ?u8 } {
    if (pattern[p] == '\\' and p + 1 < pattern.len) return .{ .len = 2, .lit = pattern[p + 1] };
    if (pattern[p] == '.') return .{ .len = 1, .lit = null };
    return .{ .len = 1, .lit = pattern[p] };
}

fn reAt(pattern: []const u8, p: usize, text: []const u8, t: usize) bool {
    if (p == pattern.len) return true;
    if (pattern[p] == '$' and p + 1 == pattern.len) return t == text.len;
    const atom = reAtom(pattern, p);
    const next = p + atom.len;
    const one = struct {
        fn f(lit: ?u8, c: u8) bool {
            return if (lit) |l| l == c else true;
        }
    }.f;
    if (next < pattern.len and (pattern[next] == '*' or pattern[next] == '+')) {
        const least: usize = if (pattern[next] == '+') 1 else 0;
        var n: usize = 0;
        while (t + n < text.len and one(atom.lit, text[t + n])) n += 1;
        var k: usize = n + 1;
        while (k > least) {
            k -= 1;
            if (reAt(pattern, next + 1, text, t + k)) return true;
        }
        return false;
    }
    if (t < text.len and one(atom.lit, text[t])) return reAt(pattern, next, text, t + 1);
    return false;
}

/// A rank's shares of the split dimensions (Python weights.load's `replace(full, heads // world, ...)`).
pub const Shares = struct { heads: u32, kv_heads: u32, nk: u32, nv: u32, moe_width: u32, shared_width: u32 };

pub const Config = struct {
    arena: std.heap.ArenaAllocator,
    model_type: []const u8,
    hidden: u32,
    layers: u32,
    layer_types: [max_layers]LayerType = @splat(.linear),
    vocab: u32,
    eps: f64,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rope: Rope,
    nk: u32,
    nv: u32,
    dk: u32,
    dv: u32,
    conv_kernel: u32,
    experts: u32,
    top_k: u32,
    moe_width: u32,
    shared_width: u32,
    norm_topk: bool,
    /// hc_count, hc_lowrank: the hyper-connection streams and their low rank
    streams: u32,
    low: u32,
    index_heads: u32,
    index_dim: u32,
    index_kv_heads: u32,
    index_budget: u32,
    index_ratio: u32,
    /// zero-indexed decoder layers with the n-gram embedding (ple_layer_ids are one-indexed)
    ple_layers: [max_ple]u32 = @splat(0),
    ple_count: u32 = 0,
    ple_dim: u32,
    ple_kernel: u32,
    ngram_size: u32,
    heads_per_ngram: u32,
    ngram_base: u64,
    ngram_divisor: u64,
    ngram_shards: u32,
    seed: u64,
    /// the n-gram resets' EOS (text_config.eos_token_id, its first entry)
    ple_eos: i64,
    /// end-of-reply ids: config.json's (the root's, else text_config's), then generation_config.json's not yet listed
    eos: [max_eos]i64 = @splat(0),
    eos_count: u32 = 0,
    bos: ?i64 = null,
    tie_embeddings: bool,
    /// MLX's affine group (32 when the block names none, as Python reads it) and bits
    group_size: u32,
    bits: u32,
    quant: Quant,
    /// NVFP4's block (config_groups.group_0.weights.group_size, else 16)
    nvfp4_group: u32,
    modelopt: ?ModelOpt = null,
    /// a GPTQ export's settings (quant == .gptq: the INT4-AutoRound checkpoint)
    gptq: ?Gptq = null,
    mtp: Mtp = .{},

    pub fn deinit(self: *Config) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// `dir`/config.json and, when present, `dir`/generation_config.json.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(text);
        const gpath = try std.fs.path.join(gpa, &.{ dir, "generation_config.json" });
        defer gpa.free(gpath);
        const gen: ?[]u8 = std.Io.Dir.cwd().readFileAlloc(io, gpath, gpa, .limited(1 << 20)) catch |e| switch (e) {
            error.FileNotFound => null,
            else => return e,
        };
        defer if (gen) |g| gpa.free(g);
        var why: Why = .{};
        return parse(gpa, text, gen, o, &why) catch |e| {
            if (why.len > 0) std.log.err("{s}: {s}", .{ path, why.text() });
            return e;
        };
    }

    pub fn layerType(self: Config, i: usize) LayerType {
        return self.layer_types[i];
    }

    pub fn count(self: Config, kind: LayerType) u32 {
        var n: u32 = 0;
        for (self.layer_types[0..self.layers]) |k| n += @intFromBool(k == kind);
        return n;
    }

    /// 2 * nk * dk + nv * dv: the DeltaNet conv's channels (q, k, v)
    pub fn convDim(self: Config) u32 {
        return 2 * self.nk * self.dk + self.nv * self.dv;
    }

    /// index_budget / index_ratio: the indexer's top blocks
    pub fn topBlocks(self: Config) u32 {
        return self.index_budget / self.index_ratio;
    }

    pub fn pleLayers(self: *const Config) []const u32 {
        return self.ple_layers[0..self.ple_count];
    }

    /// The position of decoder layer `i` among the PLE layers (Python `cfg.ple_layers.index(i)`), or null.
    pub fn pleIndex(self: Config, i: u32) ?u32 {
        for (self.ple_layers[0..self.ple_count], 0..) |l, j| if (l == i) return @intCast(j);
        return null;
    }

    pub fn stops(self: *const Config) []const i64 {
        return self.eos[0..self.eos_count];
    }

    pub fn isEos(self: Config, token: i64) bool {
        for (self.eos[0..self.eos_count]) |e| if (e == token) return true;
        return false;
    }

    /// The n-gram row ids' options for PLE layer `ple_index` (Python Config.ngram).
    pub fn ngramOptions(self: Config, ple_index: u32) ngram.Options {
        return .{ .vocab = self.vocab, .ngram_size = self.ngram_size, .heads_per_ngram = self.heads_per_ngram, .vocab_base = self.ngram_base, .divisor = self.ngram_divisor, .seed = self.seed, .eos = self.ple_eos, .embed_dim = self.ple_dim, .ple_index = ple_index };
    }

    /// The n-gram heads: (ngram_size - 1) * heads_per_ngram.
    pub fn ngramHeads(self: Config) u32 {
        return (self.ngram_size - 1) * self.heads_per_ngram;
    }

    /// Rank `rank` of `world`: every split dimension divided (refused unless it divides, and the NVFP4 groups and the
    /// 32-column pack blocks of the half expert width stay whole).
    pub fn shares(self: Config, world: u32) !Shares {
        if (world == 0) return error.BadWorld;
        for ([_]u32{ self.heads, self.kv_heads, self.nk, self.nv, self.moe_width, self.shared_width }) |d| if (d % world != 0) return error.WorldDoesNotDivide;
        const w = self.moe_width / world;
        const s = self.shared_width / world;
        if (w % 32 != 0 or w % self.nvfp4_group != 0 or s % 64 != 0) return error.WorldDoesNotDivide;
        if (self.ngramHeads() % world != 0) return error.WorldDoesNotDivide;
        return .{ .heads = self.heads / world, .kv_heads = self.kv_heads / world, .nk = self.nk / world, .nv = self.nv / world, .moe_width = w, .shared_width = s };
    }

    pub fn checkGptq(self: Config, why: *Why) !void {
        return checkGptqImpl(self, why);
    }

    /// The INT4-AutoRound format (GPTQ experts and head, block-FP8 dense linears).
    pub fn int4ar(self: Config) bool {
        return self.quant == .gptq;
    }

    /// What the engine serves: the ModelOpt NVFP4 export (routed experts NVFP4 in groups of 16, the MTP experts
    /// FP8_PB_WO in 128 blocks, the n-gram table FP8, every other linear bf16); anything else is refused with a reason.
    pub fn check(self: Config, why: *Why) !void {
        var known = false;
        for (model_types) |m| known = known or std.mem.eql(u8, m, self.model_type);
        if (!known) {
            why.set("model_type {s} is not Flash Next's (qwen4_exp, qwen3_8_flash_next)", .{self.model_type});
            return error.UnsupportedModel;
        }
        if (self.quant == .gptq) return self.checkGptq(why);
        if (self.quant != .modelopt) {
            why.set("the CUDA Flash Next engine reads the ModelOpt NVFP4 export; config.json's quantization is {t}", .{self.quant});
            return error.UnsupportedQuantization;
        }
        const mo = self.modelopt.?;
        if (!std.mem.eql(u8, mo.algo, "MIXED_PRECISION") and !std.mem.eql(u8, mo.algo, "NVFP4")) {
            why.set("ModelOpt quant_algo {s}; the engine reads MIXED_PRECISION (NVFP4 experts) exports", .{mo.algo});
            return error.UnsupportedQuantization;
        }
        if (self.nvfp4_group != 16) {
            why.set("NVFP4 group size {d}; the expert kernels read blocks of 16", .{self.nvfp4_group});
            return error.UnsupportedQuantization;
        }
        for (mo.layers) |l| {
            const experts = std.mem.endsWith(u8, l.name, ".mlp.experts");
            const ok = if (std.mem.startsWith(u8, l.name, "mtp.") and experts)
                l.algo == .fp8_pb_wo and (l.group orelse 128) == 128
            else if (experts)
                l.algo == .nvfp4 and (l.group orelse 16) == 16
            else if (std.mem.endsWith(u8, l.name, ".ngram_embedding"))
                l.algo == .fp8
            else
                false;
            if (!ok) {
                why.set("ModelOpt quantizes {s} as {s}; the engine reads NVFP4 routed experts, FP8_PB_WO MTP experts and an FP8 n-gram table, the rest bf16", .{ l.name, l.algo_name });
                return error.UnsupportedQuantization;
            }
        }
        if (self.layers == 0 or self.layers > max_layers) {
            why.set("{d} layers; the engine holds up to {d}", .{ self.layers, max_layers });
            return error.UnsupportedModel;
        }
        if (self.ngram_size < 2 or self.ngram_size > ngram.max_n or self.ngramHeads() > ngram.max_heads or self.ple_dim % self.ngramHeads() != 0) {
            why.set("n-gram size {d} with {d} heads a gram does not tile {d} dims", .{ self.ngram_size, self.heads_per_ngram, self.ple_dim });
            return error.UnsupportedModel;
        }
    }
};

pub const int4ar_group = 128;

/// The INT4-AutoRound checkpoint (GPTQ int4 g128 sym routed experts and lm_head, block-FP8 attention, DeltaNet and
/// shared expert, bf16 rest, a bf16 MTP layer): what the engine serves of a GPTQ export.
fn checkGptqImpl(self: Config, why: *Why) !void {
    const g = self.gptq.?;
    if (self.layers == 0 or self.layers > max_layers) {
        why.set("{d} layers; the engine holds up to {d}", .{ self.layers, max_layers });
        return error.UnsupportedModel;
    }
    if (g.bits != 4 or g.group != int4ar_group or !g.sym or g.desc_act) {
        why.set("GPTQ {d}-bit, group {d}, sym {}, desc_act {}; the engine reads 4-bit symmetric groups of 128 without act order", .{ g.bits, g.group, g.sym, g.desc_act });
        return error.UnsupportedQuantization;
    }
    var buf: [128]u8 = undefined;
    const expert = std.fmt.bufPrint(&buf, "model.language_model.layers.{d}.mlp.experts.0.gate_proj", .{self.layers - 1}) catch unreachable;
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
        var eb: [160]u8 = undefined;
        const name = std.fmt.bufPrint(&eb, "model.language_model.layers.0.mlp.experts.0.{s}", .{proj}) catch unreachable;
        if (g.bitsOf(name) != 4 or g.bitsOf(expert) != 4) {
            why.set("GPTQ quantizes the routed experts at {?d} bits; the engine reads 4-bit routed experts", .{g.bitsOf(name)});
            return error.UnsupportedQuantization;
        }
    }
    if (g.bitsOf("lm_head") != 4) {
        why.set("GPTQ's lm_head is {?d}-bit (null: unquantized); the engine reads the INT4-AutoRound 4-bit head", .{g.bitsOf("lm_head")});
        return error.UnsupportedQuantization;
    }
    for ([_][]const u8{ "model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.3.self_attn.q_proj", "model.language_model.layers.0.mlp.shared_expert.gate_proj", "model.language_model.layers.0.mlp.gate", "model.language_model.embed_tokens" }) |m| if (g.quantized(m)) {
        why.set("GPTQ quantizes {s}; the engine reads only the routed experts and lm_head as GPTQ int4", .{m});
        return error.UnsupportedQuantization;
    };
    if (self.layers == 0 or self.layers > max_layers) {
        why.set("{d} layers; the engine holds up to {d}", .{ self.layers, max_layers });
        return error.UnsupportedModel;
    }
    if (self.ngram_size < 2 or self.ngram_size > ngram.max_n or self.ngramHeads() > ngram.max_heads or self.ple_dim % self.ngramHeads() != 0) {
        why.set("n-gram size {d} with {d} heads a gram does not tile {d} dims", .{ self.ngram_size, self.heads_per_ngram, self.ple_dim });
        return error.UnsupportedModel;
    }
    if (self.moe_width % int4ar_group != 0 or self.hidden % int4ar_group != 0) {
        why.set("expert width {d} / hidden {d} are not whole groups of {d}", .{ self.moe_width, self.hidden, int4ar_group });
        return error.UnsupportedModel;
    }
}

const Value = std.json.Value;
const Obj = std.json.ObjectMap;

fn field(o: Obj, key: []const u8) ?Value {
    const v = o.get(key) orelse return null;
    return if (v == .null) null else v;
}

fn number(v: Value) !f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f64, s) catch error.BadConfig,
        else => error.BadConfig,
    };
}

fn integer(v: Value) !i64 {
    return switch (v) {
        .integer => |i| i,
        .float => |f| if (@trunc(f) == f) @intFromFloat(f) else error.BadConfig,
        else => error.BadConfig,
    };
}

/// A required count (> 0 where `positive`).
fn need(o: Obj, key: []const u8, why: *Why) !u32 {
    const v = field(o, key) orelse {
        why.set("config.json has no {s}", .{key});
        return error.BadConfig;
    };
    const i = integer(v) catch {
        why.set("config.json's {s} is not an integer", .{key});
        return error.BadConfig;
    };
    if (i < 0 or i > std.math.maxInt(u32)) {
        why.set("config.json's {s} is {d}", .{ key, i });
        return error.BadConfig;
    }
    return @intCast(i);
}

fn opt(o: Obj, key: []const u8, default: u32, why: *Why) !u32 {
    if (field(o, key) == null) return default;
    return need(o, key, why);
}

fn optWide(o: Obj, key: []const u8, default: u64) !u64 {
    const v = field(o, key) orelse return default;
    const i = try integer(v);
    if (i < 0) return error.BadConfig;
    return @intCast(i);
}

fn optFloat(o: Obj, key: []const u8, default: f64) !f64 {
    return number(field(o, key) orelse return default);
}

fn optBool(o: Obj, key: []const u8, default: bool) bool {
    const v = field(o, key) orelse return default;
    return if (v == .bool) v.bool else default;
}

fn object(o: Obj, key: []const u8) ?Obj {
    const v = field(o, key) orelse return null;
    return if (v == .object) v.object else null;
}

fn str(a: std.mem.Allocator, o: Obj, key: []const u8) ![]const u8 {
    const v = field(o, key) orelse return "";
    return if (v == .string) try a.dupe(u8, v.string) else "";
}

/// An id or a list of ids (eos_token_id's two spellings).
fn ids(v: ?Value, out: *std.ArrayList(i64), a: std.mem.Allocator) !void {
    const x = v orelse return;
    switch (x) {
        .array => |arr| for (arr.items) |e| try out.append(a, try integer(e)),
        else => try out.append(a, try integer(x)),
    }
}

fn algoOf(name: []const u8) Algo {
    if (std.ascii.eqlIgnoreCase(name, "NVFP4")) return .nvfp4;
    if (std.ascii.eqlIgnoreCase(name, "FP8")) return .fp8;
    if (std.ascii.eqlIgnoreCase(name, "FP8_PB_WO")) return .fp8_pb_wo;
    return .other;
}

fn spec(a: std.mem.Allocator, o: ?Obj) !?Spec {
    const s = o orelse return null;
    return .{
        .bits = @intCast(try optWide(s, "num_bits", 0)),
        .kind = try str(a, s, "type"),
        .group = if (field(s, "group_size")) |g| @as(u32, @intCast(try integer(g))) else null,
        .dynamic = optBool(s, "dynamic", false),
    };
}

fn modelOpt(a: std.mem.Allocator, q: Obj) !ModelOpt {
    var groups: std.ArrayList(Group) = .empty;
    if (object(q, "config_groups")) |cg| {
        var it = cg.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.* != .object) continue;
            const g = kv.value_ptr.object;
            var targets: std.ArrayList([]const u8) = .empty;
            if (field(g, "targets")) |t| if (t == .array) for (t.array.items) |x| if (x == .string) try targets.append(a, try a.dupe(u8, x.string));
            try groups.append(a, .{ .name = try a.dupe(u8, kv.key_ptr.*), .targets = targets.items, .weights = try spec(a, object(g, "weights")), .inputs = try spec(a, object(g, "input_activations")) });
        }
    }
    var layers: std.ArrayList(QuantizedLayer) = .empty;
    if (object(q, "quantized_layers")) |ql| {
        var it = ql.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.* != .object) return error.BadConfig;
            const l = kv.value_ptr.object;
            const name = try str(a, l, "quant_algo");
            try layers.append(a, .{ .name = try a.dupe(u8, kv.key_ptr.*), .algo = algoOf(name), .algo_name = name, .group = if (field(l, "group_size")) |g| @as(u32, @intCast(try integer(g))) else null });
        }
    }
    var ignore: std.ArrayList([]const u8) = .empty;
    if (field(q, "ignore")) |v| if (v == .array) for (v.array.items) |x| if (x == .string) try ignore.append(a, try a.dupe(u8, x.string));
    var producer: []const u8 = "";
    var version: []const u8 = "";
    if (object(q, "producer")) |p| {
        producer = try str(a, p, "name");
        version = try str(a, p, "version");
    }
    return .{ .algo = try str(a, q, "quant_algo"), .producer = producer, .version = version, .groups = groups.items, .layers = layers.items, .ignore = ignore.items };
}

fn gptqOf(a: std.mem.Allocator, q: Obj, why: *Why) !Gptq {
    var rules: std.ArrayList(Rule) = .empty;
    if (object(q, "dynamic")) |d| {
        var it = d.iterator();
        while (it.next()) |kv| {
            const key = kv.key_ptr.*;
            if (key.len < 2 or key[1] != ':' or (key[0] != '-' and key[0] != '+')) {
                why.set("GPTQ dynamic rule {s}: expected \"-:regex\" or \"+:regex\"", .{key});
                return error.BadConfig;
            }
            var bits: ?u32 = null;
            if (kv.value_ptr.* == .object) if (field(kv.value_ptr.object, "bits")) |b| {
                bits = @intCast(try integer(b));
            };
            try rules.append(a, .{ .skip = key[0] == '-', .pattern = try a.dupe(u8, key[2..]), .bits = bits });
        }
    }
    return .{
        .bits = @intCast(try optWide(q, "bits", 4)),
        .group = @intCast(try optWide(q, "group_size", 128)),
        .sym = optBool(q, "sym", true),
        .desc_act = optBool(q, "desc_act", false),
        .lm_head = optBool(q, "lm_head", false),
        .format = try str(a, q, "checkpoint_format"),
        .dynamic = rules.items,
    };
}

fn yarnOf(r: Obj, why: *Why) !Yarn {
    const factor = optFloat(r, "factor", 0) catch 0;
    const original = optWide(r, "original_max_position_embeddings", 0) catch 0;
    if (!(factor > 0) or original == 0) {
        why.set("rope_type yarn needs factor and original_max_position_embeddings", .{});
        return error.BadConfig;
    }
    return .{
        .factor = factor,
        .original_max_position_embeddings = original,
        .beta_fast = try optFloat(r, "beta_fast", 32),
        .beta_slow = try optFloat(r, "beta_slow", 1),
        .attention_factor = if (field(r, "attention_factor")) |v| try number(v) else null,
        .truncate = optBool(r, "truncate", true),
    };
}

/// config.json's text (and generation_config.json's, when the model has one) into a Config.
pub fn parse(gpa: std.mem.Allocator, text: []const u8, generation: ?[]const u8, o: Options, why: *Why) !Config {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const root_v = std.json.parseFromSliceLeaky(Value, a, text, .{}) catch {
        why.set("config.json is not JSON", .{});
        return error.BadConfig;
    };
    if (root_v != .object) return error.BadConfig;
    const raw = root_v.object;
    const t = object(raw, "text_config") orelse raw;
    const rope_o = object(t, "rope_parameters");
    const model_type = try str(a, raw, "model_type");

    const hidden = try need(t, "hidden_size", why);
    const heads = try need(t, "num_attention_heads", why);
    const head_dim = if (field(t, "head_dim") != null) try need(t, "head_dim", why) else hidden / @max(heads, 1);
    const partial = if (rope_o) |r| (if (field(r, "partial_rotary_factor")) |v| try number(v) else try optFloat(t, "partial_rotary_factor", 0.25)) else try optFloat(t, "partial_rotary_factor", 0.25);
    var rope: Rope = .{
        .theta = if (rope_o) |r| try optFloat(r, "rope_theta", 10_000_000) else 10_000_000,
        .partial = partial,
        .rotary_dim = @intFromFloat(@trunc(@as(f64, @floatFromInt(head_dim)) * partial)),
        .mrope_section = .{ 11, 11, 10 },
        .mrope_interleaved = if (rope_o) |r| optBool(r, "mrope_interleaved", true) else true,
        .max_position = try optWide(t, "max_position_embeddings", 262144),
    };
    if (rope_o) |r| {
        if (field(r, "mrope_section")) |v| {
            if (v != .array or v.array.items.len != 3) {
                why.set("rope_parameters.mrope_section must hold three sizes", .{});
                return error.BadConfig;
            }
            for (v.array.items, 0..) |x, i| rope.mrope_section[i] = @intCast(try integer(x));
        }
        if (field(r, "rope_type")) |v| if (v == .string) {
            if (std.mem.eql(u8, v.string, "yarn")) {
                rope.kind = .yarn;
                rope.yarn = try yarnOf(r, why);
            } else if (!std.mem.eql(u8, v.string, "default")) {
                why.set("rope_type {s}; the engine reads default rope (or YaRN)", .{v.string});
                return error.UnsupportedRope;
            }
        };
    }
    if (o.yarn) |y| {
        if (!(y.factor > 0) or y.original_max_position_embeddings == 0) return error.BadConfig;
        rope.kind = .yarn;
        rope.yarn = y;
    }

    var c: Config = .{
        .arena = undefined,
        .model_type = model_type,
        .hidden = hidden,
        .layers = try need(t, "num_hidden_layers", why),
        .vocab = try need(t, "vocab_size", why),
        .eps = optFloat(t, "rms_norm_eps", -1) catch -1,
        .heads = heads,
        .kv_heads = try need(t, "num_key_value_heads", why),
        .head_dim = head_dim,
        .rope = rope,
        .nk = try need(t, "linear_num_key_heads", why),
        .nv = try need(t, "linear_num_value_heads", why),
        .dk = try need(t, "linear_key_head_dim", why),
        .dv = try need(t, "linear_value_head_dim", why),
        .conv_kernel = try need(t, "linear_conv_kernel_dim", why),
        .experts = try need(t, "num_experts", why),
        .top_k = try need(t, "num_experts_per_tok", why),
        .moe_width = try need(t, "moe_intermediate_size", why),
        .shared_width = try need(t, "shared_expert_intermediate_size", why),
        .norm_topk = optBool(t, "norm_topk_prob", true),
        .streams = try opt(t, "hc_count", 4, why),
        .low = try opt(t, "hc_lowrank", 320, why),
        .index_heads = try opt(t, "indexer_n_heads", 4, why),
        .index_dim = try opt(t, "indexer_head_dim", 128, why),
        .index_kv_heads = try opt(t, "indexer_kv_heads", 1, why),
        .index_budget = try opt(t, "indexer_budget", 2048, why),
        .index_ratio = try opt(t, "indexer_compress_ratio", 4, why),
        .ple_dim = if (field(t, "ple_embed_dim") != null) try need(t, "ple_embed_dim", why) else hidden,
        .ple_kernel = try opt(t, "ple_conv_kernel_size", 4, why),
        .ngram_size = try opt(t, "ngram_size", 3, why),
        .heads_per_ngram = try opt(t, "heads_per_ngram", 8, why),
        .ngram_base = try optWide(t, "ngram_vocab_size_base", 20_000_000),
        .ngram_divisor = try optWide(t, "make_ngram_vocab_size_divisible_by", 128),
        .ngram_shards = try opt(t, "split_ngram_parts", 128, why),
        .seed = try optWide(t, "seed", 1234),
        .ple_eos = 0,
        .tie_embeddings = optBool(raw, "tie_word_embeddings", optBool(t, "tie_word_embeddings", false)),
        .group_size = 32,
        .bits = 4,
        .quant = .mlx,
        .nvfp4_group = 16,
    };
    if (c.eps <= 0) {
        why.set("config.json has no rms_norm_eps", .{});
        return error.BadConfig;
    }
    if (c.index_ratio == 0 or c.ngram_divisor == 0) return error.BadConfig;
    if (field(t, "bos_token_id")) |v| c.bos = integer(v) catch null;

    // layer types: linear_attention -> linear, every other name -> attention (Python's reading)
    const lt = field(t, "layer_types") orelse {
        why.set("config.json has no layer_types", .{});
        return error.BadConfig;
    };
    if (lt != .array or lt.array.items.len != c.layers or c.layers > max_layers) {
        why.set("layer_types lists {d} layers for num_hidden_layers {d} (at most {d})", .{ if (lt == .array) lt.array.items.len else 0, c.layers, max_layers });
        return error.BadConfig;
    }
    for (lt.array.items, 0..) |v, i| c.layer_types[i] = if (v == .string and std.mem.eql(u8, v.string, "linear_attention")) .linear else .attention;

    // ple_layer_ids are one-indexed: a sorted set of zero-indexed layers
    if (field(t, "ple_layer_ids")) |v| {
        if (v != .array) return error.BadConfig;
        for (v.array.items) |x| {
            const id = try integer(x) - 1;
            if (id < 0 or id >= c.layers) {
                why.set("ple_layer_ids names layer {d} of {d}", .{ id + 1, c.layers });
                return error.BadConfig;
            }
            const l: u32 = @intCast(id);
            if (c.pleIndex(l) != null) continue;
            if (c.ple_count == max_ple) return error.BadConfig;
            c.ple_layers[c.ple_count] = l;
            c.ple_count += 1;
        }
        std.mem.sort(u32, c.ple_layers[0..c.ple_count], {}, std.sort.asc(u32));
    }

    // end-of-reply ids: config.json's own (its root's, else text_config's), then generation_config.json's not yet listed
    var stop: std.ArrayList(i64) = .empty;
    const teos = field(t, "eos_token_id");
    if (teos) |v| c.ple_eos = switch (v) {
        .array => |arr| if (arr.items.len > 0) try integer(arr.items[0]) else 0,
        else => try integer(v),
    };
    ids(field(raw, "eos_token_id") orelse teos, &stop, a) catch {
        why.set("config.json's eos_token_id is not an id or a list of ids", .{});
        return error.BadConfig;
    };
    if (generation) |g| {
        const gv = std.json.parseFromSliceLeaky(Value, a, g, .{}) catch {
            why.set("generation_config.json is not JSON", .{});
            return error.BadConfig;
        };
        if (gv == .object) {
            var more: std.ArrayList(i64) = .empty;
            ids(field(gv.object, "eos_token_id"), &more, a) catch {
                why.set("generation_config.json's eos_token_id is not an id or a list of ids", .{});
                return error.BadConfig;
            };
            for (more.items) |id| if (std.mem.indexOfScalar(i64, stop.items, id) == null) try stop.append(a, id);
        }
    }
    // a list may repeat an id; keep its first place
    for (stop.items) |id| {
        if (std.mem.indexOfScalar(i64, c.eos[0..c.eos_count], id) != null) continue;
        if (c.eos_count == max_eos) return error.BadConfig;
        c.eos[c.eos_count] = id;
        c.eos_count += 1;
    }
    if (c.eos_count == 0) {
        why.set("no eos_token_id in config.json or generation_config.json", .{});
        return error.BadConfig;
    }

    // quantization: "quantization" (MLX) or "quantization_config" (ModelOpt), as Python reads either
    const q = object(raw, "quantization") orelse object(raw, "quantization_config");
    if (q) |qq| {
        const method = try str(a, qq, "quant_method");
        if (method.len == 0 or std.ascii.eqlIgnoreCase(method, "mlx")) {
            c.quant = .mlx;
        } else if (std.ascii.eqlIgnoreCase(method, "modelopt")) {
            c.quant = .modelopt;
            c.modelopt = try modelOpt(a, qq);
        } else if (std.ascii.eqlIgnoreCase(method, "gptq")) {
            c.quant = .gptq;
            c.gptq = try gptqOf(a, qq, why);
        } else {
            why.set("quant_method {s}; Flash Next reads MLX or ModelOpt checkpoints", .{method});
            return error.UnsupportedQuantization;
        }
        c.group_size = try opt(qq, "group_size", 32, why);
        c.bits = try opt(qq, "bits", 4, why);
        if (object(qq, "config_groups")) |cg| if (object(cg, "group_0")) |g0| if (object(g0, "weights")) |w| {
            c.nvfp4_group = try opt(w, "group_size", 16, why);
        };
    }

    // the MTP head
    c.mtp.layers = try opt(t, "mtp_num_hidden_layers", 0, why);
    c.mtp.dedicated_embeddings = optBool(t, "mtp_use_dedicated_embeddings", false);
    if (object(t, "mtp")) |m| {
        c.mtp.layers = try opt(m, "num_hidden_layers", c.mtp.layers, why);
        c.mtp.hybrid = optBool(m, "hybrid", false);
        if (field(m, "rope_theta")) |v| c.mtp.rope_theta = try number(v);
        if (field(m, "mtp_use_hidden_state_from_layer")) |v| c.mtp.hidden_from_layer = try integer(v);
        if (field(m, "layer_types")) |v| if (v == .array) for (v.array.items, 0..) |x, i| {
            if (i >= max_mtp_layers) break;
            c.mtp.layer_types[i] = if (x == .string and std.mem.eql(u8, x.string, "linear_attention")) .linear else .attention;
        };
    }
    if (c.mtp.layers > max_mtp_layers) return error.BadConfig;
    c.arena = arena;
    return c;
}

// ---- tests: the nvidia/Qwen3.8-Flash-Next-NVFP4 snapshot's config.json and generation_config.json ----------------

const fixture = @embedFile("fixtures_cuda_config.json");
const fixture_generation = @embedFile("fixtures_cuda_generation_config.json");

fn testConfig(o: Options) !Config {
    var why: Why = .{};
    return parse(std.testing.allocator, fixture, fixture_generation, o, &why);
}

test "the NVFP4 checkpoint's config.json reads as Python's Config.read" {
    var c = try testConfig(.{});
    defer c.deinit();
    try std.testing.expectEqualStrings("qwen4_exp", c.model_type);
    try std.testing.expectEqual(@as(u32, 2560), c.hidden);
    try std.testing.expectEqual(@as(u32, 48), c.layers);
    try std.testing.expectEqual(@as(u32, 248320), c.vocab);
    try std.testing.expectEqual(@as(f64, 1e-6), c.eps);
    try std.testing.expectEqual(@as(u32, 24), c.heads);
    try std.testing.expectEqual(@as(u32, 2), c.kv_heads);
    try std.testing.expectEqual(@as(u32, 256), c.head_dim);
    try std.testing.expectEqual(@as(f64, 1e7), c.rope.theta);
    try std.testing.expectEqual(@as(u32, 64), c.rope.rotary_dim);
    try std.testing.expectEqual([3]u32{ 11, 11, 10 }, c.rope.mrope_section);
    try std.testing.expect(c.rope.mrope_interleaved);
    try std.testing.expectEqual(RopeType.default, c.rope.kind);
    try std.testing.expectEqual(@as(u64, 262144), c.rope.max_position);
    try std.testing.expectEqual(@as(u32, 16), c.nk);
    try std.testing.expectEqual(@as(u32, 48), c.nv);
    try std.testing.expectEqual(@as(u32, 128), c.dk);
    try std.testing.expectEqual(@as(u32, 128), c.dv);
    try std.testing.expectEqual(@as(u32, 4), c.conv_kernel);
    try std.testing.expectEqual(@as(u32, 10240), c.convDim());
    try std.testing.expectEqual(@as(u32, 512), c.experts);
    try std.testing.expectEqual(@as(u32, 10), c.top_k);
    try std.testing.expectEqual(@as(u32, 640), c.moe_width);
    try std.testing.expectEqual(@as(u32, 640), c.shared_width);
    try std.testing.expectEqual(@as(u32, 4), c.streams);
    try std.testing.expectEqual(@as(u32, 320), c.low);
    try std.testing.expectEqual(@as(u32, 4), c.index_heads);
    try std.testing.expectEqual(@as(u32, 128), c.index_dim);
    try std.testing.expectEqual(@as(u32, 2048), c.index_budget);
    try std.testing.expectEqual(@as(u32, 4), c.index_ratio);
    try std.testing.expectEqual(@as(u32, 512), c.topBlocks());
    try std.testing.expectEqualSlices(u32, &.{1}, c.pleLayers());
    try std.testing.expectEqual(@as(?u32, 0), c.pleIndex(1));
    try std.testing.expectEqual(@as(?u32, null), c.pleIndex(2));
    try std.testing.expectEqual(@as(u32, 2560), c.ple_dim);
    try std.testing.expectEqual(@as(u32, 4), c.ple_kernel);
    try std.testing.expectEqual(@as(u32, 3), c.ngram_size);
    try std.testing.expectEqual(@as(u32, 8), c.heads_per_ngram);
    try std.testing.expectEqual(@as(u32, 16), c.ngramHeads());
    try std.testing.expectEqual(@as(u64, 20_000_000), c.ngram_base);
    try std.testing.expectEqual(@as(u64, 128), c.ngram_divisor);
    try std.testing.expectEqual(@as(u32, 128), c.ngram_shards);
    try std.testing.expectEqual(@as(u64, 1234), c.seed);
    try std.testing.expectEqual(@as(i64, 248044), c.ple_eos);
    // config.json names <|endoftext|> only (text_config); generation_config.json adds <|im_end|>
    try std.testing.expectEqualSlices(i64, &.{ 248044, 248046 }, c.stops());
    try std.testing.expect(c.isEos(248046));
    try std.testing.expectEqual(@as(u32, 32), c.group_size);
    try std.testing.expectEqual(@as(u32, 4), c.bits);
    try std.testing.expectEqual(Quant.modelopt, c.quant);
    try std.testing.expectEqual(@as(u32, 16), c.nvfp4_group);
    try std.testing.expectEqual(@as(u32, 36), c.count(.linear));
    try std.testing.expectEqual(@as(u32, 12), c.count(.attention));
    for (0..48) |i| try std.testing.expectEqual(if (i % 4 == 3) LayerType.attention else LayerType.linear, c.layerType(i));
    try std.testing.expectEqual(@as(u32, 1), c.mtp.layers);
    try std.testing.expect(c.mtp.hybrid);
    try std.testing.expectEqual(LayerType.attention, c.mtp.layer_types[0]);
    try std.testing.expectEqual(@as(?f64, 1e7), c.mtp.rope_theta);
    try std.testing.expectEqual(@as(?i64, null), c.mtp.hidden_from_layer);
    try std.testing.expect(!c.mtp.dedicated_embeddings);
    try std.testing.expect(!c.tie_embeddings);
    var why: Why = .{};
    try c.check(&why);
}

test "ModelOpt's quantization_config: groups, quantized layers and ignore globs" {
    var c = try testConfig(.{});
    defer c.deinit();
    const mo = c.modelopt.?;
    try std.testing.expectEqualStrings("MIXED_PRECISION", mo.algo);
    try std.testing.expectEqualStrings("modelopt", mo.producer);
    try std.testing.expectEqual(@as(usize, 3), mo.groups.len);
    try std.testing.expectEqual(@as(usize, 50), mo.layers.len);
    const l3 = mo.layer("model.language_model.layers.3.mlp.experts").?;
    try std.testing.expectEqual(Algo.nvfp4, l3.algo);
    try std.testing.expectEqual(@as(?u32, 16), l3.group);
    const m = mo.layer("mtp.layers.0.mlp.experts").?;
    try std.testing.expectEqual(Algo.fp8_pb_wo, m.algo);
    try std.testing.expectEqual(@as(?u32, 128), m.group);
    try std.testing.expectEqual(Algo.fp8, mo.layer("model.language_model.layers.1.ple.ple_embedding.ngram_embedding").?.algo);
    try std.testing.expectEqual(@as(usize, 292), mo.ignore.len);
    try std.testing.expect(mo.ignored("lm_head"));
    try std.testing.expect(mo.ignored("model.language_model.hyper_connection_mixer.input_mix_weight_up"));
    try std.testing.expect(mo.ignored("model.visual.blocks.3.attn.qkv"));
    try std.testing.expect(mo.ignored("model.language_model.layers.9.mlp.shared_expert_gate"));
    try std.testing.expect(!mo.ignored("model.language_model.layers.9.mlp.experts"));
    var g1: ?Group = null;
    for (mo.groups) |g| if (std.mem.eql(u8, g.name, "group_1")) {
        g1 = g;
    };
    try std.testing.expectEqual(@as(u32, 8), g1.?.weights.?.bits);
    try std.testing.expectEqual(@as(?u32, 128), g1.?.weights.?.group);
    try std.testing.expect(g1.?.inputs.?.dynamic);
    try std.testing.expectEqualStrings("mtp.layers.0.mlp.experts", g1.?.targets[0]);
}

test "a YaRN override and the alias model_type" {
    var c = try testConfig(.{ .yarn = .{ .factor = 4, .original_max_position_embeddings = 262144 } });
    defer c.deinit();
    try std.testing.expectEqual(RopeType.yarn, c.rope.kind);
    try std.testing.expectEqual(@as(f64, 4), c.rope.yarn.?.factor);
    try std.testing.expectEqual(@as(u64, 262144), c.rope.yarn.?.original_max_position_embeddings);
    try std.testing.expectEqual(@as(f64, 32), c.rope.yarn.?.beta_fast);
    try std.testing.expectEqual(@as(?f64, null), c.rope.yarn.?.attention_factor);

    const alias =
        \\{"model_type": "qwen3_8_flash_next", "eos_token_id": [248046, 248044],
        \\ "text_config": {"hidden_size": 64, "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
        \\  "vocab_size": 1000, "rms_norm_eps": 1e-6, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        \\  "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
        \\  "linear_conv_kernel_dim": 4, "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
        \\  "shared_expert_intermediate_size": 128, "eos_token_id": 7,
        \\  "rope_parameters": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 4096,
        \\   "rope_theta": 1000000, "partial_rotary_factor": 0.5}},
        \\ "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
        \\  "quantized_layers": {"model.language_model.layers.0.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16}}}}
    ;
    var why: Why = .{};
    var b = try parse(std.testing.allocator, alias, "{\"eos_token_id\": 9}", .{}, &why);
    defer b.deinit();
    try std.testing.expectEqualStrings("qwen3_8_flash_next", b.model_type);
    try std.testing.expectEqualSlices(i64, &.{ 248046, 248044, 9 }, b.stops());
    try std.testing.expectEqual(@as(i64, 7), b.ple_eos);
    try std.testing.expectEqual(RopeType.yarn, b.rope.kind);
    try std.testing.expectEqual(@as(f64, 2), b.rope.yarn.?.factor);
    try std.testing.expectEqual(@as(u32, 8), b.rope.rotary_dim);
    try std.testing.expectEqual(@as(u32, 64), b.ple_dim);
    try std.testing.expectEqual(@as(u32, 0), b.ple_count);
    try b.check(&why);
    const s = try b.shares(2);
    try std.testing.expectEqual(@as(u32, 2), s.heads);
    try std.testing.expectEqual(@as(u32, 32), s.moe_width);
    try std.testing.expectEqual(@as(u32, 64), s.shared_width);
}

test "check refuses another quantization, and shares refuse a world that does not divide" {
    var c = try testConfig(.{});
    defer c.deinit();
    const s = try c.shares(2);
    try std.testing.expectEqual(Shares{ .heads = 12, .kv_heads = 1, .nk = 8, .nv = 24, .moe_width = 320, .shared_width = 320 }, s);
    try std.testing.expectError(error.WorldDoesNotDivide, c.shares(3));
    c.nvfp4_group = 32;
    var why: Why = .{};
    try std.testing.expectError(error.UnsupportedQuantization, c.check(&why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "group size 32") != null);
    c.nvfp4_group = 16;
    c.quant = .mlx;
    try std.testing.expectError(error.UnsupportedQuantization, c.check(&why));
}

test "glob" {
    try std.testing.expect(glob("a*", "abc"));
    try std.testing.expect(glob("*.mlp.experts", "x.layers.1.mlp.experts"));
    try std.testing.expect(!glob("*.mlp.experts", "x.layers.1.mlp.experts.0"));
    try std.testing.expect(glob("a?c", "abc"));
    try std.testing.expect(glob("*", ""));
    try std.testing.expect(!glob("abc", "ab"));
    try std.testing.expect(glob("m*l*", "model.language"));
}

test "the INT4-AutoRound checkpoint's config.json: GPTQ int4 g128 experts and head, top-5 routing" {
    var why: Why = .{};
    var c = try parse(std.testing.allocator, @embedFile("fixtures_cuda_config_int4ar.json"), "{\"eos_token_id\": [248046, 248044]}", .{}, &why);
    defer c.deinit();
    try std.testing.expectEqual(Quant.gptq, c.quant);
    try std.testing.expect(c.int4ar());
    try std.testing.expectEqual(@as(u32, 5), c.top_k);
    try std.testing.expectEqual(@as(u32, 512), c.experts);
    try std.testing.expectEqual(@as(u32, 640), c.moe_width);
    try std.testing.expectEqual(@as(u32, 1280), c.shared_width);
    const g = c.gptq.?;
    try std.testing.expectEqual(@as(u32, 4), g.bits);
    try std.testing.expectEqual(@as(u32, 128), g.group);
    try std.testing.expect(g.sym and !g.desc_act and g.lm_head);
    try std.testing.expectEqual(@as(usize, 11), g.dynamic.len);
    try std.testing.expect(g.quantized("model.language_model.layers.7.mlp.experts.300.down_proj"));
    try std.testing.expect(g.quantized("lm_head"));
    for ([_][]const u8{ "model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.3.self_attn.o_proj", "model.language_model.layers.3.mlp.shared_expert.up_proj", "model.language_model.layers.3.mlp.shared_expert_gate", "model.language_model.layers.3.mlp.gate", "model.language_model.layers.1.ple.key_proj", "model.language_model.embed_tokens", "model.language_model.layers.48.mlp.experts.0.gate_proj", "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down", "model.visual.blocks.0.attn.qkv", "mtp.fc_hidden" }) |m| try std.testing.expect(!g.quantized(m));
    try c.check(&why);
    try std.testing.expectEqualSlices(i64, &.{ 248044, 248046 }, c.stops());
    const s2 = try c.shares(2);
    try std.testing.expectEqual(@as(u32, 320), s2.moe_width);
    try std.testing.expectEqual(@as(u32, 640), s2.shared_width);
}

test "regexMatch: re.match of the GPTQ rules' subset" {
    try std.testing.expect(regexMatch(".*lm_head$", "lm_head"));
    try std.testing.expect(!regexMatch(".*lm_head$", "lm_head.x"));
    try std.testing.expect(regexMatch(".*\\.gate$", "a.mlp.gate"));
    try std.testing.expect(!regexMatch(".*\\.gate$", "a.mlp.shared_expert_gate"));
    try std.testing.expect(regexMatch(".*layers\\.48\\..*", "model.layers.48.mlp"));
    try std.testing.expect(!regexMatch(".*layers\\.48\\..*", "model.layers.4.mlp"));
    try std.testing.expect(regexMatch("abc", "abcdef"));
    try std.testing.expect(!regexMatch("abd", "abcdef"));
    try std.testing.expect(regexMatch("a+b", "aaab"));
    try std.testing.expect(!regexMatch("a+b", "b"));
}

test "GPTQ checks: zero layers, a head that is not 4-bit, 8-bit experts" {
    var why: Why = .{};
    const base = @embedFile("fixtures_cuda_config_int4ar.json");
    var c = try parse(std.testing.allocator, base, null, .{}, &why);
    defer c.deinit();
    var g = c.gptq.?;
    // the head unquantized
    g.lm_head = false;
    var rules: [16]Rule = undefined;
    var n: usize = 0;
    for (c.gptq.?.dynamic) |r| if (r.skip) {
        rules[n] = r;
        n += 1;
    };
    g.dynamic = rules[0..n];
    var c2 = c;
    c2.gptq = g;
    try std.testing.expectError(error.UnsupportedQuantization, c2.check(&why));
    // an 8-bit head rule
    var rules8: [16]Rule = undefined;
    for (c.gptq.?.dynamic, 0..) |r, i| rules8[i] = if (!r.skip) .{ .skip = false, .pattern = r.pattern, .bits = 8 } else r;
    g = c.gptq.?;
    g.dynamic = rules8[0..c.gptq.?.dynamic.len];
    c2.gptq = g;
    try std.testing.expectError(error.UnsupportedQuantization, c2.check(&why));
    // 8-bit base: experts refused
    g = c.gptq.?;
    g.bits = 8;
    c2.gptq = g;
    try std.testing.expectError(error.UnsupportedQuantization, c2.check(&why));
    // zero layers: refused, no underflow
    c2 = c;
    c2.layers = 0;
    try std.testing.expectError(error.UnsupportedModel, c2.check(&why));
}
