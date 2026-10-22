//! GLM-5.3-Flash dimensions for the CUDA engine: config.json (the root and its text_config) read into every field
//! the Python CUDA engine's Config.read takes (families/glm5_next/cuda/weights.py), plus the assistant-end token
//! from tokenizer.json (weights.assistant_end, issue #60) and generation_config.json's extra eos ids.
//!
//! Python source (TensorFold, https://github.com/ashhart/TensorFold): families/glm5_next/cuda/weights.py Config.read
//! and assistant_end by Ash Hart and TensorFold contributors.
const std = @import("std");

pub const model_types = [_][]const u8{ "glm5_next", "glm5" };
pub const max_layers = 64;
pub const max_eos = 8;

pub const LayerKind = enum { kda, dsa };
pub const MlpKind = enum { dense, moe };
pub const Quant = enum { mlx, exl3 };

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

pub const Options = struct {
    /// weights.assistant_end: the id of `<|assistant|>` when it ends a reply (null: look it up from tokenizer.json)
    assistant_end: ?i64 = null,
    /// TF_GLM_ASSISTANT_ENDS: 1 (default) appends assistant_end to eos; 0 leaves it out
    assistant_ends: bool = true,
};

pub const Config = struct {
    arena: std.heap.ArenaAllocator,
    model_type: []const u8,
    hidden: u32,
    layers: u32,
    vocab: u32,
    eps: f64,
    heads: u32,
    q_lora: u32,
    kv_lora: u32,
    qk_dim: u32,
    v_dim: u32,
    lin_heads: u32,
    lin_dim: u32,
    conv: u32,
    lower: f64,
    experts: u32,
    top_k: u32,
    moe_width: u32,
    shared_width: u32,
    dense_width: u32,
    routed_scale: f64,
    norm_topk: bool,
    streams: u32,
    hc_iters: u32,
    hc_eps: f64,
    index_heads: u32,
    index_dim: u32,
    index_topk: u32,
    kpool: u32,
    limit: f64,
    kinds: [max_layers]LayerKind = @splat(.kda),
    mlp_kinds: [max_layers]MlpKind = @splat(.dense),
    eos: [max_eos]i64 = @splat(0),
    eos_count: u32 = 0,
    mtp_layers: u32,
    group_size: u32,
    bits: u32,
    quant: Quant,
    codebook: []const u8,
    max_position: u64,

    pub fn deinit(self: *Config) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// `dir`/config.json, generation_config.json when present, and tokenizer.json's `<|assistant|>` id.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options, why: *Why) !Config {
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
        var options = o;
        if (options.assistant_end == null and options.assistant_ends) {
            options.assistant_end = assistantEnd(gpa, io, dir) catch |e| switch (e) {
                error.FileNotFound => null,
                else => return e,
            };
        }
        return parse(gpa, text, gen, options, why);
    }

    pub fn layerKind(self: Config, i: usize) LayerKind {
        return self.kinds[i];
    }

    pub fn mlpKind(self: Config, i: usize) MlpKind {
        return self.mlp_kinds[i];
    }

    pub fn stops(self: *const Config) []const i64 {
        return self.eos[0..self.eos_count];
    }

    pub fn isEos(self: Config, token: i64) bool {
        for (self.eos[0..self.eos_count]) |e| if (e == token) return true;
        return false;
    }

    /// Largest context (tokens) where DSA's top-k selection keeps every visible key (Config.dense_limit).
    pub fn denseLimit(self: Config) u32 {
        return self.index_topk + self.kpool - 1;
    }

    pub fn check(self: Config, why: *Why) !void {
        var known = false;
        for (model_types) |m| known = known or std.mem.eql(u8, m, self.model_type);
        if (!known) {
            why.set("model_type {s} is not GLM-5.3-Flash's (glm5_next)", .{self.model_type});
            return error.UnsupportedModel;
        }
        if (self.quant != .exl3 and self.quant != .mlx) {
            why.set("the CUDA GLM engine reads MLX 4-bit or EXL3 checkpoints, not {t}", .{self.quant});
            return error.UnsupportedQuantization;
        }
        if (self.quant == .exl3) {
            if (self.bits != 4) {
                why.set("EXL3 bits {d}; the engine reads 4-bit mcg routed experts", .{self.bits});
                return error.UnsupportedQuantization;
            }
            if (!std.ascii.eqlIgnoreCase(self.codebook, "mcg")) {
                why.set("EXL3 codebook {s}; the engine reads mcg", .{self.codebook});
                return error.UnsupportedQuantization;
            }
        }
        if (self.layers == 0 or self.layers > max_layers) {
            why.set("{d} layers; the engine holds up to {d}", .{ self.layers, max_layers });
            return error.UnsupportedModel;
        }
        if (self.heads == 0 or self.lin_heads == 0 or self.moe_width == 0) {
            why.set("heads/lin_heads/moe_width must be positive", .{});
            return error.UnsupportedModel;
        }
    }
};

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

fn ids(v: ?Value, out: *std.ArrayList(i64), a: std.mem.Allocator) !void {
    const x = v orelse return;
    switch (x) {
        .array => |arr| for (arr.items) |e| try out.append(a, try integer(e)),
        else => try out.append(a, try integer(x)),
    }
}

/// tokenizer.json's added_tokens: the id of special `<|assistant|>`, as weights.assistant_end.
pub fn assistantEndFromTokenizer(text: []const u8) ?i64 {
    const parsed = std.json.parseFromSlice(Value, std.heap.page_allocator, text, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const added = field(parsed.value.object, "added_tokens") orelse return null;
    if (added != .array) return null;
    for (added.array.items) |item| {
        if (item != .object) continue;
        const content = field(item.object, "content") orelse continue;
        const special = optBool(item.object, "special", false);
        if (content == .string and std.mem.eql(u8, content.string, "<|assistant|>") and special) {
            const id = field(item.object, "id") orelse continue;
            return integer(id) catch null;
        }
    }
    return null;
}

fn assistantEnd(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !?i64 {
    const path = try std.fs.path.join(gpa, &.{ dir, "tokenizer.json" });
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28)) catch |e| switch (e) {
        error.FileNotFound => return null,
        else => return e,
    };
    defer gpa.free(text);
    return assistantEndFromTokenizer(text);
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
    const want = field(raw, "tensorfold_activation_dtype");
    if (want) |v| if (v == .string) {
        if (std.mem.eql(u8, v.string, "float32")) {
            why.set("tensorfold_activation_dtype float32 is the Mac engine; the CUDA engine stays bf16", .{});
            return error.UnsupportedQuantization;
        }
        if (!std.mem.eql(u8, v.string, "bfloat16")) {
            why.set("tensorfold_activation_dtype {s}: bfloat16 or float32", .{v.string});
            return error.BadConfig;
        }
    };
    const t = object(raw, "text_config") orelse raw;
    const lin = object(t, "linear_attn_config");
    const model_type = try str(a, raw, "model_type");
    const n = try need(t, "num_hidden_layers", why);
    if (n == 0 or n > max_layers) {
        why.set("{d} layers; the engine holds up to {d}", .{ n, max_layers });
        return error.BadConfig;
    }

    const qk_nope = try need(t, "qk_nope_head_dim", why);
    const qk_rope = try opt(t, "qk_rope_head_dim", 0, why);
    const n_shared = try opt(t, "n_shared_experts", 1, why);
    const moe = try need(t, "moe_intermediate_size", why);

    var c: Config = .{
        .arena = undefined,
        .model_type = model_type,
        .hidden = try need(t, "hidden_size", why),
        .layers = n,
        .vocab = try need(t, "vocab_size", why),
        .eps = optFloat(t, "rms_norm_eps", -1) catch -1,
        .heads = try need(t, "num_attention_heads", why),
        .q_lora = try need(t, "q_lora_rank", why),
        .kv_lora = try need(t, "kv_lora_rank", why),
        .qk_dim = qk_nope + qk_rope,
        .v_dim = try need(t, "v_head_dim", why),
        .lin_heads = if (lin) |l| try opt(l, "num_heads", try opt(t, "linear_num_heads", 64, why), why) else try opt(t, "linear_num_heads", 64, why),
        .lin_dim = if (lin) |l| try opt(l, "head_dim", try opt(t, "linear_head_dim", 128, why), why) else try opt(t, "linear_head_dim", 128, why),
        .conv = if (lin) |l| try opt(l, "short_conv_kernel_size", try opt(t, "linear_conv_kernel_dim", 4, why), why) else try opt(t, "linear_conv_kernel_dim", 4, why),
        .lower = if (lin) |l| (optFloat(l, "gate_lower_bound", optFloat(t, "linear_lower_bound", -5.0) catch -5.0) catch -5.0) else (optFloat(t, "linear_lower_bound", -5.0) catch -5.0),
        .experts = try need(t, "n_routed_experts", why),
        .top_k = try need(t, "num_experts_per_tok", why),
        .moe_width = moe,
        .shared_width = moe * n_shared,
        .dense_width = try need(t, "intermediate_size", why),
        .routed_scale = optFloat(t, "routed_scaling_factor", 1) catch 1,
        .norm_topk = optBool(t, "norm_topk_prob", true),
        .streams = try opt(t, "hc_mult", 4, why),
        .hc_iters = try opt(t, "hc_sinkhorn_iters", 20, why),
        .hc_eps = optFloat(t, "hc_eps", 1e-6) catch 1e-6,
        .index_heads = try opt(t, "index_n_heads", 32, why),
        .index_dim = try opt(t, "index_head_dim", 128, why),
        .index_topk = try opt(t, "index_topk", 2048, why),
        .kpool = try opt(t, "index_kpool", 4, why),
        .limit = optFloat(t, "swiglu_limit", 10.0) catch 10.0,
        .mtp_layers = try opt(t, "num_nextn_predict_layers", 0, why),
        .group_size = 64,
        .bits = 4,
        .quant = .mlx,
        .codebook = "mcg",
        .max_position = try optWide(t, "max_position_embeddings", 0),
    };
    if (c.eps <= 0) {
        why.set("config.json has no rms_norm_eps", .{});
        return error.BadConfig;
    }

    const lt = field(t, "layer_types") orelse {
        why.set("config.json has no layer_types", .{});
        return error.BadConfig;
    };
    if (lt != .array or lt.array.items.len != c.layers) {
        why.set("layer_types lists {d} layers for num_hidden_layers {d}", .{ if (lt == .array) lt.array.items.len else 0, c.layers });
        return error.BadConfig;
    }
    for (lt.array.items, 0..) |v, i| {
        c.kinds[i] = if (v == .string and std.mem.eql(u8, v.string, "linear_attention")) .kda else .dsa;
    }

    const dense_n = try opt(t, "first_k_dense_replace", 3, why);
    if (field(t, "mlp_layer_types")) |mv| {
        if (mv != .array or mv.array.items.len != c.layers) {
            why.set("mlp_layer_types lists {d} layers for num_hidden_layers {d}", .{ if (mv == .array) mv.array.items.len else 0, c.layers });
            return error.BadConfig;
        }
        for (mv.array.items, 0..) |v, i| {
            c.mlp_kinds[i] = if (v == .string and std.mem.eql(u8, v.string, "sparse")) .moe else .dense;
        }
    } else {
        for (0..c.layers) |i| c.mlp_kinds[i] = if (i < dense_n) .dense else .moe;
    }

    var stop: std.ArrayList(i64) = .empty;
    const teos = field(t, "eos_token_id");
    ids(teos orelse field(raw, "eos_token_id"), &stop, a) catch {
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
    if (o.assistant_ends) if (o.assistant_end) |end| {
        if (std.mem.indexOfScalar(i64, stop.items, end) == null) try stop.append(a, end);
    };
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

    const q = object(raw, "quantization") orelse object(raw, "quantization_config");
    if (q) |qq| {
        const method = blk: {
            const v = field(qq, "quant_method") orelse break :blk "";
            break :blk if (v == .string) v.string else "";
        };
        if (method.len == 0 or std.ascii.eqlIgnoreCase(method, "mlx")) {
            c.quant = .mlx;
        } else if (std.ascii.eqlIgnoreCase(method, "exl3")) {
            c.quant = .exl3;
        } else {
            why.set("quant_method {s}; GLM reads MLX or EXL3 checkpoints", .{method});
            return error.UnsupportedQuantization;
        }
        c.group_size = try opt(qq, "group_size", 64, why);
        c.bits = try opt(qq, "bits", 4, why);
        const cb = try str(a, qq, "codebook");
        if (cb.len > 0) c.codebook = cb;
    }

    c.arena = arena;
    return c;
}

const fixture = @embedFile("fixtures_config.json");
const fixture_generation = @embedFile("fixtures_generation_config.json");
const fixture_tokenizer = @embedFile("fixtures_tokenizer.json");

test "the EXL3 checkpoint's config.json reads as Python's Config.read" {
    var why: Why = .{};
    const end = assistantEndFromTokenizer(fixture_tokenizer);
    try std.testing.expectEqual(@as(?i64, 154828), end);
    var c = try parse(std.testing.allocator, fixture, fixture_generation, .{ .assistant_end = end }, &why);
    defer c.deinit();
    try std.testing.expectEqualStrings("glm5_next", c.model_type);
    try std.testing.expectEqual(@as(u32, 4096), c.hidden);
    try std.testing.expectEqual(@as(u32, 45), c.layers);
    try std.testing.expectEqual(@as(u32, 154880), c.vocab);
    try std.testing.expectEqual(@as(f64, 1e-5), c.eps);
    try std.testing.expectEqual(@as(u32, 64), c.heads);
    try std.testing.expectEqual(@as(u32, 1536), c.q_lora);
    try std.testing.expectEqual(@as(u32, 512), c.kv_lora);
    try std.testing.expectEqual(@as(u32, 256), c.qk_dim);
    try std.testing.expectEqual(@as(u32, 256), c.v_dim);
    try std.testing.expectEqual(@as(u32, 64), c.lin_heads);
    try std.testing.expectEqual(@as(u32, 128), c.lin_dim);
    try std.testing.expectEqual(@as(u32, 4), c.conv);
    try std.testing.expectEqual(@as(f64, -5.0), c.lower);
    try std.testing.expectEqual(@as(u32, 288), c.experts);
    try std.testing.expectEqual(@as(u32, 8), c.top_k);
    try std.testing.expectEqual(@as(u32, 2048), c.moe_width);
    try std.testing.expectEqual(@as(u32, 2048), c.shared_width);
    try std.testing.expectEqual(@as(u32, 12288), c.dense_width);
    try std.testing.expectEqual(@as(f64, 2.5), c.routed_scale);
    try std.testing.expect(c.norm_topk);
    try std.testing.expectEqual(@as(u32, 4), c.streams);
    try std.testing.expectEqual(@as(u32, 20), c.hc_iters);
    try std.testing.expectEqual(@as(u32, 32), c.index_heads);
    try std.testing.expectEqual(@as(u32, 128), c.index_dim);
    try std.testing.expectEqual(@as(u32, 2048), c.index_topk);
    try std.testing.expectEqual(@as(u32, 4), c.kpool);
    try std.testing.expectEqual(@as(u32, 2051), c.denseLimit());
    try std.testing.expectEqual(@as(u32, 1), c.mtp_layers);
    try std.testing.expectEqual(Quant.exl3, c.quant);
    try std.testing.expectEqual(@as(u32, 4), c.bits);
    try std.testing.expectEqualStrings("mcg", c.codebook);
    try std.testing.expectEqual(LayerKind.kda, c.kinds[0]);
    try std.testing.expectEqual(LayerKind.dsa, c.kinds[3]);
    try std.testing.expectEqual(MlpKind.dense, c.mlp_kinds[0]);
    try std.testing.expectEqual(MlpKind.moe, c.mlp_kinds[3]);
    try std.testing.expectEqualSlices(i64, &.{ 154820, 154827, 154829, 154828 }, c.stops());
    try c.check(&why);
}

test "assistant_ends 0 leaves the role token out of eos" {
    var why: Why = .{};
    var c = try parse(std.testing.allocator, fixture, fixture_generation, .{ .assistant_end = 154828, .assistant_ends = false }, &why);
    defer c.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 154820, 154827, 154829 }, c.stops());
}
