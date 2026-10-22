//! Flash Next's rotary frequencies on the host: the default ones and YaRN past the native 262,144 tokens (Qwen's
//! recipe: rope_type yarn, factor 4, original_max_position_embeddings 262144 -> a 1,048,576-token window), with the
//! bits of the Python engine (qwen4_exp/cuda/weights.py's inv_freq, patched by qwen4_exp/cuda/yarn.py; work/YARN.md).
//!
//! The rotary dims are the first `rotary_dim` (64) of each head, rotate-half pairs (i, i + 32). Frequencies are
//! computed in fp64 and rounded to fp32 once. YaRN follows Transformers' `_compute_yarn_parameters`: pair i keeps the
//! native frequency theta^(-i/32) below the ramp, takes it over `factor` above it, and blends linearly between the
//! correction dims of beta_fast (32) and beta_slow (1) rotations in `original_max_position_embeddings` positions
//! (floor and ceil: pairs 14..22 at theta 1e7). The attention factor (0.1 ln(factor) + 1 = 1.13862943649292 in fp32
//! at factor 4) multiplies cos and sin of the rotary dims in Transformers and vLLM; this engine folds it, rounded to
//! fp32, into the fp32 gammas of the four RMSNorms whose outputs are rotated (q_norm, k_norm, indexer q_layernorm and
//! k_layernorm of every attention layer and the MTP layer): the normalized head is multiplied by its gamma before
//! the rotation, and the rotation is linear, so no kernel changes. Dims past `rotary_dim` keep their gammas.
const std = @import("std");

/// The most rotary pairs a head may have here (Flash Next: 32).
pub const max_half = 128;

pub const Error = error{ InvalidRope, InvalidYarn };

/// The ramp's first and last rotary pair (fractional when `truncate` is off).
pub const Bounds = struct { low: f64, high: f64 };

pub const Yarn = struct {
    factor: f64,
    /// original_max_position_embeddings: the trained window
    original: u64,
    beta_fast: f64 = 32,
    beta_slow: f64 = 1,
    /// the config's attention_factor (or mscale / mscale_all_dim ratio); 0 derives it from `factor`
    attention_factor: f64 = 0,
    truncate: bool = true,
    /// the positions the correction dims count rotations in; 0 is `original` (Transformers). vLLM's
    /// MRotaryEmbedding hands its YaRN code 4x `original`, which moves the ramp to pairs 16..24.
    ramp_positions: u64 = 0,

    /// The prompt-plus-reply window YaRN serves: factor x original.
    pub fn window(self: Yarn) u64 {
        return @intFromFloat(self.factor * @as(f64, @floatFromInt(self.original)));
    }

    /// The attention factor in fp64: the config's, else 0.1 ln(factor) + 1.
    pub fn attention(self: Yarn) f64 {
        if (self.attention_factor != 0) return self.attention_factor;
        return mscale(self.factor, 1);
    }

    pub fn bounds(self: Yarn, rotary_dim: u32, theta: f64) Bounds {
        const positions: f64 = @floatFromInt(if (self.ramp_positions != 0) self.ramp_positions else self.original);
        const d: f64 = @floatFromInt(rotary_dim);
        var low = correctionDim(self.beta_fast, d, theta, positions);
        var high = correctionDim(self.beta_slow, d, theta, positions);
        if (self.truncate) {
            low = @floor(low);
            high = @ceil(high);
        }
        return .{ .low = @max(low, 0), .high = @min(high, d - 1) };
    }
};

fn mscale(scale: f64, m: f64) f64 {
    return if (scale <= 1) 1.0 else 0.1 * m * @log(scale) + 1.0;
}

/// The rotary dim at which a frequency makes `rotations` turns in `positions` positions (Transformers'
/// find_correction_dim, same operation order).
fn correctionDim(rotations: f64, dim: f64, theta: f64, positions: f64) f64 {
    return dim * @log(positions / (rotations * 2 * std.math.pi)) / (2 * @log(theta));
}

pub const Rope = struct {
    theta: f64,
    rotary_dim: u32,
    yarn: ?Yarn = null,

    pub fn half(self: Rope) u32 {
        return self.rotary_dim / 2;
    }

    /// out[i] = the fp32 frequency of rotary pair i (out.len == rotary_dim / 2), computed in fp64 with the Python
    /// engine's operations: theta ** (-i / half); under YaRN (native / factor) * (1 - keep) + native * keep with
    /// keep = 1 - clamp((i - low) / (high - low), 0, 1).
    pub fn invFreq(self: Rope, out: []f32) Error!void {
        const h = self.half();
        if (h == 0 or h > max_half or self.rotary_dim % 2 != 0 or out.len != h or !(self.theta > 1)) return error.InvalidRope;
        const hf: f64 = @floatFromInt(h);
        var span: Bounds = .{ .low = 0, .high = 0 };
        if (self.yarn) |y| {
            if (!(y.factor >= 1) or y.original == 0) return error.InvalidYarn;
            span = y.bounds(self.rotary_dim, self.theta);
            if (span.low == span.high) span.high += 0.001;
        }
        for (out, 0..) |*o, i| {
            const fi: f64 = @floatFromInt(i);
            const native = std.math.pow(f64, self.theta, -fi / hf);
            if (self.yarn) |y| {
                const ramp = std.math.clamp((fi - span.low) / (span.high - span.low), 0.0, 1.0);
                const keep = 1 - ramp;
                o.* = @floatCast((native / y.factor) * (1 - keep) + native * keep);
            } else {
                o.* = @floatCast(native);
            }
        }
    }

    /// The attention factor rounded to fp32 (1 without YaRN): what foldGamma multiplies the rotated norms by.
    pub fn scale(self: Rope) f32 {
        return if (self.yarn) |y| @floatCast(y.attention()) else 1.0;
    }

    /// The window to admit: factor x original under YaRN (at least `native`), else `native`.
    pub fn window(self: Rope, native: u64) u64 {
        return if (self.yarn) |y| @max(native, y.window()) else native;
    }
};

/// gamma[0 .. rotary_dim] *= scale, one IEEE fp32 multiply each (the Python fold's bits); the rest is untouched.
/// Apply once per loaded gamma: q_norm, k_norm, indexer q_layernorm and k_layernorm (fp32, as the engine stores
/// them: 1 + w for centered checkpoints), in every attention layer and the MTP layer. A no-op at scale 1.
pub fn foldGamma(gamma: []f32, rotary_dim: u32, scale: f32) Error!void {
    if (gamma.len < rotary_dim) return error.InvalidRope;
    if (scale == 1.0) return;
    for (gamma[0..rotary_dim]) |*g| g.* = g.* * scale;
}

pub const env_factor = "TF_FLASHNEXT_YARN";
pub const env_ramp = "TF_FLASHNEXT_YARN_RAMP_POSITIONS";

/// The YaRN settings of a text config, as qwen4_exp/cuda/yarn.py's `read`: `rope` is text_config.rope_parameters
/// (empty when absent), `native` max_position_embeddings; `factor_env` is TF_FLASHNEXT_YARN (a factor above 1 forces
/// yarn with that factor; 0/off/no/false/none/default forces the default rope) and `ramp_env` is
/// TF_FLASHNEXT_YARN_RAMP_POSITIONS. Null: the default rope.
pub fn fromConfig(rope: std.json.ObjectMap, native: u64, factor_env: ?[]const u8, ramp_env: ?[]const u8) Error!?Yarn {
    var forced: ?f64 = null;
    if (factor_env) |raw| {
        const text = std.mem.trim(u8, raw, " \t\r\n");
        for ([_][]const u8{ "0", "off", "no", "false", "none", "default" }) |off| {
            if (std.ascii.eqlIgnoreCase(text, off)) return null;
        }
        if (text.len != 0) forced = std.fmt.parseFloat(f64, text) catch return error.InvalidYarn;
    }
    if (forced == null) {
        const kind = rope.get("rope_type") orelse rope.get("type") orelse return null;
        if (kind != .string or !std.mem.eql(u8, kind.string, "yarn")) return null;
    }
    const original: u64 = if (rope.get("original_max_position_embeddings")) |v| switch (v) {
        .integer => |x| if (x > 0) @intCast(x) else native,
        .null => native,
        else => return error.InvalidYarn,
    } else native;
    const factor = forced orelse if (rope.get("factor")) |v| (try number(v)) orelse ratio(native, original) else ratio(native, original);
    if (!(factor > 1) or original == 0) return error.InvalidYarn;
    var attention: f64 = (if (rope.get("attention_factor")) |v| try number(v) else null) orelse 0;
    if (attention == 0) {
        const ms = (if (rope.get("mscale")) |v| try number(v) else null) orelse 0;
        const all = (if (rope.get("mscale_all_dim")) |v| try number(v) else null) orelse 0;
        if (ms != 0 and all != 0) attention = mscale(factor, ms) / mscale(factor, all);
    }
    var ramp: u64 = 0;
    if (ramp_env) |raw| {
        const text = std.mem.trim(u8, raw, " \t\r\n");
        if (text.len != 0) ramp = std.fmt.parseInt(u64, text, 10) catch return error.InvalidYarn;
    }
    const truncate = if (rope.get("truncate")) |v| switch (v) {
        .bool => |b| b,
        .null => true,
        else => return error.InvalidYarn,
    } else true;
    return .{
        .factor = factor,
        .original = original,
        .beta_fast = nonzero(if (rope.get("beta_fast")) |v| try number(v) else null, 32),
        .beta_slow = nonzero(if (rope.get("beta_slow")) |v| try number(v) else null, 1),
        .attention_factor = attention,
        .truncate = truncate,
        .ramp_positions = ramp,
    };
}

/// Python's `value or fallback`: absent or 0 takes the fallback.
fn nonzero(v: ?f64, fallback: f64) f64 {
    const x = v orelse return fallback;
    return if (x == 0) fallback else x;
}

fn ratio(native: u64, original: u64) f64 {
    if (original == 0) return 0;
    return @as(f64, @floatFromInt(native)) / @as(f64, @floatFromInt(original));
}

/// A JSON number (null: absent).
fn number(v: std.json.Value) Error!?f64 {
    return switch (v) {
        .integer => |x| @floatFromInt(x),
        .float => |x| x,
        .null => null,
        else => error.InvalidYarn,
    };
}

// ---------------------------------------------------------------------------------------------------------------
// tests

const testing = std.testing;
const native_window = 262144;
const qwen: Rope = .{ .theta = 10_000_000, .rotary_dim = 64 };

fn bitsOf(v: []const f32, out: []u32) void {
    for (v, out) |x, *o| o.* = @bitCast(x);
}

// fp32 bits the Python engine stores (tensorfold-upstream:609ca41, torch 2.13 on aarch64 CPU, work/patches-python/
// check_yarn.py): the stock weights.py inv_freq and yarn.inv_freq at factor 4, 2, and 4 with vLLM's 4x ramp window.
const py_default = [32]u32{ 0x3f800000, 0x3f1ab32b, 0x3ebaf81a, 0x3e61f836, 0x3e088d77, 0x3da50957, 0x3d47763f, 0x3cf11176, 0x3c91ad39, 0x3c301052, 0x3bd4ca14, 0x3b80967d, 0x3b1b690d, 0x3abbd3ec, 0x3a6301e2, 0x3a092e02, 0x39a5cb5f, 0x394860c1, 0x38f22ce3, 0x3892587f, 0x3830df51, 0x37d5c442, 0x37812dac, 0x371c1fc4, 0x36bcb0c1, 0x36640cc6, 0x3609cf4b, 0x35a68e4c, 0x35494c56, 0x34f3499c, 0x3493048e, 0x3431af44 };
const py_yarn4 = [32]u32{ 0x3f800000, 0x3f1ab32b, 0x3ebaf81a, 0x3e61f836, 0x3e088d77, 0x3da50957, 0x3d47763f, 0x3cf11176, 0x3c91ad39, 0x3c301052, 0x3bd4ca14, 0x3b80967d, 0x3b1b690d, 0x3abbd3ec, 0x3a6301e2, 0x39f8a364, 0x3986b53d, 0x3910058b, 0x38975c0e, 0x381b7e06, 0x379ac367, 0x3712f6ed, 0x36812dac, 0x361c1fc4, 0x35bcb0c1, 0x35640cc6, 0x3509cf4b, 0x34a68e4c, 0x34494c56, 0x33f3499c, 0x3393048e, 0x3331af44 };
const py_yarn2 = [32]u32{ 0x3f800000, 0x3f1ab32b, 0x3ebaf81a, 0x3e61f836, 0x3e088d77, 0x3da50957, 0x3d47763f, 0x3cf11176, 0x3c91ad39, 0x3c301052, 0x3bd4ca14, 0x3b80967d, 0x3b1b690d, 0x3abbd3ec, 0x3a6301e2, 0x3a009b22, 0x399111f3, 0x3922ce9d, 0x38b5a1aa, 0x384939ae, 0x37dd1726, 0x37707cca, 0x37012dac, 0x369c1fc4, 0x363cb0c1, 0x35e40cc6, 0x3589cf4b, 0x35268e4c, 0x34c94c56, 0x3473499c, 0x3413048e, 0x33b1af44 };
const py_yarn4_vllm = [32]u32{ 0x3f800000, 0x3f1ab32b, 0x3ebaf81a, 0x3e61f836, 0x3e088d77, 0x3da50957, 0x3d47763f, 0x3cf11176, 0x3c91ad39, 0x3c301052, 0x3bd4ca14, 0x3b80967d, 0x3b1b690d, 0x3abbd3ec, 0x3a6301e2, 0x3a092e02, 0x39a5cb5f, 0x393597af, 0x38c4c478, 0x38525f36, 0x37dd1726, 0x37632086, 0x36e20fec, 0x3656abad, 0x35bcb0c1, 0x35640cc6, 0x3509cf4b, 0x34a68e4c, 0x34494c56, 0x33f3499c, 0x3393048e, 0x3331af44 };

fn expectBits(rope: Rope, want: [32]u32) !void {
    var f: [32]f32 = undefined;
    var got: [32]u32 = undefined;
    try rope.invFreq(&f);
    bitsOf(&f, &got);
    try testing.expectEqualSlices(u32, &want, &got);
}

test "default inv_freq is the Python engine's bits" {
    try expectBits(qwen, py_default);
}

test "YaRN inv_freq is the patched Python engine's bits" {
    var r = qwen;
    r.yarn = .{ .factor = 4, .original = native_window };
    try expectBits(r, py_yarn4);
    r.yarn = .{ .factor = 2, .original = native_window };
    try expectBits(r, py_yarn2);
    r.yarn = .{ .factor = 4, .original = native_window, .ramp_positions = 4 * native_window };
    try expectBits(r, py_yarn4_vllm);
}

/// Transformers' _compute_yarn_parameters written out independently in fp64: pos_freqs = theta^(2i/d),
/// interpolation 1/(factor pos), extrapolation 1/pos, mask = 1 - ramp, inter (1 - mask) + extra mask.
fn hfYarn64(theta: f64, d: f64, factor: f64, positions: f64, out: []f64) void {
    const lo = @max(@floor(d * @log(positions / (32.0 * 2.0 * std.math.pi)) / (2.0 * @log(theta))), 0);
    var hi = @min(@ceil(d * @log(positions / (1.0 * 2.0 * std.math.pi)) / (2.0 * @log(theta))), d - 1);
    if (lo == hi) hi += 0.001;
    for (out, 0..) |*o, i| {
        const fi: f64 = @floatFromInt(i);
        const pos = std.math.pow(f64, theta, 2.0 * fi / d);
        const mask = 1 - std.math.clamp((fi - lo) / (hi - lo), 0.0, 1.0);
        o.* = (1.0 / (factor * pos)) * (1 - mask) + (1.0 / pos) * mask;
    }
}

test "YaRN inv_freq equals Transformers' formula in fp64 within fp32 rounding" {
    for ([_]f64{ 2, 4, 8 }) |factor| {
        var r = qwen;
        r.yarn = .{ .factor = factor, .original = native_window };
        var f: [32]f32 = undefined;
        var ref: [32]f64 = undefined;
        try r.invFreq(&f);
        hfYarn64(r.theta, 64, factor, native_window, &ref);
        for (f, ref) |got, want| {
            const w32: f32 = @floatCast(want);
            const ulp: f64 = @floatCast(std.math.nextAfter(f32, @abs(w32), std.math.inf(f32)) - @abs(w32));
            try testing.expect(@abs(@as(f64, got) - want) <= 0.5 * ulp * (1 + 1e-6));
        }
    }
}

test "YaRN keeps the native pairs below the ramp and divides those above it" {
    var r = qwen;
    r.yarn = .{ .factor = 4, .original = native_window };
    const b = r.yarn.?.bounds(64, r.theta);
    try testing.expectEqual(@as(f64, 14), b.low);
    try testing.expectEqual(@as(f64, 22), b.high);
    var vllm = r.yarn.?;
    vllm.ramp_positions = 4 * native_window;
    try testing.expectEqual(Bounds{ .low = 16, .high = 24 }, vllm.bounds(64, r.theta));
    var plain: [32]f32 = undefined;
    var scaled: [32]f32 = undefined;
    try qwen.invFreq(&plain);
    try r.invFreq(&scaled);
    for (0..15) |i| try testing.expectEqual(plain[i], scaled[i]);
    for (22..32) |i| {
        const native = std.math.pow(f64, r.theta, -@as(f64, @floatFromInt(i)) / 32.0);
        try testing.expectEqual(@as(f32, @floatCast(native / 4)), scaled[i]);
    }
    for (15..22) |i| try testing.expect(scaled[i] < plain[i] and scaled[i] > plain[i] / 4);
}

test "attention factor, window and their defaults" {
    var r = qwen;
    try testing.expectEqual(@as(f32, 1), r.scale());
    try testing.expectEqual(@as(u64, native_window), r.window(native_window));
    r.yarn = .{ .factor = 4, .original = native_window };
    try testing.expectEqual(@as(u32, 0x3f91be9c), @as(u32, @bitCast(r.scale()))); // Python's fp32(0.1 ln 4 + 1)
    try testing.expectEqual(@as(u64, 1 << 20), r.window(native_window));
    r.yarn = .{ .factor = 2, .original = native_window };
    try testing.expectEqual(@as(u32, 0x3f88df4e), @as(u32, @bitCast(r.scale())));
    r.yarn = .{ .factor = 4, .original = native_window, .attention_factor = 1.25 };
    try testing.expectEqual(@as(f32, 1.25), r.scale());
    var bad: [31]f32 = undefined;
    try testing.expectError(error.InvalidRope, qwen.invFreq(&bad));
}

/// One rotate-half rotation of a normalized head in f64: out = x n g rotated by pos * inv (cos/sin times cs).
fn rotate(x: []const f64, g: []const f64, inv: []const f32, pos: f64, cs: f64, out: []f64) void {
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const rinv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + 1e-6);
    const h = inv.len;
    for (out, 0..) |*o, d| {
        const xn = x[d] * rinv * g[d];
        if (d >= 2 * h) {
            o.* = xn;
            continue;
        }
        const i = if (d < h) d else d - h;
        const p = if (d < h) d + h else d - h;
        const xpn = x[p] * rinv * g[p];
        const ang = pos * @as(f64, inv[i]);
        const c = @cos(ang) * cs;
        const s = @sin(ang) * cs;
        o.* = if (d < h) xn * c - xpn * s else xpn * s + xn * c;
    }
}

test "folding the attention factor into the gammas equals scaling cos and sin" {
    var prng = std.Random.DefaultPrng.init(0x7a29);
    const rand = prng.random();
    var r = qwen;
    r.yarn = .{ .factor = 4, .original = native_window };
    var inv: [32]f32 = undefined;
    try r.invFreq(&inv);
    const m = r.scale();
    // q/k heads are 256 wide, indexer heads 128: the first 64 dims rotate in both
    inline for (.{ 256, 128 }) |width| {
        for (0..20) |_| {
            var xq: [width]f64 = undefined;
            var xk: [width]f64 = undefined;
            var gq32: [width]f32 = undefined;
            var gk32: [width]f32 = undefined;
            for (&xq, &xk, &gq32, &gk32) |*a, *b, *c, *d| {
                a.* = rand.floatNorm(f64) * 3;
                b.* = rand.floatNorm(f64) * 3;
                c.* = 0.5 + rand.float(f32);
                d.* = 0.5 + rand.float(f32);
            }
            var fq32 = gq32;
            var fk32 = gk32;
            try foldGamma(&fq32, 64, m);
            try foldGamma(&fk32, 64, m);
            for (fq32[64..], gq32[64..]) |a, b| try testing.expectEqual(b, a); // non-rotary dims untouched
            for (fq32[0..64], gq32[0..64]) |a, b| try testing.expectEqual(b * m, a); // one fp32 multiply
            var gq: [width]f64 = undefined;
            var gk: [width]f64 = undefined;
            var fq: [width]f64 = undefined;
            var fk: [width]f64 = undefined;
            for (&gq, &gk, &fq, &fk, gq32, gk32, fq32, fk32) |*a, *b, *c, *d, e, f, g, h| {
                a.* = e;
                b.* = f;
                c.* = g;
                d.* = h;
            }
            const pq: f64 = @floatFromInt(rand.uintLessThan(u32, 1 << 20));
            const pk: f64 = @floatFromInt(rand.uintLessThan(u32, 1 << 20));
            var q_ref: [width]f64 = undefined;
            var q_fold: [width]f64 = undefined;
            var k_ref: [width]f64 = undefined;
            var k_fold: [width]f64 = undefined;
            rotate(&xq, &gq, &inv, pq, m, &q_ref); // Transformers / vLLM: cos and sin times m
            rotate(&xq, &fq, &inv, pq, 1, &q_fold); // this engine: m in the gamma
            rotate(&xk, &gk, &inv, pk, m, &k_ref);
            rotate(&xk, &fk, &inv, pk, 1, &k_fold);
            var top: f64 = 0;
            for (q_ref) |v| top = @max(top, @abs(v));
            for (q_ref, q_fold) |a, b| try testing.expect(@abs(a - b) <= top * 0x1p-22); // gamma*m's fp32 rounding
            for (q_ref[64..], q_fold[64..]) |a, b| try testing.expectEqual(a, b);
            // logits: m^2 on the rotary part, the rest unchanged; the indexer's relu(q . k) is positively
            // homogeneous, so its scores agree the same way
            var dot_ref: f64 = 0;
            var dot_fold: f64 = 0;
            var rot_ref: f64 = 0;
            var pass: f64 = 0;
            for (q_ref, k_ref, q_fold, k_fold, 0..) |a, b, c, d, i| {
                dot_ref += a * b;
                dot_fold += c * d;
                if (i < 64) rot_ref += a * b else pass += a * b;
            }
            const mag = 1e-5 * (@abs(rot_ref) + @abs(pass) + 1);
            try testing.expectApproxEqAbs(dot_ref, dot_fold, mag);
            try testing.expectApproxEqAbs(@max(dot_ref, 0), @max(dot_fold, 0), mag);
        }
    }
}

test "fold is a no-op without YaRN and refuses a short gamma" {
    var g = [_]f32{ 1.5, -0.25, 3 };
    try foldGamma(&g, 2, qwen.scale());
    try testing.expectEqualSlices(f32, &[_]f32{ 1.5, -0.25, 3 }, &g);
    try testing.expectError(error.InvalidRope, foldGamma(&g, 4, 1.1));
}

fn ropeObject(gpa: std.mem.Allocator, json: []const u8) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, gpa, json, .{});
}

test "config and TF_FLASHNEXT_YARN as the Python patch reads them" {
    const gpa = testing.allocator;
    const stock = try ropeObject(gpa,
        \\{"mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
        \\ "rope_theta": 10000000, "rope_type": "default"}
    );
    defer stock.deinit();
    const card = try ropeObject(gpa,
        \\{"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000,
        \\ "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}
    );
    defer card.deinit();
    try testing.expectEqual(@as(?Yarn, null), try fromConfig(stock.value.object, native_window, null, null));
    const want: Yarn = .{ .factor = 4, .original = native_window };
    try testing.expectEqual(@as(?Yarn, want), try fromConfig(card.value.object, native_window, null, null));
    try testing.expectEqual(@as(?Yarn, want), try fromConfig(stock.value.object, native_window, "4.0", null));
    try testing.expectEqual(@as(?Yarn, null), try fromConfig(card.value.object, native_window, "off", null));
    const two = (try fromConfig(card.value.object, native_window, " 2 ", "1048576")).?;
    try testing.expectEqual(@as(f64, 2), two.factor);
    try testing.expectEqual(@as(u64, 1048576), two.ramp_positions);
    try testing.expectEqual(@as(u64, 524288), two.window());
    try testing.expectError(error.InvalidYarn, fromConfig(stock.value.object, native_window, "0.5", null));
    try testing.expectError(error.InvalidYarn, fromConfig(stock.value.object, native_window, "four", null));
    const given = try ropeObject(gpa,
        \\{"rope_type": "yarn", "factor": 4, "original_max_position_embeddings": 262144, "beta_fast": 16,
        \\ "beta_slow": 2, "mscale": 1, "mscale_all_dim": 0.5, "truncate": false}
    );
    defer given.deinit();
    const g = (try fromConfig(given.value.object, native_window, null, null)).?;
    try testing.expectEqual(@as(f64, 16), g.beta_fast);
    try testing.expectEqual(@as(f64, 2), g.beta_slow);
    try testing.expect(!g.truncate);
    try testing.expectApproxEqRel((0.1 * @log(4.0) + 1) / (0.05 * @log(4.0) + 1), g.attention(), 1e-15);
}
