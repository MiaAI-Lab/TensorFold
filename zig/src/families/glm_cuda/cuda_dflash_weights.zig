//! TensorFold glm5_next/cuda/dflash2.py:Drafter.__init__, using cuda_weights.Plan slices.
//! Ash Hart and TensorFold contributors; jschmied CUDA patterns. IncoAI weights CC BY-NC-ND.
const std = @import("std");
const cuda = @import("cuda");
const wp = @import("cuda_weights.zig");
const quant = @import("cuda_quantize.zig");
const mm = @import("cuda_matmuls.zig");
const dc = @import("cuda_dflash_config.zig");
pub const Layer = struct {
    in_norm: u64,
    post_norm: u64,
    a_base: u64,
    m_base: u64,
    q_norm: u64,
    k_norm: u64,
    a_kp: mm.Q4,
    m_kp: mm.Q4,
    qkv: mm.Q4,
    kv: mm.Q4,
    o: mm.Q4,
    gu: mm.Q4,
    down: mm.Q4,
};
/// Per-rank sliced tensor bytes decoded to BF16. Column slices read full rows, as RankReader does.
pub fn readTensor(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, draft_dir: []const u8, item: wp.Item) ![]u16 {
    const name = if (item.src.source == .dflash2) "model.safetensors" else item.src.file;
    const path = try std.fs.path.join(gpa, &.{ if (item.src.source == .dflash2) draft_dir else model_dir, name });
    defer gpa.free(path);
    var f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const bytes = try gpa.alloc(u8, @intCast(item.slice.file_end - item.slice.file_begin));
    defer gpa.free(bytes);
    if (try f.readPositionalAll(io, bytes, item.slice.file_begin) != bytes.len) return error.ShortDraftWeights;
    return decode(gpa, item, bytes);
}
fn decode(gpa: std.mem.Allocator, item: wp.Item, bytes: []const u8) ![]u16 {
    var count: usize = 1;
    for (item.slice.shape[0..item.slice.rank]) |n| count = try std.math.mul(usize, count, n);
    const size = item.src.dtype_size;
    if (size == 0 or bytes.len % size != 0) return error.BadDraftWeights;
    const out = try gpa.alloc(u16, count);
    errdefer gpa.free(out);
    var inner: usize = 1;
    if (item.slice.axis) |axis| for (item.src.shape[axis + 1 .. item.src.rank]) |n| {
        inner *= n;
    };
    for (out, 0..) |*v, i| {
        const index = if (item.slice.axis != null and item.slice.kind != .row) blk: {
            const axis = item.slice.axis.?;
            const width = (item.slice.stop - item.slice.start) * inner;
            if (width == 0) return error.BadDraftWeights;
            break :blk i / width * item.src.shape[axis] * inner + item.slice.start * inner + i % width;
        } else i;
        if (index >= bytes.len / size) return error.BadDraftWeights;
        const b = bytes[index * size ..][0..size];
        if (std.mem.eql(u8, item.src.dtype, "BF16")) v.* = std.mem.readInt(u16, b[0..2], .little) else if (std.mem.eql(u8, item.src.dtype, "F32")) v.* = quant.bf16Rne(@bitCast(std.mem.readInt(u32, b[0..4], .little))) else if (std.mem.eql(u8, item.src.dtype, "F16")) v.* = quant.bf16Rne(@as(f32, @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, b[0..2], .little)))))) else return error.UnsupportedDraftDtype;
    }
    return out;
}
/// Selector codebooks retain source FP32 precision; converting them through BF16 loses quality.
pub fn readCodebook(gpa: std.mem.Allocator, io: std.Io, draft_dir: []const u8, item: wp.Item) ![]f32 {
    if (item.slice.kind != .rep or item.src.rank != 2) return error.BadSelector;
    const path = try std.fs.path.join(gpa, &.{ draft_dir, "model.safetensors" });
    defer gpa.free(path);
    var f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const bytes = try gpa.alloc(u8, @intCast(item.slice.file_end - item.slice.file_begin));
    defer gpa.free(bytes);
    if (try f.readPositionalAll(io, bytes, item.slice.file_begin) != bytes.len) return error.ShortDraftWeights;
    const n = @as(usize, item.src.shape[0]) * item.src.shape[1];
    const out = try gpa.alloc(f32, n);
    errdefer gpa.free(out);
    for (out, 0..) |*v, i| {
        const b = bytes[i * item.src.dtype_size ..][0..item.src.dtype_size];
        if (std.mem.eql(u8, item.src.dtype, "F32")) v.* = @bitCast(std.mem.readInt(u32, b[0..4], .little)) else if (std.mem.eql(u8, item.src.dtype, "BF16")) v.* = quant.bf16ToF32(std.mem.readInt(u16, b[0..2], .little)) else return error.UnsupportedDraftDtype;
    }
    return out;
}
pub const Loader = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: *const cuda.Driver,
    model_dir: []const u8,
    draft_dir: []const u8,
    plan: *const wp.Plan,
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
    pub fn deinit(l: *Loader) void {
        for (l.owned.items) |*b| b.free();
        l.owned.deinit(l.gpa);
    }
    pub fn item(l: *const Loader, source: wp.Source, name: []const u8) !wp.Item {
        for (l.plan.items.items) |i| if (i.src.source == source and std.mem.eql(u8, i.src.name, name)) return i;
        return error.MissingDraftWeight;
    }
    fn tensor(l: *Loader, source: wp.Source, name: []const u8, n: usize, k: usize) ![]u16 {
        const it = try l.item(source, name);
        const got = try readTensor(l.gpa, l.io, l.model_dir, l.draft_dir, it);
        errdefer l.gpa.free(got);
        if (got.len != n * k or it.slice.shape[0] != n or (it.slice.rank >= 2 and it.slice.shape[1] != k)) return error.BadDraftShape;
        return got;
    }
    pub fn upload(l: *Loader, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.fromHost(l.driver, bytes);
        errdefer b.free();
        try l.owned.append(l.gpa, b);
        return b.ptr;
    }
    fn raw(l: *Loader, name: []const u8, n: usize, k: usize) !u64 {
        const t = try l.tensor(.dflash2, name, n, k);
        defer l.gpa.free(t);
        return l.upload(std.mem.sliceAsBytes(t));
    }
    pub fn q4(l: *Loader, source: wp.Source, names: []const []const u8, ns: []const usize, k: usize) !mm.Q4 {
        if (names.len != ns.len or k % 64 != 0) return error.BadDraftShape;
        var n: usize = 0;
        for (ns) |rows| n += rows;
        const rows = try l.gpa.alloc(u16, n * k);
        defer l.gpa.free(rows);
        var at: usize = 0;
        for (names, ns) |name, width| {
            const t = try l.tensor(source, name, width, k);
            defer l.gpa.free(t);
            @memcpy(rows[at..][0..t.len], t);
            at += t.len;
        }
        const words = try l.gpa.alloc(u32, n * k / 8);
        defer l.gpa.free(words);
        const scales = try l.gpa.alloc(u16, n * k / 64);
        defer l.gpa.free(scales);
        const biases = try l.gpa.alloc(u16, scales.len);
        defer l.gpa.free(biases);
        quant.quantize4(rows, n, k, words, scales, biases, false);
        const pad = std.mem.alignForward(usize, n, 128);
        const pack_buf = try l.gpa.alloc(u32, pad * k / 8);
        defer l.gpa.free(pack_buf);
        const ps = try l.gpa.alloc(u16, pad * k / 64);
        defer l.gpa.free(ps);
        const pb = try l.gpa.alloc(u16, ps.len);
        defer l.gpa.free(pb);
        quant.pack(words, scales, biases, n, k, pack_buf, ps, pb);
        return .{ .n = n, .k = k, .w = try l.upload(std.mem.sliceAsBytes(pack_buf)), .s = try l.upload(std.mem.sliceAsBytes(ps)), .b = try l.upload(std.mem.sliceAsBytes(pb)) };
    }
    fn one(l: *Loader, name: []const u8, n: usize, k: usize) !mm.Q4 {
        return l.q4(.dflash2, &.{name}, &.{n}, k);
    }
    pub fn load(l: *Loader, rank: dc.Rank) !Weights {
        if (l.plan.rank != rank.rank or l.plan.world != rank.world) return error.BadDraftRank;
        var w: Weights = undefined;
        w.fc = try l.one("fc.weight", 4096, 5 * 4096);
        w.hidden_norm = try l.raw("hidden_norm.weight", 4096, 1);
        w.norm = try l.raw("norm.weight", 4096, 1);
        w.hproj = try l.raw("candidate_selector.hidden_projection.weight", 256, 4096);
        w.head = try l.q4(.model, &.{"lm_head.weight"}, &.{rank.vocab}, 4096);
        var buf: [512]u8 = undefined;
        for (&w.layers, 0..) |*layer, i| {
            const prefix = try std.fmt.allocPrint(l.gpa, "layers.{d}.", .{i});
            defer l.gpa.free(prefix);
            inline for (.{ "in_norm", "post_norm", "a_base", "m_base", "q_norm", "k_norm" }, .{ "input_layernorm.weight", "post_attention_layernorm.weight", "attention_conv.base_kernel", "mlp_conv.base_kernel", "self_attn.q_norm.weight", "self_attn.k_norm.weight" }, .{ 4096, 4096, 4, 4, 128, 128 }, .{ 1, 1, 4096, 4096, 1, 1 }) |field, suffix, n, k| {
                const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
                // base kernels have [2,2,D], validated separately and flattened for the replay kernel.
                if (n == 4) {
                    const it = try l.item(.dflash2, name);
                    if (it.slice.rank != 3 or it.slice.shape[0] != 2 or it.slice.shape[1] != 2 or it.slice.shape[2] != 4096) return error.BadDraftShape;
                    const t = try readTensor(l.gpa, l.io, l.model_dir, l.draft_dir, it);
                    defer l.gpa.free(t);
                    @field(layer, field) = try l.upload(std.mem.sliceAsBytes(t));
                } else @field(layer, field) = try l.raw(name, n, k);
            }
            inline for (.{ "a_kp", "m_kp", "o", "down" }, .{ "attention_conv.kernel_projection.weight", "mlp_conv.kernel_projection.weight", "self_attn.o_proj.weight", "mlp.down_proj.weight" }, .{ 1024, 1024, 4096, 4096 }, .{ @as(usize, 4096), 4096, rank.heads * 128, rank.inter }) |field, suffix, n, k| {
                @field(layer, field) = try l.one(try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix }), n, k);
            }
            const q = try std.fmt.allocPrint(l.gpa, "{s}self_attn.q_proj.weight", .{prefix});
            defer l.gpa.free(q);
            const k = try std.fmt.allocPrint(l.gpa, "{s}self_attn.k_proj.weight", .{prefix});
            defer l.gpa.free(k);
            const v = try std.fmt.allocPrint(l.gpa, "{s}self_attn.v_proj.weight", .{prefix});
            defer l.gpa.free(v);
            layer.qkv = try l.q4(.dflash2, &.{ q, k, v }, &.{ rank.heads * 128, rank.kv * 128, rank.kv * 128 }, 4096);
            layer.kv = try l.q4(.dflash2, &.{ k, v }, &.{ rank.kv * 128, rank.kv * 128 }, 4096);
            const gate = try std.fmt.allocPrint(l.gpa, "{s}mlp.gate_proj.weight", .{prefix});
            defer l.gpa.free(gate);
            const up = try std.fmt.allocPrint(l.gpa, "{s}mlp.up_proj.weight", .{prefix});
            defer l.gpa.free(up);
            layer.gu = try l.q4(.dflash2, &.{ gate, up }, &.{ rank.inter, rank.inter }, 4096);
        }
        return w;
    }
};
pub const Weights = struct { fc: mm.Q4, head: mm.Q4, hidden_norm: u64, norm: u64, hproj: u64, layers: [5]Layer };
test "column slices preserve row order and BF16 bits" {
    const item_: wp.Item = .{ .src = .{ .name = "o", .file = "unused", .dtype = "BF16", .dtype_size = 2, .rank = 2, .shape = .{ 2, 4, 1, 1, 1, 1, 1, 1 }, .data_begin = 0, .data_end = 16, .header_base = 0 }, .slice = .{ .kind = .col, .axis = 1, .start = 1, .stop = 3, .rank = 2, .shape = .{ 2, 2, 1, 1, 1, 1, 1, 1 }, .device_dtype = "BF16" }, .device_offset = 0 };
    const bytes = [_]u8{ 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0 };
    const got = try decode(std.testing.allocator, item_, &bytes);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualSlices(u16, &.{ 2, 3, 6, 7 }, got);
}
