//! GLM-5.3-Flash's EXL3 weight-loading plan on the host: config.json plus each safetensors JSON header (never the
//! weight bytes) -> per-rank list of (tensor name, file, byte range, device buffer offset, dtype, shape, slice).
//!
//! Rank `rank` of `world` takes Python's rank slices (glm5_next/cuda/split.py RankReader._span, tp.Layout.cut,
//! weights.load's vocabulary rows of lm_head, dflash2.Drafter's KV-group split). Device buffers are packed in
//! checkpoint order with 256-byte alignment (cuda_layouts.align_bytes). A 4-bit trellis is stored int16 [..., 64]
//! and held as int32 [..., 32], the same bytes (exl3_mm.words). Vision tensors are dropped.
//!
//! Python source (TensorFold, https://github.com/ashhart/TensorFold): glm5_next/cuda/weights.py load, split.py
//! RankReader._span / rule / rank_cut, tp.Layout, dflash2.Drafter, cuda/exl3/format.py, exl3_mm.words.
const std = @import("std");
const core = @import("core");
const cfgs = @import("cuda_config.zig");
const lay = @import("cuda_layouts.zig");

const st = core.safetensors;
const Config = cfgs.Config;
const Layout = lay.Layout;
const Kind = lay.Kind;
const Family = lay.Family;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const max_shape = 8;
pub const device_align = lay.align_bytes;
pub const Source = enum { model, dflash2 };

pub const Options = struct {
    rank: u32 = 0,
    world: u32 = 1,
    mtp: bool = true,
    /// include DFlash2 tensors from this directory (its model.safetensors header)
    dflash2_dir: ?[]const u8 = null,
};

pub const TensorSrc = struct {
    name: []const u8,
    file: []const u8,
    dtype: []const u8,
    dtype_size: u32,
    rank: u8,
    shape: [max_shape]u32 = @splat(1),
    /// byte offsets from the start of the data region (header["data_offsets"])
    data_begin: u64,
    data_end: u64,
    /// 8 + JSON header length
    header_base: u64,
    source: Source = .model,
};

pub const Catalog = struct {
    gpa: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    tensors: std.ArrayList(TensorSrc) = .empty,

    pub fn deinit(c: *Catalog) void {
        c.tensors.deinit(c.gpa);
        c.arena.deinit();
        c.* = undefined;
    }
};

pub const Slice = struct {
    kind: Kind,
    family: ?Family = null,
    axis: ?u8 = null,
    start: u32 = 0,
    stop: u32 = 0,
    /// shape after the rank's slice
    shape: [max_shape]u32 = @splat(1),
    rank: u8 = 0,
    /// absolute file offsets of the bytes this rank reads (row splits trim; col/dim1 read the whole tensor)
    file_begin: u64 = 0,
    file_end: u64 = 0,
    device_dtype: []const u8,
    device_shape: [max_shape]u32 = @splat(1),
    device_rank: u8 = 0,
    device_bytes: u64 = 0,

    pub fn text(self: Slice, buf: []u8) []const u8 {
        if (self.kind == .rep or self.kind == .drop) return self.kind.text();
        return std.fmt.bufPrint(buf, "{s}:{d}:{d}", .{ self.kind.text(), self.start, self.stop }) catch self.kind.text();
    }
};

pub const Item = struct {
    src: TensorSrc,
    slice: Slice,
    device_offset: u64,
};

pub const Plan = struct {
    mtp: bool = false,
    gpa: std.mem.Allocator,
    rank: u32,
    world: u32,
    items: std.ArrayList(Item) = .empty,
    dropped: u32 = 0,
    file_bytes: u64 = 0,
    device_bytes: u64 = 0,
    sha256: [32]u8 = @splat(0),

    pub fn deinit(p: *Plan) void {
        p.items.deinit(p.gpa);
        p.* = undefined;
    }

    pub fn hex(self: Plan, buf: *[64]u8) []const u8 {
        const digits = "0123456789abcdef";
        for (self.sha256, 0..) |b, i| {
            buf[2 * i] = digits[b >> 4];
            buf[2 * i + 1] = digits[b & 0xF];
        }
        return buf;
    }
};

fn intern(a: std.mem.Allocator, s: []const u8) ![]const u8 {
    return a.dupe(u8, s);
}

/// Read only the safetensors JSON header (the first 8 + n bytes). Never maps the weight payload.
pub fn readHeader(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !struct { json: []u8, base: u64, file_len: u64 } {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const file_len: u64 = @intCast(try file.length(io));
    if (file_len < 8) return error.BadSafetensors;
    var head: [8]u8 = undefined;
    if (try file.readPositionalAll(io, &head, 0) != 8) return error.BadSafetensors;
    const n = std.mem.readInt(u64, &head, .little);
    if (n > 1 << 30 or 8 + n > file_len) return error.BadSafetensors;
    const json = try gpa.alloc(u8, n);
    errdefer gpa.free(json);
    if (try file.readPositionalAll(io, json, 8) != json.len) return error.BadSafetensors;
    return .{ .json = json, .base = 8 + n, .file_len = file_len };
}

fn appendHeaderTensors(cat: *Catalog, json: []const u8, file_name: []const u8, base: u64, file_len: u64, source: Source) !void {
    const a = cat.arena.allocator();
    const data_len = file_len - base;
    const header = try st.parseHeader(a, json, data_len);
    var it = header.iterator();
    while (it.next()) |kv| {
        try cat.tensors.append(cat.gpa, try srcOf(a, kv.key_ptr.*, file_name, kv.value_ptr.*, base, source));
    }
}

fn srcOf(a: std.mem.Allocator, name: []const u8, file_name: []const u8, e: st.Entry, base: u64, source: Source) !TensorSrc {
    var shape: [max_shape]u32 = @splat(1);
    const r = @min(e.rank, max_shape);
    var i: u8 = 0;
    while (i < r) : (i += 1) shape[i] = @intCast(e.shape[i]);
    const dt = switch (e.dtype) {
        .bool => "BOOL",
        .u8 => "U8",
        .i8 => "I8",
        .u16 => "U16",
        .i16 => "I16",
        .f16 => "F16",
        .bf16 => "BF16",
        .u32 => "U32",
        .i32 => "I32",
        .f32 => "F32",
        .u64 => "U64",
        .i64 => "I64",
        .f64 => "F64",
        .f8_e4m3 => "F8_E4M3",
        .f8_e5m2 => "F8_E5M2",
        .f8_e8m0 => "F8_E8M0",
    };
    return .{
        .name = try intern(a, name),
        .file = try intern(a, file_name),
        .dtype = dt,
        .dtype_size = @intCast(e.dtype.size()),
        .rank = r,
        .shape = shape,
        .data_begin = e.begin,
        .data_end = e.end,
        .header_base = base,
        .source = source,
    };
}

/// Enumerate the checkpoint's tensors from model.safetensors.index.json and each shard's JSON header.
pub fn loadCatalog(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, dflash2_dir: ?[]const u8) !Catalog {
    var cat: Catalog = .{ .gpa = gpa, .arena = std.heap.ArenaAllocator.init(gpa) };
    errdefer cat.deinit();
    const a = cat.arena.allocator();
    const index_path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(index_path);
    const index_text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, gpa, .limited(1 << 26));
    defer gpa.free(index_text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, index_text, .{});
    defer parsed.deinit();
    const weight_map = parsed.value.object.get("weight_map") orelse return error.BadIndex;
    if (weight_map != .object) return error.BadIndex;

    var files: std.StringArrayHashMapUnmanaged(void) = .empty;
    defer files.deinit(gpa);
    {
        var it = weight_map.object.iterator();
        while (it.next()) |kv| {
            const file = kv.value_ptr.string;
            const gop = try files.getOrPut(gpa, file);
            if (!gop.found_existing) gop.key_ptr.* = try intern(a, file);
        }
    }

    const ParsedFile = struct { json: []u8, base: u64, file_len: u64, header: st.Header };
    var headers: std.StringHashMapUnmanaged(ParsedFile) = .empty;
    defer {
        var hit = headers.iterator();
        while (hit.next()) |kv| gpa.free(kv.value_ptr.json);
        headers.deinit(gpa);
    }
    var fit = files.iterator();
    while (fit.next()) |e| {
        const path = try std.fs.path.join(gpa, &.{ dir, e.key_ptr.* });
        defer gpa.free(path);
        const h = try readHeader(gpa, io, path);
        const header = try st.parseHeader(a, h.json, h.file_len - h.base);
        try headers.put(gpa, e.key_ptr.*, .{ .json = h.json, .base = h.base, .file_len = h.file_len, .header = header });
    }

    var it = weight_map.object.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        const file = kv.value_ptr.string;
        const h = headers.get(file).?;
        const e = h.header.get(name) orelse return error.MissingTensor;
        try cat.tensors.append(gpa, try srcOf(a, name, file, e, h.base, .model));
    }

    if (dflash2_dir) |dd| {
        const path = try std.fs.path.join(gpa, &.{ dd, "model.safetensors" });
        defer gpa.free(path);
        const h = try readHeader(gpa, io, path);
        defer gpa.free(h.json);
        try appendHeaderTensors(&cat, h.json, "dflash2:model.safetensors", h.base, h.file_len, .dflash2);
    }
    return cat;
}

fn numel(shape: []const u32) u64 {
    var n: u64 = 1;
    for (shape) |d| n *= d;
    return n;
}

fn copyShape(rank: u8, src: [max_shape]u32) [max_shape]u32 {
    var o: [max_shape]u32 = @splat(1);
    var i: u8 = 0;
    while (i < rank) : (i += 1) o[i] = src[i];
    return o;
}

fn dflash2Slice(name: []const u8, src: TensorSrc, q0: u32, qn: u32, kv0: u32, kvn: u32, m0: u32, mn: u32, hd: u32) ?struct { kind: Kind, start: u32, stop: u32 } {
    if (std.mem.indexOf(u8, name, ".self_attn.q_proj.weight") != null) {
        return .{ .kind = .row, .start = q0 * hd, .stop = (q0 + qn) * hd };
    }
    if (std.mem.indexOf(u8, name, ".self_attn.k_proj.weight") != null or std.mem.indexOf(u8, name, ".self_attn.v_proj.weight") != null) {
        return .{ .kind = .row, .start = kv0 * hd, .stop = (kv0 + kvn) * hd };
    }
    if (std.mem.indexOf(u8, name, ".self_attn.o_proj.weight") != null) {
        return .{ .kind = .col, .start = q0 * hd, .stop = (q0 + qn) * hd };
    }
    if (std.mem.indexOf(u8, name, ".mlp.gate_proj.weight") != null or std.mem.indexOf(u8, name, ".mlp.up_proj.weight") != null) {
        return .{ .kind = .row, .start = m0, .stop = m0 + mn };
    }
    if (std.mem.indexOf(u8, name, ".mlp.down_proj.weight") != null) {
        return .{ .kind = .col, .start = m0, .stop = m0 + mn };
    }
    _ = src;
    return null;
}

fn applyCut(src: TensorSrc, kind: Kind, a: u32, b: u32, file_full: bool) Slice {
    var shape = copyShape(src.rank, src.shape);
    const axis = kind.axis() orelse 0;
    if (kind != .rep and kind != .drop) shape[axis] = b - a;
    const stored_bytes = numel(src.shape[0..src.rank]) * src.dtype_size;
    const sliced_n = numel(shape[0..src.rank]) * src.dtype_size;
    var file_begin = src.header_base + src.data_begin;
    var file_end = src.header_base + src.data_end;
    if (!file_full and kind == .row) {
        const per = stored_bytes / src.shape[0];
        file_begin = src.header_base + src.data_begin + @as(u64, a) * per;
        file_end = src.header_base + src.data_begin + @as(u64, b) * per;
    }
    var device_dtype = src.dtype;
    var device_shape = shape;
    var device_rank = src.rank;
    var device_bytes = sliced_n;
    if (std.mem.endsWith(u8, src.name, ".trellis")) {
        if (lay.trellisDeviceLast(shape[src.rank - 1], src.dtype_size)) |w| {
            device_dtype = "I32";
            device_shape[src.rank - 1] = @intCast(w.last);
            device_rank = src.rank;
            device_bytes = numel(device_shape[0..device_rank]) * w.item;
        }
    }
    return .{
        .kind = kind,
        .axis = kind.axis(),
        .start = a,
        .stop = b,
        .shape = shape,
        .rank = src.rank,
        .file_begin = file_begin,
        .file_end = file_end,
        .device_dtype = device_dtype,
        .device_shape = device_shape,
        .device_rank = device_rank,
        .device_bytes = device_bytes,
    };
}

fn skipMtp(src: TensorSrc, cfg: *const Config, mtp: bool) bool {
    if (mtp) return false;
    const i = lay.layerIndex(src.name) orelse return false;
    return i >= cfg.layers;
}

/// One rank's load plan: every kept tensor's file range and packed device offset.
pub fn plan(gpa: std.mem.Allocator, cfg: *const Config, cat: *const Catalog, o: Options) !Plan {
    if (o.world < 1 or o.rank >= o.world) return error.BadWorld;
    const layout = try Layout.fromConfig(cfg, o.rank, o.world);
    var q: [8]u32 = undefined;
    var kv: [8]u32 = undefined;
    var mlp: [8]u32 = undefined;
    var dflash = false;
    var q0: u32 = 0;
    var qn: u32 = 0;
    var kv0: u32 = 0;
    var kvn: u32 = 0;
    var m0: u32 = 0;
    var mn: u32 = 0;
    const hd: u32 = 128;
    for (cat.tensors.items) |t| if (t.source == .dflash2) {
        dflash = true;
        break;
    };
    if (dflash) {
        // DFlash2 config is not in cfg; query/KV/MLP shares come from the first q_proj / k_proj / gate_proj shapes.
        var heads: u32 = 32;
        var kv_heads: u32 = 8;
        var inter: u32 = 12288;
        var got_h = false;
        var got_k = false;
        var got_m = false;
        for (cat.tensors.items) |t| {
            if (t.source != .dflash2) continue;
            if (!got_h and std.mem.indexOf(u8, t.name, ".self_attn.q_proj.weight") != null) {
                heads = t.shape[0] / hd;
                got_h = true;
            }
            if (!got_k and std.mem.indexOf(u8, t.name, ".self_attn.k_proj.weight") != null) {
                kv_heads = t.shape[0] / hd;
                got_k = true;
            }
            if (!got_m and std.mem.indexOf(u8, t.name, ".mlp.gate_proj.weight") != null) {
                inter = t.shape[0];
                got_m = true;
            }
        }
        try lay.draftParts(heads, kv_heads, inter, o.world, &q, &kv, &mlp);
        const qs = lay.shareOf(q[0..o.world], o.rank);
        const kvs = lay.shareOf(kv[0..o.world], o.rank);
        const ms = lay.shareOf(mlp[0..o.world], o.rank);
        q0 = qs.start;
        qn = qs.stop - qs.start;
        kv0 = kvs.start;
        kvn = kvs.stop - kvs.start;
        m0 = ms.start;
        mn = ms.stop - ms.start;
    }

    var p: Plan = .{ .gpa = gpa, .rank = o.rank, .world = o.world, .mtp = o.mtp and cfg.mtp_layers != 0 };
    errdefer p.deinit();
    var offset: u64 = 0;
    var hasher = Sha256.init(.{});
    var line_buf: [1024]u8 = undefined;

    for (cat.tensors.items) |src| {
        if (src.source == .model) {
            const kind = lay.rule(src.name) catch {
                return error.AmbiguousSplit;
            };
            if (kind == .drop) {
                p.dropped += 1;
                continue;
            }
            if (skipMtp(src, cfg, o.mtp)) {
                p.dropped += 1;
                continue;
            }
            var sl: Slice = undefined;
            if (std.mem.startsWith(u8, src.name, "lm_head.")) {
                const cut = try layout.cut(.vocab, src.shape[0]);
                sl = applyCut(src, .row, cut.a, cut.b, false);
                sl.family = .vocab;
            } else if (kind == .rep) {
                sl = applyCut(src, .rep, 0, if (src.rank > 0) src.shape[0] else 1, true);
            } else {
                const fam = layout.familyOf(src.name) orelse return error.MissingFamily;
                const axis = kind.axis().?;
                const length = src.shape[axis];
                const cut = try layout.cut(fam, length);
                const file_full = kind != .row;
                sl = applyCut(src, kind, cut.a, cut.b, file_full);
                sl.family = fam;
            }
            try appendItem(&p, src, sl, &offset, &hasher, &line_buf);
        } else {
            const spec = dflash2Slice(src.name, src, q0, qn, kv0, kvn, m0, mn, hd);
            var sl: Slice = undefined;
            if (spec) |s| {
                sl = applyCut(src, s.kind, s.start, s.stop, s.kind != .row);
            } else {
                sl = applyCut(src, .rep, 0, if (src.rank > 0) src.shape[0] else 1, true);
            }
            try appendItem(&p, src, sl, &offset, &hasher, &line_buf);
        }
    }
    p.device_bytes = offset;
    hasher.final(&p.sha256);
    return p;
}

fn appendItem(p: *Plan, src: TensorSrc, sl: Slice, offset: *u64, hasher: *Sha256, line_buf: *[1024]u8) !void {
    const item = Item{ .src = src, .slice = sl, .device_offset = offset.* };
    try p.items.append(p.gpa, item);
    p.file_bytes += sl.file_end - sl.file_begin;
    const line = canonical(item, line_buf);
    hasher.update(line);
    offset.* = lay.alignUp(offset.* + sl.device_bytes);
}

fn writeShape(w: *std.Io.Writer, rank: u8, shape: [max_shape]u32) void {
    var i: u8 = 0;
    while (i < rank) : (i += 1) {
        if (i != 0) w.writeByte('x') catch {};
        w.print("{d}", .{shape[i]}) catch {};
    }
}

pub fn canonical(item: Item, buf: []u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    var slbuf: [64]u8 = undefined;
    w.print("{s}\t{s}\t{d}\t{d}\t{d}\t{s}\t", .{
        item.src.name,
        item.src.file,
        item.slice.file_begin,
        item.slice.file_end,
        item.device_offset,
        item.src.dtype,
    }) catch {};
    writeShape(&w, item.slice.rank, item.slice.shape);
    w.print("\t{s}\t{s}\t{s}\t", .{ item.slice.kind.text(), item.slice.text(&slbuf), item.slice.device_dtype }) catch {};
    writeShape(&w, item.slice.device_rank, item.slice.device_shape);
    w.writeByte('\n') catch {};
    return buf[0..w.end];
}

pub fn find(p: *const Plan, name: []const u8) ?Item {
    for (p.items.items) |it| if (std.mem.eql(u8, it.src.name, name)) return it;
    return null;
}

test "canonical line format is stable" {
    const src = TensorSrc{
        .name = "lm_head.weight",
        .file = "model-00001-of-00083.safetensors",
        .dtype = "BF16",
        .dtype_size = 2,
        .rank = 2,
        .shape = .{ 154880, 4096, 1, 1, 1, 1, 1, 1 },
        .data_begin = 0,
        .data_end = 10,
        .header_base = 128,
    };
    var sl = applyCut(src, .row, 0, 77440, false);
    sl.family = .vocab;
    const item = Item{ .src = src, .slice = sl, .device_offset = 0 };
    var buf: [512]u8 = undefined;
    const line = canonical(item, &buf);
    try std.testing.expect(std.mem.startsWith(u8, line, "lm_head.weight\tmodel-00001-of-00083.safetensors\t"));
    try std.testing.expect(std.mem.indexOf(u8, line, "\trow:0:77440\t") != null);
}

test "MTP checkpoint plan includes replicated EH/norms and TP-sliced DSA/MoE; auto-off drops layer 45" {
    const gpa = std.testing.allocator;
    var why: cfgs.Why = .{};
    var cfg = try cfgs.parse(gpa, @embedFile("fixtures_config.json"), @embedFile("fixtures_generation_config.json"), .{}, &why);
    defer cfg.deinit();
    var cat: Catalog = .{ .gpa = gpa, .arena = std.heap.ArenaAllocator.init(gpa) };
    defer cat.deinit();
    const names = [_][]const u8{ "model.language_model.layers.45.enorm.weight", "model.language_model.layers.45.eh_proj.weight", "model.language_model.layers.45.self_attn.q_b_proj.weight", "model.language_model.layers.45.mlp.shared_experts.down_proj.weight" };
    const shapes = [_][2]u32{ .{ 4096, 1 }, .{ 4096, 8192 }, .{ 16384, 1536 }, .{ 4096, 2048 } };
    for (names, shapes, 0..) |name, shape, i| {
        try cat.tensors.append(gpa, .{ .name = name, .file = "fixture.safetensors", .dtype = "BF16", .dtype_size = 2, .rank = if (i == 0) 1 else 2, .shape = .{ shape[0], shape[1], 1, 1, 1, 1, 1, 1 }, .data_begin = 0, .data_end = @as(u64, shape[0]) * shape[1] * 2, .header_base = 128 });
    }
    var off = try plan(gpa, &cfg, &cat, .{ .world = 2, .rank = 1, .mtp = false });
    defer off.deinit();
    try std.testing.expect(!off.mtp);
    try std.testing.expectEqual(@as(usize, 0), off.items.items.len);
    var on = try plan(gpa, &cfg, &cat, .{ .world = 2, .rank = 1, .mtp = true });
    defer on.deinit();
    try std.testing.expect(on.mtp);
    try std.testing.expectEqual(@as(usize, 4), on.items.items.len);
    try std.testing.expectEqual(@as(u32, 8192), find(&on, names[1]).?.slice.shape[1]);
    try std.testing.expectEqual(@as(u32, 8192), find(&on, names[2]).?.slice.shape[0]);
    try std.testing.expectEqual(@as(u32, 1024), find(&on, names[3]).?.slice.shape[1]);
}
