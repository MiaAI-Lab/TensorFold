//! tf-flashnext-weights: Flash Next's weights loaded by flashnext.weights (on the GPU, or hashed on the host) and every
//! buffer's sha256 against the Python oracle's weights.json (zig/tests/cuda/flashnext/capture.py weight_digests).
//!
//!   tf-flashnext-weights SNAPSHOT [WEIGHTS.json] [--mode device|hash|count] [--rank R --world W] [--threads N]
//!                        [--yarn FACTOR] [--gather TOKENS] [--out DIGESTS.json]
//!
//! device: loads onto GPU 0 and downloads each buffer to hash it; hash: no GPU, the bytes a load would upload; count:
//! sizes only. Prints the device bytes the rank holds; with WEIGHTS.json, PASS when every oracle tensor is loaded with
//! its digest and shape and nothing else is (our own buffers, oracle == false, are listed apart). --gather runs a
//! window of n-gram ids through the GPU table's gather (fn_pack.cu) and the host gather and compares the rows.
const std = @import("std");
const cuda = @import("cuda");
const flashnext = @import("flashnext");

const W = flashnext.weights;

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var snapshot: ?[]const u8 = null;
    var oracle: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var o: W.Options = .{ .mode = .hash };
    var yarn: ?f64 = null;
    var gather: u32 = 0;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--mode") and i + 1 < args.len) {
            i += 1;
            o.mode = std.meta.stringToEnum(W.Mode, args[i]) orelse return usage();
        } else if (std.mem.eql(u8, a, "--rank") and i + 1 < args.len) {
            i += 1;
            o.rank = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--world") and i + 1 < args.len) {
            i += 1;
            o.world = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--threads") and i + 1 < args.len) {
            i += 1;
            o.threads = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--yarn") and i + 1 < args.len) {
            i += 1;
            yarn = try std.fmt.parseFloat(f64, args[i]);
        } else if (std.mem.eql(u8, a, "--gather") and i + 1 < args.len) {
            i += 1;
            gather = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, a, "--out") and i + 1 < args.len) {
            i += 1;
            out_path = args[i];
        } else if (snapshot == null) {
            snapshot = a;
        } else if (oracle == null) {
            oracle = a;
        } else return usage();
    }
    const dir = snapshot orelse return usage();

    var c = try flashnext.config.Config.read(gpa, io, dir, .{ .yarn = if (yarn) |f| .{ .factor = f, .original_max_position_embeddings = 262144 } else null });
    defer c.deinit();

    var driver: cuda.Driver = undefined;
    var ctx: cuda.Context = undefined;
    if (o.mode == .device) {
        driver = try cuda.Driver.open();
        ctx = try cuda.Context.init(&driver, 0);
        o.driver = &driver;
    }
    defer if (o.mode == .device) {
        ctx.deinit();
        driver.close();
    };

    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    var w = try W.load(gpa, io, dir, &c, o);
    defer w.deinit();
    const load_s = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e9;
    std.debug.print("rank {d} of {d}: {d} buffers, {d} bytes ({d:.2} GiB) on the device, {d} tensors skipped, loaded in {d:.1} s ({t})\n", .{ o.rank, o.world, w.named.items.len, w.bytes, @as(f64, @floatFromInt(w.bytes)) / (1 << 30), w.skipped, load_s, o.mode });
    if (w.table) |t| std.debug.print("n-gram table: {d} rows x {d}, {s}, scale {e}\n", .{ t.rows, t.width, if (t.gpu != null) "rank's heads on the GPU" else "host-mapped", t.scale });
    if (o.mode == .count) return 0;
    if (gather > 0) if (w.table) |*t| if (t.gpu != null and o.mode == .device) {
        if (!try checkGather(gpa, &driver, &c, t, gather)) return 1;
    };

    const ds = try W.digests(gpa, &w);
    defer W.freeDigests(gpa, ds);

    if (out_path) |p| {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        try buf.appendSlice(gpa, "{\n");
        for (ds, 0..) |d, j| {
            const line = try std.fmt.allocPrint(gpa, "\"{s}\": \"{s}\",\n\"{s}.shape\": \"{s}\"{s}\n", .{ d.name, d.sha256, d.name, d.shape, if (j + 1 == ds.len) "" else "," });
            defer gpa.free(line);
            try buf.appendSlice(gpa, line);
        }
        try buf.appendSlice(gpa, "}\n");
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = buf.items });
    }

    const path = oracle orelse return 0;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const want = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer want.deinit();
    var same: usize = 0;
    var bad: usize = 0;
    var ours: usize = 0;
    for (ds) |d| {
        if (!d.oracle) {
            ours += 1;
            std.debug.print("OURS {s} {s} (no Python counterpart)\n", .{ d.name, d.shape });
            continue;
        }
        const e = want.value.object.get(d.name) orelse {
            std.debug.print("MISSING {s}: the oracle has no tensor of this name\n", .{d.name});
            bad += 1;
            continue;
        };
        var sk: [256]u8 = undefined;
        const shape_key = try std.fmt.bufPrint(&sk, "{s}.shape", .{d.name});
        const ws = if (want.value.object.get(shape_key)) |v| v.string else "";
        if (!std.mem.eql(u8, ws, d.shape)) {
            std.debug.print("SHAPE {s}: {s}, the oracle's {s}\n", .{ d.name, d.shape, ws });
            bad += 1;
        } else if (std.mem.eql(u8, e.string, &d.sha256)) {
            same += 1;
        } else {
            std.debug.print("DIFFER {s}: {d} bytes {s}\n", .{ d.name, d.len, d.shape });
            bad += 1;
        }
    }
    var names: usize = 0;
    var it = want.value.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.endsWith(u8, kv.key_ptr.*, ".shape")) continue;
        names += 1;
        var found = false;
        for (ds) |d| found = found or std.mem.eql(u8, d.name, kv.key_ptr.*);
        if (!found) std.debug.print("UNLOADED {s}: in the oracle, not loaded\n", .{kv.key_ptr.*});
    }
    const ok = bad == 0 and same == names;
    std.debug.print("{s} weights: {d} equal, {d} wrong, {d} loaded ({d} ours), {d} in the oracle\n", .{ if (ok) "PASS" else "FAIL", same, bad, ds.len, ours, names });
    return if (ok) 0 else 1;
}

/// `tokens` random tokens' n-gram ids through the GPU gather and the host gather: the rank's heads' rows equal.
fn checkGather(gpa: std.mem.Allocator, d: *const cuda.Driver, c: *const flashnext.config.Config, t: *const W.NgramTable, tokens: u32) !bool {
    const g = try flashnext.ngram.NGram.init(c.ngramOptions(0));
    const toks = try gpa.alloc(i64, tokens);
    defer gpa.free(toks);
    for (toks, 0..) |*x, j| x.* = @intCast(flashnext.layouts.rnd(99, j) % c.vocab);
    for (toks, 0..) |*x, j| if (j % 97 == 13) {
        x.* = g.eos;
    };
    var hist: [flashnext.ngram.max_n]i64 = undefined;
    g.initialHistory(&hist);
    const ids = try gpa.alloc(i64, tokens * g.heads);
    defer gpa.free(ids);
    try g.ids(hist[0 .. g.n - 1], toks, ids);
    const gpu = t.gpu.?;
    // host: the rank's heads' rows, token-major
    const mine = try gpa.alloc(i64, tokens * gpu.heads);
    defer gpa.free(mine);
    for (0..tokens) |r| for (0..gpu.heads) |h| {
        mine[r * gpu.heads + h] = ids[r * g.heads + gpu.head0 + h];
    };
    const want = try gpa.alloc(u16, mine.len * t.width);
    defer gpa.free(want);
    try t.gather(mine, want);
    var dids = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(ids));
    defer dids.free();
    var dout = try cuda.DeviceBuffer.alloc(d, want.len * 2);
    defer dout.free();
    var s = try cuda.Stream.init(d, true);
    defer s.deinit();
    var k = try W.Gather.init(d);
    defer k.deinit();
    try k.run(s, t, dids.ptr, g.heads, tokens, dout.ptr);
    try s.synchronize();
    const got = try gpa.alloc(u16, want.len);
    defer gpa.free(got);
    try dout.download(0, std.mem.sliceAsBytes(got));
    const same = std.mem.eql(u16, want, got);
    std.debug.print("{s} n-gram gather: {d} tokens x heads {d}..{d} on the GPU {s} the host gather\n", .{ if (same) "PASS" else "FAIL", tokens, gpu.head0, gpu.head0 + gpu.heads - 1, if (same) "equal to" else "differ from" });
    return same;
}

fn usage() u8 {
    std.debug.print("usage: tf-flashnext-weights SNAPSHOT [WEIGHTS.json] [--mode device|hash|count] [--rank R --world W] [--threads N] [--yarn FACTOR] [--gather TOKENS] [--out DIGESTS.json]\n", .{});
    return 2;
}
