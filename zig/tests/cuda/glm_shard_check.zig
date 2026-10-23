//! tf-glm-shard-check (GPU): glm_shard.cu's kernels, through cuda_shard_decode's and cuda_shard_prompt's launches,
//! against their host rules (which tools/glm_shard_emulate also holds the kernels to): owner decode's counts and select
//! over random windows of 1 to 64 rows (countsRef / selectRef), and a prompt chunk's history fetch over random chunks
//! (cuda_shard.markHistory, prefixSum and renumber, cuda_shard_prompt.compactRef).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const dec = glm.shard_decode;
const sp = glm.shard_prompt;
const shard = glm.shard;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&d, true);
    defer stream.deinit();
    var k = try dec.Kernels.load(&d, cuda.kernels.glm_shard);
    defer k.unload();
    const row = dec.row_bytes;
    var counts = try cuda.DeviceBuffer.alloc(&d, 64 * 4);
    defer counts.free();
    var own = try cuda.DeviceBuffer.alloc(&d, 64 * 4);
    defer own.free();
    var ce = try cuda.DeviceBuffer.alloc(&d, 64 * 4);
    defer ce.free();
    var recv = try cuda.DeviceBuffer.alloc(&d, 64 * row);
    defer recv.free();
    var out = try cuda.DeviceBuffer.alloc(&d, 64 * row);
    defer out.free();
    const h_recv = try gpa.alloc(u8, 64 * row);
    defer gpa.free(h_recv);
    const h_out = try gpa.alloc(u8, 64 * row);
    defer gpa.free(h_out);
    const want = try gpa.alloc(u8, 64 * row);
    defer gpa.free(want);
    const got = try gpa.alloc(u8, 64 * row);
    defer gpa.free(got);
    var prng = std.Random.DefaultPrng.init(118);
    const rnd = prng.random();
    for (0..200) |trial| {
        const rows = if (trial < 64) trial + 1 else rnd.intRangeAtMost(usize, 1, 64);
        var c: [64]i32 = undefined;
        var o: [64]i32 = undefined;
        for (&c, &o) |*x, *y| {
            x.* = rnd.intRangeAtMost(i32, 0, 512);
            y.* = rnd.intRangeAtMost(i32, 0, 1);
        }
        var w: [64]i32 = @splat(-7);
        try counts.upload(0, std.mem.sliceAsBytes(&c));
        try own.upload(0, std.mem.sliceAsBytes(&o));
        try ce.upload(0, std.mem.sliceAsBytes(&w));
        dec.countsRef(c[0..rows], o[0..rows], w[0..rows]);
        rnd.bytes(h_recv);
        rnd.bytes(h_out);
        try recv.upload(0, h_recv);
        try out.upload(0, h_out);
        @memcpy(want, h_out);
        dec.selectRef(h_recv[0 .. rows * row], want[0 .. rows * row], o[0..rows], row);
        try k.counts(stream, counts.ptr, own.ptr, ce.ptr, rows);
        try k.select(stream, recv.ptr, out.ptr, own.ptr, rows, row);
        try stream.synchronize();
        var g: [64]i32 = undefined;
        try ce.download(0, std.mem.sliceAsBytes(&g));
        try out.download(0, got);
        if (!std.mem.eql(i32, &g, &w)) {
            std.debug.print("FAIL: counts, trial {d} ({d} rows)\n", .{ trial, rows });
            return error.CountsDiffer;
        }
        if (!std.mem.eql(u8, got, want)) {
            std.debug.print("FAIL: select, trial {d} ({d} rows)\n", .{ trial, rows });
            return error.SelectDiffers;
        }
    }
    std.debug.print("PASS: glm_shard_counts and glm_shard_select equal their host rules over 200 windows of 1 to 64 rows\n", .{});
    try fetch(gpa, &d, stream, rnd);
}

/// The prompt fetch's kernels: mark, scan (and its total), compact (the union and a whole history) and renumber,
/// against the host rules over random chunks either side of the dense limit, 1 to 2,048 rows, up to 64K positions.
fn fetch(gpa: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, rnd: std.Random) !void {
    var k = try sp.Kernels.load(d, cuda.kernels.glm_shard);
    defer k.unload();
    const W = glm.geometry.Geometry.tokens;
    const RS = glm.geometry.Geometry.latent_row;
    const max_rows = 2048;
    const max_pos = 65536;
    var tok_d = try cuda.DeviceBuffer.alloc(d, max_rows * W * 4);
    defer tok_d.free();
    var mark_d = try cuda.DeviceBuffer.alloc(d, max_pos * 4);
    defer mark_d.free();
    var sums_d = try cuda.DeviceBuffer.alloc(d, (sp.scan_segs + 1) * 4);
    defer sums_d.free();
    var plane_d = try cuda.DeviceBuffer.alloc(d, max_pos * RS);
    defer plane_d.free();
    var copy_d = try cuda.DeviceBuffer.alloc(d, max_pos * RS);
    defer copy_d.free();
    const toks = try gpa.alloc(i32, max_rows * W);
    defer gpa.free(toks);
    const marks = try gpa.alloc(i32, max_pos);
    defer gpa.free(marks);
    const cum = try gpa.alloc(i32, max_pos);
    defer gpa.free(cum);
    const got_i = try gpa.alloc(i32, max_rows * W);
    defer gpa.free(got_i);
    const want_i = try gpa.alloc(i32, max_rows * W);
    defer gpa.free(want_i);
    const plane = try gpa.alloc(u8, max_pos * RS);
    defer gpa.free(plane);
    const want = try gpa.alloc(u8, max_pos * RS);
    defer gpa.free(want);
    const got = try gpa.alloc(u8, max_pos * RS);
    defer gpa.free(got);
    for (0..60) |trial| {
        const rows = if (trial % 6 == 0) max_rows else rnd.intRangeAtMost(usize, 1, 256);
        const whole = trial % 5 == 0;
        const pos = if (whole) rnd.intRangeAtMost(usize, 0, 2050) else rnd.intRangeAtMost(usize, 2051, max_pos - max_rows);
        // the chunk's tokens: a row names up to 2,048 positions at or before its own, ascending, -1 after them
        @memset(toks[0 .. rows * W], -1);
        for (0..rows) |r| {
            const t = toks[r * W ..][0..W];
            const n = rnd.intRangeAtMost(usize, 0, 2048);
            for (t[0..n]) |*x| x.* = @intCast(rnd.intRangeAtMost(usize, 0, pos + r));
            std.mem.sort(i32, t[0..n], {}, std.sort.asc(i32));
        }
        try tok_d.upload(0, std.mem.sliceAsBytes(toks[0 .. rows * W]));
        rnd.bytes(plane[0 .. pos * RS]);
        try plane_d.upload(0, plane[0 .. pos * RS]);
        // mark and scan: the marks' prefix sums in place and their total
        const count = blk: {
            if (whole) break :blk pos;
            try mark_d.fill8(0x77, stream.handle);
            try k.mark(stream, tok_d.ptr, rows * W, mark_d.ptr, pos);
            try k.scan(stream, mark_d.ptr, pos, sums_d.ptr);
            try stream.synchronize();
            shard.markHistory(toks[0 .. rows * W], pos, marks[0..pos]);
            const n = shard.prefixSum(marks[0..pos], cum[0..pos]);
            const g = got_i[0..pos];
            try mark_d.download(0, std.mem.sliceAsBytes(g));
            var total: [1]i32 = undefined;
            try sums_d.download(sp.scan_segs * 4, std.mem.sliceAsBytes(&total));
            if (!std.mem.eql(i32, g, cum[0..pos]) or total[0] != @as(i32, @intCast(n))) {
                std.debug.print("FAIL: mark and scan, trial {d} ({d} rows from {d})\n", .{ trial, rows, pos });
                return error.ScanDiffers;
            }
            break :blk n;
        };
        // compact: the owner's rows of the union (or its whole history) in position order
        try copy_d.fill8(0x33, stream.handle);
        try k.compact(stream, plane_d.ptr, if (whole) 0 else mark_d.ptr, pos, copy_d.ptr);
        try stream.synchronize();
        @memset(want[0 .. (count + 1) * RS], 0x33);
        sp.compactRef(plane[0 .. pos * RS], if (whole) null else cum[0..pos], pos, want);
        try copy_d.download(0, got[0 .. (count + 1) * RS]);
        if (!std.mem.eql(u8, got[0 .. (count + 1) * RS], want[0 .. (count + 1) * RS])) {
            std.debug.print("FAIL: compact, trial {d} ({d} rows from {d})\n", .{ trial, rows, pos });
            return error.CompactDiffers;
        }
        if (whole) continue;
        // renumber: the other rank's tokens to rows of its copy, in place
        try k.renumber(stream, tok_d.ptr, rows * W, mark_d.ptr, pos);
        try stream.synchronize();
        shard.renumber(toks[0 .. rows * W], pos, cum[0..pos], want_i[0 .. rows * W]);
        try tok_d.download(0, std.mem.sliceAsBytes(got_i[0 .. rows * W]));
        if (!std.mem.eql(i32, got_i[0 .. rows * W], want_i[0 .. rows * W])) {
            std.debug.print("FAIL: renumber, trial {d} ({d} rows from {d})\n", .{ trial, rows, pos });
            return error.RenumberDiffers;
        }
    }
    std.debug.print("PASS: the prompt fetch's mark, scan, compact and renumber equal their host rules over 60 chunks\n", .{});
}
