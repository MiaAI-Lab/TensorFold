//! `fp4-check` (no checkpoint): fn_ops' K-serial NVFP4 kernel (cuda_torch_ops.Torch.fp4Serial) against Triton's
//! `_fp4mm` at one K slice (the shared expert's gate/up: N 640 a TP=2 rank, 1280 at TP=1, K 2560), bytes compared over
//! decode-sized rows, both row strides, several value ranges and scales; then both timed.
const std = @import("std");
const cuda = @import("cuda");
const tri = @import("cuda_triton.zig");
const tops = @import("cuda_torch_ops.zig");

const Shape = struct { n: usize, k: usize };
pub const shapes = [_]Shape{ .{ .n = 640, .k = 2560 }, .{ .n = 1280, .k = 2560 } };
pub const default_rows = [_]usize{ 1, 2, 7, 15, 16, 17, 36, 48, 64, 65, 113, 128, 224, 256 };

const Fill = enum { e2m1, normal, wide };

fn bf16Bits(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

/// x values: normal-ish bf16 (or a wide exponent range); w: e2m1's values (the real tables') or the same as x.
fn fill(gpa: std.mem.Allocator, buf: cuda.DeviceBuffer, count: usize, r: std.Random, how: Fill) !void {
    const h = try gpa.alloc(u16, count);
    defer gpa.free(h);
    const e2m1 = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    for (h) |*v| {
        const sign: f32 = if (r.boolean()) -1 else 1;
        v.* = switch (how) {
            .e2m1 => bf16Bits(sign * e2m1[r.uintLessThan(usize, e2m1.len)]),
            .normal => bf16Bits(@floatCast(r.floatNorm(f64))),
            .wide => bf16Bits(sign * std.math.ldexp(1.0 + r.float(f32), r.intRangeAtMost(i32, -20, 20))),
        };
    }
    try buf.upload(0, std.mem.sliceAsBytes(h));
}

pub fn check(gpa: std.mem.Allocator, d: *const cuda.Driver, t: tri.Tri, th: tops.Torch, rows: []const usize) !bool {
    var prng = std.Random.DefaultPrng.init(0xf9_4e71a1);
    const r = prng.random();
    var all = true;
    var max_rows: usize = 0;
    for (rows) |m| max_rows = @max(max_rows, m);
    for (shapes) |c| {
        if (tri.fp4SplitK(c.n, c.k) != 1) continue;
        const xs_stride = c.k + 64;
        var x = try cuda.DeviceBuffer.alloc(d, max_rows * xs_stride * 2);
        defer x.free();
        var w = try cuda.DeviceBuffer.alloc(d, c.n * c.k * 2);
        defer w.free();
        var sc = try cuda.DeviceBuffer.alloc(d, (c.k / 16) * c.n * 4);
        defer sc.free();
        var s2 = try cuda.DeviceBuffer.alloc(d, c.n * 4);
        defer s2.free();
        try s2.fill32(@bitCast(@as(f32, 1.0)), t.s.handle);
        var o1 = try cuda.DeviceBuffer.alloc(d, max_rows * c.n * 4);
        defer o1.free();
        var o2 = try cuda.DeviceBuffer.alloc(d, max_rows * c.n * 4);
        defer o2.free();
        const h1 = try gpa.alloc(u8, max_rows * c.n * 4);
        defer gpa.free(h1);
        const h2 = try gpa.alloc(u8, max_rows * c.n * 4);
        defer gpa.free(h2);
        const hs = try gpa.alloc(f32, (c.k / 16) * c.n);
        defer gpa.free(hs);
        for ([_][2]Fill{ .{ .normal, .e2m1 }, .{ .wide, .e2m1 }, .{ .normal, .normal }, .{ .wide, .wide } }, 0..) |fl, fi| {
            try fill(gpa, x, max_rows * xs_stride, r, fl[0]);
            try fill(gpa, w, c.n * c.k, r, fl[1]);
            // the shared expert's scales are 1.0; the others check the fma by the scale
            for (hs) |*v| v.* = if (fi == 0) 1.0 else std.math.ldexp(1.0 + r.float(f32), r.intRangeAtMost(i32, -6, 6));
            try sc.upload(0, std.mem.sliceAsBytes(hs));
            // bf16 out: the shared gate/up's (the kernel set has no fp32-out `_fp4mm` at these shapes)
            for ([_]bool{false}) |fp32| for (rows) |m| for ([_]usize{ c.k, xs_stride }) |stride| {
                try o1.fill8(0xA5, t.s.handle);
                try o2.fill8(0x5A, t.s.handle);
                const fp: tri.Fp4 = .{ .weight = w.ptr, .scale = sc.ptr, .scale2 = s2.ptr };
                try t.fp4mm(x.ptr, stride, fp, o1.ptr, fp32, 0, m, c.n, c.k);
                try th.fp4Serial(x.ptr, stride, w.ptr, sc.ptr, o2.ptr, fp32, m, c.n, c.k);
                try t.s.synchronize();
                const es: usize = if (fp32) 4 else 2;
                const bytes = m * c.n * es;
                try o1.download(0, h1[0..bytes]);
                try o2.download(0, h2[0..bytes]);
                var differ: usize = 0;
                var first: ?usize = null;
                var i: usize = 0;
                while (i < bytes) : (i += es) if (!std.mem.eql(u8, h1[i .. i + es], h2[i .. i + es])) {
                    differ += 1;
                    if (first == null) first = i / es;
                };
                if (differ != 0) all = false;
                std.debug.print("{s} fp4 serial N {d} K {d} {s} M {d} stride {d} x {s} w {s}", .{ if (differ == 0) "EQUAL" else "DIFFER", c.n, c.k, if (fp32) "fp32" else "bf16", m, stride, @tagName(fl[0]), @tagName(fl[1]) });
                if (first) |f0| std.debug.print(": {d} of {d} values differ, first at row {d} col {d}", .{ differ, bytes / es, f0 / c.n, f0 % c.n });
                std.debug.print("\n", .{});
            };
        }
    }
    return all;
}

/// Mean microseconds a launch of `_fp4mm` and of the K-serial kernel at m rows (bf16 out), `reps` each.
pub fn bench(gpa: std.mem.Allocator, d: *const cuda.Driver, t: tri.Tri, th: tops.Torch, rows: []const usize, reps: usize) !void {
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    for (shapes) |c| {
        var max_rows: usize = 0;
        for (rows) |m| max_rows = @max(max_rows, m);
        var x = try cuda.DeviceBuffer.alloc(d, max_rows * c.k * 2);
        defer x.free();
        var w = try cuda.DeviceBuffer.alloc(d, c.n * c.k * 2);
        defer w.free();
        var sc = try cuda.DeviceBuffer.alloc(d, (c.k / 16) * c.n * 4);
        defer sc.free();
        try sc.fill32(@bitCast(@as(f32, 1.0)), t.s.handle);
        var o = try cuda.DeviceBuffer.alloc(d, max_rows * c.n * 2);
        defer o.free();
        try fill(gpa, x, max_rows * c.k, r, .normal);
        try fill(gpa, w, c.n * c.k, r, .e2m1);
        var e0 = try cuda.Event.init(d, true);
        defer e0.deinit();
        var e1 = try cuda.Event.init(d, true);
        defer e1.deinit();
        const fp: tri.Fp4 = .{ .weight = w.ptr, .scale = sc.ptr, .scale2 = sc.ptr };
        for (rows) |m| {
            var us: [2]f64 = undefined;
            for (0..2) |which| {
                for (0..3) |_| if (which == 0) try t.fp4mm(x.ptr, c.k, fp, o.ptr, false, 0, m, c.n, c.k) else try th.fp4Serial(x.ptr, c.k, w.ptr, sc.ptr, o.ptr, false, m, c.n, c.k);
                try e0.record(t.s);
                for (0..reps) |_| if (which == 0) try t.fp4mm(x.ptr, c.k, fp, o.ptr, false, 0, m, c.n, c.k) else try th.fp4Serial(x.ptr, c.k, w.ptr, sc.ptr, o.ptr, false, m, c.n, c.k);
                try e1.record(t.s);
                try e1.synchronize();
                us[which] = @as(f64, try cuda.Event.elapsedMs(e0, e1)) * 1000.0 / @as(f64, @floatFromInt(reps));
            }
            std.debug.print("fp4 serial N {d} K {d} M {d}: _fp4mm {d:.1} us, K-serial {d:.1} us ({d:.2}x)\n", .{ c.n, c.k, m, us[0], us[1], us[0] / @max(us[1], 1e-9) });
        }
    }
}
