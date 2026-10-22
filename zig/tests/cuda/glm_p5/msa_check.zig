//! tf-glm-p5-msa: glm_msa.cu (Python msa.cu, TF_GLM_MSA=1) on the GPU against an fp32 host attention over the same FP8 rows.
//!
//!   tf-glm-p5-msa [ROWS] [CACHE_ROWS]
//!
//! For 16 heads (one m16 tile), 22 (TP=3's 22/21/21) and 32 (TP=2): random bf16 absorbed queries, a random FP8 latent
//! cache (kv8 rows), ascending token lists of varying length (1, 15, 16, 17, 2051, ...). Checks (a) every row within a
//! relative tolerance of the host reference, (b) row independence: a row computed alone (R=1) equals its bytes in the
//! full launch, (c) rows with count 0 are left alone. GPU-only; run under the GPU lock. Exit 1 on any failure.
const std = @import("std");
const cuda = @import("cuda");
const mla = @import("glm_cuda").mla;

const lw = mla.lw;

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const rows: usize = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 24;
    const n_cache: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 4096;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var msa = try mla.Msa.load(&driver);
    defer msa.deinit();

    var prng = std.Random.DefaultPrng.init(0x6d6c61);
    const rnd = prng.random();
    const w = mla.tokens;

    // the cache
    const cache = try gpa.alloc(u8, n_cache * mla.latent_row);
    defer gpa.free(cache);
    {
        var x: [lw]f32 = undefined;
        for (0..n_cache) |i| {
            for (&x) |*v| v.* = mla.bf16((rnd.float(f32) - 0.5) * 8);
            mla.quantizeRow(&x, cache[i * mla.latent_row ..][0..mla.latent_row]);
        }
    }
    var d_cache = try cuda.DeviceBuffer.fromHost(&driver, cache);
    defer d_cache.free();

    var fails: usize = 0;
    for ([_]usize{ 16, 22, 32 }) |h| {
        const q = try gpa.alloc(f32, rows * h * lw);
        defer gpa.free(q);
        const qb = try gpa.alloc(u16, q.len);
        defer gpa.free(qb);
        for (q, qb) |*v, *b| {
            v.* = mla.bf16((rnd.float(f32) - 0.5) * 2);
            b.* = @intCast(@as(u32, @bitCast(v.*)) >> 16);
        }
        const toks = try gpa.alloc(i32, rows * w);
        defer gpa.free(toks);
        @memset(toks, -1);
        const counts = try gpa.alloc(i32, rows);
        defer gpa.free(counts);
        const lens = [_]usize{ 1, 15, 16, 17, 100, 513, 2051, 0 };
        for (0..rows) |r| {
            const n = lens[r % lens.len];
            counts[r] = @intCast(n);
            // ascending distinct tokens
            var t: usize = rnd.uintLessThan(usize, 3);
            for (0..n) |i| {
                toks[r * w + i] = @intCast(@min(t, n_cache - 1));
                t += 1 + rnd.uintLessThan(usize, @max(1, (n_cache - t) / @max(1, n - i)));
                if (t >= n_cache) t = n_cache - 1;
            }
        }
        var d_q = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(qb));
        defer d_q.free();
        var d_t = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(toks));
        defer d_t.free();
        var d_c = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(counts));
        defer d_c.free();
        var d_o = try cuda.DeviceBuffer.alloc(&driver, q.len * 2);
        defer d_o.free();
        const sentinel = try gpa.alloc(u16, q.len);
        defer gpa.free(sentinel);
        @memset(sentinel, 0x7777);
        try d_o.upload(0, std.mem.sliceAsBytes(sentinel));
        try msa.prompt(stream, d_q.ptr, d_cache.ptr, d_t.ptr, d_c.ptr, d_o.ptr, rows, h, w, mla.scale);
        try stream.synchronize();
        const out = try gpa.alloc(u16, q.len);
        defer gpa.free(out);
        try d_o.download(0, std.mem.sliceAsBytes(out));

        var worst: f32 = 0;
        var ref: [32 * lw]f32 = undefined;
        for (0..rows) |r| {
            const row = out[r * h * lw ..][0 .. h * lw];
            if (counts[r] == 0) {
                for (row) |v| if (v != 0x7777) {
                    fails += 1;
                    std.debug.print("FAIL h={d} row {d}: count 0 but written\n", .{ h, r });
                    break;
                };
                continue;
            }
            try mla.msaReference(q[r * h * lw ..][0 .. h * lw], h, cache, toks[r * w ..], @intCast(counts[r]), mla.scale, &ref, gpa);
            var mx: f32 = 0;
            for (ref[0 .. h * lw]) |v| mx = @max(mx, @abs(v));
            for (row, ref[0 .. h * lw]) |g, want| {
                const got: f32 = @bitCast(@as(u32, g) << 16);
                worst = @max(worst, @abs(got - want) / @max(mx, 1e-6));
            }
        }
        const ok_ref = worst < 2e-2;
        // (b) row independence: each row alone
        var indep = true;
        const one = try gpa.alloc(u16, h * lw);
        defer gpa.free(one);
        for (0..rows) |r| {
            if (counts[r] == 0) continue;
            var d_o1 = try cuda.DeviceBuffer.alloc(&driver, h * lw * 2);
            defer d_o1.free();
            try msa.prompt(stream, d_q.ptr + r * h * lw * 2, d_cache.ptr, d_t.ptr + r * w * 4, d_c.ptr + r * 4, d_o1.ptr, 1, h, w, mla.scale);
            try stream.synchronize();
            try d_o1.download(0, std.mem.sliceAsBytes(one));
            if (!std.mem.eql(u16, one, out[r * h * lw ..][0 .. h * lw])) indep = false;
        }
        std.debug.print("{s} h={d}: worst relative error vs host {d:.5}; row independence {s}\n", .{ if (ok_ref and indep) "PASS" else "FAIL", h, worst, if (indep) "exact" else "BROKEN" });
        if (!ok_ref or !indep) fails += 1;
    }
    return if (fails == 0) 0 else 1;
}
