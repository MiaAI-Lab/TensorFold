//! Byte checks against tools/zig/glm_p23_oracle.py fixtures. GPU execution requires GPU-FREE and flock.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda").matmuls;
const Fixture = @import("fixture.zig").Fixture;
fn size(f: Fixture, name: []const u8) !usize {
    return @intCast(try f.int(name));
}
fn upload(d: *const cuda.Driver, f: Fixture, name: []const u8) !cuda.DeviceBuffer {
    const bytes = try f.bytes(name);
    defer f.gpa.free(bytes);
    return cuda.DeviceBuffer.fromHost(d, bytes);
}
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedFixtureDirectory;
    var f = try Fixture.open(init.gpa, init.io, args[1]);
    defer f.deinit();
    const kind = try f.string("kind");
    if (std.mem.eql(u8, kind, "layout") or std.mem.eql(u8, kind, "fp8")) {
        try checkLayout(init.gpa, f, kind);
        std.debug.print("PASS {s} {s}: host byte equality\n", .{ kind, args[1] });
        return;
    }
    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&d, true);
    defer stream.deinit();
    var e = try glm.Entries.load(&ctx);
    defer e.deinit();
    const rows = try size(f, "rows");
    const n = try size(f, "n");
    const k = try size(f, "k");
    var x = try upload(&d, f, "x");
    defer x.free();
    const want = try f.bytes("out");
    defer init.gpa.free(want);
    var out = try cuda.DeviceBuffer.alloc(&d, want.len);
    defer out.free();
    try out.fill8(0, null);
    if (std.mem.eql(u8, kind, "qmm")) {
        var w = try upload(&d, f, "w");
        defer w.free();
        var sc = try upload(&d, f, "s");
        defer sc.free();
        var b = try upload(&d, f, "b");
        defer b.free();
        var xs = try upload(&d, f, "xs");
        defer xs.free();
        var part = try cuda.DeviceBuffer.alloc(&d, glm.splitK(n, k) * rows * n * 4);
        defer part.free();
        const q: glm.Q4 = .{ .w = w.ptr, .s = sc.ptr, .b = b.ptr, .n = n, .k = k };
        // Both cluster and ordered-buffer reductions must equal the Python oracle.
        for ([_]bool{ false, true }) |cluster| {
            try glm.matmul(&e, stream, x.ptr, k, xs.ptr, q, out.ptr, part.ptr, rows, try f.int("f32") != 0, cluster, try f.int("prompt") != 0);
            try stream.synchronize();
            try compare(init.gpa, out, want);
        }
    } else if (std.mem.eql(u8, kind, "exl3")) {
        const slots = try size(f, "slots");
        const ne = try size(f, "experts");
        const tile = try size(f, "tile");
        const pairs = rows * slots;
        const items = glm.experts.maxItems(pairs, ne + 1, tile);
        const names = [_][]const u8{ "gt", "ut", "dt", "suh_g", "suh_u", "svh_g", "svh_u", "suh_d", "svh_d", "picks" };
        var buffers: [names.len]cuda.DeviceBuffer = undefined;
        var loaded: usize = 0;
        defer for (buffers[0..loaded]) |*v| v.free();
        for (names, 0..) |name, i| {
            buffers[i] = try upload(&d, f, name);
            loaded += 1;
        }
        var members = try cuda.DeviceBuffer.alloc(&d, pairs * 4);
        defer members.free();
        var it = try cuda.DeviceBuffer.alloc(&d, items * 3 * 4);
        defer it.free();
        var cnt = try cuda.DeviceBuffer.alloc(&d, 12);
        defer cnt.free();
        var rank = try cuda.DeviceBuffer.alloc(&d, pairs * 4);
        defer rank.free();
        var hist = try cuda.DeviceBuffer.alloc(&d, ((pairs + 1023) / 1024) * (ne + 1) * 4);
        defer hist.free();
        const p: glm.experts.Plan = .{ .members = members.ptr, .items = it.ptr, .counts = cnt.ptr, .rank = rank.ptr, .hist = hist.ptr, .tile = tile, .experts = ne + 1 };
        const ex: glm.experts.Experts = .{ .gt = buffers[0].ptr, .ut = buffers[1].ptr, .dt = buffers[2].ptr, .suh_g = buffers[3].ptr, .suh_u = buffers[4].ptr, .svh_g = buffers[5].ptr, .svh_u = buffers[6].ptr, .suh_d = buffers[7].ptr, .svh_d = buffers[8].ptr, .shared_suh = if (try f.int("shared") != 0) buffers[3].ptr else 0, .dims = k, .width = n, .count = ne };
        var xg = try cuda.DeviceBuffer.alloc(&d, pairs * k * 2);
        defer xg.free();
        var xu = try cuda.DeviceBuffer.alloc(&d, pairs * k * 2);
        defer xu.free();
        var xd = try cuda.DeviceBuffer.alloc(&d, pairs * n * 2);
        defer xd.free();
        var z = try cuda.DeviceBuffer.alloc(&d, 8 * pairs * @max(k, n) * 4);
        defer z.free();
        var done = try cuda.DeviceBuffer.alloc(&d, (items * n / 128 + 1) * 4);
        defer done.free();
        var order = try cuda.DeviceBuffer.alloc(&d, items * 4);
        defer order.free();
        for ([_]cuda.DeviceBuffer{ xg, xu, xd, z, done }) |v| try v.fill8(0, null);
        const scratch: glm.experts.Scratch = .{ .xg = xg.ptr, .xu = xu.ptr, .xd = xd.ptr, .z = z.ptr, .done = done.ptr, .order = order.ptr, .capacity = rows, .slots = slots };
        try glm.experts.plan(&e, stream, buffers[9].ptr, pairs, p);
        for ([_]bool{ false, true }) |grouped| {
            try glm.experts.routed(&e, stream, x.ptr, k, buffers[9].ptr, p, ex, scratch, out.ptr, rows, 10, grouped);
            try stream.synchronize();
            if (grouped and k == 4096 and rows < 64 and (n == 1024 or n == 768 or n == 640 or n == 512)) {
                // Python runs `dec` here; the grouped kernel is only its documented equivalent. A pick of
                // `experts` (the fixtures draw 0..experts, inclusive) names no trellis: the grouped kernel
                // reads past the arrays for it while `dec` skips it, so only in-range pairs are compared.
                const picks = try f.bytes("picks");
                defer init.gpa.free(picks);
                try compareValidPairs(init.gpa, out, want, std.mem.bytesAsSlice(i32, picks), ne, k);
            } else try compare(init.gpa, out, want);
        }
    } else return error.UnknownFixtureKind;
    std.debug.print("PASS {s} {s}: byte equality\n", .{ kind, args[1] });
}
fn compareValidPairs(gpa: std.mem.Allocator, out: cuda.DeviceBuffer, want: []const u8, picks: []align(1) const i32, count: usize, width: usize) !void {
    const got = try gpa.alloc(u8, want.len);
    defer gpa.free(got);
    try out.download(0, got);
    const stride = width * 4;
    for (picks, 0..) |pick, pair| {
        if (pick < 0 or pick >= count) continue;
        if (!std.mem.eql(u8, got[pair * stride ..][0..stride], want[pair * stride ..][0..stride])) return error.OracleMismatch;
    }
}
fn compare(gpa: std.mem.Allocator, out: cuda.DeviceBuffer, want: []const u8) !void {
    const got = try gpa.alloc(u8, want.len);
    defer gpa.free(got);
    try out.download(0, got);
    if (!std.mem.eql(u8, got, want)) return error.OracleMismatch;
}

fn checkLayout(gpa: std.mem.Allocator, f: Fixture, kind: []const u8) !void {
    const n = try size(f, "n");
    const k = try size(f, "k");
    const bytes = try f.bytes("x");
    defer gpa.free(bytes);
    const x: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, bytes));
    if (std.mem.eql(u8, kind, "fp8")) {
        const codes = try gpa.alloc(u8, n * k);
        defer gpa.free(codes);
        const scales = try gpa.alloc(f32, n * k / 128);
        defer gpa.free(scales);
        glm.layouts.makeF8(x, codes, scales);
        try equalBytes(f, "codes", codes);
        try equalBytes(f, "scales", std.mem.sliceAsBytes(scales));
        return;
    }
    const words = try gpa.alloc(u32, n * k / 8);
    defer gpa.free(words);
    const scales = try gpa.alloc(u16, n * k / 64);
    defer gpa.free(scales);
    const biases = try gpa.alloc(u16, n * k / 64);
    defer gpa.free(biases);
    glm.layouts.quantize4(x, n, k, words, scales, biases, true);
    try equalBytes(f, "words", std.mem.sliceAsBytes(words));
    try equalBytes(f, "scales", std.mem.sliceAsBytes(scales));
    try equalBytes(f, "biases", std.mem.sliceAsBytes(biases));
    const npad = (n + 127) / 128 * 128;
    const packed_words = try gpa.alloc(u32, npad * k / 8);
    defer gpa.free(packed_words);
    const sc = try gpa.alloc(u16, npad * k / 64);
    defer gpa.free(sc);
    const bi = try gpa.alloc(u16, npad * k / 64);
    defer gpa.free(bi);
    glm.layouts.pack(words, scales, biases, n, k, packed_words, sc, bi);
    try equalBytes(f, "w", std.mem.sliceAsBytes(packed_words));
    try equalBytes(f, "s", std.mem.sliceAsBytes(sc));
    try equalBytes(f, "b", std.mem.sliceAsBytes(bi));
}
fn equalBytes(f: Fixture, name: []const u8, got: []const u8) !void {
    const want = try f.bytes(name);
    defer f.gpa.free(want);
    if (!std.mem.eql(u8, want, got)) {
        std.debug.print("FAIL loader {s}\n", .{name});
        for (want, got, 0..) |w, g, i| if (w != g) {
            std.debug.print("first diff at byte {d}: got {x:0>2} want {x:0>2}\n", .{ i, g, w });
            break;
        };
        return error.OracleMismatch;
    }
}
