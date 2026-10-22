//! GPU check for GLM KDA: the wide chain's bits equal the one-block chain, a replay of the saved rows equals
//! that state (and a shorter replay equals a shorter chain), and each segment of a window equals a solo run
//! of its own rows. Chunked prompt launches are deterministic and depend on the sub-chunk offset.
//! `tf-glm-kda`. Exit 1 on any mismatch. Needs the glm_kda and glm_kda_chunk fatbins.

const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const kda = glm.kda;

const H: usize = 4;
const rows: usize = 5;
const layers: usize = 2;
const slots: usize = 2;

pub fn main(init: std.process.Init) !u8 {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    run(init.gpa, &driver) catch |e| {
        std.debug.print("FAIL glm-kda: {t}\n", .{e});
        return 1;
    };
    std.debug.print("PASS glm-kda: chain=wide=replay, segments=solo, chunk offset\n", .{});
    return 0;
}

fn bf(i: usize) u16 {
    const t = [_]u16{ 0x3f80, 0x3f00, 0xbf00, 0x4000, 0x3e80, 0x3fc0, 0x3f40, 0xbf80 };
    return t[i % t.len];
}

fn off(ptr: u64, bytes: usize) u64 {
    return ptr + @as(u64, bytes);
}

fn expectEq(name: []const u8, a: []const u8, b: []const u8) !void {
    if (std.mem.eql(u8, a, b)) return;
    var i: usize = 0;
    while (i < a.len and i < b.len and a[i] == b[i]) i += 1;
    std.debug.print("FAIL {s}: first mismatch at byte {d} (lens {d} {d})\n", .{ name, i, a.len, b.len });
    return error.Mismatch;
}

const Hold = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    items: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn deinit(h: *Hold) void {
        for (h.items.items) |*b| b.free();
        h.items.deinit(h.gpa);
    }

    fn put(h: *Hold, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(h.d, bytes);
        try h.items.append(h.gpa, b);
        return b.ptr;
    }

    fn zeros(h: *Hold, n: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(h.d, n);
        try b.fill8(0, null);
        try h.items.append(h.gpa, b);
        return b.ptr;
    }

    fn pull(h: *Hold, ptr: u64, out: []u8) !void {
        var b = cuda.DeviceBuffer{ .d = h.d, .ptr = ptr, .len = out.len };
        try b.download(0, out);
    }
};

fn run(gpa: std.mem.Allocator, d: *const cuda.Driver) !void {
    var k = try kda.Kernels.load(d);
    defer k.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    const ops: kda.Ops = .{ .k = &k, .s = stream };
    var hold = Hold{ .gpa = gpa, .d = d };
    defer hold.deinit();

    const p_stride: usize = 1800;
    const a_stride: usize = H * kda.dk;
    const g_stride: usize = H * kda.dv;
    const b_off = kda.bOff(H);
    const C = kda.convChannels(H);
    if (b_off + H > p_stride or p_stride % 8 != 0) return error.Invalid;

    const state_bytes = H * kda.dv * kda.dk * 4;
    const out_bytes = rows * H * kda.dv * 2;
    const k_bytes = rows * H * kda.dk * 4;
    const v_bytes = rows * H * kda.dv * 2;
    const gk_bytes = rows * H * kda.dk * 4;
    const b_bytes = rows * H * 4;
    const q_bytes = k_bytes;
    const y_bytes = v_bytes;

    const P = try gpa.alloc(u16, rows * p_stride);
    defer gpa.free(P);
    const A = try gpa.alloc(u16, rows * a_stride);
    defer gpa.free(A);
    const G = try gpa.alloc(u16, rows * g_stride);
    defer gpa.free(G);
    const conv = try gpa.alloc(u16, kda.conv_rows * C);
    defer gpa.free(conv);
    const cw = try gpa.alloc(u16, C * kda.taps);
    defer gpa.free(cw);
    const state = try gpa.alloc(f32, H * kda.dv * kda.dk);
    defer gpa.free(state);
    const a_log = try gpa.alloc(f32, H);
    defer gpa.free(a_log);
    const dt = try gpa.alloc(f32, H * kda.dk);
    defer gpa.free(dt);
    const norm = try gpa.alloc(u16, kda.dv);
    defer gpa.free(norm);

    for (P, 0..) |*x, i| x.* = bf(i);
    for (A, 0..) |*x, i| x.* = bf(i + 3);
    for (G, 0..) |*x, i| x.* = bf(i + 9);
    for (conv, 0..) |*x, i| x.* = bf(i + 1);
    for (cw, 0..) |*x, i| x.* = bf(i + 5);
    for (norm, 0..) |*x, i| x.* = if (i % 2 == 0) 0x3f80 else 0x3f00;
    for (state, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 9)) * 0.05 - 0.1;
    for (a_log, 0..) |*x, i| x.* = -0.3 - 0.05 * @as(f32, @floatFromInt(i));
    for (dt, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 5)) * 0.02;

    const p = try hold.put(std.mem.sliceAsBytes(P));
    const a = try hold.put(std.mem.sliceAsBytes(A));
    const g = try hold.put(std.mem.sliceAsBytes(G));
    const cs = try hold.put(std.mem.sliceAsBytes(conv));
    const w = try hold.put(std.mem.sliceAsBytes(cw));
    const s_in = try hold.put(std.mem.sliceAsBytes(state));
    const al = try hold.put(std.mem.sliceAsBytes(a_log));
    const dtb = try hold.put(std.mem.sliceAsBytes(dt));
    const nw = try hold.put(std.mem.sliceAsBytes(norm));

    const out1 = try hold.zeros(out_bytes);
    const st1 = try hold.zeros(state_bytes);
    const k1 = try hold.zeros(k_bytes);
    const v1 = try hold.zeros(v_bytes);
    const g1 = try hold.zeros(gk_bytes);
    const b1 = try hold.zeros(b_bytes);
    const out2 = try hold.zeros(out_bytes);
    const st2 = try hold.zeros(state_bytes);
    const k2 = try hold.zeros(k_bytes);
    const v2 = try hold.zeros(v_bytes);
    const g2 = try hold.zeros(gk_bytes);
    const b2 = try hold.zeros(b_bytes);
    const q = try hold.zeros(q_bytes);
    const y = try hold.zeros(y_bytes);

    try ops.chain(H, p, p_stride, b_off, a, a_stride, g, g_stride, cs, w, s_in, al, dtb, nw, kda.eps, kda.gate_lower, rows, out1, st1, k1, v1, g1, b1);
    try ops.chainWide(H, p, p_stride, b_off, a, a_stride, g, g_stride, cs, w, s_in, al, dtb, nw, kda.eps, kda.gate_lower, rows, out2, st2, k2, v2, g2, b2, q, y);
    try stream.synchronize();

    const ho1 = try gpa.alloc(u8, out_bytes);
    defer gpa.free(ho1);
    const ho2 = try gpa.alloc(u8, out_bytes);
    defer gpa.free(ho2);
    const hs1 = try gpa.alloc(u8, state_bytes);
    defer gpa.free(hs1);
    const hs2 = try gpa.alloc(u8, state_bytes);
    defer gpa.free(hs2);
    const hk = try gpa.alloc(u8, k_bytes);
    defer gpa.free(hk);
    const hk2 = try gpa.alloc(u8, k_bytes);
    defer gpa.free(hk2);
    const hv = try gpa.alloc(u8, v_bytes);
    defer gpa.free(hv);
    const hv2 = try gpa.alloc(u8, v_bytes);
    defer gpa.free(hv2);
    const hg = try gpa.alloc(u8, gk_bytes);
    defer gpa.free(hg);
    const hg2 = try gpa.alloc(u8, gk_bytes);
    defer gpa.free(hg2);
    const hb = try gpa.alloc(u8, b_bytes);
    defer gpa.free(hb);
    const hb2 = try gpa.alloc(u8, b_bytes);
    defer gpa.free(hb2);
    try hold.pull(out1, ho1);
    try hold.pull(out2, ho2);
    try hold.pull(st1, hs1);
    try hold.pull(st2, hs2);
    try hold.pull(k1, hk);
    try hold.pull(k2, hk2);
    try hold.pull(v1, hv);
    try hold.pull(v2, hv2);
    try hold.pull(g1, hg);
    try hold.pull(g2, hg2);
    try hold.pull(b1, hb);
    try hold.pull(b2, hb2);
    try expectEq("chain vs wide out", ho1, ho2);
    try expectEq("chain vs wide state", hs1, hs2);
    try expectEq("chain vs wide k", hk, hk2);
    try expectEq("chain vs wide v", hv, hv2);
    try expectEq("chain vs wide g", hg, hg2);
    try expectEq("chain vs wide beta", hb, hb2);

    const st3 = try hold.zeros(state_bytes);
    const st4 = try hold.zeros(state_bytes);
    try ops.replay(H, s_in, k2, v2, g2, b2, rows, st3);
    try ops.replay(H, s_in, k2, v2, g2, b2, 2, st4);
    const st5 = try hold.zeros(state_bytes);
    const out5 = try hold.zeros(2 * H * kda.dv * 2);
    const k5 = try hold.zeros(2 * H * kda.dk * 4);
    const v5 = try hold.zeros(2 * H * kda.dv * 2);
    const g5 = try hold.zeros(2 * H * kda.dk * 4);
    const b5 = try hold.zeros(2 * H * 4);
    const q5 = try hold.zeros(2 * H * kda.dk * 4);
    const y5 = try hold.zeros(2 * H * kda.dv * 2);
    try ops.chainWide(H, p, p_stride, b_off, a, a_stride, g, g_stride, cs, w, s_in, al, dtb, nw, kda.eps, kda.gate_lower, 2, out5, st5, k5, v5, g5, b5, q5, y5);
    try stream.synchronize();
    const hr = try gpa.alloc(u8, state_bytes);
    defer gpa.free(hr);
    const hr2 = try gpa.alloc(u8, state_bytes);
    defer gpa.free(hr2);
    const hs5 = try gpa.alloc(u8, state_bytes);
    defer gpa.free(hs5);
    try hold.pull(st3, hr);
    try hold.pull(st4, hr2);
    try hold.pull(st5, hs5);
    try expectEq("replay all rows", hs2, hr);
    try expectEq("replay prefix vs short chain", hr2, hs5);

    try segments(&hold, ops, stream, gpa, p, a, g, w, p_stride, a_stride, g_stride, b_off, C);
    try chunkCase(&hold, ops, stream, gpa, p, a, g, cs, w, s_in, al, dtb, nw, std.mem.sliceAsBytes(state), p_stride, a_stride, g_stride, b_off, out_bytes, state_bytes);
}

fn segments(hold: *Hold, ops: kda.Ops, stream: cuda.Stream, gpa: std.mem.Allocator, p: u64, a: u64, g: u64, w: u64, p_stride: usize, a_stride: usize, g_stride: usize, b_off: usize, C: usize) !void {
    var table: [3]kda.Seg = undefined;
    try kda.writeSegments(&.{
        .{ .rows = 3, .slot = 0, .parity = 0 },
        .{ .rows = 2, .slot = 1, .parity = 1 },
    }, &table);
    // a third, empty segment: the fixed table a graph keeps when fewer streams are live
    var wide_table: [3]kda.Seg = table;
    wide_table[2] = .{ .row0 = 5, .rows = 0, .slot = 0, .parity = 0, .conv = 0, .keep = 0 };
    const seg = try hold.put(std.mem.asBytes(&wide_table));

    const conv_elems = slots * layers * kda.conv_rows * C;
    const conv_h = try gpa.alloc(u16, conv_elems);
    defer gpa.free(conv_h);
    for (conv_h, 0..) |*x, i| x.* = bf(i + 11);
    const conv = try hold.put(std.mem.sliceAsBytes(conv_h));

    const state_elems = H * kda.dv * kda.dk;
    const rec_elems = slots * 2 * layers * state_elems;
    const rec_h = try gpa.alloc(f32, rec_elems);
    defer gpa.free(rec_h);
    for (rec_h, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 13)) * 0.03 - 0.2;
    const rec = try hold.put(std.mem.sliceAsBytes(rec_h));
    const rec_src = try gpa.dupe(u8, std.mem.sliceAsBytes(rec_h));
    defer gpa.free(rec_src);

    const st = kda.recStride(layers, H);
    const out_bytes = layers * rows * H * kda.dv * 2;
    const k_bytes = layers * rows * H * kda.dk * 4;
    const v_bytes = layers * rows * H * kda.dv * 2;
    const g_bytes = k_bytes;
    const b_bytes = layers * rows * H * 4;
    const out = try hold.zeros(out_bytes);
    const ks = try hold.zeros(k_bytes);
    const vs = try hold.zeros(v_bytes);
    const gs = try hold.zeros(g_bytes);
    const bs = try hold.zeros(b_bytes);
    const q = try hold.zeros(rows * H * kda.dk * 4);
    const y = try hold.zeros(rows * H * kda.dv * 2);

    var logs: [layers]u64 = undefined;
    var dts: [layers]u64 = undefined;
    var norms: [layers]u64 = undefined;
    for (0..layers) |layer| {
        const al = try gpa.alloc(f32, H);
        defer gpa.free(al);
        const dt = try gpa.alloc(f32, H * kda.dk);
        defer gpa.free(dt);
        const nw = try gpa.alloc(u16, kda.dv);
        defer gpa.free(nw);
        for (al, 0..) |*x, i| x.* = -0.2 * @as(f32, @floatFromInt(layer + 1)) - 0.01 * @as(f32, @floatFromInt(i));
        for (dt, 0..) |*x, i| x.* = 0.01 * @as(f32, @floatFromInt((i + layer) % 7));
        for (nw, 0..) |*x, i| x.* = if ((i + layer) % 2 == 0) 0x3f80 else 0x3f40;
        logs[layer] = try hold.put(std.mem.sliceAsBytes(al));
        dts[layer] = try hold.put(std.mem.sliceAsBytes(dt));
        norms[layer] = try hold.put(std.mem.sliceAsBytes(nw));
        const layer_b: u64 = @intCast(layer);
        const conv_ptr = conv + layer_b * @as(u64, kda.conv_rows * C * 2);
        const rec_ptr = rec + layer_b * @as(u64, st.layer * 4);
        const out_ptr = out + layer_b * @as(u64, rows * H * kda.dv * 2);
        const k_ptr = ks + layer_b * @as(u64, rows * H * kda.dk * 4);
        const v_ptr = vs + layer_b * @as(u64, rows * H * kda.dv * 2);
        const g_ptr = gs + layer_b * @as(u64, rows * H * kda.dk * 4);
        const b_ptr = bs + layer_b * @as(u64, rows * H * 4);
        try ops.chainWideSegments(wide_table.len, H, seg, p, p_stride, b_off, a, a_stride, g, g_stride, conv_ptr, kda.convSlotStride(layers, H), w, rec_ptr, st.slot, st.parity, logs[layer], dts[layer], norms[layer], kda.eps, kda.gate_lower, rows, out_ptr, k_ptr, v_ptr, g_ptr, b_ptr, q, y);
    }
    try stream.synchronize();

    // solo each real segment, per layer, and compare its rows and its destination state
    const seg_in = [_]kda.SegIn{
        .{ .rows = 3, .slot = 0, .parity = 0 },
        .{ .rows = 2, .slot = 1, .parity = 1 },
    };
    for (seg_in, 0..) |s, si| {
        const n: usize = @intCast(s.rows);
        const row0: usize = if (si == 0) 0 else 3;
        const solo_out = try hold.zeros(n * H * kda.dv * 2);
        const solo_k = try hold.zeros(n * H * kda.dk * 4);
        const solo_v = try hold.zeros(n * H * kda.dv * 2);
        const solo_g = try hold.zeros(n * H * kda.dk * 4);
        const solo_b = try hold.zeros(n * H * 4);
        const solo_q = try hold.zeros(n * H * kda.dk * 4);
        const solo_y = try hold.zeros(n * H * kda.dv * 2);
        const solo_st = try hold.zeros(state_elems * 4);
        for (0..layers) |layer| {
            const layer_b: u64 = @intCast(layer);
            const slot: u64 = @intCast(s.slot);
            const conv_ptr = conv + (slot * @as(u64, layers) + layer_b) * @as(u64, kda.conv_rows * C * 2);
            const src_par: u64 = @intCast(s.parity);
            const state_ptr = rec + slot * @as(u64, st.slot * 4) + src_par * @as(u64, st.parity * 4) + layer_b * @as(u64, st.layer * 4);
            // the chain already overwrote the destination parity; the source parity is still the upload.
            // re-upload that one state from the host copy so a previous solo cannot have touched it
            // (chain reads state_in and does not write it). state_ptr points into `rec`, which the
            // segmented chain wrote only at 1 - parity. source is intact.
            const row_b: u64 = @intCast(row0);
            try ops.chainWide(H, p + row_b * @as(u64, p_stride * 2), p_stride, b_off, a + row_b * @as(u64, a_stride * 2), a_stride, g + row_b * @as(u64, g_stride * 2), g_stride, conv_ptr, w, state_ptr, logs[layer], dts[layer], norms[layer], kda.eps, kda.gate_lower, n, solo_out, solo_st, solo_k, solo_v, solo_g, solo_b, solo_q, solo_y);
            try stream.synchronize();
            const got_out = try gpa.alloc(u8, n * H * kda.dv * 2);
            defer gpa.free(got_out);
            const exp_out = try gpa.alloc(u8, n * H * kda.dv * 2);
            defer gpa.free(exp_out);
            const got_st = try gpa.alloc(u8, state_elems * 4);
            defer gpa.free(got_st);
            const exp_st = try gpa.alloc(u8, state_elems * 4);
            defer gpa.free(exp_st);
            try hold.pull(solo_out, got_out);
            try hold.pull(solo_st, got_st);
            const out_off = (layer * rows + row0) * H * kda.dv * 2;
            try hold.pull(off(out, out_off), exp_out);
            const dest_par: u64 = 1 - src_par;
            const dest = rec + slot * @as(u64, st.slot * 4) + dest_par * @as(u64, st.parity * 4) + layer_b * @as(u64, st.layer * 4);
            try hold.pull(dest, exp_st);
            var name: [96]u8 = undefined;
            const no = try std.fmt.bufPrint(&name, "seg {d} layer {d} out", .{ si, layer });
            try expectEq(no, got_out, exp_out);
            const ns = try std.fmt.bufPrint(&name, "seg {d} layer {d} state", .{ si, layer });
            try expectEq(ns, got_st, exp_st);
        }
    }

    // keep == rows skips replay: destination bytes stay the chain's. Then keep < rows replays from the
    // source parity over the chain's state, matching a solo chain of that prefix.
    const before = try gpa.alloc(u8, rec_elems * 4);
    defer gpa.free(before);
    try hold.pull(rec, before);
    try ops.replayLayersSegments(wide_table.len, H, layers, seg, rec, st.slot, st.parity, st.layer, ks, vs, gs, bs, kda.kvStride(rows, H), kda.betaStride(rows, H));
    try stream.synchronize();
    const after = try gpa.alloc(u8, rec_elems * 4);
    defer gpa.free(after);
    try hold.pull(rec, after);
    try expectEq("replay skips keep==rows", before, after);

    wide_table[0].keep = 1;
    const seg2 = try hold.put(std.mem.asBytes(&wide_table));
    try ops.replayLayersSegments(wide_table.len, H, layers, seg2, rec, st.slot, st.parity, st.layer, ks, vs, gs, bs, kda.kvStride(rows, H), kda.betaStride(rows, H));
    try stream.synchronize();
    // slot 1 keep stayed at 2 == rows, so its destination (parity 0) is unchanged; slot 0's dest is the first row only
    try hold.pull(rec, after);
    for (0..layers) |layer| {
        const at = (1 * st.slot + layer * st.layer) * 4;
        try expectEq("untouched segment", before[at .. at + state_elems * 4], after[at .. at + state_elems * 4]);
        const n: usize = 1;
        const solo_out = try hold.zeros(n * H * kda.dv * 2);
        const solo_k = try hold.zeros(n * H * kda.dk * 4);
        const solo_v = try hold.zeros(n * H * kda.dv * 2);
        const solo_g = try hold.zeros(n * H * kda.dk * 4);
        const solo_b = try hold.zeros(n * H * 4);
        const solo_q = try hold.zeros(n * H * kda.dk * 4);
        const solo_y = try hold.zeros(n * H * kda.dv * 2);
        const solo_st = try hold.zeros(state_elems * 4);
        // restore is unnecessary: source parity 0 of slot 0 was not written. Use the original host bytes uploaded
        // again so we do not depend on that, and point the solo chain at a fresh copy.
        const src_off = (0 * st.slot + 0 * st.parity + layer * st.layer) * 4;
        const fresh = try hold.put(rec_src[src_off .. src_off + state_elems * 4]);
        const layer_b: u64 = @intCast(layer);
        const conv_ptr = conv + layer_b * @as(u64, kda.conv_rows * C * 2);
        try ops.chainWide(H, p, p_stride, b_off, a, a_stride, g, g_stride, conv_ptr, w, fresh, logs[layer], dts[layer], norms[layer], kda.eps, kda.gate_lower, 1, solo_out, solo_st, solo_k, solo_v, solo_g, solo_b, solo_q, solo_y);
        try stream.synchronize();
        const got = try gpa.alloc(u8, state_elems * 4);
        defer gpa.free(got);
        const exp = try gpa.alloc(u8, state_elems * 4);
        defer gpa.free(exp);
        try hold.pull(solo_st, got);
        const dest_off = (1 * st.parity + layer * st.layer) * 4;
        try hold.pull(off(rec, dest_off), exp);
        var name: [64]u8 = undefined;
        const ns = try std.fmt.bufPrint(&name, "replay keep 1 layer {d}", .{layer});
        try expectEq(ns, got, exp);
    }
}

fn chunkCase(hold: *Hold, ops: kda.Ops, stream: cuda.Stream, gpa: std.mem.Allocator, p: u64, a: u64, g: u64, cs: u64, w: u64, s_in: u64, al: u64, dtb: u64, nw: u64, state: []const u8, p_stride: usize, a_stride: usize, g_stride: usize, b_off: usize, out_bytes: usize, state_bytes: usize) !void {
    const o0 = try hold.zeros(out_bytes);
    const o1 = try hold.zeros(out_bytes);
    const o2 = try hold.zeros(out_bytes);
    const t0 = try hold.zeros(state_bytes);
    const t1 = try hold.zeros(state_bytes);
    const t2 = try hold.zeros(state_bytes);
    try ops.chunk(H, p, p_stride, b_off, a, a_stride, g, g_stride, cs, w, s_in, al, dtb, nw, kda.eps, kda.gate_lower, rows, 0, o0, t0);
    try ops.chunk(H, p, p_stride, b_off, a, a_stride, g, g_stride, cs, w, s_in, al, dtb, nw, kda.eps, kda.gate_lower, rows, 0, o1, t1);
    try ops.chunk(H, p, p_stride, b_off, a, a_stride, g, g_stride, cs, w, s_in, al, dtb, nw, kda.eps, kda.gate_lower, rows, kda.chunkOff(30), o2, t2); // 5 rows from row 30 straddle a 32-row sub-chunk boundary (an offset inside one sub-chunk only adds exact zeros)
    try stream.synchronize();
    const a0 = try gpa.alloc(u8, out_bytes);
    defer gpa.free(a0);
    const a1 = try gpa.alloc(u8, out_bytes);
    defer gpa.free(a1);
    const a2 = try gpa.alloc(u8, out_bytes);
    defer gpa.free(a2);
    const b0 = try gpa.alloc(u8, state_bytes);
    defer gpa.free(b0);
    const b1 = try gpa.alloc(u8, state_bytes);
    defer gpa.free(b1);
    const b2 = try gpa.alloc(u8, state_bytes);
    defer gpa.free(b2);
    const sin = try gpa.alloc(u8, state_bytes);
    defer gpa.free(sin);
    try hold.pull(o0, a0);
    try hold.pull(o1, a1);
    try hold.pull(o2, a2);
    try hold.pull(t0, b0);
    try hold.pull(t1, b1);
    try hold.pull(t2, b2);
    try hold.pull(s_in, sin);
    try expectEq("chunk twice out", a0, a1);
    try expectEq("chunk twice state", b0, b1);
    try expectEq("chunk leaves state_in", state, sin);
    if (std.mem.eql(u8, a0, a2) and std.mem.eql(u8, b0, b2)) return error.ChunkOffsetIgnored;
}
