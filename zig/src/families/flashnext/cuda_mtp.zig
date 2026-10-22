//! Flash Next's MTP head on CUDA (Python qwen4_exp/cuda/mtp.py and decode.py's draft draws): the next tokens'
//! embeddings and the main model's final streams, normalized and projected (fc_e, fc_h per stream), added into the
//! head's streams, its own attention layer over its own cache (State's last attention cache, `mtp_len` rows, the
//! last `mtp_drafted` of them chained drafts), the mixer, and the draft head over the draft vocabulary (the lm_head's
//! draft rows, 4-bit in groups of 32) on the last row. The draws (`sample_draft`, `sample_mapped`) read the head's
//! row on the GPU and draw on the host with the keyed rule (cuda_sampler.zig), so a draft equals Python's.
//!
//! Python source (TensorFold, https://github.com/ashhart/TensorFold, qwen4_exp/cuda/mtp.py and decode.py, authored
//! by Ash Hart (ashhart) with commits by vcruz305 inside TensorFold, which the owner allows for this port).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const tops = @import("cuda_torch_ops.zig");
const state = @import("cuda_state.zig");
const sampler = @import("cuda_sampler.zig");
const fwd = @import("cuda_forward.zig");
const nucleus = @import("cuda_nucleus.zig");

const Allocator = std.mem.Allocator;
const Forward = fwd.Forward;
const Bufs = fwd.Bufs;
const Seq = fwd.Seq;

extern "c" fn exp(x: f64) f64;
extern "c" fn log(x: f64) f64;

/// Two ranks' scratch for tp_sample_rows: a pass's device scratch, this rank's packed rows and every rank's.
pub const TpScratch = struct { s: tops.Torch.Sample, pack: u64, all: u64, max_rows: usize, max_k: usize };

/// decode.tp_sample_rows (two ranks, a sampler past the gathered candidates): every rank's top_k + MARGIN of each
/// of `rows` rows (`columns` wide at `logits`, this rank's shard) with global ids (offset, or `id_map` for the
/// draft head) and its log-sum-exp, all-gathered in rank order; each row's keyed draw at first_position + r on
/// the host, the same on both ranks, with the temperature-1 probability in `probs` when given.
pub fn tpSampleRows(f: *Forward, t: TpScratch, logits: u64, rows: usize, columns: usize, id_map: ?u64, first_position: u64, smp: lanes.Sampling, out: []u32, probs: ?[]f64) !void {
    const cm = f.comm orelse return error.NoComm;
    const k = topK(columns, smp);
    if (k > t.max_k or rows > t.max_rows) return error.TopKPastScratch;
    const width = 2 * k + 1;
    try f.th.toF32(logits, t.s.f32, rows * columns);
    try f.th.topk(t.s.f32, columns, rows, k, t.s.vals, t.s.idx, t.s.topk);
    try f.th.logsumexp(t.s.f32, rows, columns, t.s.lse, t.s.lse_scratch);
    try f.th.candidates(t.s.vals, t.s.idx, id_map, if (id_map != null) 0 else f.vocab_offset, t.s.lse, t.pack, rows, k);
    try cm.allGather(t.pack, t.all, rows * width, .f32, f.s.handle);
    const world = cm.world;
    const got = try f.gpa.alloc(f32, world * rows * width);
    defer f.gpa.free(got);
    try f.s.synchronize();
    const buf: cuda.DeviceBuffer = .{ .d = f.d, .ptr = t.all, .len = got.len * 4 };
    try buf.download(0, std.mem.sliceAsBytes(got));
    const vals = try f.gpa.alloc(f32, world * k);
    defer f.gpa.free(vals);
    const ids = try f.gpa.alloc(u64, world * k);
    defer f.gpa.free(ids);
    for (0..rows) |r| {
        var top = -std.math.inf(f64);
        for (0..world) |w| {
            const row = got[(w * rows + r) * width ..][0..width];
            for (0..k) |j| {
                vals[w * k + j] = row[j];
                ids[w * k + j] = @intCast(@as(i64, @as(i32, @bitCast(@as(u32, @bitCast(row[k + j]))))));
            }
            top = @max(top, @as(f64, row[2 * k]));
        }
        const d = try sampler.drawRow(f.gpa, vals, ids, first_position + r, smp, null);
        out[r] = @intCast(d.token);
        if (probs) |ps| {
            var sum: f64 = 0;
            for (0..world) |w| sum += exp(@as(f64, got[(w * rows + r) * width + 2 * k]) - top);
            const total = top + log(sum);
            ps[r] = 0;
            for (vals, ids) |v, id| if (id == d.token) {
                ps[r] = exp(@as(f64, v) - total);
                break;
            };
        }
    }
}

/// One stream's MTP step input: its next tokens and the main model's streams [n, S*D] they follow (device).
pub const Input = struct { seq: *Seq, next: []const u32, streams: u64 };

/// mtp_stage + mtp_compute (last_only), one stream: the draft head's logits of the last row (b.logits [1, ids]),
/// appending n entries to the head's cache at st.mtp_len; the caller advances mtp_len (Python mtp_forward).
pub fn forward(f: *Forward, seq: *Seq, x: *const Bufs, next: []const u32, streams: u64) !u64 {
    try stage(f, seq, x, next, streams);
    return compute(f, seq, x, next.len);
}

/// mtp_stage, one stream (outside a graph).
pub fn stage(f: *Forward, seq: *Seq, x: *const Bufs, next: []const u32, streams: u64) !void {
    var one: [1]fwd.Seg = undefined;
    _ = try stageMany(f, &.{.{ .seq = seq, .next = next, .streams = streams }}, x, &one);
}

/// mtp_stage: every stream's next tokens' ids and input streams into the head's buffers, one after another;
/// `segs` gets each stream's rows. Returns the rows.
pub fn stageMany(f: *Forward, inputs: []const Input, x: *const Bufs, segs: []fwd.Seg) !usize {
    if (f.w.mtp == null) return error.NoMtpHead;
    if (segs.len < inputs.len) return error.SegsTooFew;
    var n: usize = 0;
    for (inputs, 0..) |in, i| {
        if (in.next.len == 0) return error.EmptyWindow;
        const st = &in.seq.st;
        if (st.mtp_len + in.next.len > st.capacity) try f.grow(in.seq, st.mtp_len + in.next.len);
        segs[i] = .{ .seq = in.seq, .a0 = n, .a1 = n + in.next.len };
        n += in.next.len;
    }
    if (n > x.b.rows) return error.WindowPastBuffers;
    const ids = try f.gpa.alloc(u32, n);
    defer f.gpa.free(ids);
    for (inputs, segs[0..inputs.len]) |in, sg| @memcpy(ids[sg.a0..sg.a1], in.next);
    try f.stageIds(x, ids);
    const Wd = f.g.wide();
    for (inputs, segs[0..inputs.len]) |in, sg| {
        const dst = x.b.mtp_in + sg.a0 * Wd * 2;
        if (in.streams != dst) try f.th.copy(dst, in.streams, (sg.a1 - sg.a0) * Wd * 2);
    }
    return n;
}

/// mtp_compute (last_only) on staged rows, one stream from row 0 (capturable).
pub fn compute(f: *Forward, seq: *Seq, x: *const Bufs, n: usize) !u64 {
    return computeSegs(f, &.{.{ .seq = seq, .a0 = 0, .a1 = n }}, x);
}

/// mtp_compute (last_only) over every stream's rows: the draft head's logits of each stream's last row
/// (b.logits [segs, ids], contiguous); each stream's attention on its own head cache.
pub fn computeSegs(f: *Forward, segs: []const fwd.Seg, x: *const Bufs) !u64 {
    const b = &x.b;
    const g = f.g;
    const D = g.hidden;
    const S = g.streams;
    const Wd = g.wide();
    const m = &(f.w.mtp orelse return error.NoMtpHead);
    const n = segs[segs.len - 1].a1;
    const k = segs.len;
    const was = if (f.dump) |d| d.mtp else false;
    if (f.dump) |d| d.mtp = true;
    defer if (f.dump) |d| {
        d.mtp = was;
    };
    try f.mark(.stage);
    try f.t.embed(b.ids, f.w.embed, b.mtp_e, n, D, 1);
    try f.t.rmsnorm(b.mtp_e, D, m.norm_e, b.mixed, b.xs_mixed, n, D, null, f.eps);
    try f.mmx(b.mixed, D, b.xs_mixed, m.fc_e, b.mtp_eo, false, n);
    try f.t.rmsnorm(b.mtp_in, Wd, m.norm_h, b.mtp_hn, b.mtp_xh, n, Wd, null, f.eps);
    try f.mmx(b.mtp_hn, D, b.mtp_xh, m.fc_h, b.mtp_hs, false, n * S);
    try f.t.addStreams(b.mtp_eo, b.mtp_hs, b.h, n, D, S);
    try f.mark(.mtp_in);
    f.pf_moe = Forward.prefetchOf(m.mixer.down);
    const pending = try f.layerForward(&m.layer, segs, x, n, null, true);
    // several prompts in one pass: the head's cache is written by the layer above and nothing reads the draft
    // head's logits of a prompt pass (the drafts come later, from the prompts' last rows), so stop here
    if (b.prefill and k > 1) return b.logits;
    _ = try f.finish(&m.mixer, x, n, pending, false);
    // prompt buffers mix the last row into row 0; a window's last row is row n - 1; several streams' last rows
    // gathered into one block (mtp_e and mtp_xe are free by now: Python's index_select)
    var rows = b.mixed + (n - 1) * D * 2;
    var xs = b.xs_mixed + (n - 1) * (D / 32) * 4;
    if (b.prefill) {
        rows = b.mixed;
        xs = b.xs_mixed;
    } else if (k > 1) {
        for (segs, 0..) |sg, i| {
            try f.th.copy(b.mtp_e + i * D * 2, b.mixed + (sg.a1 - 1) * D * 2, D * 2);
            try f.th.copy(b.mtp_xe + i * (D / 32) * 4, b.xs_mixed + (sg.a1 - 1) * (D / 32) * 4, (D / 32) * 4);
        }
        rows = b.mtp_e;
        xs = b.mtp_xe;
    }
    try f.mark(.other);
    var columns: usize = undefined;
    if (f.w.draft_head) |q| {
        const q4: kern.Q4 = .{ .w = q.weight, .s = q.scales, .b = q.biases, .n = q.n, .k = q.k, .npad = q.npad };
        if (b.prefill) try f.ops.qmmPrefill32(rows, D, q4, b.logits, 1, false) else try f.ops.qmm32(rows, D, xs, q4, b.logits, k, false);
        columns = q.n;
    } else {
        try f.headMm(rows, D, b.logits, k);
        columns = f.w.head.n;
    }
    try f.mark(.head);
    if (f.comm) |cm| {
        try f.candidates(x, b.logits, k, columns, if (f.w.draft_head != null) f.w.draft_ids else null, f.vocab_offset, cm);
        try f.mark(.cands);
    }
    try f.put("head_logits", b.logits, k * columns * 2);
    return b.logits;
}

/// The draft head's draws on the host: its row's candidates (one rank) or the gathered ones (two ranks), the keyed
/// rule at the draft's position, and the drawn token's temperature-1 probability for the chain's stop rule.
pub const Draws = struct {
    gpa: Allocator,
    /// column -> token id (this rank's share at two ranks; the whole draft vocabulary at one), null: columns are ids
    ids: ?[]const u32,
    columns: usize,
    world: usize,
    /// the row's device scratch: fp32 copy, top-k values / columns, log-sum-exp, the packed read-back
    s: tops.Torch.Sample,
    out: u64,
    /// the most top-k values `out` holds (columns: any top_k)
    max_k: usize,
    /// two ranks: tp_sample_rows' scratch, for samplers past the gathered candidates
    tp: ?TpScratch = null,
    /// two ranks, top_k off: nucleus_rows' device scratch
    nsc: ?nucleus.Scratch = null,

    fn id(d: *const Draws, col: usize) u64 {
        return if (d.ids) |m| m[col] else col;
    }

    fn read(_: *const Draws, f: *Forward, ptr: u64, host: []u8) !void {
        try f.s.synchronize();
        const buf: cuda.DeviceBuffer = .{ .d = f.d, .ptr = ptr, .len = host.len };
        try buf.download(0, host);
    }

    /// Engine.sample_draft: the keyed draft at `position` and its probability (`logits` the head's bf16 row).
    pub fn sampleDraft(d: *const Draws, f: *Forward, x: *const Bufs, logits: u64, position: u64, s: ?lanes.Sampling) !sampler.Draw {
        return d.sampleDraftRow(f, x, logits, 0, 1, position, s);
    }

    /// sampleDraft on row `row` of the head's `rows` rows (a shared step's streams, one row each).
    pub fn sampleDraftRow(d: *const Draws, f: *Forward, x: *const Bufs, logits_base: u64, row: usize, rows: usize, position: u64, s: ?lanes.Sampling) !sampler.Draw {
        if (d.world > 1) return d.gathered(f, x, row, rows, position, s);
        const logits = logits_base + row * d.columns * 2;
        const greedy = s == null or s.?.temperature <= 0;
        if (greedy) {
            try f.th.draftGreedy(logits, d.columns, d.s, d.out);
            var got: [3]f32 = undefined;
            try d.read(f, d.out, std.mem.asBytes(&got));
            const c: usize = @intFromFloat(got[2]);
            return .{ .token = d.id(c), .prob = exp(@as(f64, got[0]) - @as(f64, got[1])) };
        }
        const k = topK(d.columns, s.?);
        if (k > d.max_k) return error.TopKPastScratch;
        try f.th.draftTopk(logits, d.columns, k, d.s, d.out);
        const got = try d.gpa.alloc(f32, 2 * k + 1);
        defer d.gpa.free(got);
        try d.read(f, d.out, std.mem.sliceAsBytes(got));
        const ids = try d.gpa.alloc(u64, k);
        defer d.gpa.free(ids);
        for (ids, got[k + 1 ..]) |*t, c| t.* = d.id(@intFromFloat(c));
        return sampler.drawRow(d.gpa, got[0..k], ids, position, s, got[k]);
    }

    /// decode.sample_mapped (a chain without the confidence rule): the keyed draw over the head's row.
    pub fn drawDraft(d: *const Draws, f: *Forward, x: *const Bufs, logits: u64, position: u64, s: ?lanes.Sampling) !u64 {
        if (d.world > 1) return (try d.gathered(f, x, 0, 1, position, s)).token;
        if (s == null or s.?.temperature <= 0) {
            try f.th.argmax(logits, d.columns, d.columns, d.s.col, 1);
            var col: i32 = undefined;
            try d.read(f, d.s.col, std.mem.asBytes(&col));
            return d.id(@intCast(col));
        }
        const k = topK(d.columns, s.?);
        if (k > d.max_k) return error.TopKPastScratch;
        try f.th.sampleTopk(logits, 1, d.columns, k, d.s);
        const vals = try d.gpa.alloc(f32, k);
        defer d.gpa.free(vals);
        const cols = try d.gpa.alloc(i64, k);
        defer d.gpa.free(cols);
        try d.read(f, d.s.vals, std.mem.sliceAsBytes(vals));
        try d.read(f, d.s.idx, std.mem.sliceAsBytes(cols));
        const ids = try d.gpa.alloc(u64, k);
        defer d.gpa.free(ids);
        for (ids, cols) |*t, c| t.* = d.id(@intCast(c));
        return (try sampler.drawRow(d.gpa, vals, ids, position, s, null)).token;
    }

    /// Two ranks: choose_gathered over the head's gathered candidates (row 0), with the probability.
    fn gathered(d: *const Draws, f: *Forward, x: *const Bufs, row: usize, rows: usize, position: u64, s: ?lanes.Sampling) !sampler.Draw {
        if (!sampler.gatheredFits(s) and s.?.top_k == 0) {
            // tp_sample_rows' nucleus branch: nucleus_rows over this row with the draft ids, its share the probability
            var tok: [1]u32 = undefined;
            var p: [1]f64 = undefined;
            try nucleus.sampleRows(f, d.nsc orelse return error.SamplerPastCandidates, x.b.logits + row * d.columns * 2, 1, d.columns, d.ids, 0, &.{position}, s.?, &tok, &p);
            return .{ .token = tok[0], .prob = p[0] };
        }
        if (!sampler.gatheredFits(s)) {
            // tp_sample_rows over this row of the head's logits, with the draft ids
            const t = d.tp orelse return error.SamplerPastCandidates;
            var tok: [1]u32 = undefined;
            var p: [1]f64 = undefined;
            try tpSampleRows(f, t, x.b.logits + row * d.columns * 2, 1, d.columns, if (d.ids != null) f.w.draft_ids else null, position, s.?, &tok, &p);
            return .{ .token = tok[0], .prob = p[0] };
        }
        const all = try d.gpa.alloc(f32, d.world * rows * (2 * state.cand + 1));
        defer d.gpa.free(all);
        try d.read(f, x.b.cand_all, std.mem.sliceAsBytes(all));
        return d.gatheredFrom(all, row, rows, position, s);
    }

    /// choose_gathered over row `row` of the step's candidates read to the host, `all` [world][rows][2 CAND + 1].
    pub fn gatheredFrom(d: *const Draws, all: []const f32, row: usize, rows: usize, position: u64, s: ?lanes.Sampling) !sampler.Draw {
        const width = 2 * state.cand + 1;
        // this row of every rank's block
        const got = try d.gpa.alloc(f32, d.world * width);
        defer d.gpa.free(got);
        for (0..d.world) |r| @memcpy(got[r * width ..][0..width], all[(r * rows + row) * width ..][0..width]);
        var out: [1]sampler.Draw = undefined;
        try sampler.chooseGathered(d.gpa, got, d.world, 1, &.{position}, s, true, &out);
        return out[0];
    }
};

/// Candidates a row's draw reads: top_k + MARGIN (at most every column), or every column without a top_k.
pub fn topK(columns: usize, s: lanes.Sampling) usize {
    return if (s.top_k != 0) @min(columns, @as(usize, s.top_k) + sampler.margin) else columns;
}

test "a draw reads top_k plus the margin, at most the row" {
    try std.testing.expectEqual(@as(usize, 28), topK(79591, .{ .seed = 1, .temperature = 1, .top_k = 20 }));
    try std.testing.expectEqual(@as(usize, 79591), topK(79591, .{ .seed = 1, .temperature = 1, .top_k = 0 }));
    try std.testing.expectEqual(@as(usize, 5), topK(5, .{ .seed = 1, .temperature = 1, .top_k = 20 }));
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Draws);
}
