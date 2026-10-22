//! TensorFold glm5_next/cuda/{dflash2_multi.py, multi.py:_round}: stream contexts and lane adapter.
//! Ash Hart and TensorFold contributors. Drafts never sample or commit target state.
const std = @import("std");
const Forward = @import("cuda_dflash_forward.zig").Forward;
const policy = @import("cuda_draft_policy.zig");
const copy = @import("cuda_copy_drafts.zig");
const rounds = @import("cuda_rounds.zig");
pub const Context = struct { end: usize = 0, capacity: usize, copies: ?copy.Copy = null };
pub const Request = struct {
    slot: usize,
    pending: u32,
    count: usize,
    sampling: ?policy.Sampling = null,
    out: []u32,
    drafted: usize = 0,
};
pub const Multi = struct {
    gpa: std.mem.Allocator,
    forward: *Forward,
    selector: policy.Selector,
    contexts: [8]Context = undefined,
    chain_policy: policy.Policy = .{},
    pub fn init(gpa: std.mem.Allocator, forward: *Forward, selector: policy.Selector, capacity: usize) !Multi {
        try selector.validate(154880);
        if (capacity == 0) return error.BadDraftCapacity;
        var m: Multi = .{ .gpa = gpa, .forward = forward, .selector = selector };
        for (&m.contexts) |*c| c.* = .{ .capacity = capacity };
        return m;
    }
    pub fn deinit(m: *Multi) void {
        for (&m.contexts) |*c| if (c.copies) |*x| x.deinit();
    }
    pub fn reset(m: *Multi, slot: usize, prompt: []const u32, settings: ?copy.Settings) !void {
        if (slot >= m.forward.streams) return error.BadDraftSlot;
        // Create the new index before discarding the old one.
        var fresh = if (settings) |s| try copy.Copy.init(m.gpa, prompt, s) else null;
        if (fresh) |*index| index.prompt = prompt.len;
        if (m.contexts[slot].copies) |*x| x.deinit();
        m.contexts[slot].copies = fresh;
        m.contexts[slot].end = 0;
    }
    /// P6 publishes only kept serial draws, including the next pending token, to prompt lookup.
    pub fn commitDraws(m: *Multi, slot: usize, draws: []const u32) !void {
        if (slot >= m.forward.streams) return error.BadDraftSlot;
        if (m.contexts[slot].copies) |*x| try x.extend(draws);
    }
    fn stageRows(m: *Multi, pos: []const i64, slots: []const i64) !void {
        try m.forward.stage(m.forward.rowpos, std.mem.sliceAsBytes(pos));
        try m.forward.stage(m.forward.rowslots, std.mem.sliceAsBytes(slots));
    }
    /// P7 Round.commit's engine.commitDrafts calls this after the target parity/position commit.
    /// source is the five planes [5,total_rows,D] produced by the target verify pass.
    pub fn commitDrafts(m: *Multi, windows: []const rounds.Window, keeps: []const usize, source: u64, total_rows: usize) !void {
        return m.commitDraftsStrided(windows, keeps, source, total_rows, total_rows);
    }
    /// Target buffers retain their allocated plane stride even for a shorter active batch.
    pub fn commitDraftsStrided(m: *Multi, windows: []const rounds.Window, keeps: []const usize, source: u64, total_rows: usize, plane_rows: usize) !void {
        if (plane_rows < total_rows) return error.BadWindow;
        if (windows.len != keeps.len or windows.len == 0 or windows.len > 8) return error.BadWindow;
        var rows: usize = 0;
        var row0: usize = 0;
        var pos: [64]i64 = undefined;
        var slots: [64]i64 = undefined;
        for (windows, keeps, 0..) |w, k, i| {
            const slot = w.state.slot;
            if (slot >= m.forward.streams or k == 0 or k > w.tokens.len or w.state.pos < k or m.contexts[slot].end != w.state.pos - k or w.state.pos > m.contexts[slot].capacity or rows + k > 64) return error.BadDraftCommit;
            for (windows[0..i]) |prev| if (prev.state.slot == slot) return error.DuplicateSlot;
            if (row0 + w.tokens.len > total_rows) return error.BadWindow;
            for (0..k) |j| {
                pos[rows + j] = @intCast(m.contexts[slot].end + j);
                slots[rows + j] = @intCast(slot);
            }
            rows += k;
            row0 += w.tokens.len;
        }
        if (row0 != total_rows) return error.BadWindow;
        // Validate the whole batch before any cache write.
        rows = 0;
        row0 = 0;
        if (!m.forward.stager.ready()) try m.forward.stream.synchronize();
        for (windows, keeps) |w, k| {
            try rounds.packTaps(m.forward.th.base, source, plane_rows, row0, k, m.forward.tap_in + rows * 5 * 8192);
            rows += k;
            row0 += w.tokens.len;
        }
        const bucket = std.mem.alignForward(usize, rows, 8);
        try m.padRows(rows, bucket, &pos, &slots);
        try m.stageRows(pos[0..bucket], slots[0..bucket]);
        try m.forward.tapsPass(bucket);
        try m.forward.tapsDone();
        for (windows, keeps) |w, k| m.contexts[w.state.slot].end += k;
    }
    /// The kept rows of several windows of one round in one tap pass, each window at its own first row of the plane
    /// (commitDraftsStrided takes the whole round in order; this takes any of its windows): at most 64 kept rows.
    pub const TapItem = struct { slot: usize, row0: usize, keep: usize };
    pub fn commitTaps(m: *Multi, items: []const TapItem, source: u64, plane_rows: usize) !void {
        if (items.len == 0 or items.len > 8) return error.BadWindow;
        var rows: usize = 0;
        var pos: [64]i64 = undefined;
        var slots: [64]i64 = undefined;
        for (items, 0..) |it, i| {
            if (it.slot >= m.forward.streams or it.keep == 0 or rows + it.keep > 64 or it.row0 > plane_rows or it.keep > plane_rows - it.row0 or it.keep > m.contexts[it.slot].capacity - m.contexts[it.slot].end) return error.BadDraftCommit;
            for (items[0..i]) |prev| if (prev.slot == it.slot) return error.DuplicateSlot;
            for (0..it.keep) |j| {
                pos[rows + j] = @intCast(m.contexts[it.slot].end + j);
                slots[rows + j] = @intCast(it.slot);
            }
            rows += it.keep;
        }
        // Validate the whole batch before any cache write.
        rows = 0;
        if (!m.forward.stager.ready()) try m.forward.stream.synchronize();
        for (items) |it| {
            try rounds.packTaps(m.forward.th.base, source, plane_rows, it.row0, it.keep, m.forward.tap_in + rows * 5 * 8192);
            rows += it.keep;
        }
        const bucket = std.mem.alignForward(usize, rows, 8);
        try m.padRows(rows, bucket, &pos, &slots);
        try m.stageRows(pos[0..bucket], slots[0..bucket]);
        try m.forward.tapsPass(bucket);
        try m.forward.check(); // the round's one check (addTaps checks each window's pass; tapsDone would defer it to a proposal that may not come)
        for (items) |it| m.contexts[it.slot].end += it.keep;
    }
    fn padRows(m: *Multi, rows: usize, bucket: usize, pos: *[64]i64, slots: *[64]i64) !void {
        for (rows..bucket) |i| {
            pos[i] = @intCast(i - rows);
            slots[i] = @intCast(m.forward.streams);
        }
        if (bucket > rows) try m.forward.th.base.fill32(m.forward.tap_in + rows * 5 * 8192, 0, (bucket - rows) * 5 * 4096 / 2);
    }
    /// Prompt taps arrive already concatenated [rows,5*D], chunks may exceed a graph bucket.
    pub fn addTaps(m: *Multi, slot: usize, source: u64, rows: usize) !void {
        if (slot >= m.forward.streams or rows > m.contexts[slot].capacity - m.contexts[slot].end) return error.BadDraftCommit;
        var at: usize = 0;
        while (at < rows) {
            const take: usize = @min(rows - at, 64);
            const bucket = std.mem.alignForward(usize, take, 8);
            var pos: [64]i64 = undefined;
            var slots: [64]i64 = undefined;
            for (0..take) |i| {
                pos[i] = @intCast(m.contexts[slot].end + i);
                slots[i] = @intCast(slot);
            }
            try m.forward.th.base.copy(m.forward.tap_in, source + at * 5 * 8192, take * 5 * 8192);
            try m.padRows(take, bucket, &pos, &slots);
            try m.stageRows(pos[0..bucket], slots[0..bucket]);
            try m.forward.tapsPass(bucket);
            try m.forward.check();
            m.contexts[slot].end += take;
            at += take;
        }
    }
    pub fn propose(m: *Multi, reqs: []Request) !void {
        if (reqs.len == 0 or reqs.len > m.forward.streams) return error.BadDraftStreams;
        // No output/context changes until request validation completes.
        for (reqs, 0..) |r, i| {
            if (r.slot >= m.forward.streams or r.pending >= 154880) return error.BadDraftSlot;
            for (reqs[0..i]) |prev| if (prev.slot == r.slot) return error.DuplicateSlot;
        }
        var active: [8]usize = undefined;
        var n: usize = 0;
        var ids: [64]u32 = @splat(m.forward.cfg.mask);
        var rowpos: [64]i64 = undefined;
        var rowslots: [64]i64 = undefined;
        var bpos: [8]i64 = undefined;
        var bslots: [8]i64 = undefined;
        for (reqs, 0..) |*r, i| {
            r.drafted = 0;
            const room = @min(r.count, r.out.len);
            if (room == 0) continue;
            if (m.contexts[r.slot].copies) |*x| {
                const proposed = x.propose(r.out[0..room]);
                if (proposed.len > 0) {
                    r.drafted = proposed.len;
                    continue;
                }
            }
            const end = m.contexts[r.slot].end;
            if (end == 0) continue;
            active[n] = i;
            bpos[n] = @intCast(end);
            bslots[n] = @intCast(r.slot);
            ids[n * 8] = r.pending;
            for (0..8) |j| {
                rowpos[n * 8 + j] = @intCast(end + j);
                rowslots[n * 8 + j] = @intCast(r.slot);
            }
            n += 1;
        }
        if (n == 0) return;
        try m.forward.stage(m.forward.ids, std.mem.sliceAsBytes(ids[0 .. n * 8]));
        try m.stageRows(rowpos[0 .. n * 8], rowslots[0 .. n * 8]);
        try m.forward.stage(m.forward.bpos, std.mem.sliceAsBytes(bpos[0..n]));
        try m.forward.stage(m.forward.bslots, std.mem.sliceAsBytes(bslots[0..n]));
        try m.forward.block(n);
        var pack_buf: [4 * 8 * 7 * 32]f32 = undefined;
        var proj: [8 * 7 * 256]f32 = undefined;
        if (m.forward.asyncOk()) {
            // one synchronization for the flags, the candidates and the projection
            try m.forward.queueProposal(n);
            try m.forward.stream.synchronize();
            try m.forward.settle(true);
            @memcpy(pack_buf[0 .. m.forward.rank.world * n * 7 * 32], m.forward.candidatesHost(n));
            @memcpy(proj[0 .. n * 7 * 256], m.forward.projectionHost(n));
        } else {
            try m.forward.check();
            try m.forward.read(m.forward.candidates, std.mem.sliceAsBytes(pack_buf[0 .. m.forward.rank.world * n * 7 * 32]));
            try m.forward.read(m.forward.projection, std.mem.sliceAsBytes(proj[0 .. n * 7 * 256]));
        }
        for (active[0..n], 0..) |which, s| {
            var one: [4 * 7 * 32]f32 = undefined;
            for (0..m.forward.rank.world) |rank| @memcpy(one[rank * 7 * 32 ..][0 .. 7 * 32], pack_buf[(rank * n + s) * 7 * 32 ..][0 .. 7 * 32]);
            var candidates: [7 * 16]policy.Candidate = undefined;
            try policy.merge(one[0 .. m.forward.rank.world * 7 * 32], m.forward.rank.world, 7, 16, 154880, &candidates);
            const r = &reqs[which];
            r.drafted = try policy.chain(&candidates, proj[s * 7 * 256 ..][0 .. 7 * 256], m.selector, r.pending, m.contexts[r.slot].end + 1, r.sampling, m.chain_policy, r.out[0..@min(r.count, r.out.len)], null);
        }
    }
};
/// Engine-facing adapter for cuda_lanes.zig's E.DraftReq. E supplies draftSlot(seq).
/// P7 commits the target window first; this adapter proposes from its committed taps.
pub fn draftMany(m: *Multi, engine: anytype, reqs: anytype) !void {
    if (reqs.len == 0 or reqs.len > 8) return error.BadDraftStreams;
    var proposals: [8]Request = undefined;
    for (reqs, 0..) |r, i| {
        if (r.next.len == 0) return error.BadDraftRequest;
        proposals[i] = .{ .slot = try engine.draftSlot(r.seq), .pending = r.next[r.next.len - 1], .count = r.count, .sampling = r.sampling, .out = r.out };
    }
    // r.next is P7's serial sample after each kept row. Prompt reset should hold only prompt IDs;
    // the initial first draw arrives here with source=prefill_last.
    for (reqs, proposals[0..reqs.len]) |r, p| try m.commitDraws(p.slot, r.next);
    try m.propose(proposals[0..reqs.len]);
    for (reqs, proposals[0..reqs.len]) |*r, p| r.drafted = p.drafted;
}
