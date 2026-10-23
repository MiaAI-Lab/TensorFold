//! Two ranks behind one lane backend (work/PLAN.md, "Two ranks in serving"): rank 0's `Leader` wraps its lane
//! backend so every call that moves an engine (prefill, first, queue, verify, keep, draft, unspeculate, release) is
//! first sent to rank 1 as one call record, then made; rank 1's `Follower` reads the records and makes the same
//! calls, with the same arguments and in the same order, on its own backend. The collectives inside the engines
//! keep the two in step; nothing else passes per round. Reads (read, probabilities, tree, alternatives) stay local:
//! both ranks hold the same bits. A stop ends rank 1.
//!
//! Transport (`Channel`): on GPUs `Records`, a fixed 16 KiB record broadcast from rank 0 over NCCL on the engine's
//! stream (a small broadcast is a steady ~15-20 us where a TCP frame over the CX7 netdev is 70-700 us, W8a's
//! out/w8a-comm5): u32 length, u8 kind (0 the batch inline, 1 the batch is the next `round` frame on the control
//! link: a request's long prompt, once per request; 2 stop; 3 idle: no stream is open, rank 1 waits on the link for
//! a `request` frame), 3 zero bytes, then the batch. A batch is a u16 count, then each call as a u32 length and its
//! bytes: calls without collectives (first, keep, release) ride with the next call that has some, so a round is one
//! record for its verify (with the last round's keeps) and one for its draft. `Frames` carries every batch as a
//! `round` frame over the control link (cuda_link.zig): the host tests' transport. `Watch` ends a rank whose peer is
//! lost; `Agreement` is the lane backend's prepare / agree / commit gate (cuda_lanes.Gate).
//!
//! A call is, little endian, no padding:
//!   op u8 (Op), then the call's fields in this order:
//!   prefill      key u64, max_new u32, drafts u8, sampling opt, prompt u32s, eos u32s, chunks u32s,
//!                resume_key u64 (rank 0's key of the kept state it resumes; 0 none), resume_at u32, marks u32s,
//!                mask u32s (structured output: the first token's allowed-token bits; a u32 count of 0 when none)
//!   first        key u64, position u64
//!   queue        key u64, feed, position u64
//!   verify       windows
//!   keep         windows, then one u32s path a window
//!   draft        u32 count, then each: key u64, follow u32s, first opt feed, rows opt u32s, start u64,
//!                position u64, depth u32, early u8, ranks u8
//!   unspeculate  key u64
//!   release      key u64
//!   note         bytes (a u32 count and the bytes): the lane backend's prompt-cache decision for rank 1
//!   spill        bytes: GLM spill v2's ask (glm_cuda/cuda_spill_wire), sent at once; every rank then answers it in one
//!                all-gather on the engine's stream (Agreement's gather), rank 1 from its lane backend's handler
//! where `u32s` is a u32 count and the values, `opt X` a u8 (0 none, 1 present) then X, `feed` a u8 (0 handle,
//! 1 value) and a u64, `sampling` seed u64, temperature f64, top_k u32, top_p f64, min_p f64 (f64 as their bits),
//! `windows` a u32 count then each: key u64, pending u32, held u32, tokens u32s, parents opt i32s, positions u64s
//! (a u32 count and u64s), early u8, masks u32s (structured output: each row's allowed-token bits, rows x words; a
//! count of 0 when none). A stream's key is rank 0's Stream address; rank 1 keeps its own Stream per key
//! from the prefill frame to the release frame (prompt, budget, sampling, eos and chunks are all a backend reads).
const std = @import("std");
const lanes = @import("lanes");
const cuda = @import("cuda");
const Link = @import("cuda_link.zig").Link;
const FlashComm = @import("cuda_comm.zig").Comm;

/// What the mirror needs of a communicator: the all-gather and broadcast on a stream, the world and the rank. Flash
/// Next's and the GLM engine's `Comm` each make one with `Coll.of`, so neither depends on the other's layout.
pub const Coll = struct {
    ptr: *const anyopaque,
    world: u32,
    rank: u32,
    vt: *const VTable,

    pub const VTable = struct {
        allGather: *const fn (*const anyopaque, cuda.abi.DevicePtr, cuda.abi.DevicePtr, usize, cuda.nccl.DataType, cuda.abi.Stream) anyerror!void,
        broadcast: *const fn (*const anyopaque, cuda.abi.DevicePtr, usize, u32, cuda.abi.Stream) anyerror!void,
    };

    /// `c` must outlive the result. T has `world`, `rank`, `allGather(send, recv, count, dtype, stream)` and
    /// `broadcast(buf, bytes, root, stream)` (cuda_comm.zig's Comm).
    pub fn of(comptime T: type, c: *const T) Coll {
        const Thunks = struct {
            fn ag(p: *const anyopaque, send: cuda.abi.DevicePtr, recv: cuda.abi.DevicePtr, count: usize, dt: cuda.nccl.DataType, st: cuda.abi.Stream) anyerror!void {
                const self: *const T = @ptrCast(@alignCast(p));
                return self.allGather(send, recv, count, dt, st);
            }
            fn bc(p: *const anyopaque, buf: cuda.abi.DevicePtr, bytes: usize, root: u32, st: cuda.abi.Stream) anyerror!void {
                const self: *const T = @ptrCast(@alignCast(p));
                return self.broadcast(buf, bytes, root, st);
            }
            const table: VTable = .{ .allGather = ag, .broadcast = bc };
        };
        return .{ .ptr = c, .world = c.world, .rank = c.rank, .vt = &Thunks.table };
    }

    pub fn allGather(c: Coll, send: cuda.abi.DevicePtr, recv: cuda.abi.DevicePtr, count: usize, dtype: cuda.nccl.DataType, stream: cuda.abi.Stream) !void {
        return c.vt.allGather(c.ptr, send, recv, count, dtype, stream);
    }

    pub fn broadcast(c: Coll, buf: cuda.abi.DevicePtr, bytes: usize, root: u32, stream: cuda.abi.Stream) !void {
        return c.vt.broadcast(c.ptr, buf, bytes, root, stream);
    }
};
const be = lanes.backend;
const Stream = lanes.Stream;
const Sampling = lanes.Sampling;
const Allocator = std.mem.Allocator;

pub const Op = enum(u8) {
    prefill = 1,
    first = 2,
    queue = 3,
    verify = 4,
    keep = 5,
    draft = 6,
    unspeculate = 7,
    release = 8,
    note = 9,
    prefill_begin = 10,
    prefill_step = 11,
    prefill_many = 12,
    // GLM spill v2 (wojo's 0110 HAS / LOADED): the lane backend's question every rank answers in one gather
    spill = 13,
    _,
};

pub const StreamSpec = struct {
    key: u64,
    max_new: u32,
    drafts: bool,
    sampling: ?Sampling,
    prompt: []const u32,
    eos: []const u32,
    chunks: []const u32,
    /// the kept prompt state the pass resumes (rank 0's key of it; 0: none), its position, and the pass's marks
    resume_key: u64 = 0,
    resume_at: u32 = 0,
    marks: []const u32 = &.{},
    /// structured output: the allowed-token bits of the prompt's first token (the grammar's words; empty: none)
    mask: []const u32 = &.{},
    /// the prompt's pictures and clips: the feature rows' count and width (the rows and features follow by NCCL
    /// broadcast inside the engine's prefill, not in the record)
    media: ?MediaSpec = null,
};

/// The media's shape for rank 1.
pub const MediaSpec = struct { rows: u32, width: u32 };

pub const WindowCall = struct {
    key: u64,
    pending: u32,
    held: u32,
    tokens: []const u32,
    parents: ?[]const i32,
    positions: []const u64,
    early: bool,
    /// structured output: each row's allowed-token bits (rows x words; empty: none)
    masks: []const u32 = &.{},

    pub fn rows(w: WindowCall) usize {
        return 1 + @as(usize, w.held) + w.tokens.len;
    }
};

pub const DraftCall = struct {
    key: u64,
    follow: []const u32,
    first: ?be.Feed,
    rows: ?[]const u32,
    start: u64,
    position: u64,
    depth: u32,
    early: bool,
    ranks: bool,
};

pub const Call = union(enum) {
    prefill: StreamSpec,
    first: struct { key: u64, position: u64 },
    queue: struct { key: u64, feed: be.Feed, position: u64 },
    verify: []const WindowCall,
    keep: struct { windows: []const WindowCall, paths: []const []const u32 },
    draft: []const DraftCall,
    unspeculate: u64,
    release: u64,
    /// the lane backend's own message to rank 1 (rank 0's prompt-cache decisions), opaque here
    note: []const u8,
    /// a prompt that fills between rounds: its start (prefill's fields), then slices of every filling stream
    prefill_begin: StreamSpec,
    prefill_step: []const u64,
    /// a burst of prompts prefilled together (each stream's prefill fields)
    prefill_many: []const StreamSpec,
    /// GLM spill v2: the lane backend's ask, opaque here; collective (every rank answers it in the gather after it)
    spill: []const u8,

    fn op(c: Call) Op {
        return switch (c) {
            .prefill => .prefill,
            .first => .first,
            .queue => .queue,
            .verify => .verify,
            .keep => .keep,
            .draft => .draft,
            .unspeculate => .unspeculate,
            .release => .release,
            .note => .note,
            .prefill_begin => .prefill_begin,
            .prefill_step => .prefill_step,
            .prefill_many => .prefill_many,
            .spill => .spill,
        };
    }
};

pub fn keyOf(s: *const Stream) u64 {
    return @intFromPtr(s);
}

// -- encoding -----------------------------------------------------------------------------------------------------

const Out = struct {
    list: *std.ArrayList(u8),
    gpa: Allocator,

    fn spec(o: Out, p: StreamSpec) !void {
        try o.int(u64, p.key);
        try o.int(u32, p.max_new);
        try o.flag(p.drafts);
        try o.flag(p.sampling != null);
        if (p.sampling) |s| {
            try o.int(u64, s.seed);
            try o.float(s.temperature);
            try o.int(u32, s.top_k);
            try o.float(s.top_p);
            try o.float(s.min_p);
        }
        try o.many(u32, p.prompt);
        try o.many(u32, p.eos);
        try o.many(u32, p.chunks);
        try o.int(u64, p.resume_key);
        try o.int(u32, p.resume_at);
        try o.many(u32, p.marks);
        try o.many(u32, p.mask);
        try o.flag(p.media != null);
        if (p.media) |m| {
            try o.int(u32, m.rows);
            try o.int(u32, m.width);
        }
    }
    fn int(o: Out, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try o.list.appendSlice(o.gpa, &b);
    }
    fn flag(o: Out, v: bool) !void {
        try o.int(u8, @intFromBool(v));
    }
    fn float(o: Out, v: f64) !void {
        try o.int(u64, @bitCast(v));
    }
    fn many(o: Out, comptime T: type, xs: []const T) !void {
        try o.int(u32, @intCast(xs.len));
        for (xs) |x| try o.int(T, x);
    }
    fn feed(o: Out, f: be.Feed) !void {
        switch (f) {
            .handle => |h| {
                try o.int(u8, 0);
                try o.int(u64, h);
            },
            .value => |v| {
                try o.int(u8, 1);
                try o.int(u64, v);
            },
        }
    }
    fn windows(o: Out, ws: []const WindowCall) !void {
        try o.int(u32, @intCast(ws.len));
        for (ws) |w| {
            try o.int(u64, w.key);
            try o.int(u32, w.pending);
            try o.int(u32, w.held);
            try o.many(u32, w.tokens);
            try o.flag(w.parents != null);
            if (w.parents) |p| try o.many(i32, p);
            try o.many(u64, w.positions);
            try o.flag(w.early);
            try o.many(u32, w.masks);
        }
    }
};

/// `call` as a `round` frame's bytes, appended to `list`.
pub fn encode(list: *std.ArrayList(u8), gpa: Allocator, call: Call) !void {
    const o: Out = .{ .list = list, .gpa = gpa };
    try o.int(u8, @backingInt(call.op()));
    switch (call) {
        .prefill, .prefill_begin => |p| try o.spec(p),
        .prefill_many => |ps| {
            try o.int(u32, @intCast(ps.len));
            for (ps) |p| try o.spec(p);
        },
        .first => |f| {
            try o.int(u64, f.key);
            try o.int(u64, f.position);
        },
        .queue => |q| {
            try o.int(u64, q.key);
            try o.feed(q.feed);
            try o.int(u64, q.position);
        },
        .verify => |ws| try o.windows(ws),
        .keep => |k| {
            if (k.paths.len != k.windows.len) return error.PathsWindowsMismatch;
            try o.windows(k.windows);
            for (k.paths) |p| try o.many(u32, p);
        },
        .draft => |rs| {
            try o.int(u32, @intCast(rs.len));
            for (rs) |r| {
                try o.int(u64, r.key);
                try o.many(u32, r.follow);
                try o.flag(r.first != null);
                if (r.first) |f| try o.feed(f);
                try o.flag(r.rows != null);
                if (r.rows) |rows| try o.many(u32, rows);
                try o.int(u64, r.start);
                try o.int(u64, r.position);
                try o.int(u32, r.depth);
                try o.flag(r.early);
                try o.flag(r.ranks);
            }
        },
        .unspeculate, .release => |key| try o.int(u64, key),
        .note, .spill => |bytes| try o.many(u8, bytes),
        .prefill_step => |keys| try o.many(u64, keys),
    }
}

// -- decoding -----------------------------------------------------------------------------------------------------

const In = struct {
    bytes: []const u8,
    at: usize = 0,
    arena: Allocator,

    fn int(i: *In, comptime T: type) !T {
        const n = @sizeOf(T);
        if (i.bytes.len - i.at < n) return error.ShortFrame;
        const v = std.mem.readInt(T, i.bytes[i.at..][0..n], .little);
        i.at += n;
        return v;
    }
    fn flag(i: *In) !bool {
        return switch (try i.int(u8)) {
            0 => false,
            1 => true,
            else => error.BadFrame,
        };
    }
    fn float(i: *In) !f64 {
        return @bitCast(try i.int(u64));
    }
    fn many(i: *In, comptime T: type) ![]T {
        const n = try i.int(u32);
        if ((i.bytes.len - i.at) / @sizeOf(T) < n) return error.ShortFrame;
        const out = try i.arena.alloc(T, n);
        for (out) |*x| x.* = try i.int(T);
        return out;
    }
    fn feed(i: *In) !be.Feed {
        const kind = try i.int(u8);
        const v = try i.int(u64);
        return switch (kind) {
            0 => .{ .handle = v },
            1 => .{ .value = std.math.cast(u32, v) orelse return error.BadFrame },
            else => error.BadFrame,
        };
    }
    fn count(i: *In) !usize {
        const n = try i.int(u32);
        // every element takes at least a byte: a corrupt count cannot ask for more than the frame holds
        if (n > i.bytes.len - i.at) return error.ShortFrame;
        return n;
    }
    fn spec(i: *In) !StreamSpec {
        return .{
            .key = try i.int(u64),
            .max_new = try i.int(u32),
            .drafts = try i.flag(),
            .sampling = if (try i.flag()) .{
                .seed = try i.int(u64),
                .temperature = try i.float(),
                .top_k = try i.int(u32),
                .top_p = try i.float(),
                .min_p = try i.float(),
            } else null,
            .prompt = try i.many(u32),
            .eos = try i.many(u32),
            .chunks = try i.many(u32),
            .resume_key = try i.int(u64),
            .resume_at = try i.int(u32),
            .marks = try i.many(u32),
            .mask = try i.many(u32),
            .media = if (try i.flag()) .{ .rows = try i.int(u32), .width = try i.int(u32) } else null,
        };
    }

    fn windows(i: *In) ![]WindowCall {
        const ws = try i.arena.alloc(WindowCall, try i.count());
        for (ws) |*w| w.* = .{
            .key = try i.int(u64),
            .pending = try i.int(u32),
            .held = try i.int(u32),
            .tokens = try i.many(u32),
            .parents = if (try i.flag()) try i.many(i32) else null,
            .positions = try i.many(u64),
            .early = try i.flag(),
            .masks = try i.many(u32),
        };
        return ws;
    }
};

/// A `round` frame's call; its slices live in `arena`.
pub fn decode(arena: Allocator, bytes: []const u8) !Call {
    var i: In = .{ .bytes = bytes, .arena = arena };
    const op: Op = @fromBackingInt(try i.int(u8));
    const call: Call = switch (op) {
        .prefill => .{ .prefill = try i.spec() },
        .prefill_begin => .{ .prefill_begin = try i.spec() },
        .prefill_step => .{ .prefill_step = try i.many(u64) },
        .prefill_many => blk: {
            const ps = try arena.alloc(StreamSpec, try i.count());
            for (ps) |*p| p.* = try i.spec();
            break :blk .{ .prefill_many = ps };
        },
        .first => .{ .first = .{ .key = try i.int(u64), .position = try i.int(u64) } },
        .queue => .{ .queue = .{ .key = try i.int(u64), .feed = try i.feed(), .position = try i.int(u64) } },
        .verify => .{ .verify = try i.windows() },
        .keep => blk: {
            const ws = try i.windows();
            const paths = try arena.alloc([]const u32, ws.len);
            for (paths) |*p| p.* = try i.many(u32);
            break :blk .{ .keep = .{ .windows = ws, .paths = paths } };
        },
        .draft => blk: {
            const rs = try arena.alloc(DraftCall, try i.count());
            for (rs) |*r| r.* = .{
                .key = try i.int(u64),
                .follow = try i.many(u32),
                .first = if (try i.flag()) try i.feed() else null,
                .rows = if (try i.flag()) try i.many(u32) else null,
                .start = try i.int(u64),
                .position = try i.int(u64),
                .depth = try i.int(u32),
                .early = try i.flag(),
                .ranks = try i.flag(),
            };
            break :blk .{ .draft = rs };
        },
        .unspeculate => .{ .unspeculate = try i.int(u64) },
        .release => .{ .release = try i.int(u64) },
        .note => .{ .note = try i.many(u8) },
        .spill => .{ .spill = try i.many(u8) },
        _ => return error.UnknownOp,
    };
    if (i.at != bytes.len) return error.TrailingBytes;
    return call;
}

fn windowCalls(a: Allocator, ws: []const be.Window) ![]WindowCall {
    const out = try a.alloc(WindowCall, ws.len);
    for (out, ws) |*o, w| o.* = .{
        .key = keyOf(w.stream),
        .pending = w.pending,
        .held = w.held,
        .tokens = w.tokens,
        .parents = w.parents,
        .positions = w.positions,
        .early = w.early,
        .masks = w.masks,
    };
    return out;
}

// -- batches ------------------------------------------------------------------------------------------------------

/// Calls in one send: u16 count, then each call as a u32 length and its bytes.
pub const Batch = struct {
    buf: std.ArrayList(u8) = .empty,
    count: u16 = 0,
    scratch: std.ArrayList(u8) = .empty,

    pub fn deinit(b: *Batch, gpa: Allocator) void {
        b.buf.deinit(gpa);
        b.scratch.deinit(gpa);
    }

    pub fn add(b: *Batch, gpa: Allocator, call: Call) !void {
        if (b.count == 0) {
            b.buf.clearRetainingCapacity();
            try b.buf.appendSlice(gpa, &.{ 0, 0 });
        }
        b.scratch.clearRetainingCapacity();
        try encode(&b.scratch, gpa, call);
        var len: [4]u8 = undefined;
        std.mem.writeInt(u32, &len, @intCast(b.scratch.items.len), .little);
        try b.buf.appendSlice(gpa, &len);
        try b.buf.appendSlice(gpa, b.scratch.items);
        b.count += 1;
        std.mem.writeInt(u16, b.buf.items[0..2], b.count, .little);
    }

    /// The batch's bytes; empty when it holds no call.
    pub fn bytes(b: *const Batch) []const u8 {
        return if (b.count == 0) &.{} else b.buf.items;
    }

    pub fn clear(b: *Batch) void {
        b.count = 0;
    }
};

/// The calls of a batch's bytes, in order.
pub const Calls = struct {
    bytes: []const u8,
    left: u16,
    at: usize = 2,

    pub fn of(bytes: []const u8) !Calls {
        if (bytes.len < 2) return error.ShortBatch;
        return .{ .bytes = bytes, .left = std.mem.readInt(u16, bytes[0..2], .little) };
    }

    pub fn next(c: *Calls) !?[]const u8 {
        if (c.left == 0) return if (c.at == c.bytes.len) null else error.TrailingBytes;
        if (c.bytes.len - c.at < 4) return error.ShortBatch;
        const n = std.mem.readInt(u32, c.bytes[c.at..][0..4], .little);
        c.at += 4;
        if (c.bytes.len - c.at < n) return error.ShortBatch;
        defer c.at += n;
        c.left -= 1;
        return c.bytes[c.at..][0..n];
    }
};

// -- transport ----------------------------------------------------------------------------------------------------

/// Where rank 0's batches go and rank 1 reads them, in order.
pub const Channel = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// rank 0: one batch
        send: *const fn (ptr: *anyopaque, bytes: []const u8) anyerror!void,
        /// rank 1: the next batch (valid until the next recv); null once rank 0 stopped
        recv: *const fn (ptr: *anyopaque) anyerror!?[]const u8,
        /// rank 0: no stream is open; rank 1 may wait for the next one where a lost peer shows (the link)
        idle: *const fn (ptr: *anyopaque) anyerror!void,
        /// rank 0: rank 1 leaves its loop
        stop: *const fn (ptr: *anyopaque) anyerror!void,
    };

    pub fn send(c: Channel, bytes: []const u8) !void {
        return c.vtable.send(c.ptr, bytes);
    }
    pub fn recv(c: Channel) !?[]const u8 {
        return c.vtable.recv(c.ptr);
    }
    pub fn idle(c: Channel) !void {
        return c.vtable.idle(c.ptr);
    }
    pub fn stop(c: Channel) !void {
        return c.vtable.stop(c.ptr);
    }
};

/// Every batch a `round` frame on the control link, `stop` a stop frame (the host tests' transport).
pub const Frames = struct {
    gpa: Allocator,
    link: Link,
    last: ?[]u8 = null,

    pub fn deinit(f: *Frames) void {
        if (f.last) |b| f.gpa.free(b);
        f.last = null;
    }

    pub fn channel(f: *Frames) Channel {
        return .{ .ptr = f, .vtable = &.{ .send = sendFn, .recv = recvFn, .idle = idleFn, .stop = stopFn } };
    }

    fn of(p: *anyopaque) *Frames {
        return @ptrCast(@alignCast(p));
    }
    fn sendFn(p: *anyopaque, bytes: []const u8) anyerror!void {
        try of(p).link.send(.round, bytes);
    }
    fn idleFn(_: *anyopaque) anyerror!void {}
    fn stopFn(p: *anyopaque) anyerror!void {
        try of(p).link.send(.stop, &.{});
    }
    fn recvFn(p: *anyopaque) anyerror!?[]const u8 {
        const f = of(p);
        f.deinit();
        const fr = try f.link.recv(f.gpa);
        switch (fr.tag) {
            .round => {
                f.last = fr.bytes;
                return fr.bytes;
            },
            .stop => {
                fr.deinit(f.gpa);
                return null;
            },
            else => {
                fr.deinit(f.gpa);
                return error.UnexpectedFrame;
            },
        }
    }
};

/// A record's bytes (the NCCL broadcast's size: 16 KiB holds a 64-stream round's windows; small ones cost alike).
pub const record_bytes = 16384;
const record_head = 8;

/// inline_call: the batch follows the head; on_link: the batch is the next `round` frame on the link (a long
/// prompt); stop: rank 1 leaves; idle: rank 0 has no stream open, the next record waits for a `request` frame.
pub const Kind = enum(u8) { inline_call = 0, on_link = 1, stop = 2, idle = 3, _ };

/// `bytes` as a record into `out` (record_bytes); a batch too long for it goes on the link (`on_link`).
pub fn packRecord(out: []u8, kind: Kind, bytes: []const u8) Kind {
    @memset(out[0..record_head], 0);
    const k: Kind = if (kind == .inline_call and bytes.len > record_bytes - record_head) .on_link else kind;
    std.mem.writeInt(u32, out[0..4], @intCast(bytes.len), .little);
    out[4] = @backingInt(k);
    if (k == .inline_call) @memcpy(out[record_head..][0..bytes.len], bytes);
    return k;
}

pub const Record = struct { kind: Kind, len: u32, bytes: []const u8 };

pub fn unpackRecord(rec: []const u8) !Record {
    if (rec.len < record_bytes) return error.ShortRecord;
    const len = std.mem.readInt(u32, rec[0..4], .little);
    const kind: Kind = @fromBackingInt(rec[4]);
    switch (kind) {
        .inline_call => {
            if (len > record_bytes - record_head) return error.BadRecord;
            return .{ .kind = kind, .len = len, .bytes = rec[record_head..][0..len] };
        },
        .on_link, .stop, .idle => return .{ .kind = kind, .len = len, .bytes = &.{} },
        _ => return error.BadRecord,
    }
}

/// How a record goes from rank 0 to rank 1: an NCCL broadcast (`Broadcast`) or, in the host tests, a link.
pub const Wire = struct {
    ptr: *anyopaque,
    put: *const fn (ptr: *anyopaque, record: []const u8) anyerror!void,
    get: *const fn (ptr: *anyopaque, record: []u8) anyerror!void,
};

/// One record broadcast from rank 0 on the engine's stream, so it lands on rank 1 in the order the collectives run.
pub const Broadcast = struct {
    comm: Coll,
    stream: cuda.abi.Stream,
    device: cuda.DeviceBuffer,
    host: cuda.HostBuffer,
    copied: cuda.Event, // rank 0: the record left the pinned buffer (it may be written again)
    pending: bool = false,

    pub fn init(d: *const cuda.Driver, comm: Coll, stream: cuda.abi.Stream) !Broadcast {
        var device = try cuda.DeviceBuffer.alloc(d, record_bytes);
        errdefer device.free();
        var host = try cuda.HostBuffer.alloc(d, record_bytes);
        errdefer host.free();
        return .{ .comm = comm, .stream = stream, .device = device, .host = host, .copied = try cuda.Event.init(d, false) };
    }

    pub fn deinit(b: *Broadcast) void {
        if (b.pending) b.copied.synchronize() catch {};
        b.copied.deinit();
        b.host.free();
        b.device.free();
    }

    pub fn wire(b: *Broadcast) Wire {
        return .{ .ptr = b, .put = putFn, .get = getFn };
    }

    fn of(p: *anyopaque) *Broadcast {
        return @ptrCast(@alignCast(p));
    }

    fn putFn(p: *anyopaque, record: []const u8) anyerror!void {
        const b = of(p);
        if (b.pending) try b.copied.synchronize();
        b.pending = false;
        const used = record_head + (if (record[4] == @backingInt(Kind.inline_call)) std.mem.readInt(u32, record[0..4], .little) else 0);
        @memcpy(b.host.bytes[0..used], record[0..used]);
        try b.device.uploadAsync(0, b.host.bytes[0..used], b.stream);
        try b.copied.record(.{ .d = b.device.d, .handle = b.stream });
        b.pending = true;
        try b.comm.broadcast(b.device.ptr, record_bytes, 0, b.stream);
    }

    fn getFn(p: *anyopaque, record: []u8) anyerror!void {
        const b = of(p);
        try b.comm.broadcast(b.device.ptr, record_bytes, 0, b.stream);
        try b.device.downloadAsync(0, b.host.bytes[0..record_bytes], b.stream);
        try cuda.Stream.synchronize(.{ .d = b.device.d, .handle = b.stream });
        @memcpy(record[0..record_bytes], b.host.bytes[0..record_bytes]);
    }
};

/// Batches as records on a wire; a batch longer than a record follows on the control link. While rank 0 has no
/// stream open both ranks leave the wire: rank 1 waits on the link (a `request` frame wakes it, a closed link ends
/// it), so an idle rank 1 neither spins in a collective nor misses rank 0's death.
pub const Records = struct {
    gpa: Allocator,
    wire: Wire,
    link: Link,
    watch: ?*Watch = null,
    idle: bool = true,
    rec: [record_bytes]u8 = undefined,
    last: ?[]u8 = null,

    pub fn deinit(r: *Records) void {
        if (r.last) |b| r.gpa.free(b);
        r.last = null;
    }

    pub fn channel(r: *Records) Channel {
        return .{ .ptr = r, .vtable = &.{ .send = sendFn, .recv = recvFn, .idle = idleFn, .stop = stopFn } };
    }

    fn of(p: *anyopaque) *Records {
        return @ptrCast(@alignCast(p));
    }

    fn put(r: *Records, kind: Kind, bytes: []const u8) !void {
        if (r.idle) try r.link.send(.request, &.{}); // rank 1 back on the wire
        r.idle = false;
        const k = packRecord(&r.rec, kind, bytes);
        // The record first, then a long batch on the link: rank 1 reads the link only after the record, so a batch larger
        // than the sockets can buffer (a few-hundred-thousand-token prompt) would block this write while rank 1 waits for
        // the record, and neither rank would move.
        try r.wire.put(r.wire.ptr, &r.rec);
        if (k == .on_link) try r.link.send(.round, bytes);
    }

    fn sendFn(p: *anyopaque, bytes: []const u8) anyerror!void {
        try of(p).put(.inline_call, bytes);
    }
    fn idleFn(p: *anyopaque) anyerror!void {
        const r = of(p);
        if (r.idle) return;
        try r.put(.idle, &.{});
        r.idle = true;
    }
    fn stopFn(p: *anyopaque) anyerror!void {
        const r = of(p);
        if (r.idle) return r.link.send(.stop, &.{});
        try r.put(.stop, &.{});
        r.idle = true;
    }
    fn recvFn(p: *anyopaque) anyerror!?[]const u8 {
        const r = of(p);
        if (r.last) |b| r.gpa.free(b);
        r.last = null;
        while (true) {
            if (r.idle) {
                // idle, rank 0 closing the link is its stop (a SIGTERM'd server sends no stop frame): exit 0
                if (r.watch) |w| w.idle.store(true, .release);
                const f = r.link.recv(r.gpa) catch |e| switch (e) {
                    error.PeerClosed => {
                        if (r.watch) |w| w.quiet();
                        std.log.info("rank 1: rank 0 closed the link while no stream was open: stopping", .{});
                        return null;
                    },
                    else => return e,
                };
                if (r.watch) |w| w.idle.store(false, .release);
                defer f.deinit(r.gpa);
                switch (f.tag) {
                    .request => r.idle = false,
                    .stop => {
                        if (r.watch) |w| w.quiet();
                        return null;
                    },
                    else => return error.UnexpectedFrame,
                }
            }
            // on the wire rank 0 is busy: its next record is due (a stall past the deadline is a lost rank)
            if (r.watch) |w| w.arm(w.base_s);
            defer if (r.watch) |w| w.disarm();
            try r.wire.get(r.wire.ptr, &r.rec);
            const rec = try unpackRecord(&r.rec);
            switch (rec.kind) {
                .inline_call => return rec.bytes,
                .stop => {
                    if (r.watch) |w| w.quiet();
                    return null;
                },
                .idle => r.idle = true,
                .on_link => {
                    const b = try r.link.expect(r.gpa, .round);
                    if (b.len != rec.len) {
                        r.gpa.free(b);
                        return error.RecordMismatch;
                    }
                    r.last = b;
                    return b;
                },
                _ => return error.BadRecord,
            }
        }
    }
};

// -- peer loss ----------------------------------------------------------------------------------------------------

/// Each rank's watchdog: the other rank is lost when the control link closes or errs (a dead process, and with
/// keepalive a dead link within ~11 s), or when one call (or rank 1's wait for rank 0's next record) passes its
/// deadline. Then the process exits at once (code 3): neither rank stays behind in NCCL.
pub const Watch = struct {
    io: std.Io,
    fd: std.posix.socket_t,
    /// TP=3/4, rank 0: the other followers' sockets (-1: none), watched like `fd`.
    more: [2]std.posix.socket_t = .{ -1, -1 },
    rank: u32,
    /// a call's deadline at least, in seconds (TF_TP_TIMEOUT_S, 300 by default: a long prompt's chunks)
    base_s: u64 = 300,
    deadline_ms: std.atomic.Value(i64) = .init(0),
    done: std.atomic.Value(bool) = .init(false),
    /// rank 1 waits on the link with no stream open: the link closing then is rank 0's stop, not a loss
    idle: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    /// what a lost peer does: abort `comm` and exit; the tests record the reason instead
    lost: *const fn (w: *Watch, why: []const u8) void = abortAndExit,
    comm: ?Coll = null,

    pub fn start(w: *Watch) !void {
        keepalive(w.fd);
        for (w.more) |m| if (m >= 0) keepalive(m);
        if (std.c.getenv("TF_TP_TIMEOUT_S")) |v| w.base_s = std.fmt.parseInt(u64, std.mem.span(v), 10) catch w.base_s;
        w.thread = try std.Thread.spawn(.{}, run, .{w});
    }

    /// A clean stop is under way: the other rank may close the link now.
    pub fn quiet(w: *Watch) void {
        w.done.store(true, .release);
    }

    pub fn deinit(w: *Watch) void {
        w.done.store(true, .release);
        if (w.thread) |t| t.join();
        w.thread = null;
    }

    fn now(w: *const Watch) i64 {
        return @intCast(@divTrunc(std.Io.Timestamp.now(w.io, .awake).toNanoseconds(), std.time.ns_per_ms));
    }

    /// A call starts that must end within `seconds`.
    pub fn arm(w: *Watch, seconds: u64) void {
        w.deadline_ms.store(w.now() + @as(i64, @intCast(seconds * 1000)), .release);
    }

    pub fn disarm(w: *Watch) void {
        w.deadline_ms.store(0, .release);
    }

    fn run(w: *Watch) void {
        const rdhup: i16 = 0x2000; // POLLRDHUP: the peer closed its end
        while (!w.done.load(.acquire)) {
            var fds: [3]std.posix.pollfd = undefined;
            var nfds: usize = 1;
            fds[0] = .{ .fd = w.fd, .events = rdhup | std.posix.POLL.IN, .revents = 0 };
            for (w.more) |m| if (m >= 0) {
                fds[nfds] = .{ .fd = m, .events = rdhup | std.posix.POLL.IN, .revents = 0 };
                nfds += 1;
            };
            const n = std.posix.poll(fds[0..nfds], 200) catch 0;
            const gone: i16 = rdhup | std.posix.POLL.HUP | std.posix.POLL.ERR | std.posix.POLL.NVAL;
            var any_gone = false; // some follower's link closed or erred
            var any_in = false; // some link holds bytes for the loop to read
            for (fds[0..nfds]) |p| {
                if (p.revents & gone != 0) any_gone = true;
                if (p.revents & std.posix.POLL.IN != 0) any_in = true;
            }
            // readable data alone is a frame for the loop: wait for it to be read, not for the peer
            if (n > 0 and !any_gone) {
                std.Io.sleep(w.io, .fromMilliseconds(50), .awake) catch {};
            }
            if (n > 0 and any_gone) {
                // with bytes still unread (rank 0's stop) the loop gets 2 s to read them and quiet the watch
                var grace: u32 = 0;
                const unread = any_in;
                while (unread and grace < 20 and !w.done.load(.acquire)) : (grace += 1) std.Io.sleep(w.io, .fromMilliseconds(100), .awake) catch {};
                if (w.done.load(.acquire)) return;
                if (w.idle.load(.acquire)) {
                    // the follow loop reads the close and returns; give it a moment, then leave cleanly anyway
                    var settle: u32 = 0;
                    while (settle < 20 and !w.done.load(.acquire)) : (settle += 1) std.Io.sleep(w.io, .fromMilliseconds(100), .awake) catch {};
                    if (w.done.load(.acquire)) return;
                    std.log.info("rank {d}: the other rank closed the link while no stream was open: stopping", .{w.rank});
                    std.c._exit(0);
                }
                return w.lost(w, "the other rank closed the control link (it exited, or the link is down)");
            }
            const d = w.deadline_ms.load(.acquire);
            if (d != 0 and w.now() > d) return w.lost(w, "a call passed its deadline (TF_TP_TIMEOUT_S): the other rank stalled");
        }
    }

    /// Log and leave at once. Not ncclCommAbort first: it frees the communicator while this rank's other thread
    /// may be entering a collective on it (a segfault, out/w7k6); _exit runs no atexit handler that could wait on
    /// a kernel spinning for the lost peer, and the driver frees the GPU with the process.
    fn abortAndExit(w: *Watch, why: []const u8) void {
        std.log.err("rank {d}: {s}; exiting", .{ w.rank, why });
        std.c._exit(3);
    }
};

/// The lane backend's gate at two ranks (cuda_lanes.Gate): each rank's prepare outcome in one 1-int all-gather on
/// the engine's stream before the call's first collective; a failure after it goes to the watchdog (exit). GLM spill
/// v2's asks gather wider answers (`gather`) in buffers of their own, made at the first ask.
pub const Agreement = struct {
    comm: Coll,
    stream: cuda.abi.Stream,
    device: cuda.DeviceBuffer,
    host: cuda.HostBuffer,
    watch: *Watch,
    count: u64 = 0,
    /// GLM spill v2's answers: this rank's words, then every rank's (gather_words a rank, up to 4 ranks)
    words: ?cuda.DeviceBuffer = null,
    words_host: ?cuda.HostBuffer = null,
    gathers: u64 = 0,

    /// The most words a rank answers an ask with (glm_cuda/cuda_spill_wire.stats_words).
    pub const gather_words = 64;
    const gather_bytes = (1 + 4) * gather_words * 8;

    pub fn init(d: *const cuda.Driver, comm: Coll, stream: cuda.abi.Stream, watch: *Watch) !Agreement {
        var device = try cuda.DeviceBuffer.alloc(d, 32); // 2 ints of room, then up to 4 ranks' answers
        errdefer device.free();
        return .{ .comm = comm, .stream = stream, .device = device, .host = try cuda.HostBuffer.alloc(d, 32), .watch = watch };
    }

    pub fn deinit(a: *Agreement) void {
        if (a.words_host) |*h| h.free();
        if (a.words) |*d| d.free();
        a.host.free();
        a.device.free();
    }

    pub fn gate(a: *Agreement) @import("cuda_lanes.zig").Gate {
        return .{ .ptr = a, .agree = agreeFn, .fatal = fatalFn, .gather = gatherFn };
    }

    fn of(p: *anyopaque) *Agreement {
        return @ptrCast(@alignCast(p));
    }

    fn agreeFn(p: *anyopaque, ok: bool) anyerror!bool {
        const a = of(p);
        a.count += 1;
        const h = a.host.slice(i32);
        h[0] = @intFromBool(ok);
        try a.device.uploadAsync(0, a.host.bytes[0..4], a.stream);
        try a.comm.allGather(a.device.ptr, try a.device.at(8), 1, .i32, a.stream);
        const world: usize = a.comm.world;
        try a.device.downloadAsync(8, a.host.bytes[8 .. 8 + 4 * world], a.stream);
        try cuda.Stream.synchronize(.{ .d = a.device.d, .handle = a.stream });
        for (h[2 .. 2 + world]) |v| if (v != 1) return false;
        return true;
    }

    /// Every rank's `mine` (the same count on each: the ask fixes it) in one all-gather on the engine's stream, in rank
    /// order into `out`. A rank waiting here for another's answer is under the watchdog's deadline (armed when no call
    /// armed it: rank 0 asks between calls; rank 1 answers inside the mirror's call).
    fn gatherFn(p: *anyopaque, mine: []const u64, out: []u64) anyerror!usize {
        const a = of(p);
        const world: usize = a.comm.world;
        if (mine.len == 0 or mine.len > gather_words or world > 4 or out.len < world * mine.len) return error.BadGather;
        if (a.words == null) {
            var d = try cuda.DeviceBuffer.alloc(a.device.d, gather_bytes);
            errdefer d.free();
            a.words_host = try cuda.HostBuffer.alloc(a.device.d, gather_bytes);
            a.words = d;
        }
        const armed = a.watch.deadline_ms.load(.acquire) != 0;
        if (!armed) a.watch.arm(a.watch.base_s);
        defer if (!armed) a.watch.disarm();
        a.gathers += 1;
        const dev = &a.words.?;
        const h = a.words_host.?.slice(u64);
        const n = mine.len;
        @memcpy(h[0..n], mine);
        try dev.uploadAsync(0, a.words_host.?.bytes[0 .. 8 * n], a.stream);
        try a.comm.allGather(dev.ptr, try dev.at(8 * gather_words), n, .u64, a.stream);
        try dev.downloadAsync(8 * gather_words, a.words_host.?.bytes[8 * gather_words ..][0 .. 8 * n * world], a.stream);
        try cuda.Stream.synchronize(.{ .d = dev.d, .handle = a.stream });
        @memcpy(out[0 .. n * world], h[gather_words..][0 .. n * world]);
        return world;
    }

    fn fatalFn(p: *anyopaque, what: []const u8, err: anyerror) void {
        const a = of(p);
        var buf: [160]u8 = undefined;
        const why = std.fmt.bufPrint(&buf, "{s} failed after the ranks agreed ({s})", .{ what, @errorName(err) }) catch "an engine call failed after the ranks agreed";
        a.watch.lost(a.watch, why);
    }
};

/// TCP keepalive on the control link: a link that goes down errs within ~11 s (5 s idle, 3 probes 2 s apart).
fn keepalive(fd: std.posix.socket_t) void {
    const posix = std.posix;
    const on: c_int = 1;
    posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.KEEPALIVE, std.mem.asBytes(&on)) catch {};
    for ([_]struct { u32, c_int }{ .{ 4, 5 }, .{ 5, 2 }, .{ 6, 3 } }) |o| { // TCP_KEEPIDLE, KEEPINTVL, KEEPCNT
        posix.setsockopt(fd, posix.IPPROTO.TCP, o[0], std.mem.asBytes(&o[1])) catch {};
    }
}

// -- rank 0 -------------------------------------------------------------------------------------------------------

/// Rank 0's backend: each engine call goes to rank 1 first, then to `inner`. A call that runs no collective
/// (first, keep, release) waits in the batch for the next one that does (prefill, queue, verify, draft,
/// unspeculate): a round is then one record (the last round's keep with this verify) plus its draft. Calls come
/// from one thread (the lane host's), so the batch needs no lock.
pub const Leader = struct {
    gpa: Allocator,
    inner: be.Backend,
    channel: Channel,
    batch: Batch = .{},
    arena: std.heap.ArenaAllocator,
    vtable: be.Backend.VTable = undefined,
    open: std.AutoHashMapUnmanaged(u64, void) = .empty,
    watch: ?*Watch = null,
    sends: u64 = 0,
    calls: u64 = 0,
    stopped: bool = false,
    /// a send failed: rank 1 is gone, so every later call fails rather than run out of step
    broken: bool = false,

    pub fn init(gpa: Allocator, inner: be.Backend, channel: Channel) Leader {
        return .{ .gpa = gpa, .inner = inner, .channel = channel, .arena = .init(gpa) };
    }

    pub fn deinit(self: *Leader) void {
        self.stop();
        self.batch.deinit(self.gpa);
        self.open.deinit(self.gpa);
        self.arena.deinit();
    }

    /// Rank 1 leaves its loop (once, after the calls still in the batch; the channel stays open for the caller).
    pub fn stop(self: *Leader) void {
        if (self.stopped) return;
        if (!self.broken) {
            self.flush() catch {};
            self.channel.stop() catch {};
        }
        self.stopped = true;
    }

    fn flush(self: *Leader) !void {
        if (self.batch.count == 0) return;
        defer self.batch.clear();
        self.sends += 1;
        self.channel.send(self.batch.bytes()) catch |e| {
            self.broken = true;
            return e;
        };
    }

    /// `call` into the batch; a call with collectives sends the batch now (rank 1 must join them).
    fn send(self: *Leader, call: Call, collective: bool) !void {
        if (self.broken or self.stopped) return error.FollowerGone;
        try self.batch.add(self.gpa, call);
        self.calls += 1;
        if (collective) try self.flush();
    }

    /// The lane backend's notes to rank 1 (cuda_lanes.Notes): each rides with the next call that has collectives,
    /// unless the backend flushes them (GLM spill v2) or asks (a collective of its own).
    pub fn notes(self: *Leader) @import("cuda_lanes.zig").Notes {
        return .{ .ptr = self, .send = noteFn, .ask = askFn, .flush = flushFn };
    }

    fn noteFn(p: *anyopaque, bytes: []const u8) void {
        const self = of(p);
        self.send(.{ .note = bytes }, false) catch |e| std.log.err("rank 0: a prompt-cache note not sent to rank 1 ({s})", .{@errorName(e)});
    }

    /// GLM spill v2's ask: the batch goes now with it (rank 1 must reach the gather that follows; an idle rank 1 is
    /// woken by the record), then the caller's gather.
    fn askFn(p: *anyopaque, bytes: []const u8) anyerror!void {
        const self = of(p);
        try self.send(.{ .spill = bytes }, true);
    }

    /// GLM spill v2: the notes waiting in the batch go now; with no stream open rank 1 is sent back to its idle wait
    /// (after an ask too: then only once its gather is done, so both ranks' collectives stay in one order).
    fn flushFn(p: *anyopaque) void {
        const self = of(p);
        if (self.broken or self.stopped) return;
        self.flush() catch |e| return std.log.err("rank 0: spill notes not sent to rank 1 ({s})", .{@errorName(e)});
        if (self.open.count() == 0) self.channel.idle() catch |e| {
            self.broken = true;
            std.log.err("rank 0: idle not sent to rank 1 ({s})", .{@errorName(e)});
        };
    }

    fn arm(self: *Leader, seconds: u64) void {
        if (self.watch) |w| w.arm(@max(seconds, w.base_s));
    }

    fn disarm(self: *Leader) void {
        if (self.watch) |w| w.disarm();
    }

    fn scratch(self: *Leader) Allocator {
        _ = self.arena.reset(.retain_capacity);
        return self.arena.allocator();
    }

    pub fn backend(self: *Leader) be.Backend {
        const v = self.inner.vtable;
        self.vtable = .{
            .prefill = prefillFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .unspeculate = if (v.unspeculate != null) unspeculateFn else null,
            .probabilities = if (v.probabilities != null) probabilitiesFn else null,
            .tree = if (v.tree != null) treeFn else null,
            .alternatives = if (v.alternatives != null) alternativesFn else null,
            .prefill_begin = if (v.prefill_begin != null and v.fill_step != null) prefillBeginFn else null,
            .fill_step = if (v.prefill_begin != null and v.fill_step != null) prefillStepFn else null,
            .prefill_many = if (v.prefill_many != null) prefillManyFn else null,
            .first_masked = if (v.first_masked != null) firstMaskedFn else null,
            .held = if (v.held != null) heldFn else null,
            .release = releaseFn,
        };
        return .{ .ptr = self, .vtable = &self.vtable };
    }

    fn of(p: *anyopaque) *Leader {
        return @ptrCast(@alignCast(p));
    }

    fn specOf(s: *Stream) StreamSpec {
        return .{
            .key = keyOf(s),
            .max_new = s.max_new,
            .drafts = s.drafts,
            .sampling = s.sampling,
            .prompt = s.prompt(),
            .eos = s.eos,
            .chunks = s.chunks,
            .resume_key = if (s.reuse.saved) |saved| @intFromPtr(saved) else 0,
            .resume_at = s.reuse.at,
            .marks = s.reuse.marks,
            .media = if (s.media) |m| .{ .rows = @intCast(m.rowCount()), .width = m.width } else null,
        };
    }

    fn prefillBeginFn(p: *anyopaque, s: *Stream) anyerror!void {
        const self = of(p);
        try self.open.put(self.gpa, keyOf(s), {});
        try self.send(.{ .prefill_begin = specOf(s) }, true);
        self.arm(0);
        defer self.disarm();
        return self.inner.vtable.prefill_begin.?(self.inner.ptr, s);
    }

    fn prefillStepFn(p: *anyopaque, streams: []const *Stream, states: []be.FillState) anyerror!void {
        const self = of(p);
        const keys = try self.scratch().alloc(u64, streams.len);
        for (keys, streams) |*k, s| k.* = keyOf(s);
        try self.send(.{ .prefill_step = keys }, true);
        self.arm(0);
        defer self.disarm();
        return self.inner.vtable.fill_step.?(self.inner.ptr, streams, states);
    }

    fn prefillManyFn(p: *anyopaque, ss: []const *Stream) anyerror!void {
        const self = of(p);
        // a prompt with pictures or clips prefills alone: its rows go by broadcast inside its own prefill
        for (ss) |s| if (s.media != null) return error.BurstRefused;
        const specs = try self.scratch().alloc(StreamSpec, ss.len);
        var seconds: usize = 0;
        for (specs, ss) |*sp, s| {
            try self.open.put(self.gpa, keyOf(s), {});
            sp.* = specOf(s);
            seconds += s.prompt_len;
        }
        try self.send(.{ .prefill_many = specs }, true);
        self.arm(seconds / 400);
        defer self.disarm();
        return self.inner.vtable.prefill_many.?(self.inner.ptr, ss);
    }

    fn prefillFn(p: *anyopaque, s: *Stream) anyerror!void {
        const self = of(p);
        try self.open.put(self.gpa, keyOf(s), {});
        var spec = specOf(s);
        // structured output: the bits of the first token ride with the prompt, so each rank masks its own slice
        if (s.grammar) |g| spec.mask = (try g.next(self.scratch())) orelse &.{};
        try self.send(.{ .prefill = spec }, true);
        // a prompt's chunks: 300 s, or 2.5 ms a token past that (a 1M-token prompt)
        self.arm(s.prompt_len / 400);
        defer self.disarm();
        s.first_mask = spec.mask;
        defer s.first_mask = &.{};
        return self.inner.vtable.prefill(self.inner.ptr, s);
    }

    fn firstFn(p: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const self = of(p);
        try self.send(.{ .first = .{ .key = keyOf(s), .position = position } }, false);
        return self.inner.vtable.first(self.inner.ptr, s, position);
    }

    /// The first token again among the grammar's allowed tokens: rank 1 asks for the same first token (its draw was masked
    /// with the prompt, from the bits the prefill call carried).
    fn firstMaskedFn(p: *anyopaque, s: *Stream, position: u64, mask: []const u32) anyerror!u64 {
        const self = of(p);
        try self.send(.{ .first = .{ .key = keyOf(s), .position = position } }, false);
        return self.inner.vtable.first_masked.?(self.inner.ptr, s, position, mask);
    }

    /// The stream's held drafts as host tokens (read on this rank only: the window that carries them goes to both).
    fn heldFn(p: *anyopaque, s: *Stream, out: []u32) anyerror!void {
        const self = of(p);
        return self.inner.vtable.held.?(self.inner.ptr, s, out);
    }

    fn queueFn(p: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const self = of(p);
        try self.send(.{ .queue = .{ .key = keyOf(s), .feed = feed, .position = position } }, true);
        self.arm(0);
        defer self.disarm();
        return self.inner.vtable.queue(self.inner.ptr, s, feed, position);
    }

    fn readFn(p: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(p);
        return self.inner.vtable.read(self.inner.ptr, handle);
    }

    fn verifyFn(p: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(p);
        try self.send(.{ .verify = try windowCalls(self.scratch(), windows) }, true);
        self.arm(0);
        defer self.disarm();
        return self.inner.vtable.verify(self.inner.ptr, windows, out);
    }

    fn keepFn(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(p);
        try self.send(.{ .keep = .{ .windows = try windowCalls(self.scratch(), windows), .paths = paths } }, false);
        return self.inner.vtable.keep(self.inner.ptr, windows, paths);
    }

    fn draftFn(p: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(p);
        const a = self.scratch();
        const calls = try a.alloc(DraftCall, requests.len);
        for (calls, requests) |*c, r| {
            if (r.lanes != null) return error.TreesNotMirrored;
            c.* = .{ .key = keyOf(r.stream), .follow = r.follow, .first = r.first, .rows = r.rows, .start = r.start, .position = r.position, .depth = r.depth, .early = r.early, .ranks = r.ranks };
        }
        try self.send(.{ .draft = calls }, true);
        self.arm(0);
        defer self.disarm();
        return self.inner.vtable.draft(self.inner.ptr, requests);
    }

    fn unspeculateFn(p: *anyopaque, s: *Stream) anyerror!void {
        const self = of(p);
        try self.send(.{ .unspeculate = keyOf(s) }, true);
        self.arm(0);
        defer self.disarm();
        return self.inner.vtable.unspeculate.?(self.inner.ptr, s);
    }

    fn probabilitiesFn(p: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const self = of(p);
        return self.inner.vtable.probabilities.?(self.inner.ptr, s, out);
    }

    fn treeFn(p: *anyopaque, s: *Stream, gpa: Allocator) anyerror!?lanes.stream.Held {
        const self = of(p);
        return self.inner.vtable.tree.?(self.inner.ptr, s, gpa);
    }

    fn alternativesFn(p: *anyopaque, s: *Stream, out: []be.Alternative) anyerror!usize {
        const self = of(p);
        return self.inner.vtable.alternatives.?(self.inner.ptr, s, out);
    }

    fn releaseFn(p: *anyopaque, s: *Stream) void {
        const self = of(p);
        self.send(.{ .release = keyOf(s) }, false) catch |e| std.log.err("rank 0: release not sent to rank 1 ({s})", .{@errorName(e)});
        self.inner.vtable.release(self.inner.ptr, s);
        // the last open stream: rank 1 waits on the link until the next request
        _ = self.open.remove(keyOf(s));
        if (self.open.count() == 0 and !self.broken and !self.stopped) {
            self.flush() catch |e| std.log.err("rank 0: release not sent to rank 1 ({s})", .{@errorName(e)});
            self.channel.idle() catch |e| {
                self.broken = true;
                std.log.err("rank 0: idle not sent to rank 1 ({s})", .{@errorName(e)});
            };
        }
    }
};

// -- rank 1 -------------------------------------------------------------------------------------------------------

/// Rank 1's copy of a stream: what a backend reads of it, owned here.
const Mirrored = struct {
    stream: Stream,
    eos: []u32,
    chunks: []u32,
    marks: []u32,
    mask: []u32,
    /// rank 0's media shape (its rows and features reach this rank by broadcast)
    media: ?*lanes.Media = null,
};

/// Rank 1's loop: rank 0's calls on `inner`, until a `stop` frame.
pub const Follower = struct {
    gpa: Allocator,
    inner: be.Backend,
    channel: Channel,
    streams: std.AutoHashMapUnmanaged(u64, *Mirrored) = .empty,
    arena: std.heap.ArenaAllocator,
    calls: u64 = 0,
    /// calls that failed here (rank 0 met the same failure on the same arguments)
    failed: u64 = 0,
    watch: ?*Watch = null,
    /// the lane backend's side of rank 0's notes and kept states (cuda_lanes: applyNote, savedFor)
    family: ?Family = null,

    pub const Family = struct {
        ptr: *anyopaque,
        note: *const fn (ptr: *anyopaque, bytes: []const u8) anyerror!void,
        saved: *const fn (ptr: *anyopaque, key: u64) ?*anyopaque,
        /// GLM spill v2: rank 0's ask, answered in the gather that follows (the handler reaches it whatever its own
        /// outcome: rank 0 waits there)
        ask: ?*const fn (ptr: *anyopaque, bytes: []const u8) anyerror!void = null,
    };

    pub fn init(gpa: Allocator, inner: be.Backend, channel: Channel) Follower {
        return .{ .gpa = gpa, .inner = inner, .channel = channel, .arena = .init(gpa) };
    }

    /// Releases the streams rank 0 left open.
    pub fn deinit(self: *Follower) void {
        var it = self.streams.iterator();
        while (it.next()) |kv| {
            self.inner.release(&kv.value_ptr.*.stream);
            self.free(kv.value_ptr.*);
        }
        self.streams.deinit(self.gpa);
        self.arena.deinit();
    }

    fn free(self: *Follower, m: *Mirrored) void {
        m.stream.deinit(self.gpa);
        self.gpa.free(m.eos);
        self.gpa.free(m.chunks);
        self.gpa.free(m.marks);
        self.gpa.free(m.mask);
        if (m.media) |md| self.gpa.destroy(md);
        self.gpa.destroy(m);
    }

    /// Batches until rank 0 stops (returns) or the channel fails (its error: rank 0 is gone).
    pub fn run(self: *Follower) !void {
        while (try self.channel.recv()) |bytes| {
            var calls = try Calls.of(bytes);
            while (try calls.next()) |call| try self.apply(call);
        }
    }

    /// One call's bytes. A call that fails here is logged and the loop goes on: rank 0 made the same call
    /// with the same arguments, so it failed there too and the round loop handles it.
    pub fn apply(self: *Follower, bytes: []const u8) !void {
        _ = self.arena.reset(.retain_capacity);
        const call = try decode(self.arena.allocator(), bytes);
        self.calls += 1;
        if (self.watch) |w| w.arm(w.base_s + switch (call) {
            .prefill => |p| p.prompt.len / 400,
            .prefill_many => |ps| blk: {
                var rows: usize = 0;
                for (ps) |p| rows += p.prompt.len;
                break :blk rows / 400;
            },
            else => 0,
        });
        defer if (self.watch) |w| w.disarm();
        self.make(call) catch |e| {
            self.failed += 1;
            std.log.warn("rank 1: {s} failed ({s}), as on rank 0", .{ @tagName(call), @errorName(e) });
        };
    }

    fn stream(self: *Follower, key: u64) !*Stream {
        const m = self.streams.get(key) orelse return error.UnknownStream;
        return &m.stream;
    }

    fn windows(self: *Follower, a: Allocator, ws: []const WindowCall) ![]be.Window {
        const out = try a.alloc(be.Window, ws.len);
        for (out, ws) |*o, w| o.* = .{ .stream = try self.stream(w.key), .pending = w.pending, .held = w.held, .tokens = w.tokens, .parents = w.parents, .positions = w.positions, .early = w.early, .masks = w.masks };
        return out;
    }

    /// Rank 1's own stream for rank 0's prefill (prompt, budget, sampling, eos, chunks, its resume and marks); a
    /// stream prefilled again keeps its backend lane, as on rank 0 (the backend replaces its caches).
    fn mirror(self: *Follower, p: StreamSpec) !*Stream {
        const m = try self.gpa.create(Mirrored);
        errdefer self.gpa.destroy(m);
        m.eos = try self.gpa.dupe(u32, p.eos);
        errdefer self.gpa.free(m.eos);
        m.chunks = try self.gpa.dupe(u32, p.chunks);
        errdefer self.gpa.free(m.chunks);
        m.marks = try self.gpa.dupe(u32, p.marks);
        errdefer self.gpa.free(m.marks);
        m.mask = try self.gpa.dupe(u32, p.mask);
        errdefer self.gpa.free(m.mask);
        // rank 1's own copy of the state rank 0 resumes (none held: the backend agrees to start cold)
        const saved = if (p.resume_key != 0) (if (self.family) |f| f.saved(f.ptr, p.resume_key) else null) else null;
        m.media = null;
        errdefer if (m.media) |md| self.gpa.destroy(md);
        if (p.media) |pm| {
            const md = try self.gpa.create(lanes.Media);
            md.* = .{ .rows = &.{}, .width = pm.width, .follower_rows = pm.rows };
            m.media = md;
        }
        m.stream = try Stream.init(self.gpa, .{ .id = "rank0", .prompt = p.prompt, .max_new = p.max_new, .eos = m.eos, .sampling = p.sampling, .drafts = p.drafts, .chunks = m.chunks, .reuse = .{ .saved = saved, .at = p.resume_at, .marks = m.marks }, .media = m.media });
        m.stream.first_mask = m.mask; // rank 0's grammar bits for the first token: this rank has no grammar of its own
        const gop = try self.streams.getOrPut(self.gpa, p.key);
        if (gop.found_existing) {
            const old = gop.value_ptr.*;
            self.inner.release(&old.stream);
            self.free(old);
        }
        gop.value_ptr.* = m;
        return &m.stream;
    }

    fn make(self: *Follower, call: Call) !void {
        const a = self.arena.allocator();
        const b = self.inner;
        switch (call) {
            .prefill => |p| try b.prefill(try self.mirror(p)),
            .prefill_many => |ps| {
                const f = b.vtable.prefill_many orelse return error.NoBursts;
                const ss = try a.alloc(*Stream, ps.len);
                for (ss, ps) |*x, p| x.* = try self.mirror(p);
                try f(b.ptr, ss);
            },
            .prefill_begin => |p| {
                const f = b.vtable.prefill_begin orelse return error.NoFills;
                try f(b.ptr, try self.mirror(p));
            },
            .prefill_step => |keys| {
                const f = b.vtable.fill_step orelse return error.NoFills;
                const ss = try a.alloc(*Stream, keys.len);
                for (ss, keys) |*x, k| x.* = try self.stream(k);
                const states = try a.alloc(be.FillState, keys.len);
                @memset(states, .filling);
                try f(b.ptr, ss, states);
            },
            .first => |f| _ = try b.first(try self.stream(f.key), f.position),
            .queue => |q| _ = try b.queue(try self.stream(q.key), q.feed, q.position),
            .verify => |ws| {
                const wins = try self.windows(a, ws);
                const out = try a.alloc(be.Verified, ws.len);
                for (out, ws) |*o, w| o.* = .{ .sampled = try a.alloc(u32, w.rows()), .drafts = try a.alloc(u32, w.rows() - 1) };
                try b.verify(wins, out);
            },
            .keep => |k| try b.keep(try self.windows(a, k.windows), k.paths),
            .draft => |rs| {
                const reqs = try a.alloc(be.DraftRequest, rs.len);
                for (reqs, rs) |*q, r| q.* = .{ .stream = try self.stream(r.key), .follow = r.follow, .first = r.first, .rows = r.rows, .start = r.start, .position = r.position, .depth = r.depth, .early = r.early, .ranks = r.ranks };
                try b.draft(reqs);
            },
            .unspeculate => |key| {
                const f = b.vtable.unspeculate orelse return error.NoUnspeculate;
                try f(b.ptr, try self.stream(key));
            },
            .release => |key| {
                const kv = self.streams.fetchRemove(key) orelse return;
                b.release(&kv.value.stream);
                self.free(kv.value);
            },
            .note => |bytes| {
                const f = self.family orelse return error.NoFamilyForNotes;
                try f.note(f.ptr, bytes);
            },
            .spill => |bytes| {
                const f = self.family orelse return error.NoFamilyForNotes;
                const ask = f.ask orelse return error.NoFamilyForAsks;
                try ask(f.ptr, bytes);
            },
        }
    }
};

// -- tests --------------------------------------------------------------------------------------------------------

fn roundTrip(arena: *std.heap.ArenaAllocator, call: Call) !Call {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    try encode(&list, gpa, call);
    // a frame cut anywhere is refused, never read past its end
    for (0..list.items.len) |cut| {
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        if (decode(scratch.allocator(), list.items[0..cut])) |_| return error.TestCutFrameDecoded else |_| {}
    }
    return decode(arena.allocator(), list.items);
}

test "every call survives its frame bit for bit, and a cut frame is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const ar = &arena;
    const smp: Sampling = .{ .seed = 0x7fff_0123_4567_89ab, .temperature = 0.6, .top_k = 20, .top_p = 0.95, .min_p = 0.05 };
    {
        const c = try roundTrip(ar, .{ .prefill = .{ .key = 0xdead_beef_0000_1111, .max_new = 4096, .drafts = true, .sampling = smp, .prompt = &.{ 1, 2, 3, 151643 }, .eos = &.{ 151645, 151643 }, .chunks = &.{2048} } });
        const p = c.prefill;
        try std.testing.expectEqual(@as(u64, 0xdead_beef_0000_1111), p.key);
        try std.testing.expectEqual(@as(u32, 4096), p.max_new);
        try std.testing.expect(p.drafts);
        try std.testing.expectEqual(smp, p.sampling.?);
        try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 151643 }, p.prompt);
        try std.testing.expectEqualSlices(u32, &.{ 151645, 151643 }, p.eos);
        try std.testing.expectEqualSlices(u32, &.{2048}, p.chunks);
        try std.testing.expectEqual(@as(u64, 0), p.resume_key);
        const r = try roundTrip(ar, .{ .prefill = .{ .key = 2, .max_new = 1, .drafts = true, .sampling = null, .prompt = &.{ 1, 2, 3 }, .eos = &.{}, .chunks = &.{}, .resume_key = 0xabcdef, .resume_at = 30000, .marks = &.{ 1500, 31000 } } });
        try std.testing.expectEqual(@as(u64, 0xabcdef), r.prefill.resume_key);
        try std.testing.expectEqual(@as(u32, 30000), r.prefill.resume_at);
        try std.testing.expectEqualSlices(u32, &.{ 1500, 31000 }, r.prefill.marks);
        const n = try roundTrip(ar, .{ .note = &.{ 1, 0, 0, 0, 0 } });
        try std.testing.expectEqualSlices(u8, &.{ 1, 0, 0, 0, 0 }, n.note);
        const q = try roundTrip(ar, .{ .spill = &.{ 16, 9, 9 } });
        try std.testing.expectEqualSlices(u8, &.{ 16, 9, 9 }, q.spill);
    }
    {
        const c = try roundTrip(ar, .{ .prefill = .{ .key = 1, .max_new = 0, .drafts = false, .sampling = null, .prompt = &.{}, .eos = &.{}, .chunks = &.{} } });
        try std.testing.expect(c.prefill.sampling == null and !c.prefill.drafts and c.prefill.prompt.len == 0);
    }
    {
        const specs = [_]StreamSpec{
            .{ .key = 5, .max_new = 8, .drafts = true, .sampling = null, .prompt = &.{ 1, 2, 3 }, .eos = &.{9}, .chunks = &.{} },
            .{ .key = 6, .max_new = 9, .drafts = false, .sampling = null, .prompt = &.{ 4, 5 }, .eos = &.{}, .chunks = &.{} },
        };
        const c = try roundTrip(ar, .{ .prefill_many = &specs });
        try std.testing.expectEqual(@as(usize, 2), c.prefill_many.len);
        try std.testing.expectEqual(@as(u64, 6), c.prefill_many[1].key);
        try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, c.prefill_many[0].prompt);
        try std.testing.expectEqualSlices(u32, &.{ 4, 5 }, c.prefill_many[1].prompt);
    }
    {
        const c = try roundTrip(ar, .{ .first = .{ .key = 9, .position = 1 << 40 } });
        try std.testing.expectEqual(@as(u64, 9), c.first.key);
        try std.testing.expectEqual(@as(u64, 1 << 40), c.first.position);
    }
    {
        const c = try roundTrip(ar, .{ .queue = .{ .key = 3, .feed = .{ .handle = 77 }, .position = 12 } });
        try std.testing.expectEqual(be.Feed{ .handle = 77 }, c.queue.feed);
        const d = try roundTrip(ar, .{ .queue = .{ .key = 3, .feed = .{ .value = 151644 }, .position = 12 } });
        try std.testing.expectEqual(be.Feed{ .value = 151644 }, d.queue.feed);
    }
    const ws = [_]WindowCall{
        .{ .key = 5, .pending = 42, .held = 6, .tokens = &.{}, .parents = null, .positions = &.{ 100, 101, 102, 103, 104, 105, 106 }, .early = false },
        .{ .key = 6, .pending = 7, .held = 0, .tokens = &.{ 8, 9 }, .parents = &.{ -1, 0, 0 }, .positions = &.{ 10, 11, 11 }, .early = true },
    };
    {
        const c = try roundTrip(ar, .{ .verify = &ws });
        try std.testing.expectEqual(@as(usize, 2), c.verify.len);
        for (c.verify, ws) |g, w| {
            try std.testing.expectEqual(w.key, g.key);
            try std.testing.expectEqual(w.pending, g.pending);
            try std.testing.expectEqual(w.held, g.held);
            try std.testing.expectEqual(w.early, g.early);
            try std.testing.expectEqual(w.rows(), g.rows());
            try std.testing.expectEqualSlices(u32, w.tokens, g.tokens);
            try std.testing.expectEqualSlices(u64, w.positions, g.positions);
            try std.testing.expectEqual(w.parents == null, g.parents == null);
            if (w.parents) |p| try std.testing.expectEqualSlices(i32, p, g.parents.?);
        }
    }
    {
        const c = try roundTrip(ar, .{ .keep = .{ .windows = ws[0..1], .paths = &.{&.{ 0, 1, 2 }} } });
        try std.testing.expectEqual(@as(usize, 1), c.keep.windows.len);
        try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, c.keep.paths[0]);
    }
    {
        const rs = [_]DraftCall{
            .{ .key = 5, .follow = &.{}, .first = .{ .handle = 3 }, .rows = null, .start = 99, .position = 100, .depth = 15, .early = false, .ranks = false },
            .{ .key = 5, .follow = &.{ 4, 5, 6 }, .first = null, .rows = &.{ 0, 1, 2 }, .start = 100, .position = 104, .depth = 0, .early = true, .ranks = true },
        };
        const c = try roundTrip(ar, .{ .draft = &rs });
        try std.testing.expectEqual(@as(usize, 2), c.draft.len);
        try std.testing.expectEqual(be.Feed{ .handle = 3 }, c.draft[0].first.?);
        try std.testing.expect(c.draft[0].rows == null and c.draft[0].follow.len == 0);
        try std.testing.expectEqual(@as(u32, 15), c.draft[0].depth);
        try std.testing.expect(c.draft[1].first == null);
        try std.testing.expectEqualSlices(u32, &.{ 4, 5, 6 }, c.draft[1].follow);
        try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, c.draft[1].rows.?);
        try std.testing.expectEqual(@as(u64, 104), c.draft[1].position);
        try std.testing.expect(c.draft[1].early and c.draft[1].ranks);
    }
    {
        const c = try roundTrip(ar, .{ .release = 0xffff_ffff_ffff_fff0 });
        try std.testing.expectEqual(@as(u64, 0xffff_ffff_ffff_fff0), c.release);
        const d = try roundTrip(ar, .{ .unspeculate = 2 });
        try std.testing.expectEqual(@as(u64, 2), d.unspeculate);
    }
}

test "the frame layout is the documented one" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    try encode(&list, gpa, .{ .first = .{ .key = 0x0102030405060708, .position = 5 } });
    try std.testing.expectEqualSlices(u8, &.{ 2, 8, 7, 6, 5, 4, 3, 2, 1, 5, 0, 0, 0, 0, 0, 0, 0 }, list.items);
    list.clearRetainingCapacity();
    const w = [_]WindowCall{.{ .key = 1, .pending = 2, .held = 0, .tokens = &.{3}, .parents = null, .positions = &.{ 4, 5 }, .early = false }};
    try encode(&list, gpa, .{ .verify = &w });
    try std.testing.expectEqualSlices(u8, &(.{ 4, 1, 0, 0, 0 } ++ .{ 1, 0, 0, 0, 0, 0, 0, 0 } ++ .{ 2, 0, 0, 0 } ++ .{ 0, 0, 0, 0 } ++
        .{ 1, 0, 0, 0, 3, 0, 0, 0 } ++ .{0} ++ .{ 2, 0, 0, 0 } ++ .{ 4, 0, 0, 0, 0, 0, 0, 0 } ++ .{ 5, 0, 0, 0, 0, 0, 0, 0 } ++ .{0} ++ .{ 0, 0, 0, 0 }), list.items);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    try std.testing.expectError(error.UnknownOp, decode(arena.allocator(), &.{99}));
    try std.testing.expectError(error.TrailingBytes, decode(arena.allocator(), &.{ 8, 1, 0, 0, 0, 0, 0, 0, 0, 0 }));
    try std.testing.expectError(error.BadFrame, decode(arena.allocator(), &.{ 3, 1, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }));
}

/// A channel that keeps what rank 0 sends and logs its calls in order: `s` a batch, `i` idle, `t` stop.
const Spy = struct {
    gpa: Allocator,
    sent: std.ArrayList([]u8) = .empty,
    log: std.ArrayList(u8) = .empty,

    fn channel(s: *Spy) Channel {
        return .{ .ptr = s, .vtable = &.{ .send = sendFn, .recv = recvFn, .idle = idleFn, .stop = stopFn } };
    }
    fn of(p: *anyopaque) *Spy {
        return @ptrCast(@alignCast(p));
    }
    fn sendFn(p: *anyopaque, bytes: []const u8) anyerror!void {
        const s = of(p);
        try s.sent.append(s.gpa, try s.gpa.dupe(u8, bytes));
        try s.log.append(s.gpa, 's');
    }
    fn recvFn(_: *anyopaque) anyerror!?[]const u8 {
        return error.NotUsed;
    }
    fn idleFn(p: *anyopaque) anyerror!void {
        try of(p).log.append(of(p).gpa, 'i');
    }
    fn stopFn(p: *anyopaque) anyerror!void {
        try of(p).log.append(of(p).gpa, 't');
    }
    fn deinit(s: *Spy) void {
        for (s.sent.items) |b| s.gpa.free(b);
        s.sent.deinit(s.gpa);
        s.log.deinit(s.gpa);
    }
};

/// Rank 1's lane backend side of notes and asks, recorded in order (`n` a note, `a` an ask, then its bytes).
const FamilyLog = struct {
    gpa: Allocator,
    got: std.ArrayList(u8) = .empty,

    fn family(f: *FamilyLog, asks: bool) Follower.Family {
        return .{ .ptr = f, .note = noteFn, .saved = savedFn, .ask = if (asks) askFn else null };
    }
    fn noteFn(p: *anyopaque, bytes: []const u8) anyerror!void {
        const f: *FamilyLog = @ptrCast(@alignCast(p));
        try f.got.append(f.gpa, 'n');
        try f.got.appendSlice(f.gpa, bytes);
    }
    fn askFn(p: *anyopaque, bytes: []const u8) anyerror!void {
        const f: *FamilyLog = @ptrCast(@alignCast(p));
        try f.got.append(f.gpa, 'a');
        try f.got.appendSlice(f.gpa, bytes);
    }
    fn savedFn(_: *anyopaque, _: u64) ?*anyopaque {
        return null;
    }
};

test "GLM spill v2: an ask goes at once with the notes before it, a flush sends notes without a collective and leaves an idle rank 1 idle, and rank 1 hands both to its lane backend in order" {
    const gpa = std.testing.allocator;
    var spy: Spy = .{ .gpa = gpa };
    defer spy.deinit();
    var rank0: fake.Fake = .{ .gpa = gpa };
    defer rank0.deinit();
    var leader = Leader.init(gpa, rank0.backend(), spy.channel());
    defer leader.deinit();
    const n = leader.notes();
    // 1. no stream open: a note waits in the batch; the flush sends it and rank 1 goes back to its idle wait
    n.send(n.ptr, "a");
    try std.testing.expectEqualStrings("", spy.log.items);
    n.flush.?(n.ptr);
    try std.testing.expectEqualStrings("si", spy.log.items);
    // 2. an ask goes at once with the note before it (one batch), and the flush after its gather idles rank 1 again
    n.send(n.ptr, "b");
    try n.ask.?(n.ptr, "q");
    try std.testing.expectEqualStrings("sis", spy.log.items);
    n.flush.?(n.ptr); // nothing waits: no batch, only the idle
    try std.testing.expectEqualStrings("sisi", spy.log.items);
    // 3. with a stream open the flush sends the notes and leaves rank 1 on the wire
    const prompt = [_]u32{ 3, 1, 4, 1, 5 };
    var s = try Stream.init(gpa, .{ .id = "s", .prompt = &prompt, .max_new = 4 });
    defer s.deinit(gpa);
    const b = leader.backend();
    try b.vtable.prefill(b.ptr, &s);
    n.send(n.ptr, "c");
    n.flush.?(n.ptr);
    try std.testing.expectEqualStrings("sisiss", spy.log.items); // the prefill's batch, then the note's: no idle
    b.vtable.release(b.ptr, &s); // the last stream: its release goes, then idle
    try std.testing.expectEqualStrings("sisisssi", spy.log.items);
    // rank 1: the batches in order hand the notes and the ask to its lane backend, the prefill to its engine
    var rank1: fake.Fake = .{ .gpa = gpa };
    defer rank1.deinit();
    var follower = Follower.init(gpa, rank1.backend(), spy.channel());
    defer follower.deinit();
    var log: FamilyLog = .{ .gpa = gpa };
    defer log.got.deinit(gpa);
    follower.family = log.family(true);
    for (spy.sent.items) |batch| {
        var calls = try Calls.of(batch);
        while (try calls.next()) |call| try follower.apply(call);
    }
    try std.testing.expectEqualStrings("nanbaqnc", log.got.items);
    try std.testing.expectEqual(@as(u64, 0), follower.failed);
    try std.testing.expectEqual(leader.calls, follower.calls);
    // a lane backend without asks: the ask fails there (logged; rank 0's gather then meets the watchdog)
    var bare = Follower.init(gpa, rank1.backend(), spy.channel());
    defer bare.deinit();
    var bare_log: FamilyLog = .{ .gpa = gpa };
    defer bare_log.got.deinit(gpa);
    bare.family = bare_log.family(false);
    var calls = try Calls.of(spy.sent.items[1]);
    while (try calls.next()) |call| try bare.apply(call);
    try std.testing.expectEqual(@as(u64, 1), bare.failed);
    try std.testing.expectEqualStrings("nb", bare_log.got.items);
}

const fake = lanes.fake;
const net = std.Io.net;
const link_mod = @import("cuda_link.zig");

fn testConfig(gpa: Allocator) !lanes.Config {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    return lanes.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .max_streams = 1 }, 16, 15);
}

const Case = struct { prompt: []const u32, max_new: u32, sampling: ?Sampling = null, drafts: bool = true };

/// The cases one after another on `backend` through the round loop; each stream's emitted tokens.
fn drive(gpa: Allocator, backend: be.Backend, cases: []const Case) ![][]u32 {
    var cfg = try testConfig(gpa);
    defer cfg.deinit(gpa);
    var clock: fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, backend, clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(Stream, cases.len);
    defer gpa.free(streams);
    for (cases, streams) |c, *s| s.* = try Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts });
    defer for (streams) |*s| s.deinit(gpa);
    // two streams at once (rounds alternate between them), then the rest
    for (streams[0..@min(2, streams.len)]) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    for (streams[@min(2, streams.len)..]) |*s| {
        try engine.addStream(s);
        while (engine.activeCount() > 0) try engine.step();
    }
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

fn freeRuns(gpa: Allocator, runs: [][]u32) void {
    for (runs) |r| gpa.free(r);
    gpa.free(runs);
}

const Rank1 = struct {
    target: fake.Fake,
    err: ?anyerror = null,
    calls: u64 = 0,
    failed: u64 = 0,
    sliced: bool = false,

    fn run(r: *Rank1, io: std.Io, port: u16) void {
        const gpa = std.testing.allocator;
        const l = Link.join(io, net.IpAddress.parse("127.0.0.1", port) catch unreachable, 5000) catch |e| {
            r.err = e;
            return;
        };
        defer l.close();
        var frames: Frames = .{ .gpa = gpa, .link = l };
        defer frames.deinit();
        var f = Follower.init(gpa, if (r.sliced) r.target.backendSliced() else r.target.backend(), frames.channel());
        defer f.deinit();
        f.run() catch |e| {
            r.err = e;
        };
        r.calls = f.calls;
        r.failed = f.failed;
    }
};

test "rank 1 makes rank 0's calls over a local link: the same draws, rounds and caches" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const cases = [_]Case{
        .{ .prompt = &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, .max_new = 60 },
        .{ .prompt = &.{ 2, 7, 1, 8 }, .max_new = 45, .sampling = .{ .seed = 11, .temperature = 0.8 } },
        .{ .prompt = &.{ 1, 1, 2, 3, 5, 8 }, .max_new = 30, .drafts = false },
        .{ .prompt = &.{ 9, 9 }, .max_new = 25, .sampling = .{ .seed = 5 } },
    };
    // the same cases on one rank, no link
    var alone: fake.Fake = .{ .gpa = gpa };
    defer alone.deinit();
    const want = try drive(gpa, alone.backend(), &cases);
    defer freeRuns(gpa, want);

    const server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    var rank1: Rank1 = .{ .target = .{ .gpa = gpa } };
    defer rank1.target.deinit();
    const t = try std.Thread.spawn(.{}, Rank1.run, .{ &rank1, io, server.port() });
    var rank0: fake.Fake = .{ .gpa = gpa };
    defer rank0.deinit();
    const l = try server.accept(5000);
    defer l.close();
    var frames: Frames = .{ .gpa = gpa, .link = l };
    var leader = Leader.init(gpa, rank0.backend(), frames.channel());
    const got = drive(gpa, leader.backend(), &cases) catch |e| {
        leader.deinit();
        t.join();
        return e;
    };
    defer freeRuns(gpa, got);
    leader.deinit(); // sends stop
    t.join();
    if (rank1.err) |e| return e;
    for (want, got) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    // rank 1 drew every token rank 0 drew, in order, and ran as many forwards
    try std.testing.expectEqualSlices(u32, rank0.drawn.items, rank1.target.drawn.items);
    try std.testing.expectEqual(rank0.rounds, rank1.target.rounds);
    try std.testing.expectEqual(rank0.prefill_count, rank1.target.prefill_count);
    try std.testing.expect(rank1.calls > cases.len * 3);
    try std.testing.expectEqual(@as(u64, 0), rank1.failed);
    // every stream was released on both ranks
    try std.testing.expectEqual(@as(u32, 0), rank0.lanes.count());
    try std.testing.expectEqual(@as(u32, 0), rank1.target.lanes.count());
}

test "rank 0's calls reach three followers over a fan-out link: every rank draws and rounds alike (TP=4)" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const cases = [_]Case{
        .{ .prompt = &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, .max_new = 60 },
        .{ .prompt = &.{ 2, 7, 1, 8 }, .max_new = 45, .sampling = .{ .seed = 11, .temperature = 0.8 } },
        .{ .prompt = &.{ 9, 9 }, .max_new = 25, .sampling = .{ .seed = 5 } },
    };
    var alone: fake.Fake = .{ .gpa = gpa };
    defer alone.deinit();
    const want = try drive(gpa, alone.backend(), &cases);
    defer freeRuns(gpa, want);

    const server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    var followers: [3]Rank1 = undefined;
    for (&followers) |*r| r.* = .{ .target = .{ .gpa = gpa } };
    defer for (&followers) |*r| r.target.deinit();
    var threads: [3]std.Thread = undefined;
    for (&threads, &followers) |*t, *r| t.* = try std.Thread.spawn(.{}, Rank1.run, .{ r, io, server.port() });
    var rank0: fake.Fake = .{ .gpa = gpa };
    defer rank0.deinit();
    const first = try server.accept(5000);
    var fan: Link = .{ .fd = first.fd };
    for (&fan.more) |*m| m.* = (try server.accept(5000)).fd;
    defer fan.close();
    var all: [3]std.posix.socket_t = undefined;
    try std.testing.expectEqual(@as(usize, 3), fan.sockets(&all).len);
    var frames: Frames = .{ .gpa = gpa, .link = fan };
    var leader = Leader.init(gpa, rank0.backend(), frames.channel());
    const got = drive(gpa, leader.backend(), &cases) catch |e| {
        leader.deinit();
        for (threads) |t| t.join();
        return e;
    };
    defer freeRuns(gpa, got);
    leader.deinit();
    for (threads) |t| t.join();
    for (want, got) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    for (&followers) |*r| {
        if (r.err) |e| return e;
        try std.testing.expectEqualSlices(u32, rank0.drawn.items, r.target.drawn.items);
        try std.testing.expectEqual(rank0.rounds, r.target.rounds);
        try std.testing.expectEqual(rank0.prefill_count, r.target.prefill_count);
        try std.testing.expectEqual(@as(u64, 0), r.failed);
        try std.testing.expectEqual(@as(u32, 0), r.target.lanes.count());
    }
}

/// The records' wire in the host tests: a second local link carrying each record as a frame.
const LinkWire = struct {
    link: Link,

    fn wire(w: *LinkWire) Wire {
        return .{ .ptr = w, .put = put, .get = get };
    }
    fn put(p: *anyopaque, record: []const u8) anyerror!void {
        const w: *LinkWire = @ptrCast(@alignCast(p));
        try w.link.send(.ack, record);
    }
    fn get(p: *anyopaque, record: []u8) anyerror!void {
        const w: *LinkWire = @ptrCast(@alignCast(p));
        const b = try w.link.expect(std.testing.allocator, .ack);
        defer std.testing.allocator.free(b);
        if (b.len != record_bytes) return error.TestBadRecord;
        @memcpy(record[0..record_bytes], b);
    }
};

const Rank1Records = struct {
    target: fake.Fake,
    /// set SO_RCVBUF on the payload link (the prompt's), to bound how much the kernel buffers for rank 1
    rcvbuf: ?c_int = null,
    err: ?anyerror = null,
    calls: u64 = 0,
    failed: u64 = 0,

    fn run(r: *Rank1Records, io: std.Io, port: u16, wire_port: u16) void {
        const gpa = std.testing.allocator;
        const l = Link.join(io, net.IpAddress.parse("127.0.0.1", port) catch unreachable, 5000) catch |e| {
            r.err = e;
            return;
        };
        defer l.close();
        if (r.rcvbuf) |n| std.posix.setsockopt(l.fd, std.posix.SOL.SOCKET, std.posix.SO.RCVBUF, std.mem.asBytes(&n)) catch |e| {
            r.err = e;
            return;
        };
        const wl = Link.join(io, net.IpAddress.parse("127.0.0.1", wire_port) catch unreachable, 5000) catch |e| {
            r.err = e;
            return;
        };
        defer wl.close();
        var lw: LinkWire = .{ .link = wl };
        var records: Records = .{ .gpa = gpa, .wire = lw.wire(), .link = l };
        defer records.deinit();
        var f = Follower.init(gpa, r.target.backend(), records.channel());
        defer f.deinit();
        f.run() catch |e| {
            r.err = e;
        };
        r.calls = f.calls;
        r.failed = f.failed;
    }
};

test "the mirror over records: batched rounds, idle waits on the link, long prompts on the link, stop when idle" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var long: [1500]u32 = undefined;
    for (&long, 0..) |*x, i| x.* = @intCast((i * 7) % 90);
    const cases = [_]Case{
        .{ .prompt = &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, .max_new = 60 },
        .{ .prompt = &long, .max_new = 30, .sampling = .{ .seed = 3, .temperature = 0.9 } },
        .{ .prompt = &.{ 1, 1, 2, 3, 5, 8 }, .max_new = 30, .drafts = false },
        .{ .prompt = &.{ 9, 9 }, .max_new = 25, .sampling = .{ .seed = 5 } },
    };
    var alone: fake.Fake = .{ .gpa = gpa };
    defer alone.deinit();
    const want = try drive(gpa, alone.backend(), &cases);
    defer freeRuns(gpa, want);

    const server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const wire_server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer wire_server.close();
    var rank1: Rank1Records = .{ .target = .{ .gpa = gpa } };
    defer rank1.target.deinit();
    const t = try std.Thread.spawn(.{}, Rank1Records.run, .{ &rank1, io, server.port(), wire_server.port() });
    var rank0: fake.Fake = .{ .gpa = gpa };
    defer rank0.deinit();
    const l = try server.accept(5000);
    defer l.close();
    const wl = try wire_server.accept(5000);
    defer wl.close();
    var lw: LinkWire = .{ .link = wl };
    var records: Records = .{ .gpa = gpa, .wire = lw.wire(), .link = l };
    defer records.deinit();
    var leader = Leader.init(gpa, rank0.backend(), records.channel());
    // drive() runs two streams together, then the rest one by one: rank 0 goes idle between them
    const got = drive(gpa, leader.backend(), &cases) catch |e| {
        leader.deinit();
        t.join();
        return e;
    };
    defer freeRuns(gpa, got);
    try std.testing.expect(records.idle); // every stream released
    try std.testing.expect(leader.sends < leader.calls); // keep, first and release rode with the next call
    leader.deinit(); // stop while idle: a stop frame on the link
    t.join();
    if (rank1.err) |e| return e;
    for (want, got) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    try std.testing.expectEqualSlices(u32, rank0.drawn.items, rank1.target.drawn.items);
    try std.testing.expectEqual(rank0.rounds, rank1.target.rounds);
    try std.testing.expectEqual(leader.calls, rank1.calls);
    try std.testing.expectEqual(@as(u64, 0), rank1.failed);
    try std.testing.expectEqual(@as(u32, 0), rank1.target.lanes.count());
}

test "a prompt larger than the sockets can buffer reaches rank 1: the record goes before the long batch" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    // 4,000,000 ids = 16 MB on the link. Both ends of that link get 64 KiB socket buffers (checked below), so with the
    // batch sent before its record rank 0 blocks in the write while rank 1 waits for the record, on any host.
    const huge = try gpa.alloc(u32, 4_000_000);
    defer gpa.free(huge);
    for (huge, 0..) |*x, i| x.* = @intCast((i * 13) % 97);
    const cases = [_]Case{
        .{ .prompt = huge, .max_new = 4 },
        .{ .prompt = &.{ 2, 7, 1, 8 }, .max_new = 6 },
    };
    var alone: fake.Fake = .{ .gpa = gpa };
    defer alone.deinit();
    const want = try drive(gpa, alone.backend(), &cases);
    defer freeRuns(gpa, want);

    const server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const wire_server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer wire_server.close();
    const small: c_int = 64 * 1024;
    var rank1: Rank1Records = .{ .target = .{ .gpa = gpa }, .rcvbuf = small };
    defer rank1.target.deinit();
    const t = try std.Thread.spawn(.{}, Rank1Records.run, .{ &rank1, io, server.port(), wire_server.port() });
    var rank0: fake.Fake = .{ .gpa = gpa };
    defer rank0.deinit();
    const l = try server.accept(5000);
    defer l.close();
    try std.posix.setsockopt(l.fd, std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, std.mem.asBytes(&small));
    var got_buf: c_int = 0;
    var got_len: std.posix.socklen_t = @sizeOf(c_int);
    try std.testing.expectEqual(@as(c_int, 0), std.c.getsockopt(l.fd, std.posix.SOL.SOCKET, std.posix.SO.SNDBUF, @ptrCast(&got_buf), &got_len));
    try std.testing.expect(got_buf > 0 and got_buf <= 1024 * 1024); // the kernel doubles the request; far below 16 MB
    const wl = try wire_server.accept(5000);
    defer wl.close();
    // bounded: a stuck transfer is cut after 180 s (both rank 0 sockets shut down), so the test fails instead of hanging
    const StuckCut = struct {
        done: std.atomic.Value(bool) = .init(false),
        fired: std.atomic.Value(bool) = .init(false),
        fn run(w: *@This(), wio: std.Io, fds: [2]std.posix.socket_t) void {
            var waited: u32 = 0;
            while (waited < 180_000 and !w.done.load(.acquire)) : (waited += 100) std.Io.sleep(wio, .fromMilliseconds(100), .awake) catch {};
            if (w.done.load(.acquire)) return;
            w.fired.store(true, .release);
            for (fds) |fd| _ = std.posix.system.shutdown(fd, std.posix.SHUT.RDWR);
        }
    };
    var watch: StuckCut = .{};
    const wt = try std.Thread.spawn(.{}, StuckCut.run, .{ &watch, io, [2]std.posix.socket_t{ l.fd, wl.fd } });
    var lw: LinkWire = .{ .link = wl };
    var records: Records = .{ .gpa = gpa, .wire = lw.wire(), .link = l };
    defer records.deinit();
    var leader = Leader.init(gpa, rank0.backend(), records.channel());
    const result = drive(gpa, leader.backend(), &cases);
    watch.done.store(true, .release);
    wt.join();
    const got = result catch |e| {
        leader.deinit();
        t.join();
        return e;
    };
    defer freeRuns(gpa, got);
    leader.deinit();
    t.join();
    try std.testing.expect(!watch.fired.load(.acquire));
    if (rank1.err) |e| return e;
    for (want, got) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    try std.testing.expectEqualSlices(u32, rank0.drawn.items, rank1.target.drawn.items);
    try std.testing.expectEqual(leader.calls, rank1.calls);
    try std.testing.expectEqual(@as(u64, 0), rank1.failed);
}

test "rank 0 closing the link while no stream is open stops rank 1 cleanly" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const wire_server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer wire_server.close();
    var rank1: Rank1Records = .{ .target = .{ .gpa = gpa } };
    defer rank1.target.deinit();
    const t = try std.Thread.spawn(.{}, Rank1Records.run, .{ &rank1, io, server.port(), wire_server.port() });
    const l = try server.accept(5000);
    const wl = try wire_server.accept(5000);
    defer wl.close();
    l.close(); // no stop frame: a server stopped by SIGTERM
    t.join();
    try std.testing.expectEqual(@as(?anyerror, null), rank1.err);
}

const Lost = struct {
    var why: ?[]const u8 = null;
    fn record(_: *Watch, reason: []const u8) void {
        why = reason;
    }
};

test "the watchdog sees the other rank close the link, and a call past its deadline" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const peer = try Link.join(io, try net.IpAddress.parse("127.0.0.1", server.port()), 5000);
    const l = try server.accept(5000);
    defer l.close();
    Lost.why = null;
    var w: Watch = .{ .io = io, .fd = l.fd, .rank = 0, .lost = Lost.record };
    try w.start();
    std.Io.sleep(io, .fromMilliseconds(300), .awake) catch {};
    try std.testing.expect(Lost.why == null); // an open, quiet link is fine
    peer.close();
    var waited: u32 = 0;
    while (Lost.why == null and waited < 50) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {};
    w.deinit();
    try std.testing.expect(std.mem.indexOf(u8, Lost.why orelse "", "closed") != null);

    const peer2 = try Link.join(io, try net.IpAddress.parse("127.0.0.1", server.port()), 5000);
    defer peer2.close();
    const l2 = try server.accept(5000);
    defer l2.close();
    Lost.why = null;
    var w2: Watch = .{ .io = io, .fd = l2.fd, .rank = 1, .lost = Lost.record, .base_s = 0 };
    try w2.start();
    w2.base_s = 0; // start() may read TF_TP_TIMEOUT_S
    w2.arm(0);
    waited = 0;
    while (Lost.why == null and waited < 50) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {};
    w2.deinit();
    try std.testing.expect(std.mem.indexOf(u8, Lost.why orelse "", "deadline") != null);
}

test "a batch holds its calls in order and refuses a cut one" {
    const gpa = std.testing.allocator;
    var b: Batch = .{};
    defer b.deinit(gpa);
    try b.add(gpa, .{ .first = .{ .key = 1, .position = 2 } });
    try b.add(gpa, .{ .release = 7 });
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var calls = try Calls.of(b.bytes());
    const c1 = try decode(arena.allocator(), (try calls.next()).?);
    const c2 = try decode(arena.allocator(), (try calls.next()).?);
    try std.testing.expect((try calls.next()) == null);
    try std.testing.expectEqual(@as(u64, 2), c1.first.position);
    try std.testing.expectEqual(@as(u64, 7), c2.release);
    var cut = try Calls.of(b.bytes()[0 .. b.bytes().len - 1]);
    _ = try cut.next();
    try std.testing.expectError(error.ShortBatch, cut.next());
    b.clear();
    try std.testing.expectEqual(@as(usize, 0), b.bytes().len);
}

test "a record holds a short call inline and sends a long one to the link" {
    var rec: [record_bytes]u8 = undefined;
    try std.testing.expectEqual(Kind.inline_call, packRecord(&rec, .inline_call, "abc"));
    const r = try unpackRecord(&rec);
    try std.testing.expectEqual(Kind.inline_call, r.kind);
    try std.testing.expectEqualStrings("abc", r.bytes);
    try std.testing.expectEqualSlices(u8, &.{ 3, 0, 0, 0, 0, 0, 0, 0, 'a', 'b', 'c' }, rec[0..11]);
    const long: [record_bytes - record_head + 1]u8 = @splat(7);
    try std.testing.expectEqual(Kind.on_link, packRecord(&rec, .inline_call, &long));
    const l = try unpackRecord(&rec);
    try std.testing.expectEqual(Kind.on_link, l.kind);
    try std.testing.expectEqual(@as(u32, long.len), l.len);
    const fits: [record_bytes - record_head]u8 = @splat(7);
    try std.testing.expectEqual(Kind.inline_call, packRecord(&rec, .inline_call, &fits));
    try std.testing.expectEqual(Kind.stop, packRecord(&rec, .stop, &.{}));
    try std.testing.expectEqual(Kind.stop, (try unpackRecord(&rec)).kind);
    rec[4] = 9;
    try std.testing.expectError(error.BadRecord, unpackRecord(&rec));
    try std.testing.expectError(error.ShortRecord, unpackRecord(rec[0..100]));
}

test "rank 1 ends with an error when rank 0 goes away without a stop" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    var rank1: Rank1 = .{ .target = .{ .gpa = gpa } };
    defer rank1.target.deinit();
    const t = try std.Thread.spawn(.{}, Rank1.run, .{ &rank1, io, server.port() });
    const l = try server.accept(5000);
    l.close();
    t.join();
    try std.testing.expectEqual(@as(?anyerror, error.PeerClosed), rank1.err);
}

test "prompts filling between rounds are mirrored: rank 1 fills the same slices and draws the same tokens" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const prompts = [_][]const u32{ &.{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 }, &.{ 2, 7, 1, 8 }, &.{ 1, 1, 2, 3, 5, 8, 13, 21 } };
    const Drive = struct {
        fn run(backend: be.Backend) ![3][]u32 {
            var cfg = try testConfig(gpa);
            defer cfg.deinit(gpa);
            var clock: fake.FixedClock = .{};
            var engine = lanes.Engine.init(gpa, &cfg, backend, clock.clock());
            defer engine.deinit();
            var streams: [3]Stream = undefined;
            for (&streams, prompts, 0..) |*s, p, i| s.* = try Stream.init(gpa, .{ .id = "s", .prompt = p, .max_new = 20 + 5 * @as(u32, @intCast(i)), .eos = &.{96} });
            defer for (&streams) |*s| s.deinit(gpa);
            // one fills alone, then the others arrive two steps apart while it decodes
            _ = try engine.beginStream(&streams[0]);
            var next: usize = 1;
            var steps: usize = 0;
            while (engine.activeCount() > 0 or engine.fillingCount() > 0 or next < 3) : (steps += 1) {
                if (next < 3 and steps % 2 == 1) {
                    _ = try engine.beginStream(&streams[next]);
                    next += 1;
                }
                try engine.step();
            }
            var out: [3][]u32 = undefined;
            for (&out, &streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
            return out;
        }
    };
    var alone: fake.Fake = .{ .gpa = gpa, .slice = 3 };
    defer alone.deinit();
    const want = try Drive.run(alone.backend());
    defer for (want) |w| gpa.free(w);

    const server = try link_mod.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    var rank1: Rank1 = .{ .target = .{ .gpa = gpa, .slice = 3 }, .sliced = true };
    defer rank1.target.deinit();
    const t = try std.Thread.spawn(.{}, Rank1.run, .{ &rank1, io, server.port() });
    var rank0: fake.Fake = .{ .gpa = gpa, .slice = 3 };
    defer rank0.deinit();
    const l = try server.accept(5000);
    defer l.close();
    var frames: Frames = .{ .gpa = gpa, .link = l };
    var leader = Leader.init(gpa, rank0.backendSliced(), frames.channel());
    const got = Drive.run(leader.backend()) catch |e| {
        leader.deinit();
        t.join();
        return e;
    };
    defer for (got) |g| gpa.free(g);
    leader.deinit();
    t.join();
    if (rank1.err) |e| return e;
    for (want, got) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    try std.testing.expect(rank0.fill_steps > 0);
    try std.testing.expectEqual(rank0.fill_steps, rank1.target.fill_steps);
    try std.testing.expectEqualSlices(u32, rank0.drawn.items, rank1.target.drawn.items);
    try std.testing.expectEqual(@as(u64, 0), rank1.failed);
}

test "structured output's bits ride with a prompt and with a window" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const spec: StreamSpec = .{ .key = 4, .max_new = 8, .drafts = true, .sampling = null, .prompt = &.{ 1, 2 }, .eos = &.{3}, .chunks = &.{}, .mask = &.{ 0xFFFF0000, 7 } };
    try encode(&list, gpa, .{ .prefill = spec });
    const back = try decode(arena.allocator(), list.items);
    try std.testing.expectEqualSlices(u32, &.{ 0xFFFF0000, 7 }, back.prefill.mask);
    list.clearRetainingCapacity();
    const w = [_]WindowCall{.{ .key = 1, .pending = 2, .held = 0, .tokens = &.{3}, .parents = null, .positions = &.{ 4, 5 }, .early = false, .masks = &.{ 1, 2, 3, 4 } }};
    try encode(&list, gpa, .{ .verify = &w });
    const win = try decode(arena.allocator(), list.items);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, win.verify[0].masks);
}

test "Coll wraps any communicator type with world, rank, allGather and broadcast" {
    const Other = struct {
        rank: u32,
        world: u32,
        seen: *u64,
        pub fn allGather(c: *const @This(), send: u64, recv: u64, count: usize, dtype: cuda.nccl.DataType, _: cuda.abi.Stream) !void {
            if (dtype != .f32) return error.WrongType;
            c.seen.* += send + recv + count;
        }
        pub fn broadcast(c: *const @This(), buf: u64, bytes: usize, root: u32, _: cuda.abi.Stream) !void {
            c.seen.* += buf + bytes + root * 1000;
        }
    };
    var seen: u64 = 0;
    const o: Other = .{ .rank = 2, .world = 4, .seen = &seen };
    const c = Coll.of(Other, &o);
    try std.testing.expectEqual(@as(u32, 4), c.world);
    try std.testing.expectEqual(@as(u32, 2), c.rank);
    try c.allGather(1, 2, 3, .f32, null);
    try std.testing.expectEqual(@as(u64, 6), seen);
    try c.broadcast(10, 20, 1, null);
    try std.testing.expectEqual(@as(u64, 1036), seen);
    try std.testing.expectError(error.WrongType, c.allGather(1, 2, 3, .i32, null));
}
