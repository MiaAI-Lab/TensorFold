//! Stream-ordered host-to-device staging for the decode hot path (glm5_next/cuda/verify.py's pinned metadata copies and
//! segments.py's event-guarded staging, as one ring). A put copies the bytes into pinned memory and queues an async
//! copy on the engine stream: it waits for nothing and drains nothing, because the copy runs after whatever the stream
//! already holds (the previous graph included). Pinned bytes are reused only after an event says every copy before has
//! run. This is a hot-path tool: the core DeviceBuffer.upload keeps its synchronous contract.
const std = @import("std");
const cuda = @import("cuda");

/// Where a put of `n` bytes lands in a ring of `cap` bytes already filled to `used`; `wrap` when the ring starts over
/// (the caller then waits for the copies in flight first). Puts are 16-byte aligned.
pub fn place(used: usize, n: usize, cap: usize) error{TooBig}!struct { at: usize, wrap: bool } {
    if (n > cap) return error.TooBig;
    const at = std.mem.alignForward(usize, used, 16);
    if (at + n <= cap) return .{ .at = at, .wrap = false };
    return .{ .at = 0, .wrap = true };
}

/// Run-time A/B switch (TF_GLM_AB=stage: cuda_native alternates it between generations): true makes every hot-path user
/// take its blocking path.
pub var sync_mode: bool = false;

pub const Stager = struct {
    stream: ?cuda.Stream = null,
    host: ?cuda.HostBuffer = null,
    event: ?cuda.Event = null,
    used: usize = 0,
    wraps: usize = 0,

    pub fn init(d: *const cuda.Driver, stream: cuda.Stream, bytes: usize) !Stager {
        var host = try cuda.HostBuffer.alloc(d, bytes);
        errdefer host.free();
        const event = try cuda.Event.init(d, false);
        return .{ .stream = stream, .host = host, .event = event };
    }
    pub fn deinit(s: *Stager) void {
        if (s.host) |*h| {
            if (s.stream) |st| st.synchronize() catch {};
            h.free();
        }
        if (s.event) |*e| e.deinit();
        s.* = .{};
    }
    pub fn ready(s: *const Stager) bool {
        return s.host != null and !sync_mode;
    }
    /// Queue `bytes` for `dst` (a device address) on the stream.
    pub fn put(s: *Stager, dst: u64, bytes: []const u8) !void {
        const host = s.host orelse return error.StagerNotReady;
        if (bytes.len == 0) return;
        const p = try place(s.used, bytes.len, host.bytes.len);
        if (p.wrap) {
            // every copy queued so far must have run before its pinned bytes are overwritten
            try s.event.?.record(s.stream.?);
            try s.event.?.synchronize();
            s.wraps += 1;
        }
        @memcpy(host.bytes[p.at..][0..bytes.len], bytes);
        s.used = p.at + bytes.len;
        const view: cuda.DeviceBuffer = .{ .d = host.d, .ptr = dst, .len = bytes.len };
        try view.uploadAsync(0, host.bytes[p.at..][0..bytes.len], s.stream.?.handle);
    }
    /// A single i32 or other small value.
    pub fn putValue(s: *Stager, dst: u64, value: anytype) !void {
        const v = value;
        try s.put(dst, std.mem.asBytes(&v));
    }
};

test "ring placement aligns, wraps and refuses an oversized put" {
    var p = try place(0, 5, 64);
    try std.testing.expectEqual(@as(usize, 0), p.at);
    try std.testing.expect(!p.wrap);
    p = try place(5, 10, 64);
    try std.testing.expectEqual(@as(usize, 16), p.at);
    p = try place(26, 38, 64);
    try std.testing.expectEqual(@as(usize, 0), p.at);
    try std.testing.expect(p.wrap);
    p = try place(32, 32, 64);
    try std.testing.expectEqual(@as(usize, 32), p.at);
    try std.testing.expect(!p.wrap);
    try std.testing.expectError(error.TooBig, place(0, 65, 64));
    var s: Stager = .{};
    try std.testing.expect(!s.ready());
    try std.testing.expectError(error.StagerNotReady, s.put(0, "x"));
}
