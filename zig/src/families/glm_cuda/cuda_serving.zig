//! glm5_next/cuda/multi.py rank orchestration, modeled on Flash Next cuda_native.zig.
//! P6 owns E and its CUDA resources. This wrapper owns lane state and mirrored calls.
const std = @import("std");
const lanes = @import("lanes");
const backend = @import("cuda_lanes.zig");
const mirror = @import("cuda_mirror.zig");
pub fn Serving(comptime E: type) type {
    return struct {
        const Self = @This();
        gpa: std.mem.Allocator,
        local: backend.Lanes(E),
        leader: ?mirror.Leader = null,
        follower: ?mirror.Follower = null,
        /// The object must remain at a stable address after init: mirrored backends point to local.
        pub fn init(self: *Self, gpa: std.mem.Allocator, engine: *E, context: usize, drafts: bool, streams: usize, window: usize, rank: usize, channel: ?mirror.Channel) !void {
            if (streams < 1 or streams > backend.max_streams or window < 1 or window > backend.max_rows or rank > 3) return error.BadServingGeometry;
            self.* = .{ .gpa = gpa, .local = backend.Lanes(E).init(gpa, engine, drafts, context, window) };
            self.local.streams = streams;
            self.local.follower = channel != null and rank != 0;
            if (channel) |wire| {
                if (rank == 0) {
                    self.leader = mirror.Leader.init(gpa, self.local.backend(), wire);
                    self.local.notes = self.leader.?.notes();
                } else {
                    self.follower = mirror.Follower.init(gpa, self.local.backend(), wire);
                    self.follower.?.family = .{ .ptr = &self.local, .note = noteFn, .saved = savedFn };
                }
            } else if (rank != 0) return error.MissingMirrorChannel;
        }
        fn noteFn(ptr: *anyopaque, bytes: []const u8) !void {
            const local: *backend.Lanes(E) = @ptrCast(@alignCast(ptr));
            try local.applyNote(bytes);
        }
        fn savedFn(ptr: *anyopaque, key: u64) ?*anyopaque {
            const local: *backend.Lanes(E) = @ptrCast(@alignCast(ptr));
            return local.savedFor(key);
        }
        /// Install the same prepare/agree/commit gate on every TP rank before serving.
        pub fn setAgreement(self: *Self, gate: backend.Gate, watch: ?*mirror.Watch) void {
            self.local.gate = gate;
            if (self.leader) |*lead| lead.watch = watch;
            if (self.follower) |*peer| peer.watch = watch;
        }
        pub fn laneBackend(self: *Self) lanes.backend.Backend {
            return if (self.leader) |*lead| lead.backend() else self.local.backend();
        }
        pub fn facts(self: *const Self) lanes.Model {
            return self.local.facts();
        }
        pub fn follow(self: *Self) !void {
            if (self.follower) |*peer| try peer.run() else return error.NotFollower;
        }
        pub fn deinit(self: *Self) void {
            if (self.leader) |*lead| {
                lead.stop();
                lead.deinit();
            }
            if (self.follower) |*peer| peer.deinit();
            self.local.deinit();
        }
    };
}
test "serving creates the GLM lane vtable and mirror-compatible notes" {
    const gpa = std.testing.allocator;
    const E = @import("flashnext").lane_backend.Toy;
    var e: E = .{ .gpa = gpa };
    var serving: Serving(E) = undefined;
    try serving.init(gpa, &e, 4096, true, 2, 8, 0, null);
    defer serving.deinit();
    const be = serving.laneBackend();
    try std.testing.expect(be.ptr == @as(*anyopaque, @ptrCast(&serving.local)));
    try std.testing.expectEqual(@as(u32, 2), serving.facts().max_streams);
    try std.testing.expectEqual(@as(u32, 16), serving.facts().batch_rows);
    _ = Serving(E).follow;
}
