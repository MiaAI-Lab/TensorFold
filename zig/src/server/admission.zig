//! Admission at saturation (recipe patches 0081, 0082, 0095; Python `Scheduler._check_admission`): with TF_GLM_MAX_QUEUED
//! set, a foreground request that arrives while the lanes plus that many are taken is refused with 429 and
//! `Retry-After: 5` instead of queueing invisibly. Unset queues as every scheduler always has; 0 refuses anything past
//! the lanes. Background requests (priority "background", session titles) bypass: they yield and queue by design.
const std = @import("std");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;
const Cx = errors.Cx;

pub const message = "all serving lanes are busy; retry shortly";

fn cap(srv: *const Server) ?u32 {
    if (!srv.config.recipe.glm) return null;
    const queued = srv.config.recipe.max_queued orelse return null;
    return srv.info.lanes +| queued;
}

/// The check alone, for a streamed request before its 200 goes out (`App.admit`): refuses while the cap holds.
pub fn check(srv: *Server, cx: *Cx, background: bool) errors.Refused!void {
    const limit = cap(srv) orelse return;
    if (background) return;
    if (srv.in_system.load(.acquire) >= limit) return cx.fail(.capacity, "{s}", .{message});
}

/// One request in the system, from its submit to its end.
pub const Slot = struct {
    srv: *Server,

    pub fn leave(s: *Slot) void {
        _ = s.srv.in_system.fetchSub(1, .acq_rel);
    }
};

/// Counts the request in and refuses it when the cap holds (`Scheduler.submit` checks again, as `App.admit` did).
pub fn enter(srv: *Server, cx: *Cx, background: bool) errors.Refused!Slot {
    const prev = srv.in_system.fetchAdd(1, .acq_rel);
    if (cap(srv)) |limit| if (!background and prev >= limit) {
        _ = srv.in_system.fetchSub(1, .acq_rel);
        return cx.fail(.capacity, "{s}", .{message});
    };
    return .{ .srv = srv };
}
