//! GLM-5.3-Flash picture and clip input in the server: the native frontend (zig/src/vision: split, decode, fit, expand,
//! picture cache) and the vision backend that gives the engine its feature rows. `zig`: the engine hosts the tower
//! (the canvases go to it); `python`: tools/zig/vision/glm_native_helper.py runs Python TensorFold's own tower and
//! frontend in a child process (the reference path, vision_helper.zig).
const std = @import("std");
const api = @import("engine_api");
const vision = @import("vision");
const helper_mod = @import("vision_helper.zig");
const Allocator = std.mem.Allocator;

pub const Backend = enum { zig, python };

pub const Vision = struct {
    gpa: Allocator,
    io: std.Io,
    backend: Backend,
    fe: vision.frontend.Frontend,
    slots: vision.frontend.Slots,
    fetch: ?*vision.fetch.Fetch = null,
    helper: ?*helper_mod.Helper = null,
    image_token: u32 = 154854,

    pub fn getenv(name: []const u8) ?[]const u8 {
        return vision.limits.libcGetenv(name);
    }

    /// Reads TENSORFOLD_GLM_* (the Python knobs), `allow_urls` (--vision-urls); `problem` says why not.
    pub fn start(gpa: Allocator, io: std.Io, backend: Backend, allow_urls: bool, image_token: u32, max_images: ?u32, image_tokens: ?u32, problem: *[]const u8) !*Vision {
        var diag: vision.errors.Diag = .{};
        var opts = vision.frontend.Options.fromEnv(getenv, &diag) catch {
            problem.* = try gpa.dupe(u8, diag.message());
            return error.VisionConfig;
        };
        if (max_images) |n| opts.limits.max_images = n;
        if (image_tokens) |n| opts.limits.image_tokens = n;
        opts.allow_urls = allow_urls;
        const v = try gpa.create(Vision);
        errdefer gpa.destroy(v);
        v.* = .{ .gpa = gpa, .io = io, .backend = backend, .fe = vision.frontend.Frontend.init(gpa, io, opts), .slots = vision.frontend.Slots.default(), .image_token = image_token };
        if (allow_urls) {
            const f = try gpa.create(vision.fetch.Fetch);
            f.* = vision.fetch.Fetch.init(gpa);
            v.fetch = f;
            v.fe.opts.fetcher = f.fetcher();
        }
        return v;
    }

    pub fn stop(v: *Vision, io: std.Io) void {
        if (v.helper) |h| h.stop(io);
        v.fe.deinit();
        if (v.fetch) |f| {
            f.deinit();
            v.gpa.destroy(f);
        }
        v.gpa.destroy(v);
    }
};

/// The canvases of a prepared prompt as the engine's media items (copied into `a`: the cache's buffers are let go).
pub fn engineItems(v: *Vision, a: Allocator, items: []vision.items.Item, diag: *vision.errors.Diag) (Allocator.Error || vision.errors.Error)![]api.MediaItem {
    const out = try a.alloc(api.MediaItem, items.len);
    for (items, out) |*it, *slot| {
        switch (it.*) {
            .clip => |c| slot.* = .{ .video = true, .frames = c.t, .h = c.h, .w = c.w, .data = try a.dupe(u8, c.frames) },
            .canvas => |c| slot.* = .{ .h = c.h, .w = c.w, .data = try a.dupe(u8, c.data) },
            .picture => |p| {
                const px = try v.fe.imagePixels(a, it, diag);
                slot.* = .{ .h = p.shape[1], .w = p.shape[2], .data = try a.dupe(u8, px) };
            },
        }
    }
    return out;
}
