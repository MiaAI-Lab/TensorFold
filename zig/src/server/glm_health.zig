//! The fields the recipe's /health answers (Python `cuda/health.py` and the GLM decoder's `health()`), beside the
//! native server's own: `ok`, `backend`, `busy`, `requests_running`, the totals of finished requests (a bench reads the
//! `*_total` ones before and after a run), `streams`, `context_length`, and the pool's `pool_tokens` and
//! `pool_free_tokens` when the engine reports them. The kept-state figures (`kept_*`, `spill`) belong to the prompt cache.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const Server = @import("server.zig").Server;
const Value = json.Value;
const Allocator = std.mem.Allocator;

fn putNum(a: Allocator, o: *json.Object, key: []const u8, n: anytype) !void {
    try o.put(a, key, try json.intValue(a, n));
}

pub fn add(srv: *Server, a: Allocator, o: *json.Object, status: api.Status) !void {
    const running = srv.in_system.load(.acquire);
    try o.put(a, "ok", .{ .bool = true });
    try o.put(a, "backend", .{ .string = "tensorfold" });
    try o.put(a, "busy", .{ .bool = running > 0 });
    try putNum(a, o, "requests_running", running);
    {
        const m = &srv.metrics;
        m.mutex.lockUncancelable(srv.io);
        defer m.mutex.unlock(srv.io);
        try putNum(a, o, "requests_total", m.latency.n);
        try putNum(a, o, "prompt_tokens_total", m.prompt);
        // finished replies' tokens, and the live replies' so far
        try putNum(a, o, "completion_tokens_total", m.generation + status.generation_tokens);
        try o.put(a, "prefill_seconds_total", .{ .float = @round(m.prefill.total * 1e6) / 1e6 });
        try o.put(a, "decode_seconds_total", .{ .float = @round(m.decode.total * 1e6) / 1e6 });
        try putNum(a, o, "cached_tokens_total", m.cached);
        try putNum(a, o, "rounds_total", m.rounds);
        try putNum(a, o, "drafted_total", m.drafted);
        try putNum(a, o, "accepted_total", m.accepted);
    }
    const decoding: u32 = @intCast(@min(status.streams, std.math.maxInt(u32)));
    const filling = status.running -| decoding;
    const streams = try json.newObject(a);
    try putNum(a, streams, "decoding", decoding);
    try putNum(a, streams, "prefilling", filling);
    try putNum(a, streams, "max", srv.info.lanes);
    try putNum(a, streams, "filling", filling);
    try putNum(a, streams, "paused", 0);
    try o.put(a, "streams", .{ .object = streams });
    if (status.pool_tokens) |n| try putNum(a, o, "pool_tokens", n);
    if (status.pool_free_tokens) |n| try putNum(a, o, "pool_free_tokens", n);
    if (srv.info.context_window > 0) try putNum(a, o, "context_length", srv.info.context_window);
}

fn family(w: *std.Io.Writer, name: []const u8, kind: []const u8, help: []const u8) !void {
    try w.print("# HELP tensorfold_health:{s} {s}\n# TYPE tensorfold_health:{s} {s}\n", .{ name, help, name, kind });
}

/// `/metrics`' `tensorfold_health:` families (patch 0045): /health's figures, named as /health names them.
pub fn metrics(srv: *Server, w: *std.Io.Writer) !void {
    var status: api.Status = .{};
    srv.engine.status(&status, &.{});
    const m = &srv.metrics;
    m.mutex.lockUncancelable(srv.io);
    const requests = m.latency.n;
    const completion = m.generation + status.generation_tokens;
    const cached = m.cached;
    const rounds = m.rounds;
    const prefill = m.prefill.total;
    const decode = m.decode.total;
    m.mutex.unlock(srv.io);
    try family(w, "requests_total", "counter", "Finished requests (as /health's requests_total).");
    try w.print("tensorfold_health:requests_total {d}\n", .{requests});
    try family(w, "completion_tokens_total", "counter", "Reply tokens, the running replies' tokens so far included.");
    try w.print("tensorfold_health:completion_tokens_total {d}\n", .{completion});
    try family(w, "cached_tokens_total", "counter", "Prompt tokens found cached, finished requests.");
    try w.print("tensorfold_health:cached_tokens_total {d}\n", .{cached});
    try family(w, "rounds_total", "counter", "Decode rounds of finished requests.");
    try w.print("tensorfold_health:rounds_total {d}\n", .{rounds});
    try family(w, "prefill_seconds_total", "counter", "Engine prefill seconds of finished requests.");
    try w.print("tensorfold_health:prefill_seconds_total {d}\n", .{prefill});
    try family(w, "decode_seconds_total", "counter", "Engine decode seconds of finished requests.");
    try w.print("tensorfold_health:decode_seconds_total {d}\n", .{decode});
    const decoding: u32 = @intCast(@min(status.streams, std.math.maxInt(u32)));
    const filling = status.running -| decoding;
    try family(w, "streams", "gauge", "The concurrent engine's streams by state.");
    try w.print("tensorfold_health:streams{{state=\"decoding\"}} {d}\ntensorfold_health:streams{{state=\"filling\"}} {d}\ntensorfold_health:streams{{state=\"paused\"}} 0\n", .{ decoding, filling });
    if (srv.info.context_window > 0) {
        try family(w, "context_length", "gauge", "Tokens a request's prompt and reply may hold (the server's effective window).");
        try w.print("tensorfold_health:context_length {d}\n", .{srv.info.context_window});
    }
    try family(w, "streams_max", "gauge", "Streams the concurrent engine decodes at most (--parallel).");
    try w.print("tensorfold_health:streams_max {d}\n", .{srv.info.lanes});
    if (status.pool_tokens) |n| {
        try family(w, "pool_tokens", "gauge", "Token rows in the concurrent engine's shared cache pool.");
        try w.print("tensorfold_health:pool_tokens {d}\n", .{n});
    }
    if (status.pool_free_tokens) |n| {
        try family(w, "pool_free_tokens", "gauge", "Pool rows no stream or kept prompt holds.");
        try w.print("tensorfold_health:pool_free_tokens {d}\n", .{n});
    }
}

/// Pool counts on every /health, including when the recipe block is off. Absent stays absent; a known zero is written.
pub fn put(gpa: Allocator, object: *json.Object, status: api.Status) !void {
    if (status.pool_tokens) |n| try object.put(gpa, "pool_tokens", try json.intValue(gpa, n));
    if (status.pool_free_tokens) |n| try object.put(gpa, "pool_free_tokens", try json.intValue(gpa, n));
}

test "pool health distinguishes absent from known empty" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const o = try json.newObject(a);
    try put(a, o, .{});
    try std.testing.expect(o.get("pool_tokens") == null);
    try put(a, o, .{ .pool_tokens = 8192, .pool_free_tokens = 0 });
    try std.testing.expectEqual(@as(i64, 8192), o.get("pool_tokens").?.int64().?);
    try std.testing.expectEqual(@as(i64, 0), o.get("pool_free_tokens").?.int64().?);
}
