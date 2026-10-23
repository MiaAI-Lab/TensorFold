//! GET /health, /v1/models, /metrics, and the opt-in dashboard.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const live = @import("live.zig");
const routes = @import("routes.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// With keys on, health names nothing about the model.
pub fn health(srv: *Server, conn: *Conn, a: Allocator) !void {
    if (srv.keys) |k| if (k.enabled()) return conn.sendJson(200, "{\"status\": \"ok\"}");
    const o = try json.newObject(a);
    try o.put(a, "status", .{ .string = "ok" });
    try o.put(a, "model", .{ .string = srv.config.served_name });
    const ids = try a.alloc(Value, srv.config.model_ids.len);
    for (srv.config.model_ids, ids) |id, *slot| slot.* = .{ .string = id };
    try o.put(a, "model_ids", .{ .array = ids });
    try o.put(a, "max_batch_size", try json.intValue(a, srv.info.lanes));
    if (srv.info.policy.len > 0) try o.put(a, "policy", .{ .string = srv.info.policy });
    var status: api.Status = .{};
    srv.engine.status(&status, &.{});
    try o.put(a, "warming", .{ .bool = status.warming });
    try @import("glm_health.zig").put(a, o, status);
    const memory = try json.newObject(a);
    if (srv.engine.memory(std.mem.indexOf(u8, conn.path, "reset_peak=1") != null)) |mem| {
        try memory.put(a, "active", try json.intValue(a, mem.active));
        try memory.put(a, "cache", try json.intValue(a, mem.cache));
        try memory.put(a, "peak", try json.intValue(a, mem.peak));
    }
    try o.put(a, "memory", .{ .object = memory });
    if (status.kept) |k| try keptFields(a, o, k);
    if (status.cache) |c| try putCache(a, o, c);
    try o.put(a, "live", try live.snapshot(a, srv.engine));
    if (srv.config.recipe.glm) try @import("glm_health.zig").add(srv, a, o, status);
    routes.sendValue(conn, a, 200, .{ .object = o });
}

/// The recipe's /health fields for the kept prompt states: kept_prompts, kept_bytes, kept_bytes_cap, kept_mix.
fn keptFields(a: Allocator, o: *json.Object, k: api.prompt_cache.KeptStats) !void {
    try o.put(a, "kept_prompts", try json.intValue(a, k.prompts));
    try o.put(a, "kept_bytes", try json.intValue(a, k.bytes));
    try o.put(a, "kept_bytes_cap", try json.intValue(a, k.bytes_cap));
    try o.put(a, "kept_entries_cap", try json.intValue(a, k.entries_cap));
    try o.put(a, "kept_budget", try json.intValue(a, k.budget));
    const mix = try json.newObject(a);
    try mix.put(a, "own", try json.intValue(a, k.own));
    try mix.put(a, "shared", try json.intValue(a, k.shared));
    try mix.put(a, "chats", try json.intValue(a, k.chats));
    try mix.put(a, "rows", try json.intValue(a, k.rows));
    const by = try a.alloc(Value, k.by_chat_n);
    for (by, k.by_chat[0..k.by_chat_n]) |*slot, row| {
        const triple = try a.alloc(Value, 3);
        for (triple, row) |*v, n| v.* = try json.intValue(a, n);
        slot.* = .{ .array = triple };
    }
    try mix.put(a, "by_chat", .{ .array = by });
    try o.put(a, "kept_mix", .{ .object = mix });
    const pc = try json.newObject(a);
    try pc.put(a, "hits", try json.intValue(a, k.hits));
    try pc.put(a, "misses", try json.intValue(a, k.misses));
    try pc.put(a, "evicted", try json.intValue(a, k.evicted));
    try pc.put(a, "restored_from_disk", try json.intValue(a, k.restored));
    try o.put(a, "prompt_cache", .{ .object = pc });
    if (k.spill_on) {
        const sp = try json.newObject(a);
        const names = [_][]const u8{ "entries", "bytes", "writes", "write_bytes", "reads", "read_bytes", "skipped", "failed" };
        for (names, k.spill) |name, v| try sp.put(a, name, try json.intValue(a, v));
        try o.put(a, "spill", .{ .object = sp });
    }
}

/// The prompt cache's counters with its durable tier (spill v2, the DeepSeek port's names): chats_continued and
/// chats_continued_prefill_tokens on top, every counter under prompt_cache (beside the kept states' hits, misses,
/// evicted and restored_from_disk), the tier's own under prompt_cache.spill (an array, one object per rank, rank 0
/// first).
pub fn putCache(a: Allocator, o: *json.Object, c: api.CacheStats) Allocator.Error!void {
    try o.put(a, "chats_continued", try json.intValue(a, c.chats_continued));
    try o.put(a, "chats_continued_prefill_tokens", try json.intValue(a, c.chats_continued_prefill_tokens));
    const pc = if (o.get("prompt_cache")) |v| (if (v == .object) v.object else try json.newObject(a)) else try json.newObject(a);
    const info = @typeInfo(api.CacheStats).@"struct";
    inline for (info.field_names, info.field_types) |name, T| if (T == u64) try pc.put(a, name, try json.intValue(a, @field(c, name)));
    if (c.spill) |rc| {
        const ranks = rc.slice();
        const arr = try a.alloc(Value, ranks.len);
        for (ranks, arr, 0..) |t, *v, r| {
            const sp = try json.newObject(a);
            try sp.put(a, "rank", try json.intValue(a, r));
            inline for (@typeInfo(@TypeOf(t)).@"struct".field_names) |name| try sp.put(a, name, try json.intValue(a, @field(t, name)));
            v.* = .{ .object = sp };
        }
        try pc.put(a, "spill", .{ .array = arr });
    }
    try o.put(a, "prompt_cache", .{ .object = pc });
}

pub fn models(srv: *Server, conn: *Conn, a: Allocator) !void {
    const created = std.Io.Clock.real.now(srv.io).toSeconds();
    const data = try a.alloc(Value, srv.config.model_ids.len);
    for (srv.config.model_ids, data) |id, *slot| {
        slot.* = try modelValue(a, id, srv.info, created);
    }
    const o = try json.newObject(a);
    try o.put(a, "object", .{ .string = "list" });
    try o.put(a, "data", .{ .array = data });
    routes.sendValue(conn, a, 200, .{ .object = o });
}

/// Discovery extensions report the loaded engine's prompt-plus-reply limit; zero leaves it unspecified.
fn modelValue(a: Allocator, id: []const u8, info: api.Info, created: i64) Allocator.Error!Value {
    const m = try json.newObject(a);
    try m.put(a, "id", .{ .string = id });
    try m.put(a, "object", .{ .string = "model" });
    try m.put(a, "created", try json.intValue(a, created));
    try m.put(a, "owned_by", .{ .string = "tensorfold" });
    if (info.context_window > 0) {
        try m.put(a, "context_length", try json.intValue(a, info.context_window));
        try m.put(a, "max_model_len", try json.intValue(a, info.context_window));
    }
    return .{ .object = m };
}

/// Prometheus text, version 0.0.4.
pub fn metrics(srv: *Server, conn: *Conn, a: Allocator) void {
    var out: std.Io.Writer.Allocating = .init(a);
    srv.metrics.render(srv.io, &out.writer, srv.engine, srv.info.context_window) catch return;
    if (srv.config.recipe.glm) @import("glm_health.zig").metrics(srv, &out.writer) catch return;
    const body = out.written();
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/plain; version=0.0.4; charset=utf-8") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{body.len}) catch return) catch return;
    conn.finish(body) catch {};
}

pub fn dashboard(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.config.dashboard) return routes.unknown(conn, a);
    conn.startResponse(200, null) catch return;
    conn.addHeader("Content-Type", "text/html; charset=utf-8") catch return;
    var len: [24]u8 = undefined;
    conn.addHeader("Content-Length", std.fmt.bufPrint(&len, "{d}", .{dashboard_page.len}) catch return) catch return;
    conn.finish(dashboard_page) catch {};
}

pub fn stats(srv: *Server, conn: *Conn, a: Allocator) void {
    if (!srv.config.dashboard) return routes.unknown(conn, a);
    var status: api.Status = .{};
    srv.engine.status(&status, &.{});
    const memory = srv.engine.memory(false);
    srv.metrics.mutex.lockUncancelable(srv.io);
    const prompt = srv.metrics.prompt;
    const generation = srv.metrics.generation;
    const drafted = srv.metrics.drafted;
    const accepted = srv.metrics.accepted;
    const rounds = srv.metrics.rounds;
    srv.metrics.mutex.unlock(srv.io);
    const body = statsSnapshot(a, status, memory, prompt, generation, drafted, accepted, rounds) catch return;
    conn.quiet_log = true;
    routes.sendValue(conn, a, 200, body);
}

pub fn statsSnapshot(a: Allocator, status: api.Status, memory: ?api.Memory, prompt: u64, generation: u64, drafted: u64, accepted: u64, rounds: u64) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "running", try json.intValue(a, status.running));
    try o.put(a, "waiting", try json.intValue(a, status.waiting));
    try o.put(a, "decode_tokens_per_second", .{ .float = status.decode_tokens_per_second });
    try o.put(a, "prefill_tokens_per_second", .{ .float = status.prefill_tokens_per_second });
    try o.put(a, "generation_tokens_running", try json.intValue(a, status.generation_tokens));
    try o.put(a, "prompt_tokens_total", try json.intValue(a, prompt));
    try o.put(a, "generation_tokens_total", try json.intValue(a, generation));
    try o.put(a, "mtp_drafted_total", try json.intValue(a, drafted));
    try o.put(a, "mtp_accepted_total", try json.intValue(a, accepted));
    try o.put(a, "decode_rounds_total", try json.intValue(a, rounds));
    const mem = try json.newObject(a);
    if (memory) |m| {
        try mem.put(a, "active", try json.intValue(a, m.active));
        try mem.put(a, "cache", try json.intValue(a, m.cache));
        try mem.put(a, "peak", try json.intValue(a, m.peak));
    }
    try o.put(a, "memory", .{ .object = mem });
    return .{ .object = o };
}

const dashboard_page = @embedFile("dashboard.html");

test "model discovery reports fitted engine context equally for its served name and aliases" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const info: api.Info = .{ .context_window = 8192, .context_fitted = true };
    for ([_][]const u8{ "served-model", "model-alias" }) |id| {
        const value = try modelValue(arena.allocator(), id, info, 123);
        try std.testing.expectEqualStrings(id, value.get("id").?.string);
        try std.testing.expectEqual(@as(i64, 8192), value.get("context_length").?.int64().?);
        try std.testing.expectEqual(value.get("context_length").?.int64(), value.get("max_model_len").?.int64());
        try std.testing.expectEqual(@as(i64, 123), value.get("created").?.int64().?);
    }
}

test "model discovery omits an unspecified context and preserves the full u32 limit" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const unlimited = try modelValue(arena.allocator(), "model", .{}, 0);
    try std.testing.expect(unlimited.get("context_length") == null);
    try std.testing.expect(unlimited.get("max_model_len") == null);
    const bounded = try modelValue(arena.allocator(), "model", .{ .context_window = std.math.maxInt(u32) }, 0);
    try std.testing.expectEqual(@as(i64, std.math.maxInt(u32)), bounded.get("max_model_len").?.int64().?);
}

test "stats snapshot exposes counters and omits unknown memory" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const value = try statsSnapshot(arena.allocator(), .{ .running = 1, .generation_tokens = 2 }, null, 3, 4, 5, 6, 7);
    const object = value.object;
    for ([_][]const u8{ "running", "waiting", "generation_tokens_running", "prompt_tokens_total", "generation_tokens_total", "mtp_drafted_total", "mtp_accepted_total", "decode_rounds_total", "memory" }) |key|
        try std.testing.expect(object.contains(key));
    try std.testing.expectEqual(@as(usize, 0), object.get("memory").?.object.count());
}

test "health shows the kept prompt states: count, bytes, caps, the own/shared mix by conversation, the cache's counters and the spill tier" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const o = try json.newObject(a);
    var k: api.prompt_cache.KeptStats = .{ .prompts = 3, .bytes = 1000, .bytes_cap = 4096, .own = 2, .shared = 1, .chats = 2, .rows = 300, .entries_cap = 32, .budget = 1 << 20, .hits = 5, .misses = 2, .evicted = 1, .restored = 1 };
    k.by_chat[0] = .{ 1, 2, 200 };
    k.by_chat[1] = .{ 2, 1, 100 };
    k.by_chat_n = 2;
    k.spill_on = true;
    k.spill = .{ 4, 5000, 6, 7000, 1, 1000, 2, 0 };
    try keptFields(a, o, k);
    const text = try json.stringify(a, .{ .object = o }, .{});
    for ([_][]const u8{ "\"kept_prompts\": 3", "\"kept_bytes\": 1000", "\"kept_bytes_cap\": 4096", "\"own\": 2", "\"shared\": 1", "\"chats\": 2", "\"rows\": 300", "\"by_chat\": [[1, 2, 200], [2, 1, 100]]", "\"hits\": 5", "\"restored_from_disk\": 1", "\"spill\"", "\"writes\": 6" }) |want|
        try std.testing.expect(std.mem.indexOf(u8, text, want) != null);
}

test "health's v2 tier counters: the continuation names on top, every counter under prompt_cache beside the kept states', the tier's per rank under spill" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const o = try json.newObject(a);
    try keptFields(a, o, .{ .prompts = 1, .hits = 5, .evicted = 2 }); // (a tier turns v1 off: no top-level spill)
    var ranks: api.prompt_cache.RankCounts = .{ .n = 2 };
    ranks.ranks[0].points = 5;
    ranks.ranks[1].points = 4;
    try putCache(a, o, .{ .chats_continued = 1, .memory_hits = 5, .evicted = 2, .disk_hits_committed = 3, .restores_deferred = 2, .spill = ranks });
    try std.testing.expectEqualStrings("1", o.get("chats_continued").?.int);
    try std.testing.expect(o.get("spill") == null);
    const pc = o.get("prompt_cache").?.object;
    try std.testing.expectEqualStrings("5", pc.get("hits").?.int); // the kept states' own names stay
    try std.testing.expectEqualStrings("5", pc.get("memory_hits").?.int);
    try std.testing.expectEqualStrings("3", pc.get("disk_hits_committed").?.int);
    try std.testing.expectEqualStrings("2", pc.get("restores_deferred").?.int);
    try std.testing.expectEqualStrings("0", pc.get("parked").?.int);
    const spill = pc.get("spill").?.array;
    try std.testing.expectEqual(@as(usize, 2), spill.len);
    try std.testing.expectEqualStrings("5", spill[0].object.get("points").?.int);
    try std.testing.expectEqualStrings("1", spill[1].object.get("rank").?.int);
    try std.testing.expectEqualStrings("4", spill[1].object.get("points").?.int);
    const none = try json.newObject(a);
    try putCache(a, none, .{});
    try std.testing.expect(none.get("prompt_cache").?.object.get("spill") == null);
}
