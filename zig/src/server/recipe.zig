//! The GLM-5.3-Flash EXL3 recipe's server switches, read once from the environment under the recipe's own names
//! (scripts/config.sh exports them: the Python engine's `TF_GLM_*` variables), so the recipe's launch scripts map 1:1.
//!
//! Python reference (recipe patches on TensorFold v0.6.0): `prompts.effort_tail_enabled` (0096), `cuda/app.py`'s
//! TF_GLM_CLEAR_THINKING (0060), `cuda/stream_pacing.py` (0061), TF_GLM_LOOP_GUARD (0091), `multi.py`'s
//! TF_GLM_MAX_QUEUED (0082), `kept_reasoning.py` (0036), `weights.assistant_end` (0075).
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Recipe = struct {
    /// The GLM recipe's behaviours are on (the checkpoint is GLM-5.3-Flash); the switches below are read either way.
    glm: bool = false,
    /// TF_GLM_EFFORT_TAIL=1: the reasoning-effort line goes to the prompt's tail (issue #93).
    effort_tail: bool = false,
    /// TF_GLM_CLEAR_THINKING=1: earlier turns' reasoning is dropped from the prompt, as the checkpoint's template does.
    clear_thinking: bool = false,
    /// TF_GLM_STREAM_SMOOTH=1 and TF_GLM_STREAM_SMOOTH_MS: a streamed reply's tokens go out at a steady pace.
    stream_smooth: bool = false,
    stream_smooth_ms: u32 = 400,
    /// TF_GLM_LOOP_GUARD=1: a think block that collapsed into repeating itself is closed.
    loop_guard: bool = false,
    /// TF_GLM_MAX_QUEUED: past the lanes plus this many queued foreground requests a request is refused with 429.
    max_queued: ?u32 = null,
    /// TF_GLM_KEEP_REASONING (replies kept; 0 is off) and TF_GLM_KEEP_REASONING_MB (their text at most).
    keep_reasoning: u32 = 1024,
    keep_reasoning_mb: u32 = 32,
    /// TF_GLM_ASSISTANT_ENDS (default 1): `<|assistant|>` ends a reply like the checkpoint's end tokens.
    assistant_ends: bool = true,
    /// TF_GLM_PROMPT_STOP (default 1; patch 0109): a prompt whose client left stops at its next chunk; 0: it fills to its end.
    prompt_stop: bool = true,

    /// The switches from `env` (null: none set); a malformed value fails with the Python engine's wording in `problem`.
    pub fn parse(a: Allocator, env: ?*const std.process.Environ.Map, problem: *[]const u8) error{ Invalid, OutOfMemory }!Recipe {
        var r: Recipe = .{};
        r.effort_tail = try flag(a, env, "TF_GLM_EFFORT_TAIL", problem);
        r.clear_thinking = try flag(a, env, "TF_GLM_CLEAR_THINKING", problem);
        r.stream_smooth = try flag(a, env, "TF_GLM_STREAM_SMOOTH", problem);
        if (get(env, "TF_GLM_STREAM_SMOOTH_MS")) |raw| {
            const t = trim(raw);
            if (t.len > 0) {
                const ms = std.fmt.parseInt(i64, t, 10) catch {
                    problem.* = try std.fmt.allocPrint(a, "TF_GLM_STREAM_SMOOTH_MS='{s}': expected whole milliseconds", .{t});
                    return error.Invalid;
                };
                if (ms < 1 or ms > 5000) {
                    problem.* = try std.fmt.allocPrint(a, "TF_GLM_STREAM_SMOOTH_MS={d}: expected 1-5000", .{ms});
                    return error.Invalid;
                }
                r.stream_smooth_ms = @intCast(ms);
            }
        }
        // `os.environ.get("TF_GLM_LOOP_GUARD", "0") == "1"`: any other value is off, with no error
        r.loop_guard = if (get(env, "TF_GLM_LOOP_GUARD")) |v| std.mem.eql(u8, v, "1") else false;
        // `int(queued) if queued else None`: unset or empty queues as ever; "0" refuses anything past the lanes
        if (get(env, "TF_GLM_MAX_QUEUED")) |raw| if (raw.len > 0) {
            const n = std.fmt.parseInt(i64, trim(raw), 10) catch {
                problem.* = try std.fmt.allocPrint(a, "TF_GLM_MAX_QUEUED='{s}': expected a whole number", .{raw});
                return error.Invalid;
            };
            if (n < 0) {
                problem.* = try std.fmt.allocPrint(a, "TF_GLM_MAX_QUEUED={d}: expected 0 or more", .{n});
                return error.Invalid;
            }
            r.max_queued = @intCast(@min(n, std.math.maxInt(u32)));
        };
        r.keep_reasoning = try count(a, env, "TF_GLM_KEEP_REASONING", 1024, problem);
        r.keep_reasoning_mb = try count(a, env, "TF_GLM_KEEP_REASONING_MB", 32, problem);
        // `(os.environ.get(..., "") or "1").strip()`; 0 or 1
        const ends = trim(get(env, "TF_GLM_ASSISTANT_ENDS") orelse "");
        if (ends.len > 0) {
            if (!std.mem.eql(u8, ends, "0") and !std.mem.eql(u8, ends, "1")) {
                problem.* = try std.fmt.allocPrint(a, "TF_GLM_ASSISTANT_ENDS: 0 or 1, not '{s}'", .{ends});
                return error.Invalid;
            }
            r.assistant_ends = std.mem.eql(u8, ends, "1");
        }
        const stop = trim(get(env, "TF_GLM_PROMPT_STOP") orelse "");
        if (stop.len > 0) {
            if (!std.mem.eql(u8, stop, "0") and !std.mem.eql(u8, stop, "1")) {
                problem.* = try std.fmt.allocPrint(a, "TF_GLM_PROMPT_STOP is 0 or 1, not '{s}'", .{stop});
                return error.Invalid;
            }
            r.prompt_stop = std.mem.eql(u8, stop, "1");
        }
        return r;
    }
};

fn get(env: ?*const std.process.Environ.Map, name: []const u8) ?[]const u8 {
    const m = env orelse return null;
    return m.get(name);
}

/// Python's `str.strip()` on an environment value.
fn trim(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, " \t\r\n\x0b\x0c");
}

/// `os.environ.get(name, "0").strip() or "0"`, then 0 or 1.
fn flag(a: Allocator, env: ?*const std.process.Environ.Map, name: []const u8, problem: *[]const u8) error{ Invalid, OutOfMemory }!bool {
    const t = trim(get(env, name) orelse "0");
    const v = if (t.len == 0) "0" else t;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    problem.* = try std.fmt.allocPrint(a, "{s}='{s}': expected 0 or 1", .{ name, v });
    return error.Invalid;
}

/// `_env_int` of kept_reasoning.py: a whole number of 0 or more, `default` when unset or empty.
fn count(a: Allocator, env: ?*const std.process.Environ.Map, name: []const u8, default: u32, problem: *[]const u8) error{ Invalid, OutOfMemory }!u32 {
    const t = trim(get(env, name) orelse "");
    if (t.len == 0) return default;
    const n = std.fmt.parseInt(i64, t, 10) catch {
        problem.* = try std.fmt.allocPrint(a, "{s}='{s}': expected a whole number", .{ name, t });
        return error.Invalid;
    };
    if (n < 0) {
        problem.* = try std.fmt.allocPrint(a, "{s}='{s}': expected 0 or more", .{ name, t });
        return error.Invalid;
    }
    return @intCast(@min(n, std.math.maxInt(u32)));
}

fn testEnv(a: Allocator, pairs: []const [2][]const u8) !std.process.Environ.Map {
    var m = std.process.Environ.Map.init(a);
    for (pairs) |p| try m.put(p[0], p[1]);
    return m;
}

test "the recipe's defaults are the Python engine's" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var problem: []const u8 = "";
    const r = try Recipe.parse(arena.allocator(), null, &problem);
    try std.testing.expect(!r.effort_tail and !r.clear_thinking and !r.stream_smooth and !r.loop_guard);
    try std.testing.expectEqual(@as(?u32, null), r.max_queued);
    try std.testing.expectEqual(@as(u32, 400), r.stream_smooth_ms);
    try std.testing.expectEqual(@as(u32, 1024), r.keep_reasoning);
    try std.testing.expectEqual(@as(u32, 32), r.keep_reasoning_mb);
    try std.testing.expect(r.assistant_ends);
}

test "the recipe's switches read as config.sh exports them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var env = try testEnv(a, &.{
        .{ "TF_GLM_EFFORT_TAIL", " 1 " },    .{ "TF_GLM_CLEAR_THINKING", "1" }, .{ "TF_GLM_STREAM_SMOOTH", "1" },
        .{ "TF_GLM_STREAM_SMOOTH_MS", "250" }, .{ "TF_GLM_LOOP_GUARD", "1" },    .{ "TF_GLM_MAX_QUEUED", "0" },
        .{ "TF_GLM_KEEP_REASONING", "0" },   .{ "TF_GLM_ASSISTANT_ENDS", "0" },   .{ "TF_GLM_PROMPT_STOP", "0" },
    });
    var problem: []const u8 = "";
    const r = try Recipe.parse(a, &env, &problem);
    try std.testing.expect(r.effort_tail and r.clear_thinking and r.stream_smooth and r.loop_guard);
    try std.testing.expectEqual(@as(u32, 250), r.stream_smooth_ms);
    try std.testing.expectEqual(@as(?u32, 0), r.max_queued); // "0" is a cap of nothing past the lanes
    try std.testing.expectEqual(@as(u32, 0), r.keep_reasoning);
    try std.testing.expect(!r.assistant_ends and !r.prompt_stop);
    // config.sh exports TF_GLM_MAX_QUEUED="" when MAX_QUEUED is unset: no cap
    var unset = try testEnv(a, &.{.{ "TF_GLM_MAX_QUEUED", "" }});
    try std.testing.expectEqual(@as(?u32, null), (try Recipe.parse(a, &unset, &problem)).max_queued);
    // the loop guard is on for "1" only, with no error for anything else
    var odd = try testEnv(a, &.{.{ "TF_GLM_LOOP_GUARD", "true" }});
    try std.testing.expect(!(try Recipe.parse(a, &odd, &problem)).loop_guard);
}

test "a typo fails at load with the Python engine's wording" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    var tail = try testEnv(a, &.{.{ "TF_GLM_EFFORT_TAIL", "yes" }});
    try std.testing.expectError(error.Invalid, Recipe.parse(a, &tail, &problem));
    try std.testing.expectEqualStrings("TF_GLM_EFFORT_TAIL='yes': expected 0 or 1", problem);
    var ms = try testEnv(a, &.{.{ "TF_GLM_STREAM_SMOOTH_MS", "9000" }});
    try std.testing.expectError(error.Invalid, Recipe.parse(a, &ms, &problem));
    try std.testing.expectEqualStrings("TF_GLM_STREAM_SMOOTH_MS=9000: expected 1-5000", problem);
    var ends = try testEnv(a, &.{.{ "TF_GLM_ASSISTANT_ENDS", "2" }});
    try std.testing.expectError(error.Invalid, Recipe.parse(a, &ends, &problem));
    try std.testing.expectEqualStrings("TF_GLM_ASSISTANT_ENDS: 0 or 1, not '2'", problem);
    var stop = try testEnv(a, &.{.{ "TF_GLM_PROMPT_STOP", "yes" }});
    try std.testing.expectError(error.Invalid, Recipe.parse(a, &stop, &problem));
    try std.testing.expectEqualStrings("TF_GLM_PROMPT_STOP is 0 or 1, not 'yes'", problem);
    var keep = try testEnv(a, &.{.{ "TF_GLM_KEEP_REASONING", "-3" }});
    try std.testing.expectError(error.Invalid, Recipe.parse(a, &keep, &problem));
}
