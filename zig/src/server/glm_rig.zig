//! Test rig for the GLM recipe's server behaviours: the shipped GLM-5.3-Flash chat template (fixtures_glm_chat.jinja,
//! snapshot 078455ff), byte-level token ids, and an engine that plays a script of reply rounds on CPU.
//! Template by zai-org/GLM; TensorFold prompts.py wrapper by Ash Hart and contributors.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const helper = @import("late_system_test.zig");
const template = @import("template");
const model_text = @import("model_text.zig");

const source = @embedFile("fixtures_glm_chat.jinja");

/// An engine that answers each request with `rounds` (each one a `tokens` event of the bytes of its text), then ends
/// as `reason`. It keeps what the server submitted.
pub const Script = struct {
    rounds: []const []const u8 = &.{"hi"},
    reason: api.Reason = .stop,
    lanes: u32 = 1,
    context_window: u32 = 0,
    prompt: []u32 = &.{},
    max_tokens: u32 = 0,
    loop_guard: bool = false,
    loop_recipe: bool = false,
    loop_period: ?u32 = null, // the engine says its guard closed the think block
    background: bool = false,
    busy: bool = false, // the next submit answers Busy
    structures: bool = false, // the engine says it enforces grammars
    structure: ?api.Structure = null, // the last request's grammar (its text copied: the request's arena goes)
    submits: u32 = 0,
    kept: ?api.prompt_cache.KeptStats = null, // the prompt cache's numbers its status reports (null: no cache)
    /// Called before each round with its index (a test sleeps or reads the server there).
    before_round: ?*const fn (n: usize) void = null,

    pub fn engine(e: *Script) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const e: *Script = @ptrCast(@alignCast(ctx));
        return .{ .lanes = e.lanes, .context_window = e.context_window, .loop_guard = true, .structures = e.structures };
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Script = @ptrCast(@alignCast(ctx));
        if (e.busy) return error.Busy;
        e.submits += 1;
        std.testing.allocator.free(e.prompt);
        e.prompt = std.testing.allocator.dupe(u32, request.prompt) catch return error.Busy;
        e.max_tokens = request.max_tokens;
        e.loop_guard = request.loop_guard;
        e.loop_recipe = request.loop_recipe;
        if (e.structure) |old| std.testing.allocator.free(old.text);
        e.structure = if (request.structure) |st| .{ .kind = st.kind, .text = std.testing.allocator.dupe(u8, st.text) catch "", .after = st.after } else null;
        e.background = request.background;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        for (e.rounds, 0..) |text, n| {
            if (e.before_round) |f| f(n);
            var ids: [4096]u32 = undefined;
            for (text, 0..) |byte, i| ids[i] = byte;
            sink.event(sink.ctx, id, &.{ .tokens = ids[0..text.len] });
        }
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = e.reason, .stats = .{ .loop_period = e.loop_period } } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        const e: *Script = @ptrCast(@alignCast(ctx));
        out.* = .{ .kept = e.kept };
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The three GLM call marks are whole tokens of the real tokenizer; here they are ids no byte uses.
fn tokenId(_: *anyopaque, piece: []const u8) ?u32 {
    const known = [_][]const u8{ "<tool_call>", "<arg_key>", "</arg_value>", "</think>" };
    for (known, 0..) |k, i| if (std.mem.eql(u8, k, piece)) return 1000 + @as(u32, @intCast(i));
    return null;
}

/// The server under test. Fill it in place: `try rig.start(recipe, config)`; `rig.finish()` frees it.
pub const Rig = struct {
    encoder: helper.DropLate = .{},
    hf: server.hf_text.HfText = undefined,
    vt: model_text.Text.VTable = undefined,
    backend: Script = .{},
    srv: *server.Server = undefined,

    pub fn start(rig: *Rig, r: server.recipe.Recipe, config: server.Config) !void {
        rig.hf = .{
            .gpa = std.testing.allocator,
            .arena = .init(std.testing.allocator),
            .tok = undefined,
            .glm = true,
            .source = source,
            .compiled = try template.compile(std.testing.allocator, source, null),
            .effort_tail = r.effort_tail,
            .clear_thinking = r.clear_thinking,
        };
        var text = rig.encoder.text();
        rig.vt = text.vtable.*;
        rig.vt.render = rig.hf.text().vtable.render;
        rig.vt.template_source = rig.hf.text().vtable.template_source;
        rig.hf.eos = &.{3}; // byte 0x03 ends a reply, as <|endoftext|> does
        rig.vt.eos_ids = rig.hf.text().vtable.eos_ids;
        rig.vt.token_id = tokenId;
        text.ctx = &rig.hf;
        text.vtable = &rig.vt;
        var c = config;
        c.recipe = r;
        c.recipe.glm = true;
        rig.srv = try server.Server.init(std.testing.allocator, std.testing.io, rig.backend.engine(), text, c, null);
    }

    pub fn finish(rig: *Rig) void {
        rig.srv.deinit();
        std.testing.allocator.free(rig.backend.prompt);
        if (rig.backend.structure) |st| std.testing.allocator.free(st.text);
        rig.hf.arena.deinit();
        rig.hf.compiled.?.deinit();
    }

    /// POST /v1/chat/completions with `body`; the caller frees `.body`.
    pub fn chat(rig: *Rig, body: []const u8) !helper.Reply {
        return helper.exchange(rig.srv, "/v1/chat/completions", body);
    }

    /// The prompt the engine last received, as text (the caller frees it).
    pub fn promptText(rig: *Rig) ![]u8 {
        const text = try std.testing.allocator.alloc(u8, rig.backend.prompt.len);
        for (rig.backend.prompt, text) |id, *byte| byte.* = @intCast(id);
        return text;
    }
};

pub const default_config: server.Config = .{ .served_name = "glm5_next", .enable_thinking = true, .model_ids = &.{"glm5_next"}, .use_drafts = false, .default_max_tokens = 32768 };
