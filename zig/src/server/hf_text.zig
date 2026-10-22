//! A checkpoint's tokenizer.json and chat template behind ``model_text.Text`` (our tokenizer and template engines).
const std = @import("std");
const tokenizer = @import("tokenizer");
const template = @import("template");
const json = @import("json");
const model_text = @import("model_text.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// --vision: the checkpoint template's no-media reminder becomes GLM's picture and clip markers (Glm5NextProcessor's
/// layout; ``glm5_next/cuda/app.py media_template``), a part's ``tf_mark`` going before them.
pub var media_template: bool = false;
const no_media = "{{- \"<reminder>You are unable to process this \" ~ media_type ~ \" because you don't have multi-modal input ability. Try different methods.</reminder>\" }}";
const with_media = "{%- if item.tf_mark is defined -%}{{- item.tf_mark -}}{%- endif -%}" ++
    "{%- if media_type == 'image' -%}<|begin_of_image|><|image|><|end_of_image|>" ++
    "{%- elif media_type == 'video' -%}<|begin_of_video|><|video|><|end_of_video|>" ++
    "{%- else -%}" ++ no_media ++ "{%- endif -%}";

/// The template with its media branch swapped, or null when it has none TensorFold knows how to extend.
pub fn swapMedia(a: Allocator, source: []const u8) Allocator.Error!?[]const u8 {
    if (std.mem.count(u8, source, no_media) != 1) return null;
    return try std.mem.replaceOwned(u8, a, source, no_media, with_media);
}

pub const HfText = struct {
    gpa: Allocator,
    arena: std.heap.ArenaAllocator,
    tok: tokenizer.Tokenizer,
    source: []const u8 = "",
    glm: bool = false,
    /// The recipe's TF_GLM_EFFORT_TAIL and TF_GLM_CLEAR_THINKING (recipe.zig), set after the load for a GLM checkpoint.
    effort_tail: bool = false,
    clear_thinking: bool = false,
    tool_source: []const u8 = "",
    compiled: ?template.Compiled = null,
    tool_compiled: ?template.Compiled = null,
    specials: std.json.Value = .null,
    eos: []const u32 = &.{},
    prefix: []const u32 = &.{}, // the post-processor's tokens before a text (add_special_tokens)
    suffix: []const u32 = &.{},

    /// Loads ``dir``'s tokenizer.json, chat template, special tokens and end-of-sequence ids; a failure's reason in `problem` (from `pa`).
    pub fn load(gpa: Allocator, io: std.Io, dir: []const u8, template_override: ?[]const u8, pa: Allocator, problem: *[]const u8) !*HfText {
        const t = try gpa.create(HfText);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .arena = .init(gpa), .tok = undefined };
        errdefer t.arena.deinit();
        const a = t.arena.allocator();
        t.tok = tokenizer.loadTokenizer(io, gpa, dir) catch |e| {
            problem.* = try std.fmt.allocPrint(pa, "cannot read {s}/tokenizer.json ({s})", .{ dir, @errorName(e) });
            return error.Load;
        };
        errdefer t.tok.deinit();
        const config = readJson(io, a, dir, "tokenizer_config.json");
        const model_config = readJson(io, a, dir, "config.json");
        if (model_config == .object) if (model_config.object.get("model_type")) |mt| {
            t.glm = mt == .string and (std.mem.eql(u8, mt.string, "glm5_next") or std.mem.eql(u8, mt.string, "glm5"));
        };
        if (template_override) |path| {
            t.source = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch |e| {
                problem.* = try std.fmt.allocPrint(pa, "cannot read chat template override ({s})", .{@errorName(e)});
                return error.Load;
            };
        } else try t.readTemplates(io, a, dir, config);
        t.specials = try specialsOf(a, config);
        t.eos = try t.eosIds(a, model_config, config);
        try t.readPostProcessor(io, a, dir);
        if (media_template and t.glm and t.source.len > 0) {
            t.source = (try swapMedia(a, t.source)) orelse {
                problem.* = try std.fmt.allocPrint(pa, "--vision: this checkpoint's chat template has no media branch TensorFold knows how to extend (GLM-5.3 Flash's chat_template.jinja)", .{});
                return error.Load;
            };
        }
        if (t.source.len > 0) {
            var diag: template.Diag = .{};
            defer if (diag.msg.len > 0) gpa.free(diag.msg);
            t.compiled = template.compile(gpa, t.source, &diag) catch |e| {
                problem.* = try std.fmt.allocPrint(pa, "the chat template does not compile: {s} ({s})", .{ diag.msg, @errorName(e) });
                return error.Load;
            };
        }
        if (t.tool_source.len > 0) t.tool_compiled = template.compile(gpa, t.tool_source, null) catch null;
        return t;
    }

    /// The recipe's switches for a GLM checkpoint: the effort line's place, earlier turns' reasoning, and
    /// `<|assistant|>` ending a reply (TF_GLM_ASSISTANT_ENDS, patch 0075: the model sometimes started a second answer
    /// with it, and the raw token reached the reply's text, issue #60).
    pub fn applyRecipe(t: *HfText, r: @import("recipe.zig").Recipe) Allocator.Error!void {
        t.effort_tail = r.effort_tail;
        t.clear_thinking = r.clear_thinking;
        // a special added token of the checkpoint's tokenizer (`assistant_end` of weights.py), not any vocabulary entry
        if (r.assistant_ends) if (if (t.tok.specials.contains("<|assistant|>")) t.tok.specialTokenId("<|assistant|>") else null) |id| {
            if (std.mem.indexOfScalar(u32, t.eos, id) == null) t.eos = try std.mem.concat(t.arena.allocator(), u32, &.{ t.eos, &.{id} });
        };
    }

    pub fn deinit(t: *HfText) void {
        if (t.compiled) |*c| c.deinit();
        if (t.tool_compiled) |*c| c.deinit();
        t.tok.deinit();
        t.arena.deinit();
        t.gpa.destroy(t);
    }

    fn readJson(io: std.Io, a: Allocator, dir: []const u8, name: []const u8) std.json.Value {
        const path = std.fs.path.join(a, &.{ dir, name }) catch return .null;
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)) catch return .null;
        return std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch .null;
    }

    /// chat_template.jinja, else chat_template.json, else tokenizer_config.json's (named: "default", and "tool_use").
    fn readTemplates(t: *HfText, io: std.Io, a: Allocator, dir: []const u8, config: std.json.Value) !void {
        const jinja = try std.fs.path.join(a, &.{ dir, "chat_template.jinja" });
        if (std.Io.Dir.cwd().readFileAlloc(io, jinja, a, .limited(16 << 20))) |s| {
            t.source = s;
            return;
        } else |_| {}
        const sidecar = readJson(io, a, dir, "chat_template.json");
        if (sidecar == .object) if (sidecar.object.get("chat_template")) |s| if (s == .string) {
            t.source = s.string;
            return;
        };
        if (config != .object) return;
        const found = config.object.get("chat_template") orelse return;
        switch (found) {
            .string => |s| t.source = s,
            .array => |list| for (list.items) |entry| {
                if (entry != .object) continue;
                const name = entry.object.get("name") orelse continue;
                const body = entry.object.get("template") orelse continue;
                if (name != .string or body != .string) continue;
                if (std.mem.eql(u8, name.string, "default")) t.source = body.string;
                if (std.mem.eql(u8, name.string, "tool_use")) t.tool_source = body.string;
            },
            .object => |o| {
                if (o.get("default")) |d| if (d == .string) {
                    t.source = d.string;
                };
                if (o.get("tool_use")) |d| if (d == .string) {
                    t.tool_source = d.string;
                };
            },
            else => {},
        }
    }

    /// Every ``*_token`` string (or ``{"content": ...}``) in tokenizer_config.json, as apply_chat_template passes them.
    fn specialsOf(a: Allocator, config: std.json.Value) !std.json.Value {
        var out: std.json.ObjectMap = .empty;
        if (config == .object) {
            var it = config.object.iterator();
            while (it.next()) |e| {
                if (!std.mem.endsWith(u8, e.key_ptr.*, "_token")) continue;
                switch (e.value_ptr.*) {
                    .string => try out.put(a, e.key_ptr.*, e.value_ptr.*),
                    .object => |o| if (o.get("content")) |c| if (c == .string) try out.put(a, e.key_ptr.*, c),
                    else => {},
                }
            }
        }
        return .{ .object = out };
    }

    /// config.json's eos_token_id (one or a list) and the tokenizer's eos_token, as Python's tokenizer wrapper
    /// stops on both (Qwen3.8 Flash Next: config 248044 <|endoftext|>, tokenizer 248046 <|im_end|>).
    fn eosIds(t: *HfText, a: Allocator, model_config: std.json.Value, config: std.json.Value) ![]const u32 {
        var out: std.ArrayList(u32) = .empty;
        if (model_config == .object) {
            const holders = [_]?std.json.Value{ model_config.object.get("eos_token_id"), if (model_config.object.get("text_config")) |tc| (if (tc == .object) tc.object.get("eos_token_id") else null) else null };
            for (holders) |h| {
                const v = h orelse continue;
                switch (v) {
                    .integer => |i| try out.append(a, @intCast(i)),
                    .array => |list| for (list.items) |x| if (x == .integer) try out.append(a, @intCast(x.integer)),
                    else => {},
                }
                if (out.items.len > 0) break;
            }
        }
        if (config == .object) if (config.object.get("eos_token")) |e| {
            const piece = switch (e) {
                .string => |s| s,
                .object => |o| if (o.get("content")) |c| (if (c == .string) c.string else "") else "",
                else => "",
            };
            if (t.tok.specialTokenId(piece)) |id| if (std.mem.indexOfScalar(u32, out.items, id) == null) try out.append(a, id);
        };
        return out.items;
    }

    /// The tokens ``add_special_tokens`` adds: TemplateProcessing's single template, or Bert and Roberta's cls and sep.
    fn readPostProcessor(t: *HfText, io: std.Io, a: Allocator, dir: []const u8) !void {
        const doc = readJson(io, a, dir, "tokenizer.json");
        if (doc != .object) return;
        const pp = doc.object.get("post_processor") orelse return;
        var prefix: std.ArrayList(u32) = .empty;
        var suffix: std.ArrayList(u32) = .empty;
        try addProcessor(a, pp, &prefix, &suffix);
        t.prefix = prefix.items;
        t.suffix = suffix.items;
    }

    fn addProcessor(a: Allocator, pp: std.json.Value, prefix: *std.ArrayList(u32), suffix: *std.ArrayList(u32)) !void {
        if (pp != .object) return;
        const kind = pp.object.get("type") orelse return;
        if (kind != .string) return;
        if (std.mem.eql(u8, kind.string, "Sequence")) {
            const list = pp.object.get("processors") orelse return;
            if (list == .array) for (list.array.items) |p| try addProcessor(a, p, prefix, suffix);
            return;
        }
        if (std.mem.eql(u8, kind.string, "BertProcessing") or std.mem.eql(u8, kind.string, "RobertaProcessing")) {
            if (pp.object.get("cls")) |c| if (c == .array and c.array.items.len == 2 and c.array.items[1] == .integer) try prefix.append(a, @intCast(c.array.items[1].integer));
            if (pp.object.get("sep")) |s| if (s == .array and s.array.items.len == 2 and s.array.items[1] == .integer) try suffix.append(a, @intCast(s.array.items[1].integer));
            return;
        }
        if (!std.mem.eql(u8, kind.string, "TemplateProcessing")) return;
        const single = pp.object.get("single") orelse return;
        const specials = pp.object.get("special_tokens") orelse return;
        if (single != .array or specials != .object) return;
        var after = false;
        for (single.array.items) |piece| {
            if (piece != .object) continue;
            if (piece.object.get("Sequence") != null) {
                after = true;
                continue;
            }
            const st = piece.object.get("SpecialToken") orelse continue;
            if (st != .object) continue;
            const id = st.object.get("id") orelse continue;
            if (id != .string) continue;
            const entry = specials.object.get(id.string) orelse continue;
            if (entry != .object) continue;
            const ids = entry.object.get("ids") orelse continue;
            if (ids != .array) continue;
            for (ids.array.items) |x| if (x == .integer) try (if (after) suffix else prefix).append(a, @intCast(x.integer));
        }
    }

    pub fn text(t: *HfText) model_text.Text {
        return .{ .ctx = t, .vtable = &.{
            .encode = encodeFn,
            .decode = decodeFn,
            .token_id = tokenIdFn,
            .token_string = tokenStringFn,
            .vocab_size = vocabFn,
            .eos_ids = eosFn,
            .render = renderFn,
            .template_source = sourceFn,
            .special = specialFn,
            .token_bytes = tokenBytesFn,
        } };
    }

    fn specialFn(ctx: *anyopaque, id: u32) bool {
        const t = self(ctx);
        const piece = t.tok.id_to_token.get(id) orelse return false;
        return t.tok.specials.contains(piece);
    }

    fn self(ctx: *anyopaque) *HfText {
        return @ptrCast(@alignCast(ctx));
    }

    fn encodeFn(ctx: *anyopaque, a: Allocator, s: []const u8, add_special: bool) model_text.Error![]u32 {
        const t = self(ctx);
        const ids = t.tok.encode(a, s) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else error.Template;
        if (!add_special or (t.prefix.len == 0 and t.suffix.len == 0)) return ids;
        return std.mem.concat(a, u32, &.{ t.prefix, ids, t.suffix });
    }

    fn decodeFn(ctx: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        return self(ctx).tok.decode(a, ids, false) catch |e| if (e == error.OutOfMemory) error.OutOfMemory else a.dupe(u8, "\u{fffd}");
    }

    fn tokenIdFn(ctx: *anyopaque, piece: []const u8) ?u32 {
        return self(ctx).tok.specialTokenId(piece);
    }

    fn tokenStringFn(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        const piece = self(ctx).tok.id_to_token.get(id) orelse return a.dupe(u8, "");
        return a.dupe(u8, piece);
    }

    fn tokenBytesFn(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error!?[]u8 {
        return self(ctx).tok.tokenBytes(a, id) catch |e| if (e == error.OutOfMemory) error.OutOfMemory else null;
    }

    fn vocabFn(ctx: *anyopaque) u32 {
        return @intCast(self(ctx).tok.id_to_token.tokens.items.len);
    }

    fn eosFn(ctx: *anyopaque) []const u32 {
        return self(ctx).eos;
    }

    fn sourceFn(ctx: *anyopaque) []const u8 {
        const t = self(ctx);
        return if (t.tool_source.len > 0) std.mem.concat(t.arena.allocator(), u8, &.{ t.source, " ", t.tool_source }) catch t.source else t.source;
    }

    fn renderFn(ctx: *anyopaque, a: Allocator, messages: Value, options: model_text.RenderOptions, problem: *[]const u8) model_text.Error![]u8 {
        const t = self(ctx);
        const compiled = if (options.tools != null and t.tool_compiled != null) &t.tool_compiled.? else if (t.compiled) |*c| c else {
            problem.* = "this checkpoint has no chat template: send a completion prompt instead";
            return error.Template;
        };
        var context: std.json.ObjectMap = .empty;
        if (t.specials == .object) {
            var it = t.specials.object.iterator();
            while (it.next()) |e| try context.put(a, e.key_ptr.*, e.value_ptr.*);
        }
        // earlier turns keep their reasoning unless the request (chat_template_kwargs) or TF_GLM_CLEAR_THINKING says
        // otherwise, as in zai-org's current template; a request's own switch wins
        if (t.glm or options.clear_thinking != null) try context.put(a, "clear_thinking", .{ .bool = options.clear_thinking orelse t.clear_thinking });
        try context.put(a, "enable_thinking", .{ .bool = options.enable_thinking });
        try context.put(a, "thinking_mode", .{ .string = if (options.enable_thinking) "thinking" else "chat" });
        if (options.reasoning_effort) |e| try context.put(a, "reasoning_effort", .{ .string = e });
        var diag: template.Diag = .{};
        const tools: std.json.Value = if (options.tools) |tl| try toStd(a, tl) else .null;
        const out = template.render(a, compiled, try toStd(a, messages), tools, .{ .object = context }, options.add_generation_prompt, &diag) catch |e| {
            if (e == error.OutOfMemory) return error.OutOfMemory;
            problem.* = diag.msg;
            return error.Template;
        };
        if (!t.glm) return out;
        const prompts = @import("glm_prompts.zig");
        // TF_GLM_EFFORT_TAIL=1 (issue #93): the effort line goes last, not first; then thinking off, as ThinkingOffTemplate
        const placed = if (t.effort_tail) try prompts.effortToTail(a, out, options.enable_thinking) else out;
        if (!options.enable_thinking) return prompts.thinkingOff(a, placed);
        return placed;
    }
};

test "model EOS does not hide a different tokenizer chat end" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var t: HfText = undefined;
    t.tok.vocab = std.StringHashMap(u32).init(a);
    try t.tok.vocab.put("<|im_end|>", 248046);
    const model = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"text_config\":{\"eos_token_id\":248044}}", .{});
    const config = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"eos_token\":\"<|im_end|>\"}", .{});
    try std.testing.expectEqualSlices(u32, &.{ 248044, 248046 }, try t.eosIds(a, model, config));
    const same = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"eos_token_id\":[248046]}", .{});
    try std.testing.expectEqualSlices(u32, &.{248046}, try t.eosIds(a, same, config));
    const added = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"eos_token\":{\"content\":\"<|im_end|>\"}}", .{});
    try std.testing.expectEqualSlices(u32, &.{ 248044, 248046 }, try t.eosIds(a, model, added));
    try std.testing.expectEqualSlices(u32, &.{248046}, try t.eosIds(a, .null, config));
}

/// Our JSON value as std.json's, ints exact (past i64 as number strings) and keys in order.
pub fn toStd(a: Allocator, v: Value) Allocator.Error!std.json.Value {
    return switch (v) {
        .null => .null,
        .bool => |b| .{ .bool = b },
        .int => |s| if (std.fmt.parseInt(i64, s, 10)) |i| .{ .integer = i } else |_| .{ .number_string = s },
        .float => |f| .{ .float = f },
        .string => |s| .{ .string = s },
        .array => |items| blk: {
            var list: std.json.Array = .init(a);
            try list.ensureTotalCapacity(items.len);
            for (items) |item| list.appendAssumeCapacity(try toStd(a, item));
            break :blk .{ .array = list };
        },
        .object => |o| blk: {
            var map: std.json.ObjectMap = .empty;
            try map.ensureTotalCapacity(a, o.count());
            for (o.keys(), o.values()) |k, item| map.putAssumeCapacity(k, try toStd(a, item));
            break :blk .{ .object = map };
        },
    };
}

test "explicit template overrides checkpoint templates including tool prompts and fails closed" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data =
        \\{"model":{"type":"BPE","vocab":{"H":0},"merges":[]},"added_tokens":[]}
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data =
        \\{"chat_template":{"default":"default{{ messages[0].content }}","tool_use":"tools{{ messages[0].content }}"}}
    });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const path = try std.fmt.allocPrint(a, "{s}/override.jinja", .{dir});
    var problem: []const u8 = "";
    const messages = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"H\"}]")).ok;
    const named = try HfText.load(gpa, io, dir, null, a, &problem);
    defer named.deinit();
    try std.testing.expectEqualStrings("defaultH", try named.text().render(a, messages, .{}, &problem));
    try std.testing.expectEqualStrings("toolsH", try named.text().render(a, messages, .{ .tools = .{ .array = &.{} } }, &problem));
    try tmp.dir.writeFile(io, .{ .sub_path = "override.jinja", .data = "override{{ messages[0].content }}" });
    const tool_override = try HfText.load(gpa, io, dir, path, a, &problem);
    defer tool_override.deinit();
    try std.testing.expectEqualStrings("overrideH", try tool_override.text().render(a, messages, .{ .tools = .{ .array = &.{} } }, &problem));
    try tmp.dir.writeFile(io, .{ .sub_path = "chat_template.jinja", .data = "sidecar{{ messages[0].content }}" });
    const sidecar = try HfText.load(gpa, io, dir, null, a, &problem);
    defer sidecar.deinit();
    try std.testing.expectEqualStrings("sidecarH", try sidecar.text().render(a, messages, .{}, &problem));
    const overridden = try HfText.load(gpa, io, dir, path, a, &problem);
    defer overridden.deinit();
    for ([_]model_text.RenderOptions{ .{}, .{ .tools = .{ .array = &.{} } } }) |options|
        try std.testing.expectEqualStrings("overrideH", try overridden.text().render(a, messages, options, &problem));
    const missing = try std.fmt.allocPrint(a, "{s}/missing.jinja", .{dir});
    try std.testing.expectError(error.Load, HfText.load(gpa, io, dir, missing, a, &problem));
    try tmp.dir.writeFile(io, .{ .sub_path = "override.jinja", .data = "{% unknown %}" });
    try std.testing.expectError(error.Load, HfText.load(gpa, io, dir, path, a, &problem));
    try tmp.dir.writeFile(io, .{ .sub_path = "override.jinja", .data = "" });
    const empty = try HfText.load(gpa, io, dir, path, a, &problem);
    defer empty.deinit();
    try std.testing.expectError(error.Template, empty.text().render(a, messages, .{}, &problem));
}

test "TF_GLM_ASSISTANT_ENDS: <|assistant|> ends a reply beside the checkpoint's end tokens (0075)" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data =
        \\{"model":{"type":"BPE","vocab":{"H":0,"<|endoftext|>":1,"<|user|>":2,"<|assistant|>":3,"<|plain|>":4},"merges":[]},"added_tokens":[{"id":1,"content":"<|endoftext|>","special":true},{"id":2,"content":"<|user|>","special":true},{"id":3,"content":"<|assistant|>","special":true},{"id":4,"content":"<|plain|>","special":false}]}
    });
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"chat_template\":\"x\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"glm5_next\",\"eos_token_id\":[1,2]}" });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var problem: []const u8 = "";
    const on = try HfText.load(gpa, io, dir, null, a, &problem);
    defer on.deinit();
    try std.testing.expect(on.glm);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, on.eos);
    try on.applyRecipe(.{ .glm = true });
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, on.eos); // the role token ends a reply too
    try on.applyRecipe(.{ .glm = true }); // once
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, on.eos);
    const off = try HfText.load(gpa, io, dir, null, a, &problem);
    defer off.deinit();
    try off.applyRecipe(.{ .glm = true, .assistant_ends = false, .effort_tail = true, .clear_thinking = true });
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, off.eos); // TF_GLM_ASSISTANT_ENDS=0 keeps the checkpoint's
    try std.testing.expect(off.effort_tail and off.clear_thinking);
}

test "the media branch of GLM's template is swapped for the picture and clip markers" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const source = "x{%- set media_type = item.type -%}" ++ no_media ++ "y";
    const out = (try swapMedia(a, source)).?;
    try std.testing.expect(std.mem.indexOf(u8, out, "<|begin_of_image|><|image|><|end_of_image|>") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "item.tf_mark is defined") != null);
    try std.testing.expect(std.mem.endsWith(u8, out, "{%- endif -%}y"));
    try std.testing.expect((try swapMedia(a, "no media branch")) == null);
    try std.testing.expect((try swapMedia(a, source ++ no_media)) == null);
}
