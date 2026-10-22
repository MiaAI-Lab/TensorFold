//! The chat template with picture and clip parts: the Python reference's marked prompts (tools/zig/vision/render_fixtures.py,
//! tensorfold-glm53:r107ref, the shipped Mia-AiLab GLM-5.3-Flash template with the media branch of glm5_next/cuda/app.py).
const std = @import("std");
const server = @import("root.zig");
const helper = @import("late_system_test.zig");
const template = @import("template");
const json = server.json;
const messages = @import("messages.zig");
const errors = server.errors;
const vision = @import("vision_helper.zig");
const pkg = @import("vision");

const source = @embedFile("fixtures_glm_chat.jinja");
const cases = @embedFile("fixtures_vision_render.json");

test "marked picture and clip prompts render as the Python reference's" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const swapped = (try server.hf_text.swapMedia(a, source)) orelse return error.TemplateHasNoMediaBranch;
    var encoder: helper.DropLate = .{};
    var text = encoder.text();
    var hf: server.hf_text.HfText = .{ .gpa = gpa, .arena = .init(gpa), .tok = undefined, .glm = true, .source = swapped, .compiled = try template.compile(gpa, swapped, null) };
    defer hf.arena.deinit();
    defer hf.compiled.?.deinit();
    var vt = text.vtable.*;
    vt.render = hf.text().vtable.render;
    vt.template_source = hf.text().vtable.template_source;
    text.ctx = &hf;
    text.vtable = &vt;
    const doc = (try json.parse(a, cases)).ok;
    var checked: usize = 0;
    for (doc.object.keys(), doc.object.values()) |name, case| {
        var cx: errors.Cx = .{ .a = a };
        var diag: pkg.errors.Diag = .{};
        const sp = try pkg.split.splitImages(a, case.get("messages").?, .{ .allow_videos = true, .max_videos = 4, .tool_media = true }, &diag);
        var nonce_bytes: [8]u8 = undefined;
        _ = try std.fmt.hexToBytes(&nonce_bytes, case.strField("nonce").?);
        _ = try pkg.split.MediaMarks.init(a, sp.messages, nonce_bytes);
        const normalized = try messages.toolArguments(&cx, try messages.normalizeWith(&cx, sp.messages, "system", false, true));
        const tools = case.get("tools").?;
        var problem: []const u8 = "";
        const thinking = case.get("thinking").?.bool;
        const rendered = try text.render(a, normalized, .{
            .tools = if (tools == .array and tools.array.len > 0) tools else null,
            .add_generation_prompt = true,
            .enable_thinking = thinking,
            .reasoning_effort = null,
        }, &problem);
        std.testing.expectEqualStrings(case.strField("marked").?, rendered) catch |e| {
            std.debug.print("case {s}\n", .{name});
            return e;
        };
        checked += 1;
    }
    try std.testing.expectEqual(@as(usize, 4), checked);
}

const fake_helper = @embedFile("fixtures_vision_fake_helper.py");

test "the helper's frames: ready, a prepared prompt, a refusal, the media the engine takes" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "helper.py", .data = fake_helper });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_len = try tmp.dir.realPathFile(std.testing.io, "helper.py", &path_buf);
    const script = path_buf[0..dir_len];
    var env = std.process.Environ.Map.init(gpa);
    defer env.deinit();
    try env.put("TENSORFOLD_VISION_HELPER", script);
    var problem: []const u8 = "";
    const hp = vision.Helper.start(gpa, std.testing.io, "/nonexistent", .{}, &env, &problem) catch |e| {
        if (problem.len > 0) gpa.free(problem);
        if (e == error.VisionHelper) return error.SkipZigTest; // no python3 here
        return e;
    };
    defer hp.stop(std.testing.io);
    hp.min_free = 0;
    try std.testing.expect(hp.videos and hp.width == 8);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const sources = [_]vision.Source{ .{ .kind = .image, .url = "data:,x" }, .{ .kind = .image, .url = "data:,y", .detail = "low" } };
    const done = try hp.prepare(a, "A\x00tf-media-0123:0\x00B\x00tf-media-0123:1\x00C", "0123", &sources, 1000);
    const ok = done.ok;
    try std.testing.expectEqual(@as(usize, 6), ok.tokens.len);
    try std.testing.expectEqual(@as(u32, 2), ok.tokens[0]);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3 }, ok.media.rows);
    try std.testing.expectEqual(@as(usize, 2 * 8 * 2), ok.media.features.len);
    try std.testing.expectEqual(@as(u16, 7), std.mem.readInt(u16, ok.media.features[16..18], .little));
    try ok.media.check(ok.tokens.len);
    const refused = try hp.prepare(a, "FAIL", "0123", &sources, null);
    try std.testing.expectEqual(@as(u16, 400), refused.refused.status);
    try std.testing.expect(std.mem.indexOf(u8, refused.refused.message, "JPEG, PNG or WebP") != null);
    const again = try hp.prepare(a, "A", "0123", sources[0..1], null); // the helper serves on after a refusal
    try std.testing.expectEqual(@as(usize, 1), again.ok.media.rows.len);
}
// 
