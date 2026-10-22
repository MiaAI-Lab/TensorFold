//! CPU HTTP checks using the shipped Mia-AiLab GLM-5.3-Flash template (snapshot 078455ff).
//! Template by zai-org/GLM; TensorFold prompts.py wrapper by Ash Hart and contributors.
const std = @import("std");
const server = @import("root.zig");
const helper = @import("late_system_test.zig");
const template = @import("template");
const source = @embedFile("fixtures_glm_chat.jinja");

test "GLM HTTP chat, SSE and health use the reference thinking-off prompt" {
    var encoder: helper.DropLate = .{};
    var text = encoder.text();
    var hf: server.hf_text.HfText = .{
        .gpa = std.testing.allocator,
        .arena = .init(std.testing.allocator),
        .tok = undefined,
        .glm = true,
        .source = source,
        .compiled = try template.compile(std.testing.allocator, source, null),
    };
    defer hf.arena.deinit();
    defer hf.compiled.?.deinit();
    var vt = text.vtable.*;
    vt.render = hf.text().vtable.render;
    vt.template_source = hf.text().vtable.template_source;
    text.ctx = &hf;
    text.vtable = &vt;
    var backend: helper.Capture = .{};
    defer std.testing.allocator.free(backend.prompt);
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text, .{ .served_name = "glm5_next", .enable_thinking = false, .model_ids = &.{"glm5_next"} }, null);
    defer srv.deinit();
    for ([_]bool{ false, true }) |thinking| {
        for ([_]bool{ false, true }) |stream| {
            const body = try std.fmt.allocPrint(std.testing.allocator, "{{\"model\":\"glm5_next\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}],\"max_tokens\":4,\"stream\":{s},\"chat_template_kwargs\":{{\"enable_thinking\":{s}}}}}", .{ if (stream) "true" else "false", if (thinking) "true" else "false" });
            defer std.testing.allocator.free(body);
            const reply = try helper.exchange(srv, "/v1/chat/completions", body);
            defer std.testing.allocator.free(reply.body);
            try std.testing.expectEqual(@as(u16, 200), reply.status);
            const expected = if (thinking) "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think>" else "[gMASK]<sop><|user|>Hi<|assistant|><think></think>";
            try std.testing.expectEqual(expected.len, backend.prompt.len);
            for (expected, backend.prompt) |byte, id| try std.testing.expectEqual(@as(u32, byte), id);
            try std.testing.expect(std.mem.indexOf(u8, reply.body, if (stream) "[DONE]" else "chat.completion") != null);
        }
    }
    const health = try helper.exchangeMethod(srv, "GET", "/health", "");
    defer std.testing.allocator.free(health.body);
    try std.testing.expectEqual(@as(u16, 200), health.status);
}
