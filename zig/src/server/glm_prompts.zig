//! TensorFold families/glm5_next/prompts.py:thinking_off (Ash Hart and contributors).
const std = @import("std");
/// Removes the first Max effort line and closes only an opened generation think block.
/// Deliberately matches the Python replacement, including its High/Low behavior.
pub fn thinkingOff(a: std.mem.Allocator, text: []const u8) ![]u8 {
    const effort = "<|system|>Reasoning Effort: Max";
    const opened = "<|assistant|><think>";
    const at = std.mem.indexOf(u8, text, effort);
    const close = std.mem.endsWith(u8, text, opened);
    const n = text.len - (if (at != null) effort.len else @as(usize, 0));
    const out = try a.alloc(u8, n + (if (close) @as(usize, 8) else 0));
    if (at) |i| {
        @memcpy(out[0..i], text[0..i]);
        @memcpy(out[i..n], text[i + effort.len ..]);
    } else @memcpy(out[0..n], text);
    if (close) @memcpy(out[n..], "</think>");
    return out;
}
const head_mask = "[gMASK]<sop>";
const effort_prefix = "<|system|>Reasoning Effort: ";
const opened_think = "<|assistant|><think>";

/// `prompts.effort_to_tail` (recipe patch 0096, TF_GLM_EFFORT_TAIL): the checkpoint text with its leading effort line
/// moved to just before the generation prompt (thinking on), or gone (thinking off), so the conversation before it
/// renders the same for every effort level and with thinking on or off. Text without the line at its head, or
/// thinking on and not ending in the opened think block, comes back as it is.
pub fn effortToTail(a: std.mem.Allocator, text: []const u8, thinking: bool) std.mem.Allocator.Error![]u8 {
    var at: usize = 0;
    if (std.mem.startsWith(u8, text, head_mask)) at = head_mask.len;
    const line_start = at;
    if (!std.mem.startsWith(u8, text[at..], effort_prefix)) return a.dupe(u8, text);
    at += effort_prefix.len;
    const levels = [_][]const u8{ "Max", "High", "Low" };
    var level: ?[]const u8 = null;
    for (levels) |l| if (std.mem.startsWith(u8, text[at..], l)) {
        level = l;
        break;
    };
    const named = level orelse return a.dupe(u8, text);
    const line_end = at + named.len;
    const head: []const u8 = text[0..line_start];
    const line = text[line_start..line_end];
    const rest = text[line_end..];
    if (!thinking) return std.mem.concat(a, u8, &.{ head, rest });
    if (!std.mem.endsWith(u8, rest, opened_think)) return a.dupe(u8, text);
    return std.mem.concat(a, u8, &.{ head, rest[0 .. rest.len - opened_think.len], line, opened_think });
}

test "effort at the tail: the line moves before the generation prompt, or goes with thinking off" {
    const a = std.testing.allocator;
    const cases = [_]struct { []const u8, bool, []const u8 }{
        .{ "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think>", true, "[gMASK]<sop><|user|>Hi<|system|>Reasoning Effort: Max<|assistant|><think>" },
        .{ "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>Hi<|assistant|><think>", true, "[gMASK]<sop><|user|>Hi<|system|>Reasoning Effort: High<|assistant|><think>" },
        .{ "<|system|>Reasoning Effort: Low<|user|>Hi<|assistant|><think>", true, "<|user|>Hi<|system|>Reasoning Effort: Low<|assistant|><think>" },
        // thinking off: the line is gone (the rendered text's think block is closed by thinkingOff afterwards)
        .{ "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>Hi<|assistant|><think>", false, "[gMASK]<sop><|user|>Hi<|assistant|><think>" },
        // not ending in the opened think block, thinking on: as it is
        .{ "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi", true, "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi" },
        // no line at its head: as it is
        .{ "[gMASK]<sop><|user|>Hi<|assistant|><think>", true, "[gMASK]<sop><|user|>Hi<|assistant|><think>" },
        .{ "[gMASK]<sop><|system|>Reasoning Effort: Medium<|user|>Hi<|assistant|><think>", true, "[gMASK]<sop><|system|>Reasoning Effort: Medium<|user|>Hi<|assistant|><think>" },
    };
    for (cases) |c| {
        const out = try effortToTail(a, c[0], c[1]);
        defer a.free(out);
        try std.testing.expectEqualStrings(c[2], out);
    }
}

test "GLM thinking off matches reference effort removal and empty block" {
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think>", "[gMASK]<sop><|user|>Hi<|assistant|><think></think>" },
        .{ "<|system|>Reasoning Effort: High<|assistant|><think>", "<|system|>Reasoning Effort: High<|assistant|><think></think>" },
        .{ "<|system|>Reasoning Effort: Max<|system|>Reasoning Effort: Max", "<|system|>Reasoning Effort: Max" },
        .{ "<|assistant|><think>past</think>answer", "<|assistant|><think>past</think>answer" },
    };
    for (cases) |c| {
        const out = try thinkingOff(std.testing.allocator, c[0]);
        defer std.testing.allocator.free(out);
        try std.testing.expectEqualStrings(c[1], out);
    }
}
