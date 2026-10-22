//! tf-glm-vision-check MODEL_DIR FIXTURE_ROOT [CASE ...] [--teacher] [--hyp] [--no-chain] [--attention flash2|flash2_fast_rcp|plain]
//!                                              [--embed-order ORDER] [--ln-rows N]
//! Runs the GPU vision tower on the fixtures of zig/tests/vision/tower/gen.py and prints, stage by stage, the maximum
//! absolute difference, the mean relative difference and the fraction of bit-identical values against the Python tower.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const vision = glm.vision;
const check = glm.vision_check;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: tf-glm-vision-check MODEL_DIR FIXTURE_ROOT [CASE ...] [--teacher] [--hyp] [--no-chain] [--attention A] [--embed-order O] [--ln-rows N] [--bench]\n", .{});
        return error.Usage;
    }
    var opts: vision.Options = .{};
    var teacher = false;
    var hyp = false;
    var chain = true;
    var bench = false;
    var sweep = false;
    var detail = false;
    var luts: ?[]const u8 = null;
    var cases: std.ArrayList([]const u8) = .empty;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--teacher")) teacher = true else if (std.mem.eql(u8, a, "--hyp")) hyp = true else if (std.mem.eql(u8, a, "--no-chain")) chain = false else if (std.mem.eql(u8, a, "--bench")) bench = true else if (std.mem.eql(u8, a, "--sweep")) sweep = true else if (std.mem.eql(u8, a, "--detail")) detail = true else if (std.mem.eql(u8, a, "--luts")) {
            i += 1;
            luts = args[i];
        } else if (std.mem.eql(u8, a, "--attention")) {
            i += 1;
            opts.attention = std.meta.stringToEnum(vision.Attention, args[i]) orelse return error.BadAttention;
        } else if (std.mem.eql(u8, a, "--embed-order")) {
            i += 1;
            opts.embed_order = std.meta.stringToEnum(vision.EmbedOrder, args[i]) orelse return error.BadOrder;
        } else if (std.mem.eql(u8, a, "--ln-rows")) {
            i += 1;
            opts.ln_rows = try std.fmt.parseInt(u32, args[i], 10);
        } else try cases.append(gpa, a);
    }
    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&d, true);
    defer stream.deinit();
    var buf: [1 << 14]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(io, &buf);
    if (luts) |path| try check.actCandidates(gpa, io, &d, stream, &out.interface, path);
    if (cases.items.len == 0) return;
    for (cases.items) |name| {
        const dir = try std.fs.path.join(gpa, &.{ args[2], name });
        defer gpa.free(dir);
        var fx = try check.Fixture.open(gpa, io, dir);
        defer fx.deinit();
        const cpath = try std.fs.path.join(gpa, &.{ dir, "canvas.bin" });
        defer gpa.free(cpath);
        const canvas = try std.Io.Dir.cwd().readFileAlloc(io, cpath, gpa, .limited(1 << 31));
        defer gpa.free(canvas);
        const frames: usize = if (std.mem.eql(u8, fx.parsed.value.object.get("kind").?.string, "image")) 1 else fx.int("frames");
        const h = fx.int("h");
        const w = fx.int("w");
        var o = opts;
        o.pass_patches = fx.int("pass_patches");
        var c: check.Checker = .{ .gpa = gpa, .io = io, .d = &d, .stream = stream, .fx = fx, .rep = .{ .out = &out.interface }, .sweep = sweep, .detail = detail };
        out.interface.print("==== {s}: {d} frames, {d}x{d}, pass_patches {d}, attention {t}, embed order {t}\n", .{ name, frames, h, w, o.pass_patches, o.attention, o.embed_order }) catch {};
        if (chain) {
            out.interface.print("-- full chain (errors accumulate through the stages)\n", .{}) catch {};
            try c.chain(args[1], o, canvas, frames, h, w);
        }
        if (teacher) try c.teacher(args[1], o, canvas, frames, h, w, hyp);
        if (bench) try c.bench(args[1], o, canvas, frames, h, w);
    }
    out.interface.flush() catch {};
}
