//! `tensorfold run` for a glm5_next checkpoint. Two ranks meet the way Flash Next's CUDA engine does:
//! rank 0 listens, rank 1 joins, both agree, then the native opener builds the engine.
//! P8 prefill and P7 exact target rounds; --drafts installs P9 proposals.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");

const usage =
    \\usage: tensorfold run MODEL --tokens ID,ID,... [--tp 2|3|4 --rank N --master IP --master-port P]
    \\         [--kernels DIR] [--device N] [--context N] [--parallel N] [--drafts --drafter DIR | --no-drafts] [--eager]
    \\
;

pub fn wants(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(gpa, &.{ dir, "config.json" }) catch return false;
    defer gpa.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24)) catch return false;
    defer gpa.free(text);
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch return false;
    defer parsed.deinit();
    const mt = parsed.value.object.get("model_type") orelse return false;
    if (mt != .string) return false;
    return wantsType(mt.string);
}

pub fn wantsType(mt: []const u8) bool {
    for (glm.weights_pkg.config.model_types) |m| if (std.mem.eql(u8, m, mt)) return true;
    return false;
}

const Options = struct {
    fn tokens_u32(o: Options) []const u32 {
        return o.tokens;
    }

    model: []const u8,
    tokens: []u32 = &.{},
    kernels: ?[]const u8 = null,
    device: u32 = 0,
    context: usize = 131072,
    drafts: bool = false,
    draft_dir: ?[]const u8 = null,
    eager: bool = false,
    parallel: u32 = 1,
    max_tokens: usize = 0,
    stop: []const u32 = &.{154827},
    tokens_file: ?[]const u8 = null,
    tp: u32 = 1,
    rank: u32 = 0,
    master: []const u8 = "",
    master_port: u16 = 29551,
};

pub fn main(init: std.process.Init, args: []const []const u8) !u8 {
    const gpa = init.gpa;
    if (args.len < 3 or !std.mem.eql(u8, args[1], "run")) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var o: Options = .{ .model = args[2] };
    defer gpa.free(o.tokens);
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        const v = if (i + 1 < args.len) args[i + 1] else "";
        if (draftFlag(&o, a, v)) |skip| {
            i += skip;
            continue;
        }
        if (std.mem.eql(u8, a, "--tokens")) {
            gpa.free(o.tokens);
            o.tokens = try parseIds(gpa, v);
            i += 1;
        } else if (std.mem.eql(u8, a, "--kernels")) {
            o.kernels = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--device")) {
            o.device = try std.fmt.parseInt(u32, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--context")) {
            o.context = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--max-tokens")) {
            o.max_tokens = try std.fmt.parseInt(usize, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--stop")) {
            o.stop = try parseIds(gpa, v);
            i += 1;
        } else if (std.mem.eql(u8, a, "--tokens-file")) {
            o.tokens_file = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--parallel")) {
            o.parallel = try std.fmt.parseInt(u32, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--tp")) {
            o.tp = try std.fmt.parseInt(u32, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--rank")) {
            o.rank = try std.fmt.parseInt(u32, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--master")) {
            o.master = v;
            i += 1;
        } else if (std.mem.eql(u8, a, "--master-port")) {
            o.master_port = try std.fmt.parseInt(u16, v, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--eager")) {
            o.eager = true;
        } else if (std.mem.startsWith(u8, a, "--")) {
            std.debug.print("unknown option {s}\n{s}", .{ a, usage });
            return 2;
        }
    }
    const kernels = o.kernels orelse init.environ_map.get("TENSORFOLD_CUDA_KERNELS") orelse {
        std.debug.print("the Triton kernel set: --kernels DIR or TENSORFOLD_CUDA_KERNELS (aot.json + cubins/)\n", .{});
        return 2;
    };
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, @intCast(o.device));
    defer ctx.deinit();
    std.debug.print("loading {s} (kernels {s}, tp {d} rank {d})\n", .{ o.model, kernels, o.tp, o.rank });
    const loaded = try glm.native.open(gpa, init.io, &ctx, o.model, kernels, .{
        .context = o.context,
        .drafts = o.drafts,
        .draft_dir = o.draft_dir,
        .tp = o.tp,
        .rank = o.rank,
        .master = o.master,
        .master_port = o.master_port,
        .eager = o.eager,
        .parallel = o.parallel,
    });
    defer loaded.deinit(loaded.ctx);
    if (o.max_tokens > 0) {
        // SPMD: every rank runs the same calls (no serving core), rank 0 prints. --tokens-file a.ids,b.ids runs each in turn.
        var files = std.mem.tokenizeScalar(u8, o.tokens_file orelse "", ',');
        var single = o.tokens_file == null;
        while (single or files.peek() != null) {
            var label: []const u8 = "tokens";
            var ids: []u32 = o.tokens;
            if (!single) {
                label = files.next().?;
                const text = try std.Io.Dir.cwd().readFileAlloc(init.io, label, gpa, .limited(1 << 26));
                defer gpa.free(text);
                ids = try parseIds(gpa, text);
            }
            defer if (!single) gpa.free(ids);
            single = false;
            if (o.tokens_file == null) files = std.mem.tokenizeScalar(u8, "", ',');
            var timing: [2]u64 = undefined;
            const out = try glm.native.generate(loaded.ctx, init.io, gpa, ids, o.max_tokens, o.stop, &timing);
            defer gpa.free(out);
            if (o.rank == 0) {
                var sha = std.crypto.hash.sha2.Sha256.init(.{});
                var buf: [16]u8 = undefined;
                for (out, 0..) |t, k| {
                    if (k > 0) sha.update(",");
                    sha.update(std.fmt.bufPrint(&buf, "{d}", .{t}) catch unreachable);
                }
                var digest: [32]u8 = undefined;
                sha.final(&digest);
                std.debug.print("RESULT {s}\nTOKENS {d}", .{ label, out[0] });
                for (out[1..]) |t| std.debug.print(",{d}", .{t});
                std.debug.print("\nTOKEN_SHA {x}\nPREFILL_MS {d} prompt {d} DECODE_MS {d} tokens {d}\n", .{ digest[0..6], timing[0], ids.len, timing[1], out.len });
                const gs = glm.native.draftStats();
                std.debug.print("DRAFT_STATS rounds {d} drafted {d} accepted {d} tokens_per_round {d:.2}\n", .{ gs.rounds, gs.drafted, gs.accepted, @as(f64, @floatFromInt(out.len)) / @as(f64, @floatFromInt(@max(gs.rounds, 1))) });
            }
        }
        return 0;
    }
    if (loaded.follow) |follow| {
        try follow(loaded.ctx);
        return 0;
    }
    std.log.err("GLM CUDA engine is loaded ({d} token ids queued); pass --max-tokens N to generate", .{o.tokens.len});
    return 0;
}

fn parseIds(gpa: std.mem.Allocator, text: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n[]");
    while (it.next()) |t| try out.append(gpa, try std.fmt.parseInt(u32, t, 10));
    return out.toOwnedSlice(gpa);
}

test "glm5_next and glm5 are this command's model types" {
    try std.testing.expect(wantsType("glm5_next"));
    try std.testing.expect(wantsType("glm5"));
    try std.testing.expect(!wantsType("qwen4_exp"));
}

fn draftFlag(o: *Options, name: []const u8, value: []const u8) ?usize {
    if (std.mem.eql(u8, name, "--drafts")) {
        o.drafts = true;
        return 0;
    }
    if (std.mem.eql(u8, name, "--no-drafts")) {
        o.drafts = false;
        return 0;
    }
    if (std.mem.eql(u8, name, "--drafter")) {
        o.draft_dir = value;
        return 1;
    }
    return null;
}

test "run draft flags enable P9 explicitly and preserve drafts-off override" {
    var o: Options = .{ .model = "m" };
    try std.testing.expect(!o.drafts);
    try std.testing.expectEqual(@as(?usize, 0), draftFlag(&o, "--drafts", ""));
    try std.testing.expect(o.drafts);
    try std.testing.expectEqual(@as(?usize, 1), draftFlag(&o, "--drafter", "/draft"));
    try std.testing.expectEqualStrings("/draft", o.draft_dir.?);
    try std.testing.expectEqual(@as(?usize, 0), draftFlag(&o, "--no-drafts", ""));
    try std.testing.expect(!o.drafts);
    try std.testing.expectEqual(@as(?usize, null), draftFlag(&o, "--context", "123"));
}
