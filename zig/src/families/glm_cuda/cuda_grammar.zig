//! Structured output on the GLM engine (the recipe's xgrammar path): the logits rows a grammar constrains have their
//! disallowed tokens set to -inf before the draw, as Python's `constraint.mask(logits, window, vocab_offset)` does on each
//! rank's vocabulary slice. The bits are the lanes core's (lanes/grammar.zig: token t at bit t % 32 of word t / 32, a row
//! per window row); the kernel is sample.cu's `tf_grammar_mask`, shared with Nemotron.
const std = @import("std");
const cuda = @import("cuda");

pub const max_rows = 64;

/// A rank's slice of `rows` bit rows of `words` words each: the words for tokens [offset, offset + columns), packed
/// `local_words` a row; a token past the bits is not allowed. `offset` is a multiple of 32.
pub fn sliceBits(out: []u32, bits: []const u32, rows: usize, words: usize, offset: usize, columns: usize) !void {
    if (offset % 32 != 0) return error.UnalignedVocabSlice;
    const local = (columns + 31) / 32;
    if (rows == 0 or bits.len != rows * words or out.len != rows * local) return error.InvalidGrammarMasks;
    const first = offset / 32;
    for (0..rows) |r| for (0..local) |w| {
        out[r * local + w] = if (first + w < words) bits[r * words + first + w] else 0;
    };
}

pub const Mask = struct {
    mod: cuda.Module,
    f: cuda.Function,
    bits: cuda.DeviceBuffer,

    /// The kernel module and a bit buffer for a window's rows of `columns` tokens.
    pub fn load(d: *const cuda.Driver, columns: usize) !Mask {
        const mod = try cuda.Module.load(d, cuda.kernels.sample);
        const f = try mod.function("tf_grammar_mask");
        const bits = try cuda.DeviceBuffer.alloc(d, max_rows * ((columns + 31) / 32) * 4);
        return .{ .mod = mod, .f = f, .bits = bits };
    }

    pub fn deinit(m: *Mask) void {
        m.bits.free();
    }

    /// `rows` bf16 logits rows of `columns` tokens at `logits` (this rank's slice, starting at token `offset`): the
    /// tokens `bits` (`rows` rows of `words` words over the whole vocabulary) do not allow become -inf.
    pub fn apply(m: *Mask, gpa: std.mem.Allocator, s: cuda.Stream, logits: u64, rows: usize, columns: usize, offset: usize, bits: []const u32, words: usize) !void {
        if (rows == 0 or rows > max_rows) return error.InvalidGrammarMasks;
        const local = (columns + 31) / 32;
        const mine = try gpa.alloc(u32, rows * local);
        defer gpa.free(mine);
        try sliceBits(mine, bits, rows, words, offset, columns);
        // the previous draw synchronised its stream, so nothing in flight reads the buffer
        try m.bits.upload(0, std.mem.sliceAsBytes(mine));
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u32, @intCast(columns)));
        a.add(m.bits.ptr);
        a.add(@as(u32, @intCast(local)));
        try cuda.launch.launch(m.f, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 256 }, .shared = 0 }, s, &a);
    }
};

test "a rank's slice of the bits" {
    // 3 rows of 4 words (128 tokens); the second rank holds tokens 64..127
    const bits = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var out: [6]u32 = undefined;
    try sliceBits(&out, &bits, 3, 4, 64, 64);
    try std.testing.expectEqualSlices(u32, &.{ 3, 4, 7, 8, 11, 12 }, &out);
    var first: [6]u32 = undefined;
    try sliceBits(&first, &bits, 3, 4, 0, 64);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 5, 6, 9, 10 }, &first);
    // columns past the bits (the engine's vocabulary is wider than the grammar's) allow nothing
    var wide: [9]u32 = undefined;
    try sliceBits(&wide, &bits, 3, 4, 64, 96);
    try std.testing.expectEqualSlices(u32, &.{ 3, 4, 0, 7, 8, 0, 11, 12, 0 }, &wide);
    try std.testing.expectError(error.UnalignedVocabSlice, sliceBits(&out, &bits, 3, 4, 40, 64));
    try std.testing.expectError(error.InvalidGrammarMasks, sliceBits(&out, bits[0..11], 3, 4, 0, 64));
}
