//! Flash Next's forward on CUDA (Python qwen4_exp/cuda/forward.py, its NVFP4 checkpoint path): the host staging
//! (token ids, the n-gram rows gathered on the host), the embedding, each decoder layer (the n-gram branch, the
//! attention hyper-connection's write-back and read-out, the DeltaNet or attention mixer, the MLP hyper-connection,
//! the routed NVFP4 experts with the shared expert), the last write-back, the mixer and the head, and `commit`
//! (DeltaNet replays, the conv windows' and n-gram tail's shifts, the n-gram history). Every launch is the Python
//! wrapper's own (cuda_triton.zig for the Triton cubins, cuda_kernels.zig for the extensions, cuda_torch_ops.zig
//! for torch's ops), in Python's order, on the same buffers (cuda_state.zig), so one GPU gives Python's bits.
//!
//! Two ranks (work/PLAN.md "TP=2 design"): the bf16 out_proj / o_proj and the MoE leave fp32 partials that are
//! all-gathered in rank order and summed by `_hc_writeback` mode 3; the head's candidates are gathered.
//!
//! Python source (TensorFold, https://github.com/ashhart/TensorFold, qwen4_exp/cuda/forward.py and mtp.py, authored
//! by Ash Hart (ashhart) with commits by vcruz305 inside TensorFold, which the owner allows for this port).
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const tops = @import("cuda_torch_ops.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const cfgs = @import("cuda_config.zig");
const comms = @import("cuda_comm.zig");
const vmm = @import("cuda_vmm.zig");
const profs = @import("cuda_prof.zig");
const prompt = @import("cuda_prompt.zig");
const moep = @import("cuda_moe_prompt.zig");
const fp8 = @import("cuda_fp8.zig");
const int4 = @import("cuda_int4.zig");

const Allocator = std.mem.Allocator;
const W = weights;

/// A pending write-back: the branch `_hc_writeback` adds next and the inject gates it reads (Python's
/// (mode, branch, weights, inject) tuple).
pub const Pending = struct { branch: tri.Branch, inject: u64 };

/// One window's buffers with what Python's Buffers holds beside the pointers: the experts' slot dtype (fp32 in the
/// MTP head's decode buffers, bf16 in the main ones and the prompt's) and the attention scratch's geometry.
pub const Bufs = struct {
    b: state.Buffers,
    y_f32: bool,
    attn: tri.AttnGeometry,

    pub fn deinit(x: *Bufs) void {
        x.b.deinit();
    }
};

/// One sequence: its committed caches (cuda_state.State) and the window `stage` staged for `commit`.
pub const Seq = struct {
    st: state.State,
    /// the staged window's tokens (the n-gram history takes the kept ones at commit)
    window: std.ArrayList(i64) = .empty,
    staged: bool = false,
    /// the prompt's last row's final streams [1, S*D] bf16 (decode.Engine.last_streams), the first absorb's input
    last_streams: ?cuda.DeviceBuffer = null,
    /// bytes this sequence holds of the engine's Budget
    charged: usize = 0,
    /// its caches in reserved address space (cuda_vmm.zig): per cache layer keys, values, the indexer-key ring,
    /// pooled blocks; empty: State's own allocation (grown by copying)
    regions: std.ArrayList(vmm.Region) = .empty,
    /// rows the indexer-key ring holds (0: no ring)
    ring_rows: usize = 0,
    /// the last round's kept rows not yet folded into its DeltaNet states: rows [held_a0, held_a0 + held) of the
    /// round scratch of parity held_par (the next round's tree folds them in first, or `flush`)
    held: usize = 0,
    held_a0: usize = 0,
    held_par: u1 = 0,

    pub fn deinit(s: *Seq, gpa: Allocator) void {
        for (s.regions.items) |*r| r.deinit(gpa);
        s.regions.deinit(gpa);
        if (s.last_streams) |*b| b.free();
        s.window.deinit(gpa);
        s.st.deinit();
    }
};

/// The rank's memory for sequences' caches (Python's MemoryGate): a sequence takes its fixed state and its
/// context caches from it, grows in steps while the budget holds the grown caches beside the old ones (a resize
/// holds both), and gives everything back when freed. A step past the budget is refused (error.NoRoom).
pub const Budget = struct {
    limit: usize,
    used: usize = 0,
    /// MemAvailable a growth step must leave (the owner's 10 GiB floor), checked against the kernel's own count
    /// at each step whatever else took memory since load; 0: unchecked
    floor: usize = 0,

    /// Whether `bytes` more leave MemAvailable at or above the floor (true when it cannot be read).
    pub fn leaves(b: *const Budget, bytes: usize) bool {
        if (b.floor == 0) return true;
        const avail = memAvailable() orelse return true;
        return avail >= b.floor + bytes;
    }

    pub fn room(b: *const Budget) usize {
        return b.limit -| b.used;
    }
};

extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn close(fd: c_int) c_int;
const c_open = open;
const c_read = read;
const c_close = close;

/// /proc/meminfo's MemAvailable in bytes, or null.
pub fn memAvailable() ?usize {
    const fd = c_open("/proc/meminfo", 0);
    if (fd < 0) return null;
    defer _ = c_close(fd);
    var buf: [4096]u8 = undefined;
    const got = c_read(fd, &buf, buf.len);
    if (got <= 0) return null;
    const n: usize = @intCast(got);
    var it = std.mem.tokenizeScalar(u8, buf[0..n], '\n');
    while (it.next()) |line| if (std.mem.startsWith(u8, line, "MemAvailable:")) {
        var f = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \t");
        const kb = std.fmt.parseInt(usize, f.next() orelse return null, 10) catch return null;
        return kb * 1024;
    };
    return null;
}

/// A stream's rows [a0, a1) of a window shared by several streams (Python's Seg = (State, a0, a1)).
pub const Seg = struct { seq: *Seq, a0: usize, a1: usize };

/// One stream's window of a shared round: its tokens at its own seq.st.pos.
pub const Window = struct { seq: *Seq, tokens: []const u32 };

/// Scratch Python's wrappers allocate per call from torch's caching allocator (split-K partials, the fp32 down
/// projection of a read-out, the DeltaNet prompt front's outputs, the shared expert's), sized for the widest window.
pub const Scratch = struct {
    arena: state.Arena,
    part: u64,
    dn: u64, // [rows, low + streams] fp32
    shared_g: u64, // [rows, 2 NI] bf16
    shared_y: u64, // [rows, D] fp32 or bf16
    gq: u64, // [rows, nk, 128] fp32
    gk: u64,
    gv: u64, // [rows, nv, 128] bf16
    gg: u64, // [rows, nv] fp32
    gb: u64,
    gy: u64, // [rows, nv, 128] bf16
    kv_scale: u64, // a bf16 cache's one-element fp16 scales (kvcache.KVCache)
    invalid: u64, // gatherRows' bad-index flag
    vote: u64, // two ranks' growth agreement (this rank's word, then both)
    gdn_t: u64, // a prompt chunk's DeltaNet state of one layer, copied back over the sequence's
    // two ranks, the n-gram table on the GPU: every head's row ids, this rank's heads' embedding and group sums,
    // and both ranks' gathered
    ngram_ids: u64, // [rows, heads] int64
    ple_half: u64, // [rows, heads / world * dh] bf16
    xs_half: u64, // [rows, heads / world * dh / 32] fp32
    ple_all: u64, // [world, rows, heads / world * dh] bf16
    xs_all: u64,
    // INT4-AutoRound: a projection's block-FP8 columns and its bf16 rows' columns before they are copied into place,
    // the healed shared expert's gate|up and its SwiGLU
    p8: u64, // [rows, the widest block-FP8 face] bf16
    pb: u64, // [rows, the widest bf16 part] bf16
    sh_g: u64, // [rows, 2 shared width] bf16
    sh_a: u64, // [rows, shared width] bf16

    qsa_pos: u64, // [1] int32: a row-split indexer block's first own row position (cuda_prompt.zig)

    fn sizes(g: state.Geometry, rows: usize, part: usize) [24]usize {
        const ni = g.moe_width;
        const two = g.world > 1;
        const emb: usize = if (two) rows * g.ple_dim * 2 else 0;
        const xs: usize = if (two) rows * (g.ple_dim / 32) * 4 else 0;
        const on4 = g.int4ar;
        const face8 = @max(g.gdnConv() + g.nv * state.gdn_dv, g.heads * 2 * g.head_dim + 2 * g.kv_heads * g.head_dim);
        const faceb = @max(2 * g.nv, (g.index_heads + 1) * g.index_dim);
        const sw = g.sharedWidth();
        return .{ part, rows * (g.low + g.streams) * 4, rows * 2 * ni * 2, rows * g.hidden * 4, rows * g.nk * state.gdn_dk * 4, rows * g.nk * state.gdn_dk * 4, rows * g.nv * state.gdn_dv * 2, rows * g.nv * 4, rows * g.nv * 4, rows * g.nv * state.gdn_dv * 2, 256, 256, 256, g.nv * state.gdn_dv * state.gdn_dk * 4, if (two) rows * g.pleHeads() * 8 else 0, emb, xs, emb, xs, 256, if (on4) rows * face8 * 2 else 0, if (on4) rows * faceb * 2 else 0, if (on4) rows * 2 * sw * 2 else 0, if (on4) rows * sw * 2 else 0 };
    }

    pub fn init(d: *const cuda.Driver, g: state.Geometry, rows: usize, part: usize) !Scratch {
        const sz = sizes(g, rows, part);
        var total: usize = 0;
        for (sz) |n| total += std.mem.alignForward(usize, n, 256);
        var s: Scratch = undefined;
        s.arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, total) };
        errdefer s.arena.buf.free();
        try s.arena.buf.fill8(0, null);
        const ptrs = [_]*u64{ &s.part, &s.dn, &s.shared_g, &s.shared_y, &s.gq, &s.gk, &s.gv, &s.gg, &s.gb, &s.gy, &s.kv_scale, &s.invalid, &s.vote, &s.gdn_t, &s.ngram_ids, &s.ple_half, &s.xs_half, &s.ple_all, &s.xs_all, &s.qsa_pos, &s.p8, &s.pb, &s.sh_g, &s.sh_a };
        for (ptrs, sz) |p, n| p.* = s.arena.take(n);
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        s.arena.buf.free();
    }
};

/// A round's DeltaNet scratch and tables (Python gdn_multi.Scratch and Tables): the window rows' queries, and their
/// keys, values, gates and betas kept a round per parity (the next round's trees fold a stream's kept rows from
/// them); every row's stream and conv taps, every stream's row range, kept rows to fold, conv and state pointers.
/// The most rows the K-serial NVFP4 kernel takes for `n` columns: where it beat `_fp4mm` on GB10 (fp4-check's timing,
/// out/w6-i-fp4: N 640 2.1x at 36-64 rows, 1.2x at 113-128, slower past; N 1280 1.7x at 36-48, slower from 113; both
/// slower at 1-8 rows, which keep `_fp4mm`).
pub fn fp4SerialRows(n: usize) usize {
    return if (n <= 640) 128 else if (n <= 1280) 64 else 0;
}

/// attn_multi.PTRS: a stream's pointers a layer (keys, values, key scales, value scales, index keys, pooled).
pub const multi_ptrs = 6;

pub const Side = struct { s: cuda.Stream, fork: cuda.Event, join: cuda.Event };

pub const Round = struct {
    arena: state.Arena,
    pin: cuda.HostBuffer,
    rows: usize,
    streams: usize,
    lin: usize,
    q: u64,
    kv: [2][4]u64, // per parity: k, v, g, beta of every layer ([lin][rows][...])
    sizes: [4]usize, // a layer's k, v, g, beta bytes
    sid: u64,
    win: u64,
    starts: u64,
    held: u64,
    counts: u64,
    conv: u64,
    state: u64,
    rtable: u64,
    rrows: u64,
    rcount: u64,
    host_bytes: usize,
    /// attn_multi.Step's tables (the batched attention): each row's position, each stream's first position and
    /// rows, and every attention layer's cache pointers [layer][6][streams] int64; their own pinned buffer
    att: usize,
    posr: u64,
    first: u64,
    acounts: u64,
    aptr: u64,
    apin: cuda.HostBuffer,

    pub const held_stride = state.max_rows;

    pub const Layer = struct { k: u64, v: u64, g: u64, beta: u64 };

    pub fn layer(r: *const Round, p: usize, li: usize) Layer {
        return .{ .k = r.kv[p][0] + li * r.sizes[0], .v = r.kv[p][1] + li * r.sizes[1], .g = r.kv[p][2] + li * r.sizes[2], .beta = r.kv[p][3] + li * r.sizes[3] };
    }

    pub fn init(d: *const cuda.Driver, g: state.Geometry, rows: usize, streams: usize) !Round {
        const lin = g.linear_layers;
        const sz = [4]usize{ rows * g.nk * state.gdn_dk * 4, rows * g.nv * state.gdn_dv * 2, rows * g.nv * 4, rows * g.nv * 4 };
        const tables = [_]usize{ rows * 4, rows * 16, (streams + 1) * 4, streams * held_stride * 4, streams * 4, lin * streams * 8, lin * streams * 8, 5 * lin * 8, held_stride * 4, 4 };
        var total: usize = std.mem.alignForward(usize, rows * g.nk * state.gdn_dk * 4, 256);
        for (0..2) |_| for (sz) |b| {
            total += std.mem.alignForward(usize, lin * b, 256);
        };
        for (tables) |b| total += std.mem.alignForward(usize, b, 256);
        for ([_]usize{ rows * 4, streams * 4, streams * 4, g.attention_layers * multi_ptrs * streams * 8 }) |b| total += std.mem.alignForward(usize, b, 256);
        var r: Round = undefined;
        r.arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, total) };
        errdefer r.arena.buf.free();
        try r.arena.buf.fill8(0, null);
        r.rows = rows;
        r.streams = streams;
        r.lin = lin;
        r.sizes = sz;
        r.q = r.arena.take(rows * g.nk * state.gdn_dk * 4);
        for (0..2) |p| for (0..4) |j| {
            r.kv[p][j] = r.arena.take(lin * sz[j]);
        };
        const ptrs = [_]*u64{ &r.sid, &r.win, &r.starts, &r.held, &r.counts, &r.conv, &r.state, &r.rtable, &r.rrows, &r.rcount };
        for (ptrs, tables) |q, b| q.* = r.arena.take(b);
        r.host_bytes = 0;
        for (tables) |b| r.host_bytes += std.mem.alignForward(usize, b, 64);
        r.pin = try cuda.HostBuffer.alloc(d, r.host_bytes);
        errdefer r.pin.free();
        r.att = g.attention_layers;
        const at = r.attSizes();
        r.posr = r.arena.take(at[0]);
        r.first = r.arena.take(at[1]);
        r.acounts = r.arena.take(at[2]);
        r.aptr = r.arena.take(at[3]);
        r.apin = try cuda.HostBuffer.alloc(d, attHost(at));
        return r;
    }

    /// posr, first, counts and the pointer tables' bytes.
    fn attSizes(r: *const Round) [4]usize {
        return .{ r.rows * 4, r.streams * 4, r.streams * 4, r.att * multi_ptrs * r.streams * 8 };
    }

    fn attHost(at: [4]usize) usize {
        var n: usize = 0;
        for (at) |b| n += std.mem.alignForward(usize, b, 64);
        return n;
    }

    pub fn deinit(r: *Round) void {
        r.apin.free();
        r.pin.free();
        r.arena.buf.free();
    }
};

/// The split-K partials the widest window's matmuls need (bf16.matmul and nvfp4.matmul allocate them per call).
pub fn partBytes(g: state.Geometry, rows: usize) usize {
    const D = g.hidden;
    const Wd = g.wide();
    const b16 = [_][2]usize{
        .{ g.low + g.streams, Wd },                .{ g.low, Wd },        .{ Wd, g.low },               .{ g.gdnWidth(), D },
        .{ D, g.nv * state.gdn_dv },               .{ g.attnWidth(), D }, .{ D, g.heads * g.head_dim }, .{ Wd, g.ple_dim },
        .{ D, g.ple_dim },                         .{ D, D },             .{ g.head_n, D },
        // INT4-AutoRound's bf16 parts beside block FP8: DeltaNet b|a, the indexer
                    .{ 2 * g.nv, D },
        .{ (g.index_heads + 1) * g.index_dim, D },
    };
    var most: usize = 0;
    for (b16) |nk| most = @max(most, tri.b16PartBytes(rows * (if (nk[0] == D and nk[1] == D) g.streams else 1), nk[0], nk[1]));
    most = @max(most, tri.fp4PartBytes(rows, 2 * g.moe_width, D));
    most = @max(most, tri.fp4PartBytes(rows, D, g.moe_width));
    return @max(most, 256);
}

/// Per-layer tensors as Python's capture names them (zig/tests/cuda/flashnext/capture.py Dumps): each one's sha256,
/// compared with the oracle's digests.json when given, written raw into `dir` when given.
pub const Dump = struct {
    gpa: Allocator,
    io: std.Io,
    want: ?*const std.json.ObjectMap = null,
    dir: ?[]const u8 = null,
    prefix: []const u8 = "",
    mtp: bool = false,
    seen: std.StringHashMapUnmanaged(usize) = .empty,
    names: std.ArrayList([]u8) = .empty,
    equal: usize = 0,
    differ: usize = 0,
    missing: usize = 0,
    /// mismatches printed at most
    show: usize = 12,
    prefix_buf: [16]u8 = undefined,

    pub fn deinit(d: *Dump) void {
        d.seen.deinit(d.gpa);
        for (d.names.items) |n| d.gpa.free(n);
        d.names.deinit(d.gpa);
    }

    pub fn setChunk(d: *Dump, chunk: ?usize) void {
        d.prefix = if (chunk) |c| std.fmt.bufPrint(&d.prefix_buf, "c{d:0>2}_", .{c}) catch "" else "";
    }

    fn put(d: *Dump, f: *Forward, name: []const u8, ptr: u64, bytes: usize) !void {
        var nb: [160]u8 = undefined;
        const base = try std.fmt.bufPrint(&nb, "{s}{s}{s}", .{ d.prefix, if (d.mtp) "mtp_" else "", name });
        const key = try d.gpa.dupe(u8, base);
        try d.names.append(d.gpa, key);
        const gop = try d.seen.getOrPut(d.gpa, key);
        var full_buf: [176]u8 = undefined;
        var full: []const u8 = key;
        if (gop.found_existing) {
            gop.value_ptr.* += 1;
            full = try std.fmt.bufPrint(&full_buf, "{s}#{d}", .{ key, gop.value_ptr.* });
        } else gop.value_ptr.* = 1;
        const host = try d.gpa.alloc(u8, bytes);
        defer d.gpa.free(host);
        try f.s.synchronize();
        const buf: cuda.DeviceBuffer = .{ .d = f.d, .ptr = ptr, .len = bytes };
        try buf.download(0, host);
        var sum: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(host, &sum, .{});
        const hex = std.fmt.bytesToHex(sum, .lower);
        if (d.dir) |dir| {
            var pb: [512]u8 = undefined;
            var fb: [176]u8 = undefined;
            const file = try std.fmt.bufPrint(&fb, "{s}", .{full});
            for (file) |*ch| if (ch.* == '#') {
                ch.* = '_';
            };
            const path = try std.fmt.bufPrint(&pb, "{s}/{s}.bin", .{ dir, file });
            try std.Io.Dir.cwd().writeFile(d.io, .{ .sub_path = path, .data = host });
        }
        const want = d.want orelse return;
        const entry = want.get(full) orelse {
            d.missing += 1;
            if (d.missing <= d.show) std.debug.print("  dump {s}: not in the oracle's digests\n", .{full});
            return;
        };
        const ws = entry.object.get("sha256").?.string;
        if (std.mem.eql(u8, ws, &hex)) {
            d.equal += 1;
        } else {
            d.differ += 1;
            if (d.differ <= d.show) std.debug.print("  dump {s}: DIFFER ({d} bytes)\n", .{ full, bytes });
        }
    }
};

/// Two ranks' prompt glue split by rows (work/research/R2-prefill.md W-C, the GLM recipe's split): rank r runs the
/// write-backs and read-outs of rows [o, o + rh), rh = ceil(R / 2) rounded up to 16 (rank 1's rows past R are pad), and
/// the read-out rows are all-gathered for the mixers and experts, which run on every row as before. A mixer's or
/// the MoE's fp32 partial is made in two halves: this rank's rows into its slot of the [2, rh, D] block the
/// write-back adds (mode 3, rank 0 then rank 1, one rounding: unchanged), the peer's rows sent to the peer while the
/// peer's partial of this rank's rows arrives in the other slot. Every row's values are the unsplit path's bits:
/// each kernel is row-independent and runs the same arithmetic on the same inputs.
pub const Split = struct {
    rank: usize,
    /// rows a rank runs: ceil(R / 2) rounded up to 16 (the second rank's last ones pad rows past R)
    rh: usize,
    /// this rank's first row and the peer's
    o: usize,
    po: usize,
};

/// Rows from which prompt chunks split the glue (smaller chunks: little to gain).
pub const split_min_rows = 64;

/// How a forward reaches the GPU: the stream, the three kernel sets, the weights and the shapes of this rank.
pub const Forward = struct {
    gpa: Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    ops: kern.Ops,
    t: tri.Tri,
    th: tops.Torch,
    w: *const W.Weights,
    c: *const cfgs.Config,
    dims: tri.Dims,
    g: state.Geometry,
    sc: Scratch,
    comm: ?*const comms.Comm = null,
    /// pinned staging: token ids (i32), then the n-gram rows (bf16)
    pin: cuda.HostBuffer,
    pin_rows: usize,
    staged: cuda.Event,
    /// each decoder layer's index among the DeltaNet layers or among the attention layers
    sub: [cfgs.max_layers]usize = @splat(0),
    dump: ?*Dump = null,
    /// a decode window's attention keys bound (a graph's context bucket, Python compute's `context`); null: exact
    context: ?usize = null,
    /// a shared round's graph: each segment's keys bound (its context bucket), parallel to the segments
    contexts: ?[]const usize = null,
    /// the sequences' memory (null: unbounded)
    budget: ?*Budget = null,
    /// decode rounds' DeltaNet scratch (initRound), the parity the last decode stage wrote, and every live
    /// sequence (those holding rows of a round that a later one overwrites are folded first)
    round: ?Round = null,
    cur_par: u1 = 1,
    seqs: std.ArrayList(*Seq) = .empty,
    /// two ranks: the n-gram table's GPU gather of this rank's heads (fn_pack.cu)
    gather: ?W.Gather = null,
    eps: f32,
    vocab_offset: i64 = 0,
    /// the sampling scratch (cuda_engine.zig sizes it for the head rows); `candidates` reads it at two ranks
    sample: ?tops.Torch.Sample = null,
    /// TF_FLASHNEXT_PROFILE: the prompt chunk's parts timed (cuda_prof.zig); null: no events
    prof: ?*profs.Prof = null,
    /// TF_FLASHNEXT_COUNT_EXPERTS=1 (with the profile): every decode MoE layer's distinct routed experts and rows
    /// counted on the host (a wait a layer); the main model's and the MTP head's, since the last report
    count_experts: bool = false,
    /// the shared expert's stream (initSide): it runs beside the routed experts, joined before their sum
    side: ?Side = null,
    /// `_fp4mm` SK 1 launches of 9 to fp4SerialRows(n) rows on fn_ops' K-serial kernel (fp4Serial)
    fp4_serial: bool = false,
    /// a shared round of at least this many streams runs its attention in one launch a kernel (attn_multi; 0 off,
    /// or a kernel set without it); `round_multi` says whether the staged round does
    attn_multi: usize = 0,
    round_multi: bool = false,
    experts_seen: [2]struct { layers: usize = 0, distinct: usize = 0, rows: usize = 0 } = .{ .{}, .{} },
    /// prompt matmuls with split K run their slices in one program (cuda_prompt.zig, the same bits); false:
    /// Python's split kernels and `_reduce` for every row count
    prompt_mm: bool = false,
    /// prompt chunks' routed experts read each expert's weights once a call (cuda_moe_prompt.zig, the same bits)
    px: ?*const moep.Prompt4 = null,
    /// two ranks, prompt chunks: the hyper-connection glue split by rows (`Split`); false: every rank runs it on
    /// every row (the same bits either way)
    glue_split: bool = false,
    /// INT4-AutoRound: the block-FP8 lane matmul and the GPTQ int4 kernels (cuda_engine loads them)
    k8: ?*const fp8.Kernels = null,
    k4: ?*const int4.Kernels = null,
    /// decode D2 (TF_FLASHNEXT_L2PF=1, cuda_comm.zig): the weights the kernels after the token mixer's gather and
    /// after the MoE's gather read first (the MLP read-out's down projection; the next layer's or the mixer's)
    pf_mixer: comms.Prefetch = .{},
    pf_moe: comms.Prefetch = .{},
    /// decode D5 (TF_FLASHNEXT_MTP_Q4=1): the MTP head's matrices in 4-bit, run by `mmx` (drafts only)
    mtp_q4: ?*const @import("cuda_mtp_q4.zig").Table = null,
    /// two ranks, split prompt glue: the exchanges run on this stream beside the next rows' work (TF_FLASHNEXT_OVERLAP)
    cs: ?cuda.Stream = null,
    cs_ev: [4]cuda.Event = undefined,
    cs_i: usize = 0,
    /// what the comm stream overlaps (TF_FLASHNEXT_OVERLAP bits): 1 the mixers' out_proj (attention: the peer's
    /// rows first), 2 the MoE in two row halves (the peer's first)
    overlap: u8 = 1,
    /// two ranks, prompt chunks: an attention block's indexer rows scored and selected half on each rank, the key
    /// lists exchanged, from this many keys (TF_FLASHNEXT_QSA_SPLIT_KEYS; 0: never)
    qsa_split_keys: usize = 65536,
    /// prompt indexer kernels: `_scores_rows` with this many rows a program (0: Python's `_scores`), and
    /// `_select_tiles` from this many blocks (Python's from past `_select`'s registers); the same bits either way
    qsa_rt: usize = 0,
    qsa_tiles_from: usize = 8192,

    pub fn init(gpa: Allocator, d: *const cuda.Driver, s: cuda.Stream, ops: kern.Ops, t: tri.Tri, th: tops.Torch, w: *const W.Weights, c: *const cfgs.Config, g: state.Geometry, dims: tri.Dims, max_rows: usize) !Forward {
        var f: Forward = .{ .gpa = gpa, .d = d, .s = s, .ops = ops, .t = t, .th = th, .w = w, .c = c, .dims = dims, .g = g, .sc = undefined, .pin = undefined, .pin_rows = max_rows, .staged = undefined, .eps = @floatCast(c.eps), .vocab_offset = w.vocab_offset };
        f.sc = try Scratch.init(d, g, max_rows, partBytes(g, max_rows));
        errdefer f.sc.deinit();
        f.pin = try cuda.HostBuffer.alloc(d, max_rows * 4 + max_rows * g.pleHeads() * g.pleHeadDim() * 2 + 256);
        errdefer f.pin.free();
        f.staged = try cuda.Event.init(d, false);
        try f.staged.record(s);
        var lin: usize = 0;
        var att: usize = 0;
        for (w.layers, 0..) |l, i| {
            if (l.linear) {
                f.sub[i] = lin;
                lin += 1;
            } else {
                f.sub[i] = att;
                att += 1;
            }
        }
        if (lin != g.linear_layers or att != g.attention_layers) return error.LayerCountsDiffer;
        var ple: usize = 0;
        for (w.layers) |l| ple += @intFromBool(l.ple != null);
        if (ple > 1) return error.UnsupportedModel; // stage's rows are one PLE layer's (Python stages each at the same rows)
        if (w.table) |tab| if (tab.gpu != null) {
            f.gather = try W.Gather.init(d);
        };
        return f;
    }

    /// The decode rounds' DeltaNet scratch for `rows` rows of up to `streams` streams.
    pub fn initRound(f: *Forward, rows: usize, streams: usize) !void {
        f.round = try Round.init(f.d, f.g, rows, streams);
    }

    pub fn register(f: *Forward, seq: *Seq) !void {
        try f.seqs.append(f.gpa, seq);
    }

    pub fn unregister(f: *Forward, seq: *Seq) void {
        for (f.seqs.items, 0..) |q, i| if (q == seq) {
            _ = f.seqs.swapRemove(i);
            return;
        };
    }

    /// The sequence's kept rows folded into its DeltaNet states now (gdn.replay in place, every layer at once).
    pub fn flush(f: *Forward, seq: *Seq) !void {
        if (seq.held == 0) return;
        const r = &(f.round orelse return error.NoRoundScratch);
        const g = f.g;
        try f.staged.synchronize();
        const base = r.host_bytes - std.mem.alignForward(usize, 4, 64) - std.mem.alignForward(usize, Round.held_stride * 4, 64) - std.mem.alignForward(usize, 5 * r.lin * 8, 64);
        const tbl = std.mem.bytesAsSlice(u64, r.pin.bytes[base..][0 .. 5 * r.lin * 8]);
        for (0..r.lin) |li| {
            const l = r.layer(seq.held_par, li);
            tbl[4 * li + 0] = l.k;
            tbl[4 * li + 1] = l.v;
            tbl[4 * li + 2] = l.g;
            tbl[4 * li + 3] = l.beta;
            tbl[4 * r.lin + li] = f.rec(&seq.st, li);
        }
        const rows_at = base + std.mem.alignForward(usize, 5 * r.lin * 8, 64);
        const rows = std.mem.bytesAsSlice(i32, r.pin.bytes[rows_at..][0 .. Round.held_stride * 4]);
        for (0..seq.held) |j| rows[j] = @intCast(seq.held_a0 + j);
        const count_at = rows_at + std.mem.alignForward(usize, Round.held_stride * 4, 64);
        std.mem.bytesAsSlice(i32, r.pin.bytes[count_at..][0..4])[0] = @intCast(seq.held);
        try f.uploadRound(r.rtable, base, 5 * r.lin * 8);
        try f.uploadRound(r.rrows, rows_at, Round.held_stride * 4);
        try f.uploadRound(r.rcount, count_at, 4);
        try f.ops.gdnTreeReplay(r.rtable, r.lin, 1, r.rrows, Round.held_stride, r.rcount, 1, 0, g.nk, g.nv, state.gdn_dv);
        try f.staged.record(f.s);
        seq.held = 0;
    }

    /// A decode round's DeltaNet tables (Python gdn_multi.Tables) for `segs`, writing parity `p`: rows' streams and
    /// conv taps (< 3 the stream's conv state row, else window row a0 + tap - 3), row ranges, the kept rows to fold
    /// (last round's, parity 1 - p), conv and state pointers a layer. Streams whose kept rows sit in parity p are
    /// folded first, since this round overwrites them.
    fn roundTables(f: *Forward, segs: []const Seg, p: u1) !void {
        const r = &(f.round orelse return error.NoRoundScratch);
        const n = segs.len;
        const R = segs[n - 1].a1;
        if (n > r.streams or R > r.rows) return error.WindowPastBuffers;
        for (f.seqs.items) |q| if (q.held > 0 and q.held_par == p) try f.flush(q);
        try f.staged.synchronize();
        var at: usize = 0;
        const take = struct {
            fn next(a: *usize, bytes: usize) usize {
                const o = a.*;
                a.* += std.mem.alignForward(usize, bytes, 64);
                return o;
            }
        }.next;
        const o_sid = take(&at, r.rows * 4);
        const o_win = take(&at, r.rows * 16);
        const o_starts = take(&at, (r.streams + 1) * 4);
        const o_held = take(&at, r.streams * Round.held_stride * 4);
        const o_counts = take(&at, r.streams * 4);
        const o_conv = take(&at, r.lin * r.streams * 8);
        const o_state = take(&at, r.lin * r.streams * 8);
        const b = r.pin.bytes;
        const sid = std.mem.bytesAsSlice(i32, b[o_sid..][0 .. R * 4]);
        const win = std.mem.bytesAsSlice(i32, b[o_win..][0 .. R * 16]);
        const starts = std.mem.bytesAsSlice(i32, b[o_starts..][0 .. (n + 1) * 4]);
        const held = std.mem.bytesAsSlice(i32, b[o_held..][0 .. n * Round.held_stride * 4]);
        const counts = std.mem.bytesAsSlice(i32, b[o_counts..][0 .. n * 4]);
        const convs = std.mem.bytesAsSlice(u64, b[o_conv..][0 .. r.lin * n * 8]);
        const st_ptrs = std.mem.bytesAsSlice(u64, b[o_state..][0 .. r.lin * n * 8]);
        for (segs, 0..) |sg, i| {
            for (sg.a0..sg.a1) |row| {
                sid[row] = @intCast(i);
                for (0..4) |tap| {
                    const j = row - sg.a0 + tap;
                    win[row * 4 + tap] = @intCast(if (j < 3) j else sg.a0 + j);
                }
            }
            starts[i] = @intCast(sg.a0);
            const q = sg.seq;
            const folding = q.held > 0 and q.held_par != p;
            counts[i] = if (folding) @intCast(q.held) else 0;
            if (folding) for (0..q.held) |j| {
                held[i * Round.held_stride + j] = @intCast(q.held_a0 + j);
            };
            for (0..r.lin) |li| {
                convs[li * n + i] = f.conv(&q.st, li);
                st_ptrs[li * n + i] = f.rec(&q.st, li);
            }
        }
        starts[n] = @intCast(R);
        try f.uploadRound(r.sid, o_sid, R * 4);
        try f.uploadRound(r.win, o_win, R * 16);
        try f.uploadRound(r.starts, o_starts, (n + 1) * 4);
        try f.uploadRound(r.held, o_held, n * Round.held_stride * 4);
        try f.uploadRound(r.counts, o_counts, n * 4);
        try f.uploadRound(r.conv, o_conv, r.lin * n * 8);
        try f.uploadRound(r.state, o_state, r.lin * n * 8);
        f.round_multi = f.attn_multi != 0 and n >= f.attn_multi;
        if (f.round_multi) try f.multiTables(r, segs);
        // the kept rows ride in this round's trees: the streams hold nothing until they commit again
        for (segs) |sg| sg.seq.held = 0;
        f.cur_par = p;
        try f.staged.record(f.s);
    }

    /// A second stream for the shared expert: its matmuls are a few programs each, latency-bound over K, and run
    /// beside the routed experts' bandwidth-bound kernels (the same kernels and inputs: the same bits).
    pub fn initSide(f: *Forward) !void {
        var s = try cuda.Stream.init(f.d, true);
        errdefer s.deinit();
        var fork = try cuda.Event.init(f.d, false);
        errdefer fork.deinit();
        const join = try cuda.Event.init(f.d, false);
        f.side = .{ .s = s, .fork = fork, .join = join };
    }

    /// attn_multi.Step for `segs` (positions before the round commits): rows' positions, streams' first positions
    /// and rows, every attention layer's cache pointers; uploaded from the round's own pinned buffer.
    fn multiTables(f: *Forward, r: *Round, segs: []const Seg) !void {
        const n = segs.len;
        const R = segs[n - 1].a1;
        const at = r.attSizes();
        var o: [4]usize = undefined;
        var off: usize = 0;
        for (at, &o) |b, *x| {
            x.* = off;
            off += std.mem.alignForward(usize, b, 64);
        }
        const hb = r.apin.bytes;
        const posr = std.mem.bytesAsSlice(i32, hb[o[0]..][0 .. R * 4]);
        const first = std.mem.bytesAsSlice(i32, hb[o[1]..][0 .. n * 4]);
        const counts = std.mem.bytesAsSlice(i32, hb[o[2]..][0 .. n * 4]);
        const ptrs = std.mem.bytesAsSlice(u64, hb[o[3]..][0 .. r.att * multi_ptrs * n * 8]);
        for (segs, 0..) |sg, s| {
            const st = &sg.seq.st;
            for (sg.a0..sg.a1) |row| posr[row] = @intCast(st.pos + row - sg.a0);
            first[s] = @intCast(st.pos);
            counts[s] = @intCast(sg.a1 - sg.a0);
            for (0..r.att) |ai| {
                const t = ptrs[ai * multi_ptrs * n ..][0 .. multi_ptrs * n];
                const vals = [multi_ptrs]u64{ st.kc[ai], st.vc[ai], f.sc.kv_scale, f.sc.kv_scale, st.ikc[ai], st.pooled[ai] };
                for (vals, 0..) |v, j| t[j * n + s] = v;
            }
        }
        // the device tables at the round's first n rows / streams (the pointer tables a layer [6][n] apart)
        for ([_]u64{ r.posr, r.first, r.acounts, r.aptr }, o, [_]usize{ R * 4, n * 4, n * 4, r.att * multi_ptrs * n * 8 }) |dst, src, bytes|
            try f.d.check(f.d.api.cuMemcpyHtoDAsync_v2(dst, r.apin.bytes[src..].ptr, bytes, f.s.handle), "cuMemcpyHtoDAsync");
    }

    pub fn deinit(f: *Forward) void {
        if (f.side) |*sd| {
            sd.join.deinit();
            sd.fork.deinit();
            sd.s.deinit();
        }
        if (f.round) |*r| r.deinit();
        f.seqs.deinit(f.gpa);
        if (f.gather) |*g| g.deinit();
        f.staged.deinit();
        f.pin.free();
        f.sc.deinit();
    }

    // -- small helpers ----------------------------------------------------------------------------------------

    /// The row split of a prompt chunk's glue at two ranks (null: unsplit).
    pub fn splitOf(f: *const Forward, x: *const Bufs, R: usize) ?Split {
        const cm = f.comm orelse return null;
        if (!f.glue_split or !x.b.prefill or cm.world != 2 or R < split_min_rows) return null;
        // a multiple of 16 rows a rank: every row offset keeps the 16-byte alignment the kernels were specialized on
        const rh = std.mem.alignForward(usize, (R + 1) / 2, 16);
        if (2 * rh > x.b.rows) return null;
        const r: usize = cm.rank;
        return .{ .rank = r, .rh = rh, .o = r * rh, .po = (1 - r) * rh };
    }

    /// Every rank's rows [rank * rh, (rank + 1) * rh) of `buf` ([2 rh, row] elements of `dtype`) gathered in place.
    fn gatherRows(f: *Forward, buf: u64, sp: Split, row_bytes: usize, dtype: cuda.nccl.DataType, elem: usize) !void {
        const cm = f.comm.?;
        const n = sp.rh * row_bytes;
        // in place (NCCL's form): RoCE refuses overlapping buffers, so these stay on NCCL at every size
        try cm.allGatherInPlace(buf + sp.o * row_bytes, buf, n / elem, dtype, f.s.handle);
    }

    fn nextEvent(f: *Forward) cuda.Event {
        f.cs_i = (f.cs_i + 1) % f.cs_ev.len;
        return f.cs_ev[f.cs_i];
    }

    /// The exchange of `count` elements: `send` to the peer, the peer's into `recv`; on the comm stream after the
    /// work queued so far (`exchangeWait` joins it), or in line without one.
    fn exchangeAsync(f: *Forward, send: u64, recv: u64, count: usize, dtype: cuda.nccl.DataType) !void {
        const cm = f.comm.?;
        const cs = f.cs orelse return cm.exchange(send, recv, count, dtype, f.s.handle);
        const e = f.nextEvent();
        try e.record(f.s);
        try cs.wait(e);
        try cm.exchange(send, recv, count, dtype, cs.handle);
    }

    fn exchangeWait(f: *Forward) !void {
        const cs = f.cs orelse return;
        const e = f.nextEvent();
        try e.record(cs);
        try f.s.wait(e);
    }

    /// The GPU time since the previous mark is `part`'s (profiling only).
    pub fn mark(f: *Forward, part: profs.Part) !void {
        if (f.prof) |p| try p.mark(f.s, part);
    }

    pub fn put(f: *Forward, name: []const u8, ptr: u64, bytes: usize) !void {
        if (f.dump) |d| try d.put(f, name, ptr, bytes);
    }

    fn putLayer(f: *Forward, layer: *const W.Layer, what: []const u8, ptr: u64, bytes: usize) !void {
        if (f.dump == null) return;
        var buf: [64]u8 = undefined;
        const name = if (layer.index < 0)
            try std.fmt.bufPrint(&buf, "L{d}_{s}", .{ layer.index, what })
        else
            try std.fmt.bufPrint(&buf, "L{d:0>2}_{s}", .{ @as(u32, @intCast(layer.index)), what });
        try f.put(name, ptr, bytes);
    }

    pub fn mm(f: *Forward, x: u64, x_stride: usize, r: W.Rows, out: u64, fp32: bool, m: usize) !void {
        try f.mmAt(null, x, x_stride, r, out, fp32, m);
    }

    /// `mm`, its time marked as `part` when profiling (a split's `_reduce` apart, as `.reduce`).
    pub fn mmAt(f: *Forward, part: ?profs.Part, x: u64, x_stride: usize, r: W.Rows, out: u64, fp32: bool, m: usize) !void {
        if (prompt.b16Takes(f.prompt_mm, m, r.n, r.k)) {
            try prompt.b16(f.t, x, x_stride, r.weight, out, fp32, m, r.n, r.k);
        } else if (f.prof != null and part != null and tri.b16SplitK(r.n, r.k) > 1 and f.t.rec == null) {
            try prompt.b16Slices(f.t, x, x_stride, r.weight, f.sc.part, fp32, m, r.n, r.k);
            try f.mark(part.?);
            try prompt.reduce(f.t, f.sc.part, out, fp32, m * r.n, tri.b16SplitK(r.n, r.k));
            try f.mark(.reduce);
            return;
        } else try f.t.b16mm(x, x_stride, r.weight, out, fp32, f.sc.part, m, r.n, r.k);
        if (part) |pp| try f.mark(pp);
    }

    /// A linear on its kernel: block FP8 (`l8`) or bf16 rows, marked as `part` when profiling.
    fn denseAt(f: *Forward, part: profs.Part, x: u64, x_stride: usize, xs: u64, r: W.Rows, l8: ?fp8.Linear, out: u64, fp32: bool, m: usize) !void {
        if (l8) |l| {
            try fp8.matmul(f.k8 orelse return error.NoFp8Kernels, f.s, x, x_stride, l, out, fp32, m);
            return f.mark(part);
        }
        return f.mmxAt(part, x, x_stride, xs, r, out, fp32, m);
    }

    /// INT4-AutoRound's projections: the first columns block FP8 (`l8`), the rest bf16 rows (`rb`), each on its kernel
    /// into a scratch and copied into `out`'s columns (rows `out_w` wide; Python's Concat); without `l8`, `rb` alone.
    fn projFace(f: *Forward, part: profs.Part, x: u64, x_stride: usize, l8: ?fp8.Linear, rb: W.Rows, out: u64, out_w: usize, m: usize) !void {
        const l = l8 orelse return f.mmAt(part, x, x_stride, rb, out, false, m);
        try fp8.matmul(f.k8 orelse return error.NoFp8Kernels, f.s, x, x_stride, l, f.sc.p8, false, m);
        try f.th.slotCopy(f.sc.p8, l.n * 2, out, out_w * 2, l.n * 2, m);
        try f.mmAt(null, x, x_stride, rb, f.sc.pb, false, m);
        try f.th.slotCopy(f.sc.pb, rb.n * 2, out + l.n * 2, out_w * 2, rb.n * 2, m);
        if (l.n + rb.n != out_w) return error.ProjectionWidth;
        try f.mark(part);
    }

    /// The head's logits of `m` rows: bf16 rows, or INT4-AutoRound's GPTQ int4 columns.
    pub fn headMm(f: *Forward, x: u64, x_stride: usize, out: u64, m: usize) !void {
        if (f.w.head4) |h| return int4.dense(f.k4 orelse return error.NoInt4Kernels, f.s, x, x_stride, h, out, false, m);
        return f.mm(x, x_stride, f.w.head, out, false, m);
    }

    /// The shared expert's NVFP4-table matmul (nvfp4.matmul), marked as `part` when profiling.
    /// `fp4At`'s matmul on `t`'s stream, unmarked (the side stream's).
    fn fp4On(f: *Forward, t: tri.Tri, th: tops.Torch, x: u64, x_stride: usize, fp: tri.Fp4, out: u64, fp32: bool, m: usize, n: usize, k: usize) !void {
        if (f.fp4Serial(fp, x_stride, m, n, k)) {
            try th.fp4Serial(x, x_stride, fp.weight, fp.scale, out, fp32, m, n, k);
        } else if (prompt.fp4Takes(f.prompt_mm, m, n, k)) {
            try prompt.fp4(t, x, x_stride, fp, out, fp32, m, n, k);
        } else try t.fp4mm(x, x_stride, fp, out, fp32, f.sc.part, m, n, k);
    }

    fn fp4At(f: *Forward, part: profs.Part, x: u64, x_stride: usize, fp: tri.Fp4, out: u64, fp32: bool, m: usize, n: usize, k: usize) !void {
        try f.fp4On(f.t, f.th, x, x_stride, fp, out, fp32, m, n, k);
        try f.mark(part);
    }

    /// Whether `_fp4mm`'s single-slice launch (SK 1: the shared expert's gate/up) runs as fn_ops' K-serial kernel
    /// (TF_FLASHNEXT_FP4_SERIAL=1; its bits are `_fp4mm`'s): bf16-pattern tables, decode-sized rows.
    fn fp4Serial(f: *const Forward, fp: tri.Fp4, x_stride: usize, m: usize, n: usize, k: usize) bool {
        return f.fp4_serial and !fp.codes and m > 8 and m <= fp4SerialRows(n) and tri.fp4SplitK(n, k) == 1 and tops.fp4SerialFits(x_stride, n, k);
    }

    /// mm with the input's group sums `xs` known: a matrix with a 4-bit copy in mtp_q4 (the MTP head's, D5) runs qmm
    /// on it; every other matrix (and xs 0) the bf16 matmul as `mm`.
    pub fn mmx(f: *Forward, x: u64, x_stride: usize, xs: u64, r: W.Rows, out: u64, fp32: bool, m: usize) !void {
        if (xs != 0) if (f.mtp_q4) |t| if (t.find(r.weight)) |q| return f.ops.qmm32Split(x, x_stride, xs, q, out, m, fp32);
        try f.mm(x, x_stride, r, out, fp32, m);
    }

    /// mmAt (the profiled part) with the 4-bit MTP copy as `mmx`.
    pub fn mmxAt(f: *Forward, part: ?profs.Part, x: u64, x_stride: usize, xs: u64, r: W.Rows, out: u64, fp32: bool, m: usize) !void {
        if (xs != 0) if (f.mtp_q4) |t| if (t.find(r.weight)) |q| {
            try f.ops.qmm32Split(x, x_stride, xs, q, out, m, fp32);
            if (part) |p| try f.mark(p);
            return;
        };
        try f.mmAt(part, x, x_stride, r, out, fp32, m);
    }

    pub fn setPos(f: *Forward, st: *state.State, pos: usize) !void {
        st.pos = pos;
        try f.th.fill32(st.pos_dev, @intCast(pos), 1);
    }

    pub fn setMtpLen(f: *Forward, st: *state.State, n: usize) !void {
        st.mtp_len = n;
        try f.th.fill32(st.mtp_pos, @intCast(n), 1);
    }

    /// The DeltaNet state of linear layer `li` (one buffer: kept rows are folded in place).
    pub fn rec(f: *const Forward, st: *const state.State, li: usize) u64 {
        return st.rec + li * f.g.nv * state.gdn_dv * state.gdn_dk * 4;
    }

    fn conv(f: *const Forward, st: *const state.State, li: usize) u64 {
        return st.conv + li * (f.g.conv_kernel - 1) * f.g.gdnConv() * 2;
    }

    /// The upload of `bytes` pinned bytes at `at` to `dst`, on the stream.
    pub fn upload(f: *Forward, dst: u64, at: usize, bytes: usize) !void {
        if (bytes == 0) return;
        try f.d.check(f.d.api.cuMemcpyHtoDAsync_v2(dst, f.pin.bytes[at..].ptr, bytes, f.s.handle), "cuMemcpyHtoDAsync");
    }

    /// `upload` from the round tables' own pinned buffer.
    fn uploadRound(f: *Forward, dst: u64, at: usize, bytes: usize) !void {
        if (bytes == 0) return;
        const r = &(f.round orelse return error.NoRoundScratch);
        try f.d.check(f.d.api.cuMemcpyHtoDAsync_v2(dst, r.pin.bytes[at..].ptr, bytes, f.s.handle), "cuMemcpyHtoDAsync");
    }

    /// State.reset: an empty sequence in the same buffers.
    pub fn reset(f: *Forward, seq: *Seq) !void {
        const st = &seq.st;
        const g = f.g;
        try f.th.zero(st.conv, g.linear_layers * (g.conv_kernel - 1) * g.gdnConv() * 2);
        try f.th.zero(st.rec, g.linear_layers * g.nv * state.gdn_dv * state.gdn_dk * 4);
        st.cur = @splat(0);
        seq.held = 0;
        try f.th.zero(st.ple_tail, (g.ple_kernel - 1) * g.ngram_size * g.wide() * 2);
        for (f.w.layers) |l| if (l.ple) |p| p.ngram.initialHistory(st.history[0 .. p.ngram.n - 1]);
        seq.staged = false;
        try f.setPos(st, 0);
        st.rope_delta = 0;
        try f.th.fill32(st.rope_delta_dev, 0, 1);
        st.mtp_drafted = 0;
        try f.setMtpLen(st, 0);
    }

    // -- staging ----------------------------------------------------------------------------------------------

    /// A sequence's caches grown to hold `rows` rows (a growth step at a time, State.ensure) within the budget: the
    /// grown caches must fit beside the old ones while the copy runs. The stream drains first (the resize copies on
    /// the legacy stream and frees the old caches).
    pub fn grow(f: *Forward, seq: *Seq, rows: usize) !void {
        const st = &seq.st;
        if (rows <= st.capacity) return;
        if (rows > st.limit) return error.ContextPastWindow;
        const target = @min(st.limit, std.mem.alignForward(usize, rows, state.grow_step));
        if (seq.regions.items.len > 0) return f.growMapped(seq, target);
        const now = if (st.ctx) |c| c.buf.len else 0;
        const next = state.State.cacheBytes(f.g, target) + 64 * 256;
        if (!try f.agree(if (f.budget) |b| b.used + next <= b.limit and b.leaves(next) else true)) return error.NoRoom;
        try f.s.synchronize();
        _ = try st.ensure(rows);
        const after = if (st.ctx) |c| c.buf.len else 0;
        if (f.budget) |b| {
            b.used = b.used - now + after;
            seq.charged = seq.charged - now + after;
        }
    }

    /// Two ranks: a growth goes ahead only when both may take it (each rank's MemAvailable is its own), so both
    /// refuse together and stay in step; one rank: its own answer.
    pub fn agree(f: *Forward, ok: bool) !bool {
        const cm = f.comm orelse return ok;
        const mine: u64 = @intFromBool(ok);
        const v: cuda.DeviceBuffer = .{ .d = f.d, .ptr = f.sc.vote, .len = 8 };
        try f.s.synchronize();
        try v.upload(0, std.mem.asBytes(&mine));
        try cm.allGather(f.sc.vote, f.sc.vote + 64, 1, .u64, f.s.handle);
        var both: [2]u64 = undefined;
        try f.s.synchronize();
        const buf: cuda.DeviceBuffer = .{ .d = f.d, .ptr = f.sc.vote + 64, .len = 16 };
        try buf.download(0, std.mem.sliceAsBytes(&both));
        return both[0] != 0 and both[1] != 0;
    }

    /// grow in place: more physical memory under each cache's reserved range (zeroed, as Python's caches are),
    /// nothing copied, no pointer moves (State.version stays: captured graphs stay valid).
    fn growMapped(f: *Forward, seq: *Seq, target: usize) !void {
        const st = &seq.st;
        const g = f.g;
        const kv_row = g.kv_heads * g.head_dim * 2;
        const pooled = g.blocks(target) * g.index_dim * 2;
        var need: usize = 0;
        const layers = seq.regions.items.len / 4;
        for (0..layers) |i| {
            const r = seq.regions.items[4 * i ..][0..4];
            for ([_]usize{ 0, 1, 3 }, [_]usize{ target * kv_row, target * kv_row, pooled }) |j, bytes| need += r[j].v.up(bytes) -| r[j].mapped;
        }
        if (!try f.agree(if (f.budget) |b| b.used + need <= b.limit and b.leaves(need) else true)) return error.NoRoom;
        var added: usize = 0;
        for (0..layers) |i| {
            const r = seq.regions.items[4 * i ..][0..4];
            for ([_]usize{ 0, 1, 3 }, [_]usize{ target * kv_row, target * kv_row, pooled }) |j, bytes| {
                const before = r[j].mapped;
                const got = try r[j].growTo(f.gpa, bytes);
                if (got > 0) try f.th.zero(r[j].base + before, got);
                added += got;
            }
        }
        st.capacity = target;
        if (f.budget) |b| b.used += added;
        seq.charged += added;
    }

    /// Token ids into the window's `ids` through the pinned buffer (after the last staging's copies are done).
    pub fn stageIds(f: *Forward, x: *const Bufs, tokens: []const u32) !void {
        if (tokens.len > x.b.rows or tokens.len > f.pin_rows) return error.WindowPastBuffers;
        try f.staged.synchronize();
        const ids = std.mem.bytesAsSlice(i32, f.pin.bytes[0 .. tokens.len * 4]);
        for (ids, tokens) |*a, t| a.* = @intCast(t);
        try f.upload(x.b.ids, 0, tokens.len * 4);
        try f.staged.record(f.s);
    }

    /// forward.stage: every window's token ids and n-gram rows (each stream's own history) into one buffer, the
    /// windows' rows one after another; `segs` gets each stream's rows. Returns the rows.
    pub fn stageMany(f: *Forward, windows: []const Window, x: *const Bufs, segs: []Seg) !usize {
        var R: usize = 0;
        if (segs.len < windows.len) return error.SegsTooFew;
        for (windows, 0..) |w, i| {
            if (w.tokens.len == 0) return error.EmptyWindow;
            for (windows[0..i]) |o| if (o.seq == w.seq) return error.StreamTwiceInRound;
            segs[i] = .{ .seq = w.seq, .a0 = R, .a1 = R + w.tokens.len };
            R += w.tokens.len;
        }
        if (R > x.b.rows or R > f.pin_rows) return error.WindowPastBuffers;
        for (windows) |w| {
            const st = &w.seq.st;
            if (st.pos + w.tokens.len > st.capacity) try f.grow(w.seq, st.pos + w.tokens.len);
            try w.seq.window.resize(f.gpa, w.tokens.len);
            for (w.seq.window.items, w.tokens) |*a, t| a.* = t;
        }
        // a prompt chunk runs from the committed DeltaNet state: last round's kept rows folded in first
        if (x.b.prefill) for (windows) |w| try f.flush(w.seq);
        // the previous step's copies out of the pinned buffer are done
        try f.staged.synchronize();
        const ids = std.mem.bytesAsSlice(i32, f.pin.bytes[0 .. R * 4]);
        for (segs[0..windows.len], windows) |sg, w| for (ids[sg.a0..sg.a1], w.tokens) |*a, t| {
            a.* = @intCast(t);
        };
        try f.upload(x.b.ids, 0, R * 4);
        const rows_at = std.mem.alignForward(usize, f.pin_rows * 4, 256);
        for (f.w.layers) |l| if (l.ple) |p| {
            const heads = p.ngram.heads;
            const nid = try f.gpa.alloc(i64, R * heads);
            defer f.gpa.free(nid);
            for (segs[0..windows.len]) |sg| try p.ngram.ids(sg.seq.st.history[0 .. p.ngram.n - 1], sg.seq.window.items, nid[sg.a0 * heads .. sg.a1 * heads]);
            const table = &(f.w.table orelse return error.NoNgramTable);
            if (table.gpu != null) {
                // the ids on the device, then the rank's heads' rows gathered there (W4's Gather, the host bits)
                @memcpy(std.mem.bytesAsSlice(i64, f.pin.bytes[rows_at..][0 .. R * heads * 8]), nid);
                try f.upload(f.sc.ngram_ids, rows_at, R * heads * 8);
                try f.gather.?.run(f.s, table, f.sc.ngram_ids, heads, @intCast(R), x.b.ple_v);
                continue;
            }
            const out = std.mem.bytesAsSlice(u16, f.pin.bytes[rows_at..][0 .. R * heads * table.width * 2]);
            try table.gather(nid, @alignCast(out));
            try f.upload(x.b.ple_v, rows_at, R * heads * table.width * 2);
        };
        for (windows) |w| w.seq.staged = true;
        try f.staged.record(f.s);
        if (!x.b.prefill and f.round != null and f.g.linear_layers > 0) try f.roundTables(segs[0..windows.len], 1 - f.cur_par);
        return R;
    }

    /// forward.stage for one stream (its rows from 0).
    pub fn stage(f: *Forward, seq: *Seq, x: *const Bufs, tokens: []const u32) !void {
        var one: [1]Seg = undefined;
        _ = try f.stageMany(&.{.{ .seq = seq, .tokens = tokens }}, x, &one);
    }

    // -- the forward ------------------------------------------------------------------------------------------

    /// forward.compute on staged rows: the embedding, every layer, then `finish`; the head's logits (`logits`) of
    /// every row (a window) or the last (a prompt chunk), or null. `segs` are the streams' rows (one for a prompt).
    pub fn computeSegs(f: *Forward, segs: []const Seg, x: *const Bufs, logits: bool) !?u64 {
        const b = &x.b;
        const R = segs[segs.len - 1].a1;
        try f.mark(.stage);
        try f.t.embed(b.ids, f.w.embed, b.h, R, f.g.hidden, f.g.streams);
        try f.mark(.embed);
        var pending: ?Pending = null;
        for (f.w.layers, 0..) |*layer, i| {
            f.pf_moe = prefetchHc(if (i + 1 < f.w.layers.len) &f.w.layers[i + 1].attn_hc else &f.w.mixer);
            pending = try f.layerForward(layer, segs, x, R, pending, false);
        }
        return f.finishSegs(&f.w.mixer, x, R, pending.?, logits, segs);
    }

    /// A weight's first bytes to prefetch: all of it up to TF_FLASHNEXT_L2PF_BYTES (default 8 MiB).
    pub fn prefetchOf(r: W.Rows) comms.Prefetch {
        const cap: usize = if (std.c.getenv("TF_FLASHNEXT_L2PF_BYTES")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch (8 << 20) else (8 << 20);
        return .{ .ptr = r.weight, .bytes = @min(@as(usize, r.n) * r.k * 2, cap) };
    }

    /// A read-out's down projection to prefetch: its block-FP8 bytes (fast-fp8), else its bf16 rows.
    pub fn prefetchHc(h: *const W.Hc) comms.Prefetch {
        const d8 = h.down8 orelse return prefetchOf(h.down);
        const cap: usize = if (std.c.getenv("TF_FLASHNEXT_L2PF_BYTES")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch (8 << 20) else (8 << 20);
        return .{ .ptr = d8.w8, .bytes = @min(@as(usize, d8.npad) * d8.k, cap) };
    }

    /// One stream's staged rows [0, R).
    pub fn compute(f: *Forward, seq: *Seq, x: *const Bufs, R: usize, logits: bool) !?u64 {
        return f.computeSegs(&.{.{ .seq = seq, .a0 = 0, .a1 = R }}, x, logits);
    }

    /// The window `tokens` at st.pos ..: staged, then computed (Python forward; the state commits later).
    pub fn forward(f: *Forward, seq: *Seq, x: *const Bufs, tokens: []const u32, logits: bool) !?u64 {
        try f.stage(seq, x, tokens);
        return f.compute(seq, x, tokens.len, logits);
    }

    /// layer_forward: one decoder layer on b.h[:R] (every segment's rows); returns the new pending write-back.
    pub fn layerForward(f: *Forward, layer: *const W.Layer, segs: []const Seg, x: *const Bufs, R: usize, pending: ?Pending, mtp: bool) !Pending {
        const b = &x.b;
        const g = f.g;
        try f.putLayer(layer, "h_in", b.h, R * g.wide() * 2);
        var pend = pending;
        const sp = f.splitOf(x, R);
        if (layer.ple) |*p| {
            if (pend) |pp| {
                try f.writebackRows(b.h, x, R, pp, sp);
                pend = null;
            }
            // the n-gram block runs on every row (its conv reads the rows before each): the halves gathered first
            if (sp) |q| try f.gatherRows(b.h, q, g.wide() * 2, .bf16, 2);
            try f.mark(.hc_writeback);
            try f.pleBlock(p, segs, x, R);
            try f.mark(.ple);
            try f.putLayer(layer, "ple_emb", b.ple_emb, R * g.ple_dim * 2);
            try f.putLayer(layer, "h_ple", b.h, R * g.wide() * 2);
        }
        try f.hcBlock(&layer.attn_hc, x, R, b.h, pend, b.inj_a, sp);
        try f.putLayer(layer, "hc_a", b.mixed, R * g.hidden * 2);
        f.pf_mixer = prefetchHc(&layer.mlp_hc);
        const br = if (layer.linear) try f.gdnBlock(layer, segs, x, R) else try f.attnBlock(layer, segs, x, R, mtp);
        try f.putBranch(layer, if (layer.linear) "gdn" else "attn", br, R);
        try f.hcBlock(&layer.mlp_hc, x, R, b.h, .{ .branch = br, .inject = b.inj_a }, b.inj_m, sp);
        try f.putLayer(layer, "hc_m", b.mixed, R * g.hidden * 2);
        const moe = try f.moeBlock(&layer.moe, x, R, mtp);
        try f.putLayer(layer, "router_logits", b.moe_logits, R * (g.experts + 1) * 4);
        try f.putLayer(layer, "router_pick", b.moe_pick, R * (g.topKFor(mtp) + 1) * 4);
        try f.putLayer(layer, "router_wts", b.moe_wts, R * (g.topKFor(mtp) + 1) * 4);
        try f.putBranch(layer, "moe_y", moe, R);
        return .{ .branch = moe, .inject = b.inj_m };
    }

    fn putBranch(f: *Forward, layer: *const W.Layer, what: []const u8, br: tri.Branch, R: usize) !void {
        if (f.dump == null) return;
        const D = f.g.hidden;
        var buf: [32]u8 = undefined;
        switch (br) {
            .bf16 => |p| try f.putLayer(layer, try std.fmt.bufPrint(&buf, "{s}_m1", .{what}), p, R * D * 2),
            .moe => |m| try f.putLayer(layer, try std.fmt.bufPrint(&buf, "{s}_m2", .{what}), m.y, R * m.slots * D * @as(usize, if (m.y_f32) 4 else 2)),
            .ranks => |r| try f.putLayer(layer, try std.fmt.bufPrint(&buf, "{s}_m3", .{what}), r.part, r.world * R * D * 4),
            .slices => |s| try f.putLayer(layer, try std.fmt.bufPrint(&buf, "{s}_m4", .{what}), s.part, s.sk * R * D * 4),
            .none => {},
        }
    }

    /// _writeback: a pending branch into the streams h (in place), no read-out.
    fn writeback(f: *Forward, h: u64, x: *const Bufs, R: usize, p: Pending) !void {
        try f.t.hcWriteback(h, h, x.b.pss, p.inject, p.branch, R, f.g.hidden, f.g.streams);
    }

    /// `writeback`, split: this rank's rows only (their streams, squared sums and inject gates at their offset).
    fn writebackRows(f: *Forward, h: u64, x: *const Bufs, R: usize, p: Pending, sp: ?Split) !void {
        const q = sp orelse return f.writeback(h, x, R, p);
        const g = f.g;
        try f.t.hcWriteback(h + q.o * g.wide() * 2, h + q.o * g.wide() * 2, x.b.pss + q.o * pssRow(g), p.inject + q.o * g.streams * 2, p.branch, q.rh, g.hidden, g.streams);
    }

    fn pssRow(g: state.Geometry) usize {
        return (g.hidden / 256) * g.streams * 4;
    }

    /// hc_block: the pending branch written back into the streams h, then the hyper-connection's read-out into
    /// b.mixed (and its inject gates into `inject_out` when it has them). Split: this rank's rows, then the read-out
    /// rows of both ranks gathered into b.mixed.
    fn hcBlock(f: *Forward, hc: *const W.Hc, x: *const Bufs, R: usize, h: u64, pending: ?Pending, inject_out: u64, sp: ?Split) !void {
        const g = f.g;
        const b = &x.b;
        if (sp) |q| {
            const o = q.o;
            const hh = h + o * g.wide() * 2;
            const ps = b.pss + o * pssRow(g);
            if (pending) |p| {
                try f.t.hcWriteback(hh, hh, ps, p.inject + o * g.streams * 2, p.branch, q.rh, g.hidden, g.streams);
            } else try f.t.hcWriteback(hh, hh, ps, null, .none, q.rh, g.hidden, g.streams);
            try f.mark(.hc_writeback);
            try f.readoutAt(hc, x, hh, ps, q.rh, if (hc.inject) inject_out + o * g.streams * 2 else null, b.mixed + o * g.hidden * 2, b.xs_mixed + o * (g.hidden / 32) * 4);
            try f.gatherRows(b.mixed, q, g.hidden * 2, .bf16, 2);
            try f.mark(.gather);
            return;
        }
        if (pending) |p| {
            try f.t.hcWriteback(h, h, x.b.pss, p.inject, p.branch, R, g.hidden, g.streams);
        } else try f.t.hcWriteback(h, h, x.b.pss, null, .none, R, g.hidden, g.streams);
        try f.mark(.hc_writeback);
        try f.readout(hc, x, h, R, if (hc.inject) inject_out else null);
    }

    /// _readout_b16: norm, the down projection in fp32, SiLU and the inject gates, the up projection, the mix.
    fn readout(f: *Forward, hc: *const W.Hc, x: *const Bufs, h: u64, R: usize, inject: ?u64) !void {
        try f.readoutAt(hc, x, h, x.b.pss, R, inject, x.b.mixed, x.b.xs_mixed);
    }

    /// `readout` of rows whose squared sums are at `pss`, the mix into `mixed` / `xs` (the scratch from row 0).
    fn readoutAt(f: *Forward, hc: *const W.Hc, x: *const Bufs, h: u64, pss: u64, R: usize, inject: ?u64, mixed: u64, xs: u64) !void {
        const b = &x.b;
        const g = f.g;
        const Wd = g.wide();
        try f.t.hcNormed(h, pss, hc.scale, b.normed, b.xs_normed, R, g.hidden, g.streams, f.eps);
        try f.mark(.hc_normed);
        const dn_n: usize = hc.downN();
        if (hc.down8) |d8| {
            // fast-fp8: the low-rank rows on block FP8 and the inject rows bf16, each into its columns of dn (fp32)
            try fp8.matmul(f.k8 orelse return error.NoFp8Kernels, f.s, b.normed, Wd, d8, f.sc.p8, true, R);
            try f.th.slotCopy(f.sc.p8, d8.n * 4, f.sc.dn, dn_n * 4, d8.n * 4, R);
            if (hc.down.n > 0) {
                try f.mmAt(null, b.normed, Wd, hc.down, f.sc.pb, true, R);
                try f.th.slotCopy(f.sc.pb, hc.down.n * 4, f.sc.dn + d8.n * 4, dn_n * 4, hc.down.n * 4, R);
            }
            try f.mark(.hc_down);
        } else try f.mmxAt(.hc_down, b.normed, Wd, b.xs_normed, hc.down, f.sc.dn, true, R);
        try f.t.hcAct(f.sc.dn, b.act, b.xs_act, inject, R, dn_n, g.streams, g.low);
        try f.mark(.hc_act);
        try f.denseAt(.hc_up, b.act, g.low, b.xs_act, hc.up, hc.up8, b.up, false, R);
        try f.t.hcMix(b.up, b.normed, mixed, xs, R, g.hidden, g.streams);
        try f.mark(.hc_mix);
    }

    /// _out_proj: one GPU, the bf16 branch (mode 1); two ranks, the fp32 partials gathered in rank order (mode 3).
    fn outProj(f: *Forward, x: *const Bufs, in: u64, r: W.Rows, l8: ?fp8.Linear, R: usize) !tri.Branch {
        return f.outProjX(x, in, 0, r, l8, R);
    }

    /// outProj with the input's group sums (the MTP head's o_proj in 4-bit, D5).
    fn outProjX(f: *Forward, x: *const Bufs, in: u64, xs: u64, r: W.Rows, l8: ?fp8.Linear, R: usize) !tri.Branch {
        const b = &x.b;
        if (f.splitOf(x, R)) |q| {
            // this rank's rows into its slot, the peer's rows sent while the peer's partial of ours arrives
            const D = f.g.hidden;
            const cm = f.comm.?;
            _ = cm;
            try f.denseAt(.out_proj, in + q.po * r.k * 2, r.k, 0, r, l8, b.part_branch, true, q.rh);
            try f.exchangeAsync(b.part_branch, b.g_branch + (1 - q.rank) * q.rh * D * 4, q.rh * D, .f32);
            try f.denseAt(.out_proj, in + q.o * r.k * 2, r.k, 0, r, l8, b.g_branch + q.rank * q.rh * D * 4, true, q.rh);
            try f.exchangeWait();
            try f.mark(.gather);
            return .{ .ranks = .{ .part = b.g_branch, .world = 2 } };
        }
        if (f.comm) |cm| {
            try f.denseAt(.out_proj, in, r.k, xs, r, l8, b.part_branch, true, R);
            try cm.gatherPartialsThen(b.part_branch, b.g_branch, R, f.g.hidden, f.s.handle, f.pf_mixer);
            try f.mark(.gather);
            return .{ .ranks = .{ .part = b.g_branch, .world = cm.world } };
        }
        try f.denseAt(.out_proj, in, r.k, xs, r, l8, b.branch, false, R);
        return .{ .bf16 = b.branch };
    }

    /// ple_block: h += the n-gram embedding branch through the stream's conv tail (rows staged by `stage`).
    fn pleBlock(f: *Forward, p: *const W.Ple, segs: []const Seg, x: *const Bufs, R: usize) !void {
        const b = &x.b;
        const g = f.g;
        const scale: f32 = if (f.w.table) |t| (if (t.fp8) 1.0 else t.scale) else 1.0;
        const table = f.w.table.?;
        if (table.gpu) |gp| {
            // two ranks: this rank's heads embedded (each head alone, so the split is exact), both ranks' halves
            // gathered in rank (head) order, then interleaved into the [R, heads * dh] rows and their group sums
            const cm = f.comm orelse return error.TwoRanksNeedComm;
            const half = gp.heads * p.ngram.dims;
            try f.t.pleEmbedBf16(b.ple_v, f.sc.ple_half, f.sc.xs_half, R, gp.heads, p.ngram.dims, scale);
            try cm.allGather(f.sc.ple_half, f.sc.ple_all, R * half, .bf16, f.s.handle);
            try cm.allGather(f.sc.xs_half, f.sc.xs_all, R * half / 32, .f32, f.s.handle);
            for (0..cm.world) |r| {
                try f.th.slotCopy(f.sc.ple_all + r * R * half * 2, half * 2, b.ple_emb + r * half * 2, cm.world * half * 2, half * 2, R);
                try f.th.slotCopy(f.sc.xs_all + r * R * (half / 32) * 4, (half / 32) * 4, b.xs_ple + r * (half / 32) * 4, cm.world * (half / 32) * 4, (half / 32) * 4, R);
            }
        } else try f.t.pleEmbedBf16(b.ple_v, b.ple_emb, b.xs_ple, R, p.ngram.heads, p.ngram.dims, scale);
        try f.mm(b.ple_emb, g.ple_dim, p.key, b.ple_keys, false, R);
        try f.mm(b.ple_emb, g.ple_dim, p.value, b.ple_vals, false, R);
        try f.t.pleGate(b.ple_keys, b.ple_vals, b.h, p.norm_key, p.norm_query, b.ple_gated, b.ple_pss, R, g.hidden, g.streams, f.eps);
        const Wd = g.wide();
        for (segs) |sg| {
            const o = sg.a0;
            try f.t.pleConv(b.ple_gated + o * Wd * 2, b.ple_pss + o * g.streams * 4, p.norm_conv, sg.seq.st.ple_tail, p.conv, b.h + o * Wd * 2, b.h + o * Wd * 2, b.ple_nrow + o * Wd * 2, sg.a1 - sg.a0, g.hidden, g.streams, g.ple_kernel, g.ngram_size, f.eps);
        }
    }

    /// gdn_block: the projection, then the decode chain (rows saved for the replay at commit) or, in a prompt chunk,
    /// the front, the chunk-invariant chain and the back, the layer committed at once.
    fn gdnBlock(f: *Forward, layer: *const W.Layer, segs: []const Seg, x: *const Bufs, R: usize) !tri.Branch {
        const b = &x.b;
        const g = f.g;
        const gd = &layer.gdn.?;
        const li = f.sub[@intCast(layer.index)];
        const pw = g.gdnWidth();
        const nv = g.nv;
        if (b.prefill) {
            // the projection over every row, then each prompt's own chain: its conv taps and state, as alone (the
            // front and back are row-wise, the chain runs over one sequence's rows; the segment's rows start the
            // windows table over again, so a prompt's taps never read the rows before it)
            try f.projFace(.gdn_proj, b.mixed, g.hidden, gd.proj8, gd.proj, b.proj, pw, R);
            const C = g.gdnConv();
            for (segs) |sg| {
                const st = &sg.seq.st;
                const o = sg.a0;
                const n = sg.a1 - sg.a0;
                if (sg.seq.held != 0) return error.UnfoldedRows; // a prompt starts from reset or restore
                const p = b.proj + o * pw * 2;
                const qo = o * g.nk * state.gdn_dk * 4;
                const vo = o * nv * state.gdn_dv * 2;
                const so = o * nv * 4;
                try f.th.fill64(b.conv_ptr, f.conv(st, li), 1);
                try f.ops.gdnFront(nv, p, b.conv_ptr, b.sid, b.windows, n, gd.conv, gd.a_log, gd.dt_bias, f.sc.gq + qo, f.sc.gk + qo, f.sc.gv + vo, f.sc.gg + so, f.sc.gb + so);
                try f.mark(.gdn_front);
                // the chunk-invariant chain into the scratch state, then over the sequence's one state
                try f.ops.gdnPrefill(true, f.sc.gq + qo, f.sc.gk + qo, f.sc.gv + vo, f.sc.gg + so, f.sc.gb + so, f.rec(st, li), f.sc.gdn_t, f.sc.gy + vo, n, g.nk, nv);
                try f.th.copy(f.rec(st, li), f.sc.gdn_t, nv * state.gdn_dv * state.gdn_dk * 4);
                try f.mark(.gdn_chain);
                try f.ops.gdnBack(nv, f.sc.gy + vo, p, gd.norm, f.eps, b.gout + vo, b.gxs + o * (nv * state.gdn_dv / 32) * 4, n);
                try f.mark(.gdn_back);
                try f.t.shiftWindows(f.conv(st, li), p, n, (g.conv_kernel - 1) * C, b.rows * pw, pw, 1, C, g.conv_kernel - 1);
                try f.mark(.gdn_shift);
            }
            return f.outProj(x, b.gout, gd.out, gd.out8, R);
        }
        const proj = b.proj + li * b.rows * pw * 2;
        if (gd.proj8 != null) try f.projFace(.gdn_proj, b.mixed, g.hidden, gd.proj8, gd.proj, proj, pw, R) else {
            try f.mm(b.mixed, g.hidden, gd.proj, proj, false, R);
            try f.mark(.gdn_proj);
        }
        // every stream's rows in one launch a step (gdn_multi.block): conv taps and states found by table, each
        // tree first folding in its stream's last kept rows (in place), the window's inputs kept for the next fold
        const r = &(f.round orelse return error.NoRoundScratch);
        const n = segs.len;
        const p: usize = f.cur_par;
        var most: usize = 1;
        for (segs) |sg| most = @max(most, sg.a1 - sg.a0);
        const at = r.layer(p, li);
        const prev = r.layer(1 - p, li);
        try f.ops.gdnFront(nv, proj, r.conv + li * n * 8, r.sid, r.win, R, gd.conv, gd.a_log, gd.dt_bias, r.q, at.k, at.v, at.g, at.beta);
        try f.mark(.gdn_front);
        try f.ops.gdnTree(r.q, at.k, at.v, at.g, at.beta, 0, r.state + li * n * 8, r.starts, r.starts, R, 0, n, most, f.sc.gy, g.nk, nv, state.gdn_dv, .{ .k = prev.k, .v = prev.v, .g = prev.g, .beta = prev.beta, .rows = r.held, .row_stride = @intCast(Round.held_stride), .counts = r.counts, .count_stride = 1 }, 0, 0);
        try f.mark(.gdn_chain);
        try f.ops.gdnBack(nv, f.sc.gy, proj, gd.norm, f.eps, b.gout, b.gxs, R);
        try f.mark(.gdn_back);
        return f.outProj(x, b.gout, gd.out, gd.out8, R);
    }

    /// attn_block: the projection, attn_prep (norms, rope, cache writes), the indexer's pool/select when the
    /// capacity passes its budget, the chunked attention and merge (prompt chunks in 256-row blocks), the gate.
    fn attnBlock(f: *Forward, layer: *const W.Layer, segs: []const Seg, x: *const Bufs, R: usize, mtp: bool) !tri.Branch {
        const b = &x.b;
        const g = f.g;
        const a = &layer.attn.?;
        if (a.proj8 != null) try f.projFace(.attn_proj, b.mixed, g.hidden, a.proj8, a.proj, b.pa, g.attnWidth(), R) else try f.mmxAt(.attn_proj, b.mixed, g.hidden, b.xs_mixed, a.proj, b.pa, false, R);
        const ci: usize = if (mtp) g.attention_layers else f.sub[@intCast(layer.index)];
        const d = f.dims;
        const ag = x.attn;
        const sc: tri.AttnScratch = .{ .po = b.att_po, .pm = b.att_pm, .pl = b.att_pl, .ids = b.att_ids, .nk = b.att_nk, .sparse = b.att_sparse, .scores = b.att_scores };
        const qrow = g.heads * g.head_dim * 2;
        const irow = g.index_heads * g.index_dim * 2;
        const prow = g.attnWidth() * 2;
        var o = b.att_out;
        // eager rounds only: a graph's launches are fixed at capture, the sparse selects and chunks are not
        if (!b.prefill and !mtp and f.round_multi and f.contexts == null and f.context == null) {
            try f.attnMulti(layer, segs, x, R);
            o = b.attn_o;
            try f.t.attnGate(o, b.pa, b.gated, b.xs_gated, R, d);
            try f.mark(.attn_gate);
            return f.outProjX(x, b.gated, b.xs_gated, a.o, a.o8, R);
        }
        for (segs, 0..) |sg, si| {
            const st = &sg.seq.st;
            const n = sg.a1 - sg.a0;
            const pos = if (mtp) st.mtp_pos else st.pos_dev;
            const host_pos = if (mtp) st.mtp_len else st.pos;
            const cache: tri.Cache = .{ .k = st.kc[ci], .v = st.vc[ci], .ks = f.sc.kv_scale, .vs = f.sc.kv_scale };
            const ikc = st.ikc[ci];
            const pooled = st.pooled[ci];
            const q = b.q + sg.a0 * qrow;
            const iq = b.iq + sg.a0 * irow;
            try f.t.attnPrep(.{ .p = b.pa + sg.a0 * prow, .pos0 = pos, .qw = a.q_scale, .kw = a.k_scale, .iw = a.iq_scale, .inv = f.w.inv_freq, .q = q, .cache = cache, .iq = iq, .ikc = ikc }, n, d);
            try f.mark(.attn_prep);
            if (b.prefill) {
                if (ag.qsa) try f.t.qsaPool(ikc, pooled, pos, a.ik_scale, f.w.inv_freq, n, ag, d);
                try f.mark(.qsa_pool);
                if (f.splitOf(x, R)) |sp| if (segs.len == 1 and f.cs != null and f.overlap & 1 != 0) {
                    // the peer's rows first (their 256-row blocks, the same blocks as below), their gate and
                    // out_proj, sent while this rank's own blocks run
                    const D = g.hidden;
                    const pa = sp.po;
                    const pb = @min(sp.po + sp.rh, n);
                    var phase: usize = 0;
                    while (phase < 2) : (phase += 1) {
                        var r1: usize = 0;
                        while (r1 < n) : (r1 += state.att_rows) {
                            const m = @min(state.att_rows, n - r1);
                            const peer = r1 < pb and r1 + m > pa;
                            if (peer != (phase == 0)) continue;
                            try f.th.fill32(b.pos_blk, @intCast(host_pos + r1), 1);
                            const ends = host_pos + r1 + m;
                            if (ag.qsa) try f.qsaBlock(iq + r1 * irow, pooled, b.pos_blk, host_pos + r1, sc, m, ag, ends);
                            try f.t.attention(q + r1 * qrow, cache, b.pos_blk, sc, b.attn_o + r1 * qrow, m, ag, ends, d);
                            try f.mark(.attention);
                        }
                        const rows0 = if (phase == 0) sp.po else sp.o;
                        try f.t.attnGate(b.attn_o + rows0 * qrow, b.pa + rows0 * prow, b.gated + rows0 * qrow, b.xs_gated + rows0 * (qrow / 64) * 4, sp.rh, d);
                        try f.mark(.attn_gate);
                        if (phase == 0) {
                            try f.mmAt(.out_proj, b.gated + sp.po * a.o.k * 2, a.o.k, a.o, b.part_branch, true, sp.rh);
                            try f.exchangeAsync(b.part_branch, b.g_branch + (1 - sp.rank) * sp.rh * D * 4, sp.rh * D, .f32);
                        } else {
                            try f.mmAt(.out_proj, b.gated + sp.o * a.o.k * 2, a.o.k, a.o, b.g_branch + sp.rank * sp.rh * D * 4, true, sp.rh);
                            try f.exchangeWait();
                            try f.mark(.gather);
                        }
                    }
                    return .{ .ranks = .{ .part = b.g_branch, .world = 2 } };
                };
                var r0: usize = 0;
                while (r0 < n) : (r0 += state.att_rows) {
                    const m = @min(state.att_rows, n - r0);
                    try f.th.fill32(b.pos_blk, @intCast(host_pos + r0), 1);
                    const ends = host_pos + r0 + m;
                    if (ag.qsa) try f.qsaBlock(iq + r0 * irow, pooled, b.pos_blk, host_pos + r0, sc, m, ag, ends);
                    try f.t.attention(q + r0 * qrow, cache, b.pos_blk, sc, b.attn_o + (sg.a0 + r0) * qrow, m, ag, ends, d);
                    try f.mark(.attention);
                }
                continue;
            }
            const keys = if (f.contexts) |cs| cs[si] else f.context orelse host_pos + n;
            if (ag.qsa) try f.t.qsaSelect(iq, ikc, pooled, pos, a.ik_scale, f.w.inv_freq, sc, n, ag, keys, d);
            try f.t.attention(q, cache, pos, sc, b.att_out, n, ag, keys, d);
            // the scratch output is the next stream's too
            if (segs.len > 1) try f.th.copy(b.attn_o + sg.a0 * qrow, b.att_out, n * qrow);
            try f.mark(.attention);
        }
        if (b.prefill or segs.len > 1) o = b.attn_o;
        try f.t.attnGate(o, b.pa, b.gated, b.xs_gated, R, d);
        try f.mark(.attn_gate);
        return f.outProjX(x, b.gated, b.xs_gated, a.o, a.o8, R);
    }

    /// The indexer's scores and key lists of a prompt block's `m` rows (first row at `pos`, its device copy `pos0`).
    /// Two ranks past `qsa_split_keys`: each rank scores and selects its half of the rows (rank 0 the first) and the
    /// lists, lengths and flags are exchanged: the same kernels on the same rows, so the same lists.
    fn qsaBlock(f: *Forward, iq: u64, pooled: u64, pos0: u64, pos: usize, sc: tri.AttnScratch, m: usize, ag: tri.AttnGeometry, ends: usize) !void {
        const d = f.dims;
        const irow = d.index_heads * d.index_dim * 2;
        const split = f.comm != null and f.qsa_split_keys > 0 and ends >= f.qsa_split_keys and m >= 32;
        if (!split) {
            if (f.prof == null and f.qsa_rt == 0 and ag.blocks(ends) < f.qsa_tiles_from) return f.t.qsaRows(iq, pooled, pos0, sc, m, ag, ends, d);
            try f.qsaScoresSelect(iq, pooled, pos0, sc, m, ag, ends);
            return;
        }
        const cm = f.comm.?;
        const h = std.mem.alignForward(usize, (m + 1) / 2, 16);
        const own0: usize = if (cm.rank == 0) 0 else h;
        const own1: usize = if (cm.rank == 0) h else m;
        const peer0: usize = if (cm.rank == 0) h else 0;
        const peer1: usize = if (cm.rank == 0) m else h;
        try f.th.fill32(f.sc.qsa_pos, @intCast(pos + own0), 1);
        const so = prompt.scratchAt(sc, ag, own0);
        try f.qsaScoresSelect(iq + own0 * irow, pooled, f.sc.qsa_pos, so, own1 - own0, ag, ends);
        const sp = prompt.scratchAt(sc, ag, peer0);
        const no = own1 - own0;
        const np = peer1 - peer0;
        try cm.exchangeBytes(&.{ .{ so.ids, no * ag.idw * 4 }, .{ so.nk, no * 4 }, .{ so.sparse, no * 4 } }, &.{ .{ sp.ids, np * ag.idw * 4 }, .{ sp.nk, np * 4 }, .{ sp.sparse, np * 4 } }, f.s.handle);
        try f.mark(.qsa_exchange);
    }

    fn qsaScoresSelect(f: *Forward, iq: u64, pooled: u64, pos0: u64, sc: tri.AttnScratch, m: usize, ag: tri.AttnGeometry, ends: usize) !void {
        if (f.qsa_rt > 0 and m >= 16) {
            try prompt.qsaScoresRows(f.t, iq, pooled, pos0, sc, m, ag, ends, f.dims, f.qsa_rt);
        } else try prompt.qsaScores(f.t, iq, pooled, pos0, sc, m, ag, ends, f.dims);
        try f.mark(.qsa_scores);
        if (ag.blocks(ends) >= f.qsa_tiles_from) {
            try prompt.qsaSelectTiles(f.t, pos0, sc, m, ag);
        } else try prompt.qsaSelect(f.t, pos0, sc, m, ag, ends);
        try f.mark(.qsa_select);
    }

    /// attn_multi.layer: every stream's rows of a decode round in one launch a kernel (prep, pool, the sparse
    /// streams' own selects, chunks, merge) into b.attn_o; the round's tables from multiTables.
    fn attnMulti(f: *Forward, layer: *const W.Layer, segs: []const Seg, x: *const Bufs, R: usize) !void {
        const b = &x.b;
        const g = f.g;
        const a = &layer.attn.?;
        const d = f.dims;
        const ag = x.attn;
        const r = &(f.round orelse return error.NoRoundScratch);
        const n = segs.len;
        const ai: usize = f.sub[@intCast(layer.index)];
        const cp = r.aptr + ai * multi_ptrs * n * 8;
        const m: tri.Tri.Multi = .{ .posr = r.posr, .sid = r.sid, .first = r.first, .counts = r.acounts, .n = n };
        try f.t.prepMulti(m, cp, b.pa, a.q_scale, a.k_scale, a.iq_scale, f.w.inv_freq, b.q, b.iq, R, d);
        try f.mark(.attn_prep);
        const top = ag.budget / ag.ratio;
        const irow = g.index_heads * g.index_dim * 2;
        var keys: usize = 0;
        var most: usize = 0;
        for (segs) |sg| {
            keys = @max(keys, sg.seq.st.pos + sg.a1 - sg.a0);
            most = @max(most, sg.a1 - sg.a0);
        }
        if (ag.qsa) {
            try f.t.poolMulti(m, cp, a.ik_scale, f.w.inv_freq, most, ag, d);
            try f.mark(.qsa_pool);
            // a stream whose rows pass TOP blocks selects its own keys (the one-stream kernels on its rows)
            for (segs) |sg| {
                const st = &sg.seq.st;
                const end = st.pos + sg.a1 - sg.a0;
                if (end / ag.ratio <= top) continue;
                const a0 = sg.a0;
                const sc: tri.AttnScratch = .{ .po = b.att_po, .pm = b.att_pm, .pl = b.att_pl, .ids = b.att_ids + a0 * ag.idw * 4, .nk = b.att_nk + a0 * 4, .sparse = b.att_sparse + a0 * 4, .scores = b.att_scores + a0 * ag.nb * 4 };
                try f.t.qsaRows(b.iq + a0 * irow, st.pooled[ai], st.pos_dev, sc, sg.a1 - sg.a0, ag, end, d);
            }
            try f.mark(.qsa_rows);
        }
        const sc: tri.AttnScratch = .{ .po = b.att_po, .pm = b.att_pm, .pl = b.att_pl, .ids = b.att_ids, .nk = b.att_nk, .sparse = b.att_sparse, .scores = b.att_scores };
        try f.t.attentionMulti(m, cp, b.q, sc, b.attn_o, R, ag, keys, d);
        try f.mark(.attention);
    }

    /// moe_block (nvfp4_moe.moe): the router with the shared expert's gate, each row's top_k and the shared slot,
    /// the plan, the routed gate/up SwiGLU, the shared expert's SwiGLU into its slot, the routed down, the shared
    /// down into its slot. One GPU: the pending (mode 2, slots y, weights); two ranks: the gathered partials. `mtp`:
    /// the MTP layer (its own top-k with TF_FLASHNEXT_MTP_TOPK; every stride follows the call's slots).
    fn moeBlock(f: *Forward, m: *const W.MoE, x: *const Bufs, R: usize, mtp: bool) !tri.Branch {
        const b = &x.b;
        const g = f.g;
        const D = g.hidden;
        const E = g.experts;
        const top = g.topKFor(mtp);
        const slots = top + 1;
        const es: usize = if (x.y_f32) 4 else 2;
        try f.t.router(b.mixed, D, m.router, b.moe_logits, R, D, E + 1);
        try f.mark(.router);
        try f.t.topkRows(b.moe_logits, b.moe_pick, b.moe_wts, R, E, top);
        if (f.count_experts and f.prof != null and !b.prefill) try f.countExperts(b.moe_pick, R, slots, E);
        if (f.splitOf(x, R)) |q| if (f.cs != null and f.overlap & 2 != 0) {
            // the peer's rows first: their experts and partial, sent while this rank's own rows run
            const yrow = slots * D * es;
            try f.moeRows(m, x, q.po, @min(q.rh, R - q.po), top);
            try f.t.moePartial(b.moe_y + q.po * yrow, x.y_f32, b.moe_wts + q.po * slots * 4, b.part_moe, q.rh, D, slots);
            try f.mark(.moe_partial);
            try f.exchangeAsync(b.part_moe, b.g_moe + (1 - q.rank) * q.rh * D * 4, q.rh * D, .f32);
            try f.moeRows(m, x, q.o, @min(q.rh, R - q.o), top);
            try f.t.moePartial(b.moe_y + q.o * yrow, x.y_f32, b.moe_wts + q.o * slots * 4, b.g_moe + q.rank * q.rh * D * 4, q.rh, D, slots);
            try f.mark(.moe_partial);
            try f.exchangeWait();
            try f.mark(.moe_gather);
            return .{ .ranks = .{ .part = b.g_moe, .world = 2 } };
        };
        try f.moeRows(m, x, 0, R, top);
        if (f.splitOf(x, R)) |q| {
            // the peer's rows' partial first, sent while this rank's own rows' partial runs
            const yrow = slots * D * es;
            try f.t.moePartial(b.moe_y + q.po * yrow, x.y_f32, b.moe_wts + q.po * slots * 4, b.part_moe, q.rh, D, slots);
            try f.exchangeAsync(b.part_moe, b.g_moe + (1 - q.rank) * q.rh * D * 4, q.rh * D, .f32);
            try f.t.moePartial(b.moe_y + q.o * yrow, x.y_f32, b.moe_wts + q.o * slots * 4, b.g_moe + q.rank * q.rh * D * 4, q.rh, D, slots);
            try f.mark(.moe_partial);
            try f.exchangeWait();
            try f.mark(.moe_gather);
            return .{ .ranks = .{ .part = b.g_moe, .world = 2 } };
        }
        if (f.comm) |cm| {
            try f.t.moePartial(b.moe_y, x.y_f32, b.moe_wts, b.part_moe, R, D, slots);
            try f.mark(.moe_partial);
            try cm.gatherPartialsThen(b.part_moe, b.g_moe, R, D, f.s.handle, f.pf_moe);
            try f.mark(.moe_gather);
            return .{ .ranks = .{ .part = b.g_moe, .world = cm.world } };
        }
        return .{ .moe = .{ .y = b.moe_y, .y_f32 = x.y_f32, .wts = b.moe_wts, .slots = slots } };
    }

    /// TF_FLASHNEXT_COUNT_EXPERTS: this layer's distinct routed experts over its R rows' picks (the shared slot apart).
    fn countExperts(f: *Forward, picks: u64, R: usize, slots: usize, E: usize) !void {
        const host = try f.gpa.alloc(i32, R * slots);
        defer f.gpa.free(host);
        try f.s.synchronize();
        const buf: cuda.DeviceBuffer = .{ .d = f.d, .ptr = picks, .len = host.len * 4 };
        try buf.download(0, std.mem.sliceAsBytes(host));
        var seen = try std.DynamicBitSet.initEmpty(f.gpa, E + 1);
        defer seen.deinit();
        for (host) |x| if (x >= 0 and x < E) seen.set(@intCast(x));
        const c = &f.experts_seen[@intFromBool(f.prof.?.mtp)];
        c.layers += 1;
        c.distinct += seen.count();
        c.rows += R;
    }

    /// The experts, the shared expert and the slots of rows [a0, a0 + n) (the router's picks made for every row), `top`
    /// routed experts a row.
    fn moeRows(f: *Forward, m: *const W.MoE, x: *const Bufs, a0: usize, n: usize, top: usize) !void {
        if (m.int4) |ex4| return f.moeRowsInt4(m, ex4, x, a0, n, top);
        const b = &x.b;
        const g = f.g;
        const D = g.hidden;
        const E = g.experts;
        const slots = top + 1;
        const ni = g.moe_width;
        const es: usize = if (x.y_f32) 4 else 2;
        const R = n;
        const pick = b.moe_pick + a0 * slots * 4;
        const xin = b.mixed + a0 * D * 2;
        const act = b.moe_act + a0 * slots * ni * 2;
        const y = b.moe_y + a0 * slots * D * es;
        const plan: kern.Plan = .{ .members = b.plan_members, .items = b.plan_items, .counts = b.plan_counts, .rank = b.plan_rank, .hist = b.plan_hist };
        // prompt chunks: gate/up and down on cuda_moe_prompt's items from their row counts (the same bits)
        const gu_px: ?*const moep.Prompt4 = if (f.px) |q| (if (R >= @max(moep.min_rows, q.gu_rows)) q else null) else null;
        const dn_px: ?*const moep.Prompt4 = if (f.px) |q| (if (R >= @max(moep.min_rows, q.down_rows)) q else null) else null;
        if (gu_px) |q| try q.plan(f.ops, pick, R * slots, E + 1, plan) else try f.ops.plan(pick, R * slots, E + 1, kern.plan_tile, plan);
        const ex: kern.Experts4 = .{ .up = m.routed.up, .down = m.routed.down, .up_scale = m.routed.up_scale, .down_scale = m.routed.down_scale, .width = m.routed.width, .dims = m.routed.dims };
        if (ex.width != ni or ex.dims != D) return error.ExpertShape;
        const skip: c_int = @intCast(E);
        try f.mark(.topk_plan);
        const gu = m.shared.gu;
        const dn = m.shared.down;
        const shared_act = act + top * ni * 2;
        if (f.side) |sd| {
            // the shared expert on the side stream: its slot of moe_act and moe_y, which the routed kernels skip
            try sd.fork.record(f.s);
            try sd.s.wait(sd.fork);
            var t2 = f.t;
            t2.s = sd.s;
            var th2 = f.th;
            th2.s = sd.s;
            try f.fp4On(t2, th2, xin, D, .{ .weight = gu.weight, .scale = gu.scale, .scale2 = gu.scale2 }, f.sc.shared_g, false, R, gu.n, gu.k);
            try th2.sharedSwiglu(f.sc.shared_g, shared_act, R, ni, slots * ni);
            try f.fp4On(t2, th2, shared_act, slots * ni, .{ .weight = dn.weight, .scale = dn.scale, .scale2 = dn.scale2 }, f.sc.shared_y, x.y_f32, R, dn.n, dn.k);
            try th2.slotCopy(f.sc.shared_y, D * es, y + top * D * es, slots * D * es, D * es, R);
            try sd.join.record(sd.s);
        }
        if (gu_px) |q| try q.gateUp(f.s, xin, D, ex, plan, slots, E + 1, act, R, skip) else try f.ops.nvfp4GateUp(xin, D, ex, plan, slots, E + 1, act, R, skip);
        try f.mark(.experts_gate_up);
        if (f.side == null) {
            try f.fp4At(.shared_gate_up, xin, D, .{ .weight = gu.weight, .scale = gu.scale, .scale2 = gu.scale2 }, f.sc.shared_g, false, R, gu.n, gu.k);
            try f.th.sharedSwiglu(f.sc.shared_g, shared_act, R, ni, slots * ni);
            try f.mark(.shared_swiglu);
        }
        if ((gu_px == null) != (dn_px == null)) {
            if (dn_px) |q| try q.plan(f.ops, pick, R * slots, E + 1, plan) else try f.ops.plan(pick, R * slots, E + 1, kern.plan_tile, plan);
        }
        if (dn_px) |q| try q.down(f.s, act, ni, ex, plan, slots, E + 1, y, x.y_f32, R, skip) else try f.ops.nvfp4Down(act, ni, ex, plan, slots, E + 1, y, x.y_f32, R, skip);
        try f.mark(.experts_down);
        if (f.side) |sd| {
            try f.s.wait(sd.join);
            try f.mark(.shared_down);
        } else {
            try f.fp4At(.shared_down, shared_act, slots * ni, .{ .weight = dn.weight, .scale = dn.scale, .scale2 = dn.scale2 }, f.sc.shared_y, x.y_f32, R, dn.n, dn.k);
            try f.th.slotCopy(f.sc.shared_y, D * es, y + top * D * es, slots * D * es, D * es, R);
            try f.mark(.slot_copy);
        }
    }

    /// INT4-AutoRound's experts of rows [a0, a0 + n): the GPTQ int4 gate/up SwiGLU, the block-FP8 shared expert's
    /// gate|up and SwiGLU (its own scratch: the healed shared expert is wider than the routed ones), the int4 down,
    /// the shared down into its slot.
    fn moeRowsInt4(f: *Forward, m: *const W.MoE, ex4: int4.Experts, x: *const Bufs, a0: usize, n: usize, top: usize) !void {
        const b = &x.b;
        const g = f.g;
        const D = g.hidden;
        const E = g.experts;
        const slots = top + 1;
        const ni = g.moe_width;
        const es: usize = if (x.y_f32) 4 else 2;
        const R = n;
        const pick = b.moe_pick + a0 * slots * 4;
        const xin = b.mixed + a0 * D * 2;
        const act = b.moe_act + a0 * slots * ni * 2;
        const y = b.moe_y + a0 * slots * D * es;
        const plan: kern.Plan = .{ .members = b.plan_members, .items = b.plan_items, .counts = b.plan_counts, .rank = b.plan_rank, .hist = b.plan_hist };
        const k4 = f.k4 orelse return error.NoInt4Kernels;
        const k8 = f.k8 orelse return error.NoFp8Kernels;
        if (ex4.width != ni or ex4.dims != D) return error.ExpertShape;
        const skip: c_int = @intCast(E);
        // items of 16 pairs (int4.tileFor; the prompt tile is opt-in, the same bits)
        const tile = int4.tileFor(R * slots);
        try f.ops.plan(pick, R * slots, E + 1, tile, plan);
        try f.mark(.topk_plan);
        try int4.gateUp(k4, f.s, xin, D, ex4, plan, slots, E + 1, act, R, skip, tile);
        try f.mark(.experts_gate_up);
        const sh = m.shared8 orelse return error.NoSharedExpert;
        try fp8.matmul(k8, f.s, xin, D, sh.gu, f.sc.sh_g, false, R);
        try f.th.sharedSwiglu(f.sc.sh_g, f.sc.sh_a, R, sh.width, sh.width);
        try f.mark(.shared_swiglu);
        try int4.down(k4, f.s, act, ex4.width, ex4, plan, slots, E + 1, y, x.y_f32, R, skip, tile);
        try f.mark(.experts_down);
        try fp8.matmul(k8, f.s, f.sc.sh_a, sh.width, sh.down, f.sc.shared_y, x.y_f32, R);
        try f.mark(.shared_down);
        try f.th.slotCopy(f.sc.shared_y, D * es, y + top * D * es, slots * D * es, D * es, R);
        try f.mark(.slot_copy);
    }

    /// finish: the last write-back into a copy of the streams (b.streams), the mixer's read-out (a prompt chunk's
    /// last row only), the head (`logits`), the candidates gathered across ranks.
    pub fn finish(f: *Forward, mixer: *const W.Hc, x: *const Bufs, R: usize, pending: Pending, logits: bool) !?u64 {
        return f.finishSegs(mixer, x, R, pending, logits, &.{});
    }

    /// `finish` of a prompt chunk that carries several prompts (`segs`, each ending its own prompt): the read-out
    /// and the head on each prompt's last row, one row a prompt (b.logits and the candidates hold `segs.len` rows).
    pub fn finishSegs(f: *Forward, mixer: *const W.Hc, x: *const Bufs, R: usize, pending: Pending, logits: bool, segs: []const Seg) !?u64 {
        const b = &x.b;
        const g = f.g;
        const Wd = g.wide();
        try f.mark(.other);
        if (f.splitOf(x, R)) |q| {
            // this rank's rows written back, then both halves of the streams and their squared sums gathered
            try f.th.copy(b.streams + q.o * Wd * 2, b.h + q.o * Wd * 2, q.rh * Wd * 2);
            try f.writebackRows(b.streams, x, R, pending, q);
            try f.gatherRows(b.streams, q, Wd * 2, .bf16, 2);
            try f.gatherRows(b.pss, q, pssRow(g), .f32, 4);
        } else {
            try f.th.copy(b.streams, b.h, R * Wd * 2);
            try f.writeback(b.streams, x, R, pending);
        }
        try f.mark(.finish);
        var rows = b.streams;
        var n = R;
        if (b.prefill) {
            const prow = (g.hidden / 256) * g.streams * 4;
            if (segs.len > 1) {
                if (segs.len > state.max_rows) return error.TooManyPrompts;
                // the last rows gathered into rows 0.. of h (free once the streams hold its rows) and of the sums
                // (a row's source is never before its slot, so the copies in order do not overwrite one to come)
                for (segs, 0..) |sg, i| {
                    try f.th.copy(b.h + i * Wd * 2, b.streams + (sg.a1 - 1) * Wd * 2, Wd * 2);
                    if (sg.a1 - 1 != i) try f.th.copy(b.pss + i * prow, b.pss + (sg.a1 - 1) * prow, prow);
                }
                rows = b.h;
                n = segs.len;
            } else {
                try f.th.copy(b.pss, b.pss + (R - 1) * prow, prow);
                rows = b.streams + (R - 1) * Wd * 2;
                n = 1;
            }
        }
        try f.readout(mixer, x, rows, n, null);
        try f.put("final_streams", b.streams, R * Wd * 2);
        try f.put("final_mixed", b.mixed, n * g.hidden * 2);
        if (!logits) return null;
        const head = f.w.head;
        try f.headMm(b.mixed, g.hidden, b.logits, n);
        try f.mark(.head);
        if (f.comm) |cm| {
            try f.candidates(x, b.logits, n, head.n, null, f.vocab_offset, cm);
            try f.mark(.cands);
        }
        try f.put("logits", b.logits, n * head.n * 2);
        return b.logits;
    }

    /// forward.candidates: each rank's top CAND values, global ids and log-sum-exp of `rows` rows, gathered into
    /// b.cand_all [world, rows, 2 CAND + 1].
    pub fn candidates(f: *Forward, x: *const Bufs, logits: u64, rows: usize, columns: usize, id_map: ?u64, offset: i64, cm: *const comms.Comm) !void {
        const b = &x.b;
        const s = f.sample orelse return error.NoSampleScratch;
        try f.th.rankCandidates(logits, rows, columns, state.cand, id_map, offset, s, b.cand);
        try cm.allGather(b.cand, b.cand_all, rows * (2 * state.cand + 1), .f32, f.s.handle);
    }

    // -- commit -----------------------------------------------------------------------------------------------

    /// forward.commit: keep the first `keep` of the window's R rows (a prompt chunk keeps all, its DeltaNet layers
    /// committed during the forward): the replays, the conv windows' shift, the n-gram history and tail, pos.
    pub fn commit(f: *Forward, seq: *Seq, x: *const Bufs, R: usize, keep: usize) !void {
        return f.commitAt(seq, x, R, keep, 0);
    }

    /// commit(..., at): the stream ran the window's rows [at, at + R) of a shared round.
    pub fn commitAt(f: *Forward, seq: *Seq, x: *const Bufs, R: usize, keep: usize, at: usize) !void {
        const b = &x.b;
        const g = f.g;
        const st = &seq.st;
        if (keep < 1 or keep > R or (b.prefill and keep != R)) return error.BadKeep;
        if (!b.prefill) {
            // the kept rows wait in the round scratch; the stream's next tree (or a flush) folds them in
            seq.held = keep;
            seq.held_a0 = at;
            seq.held_par = f.cur_par;
            const C = g.gdnConv();
            const pw = g.gdnWidth();
            try f.t.shiftWindows(st.conv, b.proj + at * pw * 2, keep, (g.conv_kernel - 1) * C, b.rows * pw, pw, g.linear_layers, C, g.conv_kernel - 1);
        }
        if (seq.staged) {
            for (f.w.layers) |l| if (l.ple) |p| p.ngram.advance(st.history[0 .. p.ngram.n - 1], seq.window.items[0..keep]);
            seq.staged = false;
            const Wd = g.wide();
            const taps = (g.ple_kernel - 1) * g.ngram_size;
            try f.t.shiftWindows(st.ple_tail, b.ple_nrow + at * Wd * 2, keep, taps * Wd, b.rows * Wd, Wd, 1, Wd, taps);
        }
        try f.setPos(st, st.pos + keep);
    }
};

test "the partial scratch covers the widest split-K matmul of a prompt chunk" {
    const g: state.Geometry = .{ .hidden = 2560, .streams = 4, .low = 320, .heads = 24, .kv_heads = 2, .head_dim = 256, .index_heads = 4, .index_dim = 128, .index_budget = 2048, .index_ratio = 4, .nk = 16, .nv = 48, .conv_kernel = 4, .experts = 512, .top_k = 10, .moe_width = 640, .ple_dim = 2560, .heads_per_ngram = 8, .ngram_size = 3, .ple_kernel = 4, .head_n = 248320, .linear_layers = 36, .attention_layers = 12, .mtp = true, .world = 1 };
    // the MTP head's fc_h over a chunk's 4 streams a row: 4 K slices of [4 * 2048, 2560] fp32 (past the shared
    // expert's down, 8 slices of [2048, 2560])
    try std.testing.expectEqual(@as(usize, 4 * 4 * 2048 * 2560 * 4), partBytes(g, 2048));
    // the MTP head's fc_h at 8 rows takes 32 stream rows of [2560, 2560]
    try std.testing.expect(partBytes(g, 8) >= tri.b16PartBytes(32, 2560, 2560));
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Forward);
    std.testing.refAllDecls(Dump);
    std.testing.refAllDecls(Scratch);
}
