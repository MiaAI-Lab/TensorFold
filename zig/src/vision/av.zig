//! Video decoding through FFmpeg's shared libraries, loaded at run time (dlopen): the same call sequence PyAV 18.1
//! runs for `av.open(io.BytesIO(data))` + `container.decode(stream)` + `frame.to_ndarray(format="rgb24")`, so the
//! frames are byte-identical to PyAV's.
//!
//! ABI assumptions (PyAV 18.1's FFmpeg 8: libavutil 60, libavcodec 62, libavformat 62, libswscale 9): everything is
//! called through exported functions except a handful of public struct fields that have no accessor, read at fixed
//! offsets on LP64 (aarch64 and x86-64 share them). The loader refuses any other major version, and checks the
//! stream/codec-parameter layout with sanity tests before trusting it.
//!   AVFormatContext: pb @32, nb_streams @44, streams @48, duration @104
//!   AVStream:        index @8, codecpar @16, time_base @32, duration @48, nb_frames @56, avg_frame_rate @88
//!   AVCodecParameters: codec_type @0, codec_id @4, width @72, height @76
//!   AVPacket: stream_index @36        AVIOContext: buffer @8
//!   AVFrame: data @0, linesize @64, width @104, height @108, format @116, color_range @280, color_primaries @284,
//!            color_trc @288, colorspace @292
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Error = error{ LibrariesMissing, UnsupportedVersion, SymbolMissing, OutOfMemory };

const Rational = extern struct { num: c_int, den: c_int };

const ReadFn = *const fn (?*anyopaque, [*]u8, c_int) callconv(.c) c_int;
const SeekFn = *const fn (?*anyopaque, i64, c_int) callconv(.c) i64;

pub const AVERROR_EOF: c_int = -541478725;
pub const AVERROR_INVALIDDATA: c_int = -1094995529;
pub const AVERROR_DECODER_NOT_FOUND: c_int = -1128547576;
pub const EAGAIN: c_int = 11;
const AVERROR_EAGAIN: c_int = -EAGAIN;
const AV_NOPTS_VALUE: i64 = std.math.minInt(i64);

pub const Libs = struct {
    util: std.DynLib,
    codec: std.DynLib,
    format: std.DynLib,
    scale: std.DynLib,
    // libavutil
    av_malloc: *const fn (usize) callconv(.c) ?*anyopaque,
    av_free: *const fn (?*anyopaque) callconv(.c) void,
    av_frame_alloc: *const fn () callconv(.c) ?*anyopaque,
    av_frame_free: *const fn (*?*anyopaque) callconv(.c) void,
    av_frame_unref: *const fn (?*anyopaque) callconv(.c) void,
    av_frame_copy_props: *const fn (?*anyopaque, ?*const anyopaque) callconv(.c) c_int,
    av_opt_set: *const fn (?*anyopaque, [*:0]const u8, [*:0]const u8, c_int) callconv(.c) c_int,
    av_opt_set_int: *const fn (?*anyopaque, [*:0]const u8, i64, c_int) callconv(.c) c_int,
    av_log_set_level: *const fn (c_int) callconv(.c) void,
    // libavcodec
    av_packet_alloc: *const fn () callconv(.c) ?*anyopaque,
    av_packet_free: *const fn (*?*anyopaque) callconv(.c) void,
    av_packet_unref: *const fn (?*anyopaque) callconv(.c) void,
    avcodec_find_decoder: *const fn (c_int) callconv(.c) ?*const anyopaque,
    avcodec_alloc_context3: *const fn (?*const anyopaque) callconv(.c) ?*anyopaque,
    avcodec_free_context: *const fn (*?*anyopaque) callconv(.c) void,
    avcodec_parameters_to_context: *const fn (?*anyopaque, ?*const anyopaque) callconv(.c) c_int,
    avcodec_open2: *const fn (?*anyopaque, ?*const anyopaque, ?*anyopaque) callconv(.c) c_int,
    avcodec_send_packet: *const fn (?*anyopaque, ?*const anyopaque) callconv(.c) c_int,
    avcodec_receive_frame: *const fn (?*anyopaque, ?*anyopaque) callconv(.c) c_int,
    // libavformat
    avformat_alloc_context: *const fn () callconv(.c) ?*anyopaque,
    avformat_open_input: *const fn (*?*anyopaque, ?[*:0]const u8, ?*const anyopaque, ?*anyopaque) callconv(.c) c_int,
    avformat_find_stream_info: *const fn (?*anyopaque, ?*anyopaque) callconv(.c) c_int,
    avformat_close_input: *const fn (*?*anyopaque) callconv(.c) void,
    av_read_frame: *const fn (?*anyopaque, ?*anyopaque) callconv(.c) c_int,
    av_guess_frame_rate: *const fn (?*anyopaque, ?*anyopaque, ?*anyopaque) callconv(.c) Rational,
    avio_alloc_context: *const fn (?*anyopaque, c_int, c_int, ?*anyopaque, ?ReadFn, ?*const anyopaque, ?SeekFn) callconv(.c) ?*anyopaque,
    avio_context_free: *const fn (*?*anyopaque) callconv(.c) void,
    // libswscale
    sws_alloc_context: *const fn () callconv(.c) ?*anyopaque,
    sws_free_context: *const fn (*?*anyopaque) callconv(.c) void,
    sws_scale_frame: *const fn (?*anyopaque, ?*anyopaque, ?*const anyopaque) callconv(.c) c_int,

    pub fn close(l: *Libs) void {
        l.scale.close();
        l.format.close();
        l.codec.close();
        l.util.close();
    }
};

fn startsWithAny(name: []const u8, prefix: []const u8) bool {
    return std.mem.startsWith(u8, name, prefix) and std.mem.indexOf(u8, name, ".so") != null;
}

/// The full path of the library whose file name starts with `prefix` (`libavutil-`, then `libavutil.so`) in `dir`.
fn findIn(a: Allocator, io: std.Io, dir_path: []const u8, stem: []const u8) ?[]u8 {
    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return null;
    defer dir.close(io);
    var best: ?[]u8 = null;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        var pre1: [64]u8 = undefined;
        var pre2: [64]u8 = undefined;
        const p1 = std.fmt.bufPrint(&pre1, "{s}-", .{stem}) catch continue;
        const p2 = std.fmt.bufPrint(&pre2, "{s}.so", .{stem}) catch continue;
        if (startsWithAny(entry.name, p1) or startsWithAny(entry.name, p2)) {
            const full = std.fs.path.join(a, &.{ dir_path, entry.name }) catch return best;
            if (best) |b| {
                // prefer the longer (more specific) name, e.g. libavutil-xxxx.so.60.26.102
                if (full.len > b.len) {
                    a.free(b);
                    best = full;
                } else a.free(full);
            } else best = full;
        }
    }
    return best;
}

fn tryDir(a: Allocator, io: std.Io, dir: []const u8) ?[4][]u8 {
    var out: [4][]u8 = undefined;
    const stems = [4][]const u8{ "libavutil", "libavcodec", "libavformat", "libswscale" };
    var n: usize = 0;
    for (stems) |s| {
        out[n] = findIn(a, io, dir, s) orelse break;
        n += 1;
    }
    if (n == 4) return out;
    for (out[0..n]) |p| a.free(p);
    return null;
}

fn getenv(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

/// Candidate library directories: TENSORFOLD_AV_LIBDIR, PyAV wheels' `av.libs` next to the Python packages, then
/// the system's default loader path (empty string).
fn candidateDirs(a: Allocator, io: std.Io) !std.ArrayList([]u8) {
    var list: std.ArrayList([]u8) = .empty;
    errdefer {
        for (list.items) |p| a.free(p);
        list.deinit(a);
    }
    if (getenv("TENSORFOLD_AV_LIBDIR")) |d| if (d.len != 0) try list.append(a, try a.dupe(u8, d));
    var bases: std.ArrayList([]u8) = .empty;
    defer {
        for (bases.items) |p| a.free(p);
        bases.deinit(a);
    }
    if (getenv("PYTHONPATH")) |pp| {
        var it = std.mem.tokenizeScalar(u8, pp, ':');
        while (it.next()) |e| try bases.append(a, try a.dupe(u8, e));
    }
    const fixed = [_][]const u8{ "/usr/local/lib", "/usr/lib", "/opt/venv/lib", "/opt/conda/lib" };
    for (fixed) |f| try bases.append(a, try a.dupe(u8, f));
    if (getenv("VIRTUAL_ENV")) |v| try bases.append(a, try std.fmt.allocPrint(a, "{s}/lib", .{v}));
    if (getenv("HOME")) |h| try bases.append(a, try std.fmt.allocPrint(a, "{s}/.local/lib", .{h}));
    for (bases.items) |b| {
        // b itself may be a site directory (PYTHONPATH), or a lib dir holding python3.x/{site,dist}-packages
        try list.append(a, try std.fmt.allocPrint(a, "{s}/av.libs", .{b}));
        var dir = std.Io.Dir.cwd().openDir(io, b, .{ .iterate = true }) catch continue;
        defer dir.close(io);
        var it = dir.iterate();
        while (it.next(io) catch null) |entry| {
            if (entry.kind != .directory or !std.mem.startsWith(u8, entry.name, "python3")) continue;
            inline for (.{ "dist-packages", "site-packages" }) |sp| {
                try list.append(a, try std.fmt.allocPrint(a, "{s}/{s}/{s}/av.libs", .{ b, entry.name, sp }));
            }
        }
    }
    try list.append(a, try a.dupe(u8, ""));
    return list;
}

const util_fns = .{ "av_malloc", "av_free", "av_frame_alloc", "av_frame_free", "av_frame_unref", "av_frame_copy_props", "av_opt_set", "av_opt_set_int", "av_log_set_level" };
const codec_fns = .{ "av_packet_alloc", "av_packet_free", "av_packet_unref", "avcodec_find_decoder", "avcodec_alloc_context3", "avcodec_free_context", "avcodec_parameters_to_context", "avcodec_open2", "avcodec_send_packet", "avcodec_receive_frame" };
const format_fns = .{ "avformat_alloc_context", "avformat_open_input", "avformat_find_stream_info", "avformat_close_input", "av_read_frame", "av_guess_frame_rate", "avio_alloc_context", "avio_context_free" };
const scale_fns = .{ "sws_alloc_context", "sws_free_context", "sws_scale_frame" };

fn lookupAll(l: *Libs) Error!void {
    inline for (util_fns) |n| @field(l, n) = l.util.lookup(@TypeOf(@field(l, n)), n) orelse return error.SymbolMissing;
    inline for (codec_fns) |n| @field(l, n) = l.codec.lookup(@TypeOf(@field(l, n)), n) orelse return error.SymbolMissing;
    inline for (format_fns) |n| @field(l, n) = l.format.lookup(@TypeOf(@field(l, n)), n) orelse return error.SymbolMissing;
    inline for (scale_fns) |n| @field(l, n) = l.scale.lookup(@TypeOf(@field(l, n)), n) orelse return error.SymbolMissing;
}

fn major(l: *std.DynLib, name: [:0]const u8) ?u32 {
    const f = l.lookup(*const fn () callconv(.c) c_uint, name) orelse return null;
    return f() >> 16;
}

pub fn load(a: Allocator, io: std.Io) Error!Libs {
    var dirs = candidateDirs(a, io) catch return error.OutOfMemory;
    defer {
        for (dirs.items) |p| a.free(p);
        dirs.deinit(a);
    }
    var last: Error = error.LibrariesMissing;
    for (dirs.items) |dir| {
        var paths: [4][]u8 = undefined;
        var have_paths = false;
        var l: Libs = undefined;
        if (dir.len == 0) {
            const names = [4][:0]const u8{ "libavutil.so.60", "libavcodec.so.62", "libavformat.so.62", "libswscale.so.9" };
            l.util = std.DynLib.open(names[0]) catch continue;
            l.codec = std.DynLib.open(names[1]) catch {
                l.util.close();
                continue;
            };
            l.format = std.DynLib.open(names[2]) catch {
                l.codec.close();
                l.util.close();
                continue;
            };
            l.scale = std.DynLib.open(names[3]) catch {
                l.format.close();
                l.codec.close();
                l.util.close();
                continue;
            };
        } else {
            paths = tryDir(a, io, dir) orelse continue;
            have_paths = true;
            defer if (have_paths) for (paths) |p| a.free(p);
            l.util = std.DynLib.open(paths[0]) catch continue;
            l.codec = std.DynLib.open(paths[1]) catch {
                l.util.close();
                continue;
            };
            l.format = std.DynLib.open(paths[2]) catch {
                l.codec.close();
                l.util.close();
                continue;
            };
            l.scale = std.DynLib.open(paths[3]) catch {
                l.format.close();
                l.codec.close();
                l.util.close();
                continue;
            };
        }
        const ok = (major(&l.util, "avutil_version") orelse 0) == 60 and (major(&l.codec, "avcodec_version") orelse 0) == 62 and
            (major(&l.format, "avformat_version") orelse 0) == 62 and (major(&l.scale, "swscale_version") orelse 0) == 9;
        if (!ok) {
            l.close();
            last = error.UnsupportedVersion;
            continue;
        }
        lookupAll(&l) catch |e| {
            l.close();
            last = e;
            continue;
        };
        l.av_log_set_level(-8); // AV_LOG_QUIET: the server's log is not FFmpeg's
        return l;
    }
    return last;
}

// ---- struct field access -----------------------------------------------------------------------------------------
inline fn at(comptime T: type, p: ?*anyopaque, off: usize) T {
    const bytes: [*]const u8 = @ptrCast(p.?);
    return std.mem.bytesToValue(T, bytes[off..][0..@sizeOf(T)]);
}

inline fn setAt(comptime T: type, p: ?*anyopaque, off: usize, v: T) void {
    const bytes: [*]u8 = @ptrCast(p.?);
    bytes[off..][0..@sizeOf(T)].* = std.mem.toBytes(v);
}

const Memory = struct {
    data: []const u8,
    pos: usize = 0,

    fn read(opaque_: ?*anyopaque, buf: [*]u8, size: c_int) callconv(.c) c_int {
        const m: *Memory = @ptrCast(@alignCast(opaque_.?));
        if (m.pos >= m.data.len) return AVERROR_EOF;
        const n = @min(@as(usize, @intCast(size)), m.data.len - m.pos);
        @memcpy(buf[0..n], m.data[m.pos..][0..n]);
        m.pos += n;
        return @intCast(n);
    }

    fn seek(opaque_: ?*anyopaque, offset: i64, whence_in: c_int) callconv(.c) i64 {
        const m: *Memory = @ptrCast(@alignCast(opaque_.?));
        if ((whence_in & 0x10000) != 0) return @intCast(m.data.len); // AVSEEK_SIZE
        const whence = whence_in & 3;
        const base: i64 = switch (whence) {
            0 => 0,
            1 => @intCast(m.pos),
            2 => @intCast(m.data.len),
            else => return -22,
        };
        const p = base + offset;
        if (p < 0) return -22;
        m.pos = @intCast(p);
        return p;
    }
};

/// Python's exception class name for an FFmpeg error code, as PyAV maps it (for the 400 text).
pub fn errorName(code: c_int) []const u8 {
    if (code == AVERROR_INVALIDDATA) return "InvalidDataError";
    if (code == -22) return "ValueError";
    if (code == -12) return "MemoryError";
    if (code == AVERROR_DECODER_NOT_FOUND) return "DecoderNotFoundError";
    if (code == -1296385272) return "DemuxerNotFoundError"; // FFERRTAG(0xF8,'D','E','M')
    return "FFmpegError";
}

pub const StreamInfo = struct {
    width: u32,
    height: u32,
    avg_num: i32,
    avg_den: i32,
    guessed_num: i32,
    guessed_den: i32,
    nb_frames: i64,
    duration: i64, // AV_NOPTS_VALUE when unknown
    time_base_num: i32,
    time_base_den: i32,
    container_duration: i64, // microseconds, AV_NOPTS_VALUE when unknown
};

pub const nopts = AV_NOPTS_VALUE;

/// An open clip: demuxer, decoder and converter.
pub const Clip = struct {
    libs: *const Libs,
    mem: *Memory,
    allocator: Allocator,
    ic: ?*anyopaque = null,
    pb: ?*anyopaque = null,
    stream: ?*anyopaque = null,
    stream_index: i32 = 0,
    ctx: ?*anyopaque = null,
    pkt: ?*anyopaque = null,
    frame: ?*anyopaque = null,
    sws: ?*anyopaque = null,
    flushed: bool = false,
    info: StreamInfo = undefined,
    code: c_int = 0, // the FFmpeg error behind the last failure

    pub fn open(a: Allocator, libs: *const Libs, data: []const u8) (Allocator.Error || error{ Failed, NoVideoStream, Unsupported })!*Clip {
        const c = try a.create(Clip);
        const mem = a.create(Memory) catch |e| {
            a.destroy(c);
            return e;
        };
        mem.* = .{ .data = data };
        c.* = .{ .libs = libs, .mem = mem, .allocator = a };
        errdefer c.close();
        const buf = libs.av_malloc(32768) orelse return error.OutOfMemory;
        c.pb = libs.avio_alloc_context(buf, 32768, 0, mem, Memory.read, null, Memory.seek) orelse {
            libs.av_free(buf);
            return error.OutOfMemory;
        };
        c.ic = libs.avformat_alloc_context() orelse return error.OutOfMemory;
        setAt(?*anyopaque, c.ic, 32, c.pb); // AVFormatContext.pb
        var r = libs.avformat_open_input(&c.ic, null, null, null);
        if (r < 0) {
            c.code = r;
            return error.Failed;
        }
        r = libs.avformat_find_stream_info(c.ic, null);
        if (r < 0) {
            c.code = r;
            return error.Failed;
        }
        const nb = at(u32, c.ic, 44);
        const streams = at(?[*]?*anyopaque, c.ic, 48);
        if (nb > 4096 or (nb != 0 and streams == null)) return error.Unsupported;
        var found: ?*anyopaque = null;
        for (0..nb) |i| {
            const s = streams.?[i];
            if (s == null) return error.Unsupported;
            if (at(i32, s, 8) != @as(i32, @intCast(i))) return error.Unsupported; // layout check
            const par = at(?*anyopaque, s, 16);
            if (par == null) return error.Unsupported;
            if (at(i32, par, 0) == 0 and found == null) found = s; // AVMEDIA_TYPE_VIDEO
        }
        const st = found orelse return error.NoVideoStream;
        c.stream = st;
        c.stream_index = at(i32, st, 8);
        const par = at(?*anyopaque, st, 16);
        const w = at(i32, par, 72);
        const h = at(i32, par, 76);
        if (w < 0 or h < 0 or w > (1 << 20) or h > (1 << 20)) return error.Unsupported;
        const g = libs.av_guess_frame_rate(c.ic, st, null);
        c.info = .{
            .width = @intCast(w),
            .height = @intCast(h),
            .avg_num = at(i32, st, 88),
            .avg_den = at(i32, st, 92),
            .guessed_num = g.num,
            .guessed_den = g.den,
            .nb_frames = at(i64, st, 56),
            .duration = at(i64, st, 48),
            .time_base_num = at(i32, st, 32),
            .time_base_den = at(i32, st, 36),
            .container_duration = at(i64, c.ic, 104),
        };
        return c;
    }

    /// Open the decoder (after the caller accepted the stream's properties).
    pub fn openDecoder(c: *Clip) error{ Failed, Unsupported }!void {
        const libs = c.libs;
        const par = at(?*anyopaque, c.stream, 16);
        const dec = libs.avcodec_find_decoder(at(i32, par, 4)) orelse {
            c.code = AVERROR_DECODER_NOT_FOUND;
            return error.Failed;
        };
        c.ctx = libs.avcodec_alloc_context3(dec) orelse return error.Failed;
        var r = libs.avcodec_parameters_to_context(c.ctx, par);
        if (r < 0) {
            c.code = r;
            return error.Failed;
        }
        _ = libs.av_opt_set(c.ctx, "thread_type", "frame+slice", 1);
        r = libs.avcodec_open2(c.ctx, dec, null);
        if (r < 0) {
            c.code = r;
            return error.Failed;
        }
        c.pkt = libs.av_packet_alloc() orelse return error.Failed;
        c.frame = libs.av_frame_alloc() orelse return error.Failed;
        c.sws = libs.sws_alloc_context() orelse return error.Failed;
        _ = libs.av_opt_set_int(c.sws, "sws_flags", 2, 0); // SWS_BILINEAR, PyAV's default interpolation
        _ = libs.av_opt_set_int(c.sws, "threads", 0, 0);
    }

    pub const Rgb = struct { w: u32, h: u32, pixels: []u8 };

    /// Decode the next frame into the clip's frame slot; false at the end of the stream. The slot is released
    /// by the next call (or `close`).
    pub fn advance(c: *Clip) error{Failed}!bool {
        const libs = c.libs;
        libs.av_frame_unref(c.frame);
        while (true) {
            const r = libs.avcodec_receive_frame(c.ctx, c.frame);
            if (r >= 0) return true;
            if (r == AVERROR_EOF) return false;
            if (r != AVERROR_EAGAIN) {
                c.code = r;
                return error.Failed;
            }
            // feed the decoder
            if (c.flushed) return false;
            const rf = libs.av_read_frame(c.ic, c.pkt);
            if (rf < 0) {
                if (rf != AVERROR_EOF) {
                    // PyAV stops the demux loop on errors other than EOF by raising; mirror that
                    c.code = rf;
                    return error.Failed;
                }
                const rs = libs.avcodec_send_packet(c.ctx, null);
                if (rs < 0 and rs != AVERROR_EOF) {
                    c.code = rs;
                    return error.Failed;
                }
                c.flushed = true;
                continue;
            }
            defer libs.av_packet_unref(c.pkt);
            if (at(i32, c.pkt, 36) != c.stream_index) continue;
            const rs = libs.avcodec_send_packet(c.ctx, c.pkt);
            if (rs < 0 and rs != AVERROR_EAGAIN) {
                c.code = rs;
                return error.Failed;
            }
        }
    }

    fn setColorspace(c: *Clip, f: ?*anyopaque, cs: i32) void {
        _ = c;
        // PyAV's _set_frame_colorspace(frame, cs, UNSPECIFIED): SWS_CS_* -> AVCOL_SPC_*
        const mapped: ?i32 = switch (cs) {
            1 => 1,
            4 => 4,
            5 => 6,
            7 => 7,
            9 => 9,
            else => null,
        };
        if (mapped) |m| setAt(i32, f, 292, m);
    }

    /// The current frame converted to rgb24 (caller frees `pixels`).
    pub fn convert(c: *Clip, a: Allocator) (Allocator.Error || error{Failed})!Rgb {
        const libs = c.libs;
        const src = c.frame;
        const w = at(i32, src, 104);
        const h = at(i32, src, 108);
        if (w <= 0 or h <= 0) {
            c.code = AVERROR_INVALIDDATA;
            return error.Failed;
        }
        var dst: ?*anyopaque = libs.av_frame_alloc() orelse return error.OutOfMemory;
        defer libs.av_frame_free(&dst);
        if (libs.av_frame_copy_props(dst, src) < 0) return error.OutOfMemory;
        setAt(i32, dst, 116, 2); // AV_PIX_FMT_RGB24
        setAt(i32, dst, 104, w);
        setAt(i32, dst, 108, h);
        const cs = at(i32, src, 292);
        inline for (.{ src, dst }) |f| {
            setAt(i32, f, 284, 2); // color_primaries = UNSPECIFIED
            setAt(i32, f, 288, 2); // color_trc = UNSPECIFIED
        }
        c.setColorspace(src, cs);
        c.setColorspace(dst, cs);
        const r = libs.sws_scale_frame(c.sws, dst, src);
        if (r < 0) {
            c.code = r;
            return error.Failed;
        }
        const base = at(?[*]const u8, dst, 0) orelse return error.Failed;
        const stride: usize = @intCast(at(i32, dst, 64));
        const uw: usize = @intCast(w);
        const uh: usize = @intCast(h);
        const out = try a.alloc(u8, uw * uh * 3);
        for (0..uh) |y| @memcpy(out[y * uw * 3 ..][0 .. uw * 3], base[y * stride ..][0 .. uw * 3]);
        return .{ .w = @intCast(w), .h = @intCast(h), .pixels = out };
    }

    pub fn close(c: *Clip) void {
        const libs = c.libs;
        const a = c.allocator;
        if (c.sws != null) libs.sws_free_context(&c.sws);
        if (c.frame != null) libs.av_frame_free(&c.frame);
        if (c.pkt != null) libs.av_packet_free(&c.pkt);
        if (c.ctx != null) libs.avcodec_free_context(&c.ctx);
        if (c.ic != null) libs.avformat_close_input(&c.ic);
        if (c.pb != null) {
            const buffer = at(?*anyopaque, c.pb, 8); // AVIOContext.buffer
            libs.av_free(buffer);
            libs.avio_context_free(&c.pb);
        }
        a.destroy(c.mem);
        a.destroy(c);
    }
};
