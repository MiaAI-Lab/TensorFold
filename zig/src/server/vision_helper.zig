//! ``--vision-backend python``: GLM-5.3-Flash picture and clip input through Python TensorFold's own frontend and tower
//! (``tensorfold.vision.glm``), run by a helper process (tools/zig/vision/glm_native_helper.py), after the owner's Flash
//! Next helper protocol (TensorFold-vision/zig/src/server/vision.zig). The server renders the chat template with
//! image and video parts marked (``tf_mark`` nonce) and sends the marked prompt with the request's media (data URLs,
//! or https URLs with ``--vision-urls``); the helper reads the marks, decodes (Pillow, PyAV), runs GlmVision.prepare
//! (resize, normalize, the expanded placeholders, the tokenizer) and the 24-block tower on this GPU, and answers with
//! the prompt's token ids, the feature rows and the bf16 features: Python's bits. The engine moves the features
//! into the prompt rows (``api.Media``). This is the reference path the native Zig frontend and tower are checked
//! against.
//!
//! Frames on the helper's stdin / stdout (little endian): a request is a u32 length and its JSON; a reply is a u32
//! header length, the header JSON, a u64 payload length and the payload (token ids u32 [n], rows u32 [k],
//! features bf16 [k, width]). One request at a time.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const log = @import("log.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

/// One image or video of a request, in the order the template renders them.
pub const Source = struct { kind: enum { image, video }, url: []const u8, detail: []const u8 = "auto" };

/// What the helper made of a request: Python's token ids and the media the engine takes.
pub const Prepared = struct {
    tokens: []const u32,
    media: *api.Media,
    images: u32,
    videos: u32,
    visual_tokens: u32,
    prepare_s: f64,
    encode_s: f64,
    features_sha: []const u8,
    /// the helper's torch reservation peak during the request (bytes)
    peak: u64 = 0,
};

pub const Refusal = struct { status: u16, message: []const u8, context: bool = false };

pub const Result = union(enum) { ok: Prepared, refused: Refusal };

pub const Settings = struct {
    allow_urls: bool = false,
    max_images: u32 = 50,
    image_tokens: u32 = 2048,
};

extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
extern "c" fn poll(fds: [*]std.posix.pollfd, n: c_ulong, timeout: c_int) c_int;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;

/// /proc/meminfo's MemAvailable in bytes, or null.
fn memAvailable() ?u64 {
    const fd = open("/proc/meminfo", 0);
    if (fd < 0) return null;
    defer _ = close(fd);
    var buf: [4096]u8 = undefined;
    const got = read(fd, &buf, buf.len);
    if (got <= 0) return null;
    var it = std.mem.tokenizeScalar(u8, buf[0..@intCast(got)], '\n');
    while (it.next()) |line| if (std.mem.startsWith(u8, line, "MemAvailable:")) {
        var f = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \t");
        const kb = std.fmt.parseInt(u64, f.next() orelse return null, 10) catch return null;
        return kb * 1024;
    };
    return null;
}

/// Where the helper script lives in the image (TENSORFOLD_VISION_HELPER overrides).
const default_helper = "/opt/tensorfold/share/vision/glm_native_helper.py";

pub const Helper = struct {
    gpa: Allocator,
    child: std.process.Child,
    to: c_int,
    from: c_int,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    dead: bool = false,
    settings: Settings,
    videos: bool = false,
    width: u32 = 0,
    tower_bytes: u64 = 0,
    /// seconds a request may take (writing it, decoding, the tower, reading the answer); past it the helper is
    /// taken as stuck: killed, and media requests refused from then on
    timeout_s: u32 = 600,
    /// requests waiting for the helper (each holds its body in host memory): past `max_waiting` a request is refused
    /// at once with 503, and none waits past `wait_s` (Python's bounded image slots and 60 s wait)
    waiting: std.atomic.Value(u32) = .init(0),
    max_waiting: u32 = 8,
    wait_s: u32 = 60,
    /// the features sha of the helper's start-up self-test image (a fixed picture through the whole frontend and
    /// tower): equal on every box running the same stack; TENSORFOLD_VISION_SELFTEST=<sha> refuses another
    selftest: []const u8 = "",
    /// MemAvailable the host must keep past a request's media (TENSORFOLD_VISION_MIN_FREE_MIB). The GLM engine takes
    /// nearly all of the Spark's memory by design, so the floor is small: the helper's decode is bounded by the limits
    min_free: u64 = 1 << 30,
    /// the transaction's deadline (monotonic ns) for readAll / writeAll
    deadline: i128 = 0,

    /// Starts the helper and waits for its ready frame (the tower loaded and warm on this GPU).
    pub fn start(gpa: Allocator, io: std.Io, dir: []const u8, settings: Settings, environ: ?*const std.process.Environ.Map, problem: *[]const u8) !*Helper {
        const a = gpa;
        const python = if (environ) |m| m.get("TENSORFOLD_VISION_PYTHON") orelse "python3" else "python3";
        const helper_script = if (environ) |m| m.get("TENSORFOLD_VISION_HELPER") orelse default_helper else default_helper;
        var argv: std.ArrayList([]const u8) = .empty;
        defer argv.deinit(a);
        var nums: [2][16]u8 = undefined;
        try argv.appendSlice(a, &.{ python, helper_script, "--model", dir, "--max-images", try std.fmt.bufPrint(&nums[0], "{d}", .{settings.max_images}), "--image-tokens", try std.fmt.bufPrint(&nums[1], "{d}", .{settings.image_tokens}) });
        if (settings.allow_urls) try argv.append(a, "--urls");
        // TENSORFOLD_VISION_PYTHONPATH: where TensorFold's Python tree is (the image puts it beside the binary)
        var env_copy: ?std.process.Environ.Map = null;
        defer if (env_copy) |*m| m.deinit();
        if (environ) |m| if (m.get("TENSORFOLD_VISION_PYTHONPATH")) |extra| {
            env_copy = std.process.Environ.Map.init(a);
            for (m.array_hash_map.keys(), m.array_hash_map.values()) |k, v| try env_copy.?.put(k, v);
            const joined = if (m.get("PYTHONPATH")) |old| try std.fmt.allocPrint(a, "{s}:{s}", .{ extra, old }) else try a.dupe(u8, extra);
            defer a.free(joined);
            try env_copy.?.put("PYTHONPATH", joined);
        };
        const child = std.process.spawn(io, .{ .argv = argv.items, .stdin = .pipe, .stdout = .pipe, .stderr = .inherit, .environ_map = if (env_copy) |*m| m else environ }) catch |e| {
            problem.* = try std.fmt.allocPrint(a, "--vision: cannot start {s} {s} ({s})", .{ python, helper_script, @errorName(e) });
            return error.VisionHelper;
        };
        const h = try gpa.create(Helper);
        h.* = .{ .gpa = gpa, .io = io, .child = child, .to = child.stdin.?.handle, .from = child.stdout.?.handle, .settings = settings };
        // loading the tower takes seconds; importing torch and transformers the first time can take a minute
        h.timeout_s = 900;
        h.deadline = now() + @as(i128, h.timeout_s) * std.time.ns_per_s;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const frame = h.readFrame(arena.allocator()) catch |e| {
            problem.* = try std.fmt.allocPrint(a, "--vision: the helper stopped before it was ready ({s}); see its lines above", .{@errorName(e)});
            h.stop(io);
            return error.VisionHelper;
        };
        const head = frame.header;
        if (head.get("ready") == null or head.get("ready").? != .bool or !head.get("ready").?.bool) {
            problem.* = try std.fmt.allocPrint(a, "--vision: the helper could not load the vision tower: {s}", .{head.strField("error") orelse "unknown"});
            h.stop(io);
            return error.VisionHelper;
        }
        h.timeout_s = 600;
        if (environ) |m| if (m.get("TENSORFOLD_VISION_MIN_FREE_MIB")) |t| {
            h.min_free = (std.fmt.parseInt(u64, std.mem.trim(u8, t, " "), 10) catch 1024) << 20;
        };
        h.selftest = try gpa.dupe(u8, head.strField("selftest_sha256") orelse "");
        if (environ) |m| if (m.get("TENSORFOLD_VISION_SELFTEST")) |want| if (want.len > 0 and !std.mem.eql(u8, want, h.selftest)) {
            problem.* = try std.fmt.allocPrint(a, "--vision: the helper's self-test features sha {s} is not TENSORFOLD_VISION_SELFTEST {s}: another Python, torch, transformers, Pillow or PyAV stack than the reference's", .{ h.selftest, want });
            h.stop(io);
            return error.VisionHelper;
        };
        h.videos = if (head.get("videos")) |v| v == .bool and v.bool else false;
        h.width = @intCast(if (head.get("width")) |v| v.int64() orelse 0 else 0);
        h.tower_bytes = @intCast(if (head.get("tower_bytes")) |v| v.int64() orelse 0 else 0);
        const versions = if (head.get("versions")) |v| json.stringify(arena.allocator(), v, .{ .compact = true }) catch "" else "";
        log.line("vision: image{s} input on this GPU, a {d:.2} GiB tower (helper pid {d}; {s}; self-test {s})", .{ if (h.videos) " and video" else "", @as(f64, @floatFromInt(h.tower_bytes)) / (1 << 30), if (head.get("pid")) |p| p.int64() orelse 0 else 0, versions, h.selftest });
        return h;
    }

    /// Stops the helper (killed and reaped: a stuck one cannot hold the server's exit).
    pub fn stop(h: *Helper, io: std.Io) void {
        h.child.kill(io);
        if (h.selftest.len > 0) h.gpa.free(h.selftest);
        h.gpa.destroy(h);
    }

    fn now() i128 {
        var ts: std.posix.timespec = undefined;
        _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
        return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
    }

    /// Milliseconds left of the transaction (at least 1 while any is left), 0 past it.
    fn left(h: *const Helper) c_int {
        const ms = @divTrunc(h.deadline - now(), std.time.ns_per_ms);
        return if (ms <= 0) 0 else @intCast(@min(ms + 1, std.math.maxInt(c_int)));
    }

    const Frame = struct { header: Value, payload: []u8 };

    fn readAll(h: *Helper, buf: []u8) !void {
        var got: usize = 0;
        while (got < buf.len) {
            var fds = [1]std.posix.pollfd{.{ .fd = h.from, .events = std.posix.POLL.IN, .revents = 0 }};
            const wait = h.left();
            if (wait == 0) return error.VisionTimeout;
            const ready = poll(&fds, 1, wait);
            if (ready == 0) return error.VisionTimeout;
            if (ready < 0) {
                if (std.posix.errno(ready) == .INTR) continue;
                return error.VisionRead;
            }
            const n = read(h.from, buf[got..].ptr, buf.len - got);
            if (n < 0) {
                if (std.posix.errno(n) == .INTR) continue;
                return error.VisionRead;
            }
            if (n == 0) return error.VisionStopped;
            got += @intCast(n);
        }
    }

    fn writeAll(h: *Helper, bytes: []const u8) !void {
        var done: usize = 0;
        while (done < bytes.len) {
            // a helper that stops reading cannot hold the writer past the transaction's deadline
            var fds = [1]std.posix.pollfd{.{ .fd = h.to, .events = std.posix.POLL.OUT, .revents = 0 }};
            const wait = h.left();
            if (wait == 0) return error.VisionTimeout;
            const ready = poll(&fds, 1, wait);
            if (ready == 0) return error.VisionTimeout;
            if (ready < 0) {
                if (std.posix.errno(ready) == .INTR) continue;
                return error.VisionStopped;
            }
            if (fds[0].revents & (std.posix.POLL.ERR | std.posix.POLL.HUP) != 0 and fds[0].revents & std.posix.POLL.OUT == 0) return error.VisionStopped;
            const n = write(h.to, bytes[done..].ptr, @min(bytes.len - done, 1 << 16));
            if (n < 0) {
                if (std.posix.errno(n) == .INTR) continue;
                return error.VisionStopped;
            }
            done += @intCast(n);
        }
    }

    fn readFrame(h: *Helper, a: Allocator) !Frame {
        var len4: [4]u8 = undefined;
        try h.readAll(&len4);
        const head_len = std.mem.readInt(u32, &len4, .little);
        if (head_len > 1 << 24) return error.VisionFrame;
        const head = try a.alloc(u8, head_len);
        try h.readAll(head);
        var len8: [8]u8 = undefined;
        try h.readAll(&len8);
        const size = std.mem.readInt(u64, &len8, .little);
        if (size > 4 << 30) return error.VisionFrame;
        const payload = try a.alloc(u8, @intCast(size));
        try h.readAll(payload);
        const parsed = try json.parse(a, head);
        const v = switch (parsed) {
            .ok => |x| x,
            else => return error.VisionFrame,
        };
        return .{ .header = v, .payload = payload };
    }

    /// The helper's answer for one rendered prompt and its media; `a` holds the result (the request's arena).
    pub fn prepare(h: *Helper, a: Allocator, prompt: []const u8, nonce: []const u8, sources: []const Source, max_prompt_tokens: ?u32) !Result {
        const req = try json.newObject(a);
        try req.put(a, "prompt", .{ .string = prompt });
        try req.put(a, "nonce", .{ .string = nonce });
        const list = try a.alloc(Value, sources.len);
        for (sources, list) |s, *slot| {
            const o = try json.newObject(a);
            try o.put(a, "kind", .{ .string = if (s.kind == .image) "image" else "video" });
            try o.put(a, "url", .{ .string = s.url });
            try o.put(a, "detail", .{ .string = s.detail });
            slot.* = .{ .object = o };
        }
        try req.put(a, "media", .{ .array = list });
        try req.put(a, "max_prompt_tokens", if (max_prompt_tokens) |m| try json.intValue(a, m) else .null);
        // bounded admission: a few requests wait (each holds its media in host memory), none past wait_s
        if (h.waiting.fetchAdd(1, .acq_rel) >= h.max_waiting) {
            _ = h.waiting.fetchSub(1, .acq_rel);
            return .{ .refused = .{ .status = 503, .message = "image request queue is full; retry shortly" } };
        }
        const wait_until = now() + @as(i128, h.wait_s) * std.time.ns_per_s;
        while (!h.mutex.tryLock()) {
            if (now() >= wait_until) {
                _ = h.waiting.fetchSub(1, .acq_rel);
                return .{ .refused = .{ .status = 503, .message = "image processing capacity is busy; retry shortly" } };
            }
            std.Io.sleep(h.io, .fromMilliseconds(5), .awake) catch {};
        }
        _ = h.waiting.fetchSub(1, .acq_rel);
        defer h.mutex.unlock(h.io);
        if (h.dead) return .{ .refused = .{ .status = 503, .message = "the vision helper stopped; image and video requests are unavailable until the server restarts" } };
        // the owner's floor: the request's frame and the helper's decode must leave 10 GiB MemAvailable
        if (memAvailable()) |avail| if (avail < h.min_free + 4 * @as(u64, sources.len) * (1 << 20) + 2 * total_url: {
            var t: u64 = 0;
            for (sources) |src| t += src.url.len;
            break :total_url t;
        }) return .{ .refused = .{ .status = 503, .message = "not enough free memory to process the images or video now; retry shortly" } };
        const body = try json.stringify(a, .{ .object = req }, .{ .compact = true });
        h.deadline = now() + @as(i128, h.timeout_s) * std.time.ns_per_s;
        var len4: [4]u8 = undefined;
        std.mem.writeInt(u32, &len4, @intCast(body.len), .little);
        const frame = blk: {
            h.writeAll(&len4) catch |e| break :blk e;
            h.writeAll(body) catch |e| break :blk e;
            break :blk h.readFrame(a);
        } catch |e| {
            h.dead = true;
            // a stuck helper is killed (it holds GPU memory and would hold the server's exit)
            if (h.child.id) |pid| std.posix.kill(pid, std.posix.SIG.KILL) catch {};
            log.line("vision: the helper is gone ({s}); image and video requests are refused from now on", .{@errorName(e)});
            return .{ .refused = .{ .status = 503, .message = "the vision helper stopped; image and video requests are unavailable until the server restarts" } };
        };
        const head = frame.header;
        const ok = head.get("ok");
        if (ok == null or ok.? != .bool or !ok.?.bool) {
            const status: u16 = @intCast(if (head.get("status")) |s| s.int64() orelse 400 else 400);
            const code = head.strField("code");
            return .{ .refused = .{ .status = status, .message = head.strField("error") orelse "the image or video could not be processed", .context = code != null and std.mem.eql(u8, code.?, "context_length_exceeded") } };
        }
        const n: usize = @intCast(head.get("tokens").?.int64() orelse return error.VisionFrame);
        const k: usize = @intCast(head.get("rows").?.int64() orelse return error.VisionFrame);
        const width: usize = @intCast(head.get("width").?.int64() orelse return error.VisionFrame);
        const sizes = [_]usize{ n * 4, k * 4, k * width * 2 };
        var total: usize = 0;
        for (sizes) |x| total += x;
        if (frame.payload.len != total) return error.VisionFrame;
        var at: usize = 0;
        const tokens = try a.alloc(u32, n);
        @memcpy(std.mem.sliceAsBytes(tokens), frame.payload[at..][0..sizes[0]]);
        at += sizes[0];
        const rows = try a.alloc(u32, k);
        @memcpy(std.mem.sliceAsBytes(rows), frame.payload[at..][0..sizes[1]]);
        at += sizes[1];
        const media = try a.create(api.Media);
        media.* = .{ .rows = rows, .features = frame.payload[at..][0..sizes[2]], .width = @intCast(width) };
        media.check(n) catch return error.VisionFrame;
        const count = struct {
            fn of(v: Value, key: []const u8) u32 {
                return @intCast(if (v.get(key)) |x| x.int64() orelse 0 else 0);
            }
        };
        return .{ .ok = .{ .tokens = tokens, .media = media, .images = count.of(head, "images"), .videos = count.of(head, "videos"), .visual_tokens = count.of(head, "visual_tokens"), .prepare_s = if (head.get("prepare_s")) |x| x.float64() orelse 0 else 0, .encode_s = if (head.get("encode_s")) |x| x.float64() orelse 0 else 0, .features_sha = head.strField("features_sha256") orelse "", .peak = @intCast(@max(0, if (head.get("peak")) |x| x.int64() orelse 0 else 0)) } };
    }
};
