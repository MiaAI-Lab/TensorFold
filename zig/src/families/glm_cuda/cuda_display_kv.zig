//! Ports glm5_next/cuda/display_kv.py: parsing, span_planes and map_span.
//! TensorFold by Ash Hart and contributors; recipe display support by MiaAI-Lab.
//! dispram uses the installed kindling Python client in this process; no client code is vendored.
const std = @import("std");
const cuda = @import("cuda");
extern "c" fn mmap(?*anyopaque, usize, c_int, c_int, c_int, i64) ?*anyopaque;
extern "c" fn munmap(?*anyopaque, usize) c_int;
extern "c" fn ioctl(c_int, c_ulong, ...) c_int;
extern "c" fn open([*:0]const u8, c_int, ...) c_int;
extern "c" fn dlopen([*:0]const u8, c_int) ?*anyopaque;
extern "c" fn dlsym(*anyopaque, [*:0]const u8) ?*anyopaque;

pub const Backend = enum { drm, dispram };
var claimed: std.atomic.Value(bool) = .init(false);
pub fn env(name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}
pub fn mib(raw: ?[]const u8) !usize {
    const t = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (t.len == 0) return 0;
    const n = std.fmt.parseInt(i64, t, 10) catch return error.InvalidDisplayMiB;
    if (n < 0 or n > 2032 or @mod(n, 16) != 0) return error.InvalidDisplayMiB;
    return @intCast(n);
}
pub fn backend(raw: ?[]const u8) !Backend {
    const t = std.mem.trim(u8, if (raw == null or raw.?.len == 0) "drm" else raw.?, " \t\r\n");
    if (std.ascii.eqlIgnoreCase(t, "drm")) return .drm;
    if (std.ascii.eqlIgnoreCase(t, "dispram")) return .dispram;
    return error.InvalidDisplayBackend;
}
pub fn planes(count: usize, plane: usize, display: usize) !usize {
    if (plane == 0) return error.DisplayTooLarge;
    const k = @max(1, display / plane + @intFromBool(display % plane != 0));
    if (k > count) return error.DisplayTooLarge;
    return k;
}
pub const Config = struct {
    bytes: usize = 0,
    kind: Backend = .drm,
    card: []const u8 = "/dev/dri/card0",
    pub fn read() !Config {
        const bytes = (try mib(env("TF_GLM_DISPLAY_KV_MIB") orelse env("DISPLAY_KV_MIB"))) << 20;
        return .{ .bytes = bytes, .kind = if (bytes == 0) .drm else try backend(env("TF_GLM_DISPLAY_KV_BACKEND") orelse env("DISPLAY_KV_BACKEND")), .card = env("TF_GLM_DISPLAY_KV_CARD") orelse "/dev/dri/card0" };
    }
};
pub fn headless(gpa: std.mem.Allocator, io: std.Io, sysfs: []const u8, card_path: []const u8) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, sysfs, .{ .iterate = true });
    defer dir.close(io);
    const prefix = try std.fmt.allocPrint(gpa, "{s}-", .{std.fs.path.basename(card_path)});
    defer gpa.free(prefix);
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (!std.mem.startsWith(u8, entry.name, prefix)) continue;
        const path = try std.fmt.allocPrint(gpa, "{s}/status", .{entry.name});
        defer gpa.free(path);
        var f = try dir.openFile(io, path, .{});
        defer f.close(io);
        var buf: [128]u8 = undefined;
        var r = f.readerStreaming(io, &buf);
        const text = try r.interface.allocRemaining(gpa, .limited(128));
        defer gpa.free(text);
        if (std.mem.eql(u8, std.mem.trim(u8, text, " \t\r\n"), "connected")) return error.DisplayConnected;
    }
}
pub const Span = struct {
    ptr: u64,
    base: usize = 0,
    ordinary: usize = 0,
    display: usize = 0,
    fd: c_int = -1,
    handle: u32 = 0,
    ordinary_registered: bool = false,
    display_registered: bool = false,
    d: *const cuda.Driver,
    owns_claim: bool = false,
    pub fn deinit(s: *Span) void {
        if (s.base == 0) return; // dispram's client retains its CUDA VMM allocation for the process lifetime.
        if (s.display_registered) _ = s.d.api.cuMemHostUnregister(@ptrFromInt(s.base + s.ordinary));
        if (s.ordinary_registered) _ = s.d.api.cuMemHostUnregister(@ptrFromInt(s.base));
        _ = munmap(@ptrFromInt(s.base), s.ordinary + s.display);
        var close_gem = [_]u32{ s.handle, 0 };
        _ = ioctl(s.fd, 0x40086409, &close_gem);
        _ = std.c.close(s.fd);
        s.base = 0;
        if (s.owns_claim) claimed.store(false, .release);
        s.owns_claim = false;
    }
};
fn mapped(address: usize, size: usize, prot: c_int, flags: c_int, fd: c_int, offset: u64) !usize {
    const p = mmap(if (address == 0) null else @ptrFromInt(address), size, prot, flags, fd, @intCast(offset)) orelse return error.DisplayMmap;
    const at = @intFromPtr(p);
    if (at == std.math.maxInt(usize) or (address != 0 and at != address)) return error.DisplayMmap;
    return at;
}
pub fn map(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, device: c_int, cfg: Config, ordinary: usize) !Span {
    if (claimed.cmpxchgStrong(false, true, .acq_rel, .acquire) != null) return error.DisplayAlreadyAllocated;
    errdefer claimed.store(false, .release);
    try headless(gpa, io, "/sys/class/drm", cfg.card);
    if (cfg.kind == .dispram) {
        var span = try dispram(gpa, d, device, ordinary + cfg.bytes, cfg.bytes);
        span.owns_claim = true;
        return span;
    }
    const card = try gpa.dupeSentinel(u8, cfg.card, 0);
    defer gpa.free(card);
    var system: System = .{ .d = d, .card = card };
    var span = try mapDrm(&system, ordinary, cfg.bytes);
    span.owns_claim = true;
    return span;
}

const System = struct {
    d: *const cuda.Driver,
    card: [:0]const u8,
    fn openCard(s: *System) !c_int {
        const fd = open(s.card, 2 | 0x80000);
        if (fd < 0) return error.DisplayCardUnavailable;
        return fd;
    }
    fn create(_: *System, fd: c_int, size: usize) !struct { handle: u32, size: usize } {
        var dumb: extern struct { height: u32, width: u32 = 4096, bpp: u32 = 32, flags: u32 = 0, handle: u32 = 0, pitch: u32 = 0, size: u64 = 0 } = .{ .height = @intCast(std.math.divCeil(usize, size, 16384) catch unreachable) };
        if (ioctl(fd, 0xC02064B2, &dumb) != 0) return error.DisplayCreateDumb;
        return .{ .handle = dumb.handle, .size = @intCast(dumb.size) };
    }
    fn offset(_: *System, fd: c_int, handle: u32) !u64 {
        var out: extern struct { handle: u32, pad: u32 = 0, offset: u64 = 0 } = .{ .handle = handle };
        if (ioctl(fd, 0xC01064B3, &out) != 0) return error.DisplayMapDumb;
        return out.offset;
    }
    fn reserve(_: *System, size: usize) !usize {
        return mapped(0, size, 0, 0x4022, -1, 0);
    }
    fn ordinary(_: *System, base: usize, size: usize) !void {
        _ = try mapped(base, size, 3, 0x32, -1, 0);
    }
    fn display(_: *System, base: usize, size: usize, fd: c_int, offset_value: u64) !void {
        _ = try mapped(base, size, 3, 0x11, fd, offset_value);
    }
    fn register(s: *System, base: usize, size: usize, is_io: bool) !void {
        try s.d.check(s.d.api.cuMemHostRegister_v2(@ptrFromInt(base), size, if (is_io) 6 else 2), "display register");
    }
    fn pointer(s: *System, base: usize) !u64 {
        var out: u64 = 0;
        try s.d.check(s.d.api.cuMemHostGetDevicePointer_v2(&out, @ptrFromInt(base), 0), "display device pointer");
        return out;
    }
    fn unregister(s: *System, base: usize) void {
        _ = s.d.api.cuMemHostUnregister(@ptrFromInt(base));
    }
    fn unmap(_: *System, base: usize, size: usize) void {
        _ = munmap(@ptrFromInt(base), size);
    }
    fn closeGem(_: *System, fd: c_int, handle: u32) void {
        var gem = [_]u32{ handle, 0 };
        _ = ioctl(fd, 0x40086409, &gem);
    }
    fn closeFd(_: *System, fd: c_int) void {
        _ = std.c.close(fd);
    }
};
/// The same call seam as Python's SYSTEM: tests exercise every failure without DRM or CUDA.
fn mapDrm(system: anytype, ordinary: usize, display_bytes: usize) !Span {
    const fd = try system.openCard();
    errdefer system.closeFd(fd);
    const dumb = try system.create(fd, display_bytes);
    errdefer system.closeGem(fd, dumb.handle);
    if (dumb.size < display_bytes) return error.DisplayShort;
    const offset = try system.offset(fd, dumb.handle);
    const base = try system.reserve(ordinary + display_bytes);
    errdefer system.unmap(base, ordinary + display_bytes);
    if (ordinary > 0) try system.ordinary(base, ordinary);
    try system.display(base + ordinary, display_bytes, fd, offset);
    if (ordinary > 0) try system.register(base, ordinary, false);
    errdefer if (ordinary > 0) system.unregister(base);
    try system.register(base + ordinary, display_bytes, true);
    errdefer system.unregister(base + ordinary);
    const ptr = try system.pointer(base);
    const top = try system.pointer(base + ordinary);
    if (top < ptr or top - ptr != ordinary) return error.DisplayNotContiguous;
    return .{ .ptr = ptr, .base = base, .ordinary = ordinary, .display = display_bytes, .fd = fd, .handle = dumb.handle, .d = system.d, .ordinary_registered = ordinary > 0, .display_registered = true };
}

fn symbol(comptime T: type, lib: *anyopaque, name: [:0]const u8) !T {
    return @ptrCast(@alignCast(dlsym(lib, name) orelse return error.DispramPythonSymbol));
}
fn dispram(gpa: std.mem.Allocator, d: *const cuda.Driver, device: c_int, total: usize, display: usize) !Span {
    try d.check(d.api.cuCtxSynchronize(), "dispram synchronize");
    // RTLD_GLOBAL is necessary for extension modules loaded by the installed Python client.
    var lib: ?*anyopaque = null;
    for ([_][:0]const u8{ "libpython3.13.so.1.0", "libpython3.12.so.1.0", "libpython3.11.so.1.0", "libpython3.10.so.1.0" }) |name| {
        lib = dlopen(name, 0x102);
        if (lib != null) break;
    }
    const l = lib orelse return error.DispramPythonUnavailable;
    const initialized = try symbol(*const fn () callconv(.c) c_int, l, "Py_IsInitialized");
    if (initialized() == 0) {
        (try symbol(*const fn () callconv(.c) void, l, "Py_Initialize"))();
        _ = (try symbol(*const fn () callconv(.c) ?*anyopaque, l, "PyEval_SaveThread"))();
    }
    const ensure = try symbol(*const fn () callconv(.c) c_int, l, "PyGILState_Ensure");
    const release = try symbol(*const fn (c_int) callconv(.c) void, l, "PyGILState_Release");
    const gil = ensure();
    defer release(gil);
    const module = (try symbol(*const fn ([*:0]const u8) callconv(.c) ?*anyopaque, l, "PyImport_AddModule"))("__main__") orelse return error.DispramClient;
    const dict = (try symbol(*const fn (*anyopaque) callconv(.c) *anyopaque, l, "PyModule_GetDict"))(module);
    const code = try std.fmt.allocPrintSentinel(gpa, "(__import__('dispram').map_glued({d}, {d}) if __import__('dispram').available() else None)", .{ total, device }, 0);
    defer gpa.free(code);
    const result = (try symbol(*const fn ([*:0]const u8, c_int, *anyopaque, *anyopaque, ?*anyopaque) callconv(.c) ?*anyopaque, l, "PyRun_StringFlags"))(code, 258, dict, dict, null) orelse {
        (try symbol(*const fn () callconv(.c) void, l, "PyErr_Print"))();
        return error.DispramClient;
    };
    defer (symbol(*const fn (*anyopaque) callconv(.c) void, l, "Py_DecRef") catch unreachable)(result);
    const item = try symbol(*const fn (*anyopaque, isize) callconv(.c) ?*anyopaque, l, "PyTuple_GetItem");
    const integer = try symbol(*const fn (*anyopaque) callconv(.c) u64, l, "PyLong_AsUnsignedLongLong");
    const ptr = integer(item(result, 0) orelse return error.DispramClient);
    const tail = integer(item(result, 2) orelse return error.DispramClient);
    const occurred = try symbol(*const fn () callconv(.c) ?*anyopaque, l, "PyErr_Occurred");
    if (occurred() != null or ptr == 0 or tail < display) return error.DisplayShort;
    return .{ .ptr = ptr, .d = d };
}
test "display configuration and fewest latent planes match reference boundaries" {
    try std.testing.expectEqual(@as(usize, 0), try mib(null));
    try std.testing.expectEqual(@as(usize, 1792), try mib(" 1792 "));
    for ([_][]const u8{ "-16", "1", "2048", "nan", "16.0" }) |raw| try std.testing.expectError(error.InvalidDisplayMiB, mib(raw));
    try std.testing.expectEqual(Backend.dispram, try backend(" DisPrAm "));
    try std.testing.expectError(error.InvalidDisplayBackend, backend("host"));
    try std.testing.expectEqual(@as(usize, 2), try planes(11, 1024, 1025));
    try std.testing.expectEqual(@as(usize, 1), try planes(11, 1024, 1024));
    try std.testing.expectError(error.DisplayTooLarge, planes(2, 1024, 2049));
}

const Fake = struct {
    d: *const cuda.Driver = undefined,
    step: usize = 0,
    fail: usize = 0,
    closed: usize = 0,
    gems: usize = 0,
    unmapped: usize = 0,
    unregistered: usize = 0,
    short: bool = false,
    separate: bool = false,
    fn next(f: *Fake) !void {
        f.step += 1;
        if (f.step == f.fail) return error.Injected;
    }
    fn openCard(f: *Fake) !c_int {
        try f.next();
        return 3;
    }
    fn create(f: *Fake, _: c_int, n: usize) !struct { handle: u32, size: usize } {
        try f.next();
        return .{ .handle = 7, .size = if (f.short) n - 1 else n };
    }
    fn offset(f: *Fake, _: c_int, _: u32) !u64 {
        try f.next();
        return 4096;
    }
    fn reserve(f: *Fake, _: usize) !usize {
        try f.next();
        return 0x10000;
    }
    fn ordinary(f: *Fake, base: usize, n: usize) !void {
        try f.next();
        try std.testing.expectEqual(@as(usize, 0x10000), base);
        try std.testing.expectEqual(@as(usize, 4096), n);
    }
    fn display(f: *Fake, base: usize, _: usize, _: c_int, _: u64) !void {
        try f.next();
        try std.testing.expect(base == 0x11000 or base == 0x10000);
    }
    fn register(f: *Fake, base: usize, _: usize, is_io: bool) !void {
        try f.next();
        try std.testing.expectEqual(base != 0x10000 or f.step == 6, is_io);
    }
    fn pointer(f: *Fake, base: usize) !u64 {
        try f.next();
        return base + if (f.separate and base != 0x10000) @as(usize, 1) else 0;
    }
    fn unregister(f: *Fake, _: usize) void {
        f.unregistered += 1;
    }
    fn unmap(f: *Fake, _: usize, _: usize) void {
        f.unmapped += 1;
    }
    fn closeGem(f: *Fake, _: c_int, _: u32) void {
        f.gems += 1;
    }
    fn closeFd(f: *Fake, _: c_int) void {
        f.closed += 1;
    }
};
test "DRM span unwinds every failed driver step and rejects a short or noncontiguous span" {
    for (1..11) |fail| {
        var f: Fake = .{ .fail = fail };
        try std.testing.expectError(error.Injected, mapDrm(&f, 4096, 16384));
        try std.testing.expectEqual(@as(usize, @intFromBool(fail > 1)), f.closed);
        try std.testing.expectEqual(@as(usize, @intFromBool(fail > 2)), f.gems);
        try std.testing.expectEqual(@as(usize, @intFromBool(fail > 4)), f.unmapped);
        try std.testing.expectEqual(@as(usize, @intFromBool(fail > 7) + @as(usize, @intFromBool(fail > 8))), f.unregistered);
    }
    var f: Fake = .{};
    const span = try mapDrm(&f, 4096, 16384);
    try std.testing.expectEqual(@as(u64, 0x10000), span.ptr);
    try std.testing.expectEqual(@as(usize, 0), f.closed);
    f = .{ .short = true };
    try std.testing.expectError(error.DisplayShort, mapDrm(&f, 4096, 16384));
    try std.testing.expectEqual(@as(usize, 1), f.closed);
    f = .{ .separate = true };
    try std.testing.expectError(error.DisplayNotContiguous, mapDrm(&f, 4096, 16384));
    try std.testing.expectEqual(@as(usize, 2), f.unregistered);
    f = .{};
    _ = try mapDrm(&f, 0, 16384);
    try std.testing.expectEqual(@as(usize, 8), f.step);
}

test "a connected output on the configured card refuses display memory" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    try tmp.dir.createDir(io, "card0-HDMI-A-1", .default_dir);
    try tmp.dir.createDir(io, "card1-DP-1", .default_dir);
    try tmp.dir.writeFile(io, .{ .sub_path = "card0-HDMI-A-1/status", .data = "connected\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "card1-DP-1/status", .data = "disconnected\n" });
    const path = try std.fmt.allocPrint(std.testing.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer std.testing.allocator.free(path);
    try std.testing.expectError(error.DisplayConnected, headless(std.testing.allocator, io, path, "/dev/dri/card0"));
    try headless(std.testing.allocator, io, path, "/dev/dri/card1");
    try tmp.dir.writeFile(io, .{ .sub_path = "card0-HDMI-A-1/status", .data = "disconnected\n" });
    try headless(std.testing.allocator, io, path, "/dev/dri/card0");
}
