//! The spill tier's disk side for the GLM CUDA engine: kept prompt states written to local disk when they leave memory and
//! read back when a later request extends the same tokens, also after a restart. Host only (no CUDA): the engine copies
//! a state's device buffer through `Writer.put` / `Reader.read` in pieces.
//!
//! Behaviour follows the recipe's spill tier (patch 0088-glm-spill-tier, by Robert Wojciechowski, https://github.com/wojo,
//! with the multi-rank design of Jay Leaton's patch 0250 of jayleaton/glm53-tensorfold-spark): SPILL_GIB caps the bytes
//! a rank keeps (the oldest files go first), no write leaves less than SPILL_MIN_FREE_GIB of the disk free (the spill
//! floor: old files are dropped to make that room, else the write is skipped), prompts under SPILL_MIN_TOKENS are not
//! written, a file is written to `.partial`, fsynced and renamed, every read checks a CRC-32 for each 8 MiB, and a file
//! whose size does not match its header is dropped at start. One directory per compatibility hash and per rank; other
//! hashes (our `tfspill-` directories only) are pruned at open. The file format is this engine's own.
const std = @import("std");
const lanes = @import("lanes");

pub const block: usize = 8 << 20;
const head_len = 48;
const page = 4096;
const foot_len = 16;
const magic = "TFSPILL1";
const foot_magic = "TFSPEND1";

pub const Config = struct {
    /// the spill folder (SPILL_DIR; the same absolute path on every rank's machine)
    root: []const u8,
    /// compatibility hash of everything a stored state depends on (weights, drafter, kernels, geometry, settings)
    build: u64,
    rank: u32 = 0,
    /// bytes this rank's files may take (SPILL_GIB)
    cap: u64,
    /// SPILL_MIN_TOKENS: shorter states are not written
    min_tokens: u32 = 8192,
    /// SPILL_MIN_FREE_GIB in bytes: no write leaves less of the disk free
    min_free: u64 = 50 << 30,
    /// the free bytes of the disk holding a folder (tests give their own)
    free: *const fn ([:0]const u8) ?u64 = lanes.learned_disk.available,
};

pub const Stats = struct {
    entries: u32 = 0,
    bytes: u64 = 0,
    writes: u64 = 0,
    write_bytes: u64 = 0,
    reads: u64 = 0,
    read_bytes: u64 = 0,
    skipped_short: u64 = 0,
    skipped_exists: u64 = 0,
    skipped_floor: u64 = 0,
    skipped_cap: u64 = 0,
    pruned: u64 = 0,
    failed: u64 = 0,
    crc_failures: u64 = 0,
};

pub const Meta = struct { key: u64, at: u32, tokens: []u32, payload: u64, size: u64, used: u64 };

pub const Error = error{ SpillWrite, SpillRead, SpillCrc, SpillNoRoom, SpillShort, SpillExists, SpillDir };

pub fn keyOf(tokens: []const u32) u64 {
    return std.hash.Wyhash.hash(0x5911, std.mem.sliceAsBytes(tokens));
}

fn blocksOf(payload: u64) u32 {
    return @intCast((payload + block - 1) / block);
}

fn payloadOffset(ntokens: usize) u64 {
    return std.mem.alignForward(u64, head_len + 4 * @as(u64, ntokens), page);
}

fn fileSize(ntokens: usize, payload: u64) u64 {
    return payloadOffset(ntokens) + payload + 4 * @as(u64, blocksOf(payload)) + foot_len;
}

pub const Tier = struct {
    gpa: std.mem.Allocator,
    cfg: Config,
    dir: [:0]u8,
    metas: std.ArrayList(Meta) = .empty,
    clock: u64 = 0,
    stats: Stats = .{},
    /// The index, the counters and the folder's files are shared by the lane thread (lookups, drops) and the writer
    /// thread (stores): every public call takes this; file bytes move outside it.
    mu: std.atomic.Mutex = .unlocked,
    /// keys being written now
    inflight: std.ArrayList(u64) = .empty,

    fn lock(t: *Tier) void {
        while (!t.mu.tryLock()) std.Thread.yield() catch {};
    }

    fn unlock(t: *Tier) void {
        t.mu.unlock();
    }

    /// The tier's folder for this build and rank (made when missing), its stored states indexed, damaged or foreign files
    /// and other builds' `tfspill-` folders removed.
    pub fn open(gpa: std.mem.Allocator, cfg: Config) !Tier {
        const dir = try std.fmt.allocPrintSentinel(gpa, "{s}/tfspill-{x:0>16}/rank{d}", .{ cfg.root, cfg.build, cfg.rank }, 0);
        errdefer gpa.free(dir);
        try makePath(dir);
        var t: Tier = .{ .gpa = gpa, .cfg = cfg, .dir = dir };
        errdefer t.deinit();
        t.pruneBuilds();
        t.scan();
        return t;
    }

    pub fn deinit(t: *Tier) void {
        for (t.metas.items) |m| t.gpa.free(m.tokens);
        t.metas.deinit(t.gpa);
        t.inflight.deinit(t.gpa);
        t.gpa.free(t.dir);
    }

    fn path(t: *const Tier, buf: []u8, key: u64, partial: bool) ![:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.{s}", .{ t.dir, key, if (partial) "partial" else "state" }, 0);
    }

    fn find(t: *const Tier, key: u64) ?usize {
        for (t.metas.items, 0..) |m, i| if (m.key == key) return i;
        return null;
    }

    fn hasLocked(t: *const Tier, key: u64) bool {
        return t.find(key) != null or std.mem.indexOfScalar(u64, t.inflight.items, key) != null;
    }

    pub fn has(t: *Tier, key: u64) bool {
        t.lock();
        defer t.unlock();
        return t.hasLocked(key);
    }

    pub fn total(t: *const Tier) u64 {
        var n: u64 = 0;
        for (t.metas.items) |m| n += m.size;
        return n;
    }

    pub fn snapshot(t: *Tier) Stats {
        t.lock();
        defer t.unlock();
        var s = t.stats;
        s.entries = @intCast(t.metas.items.len);
        s.bytes = t.total();
        return s;
    }

    /// The longest stored state whose tokens are a prefix of `prompt` and that leaves a row (strictly shorter), longer than
    /// `longer_than`; with `grid` set only a state on the grid.
    pub fn best(t: *Tier, prompt: []const u32, longer_than: u32, grid: u32) ?*const Meta {
        t.lock();
        defer t.unlock();
        return t.bestLocked(prompt, longer_than, grid);
    }

    /// `best`, its tokens copied into `buf` (the index may drop them as soon as the lock is gone): (position, key).
    pub fn bestInto(t: *Tier, prompt: []const u32, longer_than: u32, grid: u32, buf: *std.ArrayList(u32), min_payload_ok: *const fn (ctx: *anyopaque, at: u32, payload: u64) bool, ctx: *anyopaque) ?struct { at: u32, key: u64 } {
        t.lock();
        defer t.unlock();
        var from = longer_than;
        while (t.bestLocked(prompt, from, grid)) |m| {
            if (min_payload_ok(ctx, m.at, m.payload)) {
                buf.clearRetainingCapacity();
                buf.appendSlice(t.gpa, m.tokens) catch return null;
                return .{ .at = m.at, .key = m.key };
            }
            from = m.at;
        }
        return null;
    }

    fn bestLocked(t: *const Tier, prompt: []const u32, longer_than: u32, grid: u32) ?*const Meta {
        var out: ?*const Meta = null;
        for (t.metas.items) |*m| {
            if (m.at <= longer_than or m.tokens.len > prompt.len or m.at >= prompt.len) continue;
            if (grid > 0 and m.at % grid != 0) continue;
            if (out != null and m.at <= out.?.at) continue;
            if (!std.mem.eql(u32, prompt[0..m.tokens.len], m.tokens)) continue;
            out = m;
        }
        return out;
    }

    pub fn touch(t: *Tier, key: u64) void {
        t.lock();
        defer t.unlock();
        t.touchLocked(key);
    }

    fn touchLocked(t: *Tier, key: u64) void {
        const i = t.find(key) orelse return;
        t.clock += 1;
        t.metas.items[i].used = t.clock;
    }

    /// Whether a state of `tokens` and `payload` bytes would be written now (not short, not stored): the tier's gate before
    /// the engine copies anything off the device.
    pub fn wants(t: *Tier, tokens: []const u32, payload: u64) bool {
        t.lock();
        defer t.unlock();
        return t.wantsLocked(tokens, payload);
    }

    fn wantsLocked(t: *Tier, tokens: []const u32, payload: u64) bool {
        if (tokens.len < t.cfg.min_tokens) {
            t.stats.skipped_short += 1;
            return false;
        }
        if (t.hasLocked(keyOf(tokens))) {
            t.stats.skipped_exists += 1;
            return false;
        }
        if (fileSize(tokens.len, payload) > t.cfg.cap) {
            t.stats.skipped_cap += 1;
            return false;
        }
        return true;
    }

    /// The oldest stored state's key (the next the cap or the floor drops).
    fn oldest(t: *const Tier) ?u64 {
        var out: ?*const Meta = null;
        for (t.metas.items) |*m| if (out == null or m.used < out.?.used) {
            out = m;
        };
        return if (out) |m| m.key else null;
    }

    /// Room for a file of `size` bytes: the oldest files go until the cap holds and the disk keeps its floor. When dropping
    /// every file would not leave the floor either, nothing goes and the write is skipped (false).
    fn room(t: *Tier, size: u64) bool {
        while (t.total() + size > t.cfg.cap) {
            const k = t.oldest() orelse return false;
            t.forgetLocked(k);
            t.stats.pruned += 1;
        }
        const free = t.cfg.free(t.dir) orelse return false;
        const need = t.cfg.min_free +| size;
        if (free +| t.total() < need) return false;
        var freed: u64 = 0;
        while (free +| freed < need) {
            const k = t.oldest() orelse return false;
            freed += t.metas.items[t.find(k).?].size;
            t.forgetLocked(k);
            t.stats.pruned += 1;
        }
        return true;
    }

    pub const Writer = struct {
        tier: *Tier,
        fd: c_int,
        key: u64,
        tokens: []u32,
        at: u32,
        payload: u64,
        written: u64 = 0,
        crc: std.hash.Crc32 = .init(),
        in_block: usize = 0,
        crcs: std.ArrayList(u32) = .empty,
        partial: [1100]u8 = undefined,
        partial_len: usize = 0,

        /// The next bytes of the payload, in order; the CRC of each 8 MiB is kept.
        pub fn put(w: *Writer, bytes: []const u8) Error!void {
            if (w.written + bytes.len > w.payload) return error.SpillWrite;
            var rest = bytes;
            while (rest.len > 0) {
                const n = @min(rest.len, block - w.in_block);
                w.crc.update(rest[0..n]);
                try putAll(w.fd, rest[0..n]);
                w.in_block += n;
                w.written += n;
                rest = rest[n..];
                if (w.in_block == block) try w.closeBlock();
            }
        }

        fn closeBlock(w: *Writer) Error!void {
            w.crcs.append(w.tier.gpa, w.crc.final()) catch return error.SpillWrite;
            w.crc = .init();
            w.in_block = 0;
        }

        /// The CRC table, a footer, an fsync and the rename: the state is stored (or an error, and nothing is).
        pub fn finish(w: *Writer) Error!void {
            errdefer w.abort();
            if (w.written != w.payload) return error.SpillWrite;
            if (w.in_block > 0) try w.closeBlock();
            const table = std.mem.sliceAsBytes(w.crcs.items);
            try putAll(w.fd, table);
            var foot: [foot_len]u8 = undefined;
            @memcpy(foot[0..8], foot_magic);
            std.mem.writeInt(u32, foot[8..12], std.hash.Crc32.hash(table), .little);
            std.mem.writeInt(u32, foot[12..16], headCrc(w.at, w.tokens, w.payload, w.tier.cfg.build, w.tier.cfg.rank), .little);
            try putAll(w.fd, &foot);
            if (std.c.fsync(w.fd) != 0) return error.SpillWrite;
            _ = std.c.close(w.fd);
            w.fd = -1;
            var dest: [1100]u8 = undefined;
            const final = w.tier.path(&dest, w.key, false) catch return error.SpillWrite;
            if (std.c.rename(@ptrCast(w.partial[0..w.partial_len :0]), final) != 0) return error.SpillWrite;
            w.tier.lock();
            defer w.tier.unlock();
            w.tier.dropInflight(w.key);
            const meta: Meta = .{ .key = w.key, .at = w.at, .tokens = w.tokens, .payload = w.payload, .size = fileSize(w.tokens.len, w.payload), .used = 0 };
            w.tier.clock += 1;
            var m = meta;
            m.used = w.tier.clock;
            w.tier.metas.append(w.tier.gpa, m) catch {
                _ = std.c.unlink(final);
                return error.SpillWrite;
            };
            w.tokens = &.{}; // the index owns them now
            w.tier.stats.writes += 1;
            w.tier.stats.write_bytes += m.size;
            w.crcs.deinit(w.tier.gpa);
        }

        /// Give up: the partial file goes.
        pub fn abort(w: *Writer) void {
            w.tier.lock();
            defer w.tier.unlock();
            w.abortLocked();
        }

        fn abortLocked(w: *Writer) void {
            w.tier.dropInflight(w.key);
            if (w.fd >= 0) _ = std.c.close(w.fd);
            w.fd = -1;
            _ = std.c.unlink(@ptrCast(w.partial[0..w.partial_len :0]));
            w.tier.gpa.free(w.tokens);
            w.tokens = &.{};
            w.crcs.deinit(w.tier.gpa);
            w.tier.stats.failed += 1;
        }
    };

    /// Start storing the state of `tokens` (`at` of them the position) with `payload` bytes: room is made first. After
    /// `finish` the state is indexed.
    pub fn begin(t: *Tier, tokens: []const u32, at: u32, payload: u64) Error!Writer {
        t.lock();
        defer t.unlock();
        return t.beginLocked(tokens, at, payload);
    }

    fn dropInflight(t: *Tier, key: u64) void {
        if (std.mem.indexOfScalar(u64, t.inflight.items, key)) |i| _ = t.inflight.swapRemove(i);
    }

    fn beginLocked(t: *Tier, tokens: []const u32, at: u32, payload: u64) Error!Writer {
        if (!t.wantsLocked(tokens, payload)) return if (tokens.len < t.cfg.min_tokens) error.SpillShort else if (t.hasLocked(keyOf(tokens))) error.SpillExists else error.SpillNoRoom;
        const size = fileSize(tokens.len, payload);
        if (!t.room(size)) {
            t.stats.skipped_floor += 1;
            return error.SpillNoRoom;
        }
        const key = keyOf(tokens);
        t.inflight.append(t.gpa, key) catch return error.SpillWrite;
        errdefer t.dropInflight(key);
        var w: Writer = .{ .tier = t, .fd = -1, .key = key, .tokens = &.{}, .at = at, .payload = payload };
        const final_len = (std.fmt.bufPrint(&w.partial, "{s}/{x:0>16}.partial", .{ t.dir, key }) catch return error.SpillWrite).len;
        w.partial[final_len] = 0;
        w.partial_len = final_len;
        w.fd = std.c.open(@ptrCast(w.partial[0..w.partial_len :0]), .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0o600));
        if (w.fd < 0) {
            t.stats.failed += 1;
            return error.SpillWrite;
        }
        w.tokens = t.gpa.dupe(u32, tokens) catch {
            _ = std.c.close(w.fd);
            return error.SpillWrite;
        };
        errdefer w.abortLocked();
        // header, tokens, padding to the payload's page
        var head: [head_len]u8 = @splat(0);
        @memcpy(head[0..8], magic);
        std.mem.writeInt(u32, head[8..12], 1, .little);
        std.mem.writeInt(u32, head[12..16], at, .little);
        std.mem.writeInt(u32, head[16..20], @intCast(tokens.len), .little);
        std.mem.writeInt(u32, head[20..24], blocksOf(payload), .little);
        std.mem.writeInt(u64, head[24..32], payload, .little);
        std.mem.writeInt(u64, head[32..40], t.cfg.build, .little);
        std.mem.writeInt(u32, head[40..44], t.cfg.rank, .little);
        try putAll(w.fd, &head);
        try putAll(w.fd, std.mem.sliceAsBytes(tokens));
        const pad = payloadOffset(tokens.len) - head_len - 4 * @as(u64, tokens.len);
        var zeros: [page]u8 = @splat(0);
        try putAll(w.fd, zeros[0..@intCast(pad)]);
        return w;
    }

    pub const Reader = struct {
        tier: *Tier,
        fd: c_int,
        meta: Meta,
        offset: u64 = 0,
        crc: std.hash.Crc32 = .init(),
        in_block: usize = 0,
        crcs: []u32,
        block_index: usize = 0,
        ntokens: usize,
        bad: bool = false,
        /// the state's tokens, a copy of its own (the index may drop its)
        tokens: []u32 = &.{},

        /// The tokens, handed over to the caller (freed with its allocator, `tier.gpa`).
        pub fn takeTokens(r: *Reader) []u32 {
            defer r.tokens = &.{};
            return r.tokens;
        }

        /// The next `dest.len` payload bytes; each 8 MiB is checked against its CRC as the block completes.
        pub fn read(r: *Reader, dest: []u8) Error!void {
            if (r.offset + dest.len > r.meta.payload) return error.SpillRead;
            var done: usize = 0;
            while (done < dest.len) {
                const n = @min(dest.len - done, block - r.in_block);
                const at = payloadOffset(r.ntokens) + r.offset + done;
                if (!getAll(r.fd, dest[done..][0..n], at)) return error.SpillRead;
                r.crc.update(dest[done..][0..n]);
                r.in_block += n;
                done += n;
                const left = r.meta.payload - (r.offset + done);
                if (r.in_block == block or left == 0) {
                    if (r.crc.final() != r.crcs[r.block_index]) {
                        r.tier.lock();
                        r.tier.stats.crc_failures += 1;
                        r.tier.unlock();
                        r.bad = true;
                        return error.SpillCrc;
                    }
                    r.crc = .init();
                    r.in_block = 0;
                    r.block_index += 1;
                }
            }
            r.offset += dest.len;
        }

        /// Close; the read counts when every byte came and checked.
        pub fn close(r: *Reader) void {
            _ = std.c.close(r.fd);
            r.tier.gpa.free(r.crcs);
            r.tier.gpa.free(r.tokens);
            r.tokens = &.{};
            if (r.bad) return r.tier.forget(r.meta.key); // a block that fails its CRC: the state goes
            if (r.offset == r.meta.payload) {
                r.tier.lock();
                defer r.tier.unlock();
                r.tier.stats.reads += 1;
                r.tier.stats.read_bytes += r.meta.size;
                r.tier.touchLocked(r.meta.key);
            }
        }
    };

    /// Open the stored state `key` for reading; a file that fails its checks is forgotten.
    pub fn reader(t: *Tier, key: u64) Error!Reader {
        t.lock();
        defer t.unlock();
        const i = t.find(key) orelse return error.SpillRead;
        const m = t.metas.items[i];
        var buf: [1100]u8 = undefined;
        const p = t.path(&buf, key, false) catch return error.SpillRead;
        const fd = std.c.open(p, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
        if (fd < 0) {
            t.forgetLocked(key);
            return error.SpillRead;
        }
        errdefer _ = std.c.close(fd);
        const nblocks = blocksOf(m.payload);
        const crcs = t.gpa.alloc(u32, nblocks) catch return error.SpillRead;
        errdefer t.gpa.free(crcs);
        const table_at = payloadOffset(m.tokens.len) + m.payload;
        if (!getAll(fd, std.mem.sliceAsBytes(crcs), table_at)) {
            t.forgetLocked(key);
            return error.SpillRead;
        }
        var foot: [foot_len]u8 = undefined;
        if (!getAll(fd, &foot, table_at + 4 * @as(u64, nblocks)) or !std.mem.eql(u8, foot[0..8], foot_magic) or
            std.mem.readInt(u32, foot[8..12], .little) != std.hash.Crc32.hash(std.mem.sliceAsBytes(crcs)))
        {
            t.forgetLocked(key);
            return error.SpillRead;
        }
        const tokens = t.gpa.dupe(u32, m.tokens) catch return error.SpillRead;
        return .{ .tier = t, .fd = fd, .meta = m, .crcs = crcs, .ntokens = m.tokens.len, .tokens = tokens };
    }

    /// Drop a stored state: its file and its index entry.
    pub fn forget(t: *Tier, key: u64) void {
        t.lock();
        defer t.unlock();
        t.forgetLocked(key);
    }

    fn forgetLocked(t: *Tier, key: u64) void {
        const i = t.find(key) orelse return;
        var buf: [1100]u8 = undefined;
        if (t.path(&buf, key, false)) |p| _ = std.c.unlink(p) else |_| {}
        const m = t.metas.orderedRemove(i);
        t.gpa.free(m.tokens);
    }

    /// Start up: a state is indexed only when its file's size matches its header and its header its name.
    fn scan(t: *Tier) void {
        const d = std.c.opendir(t.dir.ptr) orelse return;
        defer _ = std.c.closedir(d);
        while (std.c.readdir(d)) |ent| {
            const name = std.mem.sliceTo(&ent.name, 0);
            if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
            var buf: [1100]u8 = undefined;
            const file = std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ t.dir, name }, 0) catch continue;
            if (!std.mem.endsWith(u8, name, ".state") or name.len != 22) {
                _ = std.c.unlink(file); // a partial write or a stranger
                continue;
            }
            if (t.load(file, name[0..16])) |m| {
                t.metas.append(t.gpa, m) catch t.gpa.free(m.tokens);
            } else {
                _ = std.c.unlink(file);
                t.stats.failed += 1;
            }
        }
        // the oldest first in the index's order: files carry no time, so the clock follows the directory's order
        for (t.metas.items) |*m| {
            t.clock += 1;
            m.used = t.clock;
        }
    }

    fn load(t: *Tier, file: [:0]const u8, hex: []const u8) ?Meta {
        const name_key = std.fmt.parseInt(u64, hex, 16) catch return null;
        const fd = std.c.open(file, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return null;
        defer _ = std.c.close(fd);
        var head: [head_len]u8 = undefined;
        if (!getAll(fd, &head, 0) or !std.mem.eql(u8, head[0..8], magic) or std.mem.readInt(u32, head[8..12], .little) != 1) return null;
        const at = std.mem.readInt(u32, head[12..16], .little);
        const n = std.mem.readInt(u32, head[16..20], .little);
        const payload = std.mem.readInt(u64, head[24..32], .little);
        if (std.mem.readInt(u64, head[32..40], .little) != t.cfg.build or std.mem.readInt(u32, head[40..44], .little) != t.cfg.rank) return null;
        if (n == 0 or n > 1 << 24 or at == 0 or at > n or std.mem.readInt(u32, head[20..24], .little) != blocksOf(payload)) return null;
        const size = fileSize(n, payload);
        const end = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (end < 0 or @as(u64, @intCast(end)) != size) return null;
        const tokens = t.gpa.alloc(u32, n) catch return null;
        if (!getAll(fd, std.mem.sliceAsBytes(tokens), head_len) or keyOf(tokens) != name_key) {
            t.gpa.free(tokens);
            return null;
        }
        return .{ .key = name_key, .at = at, .tokens = tokens, .payload = payload, .size = size, .used = 0 };
    }

    /// Other builds' folders under the root go (ours only: a `tfspill-<16 hex>` name).
    fn pruneBuilds(t: *Tier) void {
        var keep_buf: [40]u8 = undefined;
        const keep = std.fmt.bufPrint(&keep_buf, "tfspill-{x:0>16}", .{t.cfg.build}) catch return;
        var rootz: [1100]u8 = undefined;
        const root = std.fmt.bufPrintSentinel(&rootz, "{s}", .{t.cfg.root}, 0) catch return;
        const d = std.c.opendir(root.ptr) orelse return;
        defer _ = std.c.closedir(d);
        while (std.c.readdir(d)) |ent| {
            const name = std.mem.sliceTo(&ent.name, 0);
            if (name.len != 24 or !std.mem.startsWith(u8, name, "tfspill-") or std.mem.eql(u8, name, keep)) continue;
            if (!allHex(name[8..])) continue;
            var sub: [1100]u8 = undefined;
            const p = std.fmt.bufPrintSentinel(&sub, "{s}/{s}", .{ t.cfg.root, name }, 0) catch continue;
            removeTree(p);
        }
    }
};

fn allHex(text: []const u8) bool {
    for (text) |c| if (!std.ascii.isHex(c)) return false;
    return true;
}

/// A folder of ours and its files and rank folders, removed (no links followed).
fn removeTree(p: [:0]const u8) void {
    const d = std.c.opendir(p.ptr) orelse return;
    defer _ = std.c.closedir(d);
    while (std.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        var buf: [1100]u8 = undefined;
        const child = std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ p, name }, 0) catch continue;
        if (ent.type == std.c.DT.DIR) removeTree(child) else _ = std.c.unlink(child);
    }
    _ = std.c.rmdir(p.ptr);
}

fn headCrc(at: u32, tokens: []const u32, payload: u64, build: u64, rank: u32) u32 {
    var h = std.hash.Crc32.init();
    h.update(std.mem.asBytes(&at));
    h.update(std.mem.sliceAsBytes(tokens));
    h.update(std.mem.asBytes(&payload));
    h.update(std.mem.asBytes(&build));
    h.update(std.mem.asBytes(&rank));
    return h.final();
}

fn makePath(dir: [:0]const u8) !void {
    var buf: [1100]u8 = undefined;
    var i: usize = 1;
    while (i <= dir.len) : (i += 1) {
        if (i < dir.len and dir[i] != '/') continue;
        const part = std.fmt.bufPrintSentinel(&buf, "{s}", .{dir[0..i]}, 0) catch return error.SpillDir;
        const rc = std.c.mkdir(part, 0o700);
        if (rc != 0 and std.c.errno(rc) != .EXIST) return error.SpillDir;
    }
}

fn putAll(fd: c_int, bytes: []const u8) Error!void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + done, @min(bytes.len - done, 1 << 30));
        if (n <= 0) return error.SpillWrite;
        done += @intCast(n);
    }
}

fn getAll(fd: c_int, bytes: []u8, at: u64) bool {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.pread(fd, bytes.ptr + done, @min(bytes.len - done, 1 << 30), @intCast(at + done));
        if (n <= 0) return false;
        done += @intCast(n);
    }
    return true;
}

// -- tests ---------------------------------------------------------------------------------------------------------------

fn unlimited(_: [:0]const u8) ?u64 {
    return 1 << 50;
}

var fake_free: u64 = 0;
fn fakeFree(_: [:0]const u8) ?u64 {
    return fake_free;
}

fn testRoot(buf: []u8, tag: []const u8) [:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "/tmp/tf-spill-test-{d}-{s}", .{ std.c.getpid(), tag }, 0) catch unreachable;
}

fn cleanup(root: [:0]const u8) void {
    removeTree(root);
}

fn store(t: *Tier, tokens: []const u32, at: u32, payload: []const u8) !void {
    var w = try t.begin(tokens, at, payload.len);
    var i: usize = 0;
    while (i < payload.len) : (i += 5000) try w.put(payload[i..@min(payload.len, i + 5000)]);
    try w.finish();
}

fn toks(buf: []u32, seed: u32) []u32 {
    for (buf, 0..) |*t, i| t.* = seed *% 7919 +% @as(u32, @intCast(i));
    return buf;
}

test "a state is written, indexed, found by a longer prompt and read back with its CRCs" {
    const gpa = std.testing.allocator;
    var rb: [200]u8 = undefined;
    const root = testRoot(&rb, "a");
    defer cleanup(root);
    var t = try Tier.open(gpa, .{ .root = root, .build = 0xabc, .cap = 1 << 30, .min_tokens = 16, .min_free = 0, .free = unlimited });
    defer t.deinit();
    var tb: [64]u32 = undefined;
    const tokens = toks(&tb, 1);
    const payload = try gpa.alloc(u8, 3 * block + 12345);
    defer gpa.free(payload);
    var prng = std.Random.DefaultPrng.init(7);
    prng.random().bytes(payload);
    try store(&t, tokens, 64, payload);
    try std.testing.expect(t.has(keyOf(tokens)));
    var longer: [80]u32 = undefined;
    @memcpy(longer[0..64], tokens);
    for (longer[64..], 0..) |*x, i| x.* = 50000 + @as(u32, @intCast(i));
    const hit = t.best(&longer, 0, 64).?;
    try std.testing.expectEqual(@as(u32, 64), hit.at);
    try std.testing.expect(t.best(tokens, 0, 64) == null); // a prompt of exactly its tokens leaves no row
    try std.testing.expect(t.best(&longer, 64, 64) == null); // nothing longer than a kept one
    var r = try t.reader(hit.key);
    const back = try gpa.alloc(u8, payload.len);
    defer gpa.free(back);
    var off: usize = 0;
    while (off < back.len) : (off += 1 << 20) try r.read(back[off..@min(back.len, off + (1 << 20))]);
    r.close();
    try std.testing.expectEqualSlices(u8, payload, back);
    const s = t.snapshot();
    try std.testing.expectEqual(@as(u64, 1), s.writes);
    try std.testing.expectEqual(@as(u64, 1), s.reads);
    try std.testing.expectEqual(@as(u32, 1), s.entries);
}

test "a restart finds what a clean write left, and drops a truncated file, a partial one and a stranger" {
    const gpa = std.testing.allocator;
    var rb: [200]u8 = undefined;
    const root = testRoot(&rb, "b");
    defer cleanup(root);
    const cfg: Config = .{ .root = root, .build = 0x77, .cap = 1 << 30, .min_tokens = 16, .min_free = 0, .free = unlimited };
    var dir_copy: [400]u8 = undefined;
    {
        var t = try Tier.open(gpa, cfg);
        defer t.deinit();
        var tb: [40]u32 = undefined;
        var tb2: [48]u32 = undefined;
        const payload: [10000]u8 = @splat(9);
        try store(&t, toks(&tb, 1), 40, &payload);
        try store(&t, toks(&tb2, 2), 48, &payload);
        @memcpy(dir_copy[0..t.dir.len], t.dir);
        dir_copy[t.dir.len] = 0;
        // damage the second: a few bytes short
        var f: [1100]u8 = undefined;
        const p = try t.path(&f, keyOf(toks(&tb2, 2)), false);
        const tfd = std.c.open(p, .{ .ACCMODE = .WRONLY }, @as(std.c.mode_t, 0));
        try std.testing.expectEqual(@as(c_int, 0), std.c.ftruncate(tfd, @intCast(fileSize(48, 10000) - 3)));
        _ = std.c.close(tfd);
        // and a partial write and a stranger
        const part = try std.fmt.bufPrintSentinel(&f, "{s}/0000000000000001.partial", .{t.dir}, 0);
        const fd = std.c.open(part, .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(std.c.mode_t, 0o600));
        _ = std.c.close(fd);
    }
    var t = try Tier.open(gpa, cfg);
    defer t.deinit();
    try std.testing.expectEqual(@as(usize, 1), t.metas.items.len);
    try std.testing.expectEqual(@as(u32, 40), t.metas.items[0].at);
    try std.testing.expect(t.stats.failed >= 1);
    var one: [40]u32 = undefined;
    _ = toks(&one, 1);
    try std.testing.expect(t.has(keyOf(&one)));
    // a build with another hash prunes ours at open
    var t2 = try Tier.open(gpa, .{ .root = root, .build = 0x78, .cap = 1 << 30, .min_tokens = 16, .min_free = 0, .free = unlimited });
    defer t2.deinit();
    try std.testing.expectEqual(@as(usize, 0), t2.metas.items.len);
    const gone = std.c.opendir(@ptrCast(dir_copy[0..t.dir.len :0]));
    try std.testing.expect(gone == null);
}

test "a flipped payload byte fails its block's CRC and the short, the stored and the over-cap are refused" {
    const gpa = std.testing.allocator;
    var rb: [200]u8 = undefined;
    const root = testRoot(&rb, "c");
    defer cleanup(root);
    var t = try Tier.open(gpa, .{ .root = root, .build = 1, .cap = 100000, .min_tokens = 16, .min_free = 0, .free = unlimited });
    defer t.deinit();
    var tb: [32]u32 = undefined;
    const tokens = toks(&tb, 3);
    const payload: [20000]u8 = @splat(5);
    try store(&t, tokens, 32, &payload);
    var f: [1100]u8 = undefined;
    const p = try t.path(&f, keyOf(tokens), false);
    const fd = std.c.open(p, .{ .ACCMODE = .RDWR }, @as(std.c.mode_t, 0));
    try std.testing.expectEqual(@as(isize, 1), std.c.pwrite(fd, "X", 1, @intCast(payloadOffset(32) + 100)));
    _ = std.c.close(fd);
    var r = try t.reader(keyOf(tokens));
    var back: [20000]u8 = undefined;
    try std.testing.expectError(error.SpillCrc, r.read(&back));
    r.close();
    try std.testing.expectEqual(@as(u64, 1), t.stats.crc_failures);
    try std.testing.expect(!t.has(keyOf(tokens))); // forgotten
    try std.testing.expectError(error.SpillShort, t.begin(tokens[0..8], 8, 10));
    try store(&t, tokens, 32, &payload);
    try std.testing.expectError(error.SpillExists, t.begin(tokens, 32, 20000));
    var short: [8]u32 = undefined;
    try std.testing.expectError(error.SpillShort, t.begin(toks(&short, 4), 8, 100));
    var big: [32]u32 = undefined;
    try std.testing.expectError(error.SpillNoRoom, t.begin(toks(&big, 5), 32, 200000)); // past the cap altogether
}

test "the cap drops the oldest files, and the free-disk floor drops more or skips the write" {
    const gpa = std.testing.allocator;
    var rb: [200]u8 = undefined;
    const root = testRoot(&rb, "d");
    defer cleanup(root);
    const payload: [20000]u8 = @splat(1);
    const size = fileSize(32, 20000);
    var t = try Tier.open(gpa, .{ .root = root, .build = 2, .cap = 2 * size + 10, .min_tokens = 16, .min_free = 1000, .free = fakeFree });
    defer t.deinit();
    fake_free = 1 << 40;
    var a: [32]u32 = undefined;
    var b: [32]u32 = undefined;
    var c: [32]u32 = undefined;
    try store(&t, toks(&a, 1), 32, &payload);
    try store(&t, toks(&b, 2), 32, &payload);
    t.touch(keyOf(toks(&a, 1))); // a is now the newer
    try store(&t, toks(&c, 3), 32, &payload); // the cap holds two: b, the oldest, goes
    try std.testing.expect(t.has(keyOf(toks(&a, 1))) and t.has(keyOf(toks(&c, 3))) and !t.has(keyOf(toks(&b, 2))));
    try std.testing.expectEqual(@as(u64, 1), t.stats.pruned);
    // the cap keeps two files, so the new one drops the older of a and c; then the disk would be left under the floor, so the
    // last old one goes too, and the write is made
    fake_free = size + 500;
    var d: [32]u32 = undefined;
    try store(&t, toks(&d, 4), 32, &payload);
    try std.testing.expectEqual(@as(usize, 1), t.metas.items.len);
    try std.testing.expect(t.has(keyOf(toks(&d, 4))));
    // when dropping everything would not leave the floor, nothing is dropped and nothing written
    fake_free = 100;
    var e: [32]u32 = undefined;
    const before = t.metas.items.len;
    try std.testing.expectError(error.SpillNoRoom, t.begin(toks(&e, 5), 32, 20000));
    try std.testing.expectEqual(before, t.metas.items.len);
    try std.testing.expect(t.stats.skipped_floor >= 1);
}

test "a writer thread stores while the lane thread looks up, reads and forgets: the index stays whole" {
    const gpa = std.testing.allocator;
    var rb: [200]u8 = undefined;
    const root = testRoot(&rb, "e");
    defer cleanup(root);
    var t = try Tier.open(gpa, .{ .root = root, .build = 5, .cap = 1 << 30, .min_tokens = 16, .min_free = 0, .free = unlimited });
    defer t.deinit();
    const Job = struct {
        fn run(tier: *Tier, done: *std.atomic.Value(bool)) void {
            const payload: [30000]u8 = @splat(7);
            var i: u32 = 0;
            while (i < 40) : (i += 1) {
                var tb: [32]u32 = undefined;
                store(tier, toks(&tb, 100 + i), 32, &payload) catch {};
            }
            done.store(true, .release);
        }
    };
    var done: std.atomic.Value(bool) = .init(false);
    const th = try std.Thread.spawn(.{}, Job.run, .{ &t, &done });
    var forgotten: u32 = 0;
    while (!done.load(.acquire)) {
        var q: [64]u32 = undefined;
        _ = toks(&q, 100);
        if (t.best(&q, 0, 0)) |m| {
            const key = m.key;
            if (t.reader(key)) |r0| {
                var r = r0;
                var back: [30000]u8 = undefined;
                r.read(&back) catch {};
                r.close();
            } else |_| {}
        }
        const s = t.snapshot();
        if (s.entries > 10) {
            var tb: [32]u32 = undefined;
            t.forget(keyOf(toks(&tb, 100 + forgotten % 40)));
            forgotten += 1;
        }
    }
    th.join();
    const s = t.snapshot();
    try std.testing.expect(s.writes + s.failed >= 1);
    try std.testing.expectEqual(@as(usize, 0), t.inflight.items.len);
}
