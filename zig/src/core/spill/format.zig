//! Port of wojo's GLM 0088 (PR #78) and 0110 (PR #124; BadAd84's block layout, lineage and fsync order) spill tier:
//! the native spill v2 on-disk format. Fixed-schema binary records only (no JSON, no pickles, no process pointers):
//! every count is checked against limits derived from the model and context, and every length against the file,
//! before anything is allocated. A foreign or damaged record is a cache miss, never an error that stops the engine.
//!
//! Layout of one rank's store: `<root>/<prefix><compat:16 hex>/rank<N>/{owner.lock, blocks/, points/}` (the
//! family's prefix, `store.Config.prefix`).
//! - `blocks/<name>.blk`: one block's packed per-rank planes, exactly `bytes` long. Its name is a digest chained over
//!   the canonical key ids up to the block's end and salted with the computation lineage (`blockNames`), so a later
//!   turn or a fork names the blocks its earlier turns wrote. Equal bytes under another name are a hard link.
//! - `points/<key>.ids` (keyed ids, then raw ids when they differ), `points/<key>.st` (named small-state sections),
//!   then `points/<key>.man` (this file's `Manifest`), written last: a point with a manifest is complete.
//! Every file goes to a unique temporary name starting with '.', is fsynced and renamed; a restart deletes the rest.
const std = @import("std");
const builtin = @import("builtin");
const Sha256 = std.crypto.hash.sha2.Sha256;
const Crc32 = std.hash.Crc32;
const Allocator = std.mem.Allocator;

comptime {
    // ids and fields are hashed and stored as their little-endian bytes
    std.debug.assert(builtin.cpu.arch.endian() == .little);
}

pub const Digest = [32]u8;
pub const zero_digest: Digest = @splat(0);

/// The native spill format's manifest version (bumped on any byte change below). Version 3 (pictures): a point with
/// media writes 48-byte media entries naming the picture-id scheme its keyed ids were made with (`pictureId`); a text
/// point is still written, byte for byte, as version 2, and version 2 manifests without media stay valid text points.
pub const version: u32 = 3;
/// The version a point without media is written as (and the oldest one read).
pub const text_version: u32 = 2;
/// The store namespace's version, in the compat digest: unchanged by version 3, so a store's text points survive it.
/// (A change that cannot read the older points bumps it: the directory name changes and the old store is pruned.)
pub const namespace_version: u32 = 2;
pub const magic = "TFSPILL\x02".*;
/// The manifest's fixed header.
pub const header_bytes = 256;
const fork_bytes = 20;
const media_bytes = 48; // version 3: start, len, kind, scheme, SHA-256
const media_v2_bytes = 40; // version 2 (never written with media: such a manifest is foreign)
/// A version-3 media entry's kind: a picture (an image's span of the prompt).
pub const media_picture: u32 = 1;
/// The scheme a picture's keyed ids are made with: `pictureId` (the DeepSeek recipe's 0006 image ids, seeded by the
/// digest's first 8 bytes).
pub const picture_scheme: u32 = 1;
const block_ref_bytes = 80;
const section_bytes = 56;
const trailer_bytes = 8;

/// How a rank's planes are laid out on disk. Replicated and owner-sharded stores never share a namespace (the compat
/// digest carries it too); only `replicated` is written in round 1.
pub const OwnerLayout = enum(u32) { replicated = 0, owner_sharded = 1 };

/// Bounds every parse checks before it allocates, derived from the context window and the model descriptor.
pub const Limits = struct {
    /// Most tokens a point may hold (the context window).
    max_tokens: u32,
    /// Tokens a block (the engine's chunk rows: 2048).
    block_tokens: u32,
    /// A full block's bytes on this rank (exact: a full block of any other length is damaged).
    block_bytes: u64,
    /// Most small-state bytes a point carries.
    max_small_bytes: u64,
    /// `owner_sharded` (0118): a full block's bytes on this rank for a point whose part (`Lineage.part`) another rank
    /// owns: that part's owner alone holds (and stores) the sharded planes, the other ranks only the replicated ones.
    /// 0: every full block is `block_bytes` (a replicated store).
    foreign_block_bytes: u64 = 0,
    /// `owner_sharded`: the part this rank owns (a point of that part has full blocks of `block_bytes`).
    own_part: u32 = 0,
    max_media: u32 = 4096,
    max_sections: u32 = 256,
    max_forks: u32 = 4096,

    /// A full block's exact bytes for a point of part `part` (its lineage's salt): `block_bytes`, or on an
    /// owner-sharded store `block_bytes` in this rank's own part and `foreign_block_bytes` in another; null for a
    /// part-less point there (every owner-sharded point names its part).
    pub fn fullBlockBytes(l: Limits, part: ?u32) ?u64 {
        if (l.foreign_block_bytes == 0) return l.block_bytes;
        const p = part orelse return null;
        return if (p == l.own_part) l.block_bytes else l.foreign_block_bytes;
    }

    pub fn maxBlocks(l: Limits) u32 {
        return std.math.divCeil(u32, l.max_tokens, l.block_tokens) catch 0;
    }

    /// The largest manifest these limits admit.
    pub fn maxManifest(l: Limits) u64 {
        return header_bytes + @as(u64, l.max_forks) * fork_bytes + @as(u64, l.max_media) * media_bytes +
            @as(u64, l.maxBlocks()) * block_ref_bytes + @as(u64, l.max_sections) * section_bytes + trailer_bytes;
    }
};

/// A picture inside a point: its prompt positions and the full SHA-256 of its payload (per-position picture keys are
/// shortened; disk identity takes the whole digest and the span). A picture may run past the point's cut
/// (`start < at < start + len`). (extern, so a tier can hand over its prompt cache's records as they are; v1.0.7's
/// prompt cache keeps no pictures, so a tier here passes none.)
pub const Media = extern struct {
    start: u32,
    len: u32,
    digest: Digest,

    pub fn eql(a: Media, b: Media) bool {
        return a.start == b.start and a.len == b.len and std.mem.eql(u8, &a.digest, &b.digest);
    }
};

/// Keyed ids at or above this are pictures' (`pictureId`); vocabulary ids stay below it.
pub const picture_ids: u32 = 0x8000_0000;

/// Scheme 1's keyed id for position `j` of a picture whose payload's SHA-256 is `digest`. A prompt cache that keys
/// pictures must make the same ids (DeepSeek's prompt_cache.pictureId does; v1.0.7's keys no pictures).
pub fn pictureId(digest: *const Digest, j: u32) u32 {
    const seed = std.mem.readInt(u64, digest[0..8], .little);
    return picture_ids | (@as(u32, @truncate(std.hash.Wyhash.hash(seed, std.mem.asBytes(&j)))) & 0x7fff_ffff);
}

/// Whether a point's ids agree with its media: spans in order, not overlapping, not empty, each starting inside the
/// ids; every keyed id in a span is `pictureId(digest, offset)` and every other one a vocabulary id; with media the raw
/// ids are required, equal the keyed ones outside the spans and hold model ids (below `picture_ids`) inside them.
/// Bounded by the ids' length; what keep and recovery check before a point with pictures is trusted.
pub fn mediaIds(keyed: []const u32, raw: ?[]const u32, media: []const Media) bool {
    if (raw) |r| if (r.len != keyed.len) return false;
    if (media.len > 0 and raw == null) return false;
    var at: usize = 0;
    for (media) |m| {
        if (m.len == 0 or m.start < at or m.start >= keyed.len) return false;
        const stop: usize = @intCast(@min(@as(u64, m.start) + m.len, keyed.len));
        if (!textIds(keyed[at..m.start], if (raw) |r| r[at..m.start] else null)) return false;
        for (keyed[m.start..stop], 0..) |id, j| if (id != pictureId(&m.digest, @intCast(j))) return false;
        if (raw) |r| for (r[m.start..stop]) |id| if (id >= picture_ids) return false;
        at = @intCast(@as(u64, m.start) + m.len); // (past the ids when the picture straddles the cut: the loop ends)
    }
    if (at < keyed.len) return textIds(keyed[at..], if (raw) |r| r[at..] else null);
    return true;
}

fn textIds(keyed: []const u32, raw: ?[]const u32) bool {
    for (keyed) |id| if (id >= picture_ids) return false;
    if (raw) |r| if (!std.mem.eql(u32, keyed, r)) return false;
    return true;
}

/// One fork of a lineage: a computation that resumed at `at` and computed its own rows from there.
pub const Fork = struct { name: [16]u8, at: u32 };

/// The computation a state's rows belong to. Two prefills of the same ids may differ in their numerical history, so
/// only equal lineages (through a block) may share that block by name. A fresh prefill has its own root; a resume
/// inherits the source's lineage and forks it at the cut (the same names before the block holding the cut, its own
/// from there). `part` is 0118's part salt (owner-sharded pools): replaced, never stacked.
pub const Lineage = struct {
    root: [16]u8,
    part: ?u32 = null,
    forks: []const Fork = &.{},

    /// A fresh prefill's lineage: `boot` (this process's nonce) and its own sequence number.
    pub fn fresh(boot: [16]u8, seq: u64) Lineage {
        var h = Sha256.init(.{});
        h.update("tf-spill/lineage-root");
        h.update(&boot);
        h.update(std.mem.asBytes(&seq));
        const d = h.finalResult();
        return .{ .root = d[0..16].* };
    }

    /// `l` resumed at `at` by computation `name` (gpa-owned forks; free with `deinit`).
    pub fn fork(l: Lineage, gpa: Allocator, name: [16]u8, at: u32) !Lineage {
        const forks = try gpa.alloc(Fork, l.forks.len + 1);
        @memcpy(forks[0..l.forks.len], l.forks);
        forks[l.forks.len] = .{ .name = name, .at = at };
        return .{ .root = l.root, .part = l.part, .forks = forks };
    }

    /// `l` for rows that live in part `part` of an owner-sharded pool: the part replaces any earlier one (a lineage
    /// read back from disk or forked from a kept state already has one), forks untouched. Shares `l.forks`.
    pub fn inPart(l: Lineage, part: u32) Lineage {
        return .{ .root = l.root, .part = part, .forks = l.forks };
    }

    pub fn clone(l: Lineage, gpa: Allocator) !Lineage {
        return .{ .root = l.root, .part = l.part, .forks = try gpa.dupe(Fork, l.forks) };
    }

    pub fn deinit(l: *Lineage, gpa: Allocator) void {
        gpa.free(l.forks);
        l.forks = &.{};
    }

    pub fn eql(a: Lineage, b: Lineage) bool {
        if (!std.mem.eql(u8, &a.root, &b.root) or a.part != b.part or a.forks.len != b.forks.len) return false;
        for (a.forks, b.forks) |x, y| if (x.at != y.at or !std.mem.eql(u8, &x.name, &y.name)) return false;
        return true;
    }
};

/// One block of a point: tokens [start, end) and its chained name.
pub const Span = struct { start: u32, end: u32, name: Digest };

fn idsBytes(ids: []const u32) []const u8 {
    return std.mem.sliceAsBytes(ids);
}

/// Each block of `ids` under `lineage`: whole blocks, then the tail [k * block, n) if any. A name covers the lineage
/// (its root and part, then each fork whose position lies in the block), every id up to the block's end and, for a
/// tail, its length: equal names mean equal prefixes of one computation. A whole block's ids enter through `ids_chain`
/// (`chain(ids, block)`, which already commits to every id before the block's end), chained through the previous whole
/// block's name; only a tail's ids are hashed here. So naming every block costs one small hash a block on top of the
/// chain, not another pass over the ids. `out` holds ceil(n / block) spans.
pub fn blockNames(lineage: Lineage, ids_chain: []const Digest, ids: []const u32, block: u32, out: []Span) void {
    std.debug.assert(block > 0 and out.len == std.math.divCeil(usize, ids.len, block) catch unreachable);
    std.debug.assert(ids_chain.len == ids.len / block);
    var head: Digest = undefined;
    {
        var h = Sha256.init(.{});
        h.update("tf-spill/lineage");
        h.update(&lineage.root);
        if (lineage.part) |p| {
            h.update("#part");
            h.update(std.mem.asBytes(&p));
        }
        h.final(&head);
    }
    var s: usize = 0;
    var i: usize = 0;
    while (s < ids.len) : (i += 1) {
        const e = @min(ids.len, s + block);
        var h = Sha256.init(.{});
        h.update("tf-spill/block");
        h.update(&head);
        for (lineage.forks) |f| {
            if (s <= f.at and f.at < e) {
                h.update("|fork");
                h.update(&f.name);
                h.update(std.mem.asBytes(&f.at));
            }
        }
        const whole = e - s == block;
        if (whole) {
            h.update("|ids");
            h.update(&ids_chain[i]);
        } else {
            const end: u32 = @intCast(e);
            h.update("|tail");
            h.update(idsBytes(ids[s..e]));
            h.update(std.mem.asBytes(&end));
        }
        var d: Digest = undefined;
        h.final(&d);
        out[i] = .{ .start = @intCast(s), .end = @intCast(e), .name = d };
        if (whole) head = d;
        s = e;
    }
}

fn chainStep(prev: *Digest, ids: []const u32) void {
    var h = Sha256.init(.{});
    h.update(prev);
    h.update(idsBytes(ids));
    h.final(prev);
}

fn chainSeed() Digest {
    var d: Digest = undefined;
    Sha256.hash("tf-spill/ids", &d, .{});
    return d;
}

/// The id chain of `ids` (no lineage) at each whole block's end: two id lists share a prefix of k whole blocks
/// exactly when their first k chain digests agree. Appends to `out`. The one pass over a state's ids: its block
/// names and its key derive from this chain.
pub fn chain(gpa: Allocator, ids: []const u32, block: u32, out: *std.ArrayList(Digest)) !void {
    try out.ensureUnusedCapacity(gpa, ids.len / block);
    var prev = chainSeed();
    var s: usize = 0;
    while (s + block <= ids.len) : (s += block) {
        chainStep(&prev, ids[s .. s + block]);
        out.appendAssumeCapacity(prev);
    }
}

/// A point's key from its id chain (`chain(keyed, block)`): its length, the block size, the last whole block's chain
/// digest (every id before it), the tail's ids and the media identities (whichever computation produced it).
pub fn pointKeyChained(ids_chain: []const Digest, keyed: []const u32, block: u32, media: []const Media) Digest {
    std.debug.assert(block > 0 and ids_chain.len == keyed.len / block);
    const last = if (ids_chain.len > 0) ids_chain[ids_chain.len - 1] else chainSeed();
    return keyOf(last, keyed, ids_chain.len * block, block, media);
}

/// A point's key from its keyed ids alone (the chain computed here, nothing allocated): equals `pointKeyChained`.
pub fn pointKey(keyed: []const u32, block: u32, media: []const Media) Digest {
    std.debug.assert(block > 0);
    var last = chainSeed();
    const whole = keyed.len / block;
    for (0..whole) |i| chainStep(&last, keyed[i * block ..][0..block]);
    return keyOf(last, keyed, whole * block, block, media);
}

fn keyOf(last: Digest, keyed: []const u32, whole: usize, block: u32, media: []const Media) Digest {
    var h = Sha256.init(.{});
    h.update("tf-spill/point");
    const n: u32 = @intCast(keyed.len);
    h.update(std.mem.asBytes(&n));
    h.update(std.mem.asBytes(&block));
    h.update(&last);
    h.update(idsBytes(keyed[whole..]));
    const m: u32 = @intCast(media.len);
    h.update(std.mem.asBytes(&m));
    if (m > 0) { // (version 3: the scheme the keyed ids were made with; a text point's key is unchanged)
        h.update("|pictures");
        h.update(std.mem.asBytes(&picture_scheme));
    }
    for (media) |x| {
        h.update(std.mem.asBytes(&x.start));
        h.update(std.mem.asBytes(&x.len));
        h.update(&x.digest);
    }
    return h.finalResult();
}

/// The compat digest a store's directory is named by: the caller's common digest (build, kit, model, tokenizer,
/// geometry, TP plan; see PORT-SPILL-V2.md) and this format's own fields. The caller exchanges and agrees it.
pub fn compatDigest(common: Digest, layout: OwnerLayout, block_tokens: u32, block_bytes: u64) Digest {
    var h = Sha256.init(.{});
    h.update("tf-spill/compat");
    h.update(std.mem.asBytes(&namespace_version));
    h.update(&common);
    const l: u32 = @intFromEnum(layout);
    h.update(std.mem.asBytes(&l));
    h.update(std.mem.asBytes(&block_tokens));
    h.update(std.mem.asBytes(&block_bytes));
    return h.finalResult();
}

pub fn hex(d: *const Digest) [64]u8 {
    return std.fmt.bytesToHex(d.*, .lower);
}

pub fn parseHex(s: []const u8) ?Digest {
    if (s.len != 64) return null;
    var d: Digest = undefined;
    _ = std.fmt.hexToBytes(&d, s) catch return null;
    return d;
}

pub fn crc32(bytes: []const u8) u32 {
    return Crc32.hash(bytes);
}

pub fn sha256(bytes: []const u8) Digest {
    var d: Digest = undefined;
    Sha256.hash(bytes, &d, .{});
    return d;
}

/// A small-state section's name: at most 32 bytes.
pub const Name = struct {
    bytes: [32]u8 = @splat(0),
    len: u8 = 0,

    pub fn of(s: []const u8) !Name {
        if (s.len == 0 or s.len > 32) return error.BadSectionName;
        var n: Name = .{ .len = @intCast(s.len) };
        @memcpy(n.bytes[0..s.len], s);
        return n;
    }

    pub fn slice(n: *const Name) []const u8 {
        return n.bytes[0..n.len];
    }
};

/// One block a point names: its chained name, its tokens and bytes, CRC-32 (checked on every read) and content
/// SHA-256 (what dedup compares; CRC alone never authorizes a link).
pub const BlockRef = struct { name: Digest, tokens: u32, bytes: u64, crc: u32, sha: Digest };

/// A named small-state section inside the point's `.st` file.
pub const Section = struct { name: Name, offset: u64, len: u64, crc: u32 };

pub const Flags = packed struct(u32) {
    shared: bool = false,
    has_raw: bool = false,
    /// The last block was captured from rows still being written (a mutable tail; native points are chunk-aligned).
    mutable_tail: bool = false,
    _: u29 = 0,
};

/// Where a point's computation came from: the point it resumed (informational; eviction uses the id tree).
pub const Origin = struct { key: Digest, at: u32 };

/// A point's manifest: the last file written, the only one that makes a point readable.
pub const Manifest = struct {
    key: Digest,
    at: u32,
    block_tokens: u32,
    flags: Flags = .{},
    group: u64 = 0,
    created_ns: i64 = 0,
    owner_layout: OwnerLayout = .replicated,
    owner_epoch: u32 = 0,
    lineage: Lineage,
    origin: ?Origin = null,
    ids_bytes: u64,
    ids_crc: u32,
    small_bytes: u64,
    small_crc: u32,
    media: []const Media = &.{},
    blocks: []const BlockRef = &.{},
    sections: []const Section = &.{},

    pub fn deinit(m: *Manifest, gpa: Allocator) void {
        gpa.free(m.media);
        gpa.free(m.blocks);
        gpa.free(m.sections);
        gpa.free(m.lineage.forks);
        m.* = undefined;
    }

    /// The version this manifest is written as: 3 with media, else 2 (a text point's bytes as before).
    pub fn versionOf(m: *const Manifest) u32 {
        return if (m.media.len > 0) version else text_version;
    }

    pub fn encodedLen(m: *const Manifest) u64 {
        return header_bytes + m.lineage.forks.len * fork_bytes + m.media.len * media_bytes +
            m.blocks.len * block_ref_bytes + m.sections.len * section_bytes + trailer_bytes;
    }

    /// The manifest's bytes (gpa-owned).
    pub fn encode(m: *const Manifest, gpa: Allocator, compat: Digest) ![]u8 {
        const out = try gpa.alloc(u8, @intCast(m.encodedLen()));
        @memset(out, 0);
        var w: Cursor = .{ .buf = out };
        w.bytes(&magic);
        w.int(u32, m.versionOf());
        w.int(u32, header_bytes);
        w.bytes(&compat);
        w.bytes(&m.key);
        w.int(u32, m.at);
        w.int(u32, m.block_tokens);
        w.int(u32, @bitCast(m.flags));
        w.int(u32, @intCast(m.media.len));
        w.int(u32, @intCast(m.blocks.len));
        w.int(u32, @intCast(m.sections.len));
        w.int(u32, @intCast(m.lineage.forks.len));
        w.int(u32, @intFromEnum(m.owner_layout));
        w.int(u32, m.owner_epoch);
        w.int(u64, m.group);
        w.int(i64, m.created_ns);
        w.bytes(&m.lineage.root);
        w.int(u32, @intFromBool(m.lineage.part != null));
        w.int(u32, m.lineage.part orelse 0);
        w.int(u32, @intFromBool(m.origin != null));
        w.int(u32, if (m.origin) |o| o.at else 0);
        w.bytes(if (m.origin) |*o| &o.key else &zero_digest);
        w.int(u64, m.ids_bytes);
        w.int(u32, m.ids_crc);
        w.int(u32, m.small_crc);
        w.int(u64, m.small_bytes);
        std.debug.assert(w.at <= header_bytes);
        w.at = header_bytes;
        for (m.lineage.forks) |f| {
            w.bytes(&f.name);
            w.int(u32, f.at);
        }
        for (m.media) |x| {
            w.int(u32, x.start);
            w.int(u32, x.len);
            w.int(u32, media_picture);
            w.int(u32, picture_scheme);
            w.bytes(&x.digest);
        }
        for (m.blocks) |b| {
            w.bytes(&b.name);
            w.int(u32, b.tokens);
            w.int(u32, b.crc);
            w.int(u64, b.bytes);
            w.bytes(&b.sha);
        }
        for (m.sections) |s| {
            w.bytes(&s.name.bytes);
            w.int(u64, s.offset);
            w.int(u64, s.len);
            w.int(u32, s.crc);
            w.int(u32, s.name.len);
        }
        w.int(u32, crc32(out[0..w.at]));
        w.int(u32, 0);
        std.debug.assert(w.at == out.len);
        return out;
    }

    pub const ParseError = error{ Foreign, Corrupt, OutOfMemory };

    /// The manifest in `raw` (a whole file), checked against `compat` and `limits`: sizes and counts before any
    /// allocation, the trailer CRC, then every field's consistency. Foreign: another format, version, store or
    /// geometry. Corrupt: anything else that does not add up.
    pub fn parse(gpa: Allocator, raw: []const u8, compat: Digest, limits: Limits) ParseError!Manifest {
        if (raw.len < header_bytes + trailer_bytes or raw.len > limits.maxManifest()) return error.Corrupt;
        var r: Cursor = .{ .buf = @constCast(raw) };
        if (!std.mem.eql(u8, r.take(8), &magic)) return error.Foreign;
        const ver = r.get(u32);
        if ((ver != version and ver != text_version) or r.get(u32) != header_bytes) return error.Foreign;
        if (!std.mem.eql(u8, r.take(32), &compat)) return error.Foreign;
        var m: Manifest = .{ .key = r.take(32)[0..32].*, .at = 0, .block_tokens = 0, .lineage = .{ .root = @splat(0) }, .ids_bytes = 0, .ids_crc = 0, .small_bytes = 0, .small_crc = 0 };
        m.at = r.get(u32);
        m.block_tokens = r.get(u32);
        m.flags = @bitCast(r.get(u32));
        const n_media = r.get(u32);
        const n_blocks = r.get(u32);
        const n_sections = r.get(u32);
        const n_forks = r.get(u32);
        const layout = r.get(u32);
        m.owner_epoch = r.get(u32);
        m.group = r.get(u64);
        m.created_ns = r.get(i64);
        m.lineage.root = r.take(16)[0..16].*;
        const has_part = r.get(u32);
        const part = r.get(u32);
        const has_origin = r.get(u32);
        const origin_at = r.get(u32);
        const origin_key = r.take(32)[0..32].*;
        m.ids_bytes = r.get(u64);
        m.ids_crc = r.get(u32);
        m.small_crc = r.get(u32);
        m.small_bytes = r.get(u64);
        if (m.block_tokens != limits.block_tokens) return error.Foreign;
        m.owner_layout = std.enums.fromInt(OwnerLayout, layout) orelse return error.Foreign;
        if (m.flags._ != 0 or has_part > 1 or has_origin > 1) return error.Corrupt;
        if (m.at == 0 or m.at > limits.max_tokens) return error.Corrupt;
        if (n_blocks != std.math.divCeil(u32, m.at, m.block_tokens) catch unreachable) return error.Corrupt;
        if (n_media > limits.max_media or n_sections > limits.max_sections or n_forks > limits.max_forks) return error.Corrupt;
        // version 2 points are text points (no build wrote one with media; its ids name no scheme: foreign), version 3
        // points hold pictures and their raw ids
        if (ver == text_version and n_media > 0) return error.Foreign;
        if (ver == version and (n_media == 0 or !m.flags.has_raw)) return error.Corrupt;
        const ids_want = @as(u64, m.at) * 4 * (if (m.flags.has_raw) @as(u64, 2) else 1);
        if (m.ids_bytes != ids_want or m.small_bytes > limits.max_small_bytes) return error.Corrupt;
        const total = header_bytes + @as(u64, n_forks) * fork_bytes + @as(u64, n_media) * (if (ver == version) @as(u64, media_bytes) else media_v2_bytes) +
            @as(u64, n_blocks) * block_ref_bytes + @as(u64, n_sections) * section_bytes + trailer_bytes;
        if (raw.len != total) return error.Corrupt;
        const body = raw.len - trailer_bytes;
        if (std.mem.readInt(u32, raw[body..][0..4], .little) != crc32(raw[0..body])) return error.Corrupt;
        if (std.mem.readInt(u32, raw[body + 4 ..][0..4], .little) != 0) return error.Corrupt;
        if (has_part == 1) m.lineage.part = part;
        if (has_origin == 1) m.origin = .{ .key = origin_key, .at = origin_at };
        r.at = header_bytes;

        const forks = try gpa.alloc(Fork, n_forks);
        errdefer gpa.free(forks);
        for (forks) |*f| {
            f.name = r.take(16)[0..16].*;
            f.at = r.get(u32);
            if (f.at > limits.max_tokens) return error.Corrupt;
        }
        const media = try gpa.alloc(Media, n_media);
        errdefer gpa.free(media);
        var media_end: u64 = 0;
        for (media) |*x| {
            x.start = r.get(u32);
            x.len = r.get(u32);
            const kind = r.get(u32);
            const scheme = r.get(u32);
            x.digest = r.take(32)[0..32].*;
            if (kind != media_picture or scheme != picture_scheme) return error.Foreign;
            // in order, not overlapping, starting inside the point (one may run past its cut), inside the window
            if (x.len == 0 or x.start < media_end or x.start >= m.at or @as(u64, x.start) + x.len > limits.max_tokens) return error.Corrupt;
            media_end = @as(u64, x.start) + x.len;
        }
        // (a part-less point in an owner-sharded store, or a foreign block size where none is kept: damaged)
        const full = limits.fullBlockBytes(m.lineage.part) orelse return error.Corrupt;
        if (full > limits.block_bytes) return error.Corrupt;
        const blocks = try gpa.alloc(BlockRef, n_blocks);
        errdefer gpa.free(blocks);
        for (blocks, 0..) |*b, i| {
            b.name = r.take(32)[0..32].*;
            b.tokens = r.get(u32);
            b.crc = r.get(u32);
            b.bytes = r.get(u64);
            b.sha = r.take(32)[0..32].*;
            const last = i + 1 == n_blocks;
            const want = if (!last) m.block_tokens else m.at - @as(u32, @intCast(i)) * m.block_tokens;
            if (b.tokens != want) return error.Corrupt;
            if (b.tokens == m.block_tokens) {
                if (b.bytes != full) return error.Corrupt;
            } else if (b.bytes == 0 or b.bytes > limits.block_bytes) return error.Corrupt;
        }
        const sections = try gpa.alloc(Section, n_sections);
        errdefer gpa.free(sections);
        var end: u64 = 0;
        for (sections) |*s| {
            s.name.bytes = r.take(32)[0..32].*;
            s.offset = r.get(u64);
            s.len = r.get(u64);
            s.crc = r.get(u32);
            const nl = r.get(u32);
            if (nl == 0 or nl > 32) return error.Corrupt;
            s.name.len = @intCast(nl);
            for (s.name.bytes[nl..]) |c| if (c != 0) return error.Corrupt;
            const stop = std.math.add(u64, s.offset, s.len) catch return error.Corrupt;
            if (s.offset < end or stop > m.small_bytes) return error.Corrupt;
            end = stop;
        }
        m.lineage.forks = forks;
        m.media = media;
        m.blocks = blocks;
        m.sections = sections;
        return m;
    }
};

/// A bounded little-endian cursor over a buffer whose size the caller checked first.
const Cursor = struct {
    buf: []u8,
    at: usize = 0,

    fn bytes(c: *Cursor, b: []const u8) void {
        @memcpy(c.buf[c.at..][0..b.len], b);
        c.at += b.len;
    }

    fn int(c: *Cursor, comptime T: type, v: T) void {
        std.mem.writeInt(T, c.buf[c.at..][0..@sizeOf(T)], v, .little);
        c.at += @sizeOf(T);
    }

    fn take(c: *Cursor, n: usize) []const u8 {
        defer c.at += n;
        return c.buf[c.at..][0..n];
    }

    fn get(c: *Cursor, comptime T: type) T {
        defer c.at += @sizeOf(T);
        return std.mem.readInt(T, c.buf[c.at..][0..@sizeOf(T)], .little);
    }
};

/// A point's `.ids` file: its keyed ids, then its raw ids when they differ (pictures), little-endian u32.
pub fn encodeIds(gpa: Allocator, keyed: []const u32, raw: ?[]const u32) ![]u8 {
    const n = keyed.len * 4 * (if (raw != null) @as(usize, 2) else 1);
    const out = try gpa.alloc(u8, n);
    @memcpy(out[0 .. keyed.len * 4], idsBytes(keyed));
    if (raw) |r| {
        std.debug.assert(r.len == keyed.len);
        @memcpy(out[keyed.len * 4 ..], idsBytes(r));
    }
    return out;
}

// -- tests ---------------------------------------------------------------------------------------------------------
const testing = std.testing;

fn testNames(lineage: Lineage, ids: []const u32, out: []Span) !void {
    var ch: std.ArrayList(Digest) = .empty;
    defer ch.deinit(testing.allocator);
    try chain(testing.allocator, ids, 16, &ch);
    blockNames(lineage, ch.items, ids, 16, out);
}

fn testIds(gpa: Allocator, n: usize, seed: u64) ![]u32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const ids = try gpa.alloc(u32, n);
    for (ids) |*x| x.* = prng.random().intRangeAtMost(u32, 1, 150000);
    return ids;
}

test "block names: shared prefixes of one computation share names, forks diverge at the cut's block" {
    const gpa = testing.allocator;
    const ids = try testIds(gpa, 3 * 16 + 5, 1);
    defer gpa.free(ids);
    const root: Lineage = .{ .root = @splat(7) };
    var a: [4]Span = undefined;
    try testNames(root, ids, &a);
    var b: [3]Span = undefined;
    try testNames(root, ids[0..48], &b);
    for (0..3) |i| try testing.expectEqualSlices(u8, &a[i].name, &b[i].name);
    try testing.expectEqual(@as(u32, 48), a[3].start);
    try testing.expectEqual(@as(u32, 53), a[3].end);
    // a tail's name covers its length: the same ids one shorter is another name
    var c: [4]Span = undefined;
    try testNames(root, ids[0..52], &c);
    try testing.expect(!std.mem.eql(u8, &a[3].name, &c[3].name));
    // a fork at 32: blocks before it keep their names, the block holding it and every later one are new
    var forked = try root.fork(gpa, @splat(9), 32);
    defer forked.deinit(gpa);
    var f: [4]Span = undefined;
    try testNames(forked, ids, &f);
    try testing.expectEqualSlices(u8, &a[0].name, &f[0].name);
    try testing.expectEqualSlices(u8, &a[1].name, &f[1].name);
    try testing.expect(!std.mem.eql(u8, &a[2].name, &f[2].name));
    try testing.expect(!std.mem.eql(u8, &a[3].name, &f[3].name));
    // another computation of the same ids shares nothing by name
    var o: [4]Span = undefined;
    try testNames(.{ .root = @splat(8) }, ids, &o);
    for (0..4) |i| try testing.expect(!std.mem.eql(u8, &a[i].name, &o[i].name));
}

test "full block sizes: a replicated store's every point block_bytes; an owner-sharded one's by its part" {
    const rep: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    for ([_]?u32{ null, 0, 3 }) |p| try std.testing.expectEqual(@as(?u64, 4096), rep.fullBlockBytes(p));
    var sh = rep;
    sh.foreign_block_bytes = 1024;
    sh.own_part = 1;
    try std.testing.expectEqual(@as(?u64, 4096), sh.fullBlockBytes(1));
    try std.testing.expectEqual(@as(?u64, 1024), sh.fullBlockBytes(0));
    try std.testing.expectEqual(@as(?u64, null), sh.fullBlockBytes(null));
}

test "part salt: replaced, never stacked; forks untouched" {
    const gpa = testing.allocator;
    const ids = try testIds(gpa, 32, 2);
    defer gpa.free(ids);
    var l = try (Lineage{ .root = @splat(1) }).fork(gpa, @splat(2), 16);
    defer l.deinit(gpa);
    const p1 = l.inPart(1);
    const p1again = p1.inPart(1);
    const p0 = p1.inPart(0);
    try testing.expect(p1.eql(p1again));
    try testing.expect(!p1.eql(p0));
    try testing.expect(p0.eql(l.inPart(0)));
    var x: [2]Span = undefined;
    var y: [2]Span = undefined;
    var z: [2]Span = undefined;
    try testNames(p1, ids, &x);
    try testNames(p1again, ids, &y);
    try testNames(l, ids, &z);
    try testing.expectEqualSlices(u8, &x[0].name, &y[0].name);
    try testing.expect(!std.mem.eql(u8, &x[0].name, &z[0].name)); // an unsharded store's names are unchanged
}

test "chain and point key" {
    const gpa = testing.allocator;
    const ids = try testIds(gpa, 40, 3);
    defer gpa.free(ids);
    var c1: std.ArrayList(Digest) = .empty;
    defer c1.deinit(gpa);
    try chain(gpa, ids, 16, &c1);
    try testing.expectEqual(@as(usize, 2), c1.items.len);
    var c2: std.ArrayList(Digest) = .empty;
    defer c2.deinit(gpa);
    try chain(gpa, ids[0..32], 16, &c2);
    try testing.expectEqualSlices(u8, &c1.items[1], &c2.items[1]);
    const m: Media = .{ .start = 3, .len = 4, .digest = @splat(5) };
    const k1 = pointKey(ids, 16, &.{});
    const k2 = pointKey(ids, 16, &.{m});
    try testing.expect(!std.mem.eql(u8, &k1, &k2));
    var m2 = m;
    m2.digest[0] = 6;
    try testing.expect(!std.mem.eql(u8, &k2, &pointKey(ids, 16, &.{m2})));
    try testing.expectEqualSlices(u8, &k2, &pointKeyChained(c1.items, ids, 16, &.{m}));
    try testing.expectEqualSlices(u8, &k1, &pointKeyChained(c1.items, ids, 16, &.{}));
    // a tail's ids count: one id changed after the last whole block is another key
    var changed = try gpa.dupe(u32, ids);
    defer gpa.free(changed);
    changed[39] +%= 1;
    try testing.expect(!std.mem.eql(u8, &k1, &pointKey(changed, 16, &.{})));
    changed[39] -%= 1;
    changed[3] +%= 1;
    try testing.expect(!std.mem.eql(u8, &k1, &pointKey(changed, 16, &.{})));
    try testing.expect(!std.mem.eql(u8, &k1, &pointKey(ids[0..39], 16, &.{})));
}

fn sampleManifest(gpa: Allocator, l: Limits) !Manifest {
    const blocks = try gpa.alloc(BlockRef, 3);
    for (blocks, 0..) |*b, i| b.* = .{ .name = @splat(@intCast(i)), .tokens = if (i < 2) 16 else 5, .bytes = if (i < 2) l.block_bytes else 100, .crc = 7, .sha = @splat(3) };
    const sections = try gpa.alloc(Section, 2);
    sections[0] = .{ .name = try Name.of("ring"), .offset = 0, .len = 64, .crc = 1 };
    sections[1] = .{ .name = try Name.of("raw"), .offset = 64, .len = 32, .crc = 2 };
    const media = try gpa.alloc(Media, 1);
    media[0] = .{ .start = 4, .len = 3, .digest = @splat(9) };
    const forks = try gpa.alloc(Fork, 1);
    forks[0] = .{ .name = @splat(4), .at = 16 };
    return .{
        .key = @splat(1), .at = 37, .block_tokens = 16, .flags = .{ .shared = true, .has_raw = true }, .group = 99, .created_ns = 12345,
        .lineage = .{ .root = @splat(2), .part = 1, .forks = forks }, .origin = .{ .key = @splat(8), .at = 32 },
        .ids_bytes = 37 * 8, .ids_crc = 5, .small_bytes = 96, .small_crc = 6, .media = media, .blocks = blocks, .sections = sections,
    };
}

test "manifest round trip and bounded parse" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    var m = try sampleManifest(gpa, l);
    defer m.deinit(gpa);
    const compat: Digest = @splat(0xab);
    const raw = try m.encode(gpa, compat);
    defer gpa.free(raw);
    var back = try Manifest.parse(gpa, raw, compat, l);
    defer back.deinit(gpa);
    try testing.expectEqual(m.at, back.at);
    try testing.expectEqual(m.group, back.group);
    try testing.expect(back.flags.shared and back.flags.has_raw);
    try testing.expect(back.lineage.eql(m.lineage));
    try testing.expectEqual(@as(u32, 32), back.origin.?.at);
    try testing.expectEqualStrings("raw", back.sections[1].name.slice());
    try testing.expect(back.media[0].eql(m.media[0]));
    try testing.expectEqual(@as(u64, 100), back.blocks[2].bytes);

    // another store, another geometry: foreign; any flipped byte: corrupt (the trailer CRC), never a crash
    try testing.expectError(error.Foreign, Manifest.parse(gpa, raw, @splat(0xac), l));
    var l2 = l;
    l2.block_tokens = 32;
    try testing.expectError(error.Foreign, Manifest.parse(gpa, raw, compat, l2));
    const bad = try gpa.dupe(u8, raw);
    defer gpa.free(bad);
    for (0..bad.len) |i| {
        bad[i] ^= 0x41;
        if (Manifest.parse(gpa, bad, compat, l)) |*ok| {
            var x = ok.*;
            x.deinit(gpa);
            return error.TestUnexpectedResult;
        } else |_| {}
        bad[i] ^= 0x41;
    }
    // truncation at every length is refused before allocation
    for (0..raw.len) |n| try testing.expect(std.meta.isError(Manifest.parse(gpa, raw[0..n], compat, l)));
    // a count past the limits is refused even when the bytes would add up
    var l3 = l;
    l3.max_media = 0;
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, raw, compat, l3));
    // a full block of another size is damage
    var l4 = l;
    l4.block_bytes = 8192;
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, raw, compat, l4));
}

test "a huge count in a forged header with a valid CRC is refused before allocating" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    var m = try sampleManifest(gpa, l);
    defer m.deinit(gpa);
    const compat: Digest = @splat(1);
    const raw = try m.encode(gpa, compat);
    defer gpa.free(raw);
    // n_media at offset 8+4+4+32+32+4+4+4 = 92
    std.mem.writeInt(u32, raw[92..96], 0xffff_fff0, .little);
    const body = raw.len - trailer_bytes;
    std.mem.writeInt(u32, raw[body..][0..4], crc32(raw[0..body]), .little);
    var failing = testing.FailingAllocator.init(gpa, .{ .fail_index = 0 });
    try testing.expectError(error.Corrupt, Manifest.parse(failing.allocator(), raw, compat, l));
}

// -- pictures (version 3) --------------------------------------------------------------------------------------------

/// `sampleManifest` without media: a text point (version 2).
fn textManifest(gpa: Allocator, l: Limits) !Manifest {
    var m = try sampleManifest(gpa, l);
    gpa.free(m.media);
    m.media = &.{};
    m.flags.has_raw = false;
    m.ids_bytes = 37 * 4;
    return m;
}

test "manifest versions: a text point is written and read as version 2, a point with pictures as version 3" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    const compat: Digest = @splat(0x5a);
    // text: version 2, 40-byte media entries never written, the bytes those of the format before version 3
    var t = try textManifest(gpa, l);
    defer t.deinit(gpa);
    const traw = try t.encode(gpa, compat);
    defer gpa.free(traw);
    try testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, traw[8..12], .little));
    var tback = try Manifest.parse(gpa, traw, compat, l);
    tback.deinit(gpa);
    // pictures: version 3, 48-byte entries (kind, scheme)
    var p = try sampleManifest(gpa, l);
    defer p.deinit(gpa);
    const praw = try p.encode(gpa, compat);
    defer gpa.free(praw);
    try testing.expectEqual(@as(u32, 3), std.mem.readInt(u32, praw[8..12], .little));
    try testing.expectEqual(traw.len + media_bytes, praw.len);
    var pback = try Manifest.parse(gpa, praw, compat, l);
    defer pback.deinit(gpa);
    try testing.expect(pback.media[0].eql(p.media[0]));
    // the compat digest names the namespace, not the manifest version: version 3 keeps a store's text points
    try testing.expectEqual(@as(u32, 2), namespace_version);

    const Fix = struct {
        fn crc(raw: []u8) void {
            const body = raw.len - trailer_bytes;
            std.mem.writeInt(u32, raw[body..][0..4], crc32(raw[0..body]), .little);
        }
    };
    const bad = try gpa.dupe(u8, praw);
    defer gpa.free(bad);
    // a version 2 manifest with media (the scheme of its ids unknown): foreign, even with its CRC made good
    std.mem.writeInt(u32, bad[8..12], 2, .little);
    Fix.crc(bad);
    try testing.expectError(error.Foreign, Manifest.parse(gpa, bad, compat, l));
    // a version 3 manifest without media, or without raw ids: corrupt
    const t3 = try gpa.dupe(u8, traw);
    defer gpa.free(t3);
    std.mem.writeInt(u32, t3[8..12], 3, .little);
    Fix.crc(t3);
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, t3, compat, l));
    @memcpy(bad, praw);
    const flags_at = 8 + 4 + 4 + 32 + 32 + 4 + 4;
    var fl: Flags = @bitCast(std.mem.readInt(u32, bad[flags_at..][0..4], .little));
    fl.has_raw = false;
    std.mem.writeInt(u32, bad[flags_at..][0..4], @bitCast(fl), .little);
    Fix.crc(bad);
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, bad, compat, l));
    // another kind or scheme: foreign; a span out of order or past the window: corrupt
    const media_at = header_bytes + p.lineage.forks.len * fork_bytes;
    @memcpy(bad, praw);
    std.mem.writeInt(u32, bad[media_at + 8 ..][0..4], 2, .little);
    Fix.crc(bad);
    try testing.expectError(error.Foreign, Manifest.parse(gpa, bad, compat, l));
    @memcpy(bad, praw);
    std.mem.writeInt(u32, bad[media_at + 12 ..][0..4], 2, .little);
    Fix.crc(bad);
    try testing.expectError(error.Foreign, Manifest.parse(gpa, bad, compat, l));
    @memcpy(bad, praw);
    std.mem.writeInt(u32, bad[media_at + 4 ..][0..4], 2000, .little);
    Fix.crc(bad);
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, bad, compat, l));
    @memcpy(bad, praw);
    std.mem.writeInt(u32, bad[media_at..][0..4], 37, .little); // at the cut, not before it
    Fix.crc(bad);
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, bad, compat, l));
}

test "manifest with pictures: two overlapping spans are refused, a span straddling the cut is not" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    const compat: Digest = @splat(0x5b);
    var m = try sampleManifest(gpa, l);
    defer m.deinit(gpa);
    gpa.free(m.media);
    const media = try gpa.alloc(Media, 2);
    m.media = media;
    media[0] = .{ .start = 4, .len = 3, .digest = @splat(9) };
    media[1] = .{ .start = 30, .len = 20, .digest = @splat(8) }; // [30, 50) runs past the cut at 37
    const raw = try m.encode(gpa, compat);
    defer gpa.free(raw);
    var back = try Manifest.parse(gpa, raw, compat, l);
    back.deinit(gpa);
    media[1].start = 5; // inside the first span
    const raw2 = try m.encode(gpa, compat);
    defer gpa.free(raw2);
    try testing.expectError(error.Corrupt, Manifest.parse(gpa, raw2, compat, l));
}

test "manifest fuzz: random edits of a version 3 manifest never crash and never parse into an inconsistent point" {
    const gpa = testing.allocator;
    const l: Limits = .{ .max_tokens = 1024, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 1 << 20 };
    const compat: Digest = @splat(0x5c);
    var m = try sampleManifest(gpa, l);
    defer m.deinit(gpa);
    const raw = try m.encode(gpa, compat);
    defer gpa.free(raw);
    var prng = std.Random.DefaultPrng.init(0x7069637475726573);
    const rnd = prng.random();
    const buf = try gpa.alloc(u8, raw.len + 64);
    defer gpa.free(buf);
    var parsed: usize = 0;
    for (0..20_000) |_| {
        // a copy, a few random bytes changed (in the media entries half the time), a random length, the CRC made good
        // half the time so the field checks run, not only the trailer's
        var n = raw.len;
        @memcpy(buf[0..n], raw);
        const media_at = header_bytes + m.lineage.forks.len * fork_bytes;
        for (0..rnd.intRangeAtMost(usize, 1, 4)) |_| {
            const i = if (rnd.boolean()) media_at + rnd.uintLessThan(usize, media_bytes) else rnd.uintLessThan(usize, n);
            buf[i] = rnd.int(u8);
        }
        if (rnd.uintLessThan(u8, 8) == 0) n = rnd.uintLessThan(usize, raw.len + 64);
        if (n >= trailer_bytes and rnd.boolean()) {
            const body = n - trailer_bytes;
            std.mem.writeInt(u32, buf[body..][0..4], crc32(buf[0..body]), .little);
            std.mem.writeInt(u32, buf[body + 4 ..][0..4], 0, .little);
        }
        var got = Manifest.parse(gpa, buf[0..n], compat, l) catch continue;
        defer got.deinit(gpa);
        parsed += 1;
        // whatever parses is consistent: spans in order inside the point and the window, version and media agree
        var end: u64 = 0;
        for (got.media) |x| {
            try testing.expect(x.len > 0 and x.start >= end and x.start < got.at and @as(u64, x.start) + x.len <= l.max_tokens);
            end = @as(u64, x.start) + x.len;
        }
        try testing.expect(got.media.len == 0 or got.flags.has_raw);
        try testing.expect(got.at > 0 and got.at <= l.max_tokens and got.blocks.len == std.math.divCeil(u32, got.at, 16) catch unreachable);
    }
    try testing.expect(parsed > 0); // (the CRC-made-good edits of unchecked bytes, e.g. a digest, still parse)
}

test "media ids: keyed ids derive from the digest, raw ids hold model ids there and equal the keyed ones elsewhere" {
    const d: Digest = @splat(3);
    var keyed = [_]u32{ 5, 6, 0, 0, 0, 7, 0, 0 };
    var raw = [_]u32{ 5, 6, 99, 99, 99, 7, 99, 99 };
    for (2..5) |i| keyed[i] = pictureId(&d, @intCast(i - 2));
    for (6..8) |i| keyed[i] = pictureId(&d, @intCast(i - 6)); // a second picture straddling the end
    const m = [_]Media{ .{ .start = 2, .len = 3, .digest = d }, .{ .start = 6, .len = 10, .digest = d } };
    try testing.expect(mediaIds(&keyed, &raw, &m));
    try testing.expect(!mediaIds(&keyed, null, &m)); // pictures need their raw ids
    try testing.expect(!mediaIds(&keyed, &raw, m[0..1])); // a picture the media does not name
    try testing.expect(!mediaIds(&keyed, &raw, &.{ m[1], m[0] })); // out of order
    try testing.expect(!mediaIds(&keyed, &raw, &.{ m[0], .{ .start = 4, .len = 1, .digest = d } })); // overlapping
    var r2 = raw;
    r2[0] = 4;
    try testing.expect(!mediaIds(&keyed, &r2, &m)); // raw differs outside a picture
    r2 = raw;
    r2[3] = keyed[3];
    try testing.expect(!mediaIds(&keyed, &r2, &m)); // a picture id where a model id belongs
    var k2 = keyed;
    k2[3] +%= 1;
    try testing.expect(!mediaIds(&k2, &raw, &m)); // an id not made from the digest
    try testing.expect(mediaIds(&.{ 1, 2, 3 }, null, &.{})); // text
    try testing.expect(!mediaIds(&.{ 1, picture_ids, 3 }, null, &.{})); // a picture id without its media
}
