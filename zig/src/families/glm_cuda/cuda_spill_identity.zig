//! Spill v2's compatibility fingerprint on the GLM pair. Port of wojo's GLM 0088/0110 spill signature (`build_id`,
//! `weights_id`, the planes and switches its store's namespace hashes) in the shape of the DeepSeek port's
//! fingerprint.zig and rank_identity.zig: their std-only codec as is, GLM's fields and inputs. One SHA-256 digest per
//! named field over what can change the bytes a kept state holds or the messages the ranks exchange; a common digest
//! over the fields every rank must share and a per-rank digest over each rank's own. Once the settings text agreed, the
//! ranks exchange every field's digest over the control link (`agree`): a start whose ranks differ fails naming the
//! rank and the field on every rank, and each rank's store opens under the agreement narrowed to it
//! (`Agreement.storeCommon`), never under a digest one rank made alone (v1's build key, cuda_native.buildKey, never left
//! its rank, and its `catch 0` filed every build it could not hash under key 0).
//!
//! Every field must be given (Builder.finish refuses a missing one): a value a configuration lacks is given as an
//! explicit "off"/"none", never left out, so a later input can never silently join an old digest. A field may be built
//! from several parts (each length-prefixed, so ("ab","c") and ("a","bc") differ). Files are hashed by content
//! (putFile), the captured kernel archive as a sorted manifest of paths, sizes and content digests (putTree: every cubin,
//! not aot.json alone), the weights by their recorded revisions and their safetensors headers (the tens of GB of tensors
//! are not read). Never paths, inodes or mtimes (Python's weights_id used them; a copied, identical checkpoint must
//! agree), nor the host CPU model (a GB10 build resolves Cortex-X925 or A725 per binary: DeepSeek 3308e00).
const std = @import("std");
const builtin = @import("builtin");
const core = @import("core");
const link = @import("cuda_link.zig");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_weights.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Digest = [32]u8;

/// Bumped when the digest construction itself changes (every stored point and every peer then disagrees).
pub const version: u32 = 1;

/// Whether every rank must hold the same value (`common`) or each rank its own (`rank`, exchanged in rank order).
pub const Scope = enum(u8) { common = 1, rank = 2 };

/// The fingerprint's fields, by stable id (ids are on the wire and in digests: never renumber, only append).
pub const Field = enum(u16) {
    format = 1, // the kept-state format (spill_state.format_version) and core/spill's on-disk format
    wire = 2, // the ranks' spill messages (cuda_spill_wire.version)
    engine = 3, // TF_GLM_ENGINE_ID: the engine's commit or image digest
    patches = 4, // TF_GLM_RECIPE_PATCHES: the recipe's zig/patches series digest
    build = 5, // the compiler's facts (Builder.init) and TF_GLM_BUILD_FLAGS
    kernels = 6, // the captured kernel archive (TENSORFOLD_CUDA_KERNELS: aot.json and every cubin)
    model = 7, // the checkpoint: TF_GLM_MODEL_ID and TF_GLM_MODEL_REVISION, config.json and its index by content
    drafter = 8, // DFlash2: TF_GLM_DFLASH2_ID and _REVISION, its config.json by content; "mtp", or "off"
    tokenizer = 9, // tokenizer.json
    quant = 10, // the cache and weight formats the bits depend on (KV=fp8, DENSE=q4, the head, the drafter's quant)
    gpu = 11, // compute capability and CUDA driver version
    grid = 12, // the kept-state grid (64 tokens) and a block's tokens (2,048)
    layout = 13, // spill_state.Layout.digest(): the buffer, block and sections a point holds
    tp_split = 14, // the TP world, every rank's share (heads, widths, vocab) and the index ring's rows
    owner_layout = 15, // spill_state.Layout.ownerText: replicated, or owner_sharded with its home rule's version and parts
    features = 16, // the switches that change state or protocol (spill v2, its minimum, KV sharding, drafts, streams)
    template = 17, // rank 0 renders prompts with its chat template; a follower holds its own (or none)
    rank_split = 18, // this rank and its share
    rank_weights = 19, // this rank's copy of the weights: each safetensors file's name, size and header
};

pub const n_fields = @typeInfo(Field).@"enum".field_names.len;

pub fn scope(f: Field) Scope {
    return switch (f) {
        // the chat template is rank 0's alone (it renders prompts; a follower takes ids): a kept state depends on the
        // ids, not on how they were made, so the ranks need not agree on it (DeepSeek 9983b2f)
        .template, .rank_split, .rank_weights => .rank,
        else => .common,
    };
}

pub fn name(f: Field) []const u8 {
    return @tagName(f);
}

/// A field by its wire id, or null for an id this build does not know (a peer's newer field).
pub fn fieldOf(id: u16) ?Field {
    return std.enums.fromInt(Field, id);
}

fn index(f: Field) usize {
    inline for (@typeInfo(Field).@"enum".field_values, 0..) |v, i| {
        if (@backingInt(f) == v) return i;
    }
    unreachable;
}

/// The fields in id order (the order every digest walks them in).
pub const all: [n_fields]Field = blk: {
    var out: [n_fields]Field = undefined;
    for (@typeInfo(Field).@"enum".field_values, 0..) |v, i| out[i] = @fromBackingInt(@intCast(v));
    break :blk out;
};

/// Builds the per-field digests. Feed each field with part/int/putFile/putTree, then finish.
pub const Builder = struct {
    h: [n_fields]?Sha256 = @splat(null),

    /// A builder with `build` already holding the compiler's own facts (Zig version, optimize mode, target triple; not
    /// the CPU model): the caller adds the flags it chose.
    pub fn init() Builder {
        var b: Builder = .{};
        b.part(.build, builtin.zig_version_string);
        b.part(.build, @tagName(builtin.mode));
        b.part(.build, @tagName(builtin.cpu.arch));
        // not builtin.cpu.model: a GB10 has Cortex-X925 and A725 cores and a native build resolves the model per
        // compiler process (DeepSeek's server and follower binaries of one image differed), while no GPU arithmetic or
        // kept byte depends on the host CPU model
        b.part(.build, @tagName(builtin.os.tag));
        b.part(.build, @tagName(builtin.abi));
        return b;
    }

    fn hasher(b: *Builder, f: Field) *Sha256 {
        const i = index(f);
        if (b.h[i] == null) {
            var s = Sha256.init(.{});
            s.update("tf-glm-fingerprint\x00");
            var head: [6]u8 = undefined;
            std.mem.writeInt(u32, head[0..4], version, .little);
            std.mem.writeInt(u16, head[4..6], @backingInt(f), .little);
            s.update(&head);
            b.h[i] = s;
        }
        return &b.h[i].?;
    }

    /// One part of field `f`: its length (u64) then its bytes.
    pub fn part(b: *Builder, f: Field, bytes: []const u8) void {
        const s = b.hasher(f);
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, bytes.len, .little);
        s.update(&len);
        s.update(bytes);
    }

    /// An integer part (little-endian, its width fixed by T: u32 3 and u64 3 differ).
    pub fn int(b: *Builder, f: Field, comptime T: type, v: T) void {
        var buf: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &buf, v, .little);
        b.part(f, &buf);
    }

    /// A list of integers as one part (its count, then each).
    pub fn ints(b: *Builder, f: Field, comptime T: type, vs: []const T) void {
        b.int(f, u64, vs.len);
        for (vs) |v| b.int(f, T, v);
    }

    /// A file's content digest as one part (streamed; its name and location are not part of it).
    pub fn putFile(b: *Builder, f: Field, io: std.Io, dir: std.Io.Dir, sub_path: []const u8) !void {
        b.part(f, &try fileDigest(io, dir, sub_path));
    }

    /// A folder's manifest as one part: every regular file below it, sorted by relative path ('/' separated), each its
    /// path, size and content digest. `keep` (null: all) picks the files by relative path. An empty match counts (an
    /// empty folder differs from a populated one, and both from "none").
    pub fn putTree(b: *Builder, f: Field, gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, sub_path: []const u8, keep: ?*const fn ([]const u8) bool) !void {
        b.part(f, &try treeDigest(gpa, io, dir, sub_path, keep));
    }

    /// The digests; error.MissingField names (in `missing`) the first field never given.
    pub fn finish(b: *const Builder, missing: ?*Field) !Fingerprint {
        var fp: Fingerprint = .{};
        for (all, 0..) |f, i| {
            var s = b.h[i] orelse {
                if (missing) |m| m.* = f;
                return error.MissingField;
            };
            s.final(&fp.digests[i]);
        }
        return fp;
    }
};

/// A file's SHA-256 over its content.
pub fn fileDigest(io: std.Io, dir: std.Io.Dir, sub_path: []const u8) !Digest {
    var file = try dir.openFile(io, sub_path, .{});
    defer file.close(io);
    var s = Sha256.init(.{});
    var buf: [64 << 10]u8 = undefined;
    var at: u64 = 0;
    while (true) {
        const n = try file.readPositional(io, &.{&buf}, at);
        if (n == 0) break;
        s.update(buf[0..n]);
        at += n;
    }
    var out: Digest = undefined;
    s.final(&out);
    return out;
}

/// A folder's manifest digest (Builder.putTree).
pub fn treeDigest(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, sub_path: []const u8, keep: ?*const fn ([]const u8) bool) !Digest {
    var root = try dir.openDir(io, sub_path, .{ .iterate = true });
    defer root.close(io);
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    var walk = try root.walk(gpa);
    defer walk.deinit();
    while (try walk.next(io)) |e| {
        if (e.kind != .file) continue;
        const rel = try gpa.dupe(u8, e.path);
        // one separator on every host, so a manifest is the same wherever it is built
        for (rel) |*ch| if (ch.* == std.fs.path.sep) {
            ch.* = '/';
        };
        if (keep) |k| if (!k(rel)) {
            gpa.free(rel);
            continue;
        };
        names.append(gpa, rel) catch |err| {
            gpa.free(rel);
            return err;
        };
    }
    std.mem.sort([]u8, names.items, {}, struct {
        fn lt(_: void, x: []u8, y: []u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    var s = Sha256.init(.{});
    s.update("tf-glm-tree\x00");
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, names.items.len, .little);
    s.update(&len);
    for (names.items) |n| {
        std.mem.writeInt(u64, &len, n.len, .little);
        s.update(&len);
        s.update(n);
        const st = try root.statFile(io, n, .{});
        std.mem.writeInt(u64, &len, st.size, .little);
        s.update(&len);
        s.update(&try fileDigest(io, root, n));
    }
    var out: Digest = undefined;
    s.final(&out);
    return out;
}

/// Every field's digest, in id order.
pub const Fingerprint = struct {
    digests: [n_fields]Digest = undefined,

    pub fn get(fp: *const Fingerprint, f: Field) Digest {
        return fp.digests[index(f)];
    }

    /// The digest every rank must share: H(version, then each common field's id and digest in id order).
    pub fn common(fp: *const Fingerprint) Digest {
        return fp.over(.common);
    }

    /// This rank's own digest: the same over the rank-scoped fields.
    pub fn rankDigest(fp: *const Fingerprint) Digest {
        return fp.over(.rank);
    }

    fn over(fp: *const Fingerprint, sc: Scope) Digest {
        var s = Sha256.init(.{});
        s.update(if (sc == .common) "tf-glm-common\x00" else "tf-glm-rank\x00");
        var v: [4]u8 = undefined;
        std.mem.writeInt(u32, &v, version, .little);
        s.update(&v);
        for (all, 0..) |f, i| {
            if (scope(f) != sc) continue;
            var id: [2]u8 = undefined;
            std.mem.writeInt(u16, &id, @backingInt(f), .little);
            s.update(&id);
            s.update(&fp.digests[i]);
        }
        var out: Digest = undefined;
        s.final(&out);
        return out;
    }

    /// What the ranks exchange at start: every field's id, scope and digest, and the two summaries.
    pub fn exchange(fp: *const Fingerprint) Exchange {
        var x: Exchange = .{ .common = fp.common(), .rank = fp.rankDigest(), .n = n_fields };
        for (all, 0..) |f, i| x.fields[i] = .{ .id = @backingInt(f), .scope = scope(f), .digest = fp.digests[i] };
        return x;
    }
};

/// The most fields a peer's exchange may list (a newer peer may know more fields than this build; the bound keeps the
/// parse fixed).
pub const max_fields = 64;

pub const Entry = struct { id: u16, scope: Scope, digest: Digest };

/// A rank's fingerprint as exchanged at start. On the link: common [32], rank [32], n u16, then n entries of id u16,
/// scope u8 (1 common, 2 rank) and digest [32], ids strictly increasing, nothing after them.
pub const Exchange = struct {
    common: Digest,
    rank: Digest,
    n: u16 = 0,
    fields: [max_fields]Entry = undefined,

    pub const max_bytes = 32 + 32 + 2 + max_fields * 35;

    pub fn entries(x: *const Exchange) []const Entry {
        return x.fields[0..x.n];
    }

    fn find(x: *const Exchange, id: u16) ?Entry {
        for (x.entries()) |e| if (e.id == id) return e;
        return null;
    }

    /// The first common field (by id) on which `x` and `y` disagree, a field one lists and the other lacks, or a field
    /// the two scope differently; null when every common field agrees (then their common digests agree too, unless the
    /// construction differs, which `differs` reports as `wire`). Rank fields are each rank's own.
    pub fn firstDifference(x: *const Exchange, y: *const Exchange) ?u16 {
        var best: ?u16 = null;
        for ([_]*const Exchange{ x, y }, [_]*const Exchange{ y, x }) |a, b| {
            for (a.entries()) |e| {
                const o = b.find(e.id) orelse {
                    best = minId(best, e.id);
                    continue;
                };
                if (o.scope != e.scope) {
                    best = minId(best, e.id);
                    continue;
                }
                if (e.scope == .common and !std.mem.eql(u8, &e.digest, &o.digest)) best = minId(best, e.id);
            }
        }
        return best;
    }

    /// Whether the two disagree at all on what must be common: a differing field (its id), or, with every listed field
    /// equal, differing common digests (a construction difference, reported as `wire`).
    pub fn differs(x: *const Exchange, y: *const Exchange) ?u16 {
        if (x.firstDifference(y)) |id| return id;
        if (!std.mem.eql(u8, &x.common, &y.common)) return @backingInt(Field.wire);
        return null;
    }

    pub fn encode(x: *const Exchange, out: *[max_bytes]u8) []u8 {
        @memcpy(out[0..32], &x.common);
        @memcpy(out[32..64], &x.rank);
        std.mem.writeInt(u16, out[64..66], x.n, .little);
        var at: usize = 66;
        for (x.entries()) |e| {
            std.mem.writeInt(u16, out[at..][0..2], e.id, .little);
            out[at + 2] = @backingInt(e.scope);
            @memcpy(out[at + 3 ..][0..32], &e.digest);
            at += 35;
        }
        return out[0..at];
    }

    /// An exchange's bytes; refused: a short or long frame, more fields than `max_fields`, ids out of order, an unknown
    /// scope.
    pub fn decode(bytes: []const u8) !Exchange {
        if (bytes.len < 66) return error.BadExchange;
        var x: Exchange = .{ .common = bytes[0..32].*, .rank = bytes[32..64].*, .n = std.mem.readInt(u16, bytes[64..66], .little) };
        if (x.n > max_fields or bytes.len != 66 + @as(usize, x.n) * 35) return error.BadExchange;
        var last: ?u16 = null;
        for (x.fields[0..x.n], 0..) |*e, i| {
            const at = 66 + i * 35;
            const id = std.mem.readInt(u16, bytes[at..][0..2], .little);
            if (last) |l| if (id <= l) return error.BadExchange;
            last = id;
            const sc = std.enums.fromInt(Scope, bytes[at + 2]) orelse return error.BadExchange;
            e.* = .{ .id = id, .scope = sc, .digest = bytes[at + 3 ..][0..32].* };
        }
        return x;
    }
};

fn minId(a: ?u16, b: u16) u16 {
    return if (a) |x| @min(x, b) else b;
}

/// A field id in words ("engine", or "field 41" for one this build does not know).
pub fn describe(id: u16, buf: []u8) []const u8 {
    if (fieldOf(id)) |f| return name(f);
    return std.fmt.bufPrint(buf, "field {d}", .{id}) catch "field ?";
}

/// The most ranks a run links (cuda_comm.max_world).
pub const max_world = 4;

/// What the ranks agreed at start: the common digest and every rank's own, in rank order.
pub const Agreement = struct {
    common: Digest,
    world: u16,
    ranks: [max_world]Digest = @splat(@splat(0)),

    /// The digest of the whole pair (common, then the per-rank digests in order).
    pub fn pair(a: *const Agreement) Digest {
        var s = Sha256.init(.{});
        s.update("tf-glm-pair\x00");
        s.update(&a.common);
        var w: [2]u8 = undefined;
        std.mem.writeInt(u16, &w, a.world, .little);
        s.update(&w);
        for (a.ranks[0..a.world]) |r| s.update(&r);
        var out: Digest = undefined;
        s.final(&out);
        return out;
    }

    /// The `common` a rank's spill store is opened with (core/spill Config.compat via format.compatDigest): the pair's
    /// agreement narrowed to that rank, so a store is only ever read by the rank that wrote it, under the same common
    /// configuration and its own weights and split.
    pub fn storeCommon(a: *const Agreement, rank: u16) Digest {
        std.debug.assert(rank < a.world);
        var s = Sha256.init(.{});
        s.update("tf-glm-store\x00");
        s.update(&a.common);
        var w: [4]u8 = undefined;
        std.mem.writeInt(u16, w[0..2], rank, .little);
        std.mem.writeInt(u16, w[2..4], a.world, .little);
        s.update(&w);
        s.update(&a.ranks[rank]);
        var out: Digest = undefined;
        s.final(&out);
        return out;
    }
};

// -- GLM's inputs ---------------------------------------------------------------------------------------------------

/// The recipe's identity, from the environment: each rank reads its own (the recipe exports the same on every rank).
/// With spill v2 the engine, the patch series and the checkpoint's revision are required (and the drafter's with
/// DFlash2 drafts): an unidentified engine or set of weights must never read or agree on stored state (a safetensors
/// header alone does not tell two quants of one shape apart).
pub const Env = struct {
    engine: []const u8, // TF_GLM_ENGINE_ID: the engine's commit or the image's content digest
    patches: []const u8, // TF_GLM_RECIPE_PATCHES: the applied zig/patches series' digest
    build_flags: []const u8 = "", // TF_GLM_BUILD_FLAGS: the recipe's chosen build flags
    model_id: []const u8 = "", // TF_GLM_MODEL_ID: the checkpoint's repository (MODEL_ID; ABLIT=1 serves another)
    model_revision: []const u8, // TF_GLM_MODEL_REVISION: its pinned revision (MODEL_REVISION)
    dflash2_id: []const u8 = "", // TF_GLM_DFLASH2_ID
    dflash2_revision: []const u8 = "", // TF_GLM_DFLASH2_REVISION (required with DFlash2 drafts)

    pub fn read(get: *const fn ([:0]const u8) ?[]const u8, dflash2: bool) !Env {
        const e: Env = .{
            .engine = nonEmpty(get("TF_GLM_ENGINE_ID")) orelse return error.NoEngineId,
            .patches = nonEmpty(get("TF_GLM_RECIPE_PATCHES")) orelse return error.NoRecipePatches,
            .build_flags = get("TF_GLM_BUILD_FLAGS") orelse "",
            .model_id = get("TF_GLM_MODEL_ID") orelse "",
            .model_revision = nonEmpty(get("TF_GLM_MODEL_REVISION")) orelse return error.NoModelRevision,
            .dflash2_id = get("TF_GLM_DFLASH2_ID") orelse "",
            .dflash2_revision = get("TF_GLM_DFLASH2_REVISION") orelse "",
        };
        if (dflash2 and e.dflash2_revision.len == 0) return error.NoDrafterRevision;
        return e;
    }
};

fn nonEmpty(v: ?[]const u8) ?[]const u8 {
    const s = v orelse return null;
    return if (s.len == 0) null else s;
}

/// The words for a refusal the identity explains (cuda_native.refusal).
pub fn why(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.NoEngineId => "TF_GLM_SPILL_V2=1: TF_GLM_ENGINE_ID is not set (the engine's commit or image digest): spill v2 never stores state an unidentified engine made",
        error.NoRecipePatches => "TF_GLM_SPILL_V2=1: TF_GLM_RECIPE_PATCHES is not set (the zig/patches series' digest): spill v2 never stores state an unidentified engine made",
        error.NoModelRevision => "TF_GLM_SPILL_V2=1: TF_GLM_MODEL_REVISION is not set (the checkpoint's pinned revision): spill v2 never stores state unidentified weights made",
        error.NoDrafterRevision => "TF_GLM_SPILL_V2=1: TF_GLM_DFLASH2_REVISION is not set (DFlash2's pinned revision): spill v2 never stores state an unidentified drafter made",
        error.SpillIdentityMismatch => "TF_GLM_SPILL_V2=1: the ranks' spill identities differ (each rank's log names the field)",
        else => null,
    };
}

/// The drafter a run uses (cuda_mtp_choice).
pub const Drafter = enum { off, dflash2, mtp };

/// What a GLM rank's fingerprint is made from (cuda_native.open: its options, the opened engine and its context).
pub const Rank = struct {
    dir: []const u8, // the checkpoint's folder
    kernels: []const u8, // the captured kernel archive (TENSORFOLD_CUDA_KERNELS or the installed one)
    drafter: Drafter,
    draft_dir: ?[]const u8 = null, // DFlash2's folder
    draft_quant: []const u8 = "minmax", // TF_GLM_DRAFT_QUANT (the MTP head's)
    world: u32,
    rank: u32,
    ring: usize, // the index ring's rows (Caches.ring)
    layout: Digest, // spill_state.Layout.digest()
    owner_layout: []const u8 = "replicated", // spill_state.Layout.ownerText (TF_GLM_KV_SHARD: owner_sharded;...)
    capability: u32,
    driver: i32,
    state_format: u32, // spill_state.format_version
    wire: u16, // cuda_spill_wire.version
    features: []const u8, // featureString
    env: Env,
};

/// The switches that change state or protocol, canonical ("name=value;" in this order): every rank must hold the same.
pub const Features = struct {
    spill_min_tokens: u32,
    kv_shard: bool = false,
    drafts: bool,
    streams: u32,
    prefill_rows: u32,
};

pub fn featureString(buf: []u8, f: Features) ![]const u8 {
    return std.fmt.bufPrint(buf, "spill=2;spill_min={d};kv_shard={d};drafts={d};streams={d};prefill_rows={d};", .{ f.spill_min_tokens, @intFromBool(f.kv_shard), @intFromBool(f.drafts), f.streams, f.prefill_rows });
}

/// Every rank's share of the model (cuda_geometry), as text: the TP split every rank computes alike.
fn splitText(buf: []u8, world: u32) ![]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    for (0..world) |r| try shareText(&w, try geometry.Geometry.init(world, r));
    return w.buffered();
}

fn shareText(w: *std.Io.Writer, g: geometry.Geometry) !void {
    try w.print("{d}:heads={d},lin={d},dense={d},moe={d},shared={d},vocab={d}+{d};", .{ g.rank, g.heads, g.lin_heads, g.dense_width, g.moe_width, g.shared_width, g.vocab_start, g.vocab });
}

/// This rank's copy of the weights, read from its own folders: every safetensors file the checkpoint's index names
/// (sorted), each its name, size and header (the JSON of names, dtypes, shapes and offsets: what the loader reads first,
/// cuda_weights.readHeader), then DFlash2's model.safetensors the same way. The tensors themselves are not read; a
/// recorded revision (Env) is what tells two checkpoints of one layout apart.
pub fn weightsDigest(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, draft_dir: ?[]const u8) !Digest {
    var s = Sha256.init(.{});
    s.update("tf-glm-weights\x00");
    const index_path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(index_path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const map = (if (parsed.value == .object) parsed.value.object.get("weight_map") else null) orelse return error.BadIndex;
    if (map != .object) return error.BadIndex;
    var files: std.ArrayList([]const u8) = .empty;
    defer files.deinit(gpa);
    var it = map.object.iterator();
    while (it.next()) |kv| {
        if (kv.value_ptr.* != .string) return error.BadIndex;
        const f = kv.value_ptr.string;
        for (files.items) |x| {
            if (std.mem.eql(u8, x, f)) break;
        } else try files.append(gpa, f);
    }
    std.mem.sort([]const u8, files.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    for (files.items) |f| try headerPart(gpa, io, &s, dir, f);
    if (draft_dir) |dd| try headerPart(gpa, io, &s, dd, "model.safetensors") else s.update("drafter:none");
    var out: Digest = undefined;
    s.final(&out);
    return out;
}

fn headerPart(gpa: std.mem.Allocator, io: std.Io, s: *Sha256, dir: []const u8, file: []const u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, file });
    defer gpa.free(path);
    const h = try weights.readHeader(gpa, io, path);
    defer gpa.free(h.json);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, file.len, .little);
    s.update(&len);
    s.update(file);
    std.mem.writeInt(u64, &len, h.file_len, .little);
    s.update(&len);
    var d: Digest = undefined;
    Sha256.hash(h.json, &d, .{});
    s.update(&d);
}

fn exists(io: std.Io, dir: std.Io.Dir, path: []const u8) bool {
    dir.access(io, path, .{}) catch return false;
    return true;
}

/// A file's name and content as two parts when it exists, "none" otherwise.
fn fileOr(b: *Builder, f: Field, io: std.Io, dir: std.Io.Dir, path: []const u8) !void {
    if (!exists(io, dir, path)) return b.part(f, "none");
    b.part(f, path); // the name tells a template's source apart (a .jinja and a tokenizer_config.json of equal bytes)
    try b.putFile(f, io, dir, path);
}

/// This rank's fingerprint.
pub fn build(gpa: std.mem.Allocator, io: std.Io, r: Rank) !Fingerprint {
    if (r.world < 1 or r.world > max_world or r.rank >= r.world) return error.BadRank;
    const cwd = std.Io.Dir.cwd();
    var model = try cwd.openDir(io, r.dir, .{});
    defer model.close(io);
    var b = Builder.init();
    b.int(.format, u32, r.state_format);
    b.int(.format, u32, core.spill.format.version);
    b.int(.wire, u16, r.wire);
    b.part(.engine, r.env.engine);
    b.part(.patches, r.env.patches);
    b.part(.build, r.env.build_flags);
    try b.putTree(.kernels, gpa, io, cwd, r.kernels, null);
    b.part(.model, r.env.model_id);
    b.part(.model, r.env.model_revision);
    try fileOr(&b, .model, io, model, "config.json");
    try fileOr(&b, .model, io, model, "model.safetensors.index.json");
    b.part(.drafter, @tagName(r.drafter));
    switch (r.drafter) {
        .off => {},
        .mtp => b.part(.drafter, r.draft_quant),
        .dflash2 => {
            var dd = try cwd.openDir(io, r.draft_dir orelse return error.MissingDraftDirectory, .{});
            defer dd.close(io);
            b.part(.drafter, r.env.dflash2_id);
            b.part(.drafter, r.env.dflash2_revision);
            try fileOr(&b, .drafter, io, dd, "config.json");
        },
    }
    try fileOr(&b, .tokenizer, io, model, "tokenizer.json");
    // the engine's own formats: the latent cache is FP8 (528-byte rows), the dense weights 4-bit, the head FP8
    b.part(.quant, "kv=fp8;dense=q4;head=f8;");
    b.part(.quant, if (r.drafter == .mtp) r.draft_quant else "none");
    b.int(.gpu, u32, r.capability);
    b.int(.gpu, i32, r.driver);
    b.int(.grid, u32, 64);
    b.int(.grid, u32, geometry.Geometry.extent);
    b.part(.layout, &r.layout);
    var sbuf: [512]u8 = undefined;
    b.int(.tp_split, u16, @intCast(r.world));
    b.part(.tp_split, try splitText(&sbuf, r.world));
    b.int(.tp_split, u64, r.ring);
    b.part(.owner_layout, r.owner_layout);
    b.part(.features, r.features);
    // the template rank 0 renders with (server/hf_text.zig's order)
    if (exists(io, model, "chat_template.jinja")) try fileOr(&b, .template, io, model, "chat_template.jinja") else if (exists(io, model, "chat_template.json")) try fileOr(&b, .template, io, model, "chat_template.json") else try fileOr(&b, .template, io, model, "tokenizer_config.json");
    var rbuf: [128]u8 = undefined;
    var rw: std.Io.Writer = .fixed(&rbuf);
    try shareText(&rw, try geometry.Geometry.init(r.world, r.rank));
    b.int(.rank_split, u16, @intCast(r.rank));
    b.part(.rank_split, rw.buffered());
    b.part(.rank_weights, &try weightsDigest(gpa, io, r.dir, if (r.drafter == .dflash2) r.draft_dir else null));
    return b.finish(null);
}

// -- the agreement over the control link --------------------------------------------------------------------------

/// The first rank and field a start's fingerprints differ on (`agree`'s error.SpillIdentityMismatch).
pub const Mismatch = struct { rank: u16 = 0, field: u16 = 0 };

const verdict_bytes = 1 + 2 + 2 + 2 + 32 + 32 * max_world;

/// Every rank's fingerprint over the control link once the settings text agreed (cuda_link.agreeRanks' pattern): rank 0
/// sends its exchange to every follower, each follower answers with its own, rank 0 compares them in rank order and
/// sends one verdict to all: the agreement (the common digest and every rank's own), or the first rank and field that
/// differ. Every rank then holds the same Agreement, or gets error.SpillIdentityMismatch with `mismatch` set (the
/// caller logs it: `mismatchLine`). Rank 0's link is the fan-out of acceptRanks, a follower's its one socket.
pub fn agree(l: link.Link, gpa: std.mem.Allocator, rank: u32, world: u32, fp: *const Fingerprint, mismatch: *Mismatch) !Agreement {
    if (world < 2 or world > max_world or rank >= world) return error.BadRank;
    const mine = fp.exchange();
    var buf: [Exchange.max_bytes]u8 = undefined;
    var v: [verdict_bytes]u8 = @splat(0);
    if (rank == 0) {
        try l.send(.fields, mine.encode(&buf));
        var a: Agreement = .{ .common = mine.common, .world = @intCast(world) };
        a.ranks[0] = mine.rank;
        var bad: ?Mismatch = null;
        for (0..l.followers()) |i| {
            const bytes = try l.at(i).expect(gpa, .fields);
            defer gpa.free(bytes);
            const theirs = Exchange.decode(bytes) catch {
                if (bad == null) bad = .{ .rank = @intCast(i + 1), .field = @backingInt(Field.wire) };
                continue;
            };
            if (bad == null) if (mine.differs(&theirs)) |id| {
                bad = .{ .rank = @intCast(i + 1), .field = id };
            };
            if (i + 1 < max_world) a.ranks[i + 1] = theirs.rank;
        }
        v[0] = @intFromBool(bad == null);
        const m = bad orelse Mismatch{};
        std.mem.writeInt(u16, v[1..3], m.rank, .little);
        std.mem.writeInt(u16, v[3..5], m.field, .little);
        std.mem.writeInt(u16, v[5..7], a.world, .little);
        @memcpy(v[7..39], &a.common);
        for (a.ranks, 0..) |d, i| @memcpy(v[39 + 32 * i ..][0..32], &d);
        try l.send(.fields_verdict, &v);
        if (bad) |m2| return refuse(mismatch, m2);
        return a;
    }
    const first = try l.expect(gpa, .fields);
    defer gpa.free(first);
    const theirs = Exchange.decode(first) catch null;
    try l.send(.fields, mine.encode(&buf));
    const got = try l.expect(gpa, .fields_verdict);
    defer gpa.free(got);
    if (got.len != verdict_bytes) return error.UnexpectedFrame;
    if (got[0] != 1) return refuse(mismatch, .{ .rank = std.mem.readInt(u16, got[1..3], .little), .field = std.mem.readInt(u16, got[3..5], .little) });
    var a: Agreement = .{ .common = got[7..39].*, .world = std.mem.readInt(u16, got[5..7], .little) };
    for (&a.ranks, 0..) |*d, i| d.* = got[39 + 32 * i ..][0..32].*;
    // what rank 0 agreed must be this rank's own: the common digest and its own digest at its place
    const own = theirs != null and theirs.?.differs(&mine) == null;
    if (!own or a.world != world or !std.mem.eql(u8, &a.common, &mine.common) or !std.mem.eql(u8, &a.ranks[rank], &mine.rank))
        return refuse(mismatch, .{ .rank = @intCast(rank), .field = @backingInt(Field.wire) });
    return a;
}

fn refuse(out: *Mismatch, m: Mismatch) error{SpillIdentityMismatch} {
    out.* = m;
    return error.SpillIdentityMismatch;
}

/// A mismatch in words, for the start's log on every rank.
pub fn mismatchLine(buf: []u8, m: Mismatch) []const u8 {
    var fb: [16]u8 = undefined;
    return std.fmt.bufPrint(buf, "spill v2: rank {d}'s {s} differs from rank 0's (the ranks do not run one build, the same weights or the same settings)", .{ m.rank, describe(m.field, &fb) }) catch "spill v2: the ranks' identities differ";
}

// -- tests ------------------------------------------------------------------------------------------------------------

const testing = std.testing;

/// A copy of `b` (fed every field but `skip`) with each field given one part.
fn filled(skip: ?Field) Builder {
    var b: Builder = .{};
    for (all) |f| if (f != skip) b.int(f, u16, @backingInt(f));
    return b;
}

test "every field changes its own digest only, and the scope decides which summary moves" {
    const base = try filled(null).finish(null);
    const bx = base.exchange();
    for (all) |f| {
        var b = filled(f);
        b.part(f, "another value");
        const fp = try b.finish(null);
        for (all) |g| try testing.expect(std.mem.eql(u8, &fp.get(g), &base.get(g)) == (g != f));
        try testing.expect(std.mem.eql(u8, &fp.common(), &base.common()) == (scope(f) == .rank));
        try testing.expect(std.mem.eql(u8, &fp.rankDigest(), &base.rankDigest()) == (scope(f) == .common));
        const fx = fp.exchange();
        const want: ?u16 = if (scope(f) == .common) @backingInt(f) else null;
        try testing.expectEqual(want, bx.differs(&fx));
        try testing.expectEqual(want, fx.differs(&bx));
    }
    // the three per-rank fields: the template (rank 0 renders), the split and the weights
    try testing.expectEqual(Scope.rank, scope(.template));
    try testing.expectEqual(Scope.rank, scope(.rank_split));
    try testing.expectEqual(Scope.rank, scope(.rank_weights));
    try testing.expectEqual(Scope.common, scope(.layout));
}

test "a field never given is refused by name, parts are length-delimited, the construction is pinned" {
    var missing: Field = .format;
    try testing.expectError(error.MissingField, filled(.kernels).finish(&missing));
    try testing.expectEqual(Field.kernels, missing);
    var x = filled(.engine);
    var y = filled(.engine);
    x.part(.engine, "ab");
    x.part(.engine, "c");
    y.part(.engine, "a");
    y.part(.engine, "bc");
    try testing.expect(!std.mem.eql(u8, &(try x.finish(null)).get(.engine), &(try y.finish(null)).get(.engine)));
    var p = filled(.grid);
    var q = filled(.grid);
    p.int(.grid, u32, 64);
    q.int(.grid, u64, 64);
    try testing.expect(!std.mem.eql(u8, &(try p.finish(null)).get(.grid), &(try q.finish(null)).get(.grid)));
    // a change to how digests are built must bump `version` (stored points and peers then disagree on purpose)
    var hex: [64]u8 = undefined;
    _ = try std.fmt.bufPrint(&hex, "{x}", .{&(try filled(null).finish(null)).common()});
    try testing.expectEqualStrings(pinned_common, &hex);
}

const pinned_common = "4cc8176050afb54cb1b0ae36b6485ddfaa273beb86a36b3f2ddffe379653076a";

test "an exchange survives its bytes; a cut, a long one, ids out of order or an unknown scope are refused" {
    var b = filled(null);
    const x = (try b.finish(null)).exchange();
    var buf: [Exchange.max_bytes]u8 = undefined;
    const bytes = x.encode(&buf);
    try testing.expectEqual(@as(usize, 66 + n_fields * 35), bytes.len);
    const y = try Exchange.decode(bytes);
    try testing.expectEqual(@as(?u16, null), x.differs(&y));
    try testing.expectEqualSlices(u8, &x.rank, &y.rank);
    for (0..bytes.len) |cut| try testing.expectError(error.BadExchange, Exchange.decode(bytes[0..cut]));
    var long: [Exchange.max_bytes + 1]u8 = undefined;
    @memcpy(long[0..bytes.len], bytes);
    long[bytes.len] = 0;
    try testing.expectError(error.BadExchange, Exchange.decode(long[0 .. bytes.len + 1]));
    var swapped = buf;
    std.mem.writeInt(u16, swapped[66 + 35 ..][0..2], 1, .little); // the second id equal to the first
    try testing.expectError(error.BadExchange, Exchange.decode(swapped[0..bytes.len]));
    var scoped = buf;
    scoped[66 + 2] = 9;
    try testing.expectError(error.BadExchange, Exchange.decode(scoped[0..bytes.len]));
    // a peer that lacks a field (an older build) or knows one more (a newer): the field is named
    var fewer = x;
    fewer.n -= 1;
    try testing.expectEqual(@as(?u16, @backingInt(all[n_fields - 1])), x.differs(&fewer));
    var buf2: [16]u8 = undefined;
    try testing.expectEqualStrings("kernels", describe(6, &buf2));
    try testing.expectEqualStrings("field 999", describe(999, &buf2));
}

test "the agreement narrows to each rank's store; rank 0's store does not depend on rank 1's weights" {
    var a: Agreement = .{ .common = @splat(1), .world = 2 };
    a.ranks[0] = @splat(2);
    a.ranks[1] = @splat(3);
    const s0 = a.storeCommon(0);
    const s1 = a.storeCommon(1);
    try testing.expect(!std.mem.eql(u8, &s0, &s1));
    var b = a;
    b.ranks[1] = @splat(4);
    try testing.expectEqualSlices(u8, &s0, &b.storeCommon(0));
    try testing.expect(!std.mem.eql(u8, &s1, &b.storeCommon(1)));
    try testing.expect(!std.mem.eql(u8, &a.pair(), &b.pair()));
}

/// A GLM checkpoint, drafter and kernel archive in `d` (small files of the real formats).
fn writeInstall(io: std.Io, d: std.Io.Dir) !void {
    for ([_][]const u8{ "model", "draft", "kernels/cubins" }) |p| try d.createDirPath(io, p);
    try d.writeFile(io, .{ .sub_path = "model/config.json", .data = "{\"model_type\": \"glm5_next\"}" });
    try d.writeFile(io, .{ .sub_path = "model/model.safetensors.index.json", .data = "{\"metadata\": {}, \"weight_map\": {\"a.weight\": \"model-00002.safetensors\", \"b.weight\": \"model-00001.safetensors\", \"c.weight\": \"model-00001.safetensors\"}}" });
    try writeSafetensors(io, d, "model/model-00001.safetensors", "{\"b.weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]},\"c.weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[4,8]}}", "abcdefgh");
    try writeSafetensors(io, d, "model/model-00002.safetensors", "{\"a.weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}", "ijkl");
    try d.writeFile(io, .{ .sub_path = "model/tokenizer.json", .data = "tok" });
    try d.writeFile(io, .{ .sub_path = "model/chat_template.jinja", .data = "{{ messages }}" });
    try d.writeFile(io, .{ .sub_path = "draft/config.json", .data = "{\"block_size\": 16}" });
    try writeSafetensors(io, d, "draft/model.safetensors", "{\"fc.weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}", "mnop");
    try d.writeFile(io, .{ .sub_path = "kernels/aot.json", .data = "{\"kernels\": []}" });
    try d.writeFile(io, .{ .sub_path = "kernels/cubins/0a.cubin", .data = "cubin one" });
    try d.writeFile(io, .{ .sub_path = "kernels/cubins/0b.cubin", .data = "cubin two" });
}

fn writeSafetensors(io: std.Io, d: std.Io.Dir, path: []const u8, header: []const u8, body: []const u8) !void {
    var buf: [512]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], header.len, .little);
    @memcpy(buf[8..][0..header.len], header);
    @memcpy(buf[8 + header.len ..][0..body.len], body);
    try d.writeFile(io, .{ .sub_path = path, .data = buf[0 .. 8 + header.len + body.len] });
}

const test_env: Env = .{ .engine = "a1b2c3d", .patches = "sha256:0014-0020", .model_id = "Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold", .model_revision = "4f2e9a1", .dflash2_id = "z-lab/GLM-5.3-Flash-DFlash2", .dflash2_revision = "9c8b7a6" };

fn testRank(root: []const u8, a: std.mem.Allocator, rank: u32) !Rank {
    var fbuf: [256]u8 = undefined;
    return .{
        .dir = try std.fs.path.join(a, &.{ root, "model" }),
        .kernels = try std.fs.path.join(a, &.{ root, "kernels" }),
        .drafter = .dflash2,
        .draft_dir = try std.fs.path.join(a, &.{ root, "draft" }),
        .world = 2,
        .rank = rank,
        .ring = 2112,
        .layout = @splat(9),
        .capability = 121,
        .driver = 13000,
        .state_format = 1,
        .wire = 1,
        .features = try a.dupe(u8, try featureString(&fbuf, .{ .spill_min_tokens = 8192, .drafts = true, .streams = 8, .prefill_rows = 2048 })),
        .env = test_env,
    };
}

test "two ranks of one install agree on the common digest; the kernel archive's every cubin, the revisions and the switches are named; each rank's weights and split are its own" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try writeInstall(io, tmp.dir);
    var root_buf: [1024]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(io, &root_buf)];
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const f0 = try build(gpa, io, try testRank(root, a, 0));
    const f1 = try build(gpa, io, try testRank(root, a, 1));
    try testing.expectEqualSlices(u8, &f0.common(), &f1.common());
    try testing.expect(!std.mem.eql(u8, &f0.rankDigest(), &f1.rankDigest()));
    try testing.expectEqual(@as(?u16, null), f0.exchange().differs(&f1.exchange()));
    const named = struct {
        fn of(x: Fingerprint, y: Fingerprint) ?Field {
            return fieldOf(x.exchange().differs(&y.exchange()) orelse return null);
        }
    }.of;
    // a cubin changed, aot.json not: the archive is hashed whole
    try tmp.dir.writeFile(io, .{ .sub_path = "kernels/cubins/0b.cubin", .data = "cubin 2" });
    try testing.expectEqual(@as(?Field, .kernels), named(f0, try build(gpa, io, try testRank(root, a, 0))));
    try tmp.dir.writeFile(io, .{ .sub_path = "kernels/cubins/0b.cubin", .data = "cubin two" });
    // the revisions, the engine, the switches, the ring, the layout, the GPU
    var r = try testRank(root, a, 0);
    r.env.model_revision = "4f2e9a2";
    try testing.expectEqual(@as(?Field, .model), named(f0, try build(gpa, io, r)));
    r = try testRank(root, a, 0);
    r.env.model_id = "Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit";
    try testing.expectEqual(@as(?Field, .model), named(f0, try build(gpa, io, r)));
    r = try testRank(root, a, 0);
    r.env.dflash2_revision = "9c8b7a7";
    try testing.expectEqual(@as(?Field, .drafter), named(f0, try build(gpa, io, r)));
    r = try testRank(root, a, 0);
    r.env.engine = "a1b2c3e";
    try testing.expectEqual(@as(?Field, .engine), named(f0, try build(gpa, io, r)));
    r = try testRank(root, a, 0);
    r.ring = 2048;
    try testing.expectEqual(@as(?Field, .tp_split), named(f0, try build(gpa, io, r)));
    r = try testRank(root, a, 0);
    r.layout = @splat(8);
    try testing.expectEqual(@as(?Field, .layout), named(f0, try build(gpa, io, r)));
    r = try testRank(root, a, 0);
    r.owner_layout = "owner_sharded;version=1;parts=2;"; // (TF_GLM_KV_SHARD: a sharded store never meets a replicated one)
    try testing.expectEqual(@as(?Field, .owner_layout), named(f0, try build(gpa, io, r)));
    r = try testRank(root, a, 0);
    r.driver = 13010;
    try testing.expectEqual(@as(?Field, .gpu), named(f0, try build(gpa, io, r)));
    r = try testRank(root, a, 0);
    var fbuf: [256]u8 = undefined;
    r.features = try featureString(&fbuf, .{ .spill_min_tokens = 4096, .drafts = true, .streams = 8, .prefill_rows = 2048 });
    try testing.expectEqual(@as(?Field, .features), named(f0, try build(gpa, io, r)));
    r = try testRank(root, a, 0);
    r.drafter = .off;
    try testing.expectEqual(@as(?Field, .drafter), named(f0, try build(gpa, io, r)));
    // rank 1's copy of one weight file holds another header (a stale copy on its disk): its own digest moves, the
    // common one does not (it goes into rank 1's store namespace alone)
    try writeSafetensors(io, tmp.dir, "model/model-00002.safetensors", "{\"a.weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]},\"x\":{}}", "ijkl");
    const g1 = try build(gpa, io, try testRank(root, a, 1));
    try testing.expectEqualSlices(u8, &f1.common(), &g1.common());
    try testing.expect(!std.mem.eql(u8, &f1.get(.rank_weights), &g1.get(.rank_weights)));
    try testing.expect(!std.mem.eql(u8, &f1.rankDigest(), &g1.rankDigest()));
    // another template on a rank: its own field, not a mismatch
    try tmp.dir.writeFile(io, .{ .sub_path = "model/chat_template.jinja", .data = "{{ other }}" });
    const h1 = try build(gpa, io, try testRank(root, a, 1));
    try testing.expectEqualSlices(u8, &f1.common(), &h1.common());
    try testing.expect(!std.mem.eql(u8, &g1.get(.template), &h1.get(.template)));
    try testing.expectError(error.BadRank, build(gpa, io, blk: {
        var x = try testRank(root, a, 0);
        x.rank = 2;
        break :blk x;
    }));
}

test "the weights are identified by each file's header and size, in name order, never by the tensors' bytes" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try writeInstall(io, tmp.dir);
    var root_buf: [1024]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(io, &root_buf)];
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const model = try std.fs.path.join(a, &.{ root, "model" });
    const draft = try std.fs.path.join(a, &.{ root, "draft" });
    const w0 = try weightsDigest(gpa, io, model, draft);
    // other tensor bytes of the same size: the same identity (a recorded revision is what tells them apart)
    try writeSafetensors(io, tmp.dir, "model/model-00001.safetensors", "{\"b.weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]},\"c.weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[4,8]}}", "ABCDEFGH");
    try testing.expectEqualSlices(u8, &w0, &try weightsDigest(gpa, io, model, draft));
    // another size, another header, no drafter: another identity each
    try writeSafetensors(io, tmp.dir, "model/model-00001.safetensors", "{\"b.weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]},\"c.weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[4,8]}}", "ABCDEFGHI");
    const w1 = try weightsDigest(gpa, io, model, draft);
    try testing.expect(!std.mem.eql(u8, &w0, &w1));
    try testing.expect(!std.mem.eql(u8, &w1, &try weightsDigest(gpa, io, model, null)));
    try tmp.dir.writeFile(io, .{ .sub_path = "model/model-00002.safetensors", .data = &.{ 0xff, 0xff, 0, 0, 0, 0, 0, 0, 1 } });
    try testing.expectError(error.BadSafetensors, weightsDigest(gpa, io, model, draft));
    try tmp.dir.writeFile(io, .{ .sub_path = "model/model.safetensors.index.json", .data = "{\"weight_map\": [1]}" });
    try testing.expectError(error.BadIndex, weightsDigest(gpa, io, model, draft));
}

fn testEnv(n: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, n, "TF_GLM_ENGINE_ID")) return "a1b2c3d";
    if (std.mem.eql(u8, n, "TF_GLM_RECIPE_PATCHES")) return "sha256:0014";
    if (std.mem.eql(u8, n, "TF_GLM_MODEL_REVISION")) return "4f2e9a1";
    if (std.mem.eql(u8, n, "TF_GLM_DFLASH2_REVISION")) return "9c8b7a6";
    return null;
}

fn noDrafterEnv(n: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, n, "TF_GLM_DFLASH2_REVISION")) return null;
    return testEnv(n);
}

fn noModelEnv(n: [:0]const u8) ?[]const u8 {
    if (std.mem.eql(u8, n, "TF_GLM_MODEL_REVISION")) return "";
    return testEnv(n);
}

fn noEnv(_: [:0]const u8) ?[]const u8 {
    return null;
}

test "the recipe's identity is required: the engine, the patches, the checkpoint's revision and, with DFlash2, the drafter's" {
    const e = try Env.read(testEnv, true);
    try testing.expectEqualStrings("a1b2c3d", e.engine);
    try testing.expectEqualStrings("9c8b7a6", e.dflash2_revision);
    try testing.expectError(error.NoEngineId, Env.read(noEnv, false));
    try testing.expectError(error.NoModelRevision, Env.read(noModelEnv, false));
    try testing.expectError(error.NoDrafterRevision, Env.read(noDrafterEnv, true));
    _ = try Env.read(noDrafterEnv, false);
    try testing.expect(std.mem.indexOf(u8, why(error.NoModelRevision).?, "TF_GLM_MODEL_REVISION") != null);
    try testing.expect(why(error.OutOfMemory) == null);
}

/// `world` ranks over socket pairs (rank 0's link the fan-out): each runs `agree` with its own fingerprint.
fn agreeAll(world: u32, fps: []const Fingerprint, out: []?anyerror, got: []Agreement, why_out: []Mismatch) !void {
    const gpa = testing.allocator;
    var pairs: [max_world - 1][2]std.c.fd_t = undefined;
    for (pairs[0 .. world - 1]) |*p| if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, p) != 0) return error.SocketFailed;
    var zero: link.Link = .{ .fd = pairs[0][0] };
    for (pairs[1 .. world - 1], 0..) |p, i| zero.more[i] = p[0];
    const Peer = struct {
        fn run(l: link.Link, r: u32, w: u32, fp: *const Fingerprint, e: *?anyerror, a: *Agreement, m: *Mismatch) void {
            a.* = agree(l, testing.allocator, r, w, fp, m) catch |err| {
                e.* = err;
                return;
            };
        }
    };
    var threads: [max_world - 1]std.Thread = undefined;
    for (1..world) |r| threads[r - 1] = try std.Thread.spawn(.{}, Peer.run, .{ link.Link{ .fd = pairs[r - 1][1] }, @as(u32, @intCast(r)), world, &fps[r], &out[r], &got[r], &why_out[r] });
    got[0] = agree(zero, gpa, 0, world, &fps[0], &why_out[0]) catch |err| blk: {
        out[0] = err;
        break :blk undefined;
    };
    for (threads[0 .. world - 1]) |t| t.join();
    zero.close();
    for (pairs[0 .. world - 1]) |p| _ = std.c.close(p[1]);
}

/// A fingerprint whose rank fields say `rank` and whose `field` holds `value`.
fn rankPrint(rank: u16, field: ?Field, value: []const u8) !Fingerprint {
    var b: Builder = .{};
    for (all) |f| {
        if (scope(f) == .rank) b.int(f, u16, rank) else b.int(f, u16, @backingInt(f));
        if (field == f) b.part(f, value);
    }
    return b.finish(null);
}

test "the ranks agree over the control link at 2, 3 and 4 ranks; a differing field fails every rank naming the rank and the field" {
    for ([_]u32{ 2, 3, 4 }) |world| {
        var fps: [max_world]Fingerprint = undefined;
        for (0..world) |r| fps[r] = try rankPrint(@intCast(r), null, "");
        var errs: [max_world]?anyerror = @splat(null);
        var got: [max_world]Agreement = undefined;
        var whys: [max_world]Mismatch = @splat(.{});
        try agreeAll(world, &fps, &errs, &got, &whys);
        for (0..world) |r| {
            try testing.expect(errs[r] == null);
            try testing.expectEqualSlices(u8, &fps[0].common(), &got[r].common);
            try testing.expectEqual(@as(u16, @intCast(world)), got[r].world);
            for (0..world) |q| try testing.expectEqualSlices(u8, &fps[q].rankDigest(), &got[r].ranks[q]);
            try testing.expectEqualSlices(u8, &got[0].pair(), &got[r].pair());
            try testing.expectEqualSlices(u8, &got[0].storeCommon(@intCast(r)), &got[r].storeCommon(@intCast(r)));
        }
    }
    // rank 2 of 4 built with other kernels, rank 3 with another layout: every rank fails, the first one named
    var fps: [max_world]Fingerprint = undefined;
    fps[0] = try rankPrint(0, null, "");
    fps[1] = try rankPrint(1, null, "");
    fps[2] = try rankPrint(2, .kernels, "other cubins");
    fps[3] = try rankPrint(3, .layout, "other ring");
    var errs: [max_world]?anyerror = @splat(null);
    var got: [max_world]Agreement = undefined;
    var whys: [max_world]Mismatch = @splat(.{});
    try agreeAll(4, &fps, &errs, &got, &whys);
    for (0..4) |r| {
        try testing.expectEqual(@as(?anyerror, error.SpillIdentityMismatch), errs[r]);
        try testing.expectEqual(Mismatch{ .rank = 2, .field = @backingInt(Field.kernels) }, whys[r]);
    }
    var line: [160]u8 = undefined;
    try testing.expectEqualStrings("spill v2: rank 2's kernels differs from rank 0's (the ranks do not run one build, the same weights or the same settings)", mismatchLine(&line, whys[0]));
    // two ranks, rank 1's own template only: they agree
    fps[1] = try rankPrint(1, .template, "another template");
    var errs2: [max_world]?anyerror = @splat(null);
    var whys2: [max_world]Mismatch = @splat(.{});
    try agreeAll(2, &fps, &errs2, &got, &whys2);
    try testing.expect(errs2[0] == null and errs2[1] == null);
    try testing.expectEqualSlices(u8, &fps[1].rankDigest(), &got[0].ranks[1]);
}
