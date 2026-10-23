//! Port of wojo's GLM 0088 (PR #78) and 0110 (PR #124; BadAd84's block-tier design) spill tier: the native,
//! family-neutral, GPU-free spill v2 store. One store per rank directory; see docs/glm-port/spill-store.md.

const std = @import("std");
pub const format = @import("format.zig");
pub const fsio = @import("fsio.zig");
pub const slots = @import("slots.zig");
pub const store = @import("store.zig");
pub const mutants = @import("mutants.zig");

pub const Store = store.Store;
pub const Config = store.Config;
pub const Keep = store.Keep;
pub const KeepResult = store.KeepResult;
pub const Source = store.Source;
pub const StageReq = store.StageReq;
pub const Ticket = store.Ticket;
pub const Completion = store.Completion;
pub const Query = store.Query;
pub const Hit = store.Hit;
pub const ReadId = store.ReadId;
pub const JobId = store.JobId;
pub const ReadyBlock = store.ReadyBlock;
pub const ReadDone = store.ReadDone;
pub const Counters = store.Counters;
pub const Lineage = format.Lineage;
pub const Media = format.Media;
pub const Digest = format.Digest;
pub const Limits = format.Limits;
pub const OwnerLayout = format.OwnerLayout;
pub const Staging = slots.Staging;
pub const Slot = slots.Slot;

test {
    _ = format;
    _ = fsio;
    _ = slots;
    std.testing.refAllDecls(store);
    std.testing.refAllDecls(Store);
}

test {
    _ = @import("store_test.zig");
}
