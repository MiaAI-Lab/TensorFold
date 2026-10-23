//! Port of wojo's GLM 0110 `tools/spill_blocks_mutants.sh` (PR #124): each authored fix can be switched back to the
//! code before it, in test builds only, so the store's tests prove they catch its regression. Outside tests `on`
//! is comptime false and every mutant branch compiles away.
const builtin = @import("builtin");

pub const Mutant = enum {
    none,
    // the GLM fixes (spill_blocks_mutants.sh), as they apply to a single-rank, GPU-free store
    tail_staged_later, // the block a keep's end cuts staged later from the source, not captured at the keep
    loadpart_past_upto, // a step copies past the boundary rank 0 sent (LOADPART's upto ignored)
    no_upgrade, // ids kept again as a shared prefix leave their stored point unshared
    upgrade_not_durable, // ... shared in memory only (its manifest not rewritten)
    waiting_not_flagged, // ... a point still waiting to be written does not take the flag
    written_no_rewrite, // ... made shared during its own write: the manifest written unshared stays so
    rewrite_resurrects, // a point dropped during its rewrite gets its manifest back
    rewrite_not_put, // the rewrite overwrites the manifest in place (no temporary name, fsync, rename)
    rewrite_no_dir_fsync, // ... without fsyncing the points directory
    drain_skips_rewrites, // a drain returns before queued metadata rewrites ran
    lost_is_damage, // a missing name is treated as damaged bytes (every link to them dropped)
    lost_names_kept, // a missing name is not forgotten
    damage_not_widened, // damaged bytes drop only the name read, not every name linked to the inode
    small_files_keep_pages, // a point's small files keep their pages after the fsync
    pages_dropped_before_fsync, // ... dropped before the fsync (dirty pages are not dropped)
    hang_no_recycle_in_wait, // a follower's wait for a block does not give back the slots its copies hold
    budget_each_read, // every restore in flight gets the whole copy budget of a pause
    budget_same_first, // every pause starts with the same restore
    read_ahead_charges_preallocate, // preallocated read buffers charged again
    // the native port's own guards
    manifest_not_last, // the manifest written before the point's ids and small state
    scan_keeps_temps, // recovery leaves temporary files
    recover_no_size_check, // recovery accepts a block of the wrong size
    dedup_without_sha, // a link made on a CRC match alone
    guard_cancels_only, // a guard cancels a waiting point instead of staging what fits
    evict_shared_first, // shared prefixes evicted before conversations
    floor_ignored, // the free-disk floor never refuses or evicts
    owner_lock_ignored, // a second owner opens a live store
    keep_unpinned, // a keep makes room before the point holds references to the blocks it reuses
    forget_skips_reading, // forget leaves listed a point that names a dropped block while it is being read
    handover_unreserved, // blocks handed to another write leave their reservation behind
    recover_drops_on_io, // recovery drops a point (and collects its blocks) on a transient error, not only on damage
    // the GLM port's own
    prune_other_families, // open prunes every `native-spill-*` name (the lifted prefix), whatever family made it
};

pub var current: Mutant = .none;

pub inline fn on(m: Mutant) bool {
    return builtin.is_test and current == m;
}
