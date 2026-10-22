//! GLM-5.3-Flash on CUDA in Zig: config, EXL3 weight-loading plan and TP layouts matching the Python
//! glm5_next CUDA engine (families/glm5_next/cuda/weights.py, split.py, tp.py). No GPU upload yet.

pub const config = @import("cuda_config.zig");
pub const layouts = @import("cuda_layouts.zig");
pub const weights = @import("cuda_weights.zig");

test {
    _ = config;
    _ = layouts;
    _ = weights;
    _ = @import("plan_test.zig");
}
