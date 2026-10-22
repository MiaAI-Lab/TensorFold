//! GLM-5.3-Flash image and video input, the CPU side ("the processor"): decoders, resize, geometry, limits,
//! prompt expansion and request splitting. Bit-exact to the Python recipe (tensorfold.vision.*).
//!
//! Map (all under zig/src/vision/):
//!   errors, limits, geometry   Diag/`error.Input`, GlmVisionLimits + env knobs, smart_resize / content_size / sample_frames
//!   items, expand, split       Item/Picture/Clip + hashes and 62-bit keys, prompt expansion, split_images + MediaMarks
//!   patches                    canvas -> f32 patches (unit-test oracle for the GPU tower)
//!   png, jpeg, webp (+inflate, vp8, vp8l, vp8_tables), raster, convert, images    pixel-exact decoders and `_decode`
//!   resize                     torchvision's uint8 bicubic antialias `fit`
//!   av, clip                   FFmpeg through dlopen, `decode_clip`
//!   fetch                      HTTPS retrieval with the SSRF rules of images_http.py
//!   frontend                   Frontend: picture cache, loadImages / loadVideos / loadSources, expand, finish, Slots

pub const errors = @import("errors.zig");
pub const limits = @import("limits.zig");
pub const geometry = @import("geometry.zig");
pub const items = @import("items.zig");
pub const patches = @import("patches.zig");
pub const expand = @import("expand.zig");
pub const split = @import("split.zig");
pub const images = @import("images.zig");
pub const resize = @import("resize.zig");
pub const frontend = @import("frontend.zig");
pub const av = @import("av.zig");
pub const clip = @import("clip.zig");
pub const fetch = @import("fetch.zig");

test {
    _ = errors;
    _ = limits;
    _ = geometry;
    _ = items;
    _ = patches;
    _ = expand;
    _ = split;
    _ = images;
    _ = resize;
    _ = frontend;
    _ = av;
    _ = clip;
    _ = fetch;
    _ = @import("inflate.zig");
}
