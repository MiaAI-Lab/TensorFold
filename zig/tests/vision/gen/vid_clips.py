# Small clips made with PyAV 18.1 (libx264 / libvpx / native mpeg4), and what decode_clip's loop gets from them.
import hashlib, io, json, os, sys
import numpy as np, av
from fractions import Fraction

OUT = "/fix/vid"
os.makedirs(OUT, exist_ok=True)

def pattern(w, h, t):
    ys, xs = np.mgrid[0:h, 0:w]
    r = (xs * 255 // max(w - 1, 1) + t * 9) % 256
    g = (ys * 255 // max(h - 1, 1) + t * 5) % 256
    b = ((xs + ys) * 3 + t * 17) % 256
    a = np.stack([r, g, b], axis=-1).astype(np.uint8)
    a[(ys // 8 + xs // 8 + t) % 5 == 0] = (255, 255, 255)
    a[h // 3:h // 3 + 3, :] = (255, 0, 0)
    return a

def make(name, fmt, codec, w, h, n, fps, pix_fmt, options=None, color=None, container_opts=None):
    path = os.path.join(OUT, name)
    c = av.open(path, "w", format=fmt, options=container_opts or {})
    s = c.add_stream(codec, rate=Fraction(fps).limit_denominator(100000))
    s.width, s.height, s.pix_fmt = w, h, pix_fmt
    if options: s.options = options
    if color:
        for k, v in color.items(): setattr(s.codec_context, k, v)
    for t in range(n):
        f = av.VideoFrame.from_ndarray(pattern(w, h, t), format="rgb24")
        for p in s.encode(f.reformat(format=pix_fmt)): c.mux(p)
    for p in s.encode(None): c.mux(p)
    c.close()
    return path

def describe(path, tag):
    data = open(path, "rb").read()
    with av.open(io.BytesIO(data), mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        w, h = int(stream.codec_context.width or 0), int(stream.codec_context.height or 0)
        avg = stream.average_rate; gr = stream.guessed_rate
        info = {"file": os.path.basename(path), "width": w, "height": h,
                "avg_rate": [avg.numerator, avg.denominator] if avg else None,
                "guessed_rate": [gr.numerator, gr.denominator] if gr else None,
                "frames": int(stream.frames or 0),
                "duration": stream.duration, "time_base": [stream.time_base.numerator, stream.time_base.denominator],
                "container_duration": container.duration}
        hashes = []; sizes = []
        for n, frame in enumerate(container.decode(stream)):
            arr = frame.to_ndarray(format="rgb24")
            hashes.append(hashlib.sha256(arr.tobytes()).hexdigest()); sizes.append(list(arr.shape))
            if n == 0:
                info["pix_fmt"] = frame.format.name; info["colorspace"] = int(frame.colorspace); info["color_range"] = int(frame.color_range)
        info["frame_hashes"] = hashes
        info["frame_shapes"] = sizes
    return info

cases = [
    ("h264_420_64x48.mp4", "mp4", "libx264", 64, 48, 12, 24, "yuv420p", {"crf": "30", "preset": "veryfast"}, None),
    ("h264_420_30fps_odd_len.mp4", "mp4", "libx264", 48, 32, 7, Fraction(30000, 1001), "yuv420p", {"crf": "35", "preset": "veryfast"}, None),
    ("h264_420_bt709_pc.mp4", "mp4", "libx264", 64, 48, 6, 12, "yuv420p", {"crf": "28", "x264-params": "range=pc:colormatrix=bt709:colorprim=bt709:transfer=bt709"}, None),
    ("h264_420_bt709_tv.mp4", "mp4", "libx264", 64, 48, 6, 12, "yuv420p", {"crf": "28", "x264-params": "range=tv:colormatrix=bt709"}, None),
    ("h264_yuvj420p.mp4", "mp4", "libx264", 64, 48, 6, 12, "yuvj420p", {"crf": "28"}, None),
    ("h264_420_fullrange_709.mp4", "mp4", "libx264", 64, 48, 6, 12, "yuv420p", {"crf": "28", "x264-params": "fullrange=on:colormatrix=bt709"}, None),
    ("h264_420_fullrange_601.mp4", "mp4", "libx264", 48, 32, 6, 12, "yuv420p", {"crf": "28", "x264-params": "fullrange=on:colormatrix=smpte170m"}, None),
    ("h264_444_33x17.mp4", "mp4", "libx264", 34, 18, 5, 10, "yuv444p", {"crf": "25"}, None),
    ("h264_gray.mp4", "mp4", "libx264", 32, 32, 4, 5, "gray", {"crf": "25"}, None),
    ("h264_420_1fps_long.mp4", "mp4", "libx264", 32, 32, 20, 1, "yuv420p", {"crf": "35", "preset": "veryfast"}, None),
    ("h264_420_2frames.mp4", "mp4", "libx264", 32, 32, 2, 24, "yuv420p", {"crf": "30"}, None),
    ("h264_420_bframes.mp4", "mp4", "libx264", 64, 48, 24, 24, "yuv420p", {"crf": "30", "bf": "3", "preset": "veryfast"}, None),
    ("mpeg4_420.mp4", "mp4", "mpeg4", 64, 48, 8, 25, "yuv420p", {"qscale": "5"}, None),
    ("vp9_420_66x34.webm", "webm", "libvpx-vp9", 66, 34, 10, 15, "yuv420p", {"crf": "40", "b": "0", "deadline": "realtime", "cpu-used": "8"}, None),
    ("vp9_420_odd_63x35.webm", "webm", "libvpx-vp9", 63, 35, 6, 10, "yuv420p", {"crf": "40", "b": "0", "deadline": "realtime", "cpu-used": "8"}, None),
    ("vp8_420_48x32.webm", "webm", "libvpx", 48, 32, 8, 20, "yuv420p", {"crf": "30", "b": "1M"}, None),
    ("h264_420_mkv.mkv", "matroska", "libx264", 48, 32, 8, 20, "yuv420p", {"crf": "30"}, None),
    ("h264_420_mov.mov", "mov", "libx264", 48, 32, 8, 20, "yuv420p", {"crf": "30"}, None),
]
infos = []
for name, fmt, codec, w, h, n, fps, pix, opts, color in cases:
    try:
        p = make(name, fmt, codec, w, h, n, fps, pix, opts, color)
        infos.append(describe(p, name))
        print("ok", name, os.path.getsize(p), infos[-1]["pix_fmt"], infos[-1]["colorspace"], infos[-1]["color_range"], infos[-1]["frames"], infos[-1]["duration"], infos[-1]["container_duration"], len(infos[-1]["frame_hashes"]))
    except Exception as e:
        print("skip", name, type(e).__name__, e)
json.dump(infos, open("/fix/vid/expected.json", "w"))
