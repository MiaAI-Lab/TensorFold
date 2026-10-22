# Reference implementation of the libav call sequence the Zig loader uses, driven through ctypes against the
# PyAV wheel's bundled FFmpeg; checks it reproduces PyAV's frames byte for byte, and prints struct offsets.
import ctypes as C, glob, hashlib, io, json, os, sys
LIBDIR = "/usr/local/lib/python3.12/dist-packages/av.libs"
def lib(prefix):
    return C.CDLL(sorted(glob.glob(f"{LIBDIR}/{prefix}-*.so*"))[0], mode=C.RTLD_GLOBAL)
util = lib("libavutil"); codec = lib("libavcodec"); fmt = lib("libavformat"); sws = lib("libswscale")
print("versions", util.avutil_version() >> 16, codec.avcodec_version() >> 16, fmt.avformat_version() >> 16, sws.swscale_version() >> 16)

for f, r in [(util.av_malloc, C.c_void_p), (util.av_frame_alloc, C.c_void_p), (codec.av_packet_alloc, C.c_void_p), (fmt.avformat_alloc_context, C.c_void_p),
             (fmt.avio_alloc_context, C.c_void_p), (codec.avcodec_alloc_context3, C.c_void_p), (codec.avcodec_find_decoder, C.c_void_p),
             (sws.sws_getCachedContext, C.c_void_p), (sws.sws_getCoefficients, C.c_void_p), (fmt.av_guess_frame_rate, None)]:
    f.restype = r
READ = C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(C.c_uint8), C.c_int)
SEEK = C.CFUNCTYPE(C.c_int64, C.c_void_p, C.c_int64, C.c_int)

AVERROR_EOF = -541478725
AVERROR_EAGAIN = -11

class Mem:
    def __init__(self, data): self.data = data; self.pos = 0
    def read(self, opaque, buf, n):
        if self.pos >= len(self.data): return AVERROR_EOF
        k = min(n, len(self.data) - self.pos)
        C.memmove(buf, self.data[self.pos:self.pos + k], k); self.pos += k; return k
    def seek(self, opaque, off, whence):
        if whence == 0x10000: return len(self.data)   # AVSEEK_SIZE
        whence &= 3
        if whence == 0: p = off
        elif whence == 1: p = self.pos + off
        else: p = len(self.data) + off
        if p < 0: return -22
        self.pos = p; return p

def u32(p, off): return C.c_uint32.from_address(p + off).value
def i32(p, off): return C.c_int32.from_address(p + off).value
def i64(p, off): return C.c_int64.from_address(p + off).value
def ptr(p, off): return C.c_void_p.from_address(p + off).value

OFFS = [292, 280]

def decode(data, want_info=False):
    mem = Mem(data)
    rd = READ(mem.read); sk = SEEK(mem.seek)
    buf = util.av_malloc(32768)
    io_ctx = fmt.avio_alloc_context(C.c_void_p(buf), 32768, 0, None, rd, None, sk)
    ic = fmt.avformat_alloc_context()
    C.c_void_p.from_address(ic + 32).value = io_ctx          # AVFormatContext.pb
    icp = C.c_void_p(ic)
    r = fmt.avformat_open_input(C.byref(icp), None, None, None)
    assert r >= 0, r
    ic = icp.value
    assert fmt.avformat_find_stream_info(C.c_void_p(ic), None) >= 0
    nb = u32(ic, 44); streams = ptr(ic, 48)
    info = {"nb_streams": nb}
    # find the first video stream (PyAV: container.streams.video[0])
    st = None
    for i in range(nb):
        s = ptr(streams, 8 * i)
        par = ptr(s, 16)
        if i32(par, 0) == 0:     # codec_type == AVMEDIA_TYPE_VIDEO
            st = s; break
    assert st
    par = ptr(st, 16)
    info.update(index=i32(st, 8), time_base=(i32(st, 32), i32(st, 36)), start_time=i64(st, 40), duration=i64(st, 48), nb_frames=i64(st, 56),
                avg=(i32(st, 88), i32(st, 92)), codec_id=i32(par, 4), width=i32(par, 72), height=i32(par, 76), ic_duration=i64(ic, 104), ic_start=i64(ic, 96))
    class R(C.Structure): _fields_ = [("num", C.c_int), ("den", C.c_int)]
    fmt.av_guess_frame_rate.restype = R
    g = fmt.av_guess_frame_rate(C.c_void_p(ic), C.c_void_p(st), None)
    info["guessed"] = (g.num, g.den)
    dec = codec.avcodec_find_decoder(info["codec_id"])
    ctx = codec.avcodec_alloc_context3(C.c_void_p(dec))
    assert codec.avcodec_parameters_to_context(C.c_void_p(ctx), C.c_void_p(par)) >= 0
    util.av_opt_set.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_int]
    util.av_opt_set(C.c_void_p(ctx), b"thread_type", b"frame+slice", 1)   # AV_OPT_SEARCH_CHILDREN
    assert codec.avcodec_open2(C.c_void_p(ctx), C.c_void_p(dec), None) >= 0
    pkt = codec.av_packet_alloc(); frame = util.av_frame_alloc()
    out = []
    swsctx = None
    sctx = sws.sws_alloc_context()
    util.av_opt_set_int.argtypes = [C.c_void_p, C.c_char_p, C.c_int64, C.c_int]
    assert util.av_opt_set_int(C.c_void_p(sctx), b"sws_flags", 2, 0) >= 0     # SWS_BILINEAR
    assert util.av_opt_set_int(C.c_void_p(sctx), b"threads", 0, 0) >= 0
    def set_cs(fr, cs):                                                       # PyAV's _set_frame_colorspace(frame, cs, 0)
        m = {1: 1, 4: 4, 5: 6, 7: 7, 9: 9}
        if cs in m: C.c_int32.from_address(fr + OFFS[0]).value = m[cs]
    def convert(fr):
        w, h, pf = i32(fr, 104), i32(fr, 108), i32(fr, 116)
        cs = i32(fr, OFFS[0])
        info.setdefault("frame0", (w, h, pf, cs, i32(fr, OFFS[1])))
        dst = util.av_frame_alloc()
        assert util.av_frame_copy_props(C.c_void_p(dst), C.c_void_p(fr)) >= 0
        C.c_int32.from_address(dst + 116).value = 2     # AV_PIX_FMT_RGB24
        C.c_int32.from_address(dst + 104).value = w
        C.c_int32.from_address(dst + 108).value = h
        for f_ in (fr, dst):
            C.c_int32.from_address(f_ + 284).value = 2   # color_primaries = UNSPECIFIED
            C.c_int32.from_address(f_ + 288).value = 2   # color_trc = UNSPECIFIED
        set_cs(fr, cs); set_cs(dst, cs)
        r = sws.sws_scale_frame(C.c_void_p(sctx), C.c_void_p(dst), C.c_void_p(fr))
        assert r >= 0, r
        base = ptr(dst, 0); ls = i32(dst, 64)
        rows = [C.string_at(base + y * ls, w * 3) for y in range(h)]
        out.append(hashlib.sha256(b"".join(rows)).hexdigest())
        pp = C.c_void_p(dst); util.av_frame_free(C.byref(pp))
    def drain():
        while True:
            r = codec.avcodec_receive_frame(C.c_void_p(ctx), C.c_void_p(frame))
            if r == AVERROR_EAGAIN or r == AVERROR_EOF: return
            assert r >= 0, r
            convert(frame)
            util.av_frame_unref(C.c_void_p(frame))
    while True:
        r = fmt.av_read_frame(C.c_void_p(ic), C.c_void_p(pkt))
        if r < 0: break
        if i32(pkt, 36) == info["index"]:
            rr = codec.avcodec_send_packet(C.c_void_p(ctx), C.c_void_p(pkt))
            assert rr >= 0 or rr == AVERROR_EAGAIN, rr
            drain()
        codec.av_packet_unref(C.c_void_p(pkt))
    codec.avcodec_send_packet(C.c_void_p(ctx), None); drain()
    return out, info

def first_frame_ints(data):
    """The raw int32s of the first decoded AVFrame (for offset discovery)."""
    mem = Mem(data); rd = READ(mem.read); sk = SEEK(mem.seek)
    buf = util.av_malloc(32768)
    io_ctx = fmt.avio_alloc_context(C.c_void_p(buf), 32768, 0, None, rd, None, sk)
    ic = fmt.avformat_alloc_context(); C.c_void_p.from_address(ic + 32).value = io_ctx
    icp = C.c_void_p(ic); fmt.avformat_open_input(C.byref(icp), None, None, None); ic = icp.value
    fmt.avformat_find_stream_info(C.c_void_p(ic), None)
    st = ptr(ptr(ic, 48), 0); par = ptr(st, 16)
    dec = codec.avcodec_find_decoder(i32(par, 4)); ctx = codec.avcodec_alloc_context3(C.c_void_p(dec))
    codec.avcodec_parameters_to_context(C.c_void_p(ctx), C.c_void_p(par)); codec.avcodec_open2(C.c_void_p(ctx), C.c_void_p(dec), None)
    pkt = codec.av_packet_alloc(); frame = util.av_frame_alloc()
    while fmt.av_read_frame(C.c_void_p(ic), C.c_void_p(pkt)) >= 0:
        codec.avcodec_send_packet(C.c_void_p(ctx), C.c_void_p(pkt)); codec.av_packet_unref(C.c_void_p(pkt))
        if codec.avcodec_receive_frame(C.c_void_p(ctx), C.c_void_p(frame)) >= 0: break
    else:
        codec.avcodec_send_packet(C.c_void_p(ctx), None); codec.avcodec_receive_frame(C.c_void_p(ctx), C.c_void_p(frame))
    return [i32(frame, o) for o in range(0, 480, 4)]

if __name__ == "__main__":
    exp = json.load(open("/fix/vid/expected.json"))
    if sys.argv[1] == "scan":
        rows = {}
        for e in exp:
            rows[e["file"]] = (e["colorspace"], e["color_range"], first_frame_ints(open("/fix/vid/" + e["file"], "rb").read()))
        for off in range(112, 400, 4):
            idx = off // 4
            for which, name in ((0, "colorspace"), (1, "color_range")):
                if all(r[2][idx] == r[which] for r in rows.values()):
                    print("offset", off, "matches", name)
        sys.exit()
    OFFS[0], OFFS[1] = int(sys.argv[1]), int(sys.argv[2])
    ok = bad = 0
    for e in exp:
        data = open("/fix/vid/" + e["file"], "rb").read()
        out, info = decode(data)
        same = out == e["frame_hashes"]
        ok += same; bad += (not same)
        print("OK " if same else "BAD", e["file"], len(out), len(e["frame_hashes"]), info["frame0"], info["width"], info["height"], info["avg"], info["guessed"], info["nb_frames"], info["duration"], info["ic_duration"])
    print(ok, "match", bad, "differ")
