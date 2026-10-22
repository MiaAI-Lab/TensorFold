import sys
sys.argv = ["x", "none"]
exec(open("/gen/av_probe.py").read().split('if __name__ == "__main__":')[0])
import json
exp = {e["file"]: e for e in json.load(open("/fix/vid/expected.json"))}
name = "h264_420_64x48.mp4"
e = exp[name]
print({k: e[k] for k in ("width", "height", "avg_rate", "frames", "duration", "time_base", "container_duration")})
data = open("/fix/vid/" + name, "rb").read()
mem = Mem(data); rd = READ(mem.read); sk = SEEK(mem.seek)
buf = util.av_malloc(32768)
io_ctx = fmt.avio_alloc_context(C.c_void_p(buf), 32768, 0, None, rd, None, sk)
ic = fmt.avformat_alloc_context(); C.c_void_p.from_address(ic + 32).value = io_ctx
icp = C.c_void_p(ic); print("open", fmt.avformat_open_input(C.byref(icp), None, None, None)); ic = icp.value
fmt.avformat_find_stream_info(C.c_void_p(ic), None)
print("ic words", [hex(C.c_uint64.from_address(ic + 8 * i).value) for i in range(0, 20)])
print("ic ints @40..", [i32(ic, o) for o in range(40, 48, 4)])
nb = u32(ic, 44)
print("nb_streams", nb)
streams = ptr(ic, 48)
s0 = ptr(streams, 0)
print("stream words", [hex(C.c_uint64.from_address(s0 + 8 * i).value) for i in range(0, 20)])
for off in range(0, 160, 8):
    p = ptr(s0, off)
    print(off, hex(p or 0))
par = ptr(s0, 16)
print("par ints", [(o, i32(par, o)) for o in range(0, 120, 4)])
