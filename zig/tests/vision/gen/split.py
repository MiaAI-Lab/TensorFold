# split_images / MediaMarks fixtures from the reference server.
import copy, json, sys, secrets
sys.path.insert(0, "/gen")
from tensorfold.vision.images import split_images, ImageLimits, ImageSource
from tensorfold.vision.images_http import ImageInputError
from tensorfold.server.prompts import MediaMarks
import tensorfold.vision.glm as G
secrets.token_hex = lambda n=16: "0123456789abcdef"[: 2 * n] if n == 8 else "f" * (2 * n)

D = "data:image/png;base64,iVBORw0KGgo="
def img(url=D, **kw): return {"type": "image_url", "image_url": {"url": url, **kw}}
def vid(url="data:video/mp4;base64,AAAA"): return {"type": "video_url", "video_url": {"url": url}}
def txt(t): return {"type": "text", "text": t}
def msg(role, content, **kw): return {"role": role, **({"content": content} if content is not ... else {}), **kw}

cases = []
def case(messages, **opts):
    o = dict(max_images=4, allow_urls=False, allow_videos=False, tool_media=False, max_videos=None)
    o.update(opts)
    entry = {"in": messages, "opts": o}
    try:
        out, sources = split_images(copy.deepcopy(messages), limits=ImageLimits(max_images=o["max_images"]),
                                    allow_urls=o["allow_urls"], allow_videos=o["allow_videos"],
                                    max_videos=o["max_videos"], tool_media=o["tool_media"])
        entry["out"] = out
        entry["sources"] = [{"url": s.url, **({"detail": s.detail} if isinstance(s, ImageSource) else {})} for s in sources]
    except ImageInputError as e:
        entry["error"] = str(e)
    cases.append(entry)

case([msg("user", [txt("look"), img(detail="low"), txt("and"), img()])])
case([msg("system", "be nice"), msg("user", [img(detail="high")]), msg("assistant", "ok"), msg("user", "next", name="bob")])
case([msg("user", None)])
case([{"role": "user"}])
case([])
case("nope")
case(["x"])
case([{"role": "robot", "content": "x"}])
case([{"role": 3, "content": "x"}])
case([{"content": "x"}])
case([msg("user", "hi", image_url={"url": D})])
case([msg("user", "hi", audio="x")])
case([msg("user", "hi", images=[])])   # falsy: allowed
case([msg("user", 42)])
case([msg("user", ["x"])])
case([msg("user", [{"type": "text"}])])
case([msg("user", [{"type": "text", "text": 3}])])
case([msg("user", [{"type": "text", "text": "x", "image": "y"}])])
case([msg("user", [{"type": "input_audio", "input_audio": {}}])])
case([msg("user", [{"type": "input_audio", "input_audio": {}}])], allow_videos=True)
case([msg("user", [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}])])
case([msg("user", [vid()])], allow_videos=True)
case([msg("user", [vid(), vid(), vid()])], allow_videos=True, max_videos=2)
case([msg("user", [vid(), vid(), vid()])], allow_videos=True, max_videos=4)
case([msg("user", [vid(), vid(), vid()])], allow_videos=True, max_videos=None)
case([msg("user", [{"type": "video_url", "video_url": {}}])], allow_videos=True)
case([msg("user", [{"type": "video_url", "video_url": "x"}])], allow_videos=True)
case([msg("user", [{"type": "video_url", "video_url": {"url": "http://x/y.mp4"}}])], allow_videos=True)
case([msg("user", [{"type": "video_url", "video_url": {"url": "https://x/y.mp4"}}])], allow_videos=True)
case([msg("user", [{"type": "video_url", "video_url": {"url": "https://x/y.mp4"}}])], allow_videos=True, allow_urls=True)
case([msg("user", [{"type": "video_url", "video_url": {"url": "https://x/" + "a" * 4100}}])], allow_videos=True, allow_urls=True)
case([msg("user", [{"type": "video_url", "video_url": {"url": "x"}, "image": "q"}])], allow_videos=True)
case([msg("assistant", [vid()])], allow_videos=True)
case([msg("tool", [vid(), img(), txt("r")], tool_call_id="c1")], allow_videos=True, tool_media=True)
case([msg("tool", [img()], tool_call_id="c1")], tool_media=False)
case([msg("tool", [img()], tool_call_id="c1")], tool_media=True)
case([msg("assistant", [img()])], tool_media=True)
case([msg("user", [img(), img(), img(), img(), img()])])
case([msg("user", [img(), img(), img(), img(), img()])], max_images=5)
case([msg("user", [{"type": "image_url", "image_url": {}}])])
case([msg("user", [{"type": "image_url", "image_url": {"url": ""}}])])
case([msg("user", [{"type": "image_url", "image_url": {"url": 5}}])])
case([msg("user", [{"type": "image_url", "image_url": "x"}])])
case([msg("user", [{"type": "image_url"}])])
case([msg("user", [img(detail="ultra")])])
case([msg("user", [img(detail=None)])])
case([msg("user", [img(detail=3)])])
case([msg("user", [img(detail="auto")])])
case([msg("user", [img("http://example.com/a.png")])])
case([msg("user", [img("ftp://example.com/a.png")])])
case([msg("user", [img("https://example.com/a.png")])])
case([msg("user", [img("https://example.com/a.png")])], allow_urls=True)
case([msg("user", [img("https://example.com/" + "a" * 4100)])], allow_urls=True)
case([msg("user", [img("https://example.com/" + "é" * 4000)])], allow_urls=True)
case([msg("user", [img("DATA:image/png;base64,AAAA")])])
case([msg("user", [{"type": "image_url", "image_url": {"url": D}, "audio": "x"}])])
case([msg("user", [{"type": "image_url", "image_url": {"url": D}, "image_url2": 1}])])
case([msg("user", [{"type": "image_url", "image_url": {"url": D}, "video_url": {"url": D}}])], allow_videos=True)
case([msg("user", [{"type": "frame"}])])
case([msg("user", [{"type": None}])])
case([msg("user", [{"type": 7}])])
case([msg("user", [{"nothing": 1}])])
case([msg("user", [txt("é​ ü"), img(url="data:image/png;base64,é")])])
open("/fix/split.json", "w").write(json.dumps(cases, ensure_ascii=False))

# MediaMarks
bare = G.GlmVision.__new__(G.GlmVision)
mcases = []
def mk(template, reads):
    t = copy.deepcopy(template)
    entry = {"template": template, "wanted": MediaMarks.wanted(t)}
    m = MediaMarks(t)
    entry["nonce"] = "0123456789abcdef"
    entry["marked"] = t
    entry["count"] = m.count
    rs = []
    for text, escape in reads:
        r = {"text": text, "escape": escape}
        try:
            out, order = m.read(text, escape=bare.escape_quoted if escape else None)
            r["out"], r["order"] = out, order
        except ValueError as e:
            r["error"] = str(e)
        rs.append(r)
    entry["reads"] = rs
    mcases.append(entry)

IMG, VID = G.IMAGE_SPAN, G.VIDEO_SPAN
def M(k): return f"\x00tf-media-0123456789abcdef:{k}\x00"
t1 = [msg("user", [txt("a"), {"type": "image", "detail": "auto"}, txt("b"), {"type": "video"}]), msg("tool", [{"type": "image", "detail": "low"}])]
mk(t1, [(f"u {M(0)}{IMG} t {M(1)}{VID} tool {M(2)}{IMG}", True),
        (f"u {M(0)}{IMG} t {M(1)}{VID} tool {M(2)}{IMG}", False),
        (f"tool {M(2)}{IMG} u {M(0)}{IMG}", True),
        (f"quoted {IMG} and <|image|> then {M(0)}{IMG} {M(1)}{VID}", True),
        (f"quoted {IMG} and <|image|> then {M(0)}{IMG} {M(1)}{VID}", False),
        (f"bad {M(7)}{IMG}", True),
        ("no marks", True),
        ("\x00tf-media-0123456789abcdef:x\x00 and \x00tf-media-ffffffffffffffff:0\x00 " + M(1) + VID, True),
        ("é" + M(0) + IMG + "ü" + IMG + M(1) + VID, True)])
mk([msg("user", "text only")], [("text only", True)])
mk([msg("user", [txt("x")])], [])
open("/fix/marks.json", "w").write(json.dumps(mcases, ensure_ascii=False))
print(len(cases), "split cases", sum("error" in c for c in cases), "errors;", len(mcases), "mark cases")
