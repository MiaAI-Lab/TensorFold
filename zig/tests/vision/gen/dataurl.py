import base64, hashlib, json, sys
from tensorfold.vision.images import _data_bytes
from tensorfold.vision.images_http import ImageInputError
raw = bytes(range(256)) * 3
b = base64.b64encode(raw).decode()
urls = [
    ("data:image/png;base64," + b, 10**6),
    ("data:image/png;base64," + b[:-1], 10**6),
    ("data:image/png;base64," + b + "=", 10**6),
    ("data:image/png;base64,QQ==", 10),
    ("data:image/png;base64,QQ=", 10),
    ("data:image/png;base64,QQ", 10),
    ("data:image/png;base64,QUI=", 10),
    ("data:image/png;base64,QUI", 10),
    ("data:image/png;base64,QUJD", 10),
    ("data:image/png;base64,QUJDR", 10),
    ("data:image/png;base64,QUJD\n", 10),
    ("data:image/png;base64,QU JD", 10),
    ("data:image/png;base64,QUJD====", 10),
    ("data:image/png;base64,QU==JD", 10),
    ("data:image/png;base64,====", 10),
    ("data:image/png;base64,Q===", 10),
    ("data:image/png;base64,QUJD-_8=", 10),
    ("data:image/png;base64,QUJD+/8=", 10),
    ("data:image/png;base64,", 10),
    ("data:image/png;BASE64,QUJD", 10),
    ("data:image/png;base64;charset=x,QUJD", 10),
    ("data:image/png;charset=utf8,abc", 10),
    ("data:image/png,abc%20def%zz%4", 100),
    ("data:image/png,abc%41%4a%4A%", 100),
    ("data:,%00%ff%FF", 100),
    ("data:image/png", 10),
    ("data:image/png,", 10),
    ("data:image/png,é", 100),
    ("data:image/png;base64,QUJDé", 100),
    ("data:" + "x" * 300 + ",abc", 100),
    ("data:image/png;base64,QUJDREVGR0hJSktM", 5),
    ("data:image/png;base64,QUJDREVGR0hJSktM", 9),
    ("data:image/png;base64,QUJDREVGR0hJSktM", 12),
    ("data:image/png," + "a" * 30, 10),
    ("data:image/png," + "a" * 31, 10),
    ("data:image/png," + "%41" * 10, 10),
    ("data:image/png," + "%41" * 11, 10),
    ("data:image/png;base64," + "QUJD" * 4, 12),
    ("data:image/png;base64," + "QUJD" * 4, 11),
    ("data:image/png;base64," + "QUJD" * 5, 12),
    ("data:image/png;base64,QUJD=", 12),
    ("data:image/png;base64,AAAA", 12),
]
out = []
for u, m in urls:
    e = {"url": u, "max": m}
    try:
        d = _data_bytes(u, m)
        e["sha"] = hashlib.sha256(d).hexdigest(); e["len"] = len(d)
    except ImageInputError as ex:
        e["error"] = str(ex)
    out.append(e)
open("/fix/dataurl.json", "w").write(json.dumps(out, ensure_ascii=False))
print(len(out), sum("error" in e for e in out))
