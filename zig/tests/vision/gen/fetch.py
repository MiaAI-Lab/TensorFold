# images_http.py's URL validation, IP classification and redirect joining.
import ipaddress, json, random, sys
from urllib.parse import urljoin
from tensorfold.vision.images_http import _url, _public_ip, ImageInputError

urls = [
    "https://example.com/a.png", "https://example.com", "https://example.com/", "https://example.com/a b.png", "https://example.com/a%20b.png",
    "https://example.com/a?x=1&y=2", "https://example.com/a?x=a b", "https://example.com/é.png", "https://example.com/a#frag", "https://example.com/a#",
    "http://example.com/a.png", "ftp://example.com/a.png", "HTTPS://Example.COM/A.png", "https://user:pw@example.com/a.png", "https://user@example.com/a.png",
    "https://@example.com/a.png", "https://example.com:443/a.png", "https://example.com:8443/a.png", "https://example.com:/a.png", "https://example.com:0443/a.png",
    "https://example.com:99999/a.png", "https://example.com:abc/a.png", "https://localhost/a.png", "https://LOCALHOST./a.png", "https://metadata.google.internal/x",
    "https://instance-data/latest", "https://127.0.0.1/a.png", "https://[::1]/a.png", "https://[2001:4860:4860::8888]/a.png", "https://8.8.8.8/a.png",
    "https:///a.png", "https://", "https:example.com/a.png", "//example.com/a.png", "https://exa mple.com/", "https://example.com/a\\b", "https://example.com\\@evil.com/",
    "https://example.com/\x7f", "https://example.com/a\tb", "https://xn--bcher-kva.example/a.png", "https://bücher.example/a.png", "https://ex%61mple.com/a.png",
    "https://example.com/a.png?", "https://example.com/?a=b#", "https://example.com/p;q/r", "https://example.com/a'b(c)*d,e;f=g:h@i!j$k&l+m", "https://example.com/a[b]c", "https://example.com/a{b}c|d^e`f",
    "https://example.com/" + "a" * 4090, "https://example.com/" + "a" * 4100, "https://a..b/", "https://-bad-.example/", "https://a.b/" + "%zz", "https://example.com/%41%4a/ü",
]
out = {"url": [], "ip": [], "join": []}
for u in urls:
    e = {"url": u}
    try:
        host, port, target = _url(u, 4096)
        e.update(host=host, port=port, target=target)
    except ImageInputError as ex:
        e["error"] = str(ex)
    out["url"].append(e)

rnd = random.Random(5)
ips = ["0.0.0.0", "1.1.1.1", "8.8.8.8", "10.0.0.1", "100.64.0.1", "100.127.255.255", "127.0.0.1", "169.254.169.254", "172.15.255.255", "172.16.0.1", "172.31.255.255", "172.32.0.0",
       "192.0.0.0", "192.0.0.8", "192.0.0.9", "192.0.0.10", "192.0.0.11", "192.0.0.170", "192.0.0.171", "192.0.2.1", "192.88.99.1", "192.168.1.1", "198.18.0.1", "198.19.255.255", "198.20.0.0",
       "198.51.100.1", "203.0.113.5", "224.0.0.1", "239.255.255.255", "240.0.0.1", "255.255.255.255", "168.63.129.16", "168.63.129.15", "168.63.129.17", "93.184.216.34",
       "::", "::1", "::2", "::ffff:1.2.3.4", "::ffff:127.0.0.1", "64:ff9b::1", "64:ff9b:1::1", "100::1", "100:0:0:1::1", "2001::1", "2001:1::1", "2001:1::2", "2001:1::3", "2001:3::1", "2001:4:112::1", "2001:10::1", "2001:20::1", "2001:30::1", "2001:2::1",
       "2001:db8::1", "2002::1", "2002:c000:204::1", "2606:4700:4700::1111", "2a00:1450:4001::1", "fc00::1", "fd12:3456::1", "fe80::1", "fec0::1", "ff02::1", "ff00::", "e000::1", "f000::1", "fe00::1", "fe80::", "4000::1", "8000::1", "c000::1", "a000::1", "6000::1", "2000::1", "3fff::1", "2620:4f:8000::1"]
for _ in range(300):
    if rnd.random() < .5: ips.append(str(ipaddress.IPv4Address(rnd.getrandbits(32))))
    else: ips.append(str(ipaddress.IPv6Address(rnd.getrandbits(128))))
for _ in range(120):   # near boundaries of the special blocks
    base = rnd.choice(["10.0.0.0", "100.64.0.0", "169.254.0.0", "172.16.0.0", "192.0.0.0", "192.168.0.0", "198.18.0.0", "224.0.0.0", "240.0.0.0"])
    ips.append(str(ipaddress.IPv4Address(int(ipaddress.IPv4Address(base)) + rnd.choice([-1, 0, 1, 7, 255, 65535, 1 << 20, (1 << 22) - 1, (1 << 22)]) & 0xffffffff)))
for ip in ips:
    out["ip"].append({"ip": ip, "public": _public_ip(ip)})

bases = ["https://example.com/a/b/c.png", "https://example.com/a/b/", "https://example.com", "https://example.com/a?x=1", "https://example.com:443/x/y"]
locs = ["/z.png", "z.png", "../z.png", "../../z.png", "../../../z.png", "./z.png", "z/../y.png", "?q=1", "#f", "", "//other.com/p", "//other.com", "https://other.com/p", "http://other.com/p",
        "HTTPS://Other.com/P", "ftp://x/y", "/a/./b/../c", "..", ".", "../", "a/b?c#d", "https://other.com:444/x", "javascript:alert(1)", "/\\evil.com", "//\\evil.com/", "x y", "https:other", "https:/other", "  /z.png", "/z.png  "]
for b in bases:
    for l in locs:
        out["join"].append({"base": b, "loc": l, "out": urljoin(b, l)})
json.dump(out, open("/fix/fetch.json", "w"), ensure_ascii=False)
print(len(out["url"]), len(out["ip"]), len(out["join"]))
