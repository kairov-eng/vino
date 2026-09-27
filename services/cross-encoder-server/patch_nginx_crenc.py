#!/usr/bin/env python3
"""Insert /api_cross_encoder_matcher/ into vino-svoe nginx conf if missing."""
from pathlib import Path

BLOCK = """
    # CrossEncoder matcher (container "cross-encoder-matcher")
    location /api_cross_encoder_matcher/ {
        resolver 127.0.0.11 valid=30s ipv6=off;
        set $crenc_upstream http://cross-encoder-matcher:8094;
        rewrite ^/api_cross_encoder_matcher/(.*)$ /$1 break;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header Connection "";
        proxy_read_timeout 300s;
        client_max_body_size 5m;
        proxy_pass $crenc_upstream;
    }
"""

MARKER = "    # Everything else: explicit error"
ALT_MARKER = "    location / {"

for rel in (
    "nginx/conf.d/vino-svoe.conf",
    "nginx/conf.d/default.conf",
):
    p = Path("/opt/aidispatcher/distrib") / rel
    if not p.is_file():
        print(f"skip missing {p}")
        continue
    t = p.read_text(encoding="utf-8")
    if "api_cross_encoder_matcher" in t:
        print(f"already patched {p}")
        continue
    if "api_dinov3" not in t:
        print(f"no dinov3 block in {p}, skip")
        continue
    # insert after dinov3 location block closing, before catch-all
    needle = "        proxy_pass $dinov3_upstream;\n    }\n"
    if needle in t:
        t = t.replace(needle, needle + "\n" + BLOCK + "\n", 1)
        p.write_text(t, encoding="utf-8")
        print(f"patched after dinov3: {p}")
        continue
    if MARKER in t:
        t = t.replace(MARKER, BLOCK + "\n" + MARKER, 1)
        p.write_text(t, encoding="utf-8")
        print(f"patched before marker: {p}")
        continue
    print(f"FAILED to patch {p}")
