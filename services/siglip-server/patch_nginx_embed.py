#!/usr/bin/env python3
from pathlib import Path

MARKER = "        proxy_pass $safer_upstream;\n    }\n\n    location / {"
BLOCK = """        proxy_pass $safer_upstream;
    }

    # SigLIP2 embeddings (container "siglip2-embed")
    location /api_siglip2/ {
        resolver 127.0.0.11 valid=30s ipv6=off;
        rewrite ^/api_siglip2/(.*)$ /$1 break;
        set $siglip_upstream http://siglip2-embed:8090;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header Connection "";
        proxy_read_timeout 300s;
        client_max_body_size 20m;
        proxy_pass $siglip_upstream;
    }

    # DINOv3 embeddings (container "dinov3-embed")
    location /api_dinov3/ {
        resolver 127.0.0.11 valid=30s ipv6=off;
        rewrite ^/api_dinov3/(.*)$ /$1 break;
        set $dinov3_upstream http://dinov3-embed:8091;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header Connection "";
        proxy_read_timeout 300s;
        client_max_body_size 20m;
        proxy_pass $dinov3_upstream;
    }

    location / {"""

p = Path("/opt/aidispatcher/distrib/nginx/conf.d/default.conf")
t = p.read_text()
if "api_siglip2" in t:
    print("already patched")
elif MARKER not in t:
    raise SystemExit("marker not found")
else:
    p.write_text(t.replace(MARKER, BLOCK, 1))
    print("patched ok")
