#!/usr/bin/env python3
from pathlib import Path

p = Path("/opt/aidispatcher/distrib/nginx/conf.d/default.conf")
t = p.read_text()

old = """    location /api_siglip2/ {
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
    }"""

new = """    location /api_siglip2/ {
        resolver 127.0.0.11 valid=30s ipv6=off;
        set $siglip_upstream http://siglip2-embed:8090;
        rewrite ^/api_siglip2/(.*)$ /$1 break;
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
        set $dinov3_upstream http://dinov3-embed:8091;
        rewrite ^/api_dinov3/(.*)$ /$1 break;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header Connection "";
        proxy_read_timeout 300s;
        client_max_body_size 20m;
        proxy_pass $dinov3_upstream;
    }"""

if old not in t:
    raise SystemExit("block not found")
p.write_text(t.replace(old, new, 1))
print("fixed set-before-rewrite")
