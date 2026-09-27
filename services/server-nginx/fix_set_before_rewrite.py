#!/usr/bin/env python3
"""Fix nginx: set $upstream MUST be before rewrite ... break (rewrite module)."""
from pathlib import Path

FILES = [
    Path("/opt/aidispatcher/distrib/nginx/conf.d/default.conf"),
    Path("/opt/aidispatcher/distrib/nginx/https.conf.template"),
    Path("/opt/aidispatcher/distrib/nginx/http.conf.template"),
]

REPLACEMENTS = [
    (
        "rewrite ^/api_siglip2/(.*)$ /$1 break;\n"
        "        set $siglip_upstream http://siglip2-embed:8090;",
        "set $siglip_upstream http://siglip2-embed:8090;\n"
        "        rewrite ^/api_siglip2/(.*)$ /$1 break;",
    ),
    (
        "rewrite ^/api_dinov3/(.*)$ /$1 break;\n"
        "        set $dinov3_upstream http://dinov3-embed:8091;",
        "set $dinov3_upstream http://dinov3-embed:8091;\n"
        "        rewrite ^/api_dinov3/(.*)$ /$1 break;",
    ),
]


def main() -> None:
    for path in FILES:
        if not path.exists():
            print(f"MISSING {path}")
            continue
        text = path.read_text(encoding="utf-8")
        # normalize CRLF for matching
        norm = text.replace("\r\n", "\n")
        new = norm
        for old, repl in REPLACEMENTS:
            if old in new:
                new = new.replace(old, repl)
                print(f"FIXED block in {path}")
            else:
                print(f"pattern not found in {path}: {old[:40]!r}...")
        if new != norm:
            # keep original line endings if file was CRLF
            if "\r\n" in text:
                new = new.replace("\n", "\r\n")
            path.write_text(new, encoding="utf-8")
            print(f"WROTE {path}")
        else:
            print(f"NOCHANGE {path}")


if __name__ == "__main__":
    main()
