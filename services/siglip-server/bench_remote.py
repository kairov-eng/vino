"""Клиент: замер latency SigLIP2 через SSH-туннель или прямой URL.

Примеры:
  # туннель: ssh -N -L 8090:127.0.0.1:8090 aidispatcher
  python bench_remote.py C:\\dev\\Vino2026\\temp\\unbound_sample.jpg

  # или сразу через remote curl на сервере:
  python bench_remote.py --via-ssh image.jpg
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import urllib.request
except ImportError:  # pragma: no cover
    urllib = None  # type: ignore


DEFAULT_URL = "http://127.0.0.1:8090"
DEFAULT_KEY = "vino-siglip-dev"


def _post_multipart(url: str, image_path: Path, api_key: str) -> tuple[dict, float]:
    import uuid

    boundary = f"----Boundary{uuid.uuid4().hex}"
    data = image_path.read_bytes()
    filename = image_path.name
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/embed",
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "X-API-Key": api_key,
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = resp.read()
    roundtrip_ms = (time.perf_counter() - t0) * 1000
    return json.loads(raw.decode()), roundtrip_ms


def _via_ssh(image_path: Path, api_key: str, ssh_host: str, repeats: int) -> None:
    remote = f"/tmp/{image_path.name}"
    subprocess.check_call(["scp", "-q", str(image_path), f"{ssh_host}:{remote}"])
    results = []
    for i in range(repeats):
        cmd = (
            f'curl -sS -H "X-API-Key: {api_key}" '
            f'-F "image=@{remote}" http://127.0.0.1:8090/v1/embed'
        )
        t0 = time.perf_counter()
        out = subprocess.check_output(["ssh", ssh_host, cmd], text=True)
        roundtrip_ms = (time.perf_counter() - t0) * 1000
        payload = json.loads(out)
        results.append((payload, roundtrip_ms))
        print(
            f"#{i+1} server_total={payload['timings_ms']['total']}ms "
            f"infer={payload['timings_ms']['infer']}ms "
            f"preprocess={payload['timings_ms']['preprocess']}ms "
            f"ssh_roundtrip={roundtrip_ms:.0f}ms dim={payload['dim']}",
            flush=True,
        )
    _summary(results)


def _summary(results: list[tuple[dict, float]]) -> None:
    totals = [r[0]["timings_ms"]["total"] for r in results]
    infers = [r[0]["timings_ms"]["infer"] for r in results]
    print(
        json.dumps(
            {
                "runs": len(results),
                "server_total_ms": {"mean": sum(totals) / len(totals), "min": min(totals)},
                "server_infer_ms": {"mean": sum(infers) / len(infers), "min": min(infers)},
                "model_id": results[0][0]["model_id"],
                "dim": results[0][0]["dim"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--api-key", default=DEFAULT_KEY)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--via-ssh", action="store_true", help="scp+curl на сервере через SSH")
    p.add_argument("--ssh-host", default="aidispatcher")
    args = p.parse_args()

    if not args.image.is_file():
        print(f"file not found: {args.image}", file=sys.stderr)
        return 1

    if args.via_ssh:
        _via_ssh(args.image, args.api_key, args.ssh_host, args.repeats)
        return 0

    # health
    with urllib.request.urlopen(f"{args.url.rstrip('/')}/health", timeout=30) as resp:
        print("health:", resp.read().decode(), flush=True)

    results = []
    for i in range(args.repeats):
        payload, roundtrip_ms = _post_multipart(args.url, args.image, args.api_key)
        results.append((payload, roundtrip_ms))
        print(
            f"#{i+1} server_total={payload['timings_ms']['total']}ms "
            f"infer={payload['timings_ms']['infer']}ms "
            f"client_roundtrip={roundtrip_ms:.0f}ms dim={payload['dim']}",
            flush=True,
        )
    _summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
