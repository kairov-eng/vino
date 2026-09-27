#!/bin/bash
set -e
for i in $(seq 1 60); do
  code=$(curl -sS -m 5 -o /tmp/dinov3_h.json -w '%{http_code}' http://127.0.0.1:8091/health 2>/dev/null || echo 000)
  echo "try $i http=$code"
  if [ "$code" = "200" ]; then
    cat /tmp/dinov3_h.json
    echo
    docker stats dinov3-embed --no-stream --format 'mem={{.MemUsage}}'
    exit 0
  fi
  sleep 15
done
echo FAIL
docker logs dinov3-embed --tail 80
docker ps -a --filter name=dinov3-embed
free -h | head -2
exit 1
