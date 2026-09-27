#!/bin/bash
set -e
for i in $(seq 1 40); do
  code=$(curl -sS -m 5 -o /tmp/siglip_h.json -w '%{http_code}' http://127.0.0.1:8090/health 2>/dev/null || true)
  if [ -z "$code" ]; then code=000; fi
  echo "try $i http=$code"
  if [ "$code" = "200" ]; then
    cat /tmp/siglip_h.json
    echo
    docker stats siglip2-embed --no-stream --format 'mem={{.MemUsage}}'
    exit 0
  fi
  sleep 15
done
echo FAIL
docker logs siglip2-embed --tail 80
docker ps -a --filter name=siglip2-embed
free -h | head -2
exit 1
