# Gemini Vision Proxy

Зеркало прод-каталога `/opt/gemini-vision-proxy` на aidispatcher.

Запуск из репозитория — через родительский compose:

```bash
docker compose -f distrib/docker-compose.gemini-proxy.yml \
  --env-file distrib/gemini-proxy.env up -d --build
```

Файл `docker-compose.yml` здесь — копия того, что лежит в `/opt/gemini-vision-proxy`
(удобно синхронизировать 1:1 на сервер).
