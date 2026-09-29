# Nginx templates for vino-svoe.online

| Файл | Назначение |
|------|------------|
| `vino-svoe.https.conf.template` | **Канон с прода** (aidispatcher). Копия `services/server-nginx/…` |
| `vino-svoe.http.conf.template` | HTTP→HTTPS + ACME |
| `vino-svoe.app.conf.example` | Устаревший укороченный пример (не использовать для деплоя) |

На сервере шаблоны лежат в `/opt/aidispatcher/distrib/nginx/`.  
Применение: `services/server-nginx/apply-nginx-config.sh` или `cd /opt/aidispatcher/distrib && ./apply-nginx-config.sh`.
