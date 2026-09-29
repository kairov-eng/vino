# Локальный запуск frontend + backend (без Cursor)

Порты по умолчанию: **frontend `8091`**, **backend `8092`**.  
Открыть UI: http://127.0.0.1:8091 · API: http://127.0.0.1:8092

## Запуск одним кликом

Дважды кликните **`start-dev.bat`** в этой папке (`vino-svoe\`).  
Скрипт освободит порты и откроет два окна PowerShell (backend + frontend).

Перед первым запуском: `backend/.env`, Postgres с каталогом, один раз `pip install` в backend и `npm install` в frontend.

---

## Вручную (два окна PowerShell)

Нужны два **отдельных** окна. Не смешивайте процессы в одном.

### 0. Освободить порты

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\dev\Vino2026\vino-svoe\scripts\dev-free-ports.ps1
```

### 1. Backend — окно A

```powershell
Set-Location C:\dev\Vino2026\vino-svoe\backend
# при необходимости: .\.venv\Scripts\Activate.ps1
python run_dev.py
```

Готово: `Application startup complete` / `Uvicorn running on http://127.0.0.1:8092`.

### 2. Frontend — окно B

```powershell
Set-Location C:\dev\Vino2026\vino-svoe\frontend
npm run dev
```

Готово: `Local: http://127.0.0.1:8091/`.

Vite проксирует `/api` и `/media` на backend `:8092`.

---

## Остановка

В каждом окне сервера: `Ctrl+C`. Если порт занят — снова `start-dev.bat` (он сначала free-ports) или скрипт из п.0.

---

## Быстрая проверка

```powershell
curl http://127.0.0.1:8092/api/health
curl -I http://127.0.0.1:8091/
```
