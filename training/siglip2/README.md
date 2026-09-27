# Обучение / fine-tune SigLIP2

В проде эмбеддинги идут через **Hugging Face Inference Endpoint**
(`services/siglip-hf-endpoint/`). Веса `backend/models/Siglip2/` (~1.5 GB)
**не хранятся в git** (как и Cross-Encoder `*.safetensors`).

---

## 1. Подготовка данных (на ПК)

### Исходные данные

| Что | Откуда |
|-----|--------|
| PNG-кропы этикеток | `MEDIA_ROOT/crop/<stem>_crop.png` (обычно `C:\dev\Vino2026\media\uploads\crop\`) |
| Метаданные вин | PostgreSQL `wines` (`crop IN (1,2)`, `photo_name` / `slug`, имя, винодельня) |

### Шаг A — получить кропы (если ещё нет)

Скрипт **не в этом GitHub-репо** (лежит в локальном monorepo рядом):

`C:\dev\Vino2026\research\label_detect\2.crop_wines.py`

- Читает каталог `wines` + фото из `uploads/`
- Через Gemini режет этикетку → пишет `crop/*_crop.png` и `wines.crop = 1|2`
- Нужны: `DATABASE_URL`, ключи Gemini / Vertex, `UPLOADS_DIR` / media

```powershell
Set-Location C:\dev\Vino2026\research\label_detect
python 2.crop_wines.py --infinitive 1
# или точечно: python 2.crop_wines.py --id 3099
```

Если кропы уже есть (каталог после дампа/rsync) — шаг A пропускаете.

### Шаг B — собрать датасет для Colab (скрипт **в этом репо**)

```powershell
Set-Location C:\dev\Vino2026\vino-svoe
pip install pillow sqlalchemy psycopg2-binary python-dotenv
python training/siglip2/prepare_siglip2_finetune_dataset.py `
  --uploads C:\dev\Vino2026\media\uploads `
  --out C:\dev\Vino2026\vino-svoe\training\siglip2\dataset

Compress-Archive -Path C:\dev\Vino2026\vino-svoe\training\siglip2\dataset\* `
  -DestinationPath C:\dev\Vino2026\vino-svoe\training\siglip2\siglip2_finetune_dataset.zip -Force
```

Загрузите zip на Google Drive: `MyDrive/Vino/siglip2_finetune_dataset.zip`.

Массовый **инференс** эмбеддингов в БД (не обучение):  
локально `research/label_detect/4.embed_siglip2_wines.py` → таблица `embeddings_siglip2`.

---

## 2. Обучение в Google Colab

1. https://colab.research.google.com → Runtime → **GPU (T4+)**.
2. Новый notebook → вставьте целиком `training/siglip2/colab_train_siglip2.py` (ячейки с `!` — это Colab magics).
3. Запустите. Скрипт:
   - монтирует Drive;
   - распаковывает `MyDrive/Vino/siglip2_finetune_dataset.zip`;
   - fine-tune `google/siglip2-base-patch16-384`;
   - сохраняет веса в `Drive/MyDrive/Vino/siglip2_runs/siglip2_<timestamp>/final`.

---

## 3. Деплой весов

- **HF Endpoint:** скопируйте `final/` в Hub-repo с `handler.py`  
  (см. `services/siglip-hf-endpoint/README.md`), пересоздайте endpoint.
- **Локальный fallback:** положите файлы модели в `backend/models/Siglip2/`  
  (каталог в `.gitignore`, на сервер — rsync).

## 4. Без fine-tune

Можно оставить базовую `google/siglip2-base-patch16-384` на Endpoint — так работает текущий прод.
