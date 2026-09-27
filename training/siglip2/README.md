# Обучение / fine-tune SigLIP2

В проде эмбеддинги идут через **Hugging Face Inference Endpoint**
(`services/siglip-hf-endpoint/`). Веса `backend/models/Siglip2/` (~1.5 GB)
**не хранятся в git** (как и Cross-Encoder `*.safetensors`).

## 1. Подготовка данных (на ПК)

| Шаг | Скрипт / источник |
|-----|-------------------|
| Кроп этикеток из каталога | `research/label_detect/2.crop_wines.py` → `media/uploads/crop/*_crop.png` |
| Сборка датасета train/val | `training/siglip2/prepare_siglip2_finetune_dataset.py` |

```powershell
# из корня vino-svoe (нужны backend/.env → DATABASE_URL и Pillow)
pip install pillow sqlalchemy psycopg2-binary python-dotenv
python training/siglip2/prepare_siglip2_finetune_dataset.py `
  --uploads C:\dev\Vino2026\media\uploads `
  --out C:\dev\Vino2026\vino-svoe\training\siglip2\dataset

# архив для Drive
Compress-Archive -Path C:\dev\Vino2026\vino-svoe\training\siglip2\dataset\* `
  -DestinationPath C:\dev\Vino2026\vino-svoe\training\siglip2\siglip2_finetune_dataset.zip -Force
```

Загрузите zip на Google Drive: `MyDrive/Vino/siglip2_finetune_dataset.zip`.

Исходники для кропов: таблица `wines` (поля `crop`, `photo_name`, `slug`) + файлы в `MEDIA_ROOT/crop/`.

Массовый **инференс** эмбеддингов в БД (не обучение):  
`research/label_detect/4.embed_siglip2_wines.py` → таблица `embeddings_siglip2`.

## 2. Обучение в Google Colab

1. https://colab.research.google.com → Runtime → **GPU (T4+)**.
2. Откройте / вставьте `training/siglip2/colab_train_siglip2.py` в ячейку.
3. Запустите. Веса сохранятся в `Drive/MyDrive/Vino/siglip2_runs/siglip2_<timestamp>/final`.

## 3. Деплой весов

- **HF Endpoint:** скопируйте `final/` в свой Hub-repo с `handler.py`  
  (см. `services/siglip-hf-endpoint/README.md`), пересоздайте endpoint.
- **Локальный fallback:** скопируйте файлы модели в `backend/models/Siglip2/`  
  (каталог в `.gitignore`).

## 4. Без fine-tune

Можно оставить базовую `google/siglip2-base-patch16-384` на Endpoint — так работает текущий прод.
