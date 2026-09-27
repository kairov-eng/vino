# SigLIP 2 — custom handler для Hugging Face Inference Endpoints

Предупреждение «no handler.py» на `google/siglip2-...` нормально:
в чужой репозиторий файлы добавить нельзя. Нужен **свой** repo (fork),
в него кладутся `handler.py` + `requirements.txt`, endpoint создаётся **из него**.

## 1. Сделать свой репозиторий (fork весов)

Вариант A — UI: https://huggingface.co/spaces/huggingface-projects/repo_duplicator  
скопировать, например, `google/siglip2-base-patch16-384` → `ВАШ_USER/siglip2-base-patch16-384-embed`

Вариант B — CLI:

```bash
pip install -U huggingface_hub
hf auth login
hf repo create siglip2-base-patch16-384-embed --type model
# затем скопировать файлы модели из google/... или использовать duplicator
```

Рекомендация для вина: **`google/siglip2-base-patch16-384`** (768-d), не 224.

Готовый deploy-repo уже на Hub: **https://huggingface.co/cairo2000/siglip2-base-patch16-384-embed**  
(`handler.py` + `requirements.txt`; веса подтягиваются из `google/siglip2-base-patch16-384` при старте endpoint).

## 2. Добавить handler в корень repo

Файлы из этой папки положить **в корень** вашего Hub-репозитория:

- `handler.py`
- `requirements.txt`

```bash
git clone https://huggingface.co/ВАШ_USER/siglip2-base-patch16-384-embed
cd siglip2-base-patch16-384-embed
copy /path/to/siglip-hf-endpoint/handler.py .
copy /path/to/siglip-hf-endpoint/requirements.txt .
git add handler.py requirements.txt
git commit -m "add SigLIP2 embedding custom handler"
git push
```

В вкладке **Files** должны быть видны `handler.py` и `requirements.txt` рядом с весами модели.

## 3. Создать Inference Endpoint

1. https://ui.endpoints.huggingface.cloud → **New**
2. Repository: **ваш** `ВАШ_USER/siglip2-base-patch16-384-embed` (не `google/...`)
3. Task станет **Custom** (Endpoint сам подхватит `handler.py`)
4. Выберите GPU (для SigLIP2 base достаточно T4 / L4)
5. Create → дождитесь статуса Running

## 4. Вызов

```bash
curl https://YOUR_ENDPOINT_URL \
  -X POST \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"inputs\":{\"images\":[\"data:image/jpeg;base64,$(base64 -w0 photo.jpg)\"]},\"normalize\":true}"
```

Ответ: `image_embeddings` (L2), `dim`, `timings_ms`.

Локальный тест handler без endpoint:

```bash
cd vino-svoe/services/siglip-hf-endpoint
python test_handler_local.py path/to/model/or/hf-id path/to/image.jpg
```
