# =============================================================================
# Colab: fine-tune SigLIP2 (image encoder) on wine label crops
# =============================================================================
#
# Подготовка данных (на ПК, НЕ в Colab):
#   1) Этикетки: research/label_detect/2.crop_wines.py → media/uploads/crop/*_crop.png
#   2) Датасет:
#        python training/siglip2/prepare_siglip2_finetune_dataset.py \
#          --uploads C:\dev\Vino2026\media\uploads \
#          --out C:\dev\Vino2026\vino-svoe\training\siglip2\dataset
#   3) Заархивировать training/siglip2/dataset → Drive:
#        MyDrive/Vino/siglip2_finetune_dataset.zip
#
# Colab:
#   Runtime → GPU (T4+)
#   Вставьте этот файл в ячейку и запустите целиком
#
# Результат: Drive/MyDrive/Vino/siglip2_runs/<timestamp>/
# Дальше: залить веса в HF Inference Endpoint (services/siglip-hf-endpoint/)
# или положить в backend/models/Siglip2 для локального fallback.
# =============================================================================

# --- 0) GPU ---
!nvidia-smi

# --- 1) deps + Drive ---
!pip install -q -U "transformers>=4.45" accelerate pillow datasets

from google.colab import drive

drive.mount("/content/drive")

# --- 2) unpack ---
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup

ZIP = Path("/content/drive/MyDrive/Vino/siglip2_finetune_dataset.zip")
DATA = Path("/content/datasets/siglip2_finetune")
assert ZIP.is_file(), f"Upload zip to Drive: {ZIP}"

DATA.mkdir(parents=True, exist_ok=True)
!unzip -q -o "{ZIP}" -d "{DATA}"
# zip may contain top-level "dataset/" folder
root = DATA
if (DATA / "images").is_dir():
    root = DATA
else:
    subs = [p for p in DATA.iterdir() if p.is_dir()]
    root = subs[0] if len(subs) == 1 else DATA
assert (root / "images" / "train").is_dir(), root

MODEL_ID = "google/siglip2-base-patch16-384"
BATCH = 16
EPOCHS = 3
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
# train vision tower + text tower lightly (contrastive)
for p in model.parameters():
    p.requires_grad = True


class WineFolder(Dataset):
    def __init__(self, split: str):
        self.items: list[tuple[Path, int]] = []
        base = root / "images" / split
        for wine_dir in sorted(base.iterdir()):
            if not wine_dir.is_dir():
                continue
            wid = int(wine_dir.name)
            for img in wine_dir.glob("*.jpg"):
                self.items.append((img, wid))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, wid = self.items[idx]
        img = Image.open(path).convert("RGB")
        return img, wid


def collate(batch):
    images, labels = zip(*batch)
    enc = processor(images=list(images), return_tensors="pt", padding=True)
    # dummy texts = wine id as string (weak supervision); better: name/winery from meta.jsonl
    texts = [f"wine {i}" for i in labels]
    txt = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    enc["input_ids"] = txt["input_ids"]
    enc["attention_mask"] = txt.get("attention_mask")
    return enc


# Prefer captions from meta.jsonl if present
meta = {}
meta_path = root / "meta.jsonl"
if meta_path.is_file():
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        caption = " | ".join(
            x for x in [rec.get("winery"), rec.get("name"), rec.get("slug")] if x
        )
        meta[int(rec["wine_id"])] = caption or f"wine {rec['wine_id']}"


class WineFolderCaption(WineFolder):
    def __getitem__(self, idx: int):
        path, wid = self.items[idx]
        img = Image.open(path).convert("RGB")
        return img, meta.get(wid, f"wine {wid}")


def collate_cap(batch):
    images, texts = zip(*batch)
    img_enc = processor(images=list(images), return_tensors="pt", padding=True)
    txt_enc = processor(
        text=list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    return {
        "pixel_values": img_enc["pixel_values"],
        "input_ids": txt_enc["input_ids"],
        "attention_mask": txt_enc.get("attention_mask"),
    }


train_ds = WineFolderCaption("train")
val_ds = WineFolderCaption("val")
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_cap, num_workers=2)
print("train images:", len(train_ds), "val:", len(val_ds), "device:", DEVICE)

opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR, weight_decay=0.01)
steps = max(1, EPOCHS * len(train_loader))
sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

stamp = time.strftime("%Y%m%d_%H%M%S")
OUT = Path(f"/content/drive/MyDrive/Vino/siglip2_runs/siglip2_{stamp}")
OUT.mkdir(parents=True, exist_ok=True)

model.train()
step = 0
for epoch in range(EPOCHS):
    total = 0.0
    n = 0
    for batch in train_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items() if v is not None}
        out = model(**batch, return_loss=True)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        total += float(loss.item())
        n += 1
        step += 1
        if step % 20 == 0:
            print(f"epoch {epoch+1} step {step} loss={loss.item():.4f}")
    print(f"=== epoch {epoch+1} mean_loss={total / max(1, n):.4f} ===")
    model.save_pretrained(OUT / f"epoch_{epoch+1}")
    processor.save_pretrained(OUT / f"epoch_{epoch+1}")

model.save_pretrained(OUT / "final")
processor.save_pretrained(OUT / "final")
print("saved ->", OUT / "final")
