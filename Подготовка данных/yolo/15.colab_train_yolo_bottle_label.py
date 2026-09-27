# =============================================================================
# Colab: обучение YOLO (bottle + label)
# =============================================================================
#
# Как использовать:
#   1. Откройте https://colab.research.google.com
#   2. Runtime → Change runtime type → GPU (T4)
#   3. Вставьте ВЕСЬ этот файл в одну ячейку и запустите
#      (или копируйте блоки по секциям)
#
# Датасет на Drive:
#   /content/drive/MyDrive/Vino/yolo_bottle_label.zip
#   внутри zip: yolo/{images,labels,data.yaml}
#
# Веса пишутся на Drive в отдельную папку с timestamp — не затирают
# прошлые прогоны (например label_detect).
#
# =============================================================================

# --- 0) GPU check ---
!nvidia-smi

# --- 1) deps + Drive ---
!pip install -q -U ultralytics

from google.colab import drive
drive.mount("/content/drive")

# --- 2) unpack dataset ---
import os
import time
from pathlib import Path

ZIP_PATH = Path("/content/drive/MyDrive/Vino/yolo_bottle_label.zip")
DATA_ROOT = Path("/content/datasets/yolo_bottle_label")
YOLO_DIR = DATA_ROOT / "yolo"  # внутри zip папка yolo/

assert ZIP_PATH.is_file(), f"Не найден zip: {ZIP_PATH}"

DATA_ROOT.mkdir(parents=True, exist_ok=True)
!unzip -q -o "{ZIP_PATH}" -d "{DATA_ROOT}"

assert (YOLO_DIR / "images" / "train").is_dir(), f"Нет images/train в {YOLO_DIR}"
assert (YOLO_DIR / "labels" / "train").is_dir(), f"Нет labels/train в {YOLO_DIR}"

# data.yaml должен указывать на путь внутри Colab
yaml_text = f"""path: {YOLO_DIR.as_posix()}
train: images/train
val: images/val
names:
  0: bottle
  1: label
"""
data_yaml = YOLO_DIR / "data.yaml"
data_yaml.write_text(yaml_text, encoding="utf-8")
print("data.yaml ->", data_yaml)
print(data_yaml.read_text())

# --- 3) unique run name (не пересекается с прошлыми) ---
stamp = time.strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"bottle_label_{stamp}"
PROJECT = "/content/drive/MyDrive/Vino/runs"

print("project:", PROJECT)
print("name   :", RUN_NAME)
print("weights will be in:", f"{PROJECT}/{RUN_NAME}/weights/")

# --- 4) train ---
from ultralytics import YOLO

class ProgressCallback:
    def __init__(self):
        self.t0 = time.time()

    def on_train_epoch_start(self, trainer):
        ep = trainer.epoch + 1
        total = trainer.epochs
        elapsed = time.time() - self.t0
        print(
            f"\n=== Epoch {ep}/{total} | elapsed {elapsed / 60:.1f} min ===",
            flush=True,
        )

    def on_fit_epoch_end(self, trainer):
        ep = trainer.epoch + 1
        metrics = trainer.metrics or {}
        map50 = metrics.get("metrics/mAP50(B)")
        extra = f" | mAP50={map50:.3f}" if map50 is not None else ""
        print(f"=== Epoch {ep} done{extra} ===", flush=True)


model = YOLO("yolo11n.pt")  # или yolov8n.pt
cb = ProgressCallback()
model.add_callback("on_train_epoch_start", cb.on_train_epoch_start)
model.add_callback("on_fit_epoch_end", cb.on_fit_epoch_end)

results = model.train(
    data=str(data_yaml),
    epochs=80,
    imgsz=640,       # мелкие объекты на полке → можно 960
    batch=16,        # OOM → 8 или 4
    device=0,
    project=PROJECT,
    name=RUN_NAME,   # уникальное имя прогона
    exist_ok=False,  # не перезаписывать чужой run с тем же именем
    verbose=True,
)

print("best weights:", f"{PROJECT}/{RUN_NAME}/weights/best.pt")
print("last weights:", f"{PROJECT}/{RUN_NAME}/weights/last.pt")

# --- 5) быстрая проверка на val (опционально) ---
best = YOLO(f"{PROJECT}/{RUN_NAME}/weights/best.pt")
best.predict(
    source=str(YOLO_DIR / "images" / "val"),
    save=True,
    conf=0.25,
    project=PROJECT,
    name=f"{RUN_NAME}_predict",
    exist_ok=False,
)

# =============================================================================
# Если обучение оборвалось — ПРОДОЛЖИТЬ (отдельная ячейка):
#
# from ultralytics import YOLO
# model = YOLO("/content/drive/MyDrive/Vino/runs/bottle_label_YYYYMMDD_HHMMSS/weights/last.pt")
# model.train(resume=True)
# =============================================================================
