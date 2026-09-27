# YOLO weights for label detection (Ultralytics)

- `label_detect.pt` — best checkpoint from training run `label_detect-4` (class `label`)
- `args.yaml` — training args snapshot

Used by findwine when `yolo_variant=label` via `YOLO_MODEL_PATH` (default: this file).

Bottle+label model: `../yolo_bottle_label/`. Full rules: `docs/findwine-pipeline.md` §3.
