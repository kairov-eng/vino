# YOLO weights for label detection (Ultralytics)

- `label_detect.pt` — best checkpoint (`*.pt` **не в git**, см. корневой `.gitignore`)
- `args.yaml` — training args snapshot (в git)

Used by findwine when `yolo_variant=label` via `YOLO_MODEL_PATH` (default: this file).

На сервер: `rsync` в `/var/lib/vino-svoe/models/yolo/` (см. `distrib/README.md`).

Bottle+label model: `../yolo_bottle_label/`. Full rules: `docs/findwine-pipeline.md` §3.
