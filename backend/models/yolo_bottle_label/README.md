# YOLO bottle + label

Weights: `best.pt` (Ultralytics YOLO11n, classes: `0=bottle`, `1=label`).

Source training run: `bottle_label_20260922_145130`.

Used when scanner setting `yolo_variant=bottle_label`.

## Post-process (findwine)

See [`docs/findwine-pipeline.md`](../../../docs/findwine-pipeline.md) §3.2:

1. Drop orphan labels: `conf≤0.5` and ≤50% area inside any bottle, if other labels sit in bottles.
2. Merge **all** labels with ≥85% area inside the same bottle (≥2) into one crop box (top from topmost, bottom from bottommost).
3. Overlay: bottle boxes with `coef=…`, all raw labels, yellow primary (often merged).
