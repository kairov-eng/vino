# Cross-Encoder weights — not in git

Large files (`model.safetensors` ~450 MB, `tokenizer.json`, SentencePiece) are **gitignored**,
same policy as SigLIP2.

Deploy:

1. Train locally: `research/models/ocr_matcher/train_cross_encoder.py`
2. Copy artifacts into `backend/models/cross_encoder/` **or** `services/cross-encoder-server/model/`
3. On server: rsync to `/var/lib/vino-svoe/models/cross_encoder/` and mount into `vino_backend` / CE container

Small files kept in git: `README.md`, `config.json`, `metrics.json`, `train_config.json`, `special_tokens_map.json`, `tokenizer_config.json`.
