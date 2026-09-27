"""Create light HF model repo with custom SigLIP2 endpoint handler."""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
TO_ID = "cairo2000/siglip2-base-patch16-384-embed"
HANDLER_DIR = Path(r"C:\dev\Vino2026\vino-svoe\services\siglip-hf-endpoint")

api = HfApi(token=TOKEN)
print(f"create_repo {TO_ID} ...", flush=True)
url = api.create_repo(repo_id=TO_ID, repo_type="model", private=False, exist_ok=True)
print("repo:", url, flush=True)

readme = """---
library_name: transformers
tags:
- endpoints-template
- siglip2
- image-feature-extraction
pipeline_tag: image-feature-extraction
---

# SigLIP 2 embedding endpoint (custom handler)

Deploy this repo as a Hugging Face **Inference Endpoint** (Task = Custom).

Weights are loaded from `google/siglip2-base-patch16-384` at startup (`get_image_features` + L2).

## Request

```json
{"inputs": {"images": ["data:image/jpeg;base64,..."]}, "normalize": true}
```

## Response

`image_embeddings`, `dim`, `timings_ms`
"""
(HANDLER_DIR / "README_HUB.md").write_text(readme, encoding="utf-8")

files = {
    "handler.py": HANDLER_DIR / "handler.py",
    "requirements.txt": HANDLER_DIR / "requirements.txt",
    "README.md": HANDLER_DIR / "README_HUB.md",
}
for path_in_repo, local in files.items():
    print(f"upload {path_in_repo} ...", flush=True)
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=path_in_repo,
        repo_id=TO_ID,
        repo_type="model",
        commit_message=f"add {path_in_repo}",
    )
print(f"DONE https://huggingface.co/{TO_ID}", flush=True)
