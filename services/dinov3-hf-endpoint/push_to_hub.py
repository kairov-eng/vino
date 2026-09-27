"""Create light HF model repo with DINOv3 custom endpoint handler."""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

TOKEN = os.environ["HF_ACCESS_TOKEN"] if "HF_ACCESS_TOKEN" in os.environ else os.environ["HF_TOKEN"]
TO_ID = "cairo2000/dinov3-vitb16-pretrain-lvd1689m-embed"
HANDLER_DIR = Path(__file__).resolve().parent

api = HfApi(token=TOKEN)
print(f"create_repo {TO_ID} ...", flush=True)
url = api.create_repo(repo_id=TO_ID, repo_type="model", private=False, exist_ok=True)
print("repo:", url, flush=True)

files = {
    "handler.py": HANDLER_DIR / "handler.py",
    "requirements.txt": HANDLER_DIR / "requirements.txt",
    "README.md": HANDLER_DIR / "README.md",
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
