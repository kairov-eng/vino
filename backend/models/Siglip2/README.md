# Local SigLIP2 (in-process fallback)

Hub: `google/siglip2-base-patch16-384`  
Path: `backend/models/Siglip2` (`SIGLIP2_LOCAL_DIR`)

Веса (~1.5 GB) не в git. При первом запуске backend копирует их из
`~/.cache/huggingface/hub` или качает с huggingface.co.

```powershell
Set-Location C:\dev\Vino2026\vino-svoe\backend
python -c "from app.pipeline.siglip_local import ensure_model_dir; print(ensure_model_dir())"
```
