---
library_name: transformers
tags:
- endpoints-template
- dinov3
- image-feature-extraction
pipeline_tag: image-feature-extraction
---

# DINOv3 ViT-B/16 embedding endpoint (custom handler)

Deploy this repo as a Hugging Face **Inference Endpoint** (Task = Custom).

Weights are loaded from gated `facebook/dinov3-vitb16-pretrain-lvd1689m` at startup
(`pooler_output` + L2-norm → 768-d).

## Required: HF token for gated Meta model

1. Accept access on https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
2. In the Inference Endpoint → **Settings / Secrets**, add:
   - `HF_TOKEN` = your Hugging Face write/read token (account that accepted the license)
3. Optional env: `DINO_MODEL_ID=facebook/dinov3-vitb16-pretrain-lvd1689m`
4. Restart / update the endpoint after changing secrets

Without `HF_TOKEN` the container gets **401 gated repo** on boot.

## Request

```json
{"inputs": {"images": ["data:image/jpeg;base64,..."]}, "normalize": true}
```

## Response

`image_embeddings`, `dim`, `timings_ms`
