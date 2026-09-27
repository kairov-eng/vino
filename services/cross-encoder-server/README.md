# CrossEncoder text matcher (CPU) — sibling of siglip2 / dinov3 embed services.
#
# Public:
#   GET  https://vino-svoe.online/api_cross_encoder_matcher/health
#   POST https://vino-svoe.online/api_cross_encoder_matcher/v1/match
# Header: X-API-Key (same as EMBED_API_KEY / siglip / dinov3)
#
# Model files: copy from vino-svoe/backend/models/cross_encoder into ./model/
# Docs/OpenAPI disabled. Model loads + warms up in lifespan before serving.
#
# Почему медленнее локалки (aidispatcher VPS):
# - QEMU CPU без AVX/AVX2 (только SSE) → PyTorch CPU в разы медленнее ноутбука
# - на хосте 4 vCPU, рядом ещё siglip2+dinov3; раньше лимит контейнера был 2 CPU / 2 потока
# Ускорение без смены железа: больше TORCH/OMP threads + cpus в compose (сейчас 4 / 3.5).
# Дальше по эффекту: хост с AVX2/GPU, или не гонять embed+crenc параллельно на одних ядрах.
