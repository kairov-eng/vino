# Server nginx snippets for vino-svoe.online (deployed under /opt/aidispatcher/distrib/)
#
# Deploy:
#   scp vino-svoe.*.conf.template aidispatcher:/opt/aidispatcher/distrib/nginx/
#   scp issue-vino-svoe-cert.sh apply-nginx-config.sh aidispatcher:/opt/aidispatcher/distrib/
#   ssh aidispatcher 'cd /opt/aidispatcher/distrib && chmod +x issue-vino-svoe-cert.sh && ./issue-vino-svoe-cert.sh'
#
# Public endpoints after cert:
#   https://vino-svoe.online/health
#   https://vino-svoe.online/v1/vision/json
#   https://vino-svoe.online/api_siglip2/
#   https://vino-svoe.online/api_dinov3/
#   https://vino-svoe.online/api_cross_encoder_matcher/
#   https://vino-svoe.online/  → 404 JSON (not aidispatcher UI)
#
# aidispatcher.online is untouched (separate default.conf).
