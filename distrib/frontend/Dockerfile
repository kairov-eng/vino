# Build static SPA, serve with nginx.
# context = repo root (vino-svoe/)
FROM node:22-bookworm-slim AS build
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json* frontend/.npmrc* ./
RUN npm ci || npm install
COPY frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY distrib/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1/ || exit 1
