import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    holdUntilCrawlEnd: false,
    include: [
      'react',
      'react-dom',
      'react-dom/client',
      'react/jsx-runtime',
      'react/jsx-dev-runtime',
      'react-router-dom',
    ],
  },
  server: {
    host: '127.0.0.1',
    port: 8091,
    strictPort: true,
    // Tunnel Host headers: Cloudflare / ngrok
    allowedHosts: [
      '.trycloudflare.com',
      '.ngrok-free.dev',
      'growing-warehouse-planes-tracking.trycloudflare.com',
      'shriek-untidy-hurled.ngrok-free.dev',
    ],
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8092',
        changeOrigin: true,
        // findwine OCR может идти 1–3 мин — дефолтный proxy timeout рвёт запрос → Failed to fetch
        timeout: 300_000,
        proxyTimeout: 300_000,
      },
      '/media': {
        target: 'http://127.0.0.1:8092',
        changeOrigin: true,
        timeout: 120_000,
        proxyTimeout: 120_000,
      },
    },
  },
})
