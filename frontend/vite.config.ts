import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

// Where the dev server forwards same-origin API calls. On the host this is the
// server's published port; inside docker compose it is the `server` service.
function apiTarget(): string {
  // @ts-ignore - Deno global is present when running under Deno.
  if (typeof Deno !== 'undefined') {
    // @ts-ignore
    const fromDeno = Deno.env.get('API_PROXY_TARGET')
    if (fromDeno) return fromDeno
  }
  // @ts-ignore - process is provided via Node compat.
  return globalThis.process?.env?.API_PROXY_TARGET ?? 'http://localhost:9000'
}

const target = apiTarget()

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    // Polling makes file changes reliable inside containers / bind mounts,
    // which is what `docker compose watch` relies on.
    watch: { usePolling: true },
    // Proxy API calls to the backend so the browser only ever talks to its own
    // origin.
    proxy: {
      '/v1': { target, changeOrigin: true, ws: true },
      '/health': { target, changeOrigin: true },
    },
  },
})
