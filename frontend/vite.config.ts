import { defineConfig, type Plugin } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import { createReadStream, readFileSync } from 'node:fs'
import { resolve } from 'node:path'

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

// onnxruntime-web imports its thread-worker module at runtime; Vite would block
// that import ("should not be imported from source code"), so this plugin serves
// the file before transformation and emits it into the build.
function ortThreadWorkerPlugin(): Plugin {
  const fileName = 'ort-wasm-simd-threaded.mjs'
  const src = resolve(process.cwd(), 'node_modules/onnxruntime-web/dist', fileName)

  return {
    name: 'ort-thread-worker',
    // Pre-Vite middleware: intercept before module graph resolution.
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if ((req.url ?? '').split('?')[0] === `/${fileName}`) {
          res.setHeader('Content-Type', 'application/javascript')
          createReadStream(src).pipe(res)
          return
        }
        next()
      })
    },
    generateBundle() {
      this.emitFile({ type: 'asset', fileName, source: readFileSync(src) })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte(), ortThreadWorkerPlugin()],
  server: {
    allowedHosts: ['canary-dev.the-killer.app'],
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
