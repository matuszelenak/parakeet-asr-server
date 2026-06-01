# syntax=docker/dockerfile:1

# ---- shared dependency layer ----
FROM denoland/deno:2.8.1 AS deps
WORKDIR /app
COPY deno.json package.json ./
# Populates node_modules (nodeModulesDir=auto) for Vite/Svelte.
RUN deno install

# ---- dev server (used by docker compose watch) ----
FROM deps AS dev
WORKDIR /app
COPY . .
EXPOSE 5173
CMD ["deno", "task", "dev"]

# ---- production build ----
FROM deps AS build
WORKDIR /app
COPY . .
# The bundle calls the API at its own origin, so no API base is baked in.
RUN deno task build
