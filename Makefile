# Parakeet ASR server — one production image, one or two instances.
#
#   make run-parakeet      # Parakeet Ultra, GPU 0 -> :9000
#   make run-canary        # Canary 1B v2,   GPU 1 -> :9001
#   GPU=2 make run-canary  # explicit GPU wins
#
# Each instance has its own profile (.env.<instance>) and compose project, so
# they coexist. Details: docs/deployment.md. `make` lists all targets.

SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE_FILE := docker-compose.prod.yml
INSTANCES    := parakeet canary

# --- tunables ----------------------------------------------------------------
# Per-instance GPU defaults, so a bare `make run-canary` cannot land on the
# parakeet GPU (compose would recreate the container there instead of failing).
GPU_PARAKEET ?= 0
GPU_CANARY   ?= 1
HEALTH_TIMEOUT_S ?= 420

# Resolve an instance profile: e.g. `make run-parakeet` -> port/name from .env.parakeet
port_of  = $(shell set -a; . ./.env.$(1); set +a; echo $$HOST_PORT)
name_of  = $(shell set -a; . ./.env.$(1); set +a; echo $$CONTAINER_NAME)

# GPU for an instance: explicit GPU=<n> > GPU_<INSTANCE> > default
gpu_for = $(if $(filter command line environment,$(origin GPU)),$(GPU),$(GPU_$(shell echo $(1) | tr a-z A-Z)))

# docker compose invocation for one instance, `$(2)` = GPU index
compose = GPU=$(2) docker compose -f $(COMPOSE_FILE) -p asr-$(1) --env-file .env.$(1)

# Wait until the container reports healthy, then print where it is reachable.
define wait_healthy
	@priv=""; docker ps --format '{{.Names}}' | grep -qx "$(call name_of,$(1))" || priv="sudo"; \
	for i in $$(seq 1 $$(( $(HEALTH_TIMEOUT_S) / 3 ))); do \
	  st=$$($$priv docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$(call name_of,$(1))" 2>/dev/null || echo missing); \
	  case $$st in \
	    healthy)   echo "  [$(1)] healthy  ->  UI + API: http://localhost:$(call port_of,$(1))"; exit 0;; \
	    unhealthy) echo "  [$(1)] UNHEALTHY, last log lines:"; $$priv docker logs --tail 60 "$(call name_of,$(1))"; exit 1;; \
	    missing)   echo "  [$(1)] container not running (check 'make logs-$(1)')"; exit 1;; \
	  esac; \
	  sleep 3; \
	done; \
	echo "  [$(1)] still starting after $(HEALTH_TIMEOUT_S)s — follow 'make logs-$(1)'"; exit 1
endef

# --- build -------------------------------------------------------------------
.PHONY: build
build: ## Build the production image (frontend bundled) used by both instances
	docker compose -f $(COMPOSE_FILE) --env-file .env.parakeet build app
	@echo "image ready: parakeet-asr:prod"

# --- run ---------------------------------------------------------------------
.PHONY: run-parakeet
run-parakeet: ## Start Parakeet Ultra (GPU=<n>, default GPU_PARAKEET) on :9000
	@echo "starting parakeet on GPU $(call gpu_for,parakeet)"
	$(call compose,parakeet,$(call gpu_for,parakeet)) up -d --build
	$(call wait_healthy,parakeet)

.PHONY: run-canary
run-canary: ## Start Canary 1B v2 (GPU=<n>, default GPU_CANARY) on :9001
	@echo "starting canary on GPU $(call gpu_for,canary)"
	$(call compose,canary,$(call gpu_for,canary)) up -d --build
	$(call wait_healthy,canary)

.PHONY: run-both
run-both: ## Start both instances (parakeet GPU_PARAKEET, canary GPU_CANARY)
	@if [ "$(call gpu_for,parakeet)" = "$(call gpu_for,canary)" ]; then \
	  echo "refusing to start both on GPU $(call gpu_for,parakeet): set GPU_PARAKEET or GPU_CANARY"; exit 1; fi
	@echo "starting parakeet on GPU $(call gpu_for,parakeet), canary on GPU $(call gpu_for,canary)"
	$(call compose,parakeet,$(call gpu_for,parakeet)) up -d --build
	$(call wait_healthy,parakeet)
	$(call compose,canary,$(call gpu_for,canary)) up -d --build
	$(call wait_healthy,canary)

# --- stop / restart ----------------------------------------------------------
.PHONY: stop-parakeet stop-canary stop-all restart-parakeet restart-canary
stop-parakeet: ## Stop and remove the parakeet instance
	$(call compose,parakeet,$(call gpu_for,parakeet)) down

stop-canary: ## Stop and remove the canary instance
	$(call compose,canary,$(call gpu_for,canary)) down

stop-all: ## Stop and remove both instances
	-$(call compose,parakeet,$(call gpu_for,parakeet)) down
	-$(call compose,canary,$(call gpu_for,canary)) down
	@echo "both instances stopped"

restart-parakeet: stop-parakeet run-parakeet ## Recreate the parakeet instance
restart-canary: stop-canary run-canary ## Recreate the canary instance

# --- inspect -----------------------------------------------------------------
.PHONY: ps status
ps: ## Show both instances (compose project state)
	@echo "== parakeet (project asr-parakeet)"; $(call compose,parakeet,$(call gpu_for,parakeet)) ps || true
	@echo "== canary   (project asr-canary)";   $(call compose,canary,$(call gpu_for,canary)) ps || true

status: ## Show health, model, precision, stream slots and queue for both
	@for i in $(INSTANCES); do \
	  port=$$(set -a; . ./.env.$$i; set +a; echo $$HOST_PORT); \
	  echo "== $$i  (http://localhost:$$port)"; \
	  curl -fsS --max-time 5 "http://localhost:$$port/health" 2>/dev/null \
	    | python3 -c 'import json,sys; d=json.load(sys.stdin); \
	        print("   model:", d["model"]); \
	        print("   dtype:", d.get("dtype"), "| matmul:", d.get("matmul_precision")); \
	        print("   languages:", "selectable" if d.get("supports_languages") else "auto-detect"); \
	        print("   streams:", d["stream_capacity"], "| queue:", d["queue"])' \
	    || echo "   not running"; \
	done

.PHONY: logs-parakeet logs-canary
logs-parakeet: ## Follow parakeet logs
	$(call compose,parakeet,$(GPU_PARAKEET)) logs -f --tail 100

logs-canary: ## Follow canary logs
	$(call compose,canary,$(GPU_CANARY)) logs -f --tail 100

.PHONY: health-parakeet health-canary
health-parakeet: ## curl the parakeet /health
	@curl -fsS "http://localhost:$(call port_of,parakeet)/health" | python3 -m json.tool

health-canary: ## curl the canary /health
	@curl -fsS "http://localhost:$(call port_of,canary)/health" | python3 -m json.tool

.PHONY: metrics-parakeet metrics-canary
metrics-parakeet: ## Print the parakeet scheduler/streaming metrics
	@curl -fsS "http://localhost:$(call port_of,parakeet)/metrics" | grep -E "^asr_(model_ready|workers|queue|inflight|stream_)" | grep -v "_created"

metrics-canary: ## Print the canary scheduler/streaming metrics
	@curl -fsS "http://localhost:$(call port_of,canary)/metrics" | grep -E "^asr_(model_ready|workers|queue|inflight|stream_)" | grep -v "_created"

# --- development stack -------------------------------------------------------
.PHONY: dev-up dev-down
dev-up: ## Start the dev stack (hot-reload UI on :5173, API on :9000, uses .env)
	docker compose up -d

dev-down: ## Stop the dev stack (frees :9000 for the parakeet instance)
	docker compose down

# --- help --------------------------------------------------------------------
.PHONY: help
help: ## Show this help
	@echo "Parakeet ASR server recipes"
	@echo
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "GPUs: parakeet -> $(call gpu_for,parakeet), canary -> $(call gpu_for,canary) (override with GPU=<n>)"
	@echo "Profiles: .env.parakeet (:$(call port_of,parakeet))  .env.canary (:$(call port_of,canary))"
	@echo "Both instances serve their own bundled UI (production image)."
