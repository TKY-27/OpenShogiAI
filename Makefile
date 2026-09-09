SHELL := /bin/sh
.DEFAULT_GOAL := help
PROJECT_ROOT := $(CURDIR)
UV_CACHE_DIR ?= $(PROJECT_ROOT)/.uv-cache
WASM_BINDGEN ?= $(PROJECT_ROOT)/local/tooling/wasm-bindgen-0.2.127/bin/wasm-bindgen
# Keep routine development from recreating multi-gigabyte incremental caches.
CARGO_INCREMENTAL ?= 0
export UV_CACHE_DIR CARGO_INCREMENTAL

.PHONY: help bootstrap env-check deps python-sync format format-check lint rust-lint \
 python-lint test rust-test python-test build rust-build wasm-build wasm-web-generate \
 wasm-web-check boundary-check license-check provenance-check docs-check check \
 pure-build frozen-smoke phase3-validate-registry model-validate

help:
	@echo "OpenShogiAI: make check | build | test | format | lint | pure-build | frozen-smoke"
	@echo "See docs/development.md and docs/status.md. Historical campaigns are closed."

bootstrap:
	./scripts/bootstrap_macos.sh

env-check:
	./scripts/check_environment.sh

deps: python-sync
python-sync:
	uv sync --locked --group dev

format:
	cargo fmt --all
	uv run --frozen ruff format training tests/python scripts/*.py

format-check:
	cargo fmt --all --check
	uv run --frozen ruff format --check training tests/python scripts/*.py

lint: rust-lint python-lint
rust-lint:
	cargo clippy --locked --workspace --all-targets -- -D warnings
python-lint:
	uv run --frozen ruff check training tests/python scripts/*.py

test: rust-test python-test
rust-test:
	cargo test --locked --workspace
python-test:
	uv run --frozen pytest

build: rust-build wasm-web-check
rust-build:
	cargo build --locked --workspace
wasm-build: wasm-web-generate
wasm-web-generate:
	WASM_BINDGEN="$(WASM_BINDGEN)" ./scripts/build_wasm_web.sh write
wasm-web-check:
	WASM_BINDGEN="$(WASM_BINDGEN)" ./scripts/build_wasm_web.sh check
pure-build:
	CARGO_TARGET_DIR="$(PROJECT_ROOT)/target/pure" WASM_BINDGEN="$(WASM_BINDGEN)" ./scripts/build_phase10v_pure.sh
frozen-smoke:
	python3.12 scripts/check_frozen_model.py

boundary-check:
	./scripts/check_repository_boundaries.sh
license-check:
	./scripts/check_license_scope.sh
provenance-check:
	./scripts/check_provenance.sh
docs-check:
	python3.12 scripts/check_docs.py

phase3-validate-registry:
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.data validate-registry --registry configs/data_sources.yaml
model-validate:
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.models validate-config \
		--features configs/features/value_v0.toml --model configs/models/value_v0.toml \
		--training configs/training/value_v0_full_initial.toml

check: deps
	$(MAKE) env-check boundary-check license-check provenance-check docs-check
	$(MAKE) format-check lint
	$(MAKE) test
	$(MAKE) build
