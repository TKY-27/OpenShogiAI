SHELL := /bin/sh
.DEFAULT_GOAL := help

PROJECT_ROOT := $(CURDIR)
UV_CACHE_DIR ?= $(PROJECT_ROOT)/.uv-cache
WASM_BINDGEN ?= $(PROJECT_ROOT)/local/tooling/wasm-bindgen-0.2.127/bin/wasm-bindgen

export UV_CACHE_DIR

.PHONY: help bootstrap env-check deps python-sync \
	format format-check lint rust-lint python-lint \
	test rust-test python-test \
	build rust-build wasm-build wasm-web-generate wasm-web-check \
	boundary-check license-check provenance-check \
	phase1-perft phase1-fuzz \
	phase2-clean-worktree phase2-usi-test phase2-bench phase2-arena check \
	phase3-clean-worktree phase3-validate-registry phase3-dry-run \
	phase3-build-cli phase3-acquire phase3-normalize \
	phase3-build-opening phase3-export-opening \
	model-validate model-describe train-overfit train-smoke model-compare \
	feature-ablation phase4-migrate-label-manifest phase5-arena \
	phase5-arena-verify phase6-validate-config \
	phase7-validate-config phase7-plan phase7-commands phase7-prepare-games \
	phase7-analyze phase7-curate phase7-verify

PHASE2_ARENA_DIR ?= artifacts/phase2-arena
PHASE2_GIT_COMMIT ?= $(shell git rev-parse --verify HEAD)
PHASE3_REGISTRY ?= configs/data_sources.yaml
PHASE3_SOURCE ?= aobazero-no-noise
PHASE3_LIMIT ?= 100
PHASE3_RAW_ROOT ?= data/raw/phase3/aobazero-no-noise
PHASE3_PROCESSED_ROOT ?= data/processed/phase3
PHASE3_DATASET_ID ?= aobazero-no-noise-pd-sample100
PHASE3_SPLIT_SALT ?= open-shogi-ai-phase3-v1-20260803
PHASE3_CLI ?= target/release/open-shogi-cli
ENGINE_BUILD_RECEIPT ?= local/build-receipts/open-shogi-cli.pointer.json
PHASE3_DATASET_DIR ?= $(PHASE3_PROCESSED_ROOT)/$(PHASE3_DATASET_ID)
PHASE3_OPENING_DB ?= $(PHASE3_PROCESSED_ROOT)/opening/$(PHASE3_DATASET_ID).sqlite3
PHASE3_OPENING_EXPORT ?= $(PHASE3_PROCESSED_ROOT)/opening/$(PHASE3_DATASET_ID).jsonl.gz
PHASE3_OPENING_MIN_COUNT ?= 1
PHASE4_LABELS ?= artifacts/phase4/teacher/labels-v2/labels.jsonl
PHASE4_LABEL_MANIFEST ?= artifacts/phase4/teacher/labels-v2/manifest.json
PHASE4_LEGACY_LABEL_MANIFEST_SHA256 ?= cc21486da8dc92e5c192e40e66bb81097c364899bf8aa44169162ac42b266238
PHASE4_TEACHER_CONFIG ?= configs/teacher/apery-v2.0.0.yaml
PHASE4_BENCHMARK_REPORT ?= artifacts/phase4/teacher/benchmark-v3.json
PHASE4_POSITIONS ?= $(PHASE3_DATASET_DIR)/positions-00000.jsonl.gz
PHASE4_DATASET_MANIFEST ?= $(PHASE3_DATASET_DIR)/manifest.json
PHASE4_FEATURES ?= configs/features/value_v0.toml
PHASE4_MODEL ?= configs/models/value_v0.toml
PHASE4_OVERFIT_OUTPUT ?= artifacts/phase4/models/value-v0-overfit-reproduction
PHASE4_SMOKE_OUTPUT ?= artifacts/phase4/models/value-v0-smoke-reproduction
PHASE4_LEFT_CHECKPOINT ?=
PHASE4_RIGHT_CHECKPOINT ?=
PHASE4_CHECKPOINT ?=
PHASE5_GIT_COMMIT ?= $(shell git rev-parse --verify HEAD)
PHASE7_CONFIG ?= configs/evaluation/phase7_official.toml
PHASE7_REGISTRY ?= artifacts/phase6/registry-r4/model-registry.json
PHASE7_OUTPUT_ROOT ?= artifacts/phase7/bounded-official-v4
PHASE7_PLAN ?= $(PHASE7_OUTPUT_ROOT)/plan.json
PHASE7_REPORT ?= $(PHASE7_OUTPUT_ROOT)/evaluation-report.json
PHASE7_HARD_EXAMPLES ?= $(PHASE7_OUTPUT_ROOT)/hard-examples.json
PHASE7_GIT_COMMIT ?= $(shell git rev-parse --verify HEAD)

help:
	@echo "OpenShogiAI development commands"
	@echo "  make bootstrap     Inspect macOS tools and print missing setup commands"
	@echo "  make check         Run locked sync, audits, formatting, linting, tests, and builds"
	@echo "  make format        Apply Rust and Python formatting"
	@echo "  make lint          Run Clippy and Ruff"
	@echo "  make test          Run Rust and Python tests"
	@echo "  make build         Build Rust and verify deterministic Wasm bindings"
	@echo "  make phase1-perft  Traverse the initial legal-move tree to depth 2"
	@echo "  make phase1-fuzz   Run 100 deterministic random legal games"
	@echo "  make phase2-usi-test  Exercise asynchronous USI search and stop"
	@echo "  make phase2-bench     Run the fixed-position search benchmark"
	@echo "  make phase2-arena     Run 20 fixed-seed search-vs-random games"
	@echo "  make phase3-validate-registry  Validate audited source policy and catalogs"
	@echo "  make phase3-dry-run   Print the exact 100-object plan without network or writes"
	@echo "  make phase3-acquire   Acquire the approved sample into ignored raw storage"
	@echo "  make phase3-normalize Normalize the sample with the release Rust exporter"
	@echo "  make phase3-build-opening  Build the train-only opening SQLite database"
	@echo "  make phase3-export-opening Export deterministic opening JSONL"
	@echo "  make model-validate Validate the closed Phase 4 feature/model/training schemas"
	@echo "  make model-describe Show feature schema, parameters, operations, and model sizes"
	@echo "  make train-overfit Run the bounded 32-position overfit proof"
	@echo "  make train-smoke   Run the bounded two-batch training smoke proof"
	@echo "  make phase4-migrate-label-manifest  Re-audit v1 labels and publish bound v2"
	@echo "  make model-compare Compare PHASE4_LEFT_CHECKPOINT and PHASE4_RIGHT_CHECKPOINT"
	@echo "  make feature-ablation Evaluate PHASE4_CHECKPOINT with each feature group removed"
	@echo "  make phase5-arena  Run or resume the frozen five-by-forty comparison matrix"
	@echo "  make phase5-arena-verify Recompute results and Rust-replay all 200 CSA files"
	@echo "  make phase6-validate-config Validate the bounded generation and promotion policy"
	@echo "  make phase7-validate-config Validate the fixed developer-evaluation contract"
	@echo "  make phase7-plan     Publish the fixed two-game human/champion plan"
	@echo "  make phase7-commands Print the exact two interactive play commands as JSON"
	@echo "  make phase7-prepare-games Rust-replay both completed human-game artifacts"
	@echo "  make phase7-analyze  Teacher-analyze all recorded move positions"
	@echo "  make phase7-curate   Publish quarantined hard-example candidates"
	@echo "  make phase7-verify   Revalidate the complete Phase 7 evidence chain"
	@echo "  make wasm-build      Build the Wasm engine and regenerate interface bindings"
	@echo "  make wasm-web-check  Verify committed Wasm bindings are reproducible"

bootstrap:
	./scripts/bootstrap_macos.sh

env-check:
	./scripts/check_environment.sh

deps: python-sync

python-sync:
	uv sync --locked --group dev

format:
	cargo fmt --all
	uv run ruff format training tests/python

format-check:
	cargo fmt --all --check
	uv run ruff format --check training tests/python

lint: rust-lint python-lint

rust-lint:
	cargo clippy --locked --workspace --all-targets -- -D warnings

python-lint:
	uv run ruff check training tests/python

test: rust-test python-test

rust-test:
	cargo test --locked --workspace

python-test:
	uv run pytest

build: rust-build wasm-web-check

rust-build:
	cargo build --locked --workspace

wasm-build: wasm-web-generate

wasm-web-generate:
	WASM_BINDGEN="$(WASM_BINDGEN)" ./scripts/build_wasm_web.sh write

wasm-web-check:
	WASM_BINDGEN="$(WASM_BINDGEN)" ./scripts/build_wasm_web.sh check

boundary-check:
	./scripts/check_repository_boundaries.sh

license-check:
	./scripts/check_license_scope.sh

provenance-check:
	./scripts/check_provenance.sh

phase1-perft:
	cargo run --locked -p open-shogi-cli -- perft --depth 2

phase1-fuzz:
	cargo run --locked -p open-shogi-cli -- random-games --games 100 --max-plies 512 --seed 5715919092156094513

phase2-usi-test:
	cargo test --locked -p open-shogi-usi session::tests::asynchronous_search_emits_exactly_one_bestmove

phase2-clean-worktree:
	@test -z "$$(git status --porcelain --untracked-files=normal)" || \
		(echo "refusing to attribute Phase 2 evidence to HEAD while the worktree is dirty" && exit 1)

phase2-bench: phase2-clean-worktree
	cargo run --locked --release -p open-shogi-cli -- bench --nodes 25000

phase2-arena: phase2-clean-worktree
	cargo run --locked --release -p open-shogi-cli -- arena \
		--games 20 --player-a search --player-b random --a-depth 6 \
		--nodes 2000 --max-plies 128 --seed 20260729 \
		--git-commit "$(PHASE2_GIT_COMMIT)" --output-dir "$(PHASE2_ARENA_DIR)"

phase3-clean-worktree:
	@test -z "$$(git status --porcelain --untracked-files=normal)" || \
		(echo "refusing to attribute Phase 3 evidence to HEAD while the worktree is dirty" && exit 1)

phase3-validate-registry:
	PYTHONPATH=training uv run python -m open_shogi_training.data validate-registry \
		--registry "$(PHASE3_REGISTRY)"

phase3-dry-run:
	PYTHONPATH=training uv run python -m open_shogi_training.data dry-run \
		--registry "$(PHASE3_REGISTRY)" --source "$(PHASE3_SOURCE)" \
		--limit "$(PHASE3_LIMIT)"

phase3-build-cli: phase3-clean-worktree
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.selfplay.engine_receipt \
		create --repository-root "$(PROJECT_ROOT)" --git-commit "$$(git rev-parse --verify HEAD)" \
		--engine "$(PHASE3_CLI)" --output "$(ENGINE_BUILD_RECEIPT)"

phase3-acquire: phase3-clean-worktree
	PYTHONPATH=training uv run python -m open_shogi_training.data acquire \
		--registry "$(PHASE3_REGISTRY)" --source "$(PHASE3_SOURCE)" \
		--limit "$(PHASE3_LIMIT)" --sample-only --output "$(PHASE3_RAW_ROOT)"

phase3-normalize: phase3-clean-worktree phase3-build-cli
	PYTHONPATH=training uv run python -m open_shogi_training.data normalize \
		--registry "$(PHASE3_REGISTRY)" --source "$(PHASE3_SOURCE)" \
		--manifest "$(PHASE3_RAW_ROOT)/manifest.jsonl" \
		--acquisition-root "$(PHASE3_RAW_ROOT)" \
		--processed-root "$(PHASE3_PROCESSED_ROOT)" \
		--dataset-id "$(PHASE3_DATASET_ID)" --split-salt "$(PHASE3_SPLIT_SALT)" \
		--max-games "$(PHASE3_LIMIT)" --max-positions 100000 \
		--terminal-tail-positions 8 --cli "$(PHASE3_CLI)"

phase3-build-opening: phase3-clean-worktree
	PYTHONPATH=training uv run python -m open_shogi_training.data build-opening \
		--positions "$(PHASE3_DATASET_DIR)/positions-00000.jsonl.gz" \
		--output "$(PHASE3_OPENING_DB)" \
		--provenance "$(PHASE3_DATASET_DIR)/manifest.json" \
		--max-input-rows 100000 --max-input-bytes 67108864 \
		--max-input-uncompressed-bytes 268435456

phase3-export-opening: phase3-clean-worktree
	PYTHONPATH=training uv run python -m open_shogi_training.data export-opening \
		--database "$(PHASE3_OPENING_DB)" --output "$(PHASE3_OPENING_EXPORT)" \
		--min-count "$(PHASE3_OPENING_MIN_COUNT)"

model-validate:
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.models validate-config \
		--features "$(PHASE4_FEATURES)" --model "$(PHASE4_MODEL)" \
		--training configs/training/value_v0_full_initial.toml

model-describe:
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.models describe \
		--features "$(PHASE4_FEATURES)" --model "$(PHASE4_MODEL)" \
		--training configs/training/value_v0_full_initial.toml

train-overfit:
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.models overfit \
		--features "$(PHASE4_FEATURES)" --model "$(PHASE4_MODEL)" \
		--training configs/training/value_v0_overfit.toml \
		--labels "$(PHASE4_LABELS)" --positions "$(PHASE4_POSITIONS)" \
		--label-manifest "$(PHASE4_LABEL_MANIFEST)" \
		--dataset-manifest "$(PHASE4_DATASET_MANIFEST)" \
		--output-dir "$(PHASE4_OVERFIT_OUTPUT)" --examples 32

train-smoke:
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.models smoke \
		--features "$(PHASE4_FEATURES)" --model "$(PHASE4_MODEL)" \
		--training configs/training/value_v0_smoke.toml \
		--labels "$(PHASE4_LABELS)" --positions "$(PHASE4_POSITIONS)" \
		--label-manifest "$(PHASE4_LABEL_MANIFEST)" \
		--dataset-manifest "$(PHASE4_DATASET_MANIFEST)" \
		--output-dir "$(PHASE4_SMOKE_OUTPUT)"

model-compare:
	@test -n "$(PHASE4_LEFT_CHECKPOINT)" -a -n "$(PHASE4_RIGHT_CHECKPOINT)" || \
		(echo "set PHASE4_LEFT_CHECKPOINT and PHASE4_RIGHT_CHECKPOINT" && exit 1)
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.models compare \
		--left "$(PHASE4_LEFT_CHECKPOINT)" --right "$(PHASE4_RIGHT_CHECKPOINT)" \
		--labels "$(PHASE4_LABELS)" --positions "$(PHASE4_POSITIONS)" \
		--label-manifest "$(PHASE4_LABEL_MANIFEST)" \
		--dataset-manifest "$(PHASE4_DATASET_MANIFEST)"

feature-ablation:
	@test -n "$(PHASE4_CHECKPOINT)" || (echo "set PHASE4_CHECKPOINT" && exit 1)
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.models feature-ablation \
		--checkpoint "$(PHASE4_CHECKPOINT)" --labels "$(PHASE4_LABELS)" \
		--label-manifest "$(PHASE4_LABEL_MANIFEST)" \
		--positions "$(PHASE4_POSITIONS)" --dataset-manifest "$(PHASE4_DATASET_MANIFEST)"

phase4-migrate-label-manifest:
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.labeling \
		migrate-label-manifest-v2 --config "$(PHASE4_TEACHER_CONFIG)" \
		--project-root "$(PROJECT_ROOT)" --positions "$(PHASE4_POSITIONS)" \
		--dataset-manifest "$(PHASE4_DATASET_MANIFEST)" \
		--benchmark-report "$(PHASE4_BENCHMARK_REPORT)" \
		--output-dir "$$(dirname "$(PHASE4_LABEL_MANIFEST)")" \
		--expected-legacy-manifest-sha256 "$(PHASE4_LEGACY_LABEL_MANIFEST_SHA256)"

phase5-arena: phase3-build-cli
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.models arena-run \
		--git-commit "$(PHASE5_GIT_COMMIT)"

phase5-arena-verify: phase3-build-cli
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.models arena-verify \
		--git-commit "$(PHASE5_GIT_COMMIT)"

phase6-validate-config:
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.selfplay validate-config \
		--selfplay-config configs/selfplay/phase6_smoke.toml \
		--promotion-policy configs/generation/phase6_promotion.toml

phase7-validate-config:
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluation \
		--config "$(PHASE7_CONFIG)" validate-config

phase7-plan: phase3-build-cli
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluation \
		--config "$(PHASE7_CONFIG)" plan --registry "$(PHASE7_REGISTRY)" \
		--output-root "$(PHASE7_OUTPUT_ROOT)" --git-commit "$(PHASE7_GIT_COMMIT)"

phase7-commands:
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluation \
		--config "$(PHASE7_CONFIG)" commands --plan "$(PHASE7_PLAN)"

phase7-prepare-games:
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluation \
		--config "$(PHASE7_CONFIG)" prepare-games --plan "$(PHASE7_PLAN)"

phase7-analyze:
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluation \
		--config "$(PHASE7_CONFIG)" analyze --plan "$(PHASE7_PLAN)"

phase7-curate:
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluation \
		--config "$(PHASE7_CONFIG)" curate --report "$(PHASE7_REPORT)"

phase7-verify:
	PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluation \
		--config "$(PHASE7_CONFIG)" verify --report "$(PHASE7_REPORT)" \
		--hard-examples "$(PHASE7_HARD_EXAMPLES)"

check: deps
	$(MAKE) env-check
	$(MAKE) boundary-check
	$(MAKE) license-check
	$(MAKE) provenance-check
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) build
