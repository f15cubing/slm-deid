# Code-quality loop for the De-Id SLM project.
#
# `make check` is the green gate to run before every PR: lint + format-check +
# tests, in the same order CI runs them. This file is only the code-quality gate;
# the ML build / train / eval commands are documented in the README.
#
# On a Mac, MPS ops that aren't implemented fall back to CPU via PYTORCH_ENABLE_MPS_FALLBACK.

PYTEST_ENV := PYTORCH_ENABLE_MPS_FALLBACK=1

.DEFAULT_GOAL := check
.PHONY: check lint format-check format fix test install help

## check: the full code-quality gate (lint + format-check + tests) — run before every PR
check: lint format-check test

## lint: ruff lint (no changes)
lint:
	ruff check .

## format-check: verify formatting without changing files (what CI enforces)
format-check:
	ruff format --check .

## format: apply ruff formatting in place
format:
	ruff format .

## fix: auto-fix lint findings and apply formatting
fix:
	ruff check . --fix
	ruff format .

## test: run the unit + behavioral-check suite
test:
	$(PYTEST_ENV) pytest

## install: create the venv and install deps (one-time)
install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

## help: list targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
