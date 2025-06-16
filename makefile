.DEFAULT: help

help:
	@echo "install"
	@echo "        Install laplax and dependencies"
	@echo "install-dev"
	@echo "        Install laplax and development tools"
	@echo "install-notebooks"
	@echo "        Install curvlinops and tools for the tutorial notebooks"
	@echo "install-docs"
	@echo "        Install curvlinops and documentation tools"
	@echo "uninstall"
	@echo "        Uninstall laplax"
	@echo "lint"
	@echo "        Run all linting actions"
	@echo "test"
	@echo "        Run pytest on test and report coverage"
	@echo "ruff-format"
	@echo "        Run ruff format on the project"
	@echo "ruff-format-check"
	@echo "        Check if ruff format would change files"
	@echo "ruff"
	@echo "        Run ruff on the project and fix errors"
	@echo "ruff-check"
	@echo "        Run ruff check on the project without fixing errors"

.PHONY: install

install:
	@uv pip install -e .

.PHONY: install-dev

install-dev:
	@uv pip install -e '.[dev]'

.PHONY: install-notebooks

install-notebooks:
	@uv pip install -e '.[notebooks]'

.PHONY: install-docs

install-docs:
	@uv pip install -e '.[docs]'

.PHONY: uninstall

uninstall:
	@uv pip uninstall laplax

.PHONY: test

test:
	@pytest -vx --cov=laplax

.PHONY: ruff-format ruff-format-check

ruff-format:
	@ruff format .

ruff-format-check:
	@ruff format --check .

.PHONY: ruff-check

ruff:
	@ruff check . --fix

ruff-check:
	@ruff check .

.PHONY: lint

lint:
	make ruff-format-check
	make ruff-check
