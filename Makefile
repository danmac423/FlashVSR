# ── install ───────────────────────────────────────────────────────────────────
.PHONY: install
install:
	uv sync --all-groups

# ── lint / format ─────────────────────────────────────────────────────────────
.PHONY: lint format

lint:
	uv run ruff check src benchmarks tests

format:
	uv run ruff format src benchmarks tests

# ── tests ─────────────────────────────────────────────────────────────────────
.PHONY: test
test:
	uv run pytest tests/ -v

# ── clean ─────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	find . -type d -name __pycache__ -not -path "./.venv/*" | xargs rm -rf
	find . -type d -name "*.egg-info" -not -path "./.venv/*" | xargs rm -rf
