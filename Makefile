.PHONY: setup ingest predict decide test lint run clean

# ── Setup ──────────────────────────────────────────────────
setup:
	poetry install --no-root

# ── Data ingestion ─────────────────────────────────────────
ingest:
	poetry run python -m src.data.ingestion
	poetry run python -m src.data.news_ingestion

# ── Prediction pipeline ───────────────────────────────────
predict:
	poetry run python -m src.models.predict

# ── Decision pipeline ────────────────────────────────────
decide:
	poetry run python -m src.portfolio.decision_engine

# ── Testing ────────────────────────────────────────────────
test:
	poetry run pytest tests/ -v --tb=short

# ── Linting ────────────────────────────────────────────────
lint:
	poetry run ruff check src/ tests/
	poetry run mypy src/

# ── Dashboard ──────────────────────────────────────────────
run:
	poetry run streamlit run src/dashboard/app.py

# ── Cleanup ────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
