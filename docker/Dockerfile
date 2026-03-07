FROM python:3.12-slim AS base

WORKDIR /app

# System deps for psycopg2 and scipy
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy dependency files first (layer caching)
COPY pyproject.toml poetry.lock* ./

# Install production deps only
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --only main --no-interaction

# Copy source code
COPY src/ src/
COPY data/ data/

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/dashboard/app.py", \
    "--server.address=0.0.0.0"]
