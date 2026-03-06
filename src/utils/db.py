"""Database connection utilities for PostgreSQL and ChromaDB.

Provides factory functions that read configuration from environment
variables and return properly configured client instances.
"""

from __future__ import annotations

import os

import chromadb
from chromadb.api import ClientAPI
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL


def get_postgres_engine(
    *,
    pool_size: int = 5,
    max_overflow: int = 10,
    echo: bool = False,
) -> Engine:
    """Create a SQLAlchemy engine for PostgreSQL with connection pooling.

    Args:
        pool_size: Number of persistent connections in the pool.
        max_overflow: Max additional connections beyond pool_size.
        echo: If True, SQLAlchemy logs all SQL statements.

    Returns:
        A configured SQLAlchemy Engine with connection pooling.

    Raises:
        ValueError: If any required POSTGRES_* environment variable
            is missing or empty.
    """
    load_dotenv()

    required_vars = [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]
    config: dict[str, str] = {}
    missing: list[str] = []

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing.append(var)
        else:
            config[var] = value

    if missing:
        raise ValueError(
            f"Missing required environment variables for PostgreSQL: {missing}. "
            "Check your .env file against .env.example."
        )

    dsn = URL.create(
        drivername="postgresql+psycopg2",
        username=config["POSTGRES_USER"],
        password=config["POSTGRES_PASSWORD"],
        host=config["POSTGRES_HOST"],
        port=int(config["POSTGRES_PORT"]),
        database=config["POSTGRES_DB"],
    )

    engine = create_engine(
        dsn,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        echo=echo,
    )
    logger.info(
        "PostgreSQL engine created | host={} port={} db={}",
        config["POSTGRES_HOST"],
        config["POSTGRES_PORT"],
        config["POSTGRES_DB"],
    )
    return engine


def get_chroma_client(
    *,
    host: str | None = None,
    port: int | None = None,
) -> ClientAPI:
    """Create a ChromaDB HTTP client from environment variables.

    Args:
        host: Override for CHROMA_HOST env var. Useful for testing.
        port: Override for CHROMA_PORT env var. Useful for testing.

    Returns:
        A configured chromadb.HttpClient instance.

    Raises:
        ValueError: If CHROMA_HOST or CHROMA_PORT is missing and
            no override was provided.
    """
    load_dotenv()

    resolved_host = host or os.getenv("CHROMA_HOST")
    raw_port = port or os.getenv("CHROMA_PORT")

    if not resolved_host or not raw_port:
        raise ValueError(
            "Missing CHROMA_HOST and/or CHROMA_PORT. "
            "Set them in .env or pass as arguments."
        )

    client = chromadb.HttpClient(
        host=resolved_host,
        port=int(raw_port),
    )
    logger.info(
        "ChromaDB client created | host={} port={}",
        resolved_host,
        raw_port,
    )
    return client
