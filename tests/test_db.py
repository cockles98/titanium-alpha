"""Tests for src/utils/db.py — PostgreSQL and ChromaDB connection factories.

All database connections are mocked. No real database is ever contacted.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.db import get_chroma_client, get_postgres_engine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def postgres_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set all required POSTGRES_* env vars with plausible values."""
    env = {
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "titanium",
        "POSTGRES_USER": "quant_user",
        "POSTGRES_PASSWORD": "s3cret",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env


@pytest.fixture()
def chroma_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set all required CHROMA_* env vars with plausible values."""
    env = {
        "CHROMA_HOST": "localhost",
        "CHROMA_PORT": "8000",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env


# ---------------------------------------------------------------------------
# get_postgres_engine — happy path
# ---------------------------------------------------------------------------

class TestGetPostgresEngineHappyPath:
    """Verify engine creation when all env vars are present."""

    @patch("src.utils.db.create_engine")
    @patch("src.utils.db.load_dotenv")
    def test_engine_is_returned(
        self,
        _mock_dotenv: MagicMock,
        mock_create_engine: MagicMock,
        postgres_env: dict[str, str],
    ) -> None:
        """Should return whatever create_engine produces."""
        sentinel_engine = MagicMock(name="engine")
        mock_create_engine.return_value = sentinel_engine

        result = get_postgres_engine()

        assert result is sentinel_engine

    @patch("src.utils.db.create_engine")
    @patch("src.utils.db.load_dotenv")
    def test_pool_pre_ping_is_enabled(
        self,
        _mock_dotenv: MagicMock,
        mock_create_engine: MagicMock,
        postgres_env: dict[str, str],
    ) -> None:
        """pool_pre_ping=True must always be passed to create_engine."""
        get_postgres_engine()

        _, kwargs = mock_create_engine.call_args
        assert kwargs["pool_pre_ping"] is True

    @patch("src.utils.db.create_engine")
    @patch("src.utils.db.load_dotenv")
    def test_default_pool_size_and_overflow(
        self,
        _mock_dotenv: MagicMock,
        mock_create_engine: MagicMock,
        postgres_env: dict[str, str],
    ) -> None:
        """Default pool_size=5 and max_overflow=10 must reach create_engine."""
        get_postgres_engine()

        _, kwargs = mock_create_engine.call_args
        assert kwargs["pool_size"] == 5
        assert kwargs["max_overflow"] == 10

    @patch("src.utils.db.create_engine")
    @patch("src.utils.db.load_dotenv")
    def test_custom_pool_size_and_overflow(
        self,
        _mock_dotenv: MagicMock,
        mock_create_engine: MagicMock,
        postgres_env: dict[str, str],
    ) -> None:
        """Custom pool_size and max_overflow are forwarded to create_engine."""
        get_postgres_engine(pool_size=20, max_overflow=30)

        _, kwargs = mock_create_engine.call_args
        assert kwargs["pool_size"] == 20
        assert kwargs["max_overflow"] == 30

    @patch("src.utils.db.create_engine")
    @patch("src.utils.db.load_dotenv")
    def test_dsn_contains_correct_credentials(
        self,
        _mock_dotenv: MagicMock,
        mock_create_engine: MagicMock,
        postgres_env: dict[str, str],
    ) -> None:
        """The DSN URL passed to create_engine must reflect env vars."""
        get_postgres_engine()

        dsn = mock_create_engine.call_args[0][0]
        # SQLAlchemy URL objects expose these attributes
        assert dsn.host == "localhost"
        assert dsn.port == 5432
        assert dsn.database == "titanium"
        assert dsn.username == "quant_user"
        assert dsn.drivername == "postgresql+psycopg2"


# ---------------------------------------------------------------------------
# get_postgres_engine — missing env vars
# ---------------------------------------------------------------------------

class TestGetPostgresEngineMissingVars:
    """Verify ValueError when required env vars are absent."""

    @patch("src.utils.db.load_dotenv")
    def test_missing_all_vars_raises_value_error(
        self,
        _mock_dotenv: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All five vars missing should raise with all names in the message."""
        # Ensure none of the vars exist
        for var in [
            "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB",
            "POSTGRES_USER", "POSTGRES_PASSWORD",
        ]:
            monkeypatch.delenv(var, raising=False)

        with pytest.raises(ValueError, match="Missing required environment variables"):
            get_postgres_engine()

    @patch("src.utils.db.load_dotenv")
    def test_missing_single_var_raises_with_name(
        self,
        _mock_dotenv: MagicMock,
        postgres_env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Removing just POSTGRES_PASSWORD should mention it in the error."""
        monkeypatch.delenv("POSTGRES_PASSWORD")

        with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
            get_postgres_engine()

    @patch("src.utils.db.load_dotenv")
    def test_empty_string_var_treated_as_missing(
        self,
        _mock_dotenv: MagicMock,
        postgres_env: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An env var set to empty string is treated as missing."""
        monkeypatch.setenv("POSTGRES_HOST", "")

        with pytest.raises(ValueError, match="POSTGRES_HOST"):
            get_postgres_engine()


# ---------------------------------------------------------------------------
# get_chroma_client — happy path
# ---------------------------------------------------------------------------

class TestGetChromaClientHappyPath:
    """Verify ChromaDB client creation with env vars or overrides."""

    @patch("src.utils.db.chromadb")
    @patch("src.utils.db.load_dotenv")
    def test_client_created_from_env(
        self,
        _mock_dotenv: MagicMock,
        mock_chromadb: MagicMock,
        chroma_env: dict[str, str],
    ) -> None:
        """Should call HttpClient with host/port from env vars."""
        sentinel_client = MagicMock(name="chroma_client")
        mock_chromadb.HttpClient.return_value = sentinel_client

        result = get_chroma_client()

        assert result is sentinel_client
        mock_chromadb.HttpClient.assert_called_once_with(
            host="localhost",
            port=8000,
        )

    @patch("src.utils.db.chromadb")
    @patch("src.utils.db.load_dotenv")
    def test_host_port_overrides(
        self,
        _mock_dotenv: MagicMock,
        mock_chromadb: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit host/port args should override env vars."""
        # Set env vars that should be ignored
        monkeypatch.setenv("CHROMA_HOST", "env-host")
        monkeypatch.setenv("CHROMA_PORT", "9999")

        get_chroma_client(host="override-host", port=7777)

        mock_chromadb.HttpClient.assert_called_once_with(
            host="override-host",
            port=7777,
        )

    @patch("src.utils.db.chromadb")
    @patch("src.utils.db.load_dotenv")
    def test_partial_override_host_only(
        self,
        _mock_dotenv: MagicMock,
        mock_chromadb: MagicMock,
        chroma_env: dict[str, str],
    ) -> None:
        """Overriding only host should still read port from env."""
        get_chroma_client(host="custom-host")

        mock_chromadb.HttpClient.assert_called_once_with(
            host="custom-host",
            port=8000,
        )


# ---------------------------------------------------------------------------
# get_chroma_client — missing env vars
# ---------------------------------------------------------------------------

class TestGetChromaClientMissingVars:
    """Verify ValueError when CHROMA_* vars are missing and no override."""

    @patch("src.utils.db.load_dotenv")
    def test_missing_both_vars_raises(
        self,
        _mock_dotenv: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing CHROMA_HOST and CHROMA_PORT without overrides raises."""
        monkeypatch.delenv("CHROMA_HOST", raising=False)
        monkeypatch.delenv("CHROMA_PORT", raising=False)

        with pytest.raises(ValueError, match="Missing CHROMA_HOST"):
            get_chroma_client()

    @patch("src.utils.db.load_dotenv")
    def test_missing_host_only_raises(
        self,
        _mock_dotenv: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing only CHROMA_HOST (port set) should still raise."""
        monkeypatch.delenv("CHROMA_HOST", raising=False)
        monkeypatch.setenv("CHROMA_PORT", "8000")

        with pytest.raises(ValueError, match="Missing CHROMA_HOST"):
            get_chroma_client()

    @patch("src.utils.db.load_dotenv")
    def test_missing_port_only_raises(
        self,
        _mock_dotenv: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing only CHROMA_PORT (host set) should still raise."""
        monkeypatch.setenv("CHROMA_HOST", "localhost")
        monkeypatch.delenv("CHROMA_PORT", raising=False)

        with pytest.raises(ValueError, match="Missing CHROMA_HOST"):
            get_chroma_client()

    @patch("src.utils.db.chromadb")
    @patch("src.utils.db.load_dotenv")
    def test_override_bypasses_missing_env(
        self,
        _mock_dotenv: MagicMock,
        mock_chromadb: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Providing both overrides should work even with no env vars."""
        monkeypatch.delenv("CHROMA_HOST", raising=False)
        monkeypatch.delenv("CHROMA_PORT", raising=False)

        get_chroma_client(host="manual-host", port=1234)

        mock_chromadb.HttpClient.assert_called_once_with(
            host="manual-host",
            port=1234,
        )
