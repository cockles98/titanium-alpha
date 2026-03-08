"""Tests for ``src.config`` — ticker configuration loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import (
    _FALLBACK_BENCHMARK,
    _FALLBACK_CONFIG,
    _FALLBACK_TICKERS,
    load_benchmark,
    load_ticker_config,
    load_tickers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_config(tmp_path: Path) -> Path:
    """Create a valid config file and return its path."""
    cfg = {
        "market": "US",
        "benchmark": "SPY",
        "tickers": ["AAPL", "MSFT", "GOOG"],
        "trading_days_per_year": 252,
        "risk_free_rate": 0.05,
    }
    path = tmp_path / "tickers.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture()
def minimal_config(tmp_path: Path) -> Path:
    """Config with only the required keys."""
    cfg = {
        "market": "BR",
        "benchmark": "^BVSP",
        "tickers": ["PETR4.SA"],
    }
    path = tmp_path / "tickers.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture()
def missing_keys_config(tmp_path: Path) -> Path:
    """Config missing the 'benchmark' key."""
    cfg = {
        "market": "US",
        "tickers": ["AAPL"],
    }
    path = tmp_path / "tickers.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_ticker_config
# ---------------------------------------------------------------------------


class TestLoadTickerConfig:
    """Tests for ``load_ticker_config()``."""

    def test_loads_valid_config(self, valid_config: Path) -> None:
        cfg = load_ticker_config(valid_config)
        assert cfg["market"] == "US"
        assert cfg["benchmark"] == "SPY"
        assert cfg["tickers"] == ["AAPL", "MSFT", "GOOG"]
        assert cfg["trading_days_per_year"] == 252
        assert cfg["risk_free_rate"] == 0.05

    def test_loads_minimal_config(self, minimal_config: Path) -> None:
        cfg = load_ticker_config(minimal_config)
        assert cfg["market"] == "BR"
        assert cfg["benchmark"] == "^BVSP"
        assert cfg["tickers"] == ["PETR4.SA"]

    def test_fallback_when_file_missing(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nope.json"
        cfg = load_ticker_config(nonexistent)
        assert cfg == _FALLBACK_CONFIG

    def test_fallback_is_independent_copy(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nope.json"
        cfg = load_ticker_config(nonexistent)
        cfg["tickers"].append("EXTRA")
        # Original fallback must be unchanged
        assert "EXTRA" not in _FALLBACK_TICKERS

    def test_raises_on_missing_required_keys(
        self, missing_keys_config: Path
    ) -> None:
        with pytest.raises(ValueError, match="missing required keys"):
            load_ticker_config(missing_keys_config)

    def test_extra_keys_preserved(self, tmp_path: Path) -> None:
        cfg = {
            "market": "US",
            "benchmark": "SPY",
            "tickers": ["AAPL"],
            "custom_field": 42,
        }
        path = tmp_path / "tickers.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        result = load_ticker_config(path)
        assert result["custom_field"] == 42


# ---------------------------------------------------------------------------
# load_tickers
# ---------------------------------------------------------------------------


class TestLoadTickers:
    """Tests for ``load_tickers()``."""

    def test_returns_ticker_list(self, valid_config: Path) -> None:
        tickers = load_tickers(valid_config)
        assert tickers == ["AAPL", "MSFT", "GOOG"]

    def test_fallback_returns_default_tickers(self, tmp_path: Path) -> None:
        tickers = load_tickers(tmp_path / "nope.json")
        assert tickers == _FALLBACK_TICKERS


# ---------------------------------------------------------------------------
# load_benchmark
# ---------------------------------------------------------------------------


class TestLoadBenchmark:
    """Tests for ``load_benchmark()``."""

    def test_returns_benchmark(self, valid_config: Path) -> None:
        assert load_benchmark(valid_config) == "SPY"

    def test_fallback_returns_default_benchmark(
        self, tmp_path: Path
    ) -> None:
        assert load_benchmark(tmp_path / "nope.json") == _FALLBACK_BENCHMARK


# ---------------------------------------------------------------------------
# Real config file (config/tickers.json)
# ---------------------------------------------------------------------------


class TestRealConfig:
    """Validate the actual config/tickers.json shipped with the project."""

    _CONFIG_PATH = (
        Path(__file__).resolve().parent.parent / "config" / "tickers.json"
    )

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "config" / "tickers.json").exists(),
        reason="config/tickers.json not present",
    )
    def test_real_config_loads(self) -> None:
        cfg = load_ticker_config(self._CONFIG_PATH)
        assert cfg["market"] == "US"
        assert cfg["benchmark"] == "SPY"
        assert len(cfg["tickers"]) >= 50

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "config" / "tickers.json").exists(),
        reason="config/tickers.json not present",
    )
    def test_benchmark_in_tickers_or_separate(self) -> None:
        cfg = load_ticker_config(self._CONFIG_PATH)
        # SPY can be both benchmark and tradeable
        assert isinstance(cfg["benchmark"], str)

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "config" / "tickers.json").exists(),
        reason="config/tickers.json not present",
    )
    def test_no_duplicate_tickers(self) -> None:
        cfg = load_ticker_config(self._CONFIG_PATH)
        tickers = cfg["tickers"]
        assert len(tickers) == len(set(tickers)), "Duplicate tickers found"
