"""Ticker and market configuration loader.

Loads ticker universe, benchmark, and market parameters from a JSON
config file.  Falls back to hardcoded defaults when the config file
is missing (backward compatibility with the original 4-ticker setup).

Usage::

    from src.config import load_tickers, load_benchmark, load_ticker_config

    tickers = load_tickers()           # ["AAPL", "MSFT", ...]
    benchmark = load_benchmark()       # "SPY"
    config = load_ticker_config()      # full dict
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "tickers.json"

_FALLBACK_TICKERS: list[str] = ["SPY", "NVDA", "AAPL", "QQQ"]
_FALLBACK_BENCHMARK: str = "SPY"
_FALLBACK_CONFIG: dict[str, Any] = {
    "market": "US",
    "benchmark": _FALLBACK_BENCHMARK,
    "tickers": _FALLBACK_TICKERS,
    "trading_days_per_year": 252,
    "risk_free_rate": 0.05,
}

_REQUIRED_KEYS = {"tickers", "benchmark", "market"}


def load_ticker_config(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the full ticker configuration from a JSON file.

    Args:
        path: Path to the JSON config file.  Defaults to
            ``config/tickers.json`` relative to the project root.

    Returns:
        Configuration dictionary with at least ``tickers``,
        ``benchmark``, and ``market`` keys.

    Raises:
        ValueError: If the config file exists but is missing required
            keys (``tickers``, ``benchmark``, ``market``).
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        logger.warning(
            "Config file not found at {}; using fallback defaults",
            config_path,
        )
        return copy.deepcopy(_FALLBACK_CONFIG)

    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    missing = _REQUIRED_KEYS - set(config.keys())
    if missing:
        raise ValueError(
            f"Config file {config_path} is missing required keys: {missing}"
        )

    logger.info(
        "Loaded config from {} | market={} | {} tickers | benchmark={}",
        config_path,
        config["market"],
        len(config["tickers"]),
        config["benchmark"],
    )
    return config


def load_tickers(path: str | Path | None = None) -> list[str]:
    """Load the list of tradeable tickers from the config file.

    Args:
        path: Path to the JSON config file.

    Returns:
        List of ticker symbols (e.g. ``["AAPL", "MSFT", ...]``).
    """
    return load_ticker_config(path)["tickers"]


def load_benchmark(path: str | Path | None = None) -> str:
    """Load the benchmark ticker from the config file.

    Args:
        path: Path to the JSON config file.

    Returns:
        Benchmark ticker symbol (e.g. ``"SPY"``).
    """
    return load_ticker_config(path)["benchmark"]
