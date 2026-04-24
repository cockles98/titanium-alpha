"""Tests for dashboard Phase 6 — CAPM scatter with OLS regression."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl

from src.dashboard.app import _chart_capm_scatter


def _equity_from_two_streams(
    port_rets: np.ndarray,
    spy_rets: np.ndarray,
    start: date = date(2020, 1, 1),
    start_value: float = 1_000_000.0,
) -> pl.DataFrame:
    """Daily equity DataFrame such that per-day returns match the inputs exactly."""
    assert port_rets.shape == spy_rets.shape
    n = len(port_rets) + 1
    dates = [start + timedelta(days=i) for i in range(n)]
    p = [start_value]
    s = [start_value]
    for pr, sr in zip(port_rets, spy_rets):
        p.append(p[-1] * (1.0 + pr))
        s.append(s[-1] * (1.0 + sr))
    return pl.DataFrame({
        "date": dates,
        "portfolio_value": p,
        "benchmark_value": s,
    })


# ---------------------------------------------------------------------------
# Degenerate / insufficient data
# ---------------------------------------------------------------------------


def test_capm_returns_none_for_too_few_rows():
    df = _equity_from_two_streams(
        port_rets=np.array([0.01, -0.005]),
        spy_rets=np.array([0.02, -0.01]),
    )
    # 3 rows → 2 daily returns; below the 3-return minimum.
    assert _chart_capm_scatter(df) is None


def test_capm_returns_none_when_benchmark_variance_is_zero():
    rng = np.random.default_rng(0)
    port = rng.normal(scale=0.01, size=200)
    spy = np.zeros(200)
    df = _equity_from_two_streams(port, spy)
    assert _chart_capm_scatter(df) is None


# ---------------------------------------------------------------------------
# Numerical correctness of the OLS annotation
# ---------------------------------------------------------------------------


def _annotation_text(fig: go.Figure) -> str:
    return " | ".join(a.text or "" for a in (fig.layout.annotations or []))


def test_capm_recovers_known_beta_and_alpha():
    rng = np.random.default_rng(42)
    spy = rng.normal(0.0, 0.01, size=800)
    beta_true = 0.6
    alpha_daily = 0.0003
    # No noise: perfect line, R² ≈ 1.
    port = alpha_daily + beta_true * spy
    df = _equity_from_two_streams(port, spy)
    fig = _chart_capm_scatter(df)
    assert fig is not None
    text = _annotation_text(fig)
    # Beta is shown to 3 decimals in the annotation.
    assert f"{beta_true:.3f}" in text
    # Annualized alpha appears as a signed percentage.
    expected_alpha_annual_pct = alpha_daily * 252.0 * 100.0  # in %
    # Look for the numeric value ±0.01pp.
    found = False
    for offset in (0, -0.01, 0.01):
        token = f"{expected_alpha_annual_pct + offset:+.2f}%"
        if token.replace("+", "") in text or token in text:
            found = True
            break
    assert found, f"alpha token not found in {text!r}"


def test_capm_r_squared_close_to_one_for_linear_data():
    rng = np.random.default_rng(7)
    spy = rng.normal(0.0, 0.01, size=800)
    port = 0.8 * spy  # deterministic
    df = _equity_from_two_streams(port, spy)
    fig = _chart_capm_scatter(df)
    assert fig is not None
    text = _annotation_text(fig)
    assert "R²" in text
    # For noise-free data, R² is 1.000 to 3 decimals.
    assert "1.000" in text


def test_capm_low_r_squared_for_independent_streams():
    rng = np.random.default_rng(3)
    spy = rng.normal(0.0, 0.01, size=1000)
    port = rng.normal(0.0, 0.01, size=1000)  # independent
    df = _equity_from_two_streams(port, spy)
    fig = _chart_capm_scatter(df)
    assert fig is not None
    text = _annotation_text(fig)
    # Extract R² number between "R² = " and next newline/pipe.
    token = "R² = "
    idx = text.find(token)
    assert idx >= 0
    tail = text[idx + len(token):]
    number = tail.split("<br>")[0].split(" ")[0]
    r2 = float(number)
    assert 0.0 <= r2 < 0.05


# ---------------------------------------------------------------------------
# Figure shape / styling
# ---------------------------------------------------------------------------


def test_capm_has_scatter_trace_and_ols_line():
    rng = np.random.default_rng(11)
    spy = rng.normal(0.0, 0.01, size=400)
    port = 0.7 * spy + rng.normal(0.0, 0.004, size=400)
    df = _equity_from_two_streams(port, spy)
    fig = _chart_capm_scatter(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    modes = [t.mode for t in fig.data]
    assert "markers" in modes
    assert "lines" in modes


def test_capm_uses_dark_theme():
    rng = np.random.default_rng(5)
    spy = rng.normal(0.0, 0.01, size=400)
    port = 0.7 * spy
    df = _equity_from_two_streams(port, spy)
    fig = _chart_capm_scatter(df)
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#0E1117"


def test_capm_axes_use_percent_tickformat():
    rng = np.random.default_rng(6)
    spy = rng.normal(0.0, 0.01, size=400)
    port = 0.7 * spy
    df = _equity_from_two_streams(port, spy)
    fig = _chart_capm_scatter(df)
    assert fig.layout.xaxis.tickformat == ".1%"
    assert fig.layout.yaxis.tickformat == ".1%"


def test_capm_has_zero_reference_lines():
    rng = np.random.default_rng(8)
    spy = rng.normal(0.0, 0.01, size=400)
    port = 0.7 * spy
    df = _equity_from_two_streams(port, spy)
    fig = _chart_capm_scatter(df)
    shapes = list(fig.layout.shapes or [])
    # Expect two crosshair lines through origin (one hline, one vline).
    h_zeros = [sh for sh in shapes if float(sh.y0) == 0.0 and float(sh.y1) == 0.0]
    v_zeros = [sh for sh in shapes if float(sh.x0) == 0.0 and float(sh.x1) == 0.0]
    assert len(h_zeros) >= 1
    assert len(v_zeros) >= 1


def test_capm_annotation_shows_sample_size():
    rng = np.random.default_rng(9)
    spy = rng.normal(0.0, 0.01, size=500)
    port = 0.7 * spy
    df = _equity_from_two_streams(port, spy)
    fig = _chart_capm_scatter(df)
    text = _annotation_text(fig)
    assert "n = 500" in text
