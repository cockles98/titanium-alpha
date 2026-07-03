"""Tests for dashboard Phase 4 — rolling beta/alpha/correlation vs SPY."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl

from src.dashboard.app import (
    _chart_rolling_market_relationship,
    _rolling_regression,
)


def _equity_from_two_streams(
    port_rets: np.ndarray, spy_rets: np.ndarray, start: date = date(2020, 1, 1)
) -> pl.DataFrame:
    assert port_rets.shape == spy_rets.shape
    n = len(port_rets) + 1
    dates = [start + timedelta(days=i) for i in range(n)]
    p = [1_000_000.0]
    s = [1_000_000.0]
    for pr, sr in zip(port_rets, spy_rets):
        p.append(p[-1] * (1.0 + pr))
        s.append(s[-1] * (1.0 + sr))
    return pl.DataFrame({
        "date": dates,
        "portfolio_value": p,
        "benchmark_value": s,
    })


# ---------------------------------------------------------------------------
# _rolling_regression
# ---------------------------------------------------------------------------


def test_rolling_regression_recovers_known_beta_and_zero_alpha():
    rng = np.random.default_rng(42)
    spy = rng.normal(0.0, 0.01, size=1000)
    # Jensen's alpha is 0 when (port - rf) = β · (spy - rf), i.e.
    # port = β · spy + (1 - β) · rf_daily.
    beta_true = 1.5
    rf_daily = (1.0 + 0.05) ** (1.0 / 252) - 1.0
    port = beta_true * spy + (1.0 - beta_true) * rf_daily
    b, a, c = _rolling_regression(port, spy, window=126)
    np.testing.assert_allclose(np.nanmean(b), beta_true, rtol=1e-9)
    np.testing.assert_allclose(np.nanmean(a), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.nanmean(c), 1.0, atol=1e-9)


def test_rolling_regression_correlation_zero_for_independent_streams():
    rng = np.random.default_rng(7)
    spy = rng.normal(size=1000)
    port = rng.normal(size=1000)  # independent
    _b, _a, c = _rolling_regression(port, spy, window=200)
    # Rolling correlations should cluster around 0 on average.
    mean_c = np.nanmean(c)
    assert abs(mean_c) < 0.15


def test_rolling_regression_alpha_is_annualized():
    rng = np.random.default_rng(5)
    spy = rng.normal(0.0, 0.01, size=500)
    # Build port so Jensen's α = 0.0005 (daily) by construction.
    # raw_intercept = α_J + (1 - β) · rf_daily.
    daily_alpha_jensen = 0.0005
    beta_true = 0.8
    rf_daily = (1.0 + 0.05) ** (1.0 / 252) - 1.0
    raw_intercept = daily_alpha_jensen + (1.0 - beta_true) * rf_daily
    port = beta_true * spy + raw_intercept
    _b, a, _c = _rolling_regression(port, spy, window=126)
    expected_annual = daily_alpha_jensen * 252.0
    np.testing.assert_allclose(np.nanmean(a), expected_annual, rtol=1e-6)


def test_rolling_regression_warmup_is_nan():
    rng = np.random.default_rng(9)
    spy = rng.normal(size=300)
    port = 0.7 * spy
    b, a, c = _rolling_regression(port, spy, window=60)
    # Positions 0..window-2 must be NaN.
    assert np.all(np.isnan(b[:59]))
    assert np.all(np.isnan(a[:59]))
    assert np.all(np.isnan(c[:59]))
    # Position window-1 onwards must be finite (at least one).
    assert np.isfinite(b[59])


def test_rolling_regression_returns_full_nan_for_too_short_input():
    port = np.array([0.01, -0.02, 0.005])
    spy = np.array([0.02, -0.01, 0.003])
    b, a, c = _rolling_regression(port, spy, window=60)
    assert b.shape == port.shape
    assert np.all(np.isnan(b))
    assert np.all(np.isnan(a))
    assert np.all(np.isnan(c))


def test_rolling_regression_handles_zero_variance_benchmark():
    port = np.random.default_rng(1).normal(size=200)
    spy = np.zeros(200)  # degenerate
    b, a, c = _rolling_regression(port, spy, window=60)
    # Beta is undefined when Var(SPY)=0 → all NaN.
    assert np.all(np.isnan(b))


# ---------------------------------------------------------------------------
# _chart_rolling_market_relationship
# ---------------------------------------------------------------------------


def test_chart_returns_none_for_insufficient_data():
    rng = np.random.default_rng(0)
    # Only 50 rets, window=126 → should return None.
    port_rets = rng.normal(scale=0.01, size=50)
    spy_rets = rng.normal(scale=0.01, size=50)
    df = _equity_from_two_streams(port_rets, spy_rets)
    assert _chart_rolling_market_relationship(df, window=126) is None


def test_chart_returns_figure_with_three_traces():
    rng = np.random.default_rng(3)
    spy_rets = rng.normal(scale=0.01, size=600)
    port_rets = 0.7 * spy_rets + rng.normal(scale=0.003, size=600)
    df = _equity_from_two_streams(port_rets, spy_rets)
    fig = _chart_rolling_market_relationship(df, window=126)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3


def test_chart_has_reference_hlines_at_1_0_0():
    rng = np.random.default_rng(4)
    spy_rets = rng.normal(scale=0.01, size=600)
    port_rets = 0.6 * spy_rets
    df = _equity_from_two_streams(port_rets, spy_rets)
    fig = _chart_rolling_market_relationship(df, window=126)
    # add_hline creates layout shapes.
    shapes = list(fig.layout.shapes or [])
    ys = {float(sh.y0) for sh in shapes}
    assert 1.0 in ys
    assert 0.0 in ys
    assert len(shapes) >= 3


def test_chart_subplot_titles_show_mean_values():
    rng = np.random.default_rng(8)
    spy_rets = rng.normal(scale=0.01, size=600)
    port_rets = 0.8 * spy_rets + 0.0005  # alpha
    df = _equity_from_two_streams(port_rets, spy_rets)
    fig = _chart_rolling_market_relationship(df, window=126)
    annotations = [a.text for a in (fig.layout.annotations or [])]
    joined = " | ".join(annotations)
    assert "Rolling Beta" in joined
    assert "Rolling Alpha" in joined
    assert "Rolling Correlation" in joined
    assert "mean" in joined.lower()


def test_chart_uses_dark_theme():
    rng = np.random.default_rng(10)
    spy_rets = rng.normal(scale=0.01, size=500)
    port_rets = 0.7 * spy_rets
    df = _equity_from_two_streams(port_rets, spy_rets)
    fig = _chart_rolling_market_relationship(df, window=126)
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#0E1117"


def test_chart_hover_templates_present():
    rng = np.random.default_rng(11)
    spy_rets = rng.normal(scale=0.01, size=500)
    port_rets = 0.7 * spy_rets
    df = _equity_from_two_streams(port_rets, spy_rets)
    fig = _chart_rolling_market_relationship(df, window=126)
    hovers = [t.hovertemplate or "" for t in fig.data]
    joined = "|".join(hovers)
    assert "β" in joined
    assert "α" in joined
    assert "ρ" in joined
