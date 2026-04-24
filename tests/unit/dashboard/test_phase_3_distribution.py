"""Tests for dashboard Phase 3 — return distribution + QQ plot."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from src.dashboard.app import (
    _chart_return_distribution,
    _portfolio_daily_returns,
    _return_distribution_stats,
)


def _equity_from_returns(
    rets: np.ndarray, start: date = date(2024, 1, 1), start_val: float = 1_000_000.0
) -> pl.DataFrame:
    """Build an equity DataFrame from a return series; first row is the base."""
    n = len(rets) + 1
    dates = [start + timedelta(days=i) for i in range(n)]
    values = [start_val]
    for r in rets:
        values.append(values[-1] * (1.0 + r))
    return pl.DataFrame({
        "date": dates,
        "portfolio_value": values,
        "benchmark_value": values,
    })


# ---------------------------------------------------------------------------
# _portfolio_daily_returns
# ---------------------------------------------------------------------------


def test_daily_returns_length_is_n_minus_one():
    df = _equity_from_returns(np.array([0.01, -0.02, 0.015]))
    rets = _portfolio_daily_returns(df)
    assert len(rets) == 3


def test_daily_returns_match_input():
    input_rets = np.array([0.01, -0.02, 0.015, 0.005])
    df = _equity_from_returns(input_rets)
    rets = _portfolio_daily_returns(df)
    np.testing.assert_allclose(rets, input_rets, rtol=1e-9)


# ---------------------------------------------------------------------------
# _return_distribution_stats
# ---------------------------------------------------------------------------


def test_stats_normal_distribution_has_low_skew_and_kurt():
    rng = np.random.default_rng(42)
    rets = rng.normal(loc=0.0005, scale=0.01, size=5000).tolist()
    s = _return_distribution_stats(rets)
    # Normal should have skew ≈ 0, excess kurt ≈ 0.
    assert abs(s["skew"]) < 0.2
    assert abs(s["kurt"]) < 0.4
    assert s["mu"] == pytest.approx(0.0005, abs=5e-4)
    assert s["sigma"] == pytest.approx(0.01, abs=5e-4)


def test_stats_fat_tailed_t_distribution_flags_high_kurtosis():
    rng = np.random.default_rng(0)
    # Student's t with df=4 has theoretical excess kurtosis = 6/(df-4) → ∞;
    # with df=5 it's 6. Use df=5 and assert >> 0.
    import scipy.stats as sstats
    rets = sstats.t.rvs(df=5, size=10_000, random_state=rng) * 0.01
    s = _return_distribution_stats(rets.tolist())
    assert s["kurt"] > 2.0  # clearly leptokurtic


def test_stats_var5_matches_5th_percentile():
    rng = np.random.default_rng(7)
    rets = rng.normal(size=2000).tolist()
    s = _return_distribution_stats(rets)
    expected = float(np.quantile(rets, 0.05))
    assert s["var5"] == pytest.approx(expected, abs=1e-12)


def test_stats_cvar5_is_mean_of_tail():
    rng = np.random.default_rng(13)
    rets = rng.normal(size=3000).tolist()
    arr = np.asarray(rets)
    s = _return_distribution_stats(rets)
    tail = arr[arr <= s["var5"]]
    assert s["cvar5"] == pytest.approx(float(tail.mean()), rel=1e-12)


def test_stats_cvar5_not_greater_than_var5_for_losing_tail():
    """CVaR (expected loss in tail) should be at least as negative as VaR."""
    rng = np.random.default_rng(3)
    rets = rng.normal(size=1500).tolist()
    s = _return_distribution_stats(rets)
    assert s["cvar5"] <= s["var5"] + 1e-12


def test_stats_negative_skew_detected_for_left_skewed_data():
    # Construct left-skewed returns: mostly small positive, occasional big negative.
    rng = np.random.default_rng(5)
    normal = rng.normal(loc=0.001, scale=0.005, size=900)
    big_negs = rng.uniform(-0.08, -0.03, size=100)
    rets = np.concatenate([normal, big_negs]).tolist()
    s = _return_distribution_stats(rets)
    assert s["skew"] < -0.3


# ---------------------------------------------------------------------------
# _chart_return_distribution
# ---------------------------------------------------------------------------


def test_chart_returns_none_for_insufficient_data():
    df = _equity_from_returns(np.array([0.01, -0.01, 0.005]))
    assert _chart_return_distribution(df) is None


def test_chart_returns_figure_with_four_traces():
    rng = np.random.default_rng(99)
    rets = rng.normal(scale=0.01, size=500)
    df = _equity_from_returns(rets)
    fig = _chart_return_distribution(df)
    assert isinstance(fig, go.Figure)
    # histogram + normal PDF + QQ scatter + QQ reference line = 4 traces
    assert len(fig.data) == 4


def test_chart_has_histogram_and_qq_traces():
    rng = np.random.default_rng(100)
    rets = rng.normal(scale=0.01, size=500)
    df = _equity_from_returns(rets)
    fig = _chart_return_distribution(df)
    types = [t.type for t in fig.data]
    assert "histogram" in types
    # At least two scatter traces (normal PDF, QQ empirical+reference).
    assert types.count("scatter") >= 3


def test_chart_title_contains_stats_annotation():
    rng = np.random.default_rng(101)
    rets = rng.normal(scale=0.01, size=500)
    df = _equity_from_returns(rets)
    fig = _chart_return_distribution(df)
    title = fig.layout.title.text
    assert "Skew" in title
    assert "Kurt" in title
    assert "VaR" in title
    assert "CVaR" in title


def test_chart_has_var_and_cvar_vlines():
    rng = np.random.default_rng(102)
    rets = rng.normal(scale=0.01, size=500)
    df = _equity_from_returns(rets)
    fig = _chart_return_distribution(df)
    shapes = list(fig.layout.shapes or [])
    # Expect at least the vrect tail shading + VaR vline + CVaR vline = 3 shapes.
    assert len(shapes) >= 3


def test_chart_uses_dark_theme():
    rng = np.random.default_rng(103)
    rets = rng.normal(scale=0.01, size=500)
    df = _equity_from_returns(rets)
    fig = _chart_return_distribution(df)
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#0E1117"


def test_chart_qq_reference_line_uses_mu_sigma():
    """Reference line should be y = sigma*x + mu; trace #3 is the reference."""
    rng = np.random.default_rng(104)
    rets = rng.normal(loc=0.001, scale=0.01, size=1000)
    df = _equity_from_returns(rets)
    fig = _chart_return_distribution(df)
    ref = fig.data[3]
    xs = np.array(ref.x, dtype=float)
    ys = np.array(ref.y, dtype=float)
    slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    intercept = ys[0] - slope * xs[0]
    s = _return_distribution_stats(rets.tolist())
    assert slope == pytest.approx(s["sigma"], rel=1e-6)
    assert intercept == pytest.approx(s["mu"], abs=1e-9)


def test_chart_qq_empirical_is_sorted():
    """The empirical y-values in the QQ trace should be sorted ascending."""
    rng = np.random.default_rng(105)
    rets = rng.normal(scale=0.01, size=500)
    df = _equity_from_returns(rets)
    fig = _chart_return_distribution(df)
    qq_trace = fig.data[2]
    ys = list(qq_trace.y)
    assert ys == sorted(ys)
