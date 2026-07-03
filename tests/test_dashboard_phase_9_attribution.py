"""Tests for dashboard Phase 9 — return attribution waterfall."""

from __future__ import annotations

from datetime import date

import plotly.graph_objects as go
import polars as pl
import pytest

from src.dashboard.app import (
    _chart_contribution_waterfall,
    _compute_contribution_per_ticker,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _weights_long(
    rows: list[tuple[date, str, float]],
) -> pl.DataFrame:
    return pl.DataFrame({
        "date": [r[0] for r in rows],
        "ticker": [r[1] for r in rows],
        "weight": [r[2] for r in rows],
    })


def _returns_wide(
    dates: list[date], **ticker_rets: list[float]
) -> pl.DataFrame:
    cols: dict[str, list] = {"date": dates}
    cols.update(ticker_rets)
    return pl.DataFrame(cols)


# ---------------------------------------------------------------------------
# _compute_contribution_per_ticker
# ---------------------------------------------------------------------------


def test_contribution_empty_inputs_returns_empty():
    empty_w = pl.DataFrame({"date": [], "ticker": [], "weight": []})
    empty_r = pl.DataFrame({"date": []})
    assert _compute_contribution_per_ticker(empty_w, empty_r) == {}


def test_contribution_none_inputs_returns_empty():
    assert _compute_contribution_per_ticker(None, None) == {}


def test_contribution_weights_missing_columns_returns_empty():
    bad = pl.DataFrame({"date": [date(2024, 1, 1)]})
    r = _returns_wide([date(2024, 1, 2)], AAA=[0.01])
    assert _compute_contribution_per_ticker(bad, r) == {}


def test_contribution_single_rebalance_two_tickers_math():
    # Rebalance on day 1. Hold 0.6 AAA, 0.4 BBB for 3 days.
    # AAA returns: 0.02, 0.01, -0.005 → compound ≈ 1.02*1.01*0.995 - 1 = 0.02488
    # BBB returns: 0.0, -0.01, 0.015 → compound ≈ 1*0.99*1.015 - 1 = 0.004850
    w = _weights_long([
        (date(2024, 1, 1), "AAA", 0.6),
        (date(2024, 1, 1), "BBB", 0.4),
    ])
    r = _returns_wide(
        [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        AAA=[0.02, 0.01, -0.005],
        BBB=[0.0, -0.01, 0.015],
    )
    contrib = _compute_contribution_per_ticker(w, r)

    exp_aaa = 0.6 * (1.02 * 1.01 * 0.995 - 1.0) * 100.0
    exp_bbb = 0.4 * (1.0 * 0.99 * 1.015 - 1.0) * 100.0
    assert contrib["AAA"] == pytest.approx(exp_aaa, rel=1e-9)
    assert contrib["BBB"] == pytest.approx(exp_bbb, rel=1e-9)


def test_contribution_multiple_intervals_accumulate():
    # Two rebalances. Between them AAA earns +5%; after, -3%.
    w = _weights_long([
        (date(2024, 1, 1), "AAA", 0.5),
        (date(2024, 1, 3), "AAA", 0.5),
    ])
    r = _returns_wide(
        [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
        AAA=[0.05, 0.0, -0.03],
    )
    contrib = _compute_contribution_per_ticker(w, r)
    # Interval 1: r2-r3 → date>2024-01-01 and date<=2024-01-03 → [0.05, 0.0]
    #   compound = 1.05 * 1.0 - 1 = 0.05; contrib1 = 0.5 * 0.05
    # Interval 2: date>2024-01-03 and date<=2024-01-04 → [-0.03]
    #   compound = -0.03; contrib2 = 0.5 * -0.03
    expected = (0.5 * 0.05 + 0.5 * -0.03) * 100.0
    assert contrib["AAA"] == pytest.approx(expected, rel=1e-9)


def test_contribution_ignores_tickers_not_in_returns():
    w = _weights_long([
        (date(2024, 1, 1), "ONLY_IN_WEIGHTS", 1.0),
    ])
    r = _returns_wide([date(2024, 1, 2)], OTHER=[0.01])
    assert _compute_contribution_per_ticker(w, r) == {}


def test_contribution_skips_zero_weights():
    w = _weights_long([
        (date(2024, 1, 1), "AAA", 0.0),
        (date(2024, 1, 1), "BBB", 1.0),
    ])
    r = _returns_wide(
        [date(2024, 1, 2), date(2024, 1, 3)],
        AAA=[0.5, 0.5],
        BBB=[0.01, 0.01],
    )
    contrib = _compute_contribution_per_ticker(w, r)
    assert "AAA" not in contrib
    assert "BBB" in contrib


def test_contribution_nan_returns_treated_as_zero():
    w = _weights_long([(date(2024, 1, 1), "AAA", 1.0)])
    r = _returns_wide(
        [date(2024, 1, 2), date(2024, 1, 3)],
        AAA=[float("nan"), 0.01],
    )
    contrib = _compute_contribution_per_ticker(w, r)
    # NaN replaced by 0 → compound = 1.0 * 1.01 - 1 = 0.01
    assert contrib["AAA"] == pytest.approx(1.0, rel=1e-9)


def test_contribution_sum_approximates_portfolio_return():
    # Fully invested two-ticker book, one interval → sum of contributions
    # should equal weighted return exactly.
    w = _weights_long([
        (date(2024, 1, 1), "AAA", 0.3),
        (date(2024, 1, 1), "BBB", 0.7),
    ])
    r = _returns_wide(
        [date(2024, 1, 2), date(2024, 1, 3)],
        AAA=[0.02, -0.01],
        BBB=[-0.005, 0.015],
    )
    contrib = _compute_contribution_per_ticker(w, r)
    total_contrib_pp = sum(contrib.values())

    aaa_cr = 1.02 * 0.99 - 1.0
    bbb_cr = 0.995 * 1.015 - 1.0
    port_return_pct = (0.3 * aaa_cr + 0.7 * bbb_cr) * 100.0
    assert total_contrib_pp == pytest.approx(port_return_pct, rel=1e-9)


# ---------------------------------------------------------------------------
# _chart_contribution_waterfall
# ---------------------------------------------------------------------------


def test_waterfall_returns_none_for_empty_dict():
    assert _chart_contribution_waterfall({}) is None


def test_waterfall_builds_figure_with_start_and_total():
    contrib = {"AAA": 1.5, "BBB": -0.7, "CCC": 0.3}
    fig = _chart_contribution_waterfall(contrib, top_n=15)
    assert isinstance(fig, go.Figure)
    trace = fig.data[0]
    xs = list(trace.x)
    assert xs[0] == "Start"
    assert xs[-1] == "Total"
    # No "Others" when top_n >= total tickers.
    assert not any(x.startswith("Others") for x in xs)


def test_waterfall_bucket_others_when_more_than_top_n():
    contrib = {f"T{i:02d}": float(i + 1) for i in range(20)}
    fig = _chart_contribution_waterfall(contrib, top_n=5)
    trace = fig.data[0]
    xs = list(trace.x)
    others = [x for x in xs if x.startswith("Others")]
    assert others, "expected Others bucket when len(contrib) > top_n"
    # Others count = 20 - 5 = 15.
    assert "(15)" in others[0]


def test_waterfall_ordered_signed_descending_within_top_n():
    contrib = {"A": 3.0, "B": 1.0, "C": -2.0, "D": 0.5}
    fig = _chart_contribution_waterfall(contrib, top_n=4)
    trace = fig.data[0]
    xs = list(trace.x)[1:-1]  # strip Start + Total
    assert xs == ["A", "B", "D", "C"]  # 3.0, 1.0, 0.5, -2.0


def test_waterfall_total_equals_sum():
    contrib = {"A": 3.0, "B": 1.0, "C": -2.0}
    fig = _chart_contribution_waterfall(contrib, top_n=15)
    title_text = fig.layout.title.text or ""
    assert "+2.00pp" in title_text  # sum = 3 + 1 - 2 = 2


def test_waterfall_uses_dark_theme():
    contrib = {"AAA": 1.0}
    fig = _chart_contribution_waterfall(contrib)
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#0E1117"


def test_waterfall_text_uses_pp_suffix():
    contrib = {"AAA": 2.5, "BBB": -1.25}
    fig = _chart_contribution_waterfall(contrib)
    texts = [t for t in fig.data[0].text if t]
    assert any("+2.50pp" in t for t in texts)
    assert any("-1.25pp" in t for t in texts)


def test_waterfall_colors_semantic():
    contrib = {"WIN": 2.0, "LOSE": -2.0}
    fig = _chart_contribution_waterfall(contrib)
    wf = fig.data[0]
    # Increasing bars should be green (_ACCENT_GREEN), decreasing red.
    assert wf.increasing.marker.color == "#43A047"
    assert wf.decreasing.marker.color == "#E53935"
    assert wf.totals.marker.color == "#4A90D9"


def test_waterfall_top_n_selection_by_absolute_value():
    # Include one big loser — should make the top-3 over a smaller winner.
    contrib = {"BIG_LOSS": -10.0, "WIN1": 5.0, "WIN2": 4.0, "SMALL": 0.1}
    fig = _chart_contribution_waterfall(contrib, top_n=3)
    xs = list(fig.data[0].x)[1:-1]
    # Exclude Others bucket, keep individual tickers.
    tickers_shown = [x for x in xs if not x.startswith("Others")]
    assert "BIG_LOSS" in tickers_shown
    assert "SMALL" not in tickers_shown
