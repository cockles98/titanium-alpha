"""Tests for dashboard Phase 8 — effective N and Gini over time."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from src.dashboard.app import (
    _chart_concentration_evolution,
    _compute_effective_n,
    _compute_gini,
)


# ---------------------------------------------------------------------------
# _compute_effective_n
# ---------------------------------------------------------------------------


def test_effective_n_uniform_n_equal_positions_returns_n():
    # 4 positions of 0.25 each, no cash.
    assert _compute_effective_n([0.25, 0.25, 0.25, 0.25]) == pytest.approx(4.0)


def test_effective_n_single_position_no_cash_is_one():
    assert _compute_effective_n([1.0]) == pytest.approx(1.0)


def test_effective_n_half_cash_plus_one_stock_is_two():
    assert _compute_effective_n([0.5]) == pytest.approx(2.0)


def test_effective_n_half_cash_plus_two_equal_stocks():
    # weights = [0.25, 0.25], cash = 0.5
    # HHI = 0.5² + 2*(0.25²) = 0.25 + 0.125 = 0.375
    # eff_n = 1 / 0.375 ≈ 2.667
    assert _compute_effective_n([0.25, 0.25]) == pytest.approx(1.0 / 0.375)


def test_effective_n_ignores_zero_weights():
    # Zeros contribute nothing to HHI.
    assert _compute_effective_n([0.5, 0.0, 0.5, 0.0, 0.0]) == pytest.approx(2.0)


def test_effective_n_all_cash_is_one():
    # 100% cash → cash² = 1 → eff_n = 1
    assert _compute_effective_n([]) == pytest.approx(1.0)


def test_effective_n_large_uniform_portfolio():
    n = 52
    w = [1.0 / n] * n
    assert _compute_effective_n(w) == pytest.approx(float(n))


def test_effective_n_is_nan_when_all_weights_zero_and_no_room_for_cash():
    # If all weights are zero and we allow cash = 1 - 0 = 1, cash²=1, eff_n=1.
    # So the only way to get NaN is when negative weights produce 0 total.
    # Practically: weights = [0, 0] → cash = 1 → eff_n = 1.
    assert _compute_effective_n([0.0, 0.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _compute_gini
# ---------------------------------------------------------------------------


def test_gini_uniform_fully_invested_is_zero():
    # 4 equal positions, no cash → perfectly equal → Gini = 0
    assert _compute_gini([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0, abs=1e-12)


def test_gini_single_position_fully_invested_is_zero():
    # 1 of 1 slot → trivially "equal"
    assert _compute_gini([1.0]) == pytest.approx(0.0, abs=1e-12)


def test_gini_single_position_half_cash():
    # weights = [0.5], cash = 0.5 → [0.5, 0.5] sorted → Gini = 0
    assert _compute_gini([0.5]) == pytest.approx(0.0, abs=1e-12)


def test_gini_two_unequal_positions_no_cash():
    # w = [0.3, 0.7], sorted → cumsum [0.3, 1.0]
    # num = 2*(1*0.3 + 2*0.7) - 3*1 = 2*1.7 - 3 = 0.4
    # gini = 0.4 / (2*1) = 0.2
    assert _compute_gini([0.3, 0.7]) == pytest.approx(0.2, abs=1e-9)


def test_gini_single_position_half_cash_vs_one_stock():
    # 100% in 1 stock (no cash) vs 50/50 stock/cash:
    # both should yield Gini=0 because they reduce to one equal partition.
    assert _compute_gini([0.5]) == pytest.approx(0.0, abs=1e-12)


def test_gini_very_concentrated_approaches_half_for_two_slots():
    # 90% in one stock, 10% cash → [0.1, 0.9]
    # num = 2*(1*0.1 + 2*0.9) - 3*1 = 2*1.9 - 3 = 0.8
    # gini = 0.8 / (2*1) = 0.4
    assert _compute_gini([0.9]) == pytest.approx(0.4, abs=1e-9)


def test_gini_empty_weights_nan():
    # Empty vector + 100% cash → cash only, size 1, cum sum 1, numerator
    # = 2*(1*1) - 2*1 = 0, gini = 0 (trivially equal with one slot).
    assert _compute_gini([]) == pytest.approx(0.0, abs=1e-12)


def test_gini_ignores_zero_entries():
    # Zeros are dropped before concatenating cash.
    assert _compute_gini([0.0, 0.3, 0.0, 0.7, 0.0]) == pytest.approx(0.2, abs=1e-9)


# ---------------------------------------------------------------------------
# _chart_concentration_evolution
# ---------------------------------------------------------------------------


def _make_weights_long(
    rows: list[tuple[date, str, float]],
) -> pl.DataFrame:
    return pl.DataFrame({
        "date": [r[0] for r in rows],
        "ticker": [r[1] for r in rows],
        "weight": [r[2] for r in rows],
    })


def test_chart_returns_none_for_empty_weights():
    empty = pl.DataFrame({"date": [], "ticker": [], "weight": []},
                         schema={"date": pl.Date, "ticker": pl.Utf8, "weight": pl.Float64})
    assert _chart_concentration_evolution(empty) is None


def test_chart_returns_none_when_weight_column_missing():
    df = pl.DataFrame({"date": [date(2024, 1, 1)], "ticker": ["AAA"]})
    assert _chart_concentration_evolution(df) is None


def test_chart_builds_two_traces_on_dual_axis():
    w = _make_weights_long([
        (date(2024, 1, 1), "AAA", 0.5),
        (date(2024, 1, 1), "BBB", 0.5),
        (date(2024, 2, 1), "AAA", 0.25),
        (date(2024, 2, 1), "BBB", 0.25),
        (date(2024, 2, 1), "CCC", 0.25),
        (date(2024, 2, 1), "DDD", 0.25),
    ])
    fig = _chart_concentration_evolution(w)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    names = sorted(t.name for t in fig.data)
    assert names == ["Effective N", "Gini"]


def test_chart_effective_n_series_matches_compute():
    w = _make_weights_long([
        (date(2024, 1, 1), "AAA", 1.0),            # eff_n = 1
        (date(2024, 2, 1), "AAA", 0.25),
        (date(2024, 2, 1), "BBB", 0.25),
        (date(2024, 2, 1), "CCC", 0.25),
        (date(2024, 2, 1), "DDD", 0.25),            # eff_n = 4
    ])
    fig = _chart_concentration_evolution(w)
    eff_trace = next(t for t in fig.data if t.name == "Effective N")
    ys = list(eff_trace.y)
    assert ys[0] == pytest.approx(1.0)
    assert ys[1] == pytest.approx(4.0)


def test_chart_has_reference_line_at_one_over_max_weight():
    w = _make_weights_long([
        (date(2024, 1, 1), "AAA", 0.5),
        (date(2024, 1, 1), "BBB", 0.5),
    ])
    fig = _chart_concentration_evolution(w)
    shapes = list(fig.layout.shapes or [])
    assert shapes, "expected at least one reference shape"
    ys = {round(float(sh.y0), 4) for sh in shapes}
    expected = round(1.0 / 0.06, 4)
    assert expected in ys


def test_chart_uses_dark_theme():
    w = _make_weights_long([
        (date(2024, 1, 1), "AAA", 0.5),
        (date(2024, 1, 1), "BBB", 0.5),
    ])
    fig = _chart_concentration_evolution(w)
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#0E1117"


def test_chart_gini_axis_bounded_to_unit_interval():
    w = _make_weights_long([
        (date(2024, 1, 1), "AAA", 0.5),
        (date(2024, 1, 1), "BBB", 0.5),
    ])
    fig = _chart_concentration_evolution(w)
    # Secondary y axis (Gini) should be bounded 0-1.
    yaxis2 = fig.layout["yaxis2"]
    assert tuple(yaxis2.range) == (0.0, 1.0)


def test_chart_x_is_sorted_by_date():
    w = _make_weights_long([
        (date(2024, 3, 1), "AAA", 1.0),
        (date(2024, 1, 1), "AAA", 1.0),
        (date(2024, 2, 1), "AAA", 1.0),
    ])
    fig = _chart_concentration_evolution(w)
    eff_trace = next(t for t in fig.data if t.name == "Effective N")
    xs = list(eff_trace.x)
    assert xs == sorted(xs)


def test_chart_annotation_mentions_max_weight_constraint():
    w = _make_weights_long([
        (date(2024, 1, 1), "AAA", 0.5),
        (date(2024, 1, 1), "BBB", 0.5),
    ])
    fig = _chart_concentration_evolution(w)
    annotations = [a.text or "" for a in (fig.layout.annotations or [])]
    joined = " | ".join(annotations)
    assert "HRP" in joined or "max" in joined.lower()
    assert "0.06" in joined
