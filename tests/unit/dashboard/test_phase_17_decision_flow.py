"""Tests for dashboard Phase 17 — decision-flow Sankey diagram."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import pytest

from src.dashboard.app import (
    _chart_decision_flow_sankey,
    _decision_flow_breakdown,
)


def _decisions(
    actions: list[str],
    weights: list[float] | None = None,
    invested_fraction: float | None = None,
    metadata_counts: bool = True,
) -> dict[str, Any]:
    """Synthesize a decisions.json payload from a list of actions.

    When ``weights`` is omitted, BUY tickers get a constant raw weight
    chosen so the total invested fraction roughly matches a typical
    walk-forward run. When ``invested_fraction`` is omitted, it is
    derived from the weight sum (matching production fallback logic).
    """
    n = len(actions)
    if weights is None:
        weights = [0.05 if a == "BUY" else 0.01 if a == "HOLD" else 0.0
                   for a in actions]
    decs = [
        {
            "ticker": f"T{i:02d}",
            "action": a,
            "confidence": 0.5,
            "weight": w,
        }
        for i, (a, w) in enumerate(zip(actions, weights))
    ]
    meta: dict[str, Any] = {}
    if metadata_counts:
        meta["n_buy"] = sum(1 for a in actions if a == "BUY")
        meta["n_hold"] = sum(1 for a in actions if a == "HOLD")
        meta["n_sell"] = sum(1 for a in actions if a == "SELL")
    if invested_fraction is not None:
        meta["invested_fraction"] = invested_fraction
    return {
        "decisions": decs,
        "metadata": meta,
        "tickers": [d["ticker"] for d in decs],
    }


# ---------------------------------------------------------------------------
# _decision_flow_breakdown
# ---------------------------------------------------------------------------


def test_breakdown_returns_none_for_missing_payload():
    assert _decision_flow_breakdown(None) is None  # type: ignore[arg-type]
    assert _decision_flow_breakdown({}) is None
    assert _decision_flow_breakdown({"decisions": []}) is None


def test_breakdown_universe_share_sums_to_one():
    actions = ["BUY"] * 4 + ["HOLD"] * 45 + ["SELL"] * 3
    flow = _decision_flow_breakdown(_decisions(actions, invested_fraction=0.20))
    assert flow is not None
    assert flow["f_buy"] + flow["f_hold"] + flow["f_sell"] == pytest.approx(1.0)


def test_breakdown_invested_plus_cash_equals_one():
    actions = ["BUY"] * 2 + ["HOLD"] * 30 + ["SELL"] * 5
    flow = _decision_flow_breakdown(_decisions(actions, invested_fraction=0.18))
    assert flow is not None
    cash = flow["cash_from_hrp"] + flow["cash_from_excluded"]
    assert flow["invested"] + cash == pytest.approx(1.0)


def test_breakdown_excluded_cash_is_sell_share():
    actions = ["BUY"] * 1 + ["HOLD"] * 49 + ["SELL"] * 2
    flow = _decision_flow_breakdown(_decisions(actions, invested_fraction=0.12))
    assert flow is not None
    assert flow["cash_from_excluded"] == pytest.approx(2 / 52)
    assert flow["f_sell"] == pytest.approx(2 / 52)


def test_breakdown_cash_from_hrp_is_pool_minus_invested():
    actions = ["BUY"] * 3 + ["HOLD"] * 40 + ["SELL"] * 9
    flow = _decision_flow_breakdown(_decisions(actions, invested_fraction=0.30))
    assert flow is not None
    pool = flow["f_buy"] + flow["f_hold"]
    assert flow["cash_from_hrp"] == pytest.approx(pool - 0.30)


def test_breakdown_falls_back_to_counting_when_metadata_missing():
    actions = ["BUY"] * 2 + ["HOLD"] * 5 + ["SELL"] * 3
    payload = _decisions(actions, metadata_counts=False, invested_fraction=0.10)
    flow = _decision_flow_breakdown(payload)
    assert flow is not None
    assert flow["n_buy"] == 2
    assert flow["n_hold"] == 5
    assert flow["n_sell"] == 3
    assert flow["n_total"] == 10


def test_breakdown_derives_invested_from_weight_sum_when_missing():
    actions = ["BUY", "BUY", "HOLD", "HOLD", "SELL"]
    weights = [0.10, 0.05, 0.02, 0.01, 0.0]
    payload = _decisions(actions, weights=weights, invested_fraction=None)
    flow = _decision_flow_breakdown(payload)
    assert flow is not None
    assert flow["invested"] == pytest.approx(sum(weights))


def test_breakdown_clamps_invested_to_unit_interval():
    payload = _decisions(["BUY"], invested_fraction=1.5)
    flow = _decision_flow_breakdown(payload)
    assert flow is not None
    assert flow["invested"] == 1.0


def test_breakdown_handles_all_buy_no_excluded_flow():
    flow = _decision_flow_breakdown(_decisions(["BUY"] * 10, invested_fraction=1.0))
    assert flow is not None
    assert flow["cash_from_excluded"] == 0.0
    assert flow["cash_from_hrp"] == 0.0


def test_breakdown_handles_all_sell_no_hrp_flow():
    flow = _decision_flow_breakdown(_decisions(["SELL"] * 5, invested_fraction=0.0))
    assert flow is not None
    assert flow["invested"] == 0.0
    assert flow["cash_from_excluded"] == pytest.approx(1.0)
    assert flow["cash_from_hrp"] == 0.0


# ---------------------------------------------------------------------------
# _chart_decision_flow_sankey
# ---------------------------------------------------------------------------


def test_chart_returns_none_for_empty_payload():
    assert _chart_decision_flow_sankey({}) is None
    assert _chart_decision_flow_sankey({"decisions": []}) is None


def test_chart_returns_figure_with_single_sankey_trace():
    actions = ["BUY"] * 1 + ["HOLD"] * 49 + ["SELL"] * 2
    fig = _chart_decision_flow_sankey(_decisions(actions, invested_fraction=0.118))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "sankey"


def test_chart_has_eight_nodes():
    actions = ["BUY"] * 2 + ["HOLD"] * 40 + ["SELL"] * 8
    fig = _chart_decision_flow_sankey(_decisions(actions, invested_fraction=0.20))
    assert fig is not None
    labels = list(fig.data[0].node.label)
    assert len(labels) == 8


def test_chart_node_labels_include_counts_and_pct():
    actions = ["BUY"] * 1 + ["HOLD"] * 49 + ["SELL"] * 2
    fig = _chart_decision_flow_sankey(_decisions(actions, invested_fraction=0.118))
    assert fig is not None
    labels = list(fig.data[0].node.label)
    assert any("Universe (52)" in lbl for lbl in labels)
    assert any("BUY (1)" in lbl for lbl in labels)
    assert any("HOLD (49)" in lbl for lbl in labels)
    assert any("SELL (2)" in lbl for lbl in labels)
    assert any("11.8%" in lbl for lbl in labels)  # invested
    assert any("88.2%" in lbl for lbl in labels)  # cash


def test_chart_link_values_conserve_at_each_layer():
    actions = ["BUY"] * 4 + ["HOLD"] * 40 + ["SELL"] * 8
    fig = _chart_decision_flow_sankey(_decisions(actions, invested_fraction=0.25))
    assert fig is not None
    link = fig.data[0].link
    # Universe outflow should sum to 1.0 (source 0 = Universe).
    out_universe = sum(v for s, v in zip(link.source, link.value) if s == 0)
    assert out_universe == pytest.approx(1.0)
    # Cash inflow + Invested inflow should sum to 1.0 (terminal layer).
    in_invested = sum(v for t, v in zip(link.target, link.value) if t == 6)
    in_cash = sum(v for t, v in zip(link.target, link.value) if t == 7)
    assert in_invested + in_cash == pytest.approx(1.0)


def test_chart_invested_link_matches_metadata_invested_fraction():
    actions = ["BUY"] * 3 + ["HOLD"] * 30 + ["SELL"] * 5
    fig = _chart_decision_flow_sankey(_decisions(actions, invested_fraction=0.42))
    assert fig is not None
    link = fig.data[0].link
    # Target index 6 = Invested.
    invested_flow = sum(v for t, v in zip(link.target, link.value) if t == 6)
    assert invested_flow == pytest.approx(0.42)


def test_chart_zero_flow_links_omitted():
    """All-BUY portfolio: SELL/Excluded/cash_from_excluded links must not be drawn."""
    fig = _chart_decision_flow_sankey(_decisions(["BUY"] * 5, invested_fraction=1.0))
    assert fig is not None
    link = fig.data[0].link
    # Target index 5 = Excluded; should have zero incoming links.
    incoming_excluded = [v for t, v in zip(link.target, link.value) if t == 5]
    assert incoming_excluded == []
    # Target index 7 = Cash; should also have zero incoming when invested=1.
    incoming_cash = [v for t, v in zip(link.target, link.value) if t == 7]
    assert incoming_cash == []


def test_chart_uses_dark_theme():
    actions = ["BUY"] * 2 + ["HOLD"] * 5 + ["SELL"] * 3
    fig = _chart_decision_flow_sankey(_decisions(actions, invested_fraction=0.15))
    assert fig is not None
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#0E1117"


def test_chart_title_mentions_pipeline_stages():
    actions = ["BUY"] * 1 + ["HOLD"] * 1 + ["SELL"] * 1
    fig = _chart_decision_flow_sankey(_decisions(actions, invested_fraction=0.30))
    assert fig is not None
    title = fig.layout.title.text or ""
    assert "Universe" in title
    assert "Debate" in title
    assert "HRP" in title
    assert "Final" in title
