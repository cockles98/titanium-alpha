"""Tests for dashboard Phase 12 — agent-vote matrix heatmap (snapshot variant)."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import pytest

from src.dashboard.app import (
    _build_vote_matrix,
    _chart_agent_vote_matrix,
)


def _decision(
    ticker: str, action: str, confidence: float = 0.5, weight: float = 0.0,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "action": action,
        "confidence": confidence,
        "weight": weight,
    }


def _report(agent: str, signal: str, confidence: float = 0.7) -> dict[str, Any]:
    return {
        "agent": agent,
        "signal": signal,
        "confidence": confidence,
    }


def _payload(
    decisions: list[dict[str, Any]],
    debate: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    return (
        {"decisions": decisions, "tickers": [d["ticker"] for d in decisions]},
        debate,
    )


# ---------------------------------------------------------------------------
# _build_vote_matrix
# ---------------------------------------------------------------------------


def test_build_returns_none_for_empty_decisions():
    assert _build_vote_matrix({}, None) is None
    assert _build_vote_matrix({"decisions": []}, None) is None
    assert _build_vote_matrix(None, None) is None  # type: ignore[arg-type]


def test_build_orders_tickers_buy_first_then_hold_then_sell():
    decs, debate = _payload(
        [
            _decision("ZZ", "HOLD"),
            _decision("AA", "SELL"),
            _decision("MM", "BUY"),
            _decision("BB", "HOLD"),
            _decision("CC", "BUY"),
        ],
    )
    matrix = _build_vote_matrix(decs, debate)
    assert matrix is not None
    # BUY block alphabetical, then HOLD alphabetical, then SELL alphabetical.
    assert matrix["tickers"] == ["CC", "MM", "BB", "ZZ", "AA"]


def test_build_pm_column_always_populated_from_decisions():
    decs, debate = _payload(
        [
            _decision("AAA", "BUY", confidence=0.8, weight=0.05),
            _decision("BBB", "HOLD", confidence=0.3, weight=0.005),
            _decision("CCC", "SELL", confidence=0.9, weight=0.0),
        ],
        debate=None,  # no debate at all
    )
    matrix = _build_vote_matrix(decs, debate)
    assert matrix is not None
    pm_col = 3
    # Codes: BUY=1, HOLD=0, SELL=-1
    by_t = {t: i for i, t in enumerate(matrix["tickers"])}
    assert matrix["z"][by_t["AAA"]][pm_col] == 1
    assert matrix["z"][by_t["BBB"]][pm_col] == 0
    assert matrix["z"][by_t["CCC"]][pm_col] == -1
    # Confidence forwarded.
    assert matrix["confidence"][by_t["AAA"]][pm_col] == pytest.approx(0.8)


def test_build_agent_columns_use_signal_codes():
    decs, debate_dict = _payload(
        [_decision("X", "HOLD")],
        debate={
            "X": {
                "reports": [
                    _report("technical", "bullish", 0.6),
                    _report("fundamental", "neutral", 0.5),
                    _report("bear", "bearish", 0.9),
                ]
            }
        },
    )
    matrix = _build_vote_matrix(decs, debate_dict)
    assert matrix is not None
    z = matrix["z"][0]
    # Columns: 0=Technical, 1=Fundamental, 2=Bear
    assert z[0] == 1   # bullish
    assert z[1] == 0   # neutral
    assert z[2] == -1  # bearish


def test_build_missing_agent_report_leaves_cell_none():
    decs, debate_dict = _payload(
        [_decision("X", "HOLD")],
        debate={"X": {"reports": [_report("technical", "bullish")]}},
    )
    matrix = _build_vote_matrix(decs, debate_dict)
    assert matrix is not None
    # Fundamental + Bear cells must be None (gray/missing).
    assert matrix["z"][0][1] is None
    assert matrix["z"][0][2] is None
    assert matrix["confidence"][0][1] is None


def test_build_missing_debate_for_ticker_keeps_pm_only():
    decs, debate_dict = _payload(
        [_decision("X", "BUY"), _decision("Y", "HOLD")],
        debate={"X": {"reports": [_report("technical", "neutral")]}},
    )
    matrix = _build_vote_matrix(decs, debate_dict)
    assert matrix is not None
    by_t = {t: i for i, t in enumerate(matrix["tickers"])}
    # Y has no debate entry: agent cells None, PM cell filled.
    assert matrix["z"][by_t["Y"]][:3] == [None, None, None]
    assert matrix["z"][by_t["Y"]][3] == 0  # HOLD


def test_build_unknown_signal_value_does_not_crash():
    decs, debate_dict = _payload(
        [_decision("X", "BUY")],
        debate={
            "X": {
                "reports": [_report("technical", "garbled-signal")]
            }
        },
    )
    matrix = _build_vote_matrix(decs, debate_dict)
    assert matrix is not None
    # Unknown signal should leave the cell as None (not crash, not 0).
    assert matrix["z"][0][0] is None


def test_build_unknown_agent_key_is_ignored():
    decs, debate_dict = _payload(
        [_decision("X", "HOLD")],
        debate={
            "X": {
                "reports": [
                    _report("technical", "bullish"),
                    _report("portfolio_manager", "neutral"),  # not in 0..2
                ]
            }
        },
    )
    matrix = _build_vote_matrix(decs, debate_dict)
    assert matrix is not None
    assert matrix["z"][0][0] == 1  # technical recorded
    # No exception, PM column comes from decisions only.


def test_build_actions_and_weights_populated():
    decs, debate_dict = _payload(
        [
            _decision("A", "BUY", confidence=0.8, weight=0.1),
            _decision("B", "HOLD", confidence=0.2, weight=0.01),
        ],
    )
    matrix = _build_vote_matrix(decs, debate_dict)
    assert matrix is not None
    assert matrix["actions"] == {"A": "BUY", "B": "HOLD"}
    assert matrix["weights"] == pytest.approx({"A": 0.1, "B": 0.01})


def test_build_buy_letter_and_hold_letter_distinct():
    decs, _ = _payload(
        [_decision("A", "BUY"), _decision("B", "HOLD"), _decision("C", "SELL")],
    )
    matrix = _build_vote_matrix(decs, None)
    assert matrix is not None
    by_t = {t: i for i, t in enumerate(matrix["tickers"])}
    pm = 3
    assert matrix["text"][by_t["A"]][pm] == "B"
    assert matrix["text"][by_t["B"]][pm] == "H"   # HOLD letter is "H"
    assert matrix["text"][by_t["C"]][pm] == "S"


# ---------------------------------------------------------------------------
# _chart_agent_vote_matrix
# ---------------------------------------------------------------------------


def test_chart_returns_none_for_empty_decisions():
    assert _chart_agent_vote_matrix({}, None) is None
    assert _chart_agent_vote_matrix({"decisions": []}, None) is None


def test_chart_produces_single_heatmap_trace():
    decs, debate_dict = _payload(
        [_decision("A", "BUY"), _decision("B", "HOLD")],
    )
    fig = _chart_agent_vote_matrix(decs, debate_dict)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


def test_chart_axes_have_four_voices_and_n_tickers():
    decs, _ = _payload(
        [_decision(f"T{i}", "HOLD") for i in range(7)],
    )
    fig = _chart_agent_vote_matrix(decs, None)
    assert fig is not None
    x = list(fig.data[0].x)
    y = list(fig.data[0].y)
    assert x == ["Technical", "Fundamental", "Bear", "PM"]
    assert len(y) == 7


def test_chart_tickers_sorted_buy_first_in_y_axis():
    decs, _ = _payload(
        [
            _decision("ZED", "SELL"),
            _decision("ALF", "HOLD"),
            _decision("BUY1", "BUY"),
        ],
    )
    fig = _chart_agent_vote_matrix(decs, None)
    assert fig is not None
    y = list(fig.data[0].y)
    # Sorted: BUY first, HOLD next, SELL last.
    assert y == ["BUY1", "ALF", "ZED"]


def test_chart_zmin_zmid_zmax_bound_to_minus1_0_plus1():
    decs, _ = _payload([_decision("A", "BUY")])
    fig = _chart_agent_vote_matrix(decs, None)
    assert fig is not None
    h = fig.data[0]
    assert h.zmin == -1
    assert h.zmid == 0
    assert h.zmax == 1


def test_chart_title_includes_action_counts():
    decs, _ = _payload(
        [
            _decision("A", "BUY"),
            _decision("B", "HOLD"),
            _decision("C", "HOLD"),
            _decision("D", "SELL"),
        ],
    )
    fig = _chart_agent_vote_matrix(decs, None)
    assert fig is not None
    title = fig.layout.title.text or ""
    assert "1 BUY" in title
    assert "2 HOLD" in title
    assert "1 SELL" in title


def test_chart_hover_text_includes_weight_only_for_pm_column():
    decs, debate_dict = _payload(
        [_decision("X", "BUY", confidence=0.7, weight=0.08)],
        debate={"X": {"reports": [_report("technical", "bullish", 0.6)]}},
    )
    fig = _chart_agent_vote_matrix(decs, debate_dict)
    assert fig is not None
    cd = fig.data[0].customdata
    # Single row, four cells. Column 0 = Technical, Column 3 = PM.
    technical_hover = cd[0][0]
    pm_hover = cd[0][3]
    assert "Final weight" not in technical_hover
    assert "Final weight" in pm_hover
    assert "8.00%" in pm_hover


def test_chart_uses_dark_theme():
    decs, _ = _payload([_decision("A", "BUY")])
    fig = _chart_agent_vote_matrix(decs, None)
    assert fig is not None
    assert fig.layout.paper_bgcolor == "#0E1117"
    assert fig.layout.plot_bgcolor == "#0E1117"


def test_chart_xaxis_on_top_yaxis_reversed_for_readability():
    decs, _ = _payload([_decision("A", "BUY"), _decision("B", "HOLD")])
    fig = _chart_agent_vote_matrix(decs, None)
    assert fig is not None
    assert fig.layout.xaxis.side == "top"
    assert fig.layout.yaxis.autorange == "reversed"


def test_chart_height_scales_with_ticker_count():
    decs_few, _ = _payload([_decision("A", "BUY")])
    decs_many, _ = _payload([_decision(f"T{i}", "HOLD") for i in range(60)])
    fig_few = _chart_agent_vote_matrix(decs_few, None)
    fig_many = _chart_agent_vote_matrix(decs_many, None)
    assert fig_few is not None and fig_many is not None
    assert fig_many.layout.height > fig_few.layout.height
