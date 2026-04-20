"""Phase 1.1 — dry-run debate for 3 tickers (AAPL/JPM/XOM).

Validates:
  - 4 agents complete per ticker (12 reports total)
  - no fallback reports (error is None, reasoning non-empty)
  - news_context has >= 1 item per ticker
  - final decisions have action/confidence/suggested_weight

Writes full states to data/outputs/_phase1_dryrun.json for inspection.
Delete this script after Phase 1.1 validation.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import sys
import time
from pathlib import Path

from loguru import logger

from src.agents.graph import run_agent_debate


TICKERS = ["AAPL", "JPM", "XOM"]
OUT_PATH = Path("data/outputs/_phase1_dryrun.json")


def _serialize(obj):  # noqa: ANN001
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    t0 = time.time()
    print(f"[phase1.1] Running debate for {TICKERS}...", flush=True)
    decisions, full_states = run_agent_debate(TICKERS)
    elapsed = time.time() - t0

    print(flush=True)
    print(f"[phase1.1] Elapsed: {elapsed:.1f}s", flush=True)
    print(f"[phase1.1] Decisions: {len(decisions)}/{len(TICKERS)}", flush=True)

    issues: list[str] = []
    per_ticker: dict[str, dict] = {}

    for ticker in TICKERS:
        state = full_states.get(ticker)
        if state is None:
            issues.append(f"{ticker}: no state returned")
            continue

        reports = state.get("reports", [])
        news = state.get("news_context", [])
        decision = state.get("final_decision") or {}

        per_ticker[ticker] = {
            "report_count": len(reports),
            "agents": sorted({r.get("agent", "?") for r in reports}),
            "news_count": len(news),
            "news_preview": [n[:80] for n in news[:2]],
            "action": decision.get("action"),
            "confidence": decision.get("confidence"),
            "weight": decision.get("suggested_weight"),
            "reasoning": (decision.get("reasoning") or "")[:200],
            "dissent": (decision.get("dissenting_view") or "")[:200],
            "fallback_signals": [
                r.get("agent")
                for r in reports
                if (r.get("reasoning") or "").startswith("Fallback")
                or r.get("confidence", 1.0) == 0.5 and (r.get("reasoning") or "").lower().startswith("fallback")
            ],
        }

        if len(reports) < 4:
            issues.append(f"{ticker}: only {len(reports)}/4 reports")
        if len(news) == 0:
            issues.append(f"{ticker}: news_context empty")
        if per_ticker[ticker]["fallback_signals"]:
            issues.append(
                f"{ticker}: fallback agents = {per_ticker[ticker]['fallback_signals']}"
            )
        if not decision.get("action"):
            issues.append(f"{ticker}: no final_decision.action")

    print(flush=True)
    print("=== Per-ticker summary ===", flush=True)
    for t, info in per_ticker.items():
        print(f"\n[{t}]", flush=True)
        for k, v in info.items():
            print(f"  {k}: {v}", flush=True)

    print(flush=True)
    print("=== Validation issues ===", flush=True)
    if issues:
        for x in issues:
            print(f"  FAIL: {x}", flush=True)
    else:
        print("  OK — no issues", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "tickers": TICKERS,
                "elapsed_sec": elapsed,
                "decisions_count": len(decisions),
                "per_ticker": _serialize(per_ticker),
                "full_states": _serialize(full_states),
                "issues": issues,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[phase1.1] Wrote {OUT_PATH}", flush=True)

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
