"""Phase 1.2 — validate decisions.json post make decide."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PATH = Path("data/outputs/decisions.json")


def main() -> int:
    raw = json.loads(PATH.read_text(encoding="utf-8"))
    decisions = raw["decisions"]

    n = len(decisions)
    actions = Counter(d["action"] for d in decisions)
    sum_w = sum(d["weight"] for d in decisions)
    max_w = max(d["weight"] for d in decisions)
    max_t = max(decisions, key=lambda d: d["weight"])["ticker"]

    required_keys = {"ticker", "action", "weight", "confidence", "reasoning", "dissenting_view"}
    missing_keys: list[str] = []
    for d in decisions:
        miss = required_keys - set(d.keys())
        if miss:
            missing_keys.append(f"{d.get('ticker','?')}: missing {miss}")

    cap = min(0.06, 2 / n)

    print(f"timestamp: {raw.get('timestamp')}")
    print(f"n_decisions: {n}")
    print(f"action distribution: {dict(actions)}")
    print(f"sum(weight): {sum_w:.4f}  (cap: 1.0)")
    print(f"max(weight): {max_w:.4f} in {max_t}  (cap: {cap:.4f})")
    print(f"invested fraction (sum of BUY+HOLD): {sum_w:.4f}")

    issues = []
    if n != 52:
        issues.append(f"expected 52 decisions, got {n}")
    if sum_w > 1.0 + 1e-9:
        issues.append(f"sum(weight)={sum_w:.6f} > 1.0")
    if max_w > cap + 1e-9:
        issues.append(f"max(weight)={max_w:.6f} > cap={cap:.6f}")
    if missing_keys:
        issues.append("missing required keys: " + "; ".join(missing_keys))

    sells_with_weight = [d for d in decisions if d["action"] == "SELL" and d["weight"] > 0]
    if sells_with_weight:
        issues.append(f"SELL tickers with non-zero weight: {[d['ticker'] for d in sells_with_weight]}")

    print()
    print("=== BUY decisions ===")
    for d in decisions:
        if d["action"] == "BUY":
            print(f"  {d['ticker']}: weight={d['weight']:.4f}  conf={d['confidence']:.2f}")
    print()
    print("=== SELL decisions ===")
    for d in decisions:
        if d["action"] == "SELL":
            print(f"  {d['ticker']}: weight={d['weight']:.4f}  conf={d['confidence']:.2f}")

    print()
    print("=== Validation ===")
    if issues:
        for i in issues:
            print(f"  FAIL: {i}")
        return 1
    print("  OK — all criteria met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
