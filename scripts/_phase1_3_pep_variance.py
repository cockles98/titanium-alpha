"""Run debate for PEP 5 times to measure stochastic variance.

If all HOLD → systematic difference from make decide.
If mixed → confirms temperature-driven stochasticity.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import sys
from loguru import logger

from src.agents.graph import run_agent_debate


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    results = []
    for i in range(5):
        print(f"\n=== Run {i+1}/5 ===", flush=True)
        decisions, states = run_agent_debate(["PEP"])
        d = decisions[0] if decisions else None
        if d is None:
            print("  NO DECISION", flush=True)
            results.append(None)
            continue
        # Extract per-agent signals from state
        state = states.get("PEP", {})
        reports = state.get("reports", [])
        agent_sig = {r.get("agent"): (r.get("signal"), r.get("confidence")) for r in reports}
        print(
            f"  action={d['action']} weight={d['suggested_weight']:.4f} "
            f"conf={d['confidence']:.2f}",
            flush=True,
        )
        print(f"  agents: {agent_sig}", flush=True)
        results.append({
            "action": d["action"],
            "weight": d["suggested_weight"],
            "conf": d["confidence"],
            "agents": agent_sig,
        })

    print("\n=== Distribution ===", flush=True)
    actions = [r["action"] if r else "ERR" for r in results]
    print(f"Actions: {actions}", flush=True)
    confs = [r["conf"] if r else None for r in results]
    print(f"Confidences: {confs}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
