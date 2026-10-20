"""Reproducible checks for the fast crash brake's regime transitions.

Run: python test_crash_brake.py

Each case loads agent.py fresh, because the brake keeps module-level state
(_last_regime, _last_rebalance_bar_date) exactly as the engine's long-lived
agent process does.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from datetime import date, timedelta

HERE = pathlib.Path(__file__).parent
TICKERS = (
    "SPY", "QQQ", "DIA", "IWM",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "SMH",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "QLD", "SSO",
)


def load_agent():
    spec = importlib.util.spec_from_file_location("agent_under_test", HERE / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bars_from_closes(closes: list[float]) -> list[dict]:
    start = date(2025, 1, 1)
    return [
        {
            "ts": (start + timedelta(days=i)).isoformat(),
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1_000_000,
        }
        for i, price in enumerate(closes)
    ]


def build_state(closes: list[float]) -> dict:
    return {ticker: bars_from_closes(closes) for ticker in TICKERS}


def uptrend(days: int = 100) -> list[float]:
    out = []
    for i in range(days):
        wobble = 1.0008 if i % 2 else 0.9992
        out.append(100.0 * (1.0015 ** i) * wobble)
    return out


def extend(closes: list[float], moves: list[float]) -> list[float]:
    out = list(closes)
    for move in moves:
        out.append(out[-1] * (1.0 + move))
    return out


def hold(agent, state: dict, cash: float = 100_000.0):
    """First decision from cash, then a portfolio that matches the fills."""
    orders = agent.decide(state, {"cash": cash, "positions": [], "last_prices": {}}, cash)
    prices = {t: state[t][-1]["close"] for t in state}
    positions = []
    for order in orders:
        if order["side"] != "buy":
            continue
        quantity = float(order["quantity"])
        cash -= quantity * prices[order["ticker"]]
        positions.append({
            "ticker": order["ticker"],
            "quantity": quantity,
            "avg_cost": prices[order["ticker"]],
        })
    return {"cash": cash, "positions": positions, "last_prices": prices}, orders


def sells(orders: list[dict]) -> list[dict]:
    return [o for o in orders if o["side"] == "sell"]


def case(name: str, fn) -> bool:
    try:
        ok, detail = fn()
    except AssertionError as exc:
        ok, detail = False, f"fixture/assert: {exc}"
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def on_to_hard():
    agent = load_agent()
    path = extend(uptrend(), [-0.022, -0.022, -0.022])
    state_on = build_state(path[:100])
    assert agent._regime(state_on) == "on", agent._regime(state_on)
    portfolio, first = hold(agent, state_on)
    assert first, "risk-on book did not invest"

    state_hard = build_state(path[:103])
    assert agent._regime(state_hard) == "hard", agent._regime(state_hard)
    orders = agent.decide(state_hard, portfolio, portfolio["cash"])
    return bool(sells(orders)), f"{len(sells(orders))} sells"


def on_to_soft_waits():
    """A trend break alone is not a crash: it waits for the 5-day cadence.

    Forcing it costs return without cutting drawdown on the public windows
    (calm 14.65% -> 11.72%, moderate DD 6.13% -> 6.34%), so the brake stays
    crash-only and this case pins that decision.
    """
    agent = load_agent()
    path = extend(uptrend(), [-0.045])
    state_on = build_state(path[:100])
    assert agent._regime(state_on) == "on", agent._regime(state_on)
    portfolio, first = hold(agent, state_on)
    assert first, "risk-on book did not invest"

    state_soft = build_state(path[:101])
    assert agent._regime(state_soft) == "soft", agent._regime(state_soft)
    orders = agent.decide(state_soft, portfolio, portfolio["cash"])
    return not orders, f"{len(orders)} orders on a trend break"


def soft_to_hard():
    agent = load_agent()
    path = extend(uptrend(), [-0.045, -0.022, -0.022, -0.022])
    state_soft = build_state(path[:101])
    assert agent._regime(state_soft) == "soft", agent._regime(state_soft)
    portfolio, first = hold(agent, state_soft)
    assert first, "defensive book did not invest"

    state_hard = build_state(path[:104])
    assert agent._regime(state_hard) == "hard", agent._regime(state_hard)
    orders = agent.decide(state_hard, portfolio, portfolio["cash"])
    return bool(sells(orders)), f"{len(sells(orders))} sells"


def on_to_on_stays_quiet():
    agent = load_agent()
    path = uptrend()
    state = build_state(path)
    assert agent._regime(state) == "on", agent._regime(state)
    portfolio, first = hold(agent, state)
    assert first, "risk-on book did not invest"

    next_day = extend(path, [0.001])
    orders = agent.decide(build_state(next_day), portfolio, portfolio["cash"])
    return not orders, f"{len(orders)} orders inside the cadence"


def hard_to_soft_waits():
    agent = load_agent()
    path = extend(uptrend(), [-0.022, -0.022, -0.022])
    state_hard = build_state(path)
    assert agent._regime(state_hard) == "hard", agent._regime(state_hard)
    portfolio, first = hold(agent, state_hard)
    assert first, "crash book did not invest"

    recovering = extend(path, [0.030])
    assert agent._regime(build_state(recovering)) == "soft"
    orders = agent.decide(build_state(recovering), portfolio, portfolio["cash"])
    return not orders, f"{len(orders)} orders while re-risking"


def fresh_process_in_crash_sells():
    agent = load_agent()
    path = extend(uptrend(), [-0.022, -0.022, -0.022])
    state = build_state(path)
    assert agent._regime(state) == "hard", agent._regime(state)
    # A long-lived risk-on book, as if the process had just restarted mid-crash.
    prices = {t: state[t][-1]["close"] for t in state}
    portfolio = {
        "cash": 10_000.0,
        "positions": [
            {"ticker": t, "quantity": 100.0, "avg_cost": prices[t]}
            for t in ("SPY", "QQQ", "XLK", "NVDA")
        ],
        "last_prices": prices,
    }
    orders = agent.decide(state, portfolio, portfolio["cash"])
    return bool(sells(orders)), f"{len(sells(orders))} sells"


def main() -> int:
    cases = [
        ("on -> hard forces de-risk (already worked)", on_to_hard),
        ("soft -> hard forces de-risk", soft_to_hard),
        ("on -> soft waits for the cadence", on_to_soft_waits),
        ("on -> on stays quiet inside the cadence", on_to_on_stays_quiet),
        ("hard -> soft waits for the cadence", hard_to_soft_waits),
        ("fresh process in crash liquidates", fresh_process_in_crash_sells),
    ]
    results = [case(name, fn) for name, fn in cases]
    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
