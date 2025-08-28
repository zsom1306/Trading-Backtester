#!/usr/bin/env python3
"""
quant_backtester.py

A compact, well-structured backtester for a single-asset moving-average crossover strategy.

Features
- Fetch OHLCV data with yfinance
- Compute indicators (SMA short/long)
- Generate signals (long/flat) with lookahead-safe logic
- Portfolio simulation with position sizing, cash, equity, and optional costs
- Performance metrics: CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Turnover, Exposure
- Plots: price + signals, equity vs buy&hold, drawdown
- CLI with sane defaults; writes results to ./results/

Usage
------
$ python quant_backtester.py --ticker AAPL --start 2018-01-01 --end 2025-01-01 \
    --short 20 --long 50 --capital 10000 --fee-bps 1 --slip-bps 1

Dependencies
------------
- yfinance
- pandas
- numpy
- matplotlib

Install:
$ pip install yfinance pandas numpy matplotlib

Author: (you)
License: MIT (optional)
"""

from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit("yfinance is required. Install with: pip install yfinance") from e

TRADING_DAYS = 252

# --------------------------- Data & Indicators --------------------------- #

def fetch_data(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker and dates.")
    df = df[['Open','High','Low','Close','Volume']].copy()
    df.index = pd.to_datetime(df.index)
    return df


def add_indicators(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    if short_window >= long_window:
        raise ValueError("short_window must be strictly less than long_window")
    out = df.copy()
    out[f'SMA_{short_window}'] = out['Close'].rolling(short_window, min_periods=short_window).mean()
    out[f'SMA_{long_window}'] = out['Close'].rolling(long_window, min_periods=long_window).mean()
    return out.dropna()

# --------------------------- Strategy & Signals -------------------------- #

def generate_signals(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    out = add_indicators(df, short_window, long_window)
    s_short = out[f'SMA_{short_window}']
    s_long = out[f'SMA_{long_window}']

    # signal: 1 if long, 0 if flat
    out['signal'] = (s_short > s_long).astype(int)
    # trades: +1 buy when flipping to 1, -1 sell when flipping to 0
    out['trade'] = out['signal'].diff().fillna(0)

    # shift signal by 1 to avoid lookahead (enter at next bar open/close)
    out['signal_exec'] = out['signal'].shift(1).fillna(0)
    out['trade_exec'] = out['trade'].shift(1).fillna(0)
    return out

# --------------------------- Backtest Engine ----------------------------- #

@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_bps: float = 0.0          # per trade (both sides), in basis points (1 bps = 0.01%)
    slippage_bps: float = 0.0     # simulated slippage per trade in bps
    size: float = 1.0             # fraction of capital to allocate when long (0..1)


def simulate(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    out = df.copy()

    # Use Close-to-Close returns with signal_exec applied
    out['ret'] = out['Close'].pct_change().fillna(0.0)

    # Costs model: apply when a trade is executed
    cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0  # convert bps to decimal
    out['cost'] = np.where(out['trade_exec'] != 0, cost_rate, 0.0)

    # Strategy return before costs: position * market return * size
    out['strat_ret_gross'] = out['signal_exec'] * cfg.size * out['ret']

    # Net return subtracting costs on trade days (approximate as a one-off loss)
    out['strat_ret'] = out['strat_ret_gross'] - out['cost']

    # Equity curves
    out['equity'] = cfg.initial_capital * (1.0 + out['strat_ret']).cumprod()
    out['bh_equity'] = cfg.initial_capital * (1.0 + out['ret']).cumprod()

    # Trade counting & exposure
    out['turnover'] = np.abs(out['trade_exec'])  # 1 for a flip, 0 otherwise
    out['exposure'] = out['signal_exec'] * cfg.size

    return out

# --------------------------- Performance Metrics ------------------------ #

def cagr(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    start = equity.iloc[0]
    end = equity.iloc[-1]
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    return (end / start) ** (1 / years) - 1


def max_drawdown(equity: pd.Series) -> float:
    rolling_max = equity.cummax()
    dd = equity / rolling_max - 1.0
    return dd.min()


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    # rf is annual risk-free; assume daily compounding approximation
    daily_rf = (1 + rf) ** (1 / TRADING_DAYS) - 1
    excess = returns - daily_rf
    vol = excess.std(ddof=0)
    if vol == 0:
        return 0.0
    return math.sqrt(TRADING_DAYS) * excess.mean() / vol


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    daily_rf = (1 + rf) ** (1 / TRADING_DAYS) - 1
    excess = returns - daily_rf
    downside = excess[excess < 0]
    dd = downside.std(ddof=0)
    if dd == 0:
        return 0.0
    return math.sqrt(TRADING_DAYS) * excess.mean() / dd


def summarize(bt: pd.DataFrame, cfg: BacktestConfig) -> Dict[str, float]:
    eq = bt['equity']
    returns = bt['strat_ret']

    stats = {
        'Initial Capital': cfg.initial_capital,
        'Final Equity': float(eq.iloc[-1]),
        'CAGR': cagr(eq),
        'Sharpe': sharpe(returns),
        'Sortino': sortino(returns),
        'Max Drawdown': max_drawdown(eq),
        'Calmar': (cagr(eq) / abs(max_drawdown(eq)) if max_drawdown(eq) < 0 else np.nan),
        'Buy&Hold Final': float(bt['bh_equity'].iloc[-1]),
        'Trades': int(bt['turnover'].sum()),
        'Avg Daily Turnover': float(bt['turnover'].mean()),
        'Avg Daily Exposure': float(bt['exposure'].mean()),
    }
    return stats

# --------------------------- Plotting ----------------------------------- #

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_all(bt: pd.DataFrame, short: int, long: int, ticker: str, outdir: str) -> None:
    ensure_dir(outdir)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 1) Price + SMAs + buy/sell markers
    axes[0].plot(bt.index, bt['Close'], label=f'{ticker} Close')
    axes[0].plot(bt.index, bt[f'SMA_{short}'], label=f'SMA {short}', linewidth=1)
    axes[0].plot(bt.index, bt[f'SMA_{long}'], label=f'SMA {long}', linewidth=1)

    buys = bt.index[bt['trade_exec'] > 0]
    sells = bt.index[bt['trade_exec'] < 0]
    axes[0].scatter(buys, bt.loc[buys, 'Close'], marker='^')
    axes[0].scatter(sells, bt.loc[sells, 'Close'], marker='v')
    axes[0].set_title(f'{ticker} — Price & Signals')
    axes[0].legend()

    # 2) Equity curves
    axes[1].plot(bt.index, bt['equity'], label='Strategy Equity')
    axes[1].plot(bt.index, bt['bh_equity'], label='Buy & Hold Equity', linestyle='--')
    axes[1].set_title('Equity Curves')
    axes[1].legend()

    # 3) Drawdown
    peak = bt['equity'].cummax()
    drawdown = bt['equity'] / peak - 1.0
    axes[2].plot(bt.index, drawdown, label='Drawdown')
    axes[2].axhline(0, linestyle='--')
    axes[2].set_title('Drawdown')

    fig.tight_layout()
    outpath = os.path.join(outdir, 'report.png')
    fig.savefig(outpath, dpi=140)
    plt.close(fig)

# --------------------------- CLI ---------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Simple MA crossover backtester')
    p.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol (e.g., AAPL)')
    p.add_argument('--start', type=str, default='2015-01-01', help='Start date YYYY-MM-DD')
    p.add_argument('--end', type=str, default=None, help='End date YYYY-MM-DD (optional)')
    p.add_argument('--short', type=int, default=20, help='Short SMA window')
    p.add_argument('--long', type=int, default=50, help='Long SMA window')
    p.add_argument('--capital', type=float, default=10000.0, help='Initial capital')
    p.add_argument('--fee-bps', type=float, default=0.0, help='Per-trade fee (bps)')
    p.add_argument('--slip-bps', type=float, default=0.0, help='Per-trade slippage (bps)')
    p.add_argument('--size', type=float, default=1.0, help='Position size fraction (0..1)')
    p.add_argument('--outdir', type=str, default='results', help='Directory to write results')
    return p.parse_args()


def main():
    args = parse_args()

    cfg = BacktestConfig(
        initial_capital=args.capital,
        fee_bps=args.fee_bps,
        slippage_bps=args.slip_bps,
        size=args.size,
    )

    print("Downloading data…")
    raw = fetch_data(args.ticker, args.start, args.end)

    print("Generating signals…")
    sigs = generate_signals(raw, args.short, args.long)

    print("Simulating portfolio…")
    bt = simulate(sigs, cfg)

    print("Computing performance…")
    stats = summarize(bt, cfg)

    ensure_dir(args.outdir)
    # Save CSV of the backtest timeline
    csv_path = os.path.join(args.outdir, 'backtest_timeseries.csv')
    bt.to_csv(csv_path)

    # Save metrics
    metrics_path = os.path.join(args.outdir, 'metrics.csv')
    pd.Series(stats).to_csv(metrics_path)

    print("Plotting…")
    plot_all(bt, args.short, args.long, args.ticker, args.outdir)

    print("Done. Results written to:")
    print(f" - {csv_path}")
    print(f" - {metrics_path}")
    print(f" - {os.path.join(args.outdir, 'report.png')}")

    # Pretty print metrics
    print("\nPerformance Summary")
    for k, v in stats.items():
        if isinstance(v, float):
            if abs(v) < 1 and 'Final' not in k and 'Capital' not in k:
                print(f"{k:>18}: {v:.4f}")
            else:
                print(f"{k:>18}: {v:,.2f}")
        else:
            print(f"{k:>18}: {v}")


if __name__ == '__main__':
    main()
