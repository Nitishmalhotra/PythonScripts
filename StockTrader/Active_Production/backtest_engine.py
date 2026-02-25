"""
Standalone backtest engine extracted from advanced_scanner.py.
Provides walk-forward strategy backtesting with ATR-based trade simulation.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Walk-forward backtester for scanner strategies.
    """

    STRATEGY_PARAMS = {
        'Momentum Breakout': dict(stop=2.0, t1=3.5, t2=5.0, hold=25),
        'Mean Reversion':    dict(stop=1.2, t1=2.0, t2=3.0, hold=10),
        'Trend Following':   dict(stop=2.5, t1=4.0, t2=6.0, hold=30),
        'Volume Breakout':   dict(stop=2.0, t1=3.0, t2=4.5, hold=15),
        'Swing Trading':     dict(stop=2.0, t1=3.5, t2=5.0, hold=20),
        'Gap Up':            dict(stop=1.2, t1=2.0, t2=3.0, hold=10),
        'RSI Setup':         dict(stop=1.5, t1=2.5, t2=3.5, hold=12),
        'Stage 2 Uptrend':   dict(stop=2.5, t1=4.0, t2=6.0, hold=40),
        'Strong Linearity':  dict(stop=1.8, t1=3.0, t2=4.5, hold=15),
        'VCP Pattern':       dict(stop=1.5, t1=2.5, t2=4.0, hold=20),
        'Pyramiding':        dict(stop=1.5, t1=2.5, t2=3.5, hold=20),
        'Golden Crossover':  dict(stop=3.0, t1=5.0, t2=7.0, hold=40),
    }
    DEFAULT_PARAMS = dict(stop=1.5, t1=2.5, t2=3.5, hold=20)

    def __init__(self, capital: float = 100_000, risk_pct: float = 0.02):
        self.capital = capital
        self.risk_pct = risk_pct

    def _position_size(self, entry: float, stop: float) -> int:
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return 0
        return max(1, int(self.capital * self.risk_pct / risk_per_share))

    def _simulate_trade(
        self,
        df: pd.DataFrame,
        signal_idx: int,
        stop_mult: float = 1.5,
        t1_mult: float = 2.5,
        t2_mult: float = 3.5,
        max_hold: int = 20,
    ) -> dict | None:
        entry_row_idx = signal_idx + 1
        if entry_row_idx >= len(df):
            return None

        entry_price = df['open'].iloc[entry_row_idx]
        atr = df['atr'].iloc[signal_idx]
        if atr <= 0 or entry_price <= 0:
            return None

        stop = entry_price - stop_mult * atr
        t1 = entry_price + t1_mult * atr
        t2 = entry_price + t2_mult * atr
        shares = self._position_size(entry_price, stop)
        if shares == 0:
            return None

        half = max(1, shares // 2)
        t1_hit = False
        trail = stop
        pnl = 0.0
        exit_price = entry_price
        exit_reason = 'max_hold'
        exit_date = df.index[min(entry_row_idx + max_hold - 1, len(df) - 1)]

        for i in range(entry_row_idx, min(entry_row_idx + max_hold, len(df))):
            lo = df['low'].iloc[i]
            hi = df['high'].iloc[i]
            cl = df['close'].iloc[i]

            if lo <= trail:
                exit_price = trail
                exit_reason = 'stop_loss'
                exit_date = df.index[i]
                pnl = (exit_price - entry_price) * (half if t1_hit else shares)
                break

            if not t1_hit and hi >= t1:
                pnl += (t1 - entry_price) * half
                trail = entry_price
                t1_hit = True

            if t1_hit and hi >= t2:
                pnl += (t2 - entry_price) * half
                exit_price = t2
                exit_reason = 'target_2'
                exit_date = df.index[i]
                break

            if i == min(entry_row_idx + max_hold - 1, len(df) - 1):
                pnl = (cl - entry_price) * (half if t1_hit else shares)
                exit_price = cl
                exit_date = df.index[i]

        return {
            'entry_date': df.index[entry_row_idx],
            'exit_date': exit_date,
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'stop': round(stop, 2),
            'target_1': round(t1, 2),
            'target_2': round(t2, 2),
            'shares': shares,
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl / (entry_price * shares) * 100, 2),
            'exit_reason': exit_reason,
            'hold_days': max(1, i - entry_row_idx + 1),
            'atr': round(atr, 2),
        }

    def _aggregate_trades(self, trades: list, strategy: str, symbol: str) -> dict:
        if not trades:
            return {}

        df_t = pd.DataFrame(trades)
        n = len(df_t)
        wins = df_t[df_t['pnl'] > 0]
        losses = df_t[df_t['pnl'] <= 0]
        win_rate = len(wins) / n * 100
        avg_win = wins['pnl_pct'].mean() if len(wins) else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) else 0
        total_pnl = df_t['pnl'].sum()

        equity = df_t['pnl'].cumsum()
        roll_max = equity.cummax()
        drawdown = equity - roll_max
        max_dd = drawdown.min()

        pnl_pct_std = df_t['pnl_pct'].std()
        sharpe = (df_t['pnl_pct'].mean() / pnl_pct_std * np.sqrt(252) if pnl_pct_std > 0 else 0)

        gross_win = wins['pnl'].sum() if len(wins) else 0
        gross_loss = abs(losses['pnl'].sum()) if len(losses) else 1
        pf = gross_win / gross_loss if gross_loss else float('inf')

        exit_dist = df_t['exit_reason'].value_counts().to_dict()

        return {
            'strategy': strategy,
            'symbol': symbol,
            'total_trades': n,
            'win_rate_pct': round(win_rate, 1),
            'avg_win_pct': round(avg_win, 2),
            'avg_loss_pct': round(avg_loss, 2),
            'total_pnl_inr': round(total_pnl, 2),
            'max_drawdown_inr': round(max_dd, 2),
            'profit_factor': round(pf, 2),
            'sharpe_ratio': round(sharpe, 2),
            'avg_hold_days': round(df_t['hold_days'].mean(), 1),
            'stop_loss_exits': exit_dist.get('stop_loss', 0),
            'target1_exits': exit_dist.get('target_1', 0),
            'target2_exits': exit_dist.get('target_2', 0),
            'time_exits': exit_dist.get('max_hold', 0),
            'trades': trades,
        }

    def run(
        self,
        stock_data: pd.DataFrame,
        strategy_func,
        strategy_name: str,
        test_pct: float = 0.3,
    ) -> dict:
        all_summaries = []
        all_trades = []

        for symbol, sym_df in stock_data.groupby('symbol'):
            sym_df = sym_df.sort_values('date').reset_index(drop=True)

            n_rows = len(sym_df)
            if n_rows < 60:
                continue

            split = int(n_rows * (1 - test_pct))
            sim_df = sym_df.set_index('date')
            params = self.STRATEGY_PARAMS.get(strategy_name, self.DEFAULT_PARAMS)

            trades = []
            in_trade_until = -1

            for i in range(split, n_rows):
                if i <= in_trade_until:
                    continue

                window = sym_df.iloc[:i + 1].copy()
                try:
                    result = strategy_func(window)
                except Exception:
                    continue

                if result is None or result.empty:
                    continue

                entry_idx = i + 1
                if entry_idx >= n_rows:
                    continue

                try:
                    trade = self._simulate_trade(
                        sim_df,
                        i,
                        stop_mult=params['stop'],
                        t1_mult=params['t1'],
                        t2_mult=params['t2'],
                        max_hold=params['hold'],
                    )
                except Exception as e:
                    logger.debug(f"[BT] trade sim error {symbol} bar {i}: {e}")
                    continue

                if trade:
                    trade['symbol'] = symbol
                    trade['strategy'] = strategy_name
                    trades.append(trade)
                    all_trades.append(trade)
                    in_trade_until = i + trade.get('hold_days', params['hold'])

            if trades:
                summary = self._aggregate_trades(trades, strategy_name, symbol)
                if summary:
                    all_summaries.append(summary)

        overall = {}
        if all_trades:
            overall = self._aggregate_trades(all_trades, strategy_name, 'ALL_SYMBOLS')
            logger.info(
                f"[Backtest] {strategy_name}: "
                f"trades={overall['total_trades']}, "
                f"win_rate={overall['win_rate_pct']:.1f}%, "
                f"PF={overall['profit_factor']:.2f}, "
                f"Sharpe={overall['sharpe_ratio']:.2f}"
            )
        else:
            logger.info(f"[Backtest] {strategy_name}: trades=0 — no test-window signals produced")

        return {
            'strategy': strategy_name,
            'per_symbol': all_summaries,
            'overall': overall,
            'all_trades': all_trades,
        }


def run_backtest(stock_data: pd.DataFrame, strategy_map: dict, test_pct: float = 0.3) -> dict:
    """
    Run walk-forward backtests for a strategy map.

    Args:
        stock_data: DataFrame with OHLCV + indicators + symbol/date columns.
        strategy_map: Dict of {'Strategy Name': strategy_function}
        test_pct: Fraction of history to reserve as out-of-sample window.

    Returns:
        Dict keyed by strategy name with overall/per-symbol/all-trades blocks.
    """
    engine = BacktestEngine()
    backtest_results = {}

    bt_logger = logging.getLogger(__name__)
    prev_level = bt_logger.level
    bt_logger.setLevel(logging.DEBUG)

    for name, func in strategy_map.items():
        logger.info(f"[Backtest] Running {name} ...")
        try:
            backtest_results[name] = engine.run(stock_data, func, name, test_pct)
        except Exception as e:
            logger.error(f"[Backtest] {name} failed: {e}")

    bt_logger.setLevel(prev_level)
    return backtest_results


def build_default_strategy_map(scanner) -> dict:
    """
    Build the default backtest strategy map from an AdvancedStockScanner-like object.
    """
    return {
        'Momentum Breakout': scanner.strategy_momentum_breakout,
        'Mean Reversion': scanner.strategy_mean_reversion,
        'Trend Following': scanner.strategy_trend_following,
        'Volume Breakout': scanner.strategy_volume_breakout,
        'Swing Trading': scanner.strategy_swing_trading,
        'Gap Up': scanner.strategy_gap_up,
        'RSI Setup': scanner.strategy_rsi_setup,
        'Stage 2 Uptrend': scanner.strategy_stage_2,
        'Strong Linearity': scanner.strategy_strong_linearity,
        'VCP Pattern': scanner.strategy_vcp,
        'Pyramiding': scanner.strategy_pyramiding,
        'Golden Crossover': scanner.strategy_golden_crossover,
    }
