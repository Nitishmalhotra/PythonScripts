"""
Strategy Examples for Backtesting Engine
=========================================

This module contains example trading strategies that can be backtested.
Each strategy returns a DataFrame with signals and optional stop loss/target levels.

Signal format:
- signal: 1 for BUY, 0 for NO ACTION, -1 for SELL
- stop_loss: Optional stop loss price
- target: Optional target price
- trailing_stop_percent: Optional trailing stop percentage
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange


def moving_average_crossover_strategy(
    data: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
    stop_loss_percent: float = 2.0,
    target_percent: float = 5.0
) -> pd.DataFrame:
    """
    Simple Moving Average Crossover Strategy
    
    Entry: When fast MA crosses above slow MA
    Exit: Stop loss or target hit
    
    Args:
        data: OHLCV DataFrame
        fast_period: Fast MA period
        slow_period: Slow MA period
        stop_loss_percent: Stop loss percentage below entry
        target_percent: Target percentage above entry
    """
    df = data.copy()
    
    # Calculate indicators
    df['SMA_fast'] = SMAIndicator(close=df['close'], window=fast_period).sma_indicator()
    df['SMA_slow'] = SMAIndicator(close=df['close'], window=slow_period).sma_indicator()
    
    # Generate signals
    df['signal'] = 0
    df['prev_fast'] = df['SMA_fast'].shift(1)
    df['prev_slow'] = df['SMA_slow'].shift(1)
    
    # Bullish crossover
    crossover = (df['prev_fast'] <= df['prev_slow']) & (df['SMA_fast'] > df['SMA_slow'])
    df.loc[crossover, 'signal'] = 1
    
    # Calculate stop loss and target
    df['stop_loss'] = df['close'] * (1 - stop_loss_percent / 100)
    df['target'] = df['close'] * (1 + target_percent / 100)
    df['trailing_stop_percent'] = None
    
    return df[['signal', 'stop_loss', 'target', 'trailing_stop_percent']]


def rsi_mean_reversion_strategy(
    data: pd.DataFrame,
    rsi_period: int = 14,
    oversold_threshold: int = 30,
    overbought_threshold: int = 70,
    stop_loss_percent: float = 3.0,
    target_percent: float = 6.0
) -> pd.DataFrame:
    """
    RSI Mean Reversion Strategy
    
    Entry: When RSI crosses above oversold level
    Exit: When RSI crosses above overbought level OR stop loss/target
    
    Args:
        data: OHLCV DataFrame
        rsi_period: RSI calculation period
        oversold_threshold: RSI level considered oversold
        overbought_threshold: RSI level considered overbought
        stop_loss_percent: Stop loss percentage
        target_percent: Target percentage
    """
    df = data.copy()
    
    # Calculate RSI
    df['RSI'] = RSIIndicator(close=df['close'], window=rsi_period).rsi()
    
    # Generate signals
    df['signal'] = 0
    df['prev_RSI'] = df['RSI'].shift(1)
    
    # Buy when RSI crosses above oversold
    buy_signal = (df['prev_RSI'] <= oversold_threshold) & (df['RSI'] > oversold_threshold)
    df.loc[buy_signal, 'signal'] = 1
    
    # Calculate stop loss and target
    df['stop_loss'] = df['close'] * (1 - stop_loss_percent / 100)
    df['target'] = df['close'] * (1 + target_percent / 100)
    df['trailing_stop_percent'] = None
    
    return df[['signal', 'stop_loss', 'target', 'trailing_stop_percent']]


def bollinger_breakout_strategy(
    data: pd.DataFrame,
    bb_period: int = 20,
    bb_std: int = 2,
    stop_loss_atr_multiplier: float = 1.5,
    target_atr_multiplier: float = 3.0,
    trailing_stop_percent: float = 1.5
) -> pd.DataFrame:
    """
    Bollinger Bands Breakout Strategy
    
    Entry: When price closes above upper band with volume confirmation
    Exit: Trailing stop or target
    
    Args:
        data: OHLCV DataFrame
        bb_period: Bollinger Bands period
        bb_std: Number of standard deviations
        stop_loss_atr_multiplier: Stop loss as multiple of ATR
        target_atr_multiplier: Target as multiple of ATR
        trailing_stop_percent: Trailing stop percentage
    """
    df = data.copy()
    
    # Calculate Bollinger Bands
    bb = BollingerBands(close=df['close'], window=bb_period, window_dev=bb_std)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()
    
    # Calculate ATR for stop loss/target
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ATR'] = atr.average_true_range()
    
    # Calculate volume moving average
    df['Volume_MA'] = df['volume'].rolling(window=20).mean()
    
    # Generate signals
    df['signal'] = 0
    
    # Breakout signal: Close above upper band with higher volume
    breakout = (
        (df['close'] > df['BB_upper']) &
        (df['volume'] > df['Volume_MA'] * 1.2)
    )
    df.loc[breakout, 'signal'] = 1
    
    # Calculate stop loss and target based on ATR
    df['stop_loss'] = df['close'] - (df['ATR'] * stop_loss_atr_multiplier)
    df['target'] = df['close'] + (df['ATR'] * target_atr_multiplier)
    df['trailing_stop_percent'] = trailing_stop_percent
    
    return df[['signal', 'stop_loss', 'target', 'trailing_stop_percent']]


def macd_momentum_strategy(
    data: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    stop_loss_percent: float = 2.5,
    target_percent: float = 5.0
) -> pd.DataFrame:
    """
    MACD Momentum Strategy
    
    Entry: When MACD crosses above signal line and MACD > 0
    Exit: Stop loss or target
    
    Args:
        data: OHLCV DataFrame
        macd_fast: MACD fast period
        macd_slow: MACD slow period
        macd_signal: MACD signal period
        stop_loss_percent: Stop loss percentage
        target_percent: Target percentage
    """
    df = data.copy()
    
    # Calculate MACD
    macd = MACD(
        close=df['close'],
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_signal
    )
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    # Generate signals
    df['signal'] = 0
    df['prev_MACD'] = df['MACD'].shift(1)
    df['prev_MACD_signal'] = df['MACD_signal'].shift(1)
    
    # Bullish crossover when MACD is positive
    crossover = (
        (df['prev_MACD'] <= df['prev_MACD_signal']) &
        (df['MACD'] > df['MACD_signal']) &
        (df['MACD'] > 0)
    )
    df.loc[crossover, 'signal'] = 1
    
    # Calculate stop loss and target
    df['stop_loss'] = df['close'] * (1 - stop_loss_percent / 100)
    df['target'] = df['close'] * (1 + target_percent / 100)
    df['trailing_stop_percent'] = None
    
    return df[['signal', 'stop_loss', 'target', 'trailing_stop_percent']]


def momentum_breakout_strategy(
    data: pd.DataFrame,
    lookback_period: int = 20,
    volume_multiplier: float = 1.5,
    stop_loss_percent: float = 2.0,
    trailing_stop_percent: float = 2.0
) -> pd.DataFrame:
    """
    Momentum Breakout Strategy
    
    Entry: When price breaks above N-day high with volume surge
    Exit: Trailing stop
    
    Args:
        data: OHLCV DataFrame
        lookback_period: Period for high calculation
        volume_multiplier: Volume surge multiplier
        stop_loss_percent: Initial stop loss percentage
        trailing_stop_percent: Trailing stop percentage
    """
    df = data.copy()
    
    # Calculate N-day high
    df['High_N'] = df['high'].rolling(window=lookback_period).max()
    
    # Calculate volume moving average
    df['Volume_MA'] = df['volume'].rolling(window=20).mean()
    
    # Calculate RSI for confirmation
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    
    # Generate signals
    df['signal'] = 0
    
    # Breakout signal
    breakout = (
        (df['close'] > df['High_N'].shift(1)) &  # Price breaks above previous high
        (df['volume'] > df['Volume_MA'] * volume_multiplier) &  # Volume surge
        (df['RSI'] > 50)  # RSI confirmation
    )
    df.loc[breakout, 'signal'] = 1
    
    # Calculate stop loss and target
    df['stop_loss'] = df['close'] * (1 - stop_loss_percent / 100)
    df['target'] = None  # Use trailing stop instead
    df['trailing_stop_percent'] = trailing_stop_percent
    
    return df[['signal', 'stop_loss', 'target', 'trailing_stop_percent']]


def support_resistance_bounce_strategy(
    data: pd.DataFrame,
    support_lookback: int = 50,
    proximity_percent: float = 1.0,
    stop_loss_percent: float = 1.5,
    target_percent: float = 4.0
) -> pd.DataFrame:
    """
    Support/Resistance Bounce Strategy
    
    Entry: When price bounces near support level
    Exit: Stop loss or target
    
    Args:
        data: OHLCV DataFrame
        support_lookback: Period to identify support
        proximity_percent: How close price should be to support (%)
        stop_loss_percent: Stop loss percentage
        target_percent: Target percentage
    """
    df = data.copy()
    
    # Calculate support (rolling minimum)
    df['Support'] = df['low'].rolling(window=support_lookback).min()
    
    # Calculate RSI
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    
    # Generate signals
    df['signal'] = 0
    
    # Price near support with oversold RSI
    near_support = (
        (df['low'] <= df['Support'] * (1 + proximity_percent / 100)) &
        (df['close'] > df['Support']) &  # Didn't break support
        (df['RSI'] < 40)  # Oversold
    )
    df.loc[near_support, 'signal'] = 1
    
    # Calculate stop loss and target
    df['stop_loss'] = df['Support'] * 0.99  # Just below support
    df['target'] = df['close'] * (1 + target_percent / 100)
    df['trailing_stop_percent'] = None
    
    return df[['signal', 'stop_loss', 'target', 'trailing_stop_percent']]


def ema_crossover_with_volume_strategy(
    data: pd.DataFrame,
    ema_fast: int = 9,
    ema_slow: int = 21,
    volume_threshold: float = 1.3,
    stop_loss_percent: float = 2.0,
    trailing_stop_percent: float = 1.5
) -> pd.DataFrame:
    """
    EMA Crossover with Volume Confirmation
    
    Entry: Fast EMA crosses above slow EMA with volume confirmation
    Exit: Trailing stop
    
    Args:
        data: OHLCV DataFrame
        ema_fast: Fast EMA period
        ema_slow: Slow EMA period
        volume_threshold: Volume multiplier for confirmation
        stop_loss_percent: Initial stop loss percentage
        trailing_stop_percent: Trailing stop percentage
    """
    df = data.copy()
    
    # Calculate EMAs
    df['EMA_fast'] = EMAIndicator(close=df['close'], window=ema_fast).ema_indicator()
    df['EMA_slow'] = EMAIndicator(close=df['close'], window=ema_slow).ema_indicator()
    
    # Calculate volume MA
    df['Volume_MA'] = df['volume'].rolling(window=20).mean()
    
    # Calculate trend strength
    df['ADX'] = 0  # Placeholder for ADX
    
    # Generate signals
    df['signal'] = 0
    df['prev_fast'] = df['EMA_fast'].shift(1)
    df['prev_slow'] = df['EMA_slow'].shift(1)
    
    # Bullish crossover with volume
    crossover = (
        (df['prev_fast'] <= df['prev_slow']) &
        (df['EMA_fast'] > df['EMA_slow']) &
        (df['volume'] > df['Volume_MA'] * volume_threshold) &
        (df['close'] > df['EMA_slow'])  # Price above slow EMA
    )
    df.loc[crossover, 'signal'] = 1
    
    # Calculate stop loss and target
    df['stop_loss'] = df['close'] * (1 - stop_loss_percent / 100)
    df['target'] = None
    df['trailing_stop_percent'] = trailing_stop_percent
    
    return df[['signal', 'stop_loss', 'target', 'trailing_stop_percent']]


def combined_ma_rsi_volume_strategy(
    data: pd.DataFrame,
    fast_ma: int = 10,
    slow_ma: int = 30,
    rsi_period: int = 14,
    rsi_threshold: int = 50,
    volume_multiplier: float = 1.3,
    stop_loss_percent: float = 2.5,
    target_percent: float = 6.0
) -> pd.DataFrame:
    """
    Combined Strategy: MA Crossover + RSI Filter + Volume Confirmation
    
    Entry conditions (ALL must be true):
    - Fast MA crosses above Slow MA
    - RSI > threshold (confirming uptrend)
    - Volume > average volume * multiplier (volume surge)
    - Price > Slow MA (price in uptrend)
    
    Exit: Stop loss or target
    
    Args:
        data: OHLCV DataFrame
        fast_ma: Fast moving average period
        slow_ma: Slow moving average period
        rsi_period: RSI calculation period
        rsi_threshold: Minimum RSI for entry (50-60 range)
        volume_multiplier: Volume surge multiplier
        stop_loss_percent: Stop loss percentage
        target_percent: Target percentage
    """
    df = data.copy()
    
    # Calculate MAs
    df['MA_fast'] = SMAIndicator(close=df['close'], window=fast_ma).sma_indicator()
    df['MA_slow'] = SMAIndicator(close=df['close'], window=slow_ma).sma_indicator()
    
    # Calculate RSI
    df['RSI'] = RSIIndicator(close=df['close'], window=rsi_period).rsi()
    
    # Calculate volume MA
    df['Volume_MA'] = df['volume'].rolling(window=20).mean()
    
    # Calculate ATR for adaptive stop loss
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ATR'] = atr.average_true_range()
    
    # Generate signals
    df['signal'] = 0
    df['prev_fast'] = df['MA_fast'].shift(1)
    df['prev_slow'] = df['MA_slow'].shift(1)
    
    # COMBINED ENTRY CONDITIONS:
    # 1. MA Crossover
    ma_crossover = (df['prev_fast'] <= df['prev_slow']) & (df['MA_fast'] > df['MA_slow'])
    
    # 2. RSI Filter (not overbought, confirming strength)
    rsi_filter = (df['RSI'] > rsi_threshold) & (df['RSI'] < 80)
    
    # 3. Volume Confirmation
    volume_filter = df['volume'] > df['Volume_MA'] * volume_multiplier
    
    # 4. Price Trend Confirmation
    price_filter = df['close'] > df['MA_slow']
    
    # ALL CONDITIONS MUST BE TRUE
    entry_signal = ma_crossover & rsi_filter & volume_filter & price_filter
    df.loc[entry_signal, 'signal'] = 1
    
    # Calculate stop loss and target
    df['stop_loss'] = df['close'] * (1 - stop_loss_percent / 100)
    df['target'] = df['close'] * (1 + target_percent / 100)
    df['trailing_stop_percent'] = None
    
    return df[['signal', 'stop_loss', 'target', 'trailing_stop_percent']]


# Strategy registry for easy access
STRATEGY_REGISTRY = {
    'ma_crossover': moving_average_crossover_strategy,
    'rsi_mean_reversion': rsi_mean_reversion_strategy,
    'bollinger_breakout': bollinger_breakout_strategy,
    'macd_momentum': macd_momentum_strategy,
    'momentum_breakout': momentum_breakout_strategy,
    'support_resistance': support_resistance_bounce_strategy,
    'ema_crossover': ema_crossover_with_volume_strategy,
    'combined_strategy': combined_ma_rsi_volume_strategy
}


def get_strategy(strategy_name: str):
    """Get strategy function by name"""
    return STRATEGY_REGISTRY.get(strategy_name)


def list_strategies():
    """List all available strategies"""
    print("\nAvailable Strategies:")
    print("=" * 60)
    for name in STRATEGY_REGISTRY.keys():
        print(f"  - {name}")
    print("=" * 60)


if __name__ == "__main__":
    list_strategies()
