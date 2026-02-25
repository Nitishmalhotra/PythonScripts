"""
Complete Backtesting Example
=============================

This script demonstrates how to:
1. Fetch historical data from Kite API
2. Run backtests on multiple strategies
3. Optimize parameters
4. Perform walk-forward analysis
5. Generate visualizations and reports

Usage:
    python example_backtest.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
import os
import sys
from pathlib import Path

# Import backtesting modules
from backtesting_engine import BacktestingEngine, print_metrics
from strategy_examples import (
    moving_average_crossover_strategy,
    rsi_mean_reversion_strategy,
    bollinger_breakout_strategy,
    macd_momentum_strategy,
    STRATEGY_REGISTRY
)
from parameter_optimizer import ParameterOptimizer, WalkForwardAnalysis
from backtest_visualizer import BacktestVisualizer

# Get project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Global Kite instance and instruments cache
KITE_INSTANCE = None
INSTRUMENTS_CACHE = None


def load_kite_credentials():
    """Load Kite API credentials from kite_credentials.txt"""
    credentials_file = PROJECT_ROOT / 'kite_credentials.txt'
    
    if not credentials_file.exists():
        raise FileNotFoundError(f"Credentials file not found: {credentials_file}")
    
    credentials = {}
    with open(credentials_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                # Store with lowercase keys for consistency (handles both API_KEY and api_key formats)
                credentials[key.strip().lower()] = value.strip()
    
    return credentials.get('api_key'), credentials.get('access_token')


def initialize_kite():
    """Initialize Kite API connection"""
    global KITE_INSTANCE, INSTRUMENTS_CACHE
    
    if KITE_INSTANCE is not None:
        return KITE_INSTANCE
    
    try:
        print("🔌 Connecting to Kite API...")
        api_key, access_token = load_kite_credentials()
        
        if not api_key or not access_token:
            raise ValueError("API key or access token missing in credentials file")
        
        KITE_INSTANCE = KiteConnect(api_key=api_key)
        KITE_INSTANCE.set_access_token(access_token)
        
        # Load instruments for NSE
        print("📋 Loading NSE instruments...")
        INSTRUMENTS_CACHE = KITE_INSTANCE.instruments('NSE')
        print(f"✅ Loaded {len(INSTRUMENTS_CACHE)} instruments\n")
        
        return KITE_INSTANCE
        
    except Exception as e:
        print(f"❌ Failed to initialize Kite API: {e}")
        print("⚠️  Falling back to sample data\n")
        return None


def get_instrument_token(symbol: str) -> int:
    """Get instrument token for a symbol"""
    global INSTRUMENTS_CACHE
    
    if INSTRUMENTS_CACHE is None:
        return None
    
    # Search for symbol in instruments
    for inst in INSTRUMENTS_CACHE:
        if inst['tradingsymbol'] == symbol and inst['instrument_type'] == 'EQ':
            return inst['instrument_token']
    
    # Try with -EQ suffix
    for inst in INSTRUMENTS_CACHE:
        if inst['tradingsymbol'] == f"{symbol}-EQ":
            return inst['instrument_token']
    
    return None


def fetch_historical_data_from_kite(
    symbol: str,
    from_date: str,
    to_date: str,
    interval: str = 'day'
) -> pd.DataFrame:
    """
    Fetch historical data from Kite API
    
    Args:
        symbol: Trading symbol (e.g., 'SBIN')
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        interval: Candle interval
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"📥 Fetching historical data for {symbol}...")
    print(f"   Period: {from_date} to {to_date}")
    
    # Initialize Kite if needed
    kite = initialize_kite()
    
    if kite is None:
        print("⚠️  Using sample data (Kite API not available)\n")
        return generate_sample_data(from_date, to_date, symbol)
    
    try:
        # Get instrument token
        instrument_token = get_instrument_token(symbol)
        
        if instrument_token is None:
            print(f"⚠️  Symbol {symbol} not found, using sample data\n")
            return generate_sample_data(from_date, to_date, symbol)
        
        print(f"   Instrument Token: {instrument_token}")
        
        # Convert dates to datetime
        from_dt = datetime.strptime(from_date, '%Y-%m-%d')
        to_dt = datetime.strptime(to_date, '%Y-%m-%d')
        
        # Fetch historical data
        historical_data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_dt,
            to_date=to_dt,
            interval=interval
        )
        
        if not historical_data:
            print(f"⚠️  No data received for {symbol}, using sample data\n")
            return generate_sample_data(from_date, to_date, symbol)
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df['symbol'] = symbol
        
        # Rename columns to match expected format
        df = df.rename(columns={'oi': 'open_interest'})
        
        print(f"✅ Fetched {len(df)} candles for {symbol}\n")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        print("⚠️  Using sample data\n")
        return generate_sample_data(from_date, to_date, symbol)


def generate_sample_data(from_date: str, to_date: str, symbol: str = 'SAMPLE') -> pd.DataFrame:
    """Generate sample OHLCV data for demonstration"""
    dates = pd.date_range(start=from_date, end=to_date, freq='D')
    
    # Generate realistic price movement
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    
    close_prices = [1000]  # Starting price
    for ret in returns[1:]:
        close_prices.append(close_prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'date': dates,
        'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in close_prices],
        'high': [p * (1 + abs(np.random.uniform(0, 0.02))) for p in close_prices],
        'low': [p * (1 - abs(np.random.uniform(0, 0.02))) for p in close_prices],
        'close': close_prices,
        'volume': np.random.randint(100000, 1000000, len(dates)),
        'symbol': symbol
    })
    
    return data


def example_1_basic_backtest():
    """Example 1: Run a basic backtest"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Backtest - Moving Average Crossover")
    print("="*80 + "\n")
    
    # Fetch REAL data from Kite API
    data = fetch_historical_data_from_kite(
        symbol='SBIN',
        from_date='2023-01-01',
        to_date='2026-02-11',
        interval='day'
    )
    
    # Initialize backtesting engine
    engine = BacktestingEngine(
        initial_capital=100000,
        commission_percent=0.03,
        position_size_percent=10,
        max_positions=3
    )
    
    # Run backtest
    trades, equity_curve = engine.run_backtest(
        data=data,
        strategy_function=moving_average_crossover_strategy,
        strategy_name='MA Crossover',
        fast_period=20,
        slow_period=50,
        stop_loss_percent=2.0,
        target_percent=5.0
    )
    
    # Calculate metrics
    metrics = engine.calculate_metrics(trades, equity_curve)
    
    # Print results
    print_metrics(metrics, "MA Crossover Strategy")
    
    # Visualize results
    visualizer = BacktestVisualizer()
    visualizer.plot_equity_curve(equity_curve, "MA Crossover - Equity Curve")
    visualizer.plot_trade_distribution(trades)
    
    return trades, equity_curve, metrics


def example_2_parameter_optimization():
    """Example 2: Optimize strategy parameters"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Parameter Optimization - RSI Strategy")
    print("="*80 + "\n")
    
    # Fetch data
    data = fetch_historical_data_from_kite(
        symbol='TATASTEEL',
        from_date='2024-01-01',
        to_date='2026-02-01',
        interval='day'
    )
    
    # Initialize engine
    engine = BacktestingEngine(initial_capital=100000)
    
    # Define parameter grid
    param_grid = {
        'rsi_period': [10, 14, 20],
        'oversold_threshold': [25, 30, 35],
        'stop_loss_percent': [2.0, 3.0],
        'target_percent': [5.0, 7.0]
    }
    
    # Run optimization
    optimizer = ParameterOptimizer(engine, optimization_metric='sharpe_ratio')
    results = optimizer.grid_search(
        data=data,
        strategy_function=rsi_mean_reversion_strategy,
        strategy_name='RSI Mean Reversion',
        param_grid=param_grid,
        n_jobs=1
    )
    
    # Print best parameters
    optimizer.print_best_parameters(top_n=5)
    
    # Test best parameters
    best_params = results.iloc[0]
    print(f"\n🏆 Testing Best Parameters:")
    print(f"   RSI Period: {int(best_params['rsi_period'])}")
    print(f"   Oversold: {int(best_params['oversold_threshold'])}")
    print(f"   Stop Loss: {best_params['stop_loss_percent']:.1f}%")
    print(f"   Target: {best_params['target_percent']:.1f}%\n")
    
    trades, equity_curve = engine.run_backtest(
        data=data,
        strategy_function=rsi_mean_reversion_strategy,
        strategy_name='RSI Optimized',
        rsi_period=int(best_params['rsi_period']),
        oversold_threshold=int(best_params['oversold_threshold']),
        stop_loss_percent=best_params['stop_loss_percent'],
        target_percent=best_params['target_percent']
    )
    
    metrics = engine.calculate_metrics(trades, equity_curve)
    print_metrics(metrics, "RSI Optimized Strategy")
    
    return results


def example_3_walk_forward_analysis():
    """Example 3: Walk-forward analysis"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Walk-Forward Analysis - MACD Strategy")
    print("="*80 + "\n")
    
    # Fetch longer period data
    data = fetch_historical_data_from_kite(
        symbol='BEL',
        from_date='2023-01-01',
        to_date='2026-02-01',
        interval='day'
    )
    
    # Initialize engine
    engine = BacktestingEngine(initial_capital=100000)
    
    # Define parameter grid for optimization
    param_grid = {
        'macd_fast': [10, 12, 15],
        'macd_slow': [24, 26, 30],
        'stop_loss_percent': [2.0, 2.5],
        'target_percent': [4.0, 5.0]
    }
    
    # Run walk-forward analysis
    wfa = WalkForwardAnalysis(
        backtesting_engine=engine,
        in_sample_percent=0.7,
        n_splits=4
    )
    
    results = wfa.run_walk_forward(
        data=data,
        strategy_function=macd_momentum_strategy,
        strategy_name='MACD Momentum',
        param_grid=param_grid,
        optimization_metric='sharpe_ratio'
    )
    
    print("\n📊 Walk-Forward Results:")
    print(results[['split', 'is_sharpe_ratio', 'oos_sharpe_ratio', 
                   'oos_win_rate', 'oos_net_pnl']].to_string(index=False))
    
    return results


def example_4_compare_strategies():
    """Example 4: Compare multiple strategies"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Strategy Comparison")
    print("="*80 + "\n")
    
    # Fetch data
    data = fetch_historical_data_from_kite(
        symbol='AXISBANK',
        from_date='2024-01-01',
        to_date='2026-02-01',
        interval='day'
    )
    
    # Initialize engine
    engine = BacktestingEngine(initial_capital=100000)
    
    # Test multiple strategies
    strategies_to_test = {
        'MA Crossover': (moving_average_crossover_strategy, {
            'fast_period': 20, 'slow_period': 50,
            'stop_loss_percent': 2.0, 'target_percent': 5.0
        }),
        'RSI Mean Reversion': (rsi_mean_reversion_strategy, {
            'rsi_period': 14, 'oversold_threshold': 30,
            'stop_loss_percent': 2.5, 'target_percent': 6.0
        }),
        'Bollinger Breakout': (bollinger_breakout_strategy, {
            'bb_period': 20, 'bb_std': 2,
            'stop_loss_atr_multiplier': 1.5,
            'target_atr_multiplier': 3.0,
            'trailing_stop_percent': 1.5
        }),
        'MACD Momentum': (macd_momentum_strategy, {
            'macd_fast': 12, 'macd_slow': 26,
            'stop_loss_percent': 2.5, 'target_percent': 5.0
        })
    }
    
    results = {}
    
    for strategy_name, (strategy_func, params) in strategies_to_test.items():
        print(f"\n📊 Testing {strategy_name}...")
        
        trades, equity_curve = engine.run_backtest(
            data=data,
            strategy_function=strategy_func,
            strategy_name=strategy_name,
            **params
        )
        
        metrics = engine.calculate_metrics(trades, equity_curve)
        
        results[strategy_name] = {
            'equity_curve': equity_curve,
            'metrics': metrics,
            'trades': trades
        }
        
        print_metrics(metrics, strategy_name)
    
    # Compare strategies visually
    visualizer = BacktestVisualizer()
    visualizer.plot_strategy_comparison(
        results,
        title='Strategy Performance Comparison'
    )
    
    # Summary table
    print("\n📊 Strategy Comparison Summary:")
    print("=" * 100)
    
    summary_data = []
    for strategy_name, data in results.items():
        metrics = data['metrics']
        summary_data.append({
            'Strategy': strategy_name,
            'Total Trades': metrics.total_trades,
            'Win Rate': f"{metrics.win_rate:.2f}%",
            'Sharpe': f"{metrics.sharpe_ratio:.3f}",
            'Max DD': f"{metrics.max_drawdown_percent:.2f}%",
            'Net P&L': f"₹{metrics.net_pnl:,.0f}",
            'Return': f"{metrics.total_pnl_percent:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("=" * 100 + "\n")
    
    return results


def example_5_comprehensive_report():
    """Example 5: Generate comprehensive report"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Comprehensive Backtest Report")
    print("="*80 + "\n")
    
    # Fetch data
    data = fetch_historical_data_from_kite(
        symbol='RELIANCE',
        from_date='2024-01-01',
        to_date='2026-02-01',
        interval='day'
    )
    
    # Initialize engine
    engine = BacktestingEngine(
        initial_capital=100000,
        commission_percent=0.03,
        position_size_percent=15,
        max_positions=2
    )
    
    # Run backtest
    trades, equity_curve = engine.run_backtest(
        data=data,
        strategy_function=bollinger_breakout_strategy,
        strategy_name='Bollinger Breakout',
        bb_period=20,
        bb_std=2,
        stop_loss_atr_multiplier=1.5,
        target_atr_multiplier=3.0,
        trailing_stop_percent=1.5
    )
    
    # Calculate metrics
    metrics = engine.calculate_metrics(trades, equity_curve)
    
    # Generate comprehensive report
    report = engine.generate_report(
        strategy_name='Bollinger Breakout',
        trades=trades,
        metrics=metrics,
        equity_curve=equity_curve,
        save_path='backtest_report.json'
    )
    
    print(f"\n✅ Report saved to: backtest_report.json")
    
    # Generate all visualizations
    visualizer = BacktestVisualizer()
    
    # Create output directory
    import os
    os.makedirs('backtest_charts', exist_ok=True)
    
    visualizer.create_comprehensive_report(
        trades=trades,
        equity_curve=equity_curve,
        metrics=metrics,
        strategy_name='Bollinger_Breakout',
        save_dir='backtest_charts'
    )
    
    print(f"\n✅ Charts saved to: backtest_charts/")
    
    return report


def main():
    """Main function to run all examples"""
    print("\n" + "="*80)
    print("BACKTESTING ENGINE - COMPREHENSIVE EXAMPLES")
    print("="*80)
    
    examples = {
        '1': ('Basic Backtest', example_1_basic_backtest),
        '2': ('Parameter Optimization', example_2_parameter_optimization),
        '3': ('Walk-Forward Analysis', example_3_walk_forward_analysis),
        '4': ('Strategy Comparison', example_4_compare_strategies),
        '5': ('Comprehensive Report', example_5_comprehensive_report),
        'all': ('Run All Examples', None)
    }
    
    print("\nAvailable Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    choice = input("\nSelect example to run (1-5 or 'all'): ").strip()
    
    if choice == 'all':
        for key in ['1', '2', '3', '4', '5']:
            examples[key][1]()
    elif choice in examples and choice != 'all':
        examples[choice][1]()
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "="*80)
    print("✅ Examples completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Quick test mode - uncomment to test single stock quickly
    # print("\n🚀 QUICK TEST MODE - Running backtest on SBIN stock\n")
    # example_1_basic_backtest()
    
    # Interactive mode - run menu
    main()
