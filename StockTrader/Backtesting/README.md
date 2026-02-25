# 📊 Kite Trading Backtesting Engine

A comprehensive, production-ready backtesting framework for testing and optimizing trading strategies using Kite Connect API.

## 🌟 Features

### Core Backtesting
- ✅ **Historical Data Replay** - Realistic order execution with proper entry/exit logic
- ✅ **Commission & Slippage** - Accurate cost modeling (0.03% commission, 0.05% slippage)
- ✅ **Position Sizing** - Flexible position sizing (default 10% of capital per trade)
- ✅ **Risk Management** - Stop loss, targets, and trailing stops
- ✅ **Multiple Positions** - Support for concurrent positions (configurable max)

### Performance Metrics
- 📈 **Returns**: Total P&L, Net P&L, Return on Capital
- 📊 **Win/Loss**: Win Rate, Profit Factor, Avg Win/Loss
- 📉 **Risk Metrics**: Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio
- ⏱️ **Trading Stats**: Holding periods, consecutive wins/losses
- 💸 **Costs**: Commission tracking, net profitability

### Advanced Features
- 🔍 **Parameter Optimization**
  - Grid Search - Exhaustive parameter search
  - Random Search - Efficient random sampling
  - Multi-core support for faster optimization
  
- 🚶 **Walk-Forward Analysis**
  - Out-of-sample validation
  - Prevents overfitting
  - Configurable IS/OOS splits
  
- 📊 **Comprehensive Visualizations**
  - Equity curves with drawdowns
  - Trade P&L distribution
  - Win/Loss analysis
  - Monthly returns heatmap
  - Strategy comparison charts

### Strategy Library
- **Moving Average Crossover** - Classic trend following
- **RSI Mean Reversion** - Oversold/overbought entries
- **Bollinger Breakout** - Volatility breakouts
- **MACD Momentum** - Momentum trading
- **Support/Resistance Bounce** - Level-based entries
- **EMA Crossover with Volume** - Volume-confirmed trends
- **Momentum Breakout** - Price/volume breakouts

## 📁 File Structure

```
backtesting_engine/
├── backtesting_engine.py      # Core backtesting engine
├── strategy_examples.py        # Pre-built trading strategies
├── parameter_optimizer.py      # Parameter optimization tools
├── backtest_visualizer.py      # Visualization module
├── example_backtest.py         # Complete usage examples
└── README.md                   # This file
```

## 🚀 Quick Start

### Installation

```bash
# Install required dependencies
pip install pandas numpy matplotlib seaborn ta kiteconnect
```

### Basic Usage

```python
from backtesting_engine import BacktestingEngine, print_metrics
from strategy_examples import moving_average_crossover_strategy
import pandas as pd

# 1. Prepare your data (from Kite API or CSV)
data = pd.DataFrame({
    'date': [...],
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# 2. Initialize backtesting engine
engine = BacktestingEngine(
    initial_capital=100000,      # ₹1,00,000
    commission_percent=0.03,     # 0.03% per trade
    position_size_percent=10,    # 10% of capital per trade
    max_positions=3              # Max 3 concurrent positions
)

# 3. Run backtest
trades, equity_curve = engine.run_backtest(
    data=data,
    strategy_function=moving_average_crossover_strategy,
    strategy_name='MA Crossover',
    fast_period=20,
    slow_period=50,
    stop_loss_percent=2.0,
    target_percent=5.0
)

# 4. Calculate and print metrics
metrics = engine.calculate_metrics(trades, equity_curve)
print_metrics(metrics, "MA Crossover Strategy")

# 5. Visualize results
from backtest_visualizer import BacktestVisualizer

visualizer = BacktestVisualizer()
visualizer.plot_equity_curve(equity_curve)
visualizer.plot_trade_distribution(trades)
```

## 📖 Detailed Examples

### Example 1: Parameter Optimization

```python
from parameter_optimizer import ParameterOptimizer
from strategy_examples import rsi_mean_reversion_strategy

# Define parameter grid
param_grid = {
    'rsi_period': [10, 14, 20],
    'oversold_threshold': [25, 30, 35],
    'stop_loss_percent': [2.0, 3.0],
    'target_percent': [5.0, 7.0]
}

# Initialize optimizer
optimizer = ParameterOptimizer(engine, optimization_metric='sharpe_ratio')

# Run grid search
results = optimizer.grid_search(
    data=data,
    strategy_function=rsi_mean_reversion_strategy,
    strategy_name='RSI Strategy',
    param_grid=param_grid,
    n_jobs=4  # Use 4 CPU cores
)

# Get best parameters
optimizer.print_best_parameters(top_n=5)
```

### Example 2: Walk-Forward Analysis

```python
from parameter_optimizer import WalkForwardAnalysis
from strategy_examples import macd_momentum_strategy

# Initialize walk-forward analyzer
wfa = WalkForwardAnalysis(
    backtesting_engine=engine,
    in_sample_percent=0.7,  # 70% for optimization
    n_splits=5               # 5 walk-forward windows
)

# Run analysis
results = wfa.run_walk_forward(
    data=data,
    strategy_function=macd_momentum_strategy,
    strategy_name='MACD',
    param_grid={
        'macd_fast': [10, 12, 15],
        'macd_slow': [24, 26, 30]
    },
    optimization_metric='sharpe_ratio'
)

# Results show in-sample vs out-of-sample performance
print(results)
```

### Example 3: Strategy Comparison

```python
from backtest_visualizer import BacktestVisualizer

# Test multiple strategies
strategies = {
    'MA Crossover': (moving_average_crossover_strategy, {...}),
    'RSI Mean Reversion': (rsi_mean_reversion_strategy, {...}),
    'MACD Momentum': (macd_momentum_strategy, {...})
}

results = {}
for name, (strategy_func, params) in strategies.items():
    trades, equity_curve = engine.run_backtest(
        data=data,
        strategy_function=strategy_func,
        strategy_name=name,
        **params
    )
    metrics = engine.calculate_metrics(trades, equity_curve)
    results[name] = {
        'equity_curve': equity_curve,
        'metrics': metrics
    }

# Compare strategies
visualizer = BacktestVisualizer()
visualizer.plot_strategy_comparison(results)
```

## 🔧 Creating Custom Strategies

Strategies must return a DataFrame with these columns:
- `signal`: 1 for BUY, 0 for NO ACTION
- `stop_loss`: Optional stop loss price
- `target`: Optional target price
- `trailing_stop_percent`: Optional trailing stop %

```python
def my_custom_strategy(data: pd.DataFrame, param1: int, param2: float):
    """
    Your custom strategy logic
    
    Args:
        data: OHLCV DataFrame
        param1: Your parameter 1
        param2: Your parameter 2
    
    Returns:
        DataFrame with signals
    """
    df = data.copy()
    
    # Calculate your indicators
    df['indicator'] = calculate_indicator(df['close'], param1)
    
    # Generate signals
    df['signal'] = 0
    df.loc[df['indicator'] > param2, 'signal'] = 1
    
    # Set stop loss and target
    df['stop_loss'] = df['close'] * 0.98  # 2% below entry
    df['target'] = df['close'] * 1.05     # 5% above entry
    df['trailing_stop_percent'] = None
    
    return df[['signal', 'stop_loss', 'target', 'trailing_stop_percent']]
```

## 📊 Performance Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Win Rate** | % of profitable trades | > 50% |
| **Profit Factor** | Gross profit / Gross loss | > 1.5 |
| **Sharpe Ratio** | Risk-adjusted returns | > 1.0 |
| **Sortino Ratio** | Downside risk-adjusted returns | > 1.5 |
| **Max Drawdown** | Largest peak-to-trough loss | < 20% |
| **Calmar Ratio** | Return / Max Drawdown | > 1.0 |

## 🎯 Best Practices

### 1. Data Quality
```python
# Ensure your data is clean
data = data.dropna()
data = data.sort_values('date')
data = data.reset_index(drop=True)
```

### 2. Realistic Parameters
```python
engine = BacktestingEngine(
    initial_capital=100000,
    commission_percent=0.03,  # Realistic Zerodha brokerage
    slippage_percent=0.05,    # Account for market impact
    position_size_percent=10, # Don't overconcentrate
    max_positions=5           # Diversify
)
```

### 3. Avoid Overfitting
```python
# Always use walk-forward analysis
wfa = WalkForwardAnalysis(
    in_sample_percent=0.7,  # 70% for training
    n_splits=5              # Multiple validation periods
)
```

### 4. Test Multiple Timeframes
```python
# Test on different periods
periods = [
    ('Bull Market', '2020-04-01', '2021-02-01'),
    ('Bear Market', '2021-02-01', '2022-06-01'),
    ('Sideways', '2022-06-01', '2023-12-01')
]

for name, start, end in periods:
    data = fetch_data(start, end)
    # Run backtest...
```

## ⚠️ Limitations & Disclaimers

1. **Past Performance ≠ Future Results**
   - Backtesting shows historical performance only
   - Market conditions change
   
2. **Survivorship Bias**
   - Ensure your data includes delisted stocks
   
3. **Look-Ahead Bias**
   - Don't use future data in indicators
   - Our engine prevents this by design
   
4. **Market Impact**
   - Slippage model is simplified
   - May not reflect actual execution in illiquid stocks
   
5. **Fundamental Events**
   - Backtests don't account for news, earnings, etc.
   - Use with caution around major events

## 🔍 Troubleshooting

### Issue: "No trades generated"
**Solution**: Check if your strategy is too strict. Lower thresholds or adjust parameters.

### Issue: "Too many trades"
**Solution**: Add filters (volume, trend confirmation, etc.) to reduce noise.

### Issue: "Poor Sharpe ratio despite good returns"
**Solution**: Returns are volatile. Add stop losses or reduce position sizes.

### Issue: "Optimization takes too long"
**Solution**: Use random search instead of grid search, or reduce parameter combinations.

## 📈 Integration with Your Trading System

### Step 1: Fetch Data from Kite
```python
from kiteconnect import KiteConnect

kite = KiteConnect(api_key="your_api_key")
kite.set_access_token("your_access_token")

# Get instrument token
instruments = kite.instruments("NSE")
sbin = [i for i in instruments if i['tradingsymbol'] == 'SBIN'][0]

# Fetch historical data
data = kite.historical_data(
    instrument_token=sbin['instrument_token'],
    from_date="2024-01-01",
    to_date="2026-02-01",
    interval="day"
)

# Convert to DataFrame
df = pd.DataFrame(data)
```

### Step 2: Run Backtest
```python
# Use the data in backtesting engine
trades, equity_curve = engine.run_backtest(
    data=df,
    strategy_function=your_strategy,
    strategy_name='Your Strategy',
    **params
)
```

### Step 3: Deploy Live
```python
# Once satisfied with backtest results:
# 1. Create automated scanner with same logic
# 2. Integrate with your order execution system
# 3. Monitor live performance vs backtest
```

## 🤝 Contributing

To add new strategies:
1. Create strategy function in `strategy_examples.py`
2. Add to `STRATEGY_REGISTRY`
3. Test with backtesting engine
4. Submit for review

## 📝 License

MIT License - Free to use for personal and commercial trading.

## 🙏 Acknowledgments

- Built for Kite Connect API
- Uses TA-Lib for technical indicators
- Inspired by Backtrader and Zipline

## 📞 Support

For issues or questions:
1. Check the examples in `example_backtest.py`
2. Review this README
3. Test with sample data first
4. Verify Kite API integration

---

**⚠️ Risk Warning**: Trading involves substantial risk of loss. This backtesting engine is for educational and research purposes. Always paper trade before going live. Past performance does not guarantee future results.

---

**Built with ❤️ for systematic traders using Zerodha Kite**
