# 🚀 Quick Start Guide - Backtesting Engine

## Installation (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python -c "import pandas, numpy, matplotlib, ta; print('✅ All dependencies installed!')"
```

## Running Your First Backtest (2 minutes)

### Option 1: Using Sample Data
```bash
python example_backtest.py
# Select option 1 for basic backtest
```

### Option 2: With Your Kite Data

```python
from backtesting_engine import BacktestingEngine, print_metrics
from strategy_examples import moving_average_crossover_strategy
import pandas as pd

# Your Kite historical data
data = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02', ...],
    'open': [1000, 1010, ...],
    'high': [1020, 1025, ...],
    'low': [995, 1005, ...],
    'close': [1015, 1020, ...],
    'volume': [100000, 120000, ...]
})

# Convert date to datetime
data['date'] = pd.to_datetime(data['date'])

# Initialize engine
engine = BacktestingEngine(initial_capital=100000)

# Run backtest
trades, equity = engine.run_backtest(
    data=data,
    strategy_function=moving_average_crossover_strategy,
    strategy_name='MA Strategy',
    fast_period=20,
    slow_period=50
)

# See results
metrics = engine.calculate_metrics(trades, equity)
print_metrics(metrics)
```

## What You'll Get

### 1. Performance Metrics
```
📊 Trade Statistics:
   Total Trades: 45
   Win Rate: 62.22%
   Sharpe Ratio: 1.543

💰 P&L Statistics:
   Net P&L: ₹15,234.50
   Total Return: 15.23%
   Profit Factor: 1.85

📈 Risk Metrics:
   Max Drawdown: -8.45%
   Calmar Ratio: 1.80
```

### 2. Visual Charts
- Equity curve with drawdown
- Trade P&L distribution
- Win/Loss analysis
- Monthly returns heatmap

### 3. Detailed Trade Log
All trades with entry/exit prices, P&L, holding periods, etc.

## Common Use Cases

### 1. Test a Strategy on Your Stock
```python
# Fetch your stock data from Kite
# Run backtest with default parameters
# Analyze results
```

### 2. Find Best Parameters
```python
from parameter_optimizer import ParameterOptimizer

optimizer = ParameterOptimizer(engine)
results = optimizer.grid_search(
    data=data,
    strategy_function=your_strategy,
    param_grid={'param1': [10, 20, 30], 'param2': [2, 3, 5]}
)
optimizer.print_best_parameters()
```

### 3. Validate Strategy (Avoid Overfitting)
```python
from parameter_optimizer import WalkForwardAnalysis

wfa = WalkForwardAnalysis(engine)
results = wfa.run_walk_forward(
    data=data,
    strategy_function=your_strategy,
    param_grid={...}
)
# Check if out-of-sample performance is acceptable
```

### 4. Compare Strategies
```python
# Test multiple strategies
# Compare visually
# Pick the best one for live trading
```

## Next Steps

1. ✅ **Test with sample data** (5 min)
2. ✅ **Integrate with your Kite account** (10 min)
3. ✅ **Backtest your current strategy** (15 min)
4. ✅ **Optimize parameters** (30 min)
5. ✅ **Walk-forward validation** (30 min)
6. ✅ **Paper trade** (1 week)
7. ✅ **Go live** (when confident)

## Pro Tips

💡 **Start Conservative**
- Use 5-10% position sizes
- Set realistic stop losses (2-3%)
- Test on multiple stocks/timeframes

💡 **Validate Thoroughly**
- Always use walk-forward analysis
- Test in different market conditions
- Check performance on unseen data

💡 **Monitor Live Performance**
- Compare live results to backtest
- Adjust if significant deviation
- Keep detailed logs

## Troubleshooting

**Q: No trades generated?**
A: Strategy might be too strict. Try relaxing parameters.

**Q: Too many losing trades?**
A: Add filters (trend, volume) or tighter stops.

**Q: Good backtest, bad live results?**
A: Possible overfitting. Use walk-forward analysis.

## Support

📖 Full documentation: See README.md
💻 Example code: See example_backtest.py
📊 Strategy templates: See strategy_examples.py

---

**Happy Backtesting! 📈**
