# 🎯 Backtesting Quick Reference Guide

## 📝 Overview
This guide shows how to use the backtesting module to validate strategies before deploying them to live scanning.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Run Example Backtest
```bash
cd Backtesting
python example_backtest.py
```

**What it does:**
- Runs 4 pre-built strategies on sample data
- Generates performance metrics
- Creates visualization charts
- Compares strategy performance

**Expected Output:**
```
✅ Backtesting Results:
   - MA Crossover: Sharpe 1.45, Win Rate 58%
   - RSI Mean Reversion: Sharpe 1.67, Win Rate 62%
   - Bollinger Breakout: Sharpe 1.23, Win Rate 54%
   - MACD Momentum: Sharpe 1.89, Win Rate 65%

📊 Charts saved to: results/
```

---

### Step 2: Optimize Parameters (Optional)
```python
from parameter_optimizer import ParameterOptimizer
from backtesting_engine import BacktestingEngine
from strategy_examples import moving_average_crossover_strategy

# Initialize
engine = BacktestingEngine(initial_capital=100000)
optimizer = ParameterOptimizer(engine, optimization_metric='sharpe_ratio')

# Define parameter grid
param_grid = {
    'fast_period': [5, 10, 20],
    'slow_period': [50, 100, 200]
}

# Run optimization
results = optimizer.grid_search(
    data=historical_data,
    strategy_function=moving_average_crossover_strategy,
    strategy_name='MA Crossover',
    param_grid=param_grid,
    n_jobs=4  # Parallel processing
)

# Best parameters
best = results.iloc[0]
print(f"Best: fast={best['fast_period']}, slow={best['slow_period']}")
print(f"Sharpe: {best['sharpe_ratio']:.2f}")
```

---

### Step 3: Deploy to Live Scanner
```python
# 1. Copy optimized strategy to advanced_scanner.py
# 2. Add to scan_with_strategies() method
# 3. Test with automated_scanner.py
```

---

## 📊 Available Strategies

### 1. **Moving Average Crossover** 📈
```python
from strategy_examples import moving_average_crossover_strategy

# Parameters:
# - fast_period: Fast MA period (default: 20)
# - slow_period: Slow MA period (default: 50)

# Signal:
# BUY when fast MA crosses above slow MA
# SELL when fast MA crosses below slow MA
```

### 2. **RSI Mean Reversion** 🔄
```python
from strategy_examples import rsi_mean_reversion_strategy

# Parameters:
# - rsi_period: RSI calculation period (default: 14)
# - oversold: Oversold threshold (default: 30)
# - overbought: Overbought threshold (default: 70)

# Signal:
# BUY when RSI < oversold
# SELL when RSI > overbought
```

### 3. **Bollinger Breakout** 💥
```python
from strategy_examples import bollinger_breakout_strategy

# Parameters:
# - period: BB period (default: 20)
# - std_dev: Standard deviation (default: 2)

# Signal:
# BUY when price breaks above upper band
# SELL when price breaks below lower band
```

### 4. **MACD Momentum** ⚡
```python
from strategy_examples import macd_momentum_strategy

# Parameters:
# - fast: Fast EMA (default: 12)
# - slow: Slow EMA (default: 26)
# - signal: Signal line (default: 9)

# Signal:
# BUY when MACD crosses above signal
# SELL when MACD crosses below signal
```

---

## 🔍 Performance Metrics Explained

### **Return Metrics**
- `total_pnl`: Gross profit/loss ($)
- `net_pnl`: Profit after commissions ($)
- `total_pnl_percent`: Return percentage
- `return_on_capital`: ROI on initial capital

### **Win/Loss Metrics**
- `win_rate`: % of winning trades
- `profit_factor`: Total wins / Total losses
- `avg_win`: Average winning trade ($)
- `avg_loss`: Average losing trade ($)

### **Risk Metrics**
- `sharpe_ratio`: Risk-adjusted returns (>1.0 good, >2.0 excellent)
- `sortino_ratio`: Downside risk-adjusted returns
- `max_drawdown`: Biggest equity decline ($)
- `max_drawdown_percent`: Biggest decline (%)
- `calmar_ratio`: Return / Max DD (higher = better)

### **Trading Stats**
- `total_trades`: Number of completed trades
- `avg_holding_period`: Average days in trade
- `max_consecutive_wins`: Longest win streak
- `max_consecutive_losses`: Longest loss streak

---

## 📈 Interpreting Results

### **Good Strategy Characteristics:**
✅ Sharpe Ratio > 1.5
✅ Win Rate > 55%
✅ Profit Factor > 1.5
✅ Max Drawdown < 20%
✅ Calmar Ratio > 0.5

### **Warning Signs:**
⚠️ Sharpe Ratio < 0.5
⚠️ Win Rate < 45%
⚠️ Profit Factor < 1.2
⚠️ Max Drawdown > 30%
⚠️ Too few trades (< 20)

### **Overfitting Indicators:**
🚨 Win Rate > 80% (too good to be true)
🚨 Perfect performance on backtest
🚨 Poor walk-forward validation
🚨 Large IS/OOS performance gap

---

## 🛠️ Common Workflows

### **Workflow 1: Test New Strategy**
```python
from backtesting_engine import BacktestingEngine, print_metrics
import pandas as pd

# 1. Load data
data = pd.read_csv('historical_data.csv')

# 2. Initialize engine
engine = BacktestingEngine(
    initial_capital=100000,
    commission_percent=0.03,
    slippage_percent=0.05,
    position_size_percent=10,
    max_positions=5
)

# 3. Define strategy
def my_strategy(row, position, context):
    """Your custom strategy logic"""
    if position is None:  # No position
        # Buy condition
        if row['close'] > row['sma_50']:
            return {
                'action': 'BUY',
                'stop_loss': row['close'] * 0.95,
                'target': row['close'] * 1.10
            }
    else:  # Have position
        # Sell condition
        if row['close'] < row['sma_50']:
            return {'action': 'SELL'}
    return {'action': 'HOLD'}

# 4. Run backtest
metrics = engine.run_backtest(data, my_strategy, 'My Strategy')

# 5. Print results
print_metrics(metrics)
```

---

### **Workflow 2: Optimize Existing Strategy**
```python
from parameter_optimizer import ParameterOptimizer

# Initialize optimizer
optimizer = ParameterOptimizer(engine, 'sharpe_ratio')

# Define parameter ranges
param_grid = {
    'ma_period': [10, 20, 30, 50],
    'stop_loss_pct': [2, 3, 5, 7],
    'target_pct': [5, 10, 15, 20]
}

# Run grid search
results = optimizer.grid_search(
    data=data,
    strategy_function=my_strategy,
    strategy_name='Optimized Strategy',
    param_grid=param_grid,
    n_jobs=4
)

# View top 5 combinations
print(results.head())

# Use best parameters
best_params = results.iloc[0]
print(f"Optimal: MA={best_params['ma_period']}, "
      f"SL={best_params['stop_loss_pct']}%, "
      f"Target={best_params['target_pct']}%")
```

---

### **Workflow 3: Walk-Forward Validation**
```python
from parameter_optimizer import WalkForwardAnalysis

# Initialize walk-forward
wfa = WalkForwardAnalysis(engine)

# Run analysis
wf_results = wfa.run_walk_forward(
    data=data,
    strategy_function=my_strategy,
    strategy_name='My Strategy',
    param_grid=param_grid,
    train_period_months=12,  # Train on 12 months
    test_period_months=3,    # Test on 3 months
    optimization_metric='sharpe_ratio'
)

# Check consistency
print(f"In-Sample Sharpe: {wf_results['in_sample_sharpe']:.2f}")
print(f"Out-of-Sample Sharpe: {wf_results['out_sample_sharpe']:.2f}")
print(f"Degradation: {wf_results['degradation']:.1f}%")

# Good if degradation < 30%
```

---

### **Workflow 4: Strategy Comparison**
```python
from backtest_visualizer import BacktestVisualizer

# Backtest multiple strategies
strategies = [
    (moving_average_crossover_strategy, 'MA Crossover'),
    (rsi_mean_reversion_strategy, 'RSI MR'),
    (bollinger_breakout_strategy, 'BB Breakout')
]

results = {}
for strategy_func, name in strategies:
    metrics = engine.run_backtest(data, strategy_func, name)
    results[name] = metrics

# Visualize comparison
viz = BacktestVisualizer()
viz.plot_strategy_comparison(results)
viz.show()
```

---

## 🔗 Integration with Live Scanner

### **Step-by-Step Integration:**

**1. Backtest Strategy**
```python
# In Backtesting/strategy_examples.py
def my_validated_strategy(row, position, context):
    # Your strategy logic
    pass
```

**2. Extract Logic**
```python
# Convert backtest format to scanner format
def scanner_version(df):
    """Convert for use in advanced_scanner.py"""
    # df is full DataFrame with all indicators
    # Return DataFrame with signals
    pass
```

**3. Add to Scanner**
```python
# In Active_Production/advanced_scanner.py
def scan_my_strategy(self, df):
    """Your strategy in scanner format"""
    # Copy logic from backtest
    # Adapt to scanner's signal format
    signals = []
    # ... your logic ...
    return pd.DataFrame(signals)

# Add to scan_with_strategies()
results['My Strategy'] = self.scan_my_strategy(df)
```

**4. Test Live**
```bash
python Active_Production\automated_scanner.py
```

---

## 📊 File Locations

### **Backtesting Files:**
```
Backtesting/
├── backtesting_engine.py          # Core engine
├── backtest_visualizer.py         # Charts
├── parameter_optimizer.py         # Optimization
├── strategy_examples.py           # Strategy library
└── example_backtest.py            # Usage examples
```

### **Output Locations:**
```
Results/
├── equity_curve.png               # Performance chart
├── pnl_distribution.png           # P&L histogram
├── monthly_returns.png            # Heatmap
├── strategy_comparison.png        # Multi-strategy
└── backtest_report.csv            # Detailed trades
```

---

## 🎯 Best Practices

### **DO:**
✅ Always run walk-forward validation
✅ Test on at least 2-3 years of data
✅ Use realistic commission and slippage
✅ Keep position sizing reasonable (5-10%)
✅ Limit max concurrent positions
✅ Document strategy assumptions
✅ Compare multiple timeframes

### **DON'T:**
❌ Over-optimize (curve fitting)
❌ Test on too little data (< 1 year)
❌ Ignore commission/slippage
❌ Use 100% position sizes
❌ Rely on single backtest
❌ Skip walk-forward validation
❌ Deploy without live testing

---

## 🆘 Troubleshooting

### **Issue: Low number of trades**
**Solution:** Adjust strategy sensitivity, use longer data period

### **Issue: High drawdown**
**Solution:** Reduce position size, add stop losses, limit max positions

### **Issue: Poor walk-forward results**
**Solution:** Strategy overfitted - simplify logic, use broader parameters

### **Issue: Inconsistent performance**
**Solution:** Market regime change - add filters, test on different periods

---

## 📚 Next Steps

1. **Run example_backtest.py** to see how it works
2. **Modify a strategy** in strategy_examples.py
3. **Optimize parameters** using parameter_optimizer.py
4. **Validate** with walk-forward analysis
5. **Deploy** to Active_Production/advanced_scanner.py
6. **Monitor** live performance vs backtest

---

*Ready to backtest? Start with: `python example_backtest.py`*
