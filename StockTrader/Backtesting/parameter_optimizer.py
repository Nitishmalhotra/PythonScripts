"""
Parameter Optimization Module
==============================

This module provides tools for optimizing strategy parameters through:
1. Grid Search - Exhaustive search through parameter combinations
2. Random Search - Random sampling of parameter space
3. Genetic Algorithm - Evolutionary optimization (future)

Helps find optimal parameters for trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Any
from itertools import product
import random
from datetime import datetime
from backtesting_engine import BacktestingEngine, PerformanceMetrics
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


class ParameterOptimizer:
    """
    Optimize strategy parameters using various methods
    """
    
    def __init__(
        self,
        backtesting_engine: BacktestingEngine,
        optimization_metric: str = 'sharpe_ratio'
    ):
        """
        Initialize optimizer
        
        Args:
            backtesting_engine: Configured backtesting engine
            optimization_metric: Metric to optimize (sharpe_ratio, total_pnl, win_rate, etc.)
        """
        self.engine = backtesting_engine
        self.optimization_metric = optimization_metric
        self.results = []
        
    def grid_search(
        self,
        data: pd.DataFrame,
        strategy_function: Callable,
        strategy_name: str,
        param_grid: Dict[str, List[Any]],
        n_jobs: int = 1
    ) -> pd.DataFrame:
        """
        Perform grid search over parameter space
        
        Args:
            data: Historical OHLCV data
            strategy_function: Strategy function to test
            strategy_name: Name of strategy
            param_grid: Dictionary of parameters and their possible values
                       e.g., {'fast_period': [10, 20, 30], 'slow_period': [50, 100]}
            n_jobs: Number of parallel jobs (1 for sequential)
            
        Returns:
            DataFrame with results for each parameter combination
        """
        print(f"\n🔍 Starting Grid Search for {strategy_name}")
        print(f"{'='*60}")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        total_combinations = len(combinations)
        print(f"Total combinations to test: {total_combinations}\n")
        
        results = []
        
        if n_jobs == 1:
            # Sequential processing
            for idx, params in enumerate(combinations, 1):
                param_dict = dict(zip(param_names, params))
                result = self._test_single_combination(
                    data, strategy_function, strategy_name, param_dict, idx, total_combinations
                )
                results.append(result)
        else:
            # Parallel processing
            print(f"Running with {n_jobs} parallel jobs...\n")
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                for idx, params in enumerate(combinations, 1):
                    param_dict = dict(zip(param_names, params))
                    future = executor.submit(
                        self._test_single_combination,
                        data, strategy_function, strategy_name, param_dict, idx, total_combinations
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    results.append(future.result())
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by optimization metric
        results_df = results_df.sort_values(self.optimization_metric, ascending=False)
        
        self.results = results_df
        
        print(f"\n{'='*60}")
        print(f"✅ Grid Search Complete!")
        print(f"{'='*60}\n")
        
        return results_df
    
    def _test_single_combination(
        self,
        data: pd.DataFrame,
        strategy_function: Callable,
        strategy_name: str,
        params: Dict,
        iteration: int,
        total: int
    ) -> Dict:
        """Test a single parameter combination"""
        
        print(f"Testing [{iteration}/{total}]: {params}")
        
        try:
            # Run backtest
            trades, equity_curve = self.engine.run_backtest(
                data=data,
                strategy_function=strategy_function,
                strategy_name=strategy_name,
                **params
            )
            
            # Calculate metrics
            metrics = self.engine.calculate_metrics(trades, equity_curve)
            
            # Combine parameters and metrics
            result = {**params, **metrics.to_dict()}
            
            return result
            
        except Exception as e:
            print(f"❌ Error testing {params}: {str(e)}")
            # Return empty metrics
            return {**params, **{k: 0 for k in PerformanceMetrics.__annotations__.keys()}}
    
    def random_search(
        self,
        data: pd.DataFrame,
        strategy_function: Callable,
        strategy_name: str,
        param_distributions: Dict[str, Tuple[Any, Any]],
        n_iterations: int = 50,
        n_jobs: int = 1
    ) -> pd.DataFrame:
        """
        Perform random search over parameter space
        
        Args:
            data: Historical OHLCV data
            strategy_function: Strategy function to test
            strategy_name: Name of strategy
            param_distributions: Dictionary of parameters and their ranges
                                e.g., {'fast_period': (5, 50), 'slow_period': (50, 200)}
            n_iterations: Number of random combinations to test
            n_jobs: Number of parallel jobs
            
        Returns:
            DataFrame with results
        """
        print(f"\n🎲 Starting Random Search for {strategy_name}")
        print(f"{'='*60}")
        print(f"Testing {n_iterations} random combinations\n")
        
        results = []
        
        for iteration in range(1, n_iterations + 1):
            # Generate random parameters
            params = {}
            for param_name, (min_val, max_val) in param_distributions.items():
                if isinstance(min_val, int):
                    params[param_name] = random.randint(min_val, max_val)
                else:
                    params[param_name] = random.uniform(min_val, max_val)
            
            result = self._test_single_combination(
                data, strategy_function, strategy_name, params, iteration, n_iterations
            )
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(self.optimization_metric, ascending=False)
        
        self.results = results_df
        
        print(f"\n{'='*60}")
        print(f"✅ Random Search Complete!")
        print(f"{'='*60}\n")
        
        return results_df
    
    def get_best_parameters(self, top_n: int = 5) -> pd.DataFrame:
        """
        Get top N parameter combinations
        
        Args:
            top_n: Number of top results to return
            
        Returns:
            DataFrame with top results
        """
        if self.results is None or len(self.results) == 0:
            print("No optimization results available. Run optimization first.")
            return pd.DataFrame()
        
        return self.results.head(top_n)
    
    def print_best_parameters(self, top_n: int = 5):
        """Print top N parameter combinations"""
        best = self.get_best_parameters(top_n)
        
        if len(best) == 0:
            return
        
        print(f"\n🏆 Top {top_n} Parameter Combinations (by {self.optimization_metric})")
        print(f"{'='*80}\n")
        
        for idx, row in best.iterrows():
            print(f"Rank #{idx + 1}")
            print(f"Parameters: {dict((k, v) for k, v in row.items() if k not in PerformanceMetrics.__annotations__)}")
            print(f"{self.optimization_metric}: {row[self.optimization_metric]:.4f}")
            print(f"Total Trades: {row['total_trades']}")
            print(f"Win Rate: {row['win_rate']:.2f}%")
            print(f"Sharpe Ratio: {row['sharpe_ratio']:.3f}")
            print(f"Max Drawdown: {row['max_drawdown_percent']:.2f}%")
            print(f"Net P&L: ₹{row['net_pnl']:,.2f}")
            print(f"{'-'*80}\n")


class WalkForwardAnalysis:
    """
    Walk-Forward Analysis for out-of-sample testing
    
    This technique divides data into multiple in-sample (IS) and out-of-sample (OOS) periods,
    optimizes on IS and tests on OOS to avoid overfitting.
    """
    
    def __init__(
        self,
        backtesting_engine: BacktestingEngine,
        in_sample_percent: float = 0.7,
        n_splits: int = 5
    ):
        """
        Initialize walk-forward analyzer
        
        Args:
            backtesting_engine: Configured backtesting engine
            in_sample_percent: Percentage of each window for in-sample optimization
            n_splits: Number of walk-forward splits
        """
        self.engine = backtesting_engine
        self.in_sample_percent = in_sample_percent
        self.n_splits = n_splits
        self.results = []
        
    def run_walk_forward(
        self,
        data: pd.DataFrame,
        strategy_function: Callable,
        strategy_name: str,
        param_grid: Dict[str, List[Any]],
        optimization_metric: str = 'sharpe_ratio'
    ) -> pd.DataFrame:
        """
        Run walk-forward analysis
        
        Args:
            data: Full historical dataset
            strategy_function: Strategy to test
            strategy_name: Name of strategy
            param_grid: Parameter grid for optimization
            optimization_metric: Metric to optimize on
            
        Returns:
            DataFrame with walk-forward results
        """
        print(f"\n🚶 Starting Walk-Forward Analysis for {strategy_name}")
        print(f"{'='*60}")
        print(f"Splits: {self.n_splits}")
        print(f"In-Sample: {self.in_sample_percent*100:.0f}%")
        print(f"Out-of-Sample: {(1-self.in_sample_percent)*100:.0f}%\n")
        
        # Calculate window size
        total_rows = len(data)
        window_size = total_rows // self.n_splits
        is_size = int(window_size * self.in_sample_percent)
        oos_size = window_size - is_size
        
        results = []
        
        for split in range(self.n_splits):
            start_idx = split * window_size
            is_end_idx = start_idx + is_size
            oos_end_idx = min(is_end_idx + oos_size, total_rows)
            
            # Skip if we don't have enough data for OOS
            if oos_end_idx - is_end_idx < 10:
                break
            
            print(f"\n📊 Split {split + 1}/{self.n_splits}")
            print(f"{'='*60}")
            
            # In-Sample period
            is_data = data.iloc[start_idx:is_end_idx].copy()
            print(f"In-Sample: {is_data.iloc[0]['date']} to {is_data.iloc[-1]['date']} ({len(is_data)} days)")
            
            # Out-of-Sample period
            oos_data = data.iloc[is_end_idx:oos_end_idx].copy()
            print(f"Out-of-Sample: {oos_data.iloc[0]['date']} to {oos_data.iloc[-1]['date']} ({len(oos_data)} days)")
            
            # Optimize on in-sample data
            print("\n🔍 Optimizing on In-Sample data...")
            optimizer = ParameterOptimizer(self.engine, optimization_metric)
            is_results = optimizer.grid_search(
                data=is_data,
                strategy_function=strategy_function,
                strategy_name=strategy_name,
                param_grid=param_grid,
                n_jobs=1
            )
            
            # Get best parameters
            best_params = is_results.iloc[0].to_dict()
            param_names = list(param_grid.keys())
            best_param_dict = {k: best_params[k] for k in param_names}
            
            print(f"\n✅ Best IS Parameters: {best_param_dict}")
            print(f"   IS {optimization_metric}: {best_params[optimization_metric]:.4f}")
            
            # Test on out-of-sample data
            print("\n📈 Testing on Out-of-Sample data...")
            oos_trades, oos_equity = self.engine.run_backtest(
                data=oos_data,
                strategy_function=strategy_function,
                strategy_name=strategy_name,
                **best_param_dict
            )
            
            oos_metrics = self.engine.calculate_metrics(oos_trades, oos_equity)
            
            print(f"   OOS {optimization_metric}: {getattr(oos_metrics, optimization_metric):.4f}")
            
            # Store results
            results.append({
                'split': split + 1,
                'is_start': is_data.iloc[0]['date'],
                'is_end': is_data.iloc[-1]['date'],
                'oos_start': oos_data.iloc[0]['date'],
                'oos_end': oos_data.iloc[-1]['date'],
                'best_params': best_param_dict,
                f'is_{optimization_metric}': best_params[optimization_metric],
                f'oos_{optimization_metric}': getattr(oos_metrics, optimization_metric),
                'oos_trades': oos_metrics.total_trades,
                'oos_win_rate': oos_metrics.win_rate,
                'oos_net_pnl': oos_metrics.net_pnl,
                'oos_sharpe': oos_metrics.sharpe_ratio,
                'oos_max_dd_pct': oos_metrics.max_drawdown_percent
            })
        
        results_df = pd.DataFrame(results)
        self.results = results_df
        
        print(f"\n{'='*60}")
        print(f"✅ Walk-Forward Analysis Complete!")
        print(f"{'='*60}")
        
        # Print summary
        self.print_summary(optimization_metric)
        
        return results_df
    
    def print_summary(self, metric: str):
        """Print walk-forward summary statistics"""
        if len(self.results) == 0:
            return
        
        print(f"\n📊 Walk-Forward Summary")
        print(f"{'='*60}")
        
        is_col = f'is_{metric}'
        oos_col = f'oos_{metric}'
        
        avg_is = self.results[is_col].mean()
        avg_oos = self.results[oos_col].mean()
        
        print(f"Average In-Sample {metric}: {avg_is:.4f}")
        print(f"Average Out-of-Sample {metric}: {avg_oos:.4f}")
        
        if avg_is != 0:
            degradation = ((avg_is - avg_oos) / abs(avg_is)) * 100
            print(f"Performance Degradation: {degradation:.2f}%")
        
        print(f"\nOut-of-Sample Results:")
        print(f"  Total Trades: {self.results['oos_trades'].sum()}")
        print(f"  Avg Win Rate: {self.results['oos_win_rate'].mean():.2f}%")
        print(f"  Total Net P&L: ₹{self.results['oos_net_pnl'].sum():,.2f}")
        print(f"  Avg Sharpe Ratio: {self.results['oos_sharpe'].mean():.3f}")
        print(f"  Avg Max Drawdown: {self.results['oos_max_dd_pct'].mean():.2f}%")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Parameter Optimization Module initialized!")
    print("Use ParameterOptimizer and WalkForwardAnalysis classes.")
