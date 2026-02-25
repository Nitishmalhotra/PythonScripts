"""
Backtesting Visualization Module
=================================

Generate comprehensive visualizations for backtesting results:
- Equity curves
- Drawdown charts
- Trade distribution
- Win/Loss analysis
- Monthly returns heatmap
- Strategy comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Optional
from backtesting_engine import Trade, PerformanceMetrics
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class BacktestVisualizer:
    """
    Create visualizations for backtesting results
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
    
    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Equity Curve",
        save_path: Optional[str] = None
    ):
        """
        Plot equity curve with drawdown
        
        Args:
            equity_curve: DataFrame with equity data
            title: Chart title
            save_path: Optional path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
        
        # Equity curve
        ax1.plot(equity_curve['date'], equity_curve['equity'], 
                label='Portfolio Value', linewidth=2, color='#2E86AB')
        ax1.fill_between(equity_curve['date'], equity_curve['equity'], 
                         alpha=0.3, color='#2E86AB')
        
        # Add buy & hold benchmark if initial capital is available
        initial_value = equity_curve['equity'].iloc[0]
        ax1.axhline(y=initial_value, color='gray', linestyle='--', 
                   label='Initial Capital', alpha=0.6)
        
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Portfolio Value (₹)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Format y-axis with Indian number system
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
        
        # Drawdown
        ax2.fill_between(equity_curve['date'], equity_curve['drawdown_percent'], 
                        0, alpha=0.5, color='red', label='Drawdown %')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower left', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Equity curve saved to {save_path}")
        
        plt.show()
    
    def plot_trade_distribution(
        self,
        trades: List[Trade],
        title: str = "Trade P&L Distribution",
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of trade P&L
        
        Args:
            trades: List of Trade objects
            title: Chart title
            save_path: Optional path to save figure
        """
        if not trades:
            print("No trades to plot")
            return
        
        pnl_values = [t.pnl for t in trades]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(pnl_values, bins=30, edgecolor='black', alpha=0.7, color='#A23B72')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Break-even')
        ax1.axvline(x=np.mean(pnl_values), color='green', linestyle='-', 
                   linewidth=2, label=f'Mean: ₹{np.mean(pnl_values):,.2f}')
        ax1.set_xlabel('P&L (₹)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('P&L Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(pnl_values, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#F18F01', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_ylabel('P&L (₹)', fontsize=12)
        ax2.set_title('P&L Box Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Trade distribution saved to {save_path}")
        
        plt.show()
    
    def plot_win_loss_analysis(
        self,
        trades: List[Trade],
        title: str = "Win/Loss Analysis",
        save_path: Optional[str] = None
    ):
        """
        Plot win/loss analysis
        
        Args:
            trades: List of Trade objects
            title: Chart title
            save_path: Optional path to save figure
        """
        if not trades:
            print("No trades to plot")
            return
        
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Win/Loss count
        categories = ['Wins', 'Losses']
        counts = [len(wins), len(losses)]
        colors = ['#06A77D', '#D62839']
        ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Number of Trades', fontsize=12)
        ax1.set_title('Win vs Loss Count', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(counts):
            ax1.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Win/Loss amount
        win_amount = sum(t.pnl for t in wins) if wins else 0
        loss_amount = abs(sum(t.pnl for t in losses)) if losses else 0
        amounts = [win_amount, loss_amount]
        ax2.bar(categories, amounts, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Amount (₹)', fontsize=12)
        ax2.set_title('Win vs Loss Amount', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
        
        # Add value labels
        for i, v in enumerate(amounts):
            ax2.text(i, v, f'₹{v:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Average win vs average loss
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 0
        avg_amounts = [avg_win, avg_loss]
        ax3.bar(categories, avg_amounts, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Average Amount (₹)', fontsize=12)
        ax3.set_title('Average Win vs Loss', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))
        
        # Holding period comparison
        win_holding = [t.holding_period_days for t in wins]
        loss_holding = [t.holding_period_days for t in losses]
        
        bp_data = [win_holding, loss_holding]
        bp = ax4.boxplot(bp_data, labels=['Wins', 'Losses'], patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Holding Period (days)', fontsize=12)
        ax4.set_title('Holding Period: Wins vs Losses', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Win/Loss analysis saved to {save_path}")
        
        plt.show()
    
    def plot_monthly_returns(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Monthly Returns Heatmap",
        save_path: Optional[str] = None
    ):
        """
        Plot monthly returns heatmap
        
        Args:
            equity_curve: DataFrame with equity data
            title: Chart title
            save_path: Optional path to save figure
        """
        # Calculate monthly returns
        df = equity_curve.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Resample to monthly and calculate returns
        monthly = df['equity'].resample('ME').last()
        monthly_returns = monthly.pct_change() * 100
        
        # Create year-month pivot
        monthly_returns_df = pd.DataFrame({
            'return': monthly_returns.values
        }, index=monthly_returns.index)
        
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        
        pivot = monthly_returns_df.pivot_table(
            values='return',
            index='year',
            columns='month',
            aggfunc='first'
        )
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Return (%)'},
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Set month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_names[:len(pivot.columns)])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Monthly returns heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_strategy_comparison(
        self,
        results: Dict[str, Dict],
        metric: str = 'sharpe_ratio',
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Compare multiple strategies
        
        Args:
            results: Dictionary mapping strategy name to results dict
                    {'Strategy1': {'equity_curve': df, 'metrics': PerformanceMetrics}}
            metric: Metric to compare
            title: Chart title
            save_path: Optional path to save figure
        """
        if not results:
            print("No results to compare")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot equity curves
        for strategy_name, data in results.items():
            equity_curve = data['equity_curve']
            ax1.plot(equity_curve['date'], equity_curve['equity'], 
                    label=strategy_name, linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Portfolio Value (₹)', fontsize=12)
        ax1.set_title('Equity Curves Comparison', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Compare metrics
        strategies = list(results.keys())
        sharpe_ratios = [results[s]['metrics'].sharpe_ratio for s in strategies]
        win_rates = [results[s]['metrics'].win_rate for s in strategies]
        max_dds = [abs(results[s]['metrics'].max_drawdown_percent) for s in strategies]
        returns = [results[s]['metrics'].total_pnl_percent for s in strategies]
        
        # Sharpe ratio comparison
        colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))
        ax2.bar(strategies, sharpe_ratios, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Sharpe Ratio', fontsize=12)
        ax2.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        
        # Win rate comparison
        ax3.bar(strategies, win_rates, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Win Rate (%)', fontsize=12)
        ax3.set_title('Win Rate Comparison', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(axis='x', rotation=45)
        
        # Scatter: Return vs Risk (Max DD)
        ax4.scatter(max_dds, returns, s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
        
        for i, strategy in enumerate(strategies):
            ax4.annotate(strategy, (max_dds[i], returns[i]), 
                        fontsize=9, ha='center', va='bottom')
        
        ax4.set_xlabel('Max Drawdown (%)', fontsize=12)
        ax4.set_ylabel('Total Return (%)', fontsize=12)
        ax4.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add quadrants
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.axvline(x=np.mean(max_dds), color='black', linestyle='--', alpha=0.5)
        
        if title:
            plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        else:
            plt.suptitle('Strategy Comparison', fontsize=16, fontweight='bold', y=1.00)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Strategy comparison saved to {save_path}")
        
        plt.show()
    
    def create_comprehensive_report(
        self,
        trades: List[Trade],
        equity_curve: pd.DataFrame,
        metrics: PerformanceMetrics,
        strategy_name: str,
        save_dir: Optional[str] = None
    ):
        """
        Create comprehensive visual report
        
        Args:
            trades: List of trades
            equity_curve: Equity curve DataFrame
            metrics: Performance metrics
            strategy_name: Name of strategy
            save_dir: Directory to save all charts
        """
        print(f"\n📊 Generating Comprehensive Report for {strategy_name}")
        print(f"{'='*60}\n")
        
        # Equity curve
        save_path = f"{save_dir}/{strategy_name}_equity_curve.png" if save_dir else None
        self.plot_equity_curve(equity_curve, f"{strategy_name} - Equity Curve", save_path)
        
        # Trade distribution
        save_path = f"{save_dir}/{strategy_name}_trade_distribution.png" if save_dir else None
        self.plot_trade_distribution(trades, f"{strategy_name} - Trade Distribution", save_path)
        
        # Win/Loss analysis
        save_path = f"{save_dir}/{strategy_name}_win_loss.png" if save_dir else None
        self.plot_win_loss_analysis(trades, f"{strategy_name} - Win/Loss Analysis", save_path)
        
        # Monthly returns
        save_path = f"{save_dir}/{strategy_name}_monthly_returns.png" if save_dir else None
        self.plot_monthly_returns(equity_curve, f"{strategy_name} - Monthly Returns", save_path)
        
        print(f"\n✅ Comprehensive report generated!")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Backtesting Visualization Module initialized!")
    print("Use BacktestVisualizer class to create charts.")
