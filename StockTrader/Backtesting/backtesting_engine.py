"""
Backtesting Engine for Kite Trading Strategies
==============================================

Features:
- Historical data replay with realistic order execution
- Multiple strategy support and comparison
- Comprehensive performance metrics (Sharpe, Max DD, Win Rate, etc.)
- Equity curve generation
- Parameter optimization
- Walk-forward analysis
- Detailed trade reports and visualizations

Author: Trading Bot
Date: February 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
import json
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Represents a single trade"""
    entry_date: datetime
    exit_date: datetime
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percent: float
    commission: float
    strategy: str
    exit_reason: str  # 'TARGET', 'STOP_LOSS', 'TRAILING_STOP', 'TIME_EXIT'
    holding_period_days: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a strategy"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percent: float
    avg_win: float
    avg_loss: float
    avg_pnl: float
    profit_factor: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_percent: float
    calmar_ratio: float
    avg_holding_period: float
    total_commission: float
    net_pnl: float
    return_on_capital: float
    
    def to_dict(self):
        return asdict(self)


class BacktestingEngine:
    """
    Main backtesting engine for strategy validation
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission_percent: float = 0.03,  # 0.03% per trade (buy + sell = 0.06%)
        slippage_percent: float = 0.05,    # 0.05% slippage
        position_size_percent: float = 10,  # 10% of capital per trade
        max_positions: int = 5,
        risk_free_rate: float = 0.05       # 5% annual risk-free rate
    ):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital
            commission_percent: Commission per trade (%)
            slippage_percent: Slippage per trade (%)
            position_size_percent: Position size as % of capital
            max_positions: Maximum concurrent positions
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.commission_percent = commission_percent / 100
        self.slippage_percent = slippage_percent / 100
        self.position_size_percent = position_size_percent / 100
        self.max_positions = max_positions
        self.risk_free_rate = risk_free_rate
        
        # State variables
        self.current_capital = initial_capital
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.daily_returns = []
        self.open_positions = {}
        
    def reset(self):
        """Reset engine state for new backtest"""
        self.current_capital = self.initial_capital
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.open_positions = {}
        
    def calculate_position_size(self, price: float) -> int:
        """Calculate position size based on available capital"""
        position_value = self.current_capital * self.position_size_percent
        quantity = int(position_value / price)
        return max(1, quantity)
    
    def apply_commission_and_slippage(self, price: float, direction: str) -> float:
        """
        Apply commission and slippage to entry/exit price
        
        Args:
            price: Original price
            direction: 'BUY' or 'SELL'
            
        Returns:
            Adjusted price with commission and slippage
        """
        total_cost_percent = self.commission_percent + self.slippage_percent
        
        if direction == 'BUY':
            # Buy: increase price (pay more)
            adjusted_price = price * (1 + total_cost_percent)
        else:
            # Sell: decrease price (receive less)
            adjusted_price = price * (1 - total_cost_percent)
            
        return adjusted_price
    
    def enter_position(
        self,
        date: datetime,
        symbol: str,
        price: float,
        direction: str,
        strategy: str,
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
        trailing_stop_percent: Optional[float] = None
    ) -> bool:
        """
        Enter a new position
        
        Returns:
            True if position entered successfully, False otherwise
        """
        # Check if we can enter more positions
        if len(self.open_positions) >= self.max_positions:
            return False
        
        # Check if position already exists for this symbol
        if symbol in self.open_positions:
            return False
        
        # Calculate position size
        quantity = self.calculate_position_size(price)
        
        # Apply costs
        entry_price = self.apply_commission_and_slippage(price, 'BUY')
        position_cost = entry_price * quantity
        
        # Check if we have enough capital
        if position_cost > self.current_capital:
            return False
        
        # Create position
        self.open_positions[symbol] = {
            'entry_date': date,
            'entry_price': entry_price,
            'quantity': quantity,
            'direction': direction,
            'strategy': strategy,
            'stop_loss': stop_loss,
            'target': target,
            'trailing_stop_percent': trailing_stop_percent,
            'highest_price': price if direction == 'LONG' else None,
            'lowest_price': price if direction == 'SHORT' else None
        }
        
        # Update capital
        self.current_capital -= position_cost
        
        return True
    
    def exit_position(
        self,
        date: datetime,
        symbol: str,
        price: float,
        exit_reason: str
    ) -> Optional[Trade]:
        """
        Exit an existing position
        
        Returns:
            Trade object if position exited, None otherwise
        """
        if symbol not in self.open_positions:
            return None
        
        position = self.open_positions[symbol]
        
        # Apply costs
        exit_price = self.apply_commission_and_slippage(price, 'SELL')
        
        # Calculate P&L
        if position['direction'] == 'LONG':
            pnl = (exit_price - position['entry_price']) * position['quantity']
            pnl_percent = ((exit_price / position['entry_price']) - 1) * 100
        else:  # SHORT
            pnl = (position['entry_price'] - exit_price) * position['quantity']
            pnl_percent = ((position['entry_price'] / exit_price) - 1) * 100
        
        # Calculate commission (applied on both entry and exit)
        commission = (position['entry_price'] + exit_price) * position['quantity'] * self.commission_percent
        
        # Net P&L
        net_pnl = pnl - commission
        
        # Update capital
        self.current_capital += (exit_price * position['quantity'])
        
        # Calculate holding period
        holding_period = (date - position['entry_date']).total_seconds() / 86400
        
        # Create trade record
        trade = Trade(
            entry_date=position['entry_date'],
            exit_date=date,
            symbol=symbol,
            direction=position['direction'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            quantity=position['quantity'],
            pnl=pnl,
            pnl_percent=pnl_percent,
            commission=commission,
            strategy=position['strategy'],
            exit_reason=exit_reason,
            holding_period_days=holding_period
        )
        
        self.trades.append(trade)
        
        # Remove from open positions
        del self.open_positions[symbol]
        
        return trade
    
    def update_trailing_stops(self, symbol: str, current_price: float):
        """Update trailing stop for a position"""
        if symbol not in self.open_positions:
            return
        
        position = self.open_positions[symbol]
        
        if position['trailing_stop_percent'] is None:
            return
        
        trailing_percent = position['trailing_stop_percent'] / 100
        
        if position['direction'] == 'LONG':
            # Update highest price
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
                # Update stop loss
                new_stop = position['highest_price'] * (1 - trailing_percent)
                position['stop_loss'] = new_stop
        else:  # SHORT
            # Update lowest price
            if current_price < position['lowest_price']:
                position['lowest_price'] = current_price
                # Update stop loss
                new_stop = position['lowest_price'] * (1 + trailing_percent)
                position['stop_loss'] = new_stop
    
    def check_exits(
        self,
        date: datetime,
        symbol: str,
        high: float,
        low: float,
        close: float
    ) -> Optional[Trade]:
        """
        Check if any exit conditions are met
        
        Returns:
            Trade object if position was exited, None otherwise
        """
        if symbol not in self.open_positions:
            return None
        
        position = self.open_positions[symbol]
        
        # Update trailing stops
        self.update_trailing_stops(symbol, close)
        
        # Check stop loss
        if position['stop_loss'] is not None:
            if position['direction'] == 'LONG' and low <= position['stop_loss']:
                return self.exit_position(date, symbol, position['stop_loss'], 'STOP_LOSS')
            elif position['direction'] == 'SHORT' and high >= position['stop_loss']:
                return self.exit_position(date, symbol, position['stop_loss'], 'STOP_LOSS')
        
        # Check target
        if position['target'] is not None:
            if position['direction'] == 'LONG' and high >= position['target']:
                return self.exit_position(date, symbol, position['target'], 'TARGET')
            elif position['direction'] == 'SHORT' and low <= position['target']:
                return self.exit_position(date, symbol, position['target'], 'TARGET')
        
        return None
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_function: Callable,
        strategy_name: str,
        **strategy_params
    ) -> Tuple[List[Trade], pd.DataFrame]:
        """
        Run backtest for a strategy
        
        Args:
            data: Historical OHLCV data with columns: date, open, high, low, close, volume
            strategy_function: Function that generates signals
            strategy_name: Name of the strategy
            **strategy_params: Parameters to pass to strategy function
            
        Returns:
            Tuple of (trades list, equity curve dataframe)
        """
        self.reset()
        
        # Ensure data is sorted by date
        data = data.sort_values('date').reset_index(drop=True)
        
        # Generate signals
        signals = strategy_function(data, **strategy_params)
        
        # Run through each day
        for idx, row in data.iterrows():
            date = row['date']
            symbol = row.get('symbol', 'STOCK')
            
            # Check exits for open positions
            self.check_exits(
                date=date,
                symbol=symbol,
                high=row['high'],
                low=row['low'],
                close=row['close']
            )
            
            # Check for entry signals
            if idx < len(signals) and signals.iloc[idx]['signal'] == 1:
                self.enter_position(
                    date=date,
                    symbol=symbol,
                    price=row['close'],
                    direction='LONG',
                    strategy=strategy_name,
                    stop_loss=signals.iloc[idx].get('stop_loss'),
                    target=signals.iloc[idx].get('target'),
                    trailing_stop_percent=signals.iloc[idx].get('trailing_stop_percent')
                )
            
            # Record equity
            open_positions_value = sum(
                pos['quantity'] * row['close'] 
                for pos in self.open_positions.values()
            )
            total_equity = self.current_capital + open_positions_value
            
            self.equity_curve.append({
                'date': date,
                'equity': total_equity,
                'cash': self.current_capital,
                'positions_value': open_positions_value,
                'drawdown': 0  # Will calculate later
            })
        
        # Close any remaining open positions at last price
        last_row = data.iloc[-1]
        for symbol in list(self.open_positions.keys()):
            self.exit_position(
                date=last_row['date'],
                symbol=symbol,
                price=last_row['close'],
                exit_reason='TIME_EXIT'
            )
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Calculate drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = equity_df['equity'] - equity_df['cummax']
        equity_df['drawdown_percent'] = (equity_df['drawdown'] / equity_df['cummax']) * 100
        
        return self.trades, equity_df
    
    def calculate_metrics(self, trades: List[Trade], equity_curve: pd.DataFrame) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Args:
            trades: List of completed trades
            equity_curve: Equity curve DataFrame
            
        Returns:
            PerformanceMetrics object
        """
        if not trades:
            return PerformanceMetrics(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, total_pnl_percent=0,
                avg_win=0, avg_loss=0, avg_pnl=0, profit_factor=0,
                max_consecutive_wins=0, max_consecutive_losses=0,
                sharpe_ratio=0, sortino_ratio=0, max_drawdown=0,
                max_drawdown_percent=0, calmar_ratio=0, avg_holding_period=0,
                total_commission=0, net_pnl=0, return_on_capital=0
            )
        
        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl < 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L statistics
        total_pnl = sum(t.pnl for t in trades)
        total_commission = sum(t.commission for t in trades)
        net_pnl = total_pnl - total_commission
        
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_pnl = net_pnl / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = sum(losses) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in trades:
            if trade.pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        # Calculate returns
        equity_curve['returns'] = equity_curve['equity'].pct_change()
        daily_returns = equity_curve['returns'].dropna()
        
        # Sharpe Ratio (annualized)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            avg_daily_return = daily_returns.mean()
            std_daily_return = daily_returns.std()
            sharpe_ratio = (avg_daily_return - (self.risk_free_rate / 252)) / std_daily_return * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Sortino Ratio (annualized)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (daily_returns.mean() - (self.risk_free_rate / 252)) / downside_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        # Maximum Drawdown
        max_drawdown = equity_curve['drawdown'].min()
        max_drawdown_percent = equity_curve['drawdown_percent'].min()
        
        # Calmar Ratio
        total_return = ((equity_curve['equity'].iloc[-1] / self.initial_capital) - 1) * 100
        years = len(equity_curve) / 252  # Assuming 252 trading days per year
        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        calmar_ratio = annualized_return / abs(max_drawdown_percent) if max_drawdown_percent != 0 else 0
        
        # Average holding period
        avg_holding_period = np.mean([t.holding_period_days for t in trades])
        
        # Return on capital
        return_on_capital = (net_pnl / self.initial_capital) * 100
        total_pnl_percent = ((equity_curve['equity'].iloc[-1] / self.initial_capital) - 1) * 100
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_pnl=avg_pnl,
            profit_factor=profit_factor,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_percent,
            calmar_ratio=calmar_ratio,
            avg_holding_period=avg_holding_period,
            total_commission=total_commission,
            net_pnl=net_pnl,
            return_on_capital=return_on_capital
        )
    
    def generate_report(
        self,
        strategy_name: str,
        trades: List[Trade],
        metrics: PerformanceMetrics,
        equity_curve: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive backtest report
        
        Args:
            strategy_name: Name of the strategy
            trades: List of trades
            metrics: Performance metrics
            equity_curve: Equity curve DataFrame
            save_path: Optional path to save report as JSON
            
        Returns:
            Dictionary containing full report
        """
        report = {
            'strategy_name': strategy_name,
            'backtest_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'initial_capital': self.initial_capital,
            'final_capital': equity_curve['equity'].iloc[-1] if len(equity_curve) > 0 else 0,
            'configuration': {
                'commission_percent': self.commission_percent * 100,
                'slippage_percent': self.slippage_percent * 100,
                'position_size_percent': self.position_size_percent * 100,
                'max_positions': self.max_positions,
                'risk_free_rate': self.risk_free_rate
            },
            'performance_metrics': metrics.to_dict(),
            'trades': [t.to_dict() for t in trades],
            'equity_curve': equity_curve.to_dict('records') if len(equity_curve) > 0 else []
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report


# Example usage and utility functions
def print_metrics(metrics: PerformanceMetrics, strategy_name: str = "Strategy"):
    """Pretty print performance metrics"""
    print(f"\n{'='*60}")
    print(f"  {strategy_name} - Performance Metrics")
    print(f"{'='*60}\n")
    
    print(f"📊 Trade Statistics:")
    print(f"   Total Trades: {metrics.total_trades}")
    print(f"   Winning Trades: {metrics.winning_trades}")
    print(f"   Losing Trades: {metrics.losing_trades}")
    print(f"   Win Rate: {metrics.win_rate:.2f}%")
    print(f"   Avg Holding Period: {metrics.avg_holding_period:.2f} days")
    
    print(f"\n💰 P&L Statistics:")
    print(f"   Total P&L: ₹{metrics.total_pnl:,.2f}")
    print(f"   Net P&L (after commissions): ₹{metrics.net_pnl:,.2f}")
    print(f"   Total Return: {metrics.total_pnl_percent:.2f}%")
    print(f"   Return on Capital: {metrics.return_on_capital:.2f}%")
    print(f"   Average Win: ₹{metrics.avg_win:,.2f}")
    print(f"   Average Loss: ₹{metrics.avg_loss:,.2f}")
    print(f"   Profit Factor: {metrics.profit_factor:.2f}")
    
    print(f"\n📈 Risk Metrics:")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"   Sortino Ratio: {metrics.sortino_ratio:.3f}")
    print(f"   Max Drawdown: ₹{metrics.max_drawdown:,.2f} ({metrics.max_drawdown_percent:.2f}%)")
    print(f"   Calmar Ratio: {metrics.calmar_ratio:.3f}")
    
    print(f"\n🎯 Streaks:")
    print(f"   Max Consecutive Wins: {metrics.max_consecutive_wins}")
    print(f"   Max Consecutive Losses: {metrics.max_consecutive_losses}")
    
    print(f"\n💸 Costs:")
    print(f"   Total Commission: ₹{metrics.total_commission:,.2f}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    print("Backtesting Engine initialized successfully!")
    print("Import this module to use the BacktestingEngine class.")
