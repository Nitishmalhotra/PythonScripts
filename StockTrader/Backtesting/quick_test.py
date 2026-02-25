"""
Quick Backtest Demo - Test with Real Kite Data
===============================================

Simple script to quickly test backtesting with actual Kite API data.
Tests a Moving Average Crossover strategy on SBIN stock.

Usage:
    python quick_test.py
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from kiteconnect import KiteConnect
import traceback
import json
import os

# Import backtesting modules
from backtesting_engine import BacktestingEngine, print_metrics
from strategy_examples import (
    moving_average_crossover_strategy,
    rsi_mean_reversion_strategy,
    bollinger_breakout_strategy,
    macd_momentum_strategy,
    momentum_breakout_strategy,
    combined_ma_rsi_volume_strategy
)
from backtest_visualizer import BacktestVisualizer

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Stock to Sector/Index Mapping
STOCK_SECTOR_MAP = {
    # Banking
    'SBIN': {'sector': 'Banking', 'index': 'Nifty Bank', 'competitors': ['HDFCBANK', 'ICICIBANK', 'AXISBANK', 'KOTAKBANK', 'INDUSINDBK']},
    'HDFCBANK': {'sector': 'Banking', 'index': 'Nifty Bank', 'competitors': ['SBIN', 'ICICIBANK', 'AXISBANK', 'KOTAKBANK', 'INDUSINDBK']},
    'ICICIBANK': {'sector': 'Banking', 'index': 'Nifty Bank', 'competitors': ['SBIN', 'HDFCBANK', 'AXISBANK', 'KOTAKBANK', 'INDUSINDBK']},
    'AXISBANK': {'sector': 'Banking', 'index': 'Nifty Bank', 'competitors': ['SBIN', 'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'INDUSINDBK']},
    'KOTAKBANK': {'sector': 'Banking', 'index': 'Nifty Bank', 'competitors': ['SBIN', 'HDFCBANK', 'ICICIBANK', 'AXISBANK', 'INDUSINDBK']},
    'INDUSINDBK': {'sector': 'Banking', 'index': 'Nifty Bank', 'competitors': ['SBIN', 'HDFCBANK', 'ICICIBANK', 'AXISBANK', 'KOTAKBANK']},
    
    # IT
    'TCS': {'sector': 'IT', 'index': 'Nifty IT', 'competitors': ['INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM']},
    'INFY': {'sector': 'IT', 'index': 'Nifty IT', 'competitors': ['TCS', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM']},
    'WIPRO': {'sector': 'IT', 'index': 'Nifty IT', 'competitors': ['TCS', 'INFY', 'HCLTECH', 'TECHM', 'LTIM']},
    'HCLTECH': {'sector': 'IT', 'index': 'Nifty IT', 'competitors': ['TCS', 'INFY', 'WIPRO', 'TECHM', 'LTIM']},
    'TECHM': {'sector': 'IT', 'index': 'Nifty IT', 'competitors': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'LTIM']},
    'LTIM': {'sector': 'IT', 'index': 'Nifty IT', 'competitors': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM']},
    
    # Auto
    'MARUTI': {'sector': 'Auto', 'index': 'Nifty Auto', 'competitors': ['TMPV', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT', 'HEROMOTOCO']},
    'TMPV': {'sector': 'Auto', 'index': 'Nifty Auto', 'competitors': ['MARUTI', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT', 'HEROMOTOCO']},
    'M&M': {'sector': 'Auto', 'index': 'Nifty Auto', 'competitors': ['MARUTI', 'TMPV', 'BAJAJ-AUTO', 'EICHERMOT', 'HEROMOTOCO']},
    'BAJAJ-AUTO': {'sector': 'Auto', 'index': 'Nifty Auto', 'competitors': ['MARUTI', 'TMPV', 'M&M', 'EICHERMOT', 'HEROMOTOCO']},
    
    # Pharma
    'SUNPHARMA': {'sector': 'Pharma', 'index': 'Nifty Pharma', 'competitors': ['DRREDDY', 'CIPLA', 'DIVISLAB', 'APOLLOHOSP', 'LAURUSLABS']},
    'DRREDDY': {'sector': 'Pharma', 'index': 'Nifty Pharma', 'competitors': ['SUNPHARMA', 'CIPLA', 'DIVISLAB', 'APOLLOHOSP', 'LAURUSLABS']},
    'CIPLA': {'sector': 'Pharma', 'index': 'Nifty Pharma', 'competitors': ['SUNPHARMA', 'DRREDDY', 'DIVISLAB', 'APOLLOHOSP', 'LAURUSLABS']},
    'DIVISLAB': {'sector': 'Pharma', 'index': 'Nifty Pharma', 'competitors': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'APOLLOHOSP', 'LAURUSLABS']},
    
    # FMCG/Consumer
    'HINDUNILVR': {'sector': 'FMCG', 'index': 'Nifty FMCG', 'competitors': ['ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'MARICO']},
    'ITC': {'sector': 'FMCG', 'index': 'Nifty FMCG', 'competitors': ['HINDUNILVR', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'MARICO']},
    'NESTLEIND': {'sector': 'FMCG', 'index': 'Nifty FMCG', 'competitors': ['HINDUNILVR', 'ITC', 'BRITANNIA', 'DABUR', 'MARICO']},
    'BRITANNIA': {'sector': 'FMCG', 'index': 'Nifty FMCG', 'competitors': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'DABUR', 'MARICO']},
    
    # Metals
    'TATASTEEL': {'sector': 'Metals', 'index': 'Nifty Metal', 'competitors': ['HINDALCO', 'JSWSTEEL', 'COALINDIA', 'VEDL', 'NMDC']},
    'HINDALCO': {'sector': 'Metals', 'index': 'Nifty Metal', 'competitors': ['TATASTEEL', 'JSWSTEEL', 'COALINDIA', 'VEDL', 'NMDC']},
    'JSWSTEEL': {'sector': 'Metals', 'index': 'Nifty Metal', 'competitors': ['TATASTEEL', 'HINDALCO', 'COALINDIA', 'VEDL', 'NMDC']},
    
    # Energy/Oil
    'RELIANCE': {'sector': 'Energy', 'index': 'Nifty Energy', 'competitors': ['ONGC', 'BPCL', 'IOC', 'ADANIPORTS', 'NTPC']},
    'ONGC': {'sector': 'Energy', 'index': 'Nifty Energy', 'competitors': ['RELIANCE', 'BPCL', 'IOC', 'NTPC', 'POWERGRID']},
    'BPCL': {'sector': 'Energy', 'index': 'Nifty Energy', 'competitors': ['RELIANCE', 'ONGC', 'IOC', 'NTPC', 'POWERGRID']},
    'NTPC': {'sector': 'Energy', 'index': 'Nifty Energy', 'competitors': ['RELIANCE', 'ONGC', 'BPCL', 'POWERGRID', 'IOC']},
}

# Index constituents (top stocks)
INDEX_CONSTITUENTS = {
    'Nifty Bank': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK', 'INDUSINDBK'],
    'Nifty IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM'],
    'Nifty Auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT'],
    'Nifty Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'APOLLOHOSP'],
    'Nifty FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR'],
    'Nifty Metal': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'COALINDIA', 'VEDL'],
    'Nifty Energy': ['RELIANCE', 'ONGC', 'BPCL', 'NTPC', 'POWERGRID'],
}


def load_kite_credentials():
    """Load Kite API credentials"""
    credentials_file = PROJECT_ROOT / 'kite_credentials.txt'
    
    print(f"📂 Looking for credentials at: {credentials_file}")
    
    if not credentials_file.exists():
        print(f"❌ Credentials file not found!")
        print(f"   Expected location: {credentials_file}")
        print(f"   Please create kite_credentials.txt with:")
        print(f"   api_key=your_api_key")
        print(f"   access_token=your_access_token")
        return None, None
    
    credentials = {}
    with open(credentials_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                # Store with lowercase keys for consistency (handles both API_KEY and api_key formats)
                credentials[key.strip().lower()] = value.strip()
    
    print(f"📋 Found keys in credentials: {list(credentials.keys())}")
    
    api_key = credentials.get('api_key')
    access_token = credentials.get('access_token')
    
    if not api_key:
        print("❌ api_key not found in credentials file")
        return None, None
    
    if not access_token:
        print("❌ access_token not found in credentials file")
        return None, None
    
    print(f"✅ Credentials loaded successfully")
    print(f"   API Key: {api_key[:10]}... (hidden)")
    
    return api_key, access_token


def fetch_real_data(symbol, days_back=365):
    """Fetch real historical data from Kite API"""
    print(f"\n📊 Fetching data for {symbol}...")
    
    # Load credentials
    api_key, access_token = load_kite_credentials()
    if not api_key or not access_token:
        print("\n❌ Cannot proceed without valid credentials")
        return None
    
    try:
        # Initialize Kite
        print(f"\n🔌 Connecting to Kite API...")
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        # Test connection by getting profile
        try:
            profile = kite.profile()
            print(f"✅ Connected to Kite API")
            print(f"   User: {profile.get('user_name', 'Unknown')}")
            print(f"   User ID: {profile.get('user_id', 'Unknown')}")
        except Exception as e:
            print(f"❌ Failed to connect to Kite API")
            print(f"   Error: {e}")
            print(f"\n💡 Possible issues:")
            print(f"   1. Access token expired - generate new token")
            print(f"   2. API key incorrect")
            print(f"   3. No internet connection")
            return None
        
        # Get instruments
        print(f"\n📋 Loading NSE instruments...")
        try:
            instruments = kite.instruments('NSE')
            print(f"✅ Loaded {len(instruments)} instruments")
        except Exception as e:
            print(f"❌ Failed to load instruments: {e}")
            return None
        
        # Find instrument token
        print(f"\n🔍 Searching for {symbol}...")
        instrument_token = None
        instrument_info = None
        for inst in instruments:
            if inst['tradingsymbol'] == symbol and inst['instrument_type'] == 'EQ':
                instrument_token = inst['instrument_token']
                instrument_info = inst
                break
        
        if not instrument_token:
            print(f"❌ Symbol {symbol} not found in NSE instruments")
            print(f"\n💡 Trying common symbols:")
            common_symbols = ['SBIN', 'RELIANCE', 'TCS', 'INFY', 'HDFCBANK']
            for sym in common_symbols[:5]:
                for inst in instruments:
                    if inst['tradingsymbol'] == sym and inst['instrument_type'] == 'EQ':
                        print(f"   ✅ {sym} - Token: {inst['instrument_token']}")
                        break
            return None
        
        print(f"✅ Found: {instrument_info['name']} ({symbol})")
        print(f"   Token: {instrument_token}")
        print(f"   Exchange: {instrument_info['exchange']}")
        
        # Calculate date range
        to_date = datetime.now()
        from_date = datetime.now() - pd.Timedelta(days=days_back)
        
        print(f"\n📅 Fetching historical data...")
        print(f"   From: {from_date.date()}")
        print(f"   To: {to_date.date()}")
        print(f"   Interval: day")
        
        # Fetch historical data
        try:
            data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval='day'
            )
        except Exception as e:
            print(f"❌ Failed to fetch historical data: {e}")
            print(f"\n💡 Possible issues:")
            print(f"   1. Date range too large (try reducing days_back)")
            print(f"   2. No data available for this period")
            print(f"   3. Rate limit exceeded")
            return None
        
        if not data:
            print(f"❌ No data received from API")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['symbol'] = symbol
        
        print(f"✅ Successfully fetched {len(df)} candles")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Price range: ₹{df['close'].min():.2f} - ₹{df['close'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"\n📋 Full traceback:")
        traceback.print_exc()
        return None


def save_backtest_results(results, symbol, output_dir='backtest_results'):
    """Save backtest results to files (CSV, JSON, charts)"""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Save comparison table as CSV
    summary_data = []
    for strategy_name, result in results.items():
        metrics = result['metrics']
        summary_data.append({
            'Strategy': strategy_name,
            'Trades': metrics.total_trades,
            'Win_Rate_%': metrics.win_rate,
            'Sharpe_Ratio': metrics.sharpe_ratio,
            'Sortino_Ratio': metrics.sortino_ratio,
            'Max_Drawdown_%': metrics.max_drawdown_percent,
            'Net_PnL': metrics.net_pnl,
            'Return_%': metrics.total_pnl_percent,
            'Profit_Factor': metrics.profit_factor,
            'Avg_Win': metrics.avg_win,
            'Avg_Loss': metrics.avg_loss,
            'Winning_Trades': metrics.winning_trades,
            'Losing_Trades': metrics.losing_trades,
            'Avg_Holding_Days': metrics.avg_holding_period
        })
    
    df = pd.DataFrame(summary_data)
    csv_filename = f'{symbol}_comparison_{timestamp}.csv'
    csv_path = output_path / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"   ✅ CSV saved: {csv_path}")
    
    # 2. Save detailed JSON report
    json_data = {
        'symbol': symbol,
        'timestamp': timestamp,
        'test_date': datetime.now().isoformat(),
        'strategies': {}
    }
    
    for strategy_name, result in results.items():
        m = result['metrics']
        json_data['strategies'][strategy_name] = {
            'description': result['description'],
            'total_trades': m.total_trades,
            'winning_trades': m.winning_trades,
            'losing_trades': m.losing_trades,
            'win_rate': m.win_rate,
            'net_pnl': m.net_pnl,
            'total_return_percent': m.total_pnl_percent,
            'sharpe_ratio': m.sharpe_ratio,
            'sortino_ratio': m.sortino_ratio,
            'max_drawdown_percent': m.max_drawdown_percent,
            'profit_factor': m.profit_factor,
            'average_win': m.avg_win,
            'average_loss': m.avg_loss,
            'average_holding_period': m.avg_holding_period,
            'calmar_ratio': m.calmar_ratio,
            'max_consecutive_wins': m.max_consecutive_wins,
            'max_consecutive_losses': m.max_consecutive_losses
        }
    
    # Add best strategy recommendations
    best_sharpe = max(results.items(), key=lambda x: x[1]['metrics'].sharpe_ratio)
    best_return = max(results.items(), key=lambda x: x[1]['metrics'].total_pnl_percent)
    best_winrate = max(results.items(), key=lambda x: x[1]['metrics'].win_rate)
    safest = min(results.items(), key=lambda x: abs(x[1]['metrics'].max_drawdown_percent))
    
    json_data['recommendations'] = {
        'best_sharpe': {
            'strategy': best_sharpe[0],
            'sharpe': best_sharpe[1]['metrics'].sharpe_ratio
        },
        'best_return': {
            'strategy': best_return[0],
            'return_percent': best_return[1]['metrics'].total_pnl_percent
        },
        'best_win_rate': {
            'strategy': best_winrate[0],
            'win_rate': best_winrate[1]['metrics'].win_rate
        },
        'safest': {
            'strategy': safest[0],
            'max_drawdown_percent': safest[1]['metrics'].max_drawdown_percent
        }
    }
    
    json_filename = f'{symbol}_report_{timestamp}.json'
    json_path = output_path / json_filename
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"   ✅ JSON report saved: {json_path}")
    
    # 3. Save text summary report
    txt_filename = f'{symbol}_summary_{timestamp}.txt'
    txt_path = output_path / txt_filename
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"BACKTEST SUMMARY REPORT - {symbol}\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Strategies Tested: {len(results)}\n")
        f.write("="*80 + "\n\n")
        
        f.write("STRATEGY COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"🏆 Best Risk-Adjusted Returns (Sharpe Ratio):\n")
        f.write(f"   {best_sharpe[0]}\n")
        f.write(f"   Sharpe: {best_sharpe[1]['metrics'].sharpe_ratio:.2f}\n\n")
        
        f.write(f"💰 Best Absolute Returns:\n")
        f.write(f"   {best_return[0]}\n")
        f.write(f"   Return: {best_return[1]['metrics'].total_pnl_percent:.1f}%\n\n")
        
        f.write(f"🎯 Best Win Rate:\n")
        f.write(f"   {best_winrate[0]}\n")
        f.write(f"   Win Rate: {best_winrate[1]['metrics'].win_rate:.1f}%\n\n")
        
        f.write(f"🛡️  Safest (Lowest Drawdown):\n")
        f.write(f"   {safest[0]}\n")
        f.write(f"   Max DD: {safest[1]['metrics'].max_drawdown_percent:.1f}%\n\n")
    
    print(f"   ✅ Summary report saved: {txt_path}")
    
    return {
        'csv_path': str(csv_path),
        'json_path': str(json_path),
        'txt_path': str(txt_path),
        'output_dir': str(output_path)
    }


def run_comprehensive_backtest(symbol='SBIN', days_back=730):
    """Run comprehensive backtest with multiple strategies"""
    print("\n" + "="*80)
    print(f"🚀 COMPREHENSIVE BACKTEST - Multiple Strategies on {symbol}")
    print("="*80)
    
    # Fetch real data
    data = fetch_real_data(symbol, days_back=days_back)
    
    if data is None:
        print("\n" + "="*80)
        print("❌ BACKTEST FAILED - Could not fetch data")
        print("="*80)
        print("\n💡 TROUBLESHOOTING STEPS:")
        print("\n1. Check your credentials:")
        print(f"   - File location: {PROJECT_ROOT / 'kite_credentials.txt'}")
        print("   - Should contain: api_key=xxx and access_token=xxx")
        print("\n2. Generate fresh access token:")
        print("   cd ..")
        print("   python Utilities/generate_token.py")
        print("\n3. Check internet connection")
        print("\n4. Verify Kite API status at: https://kite.trade/")
        print("\n5. Verify symbol name is correct (e.g., SBIN, RELIANCE, TCS)")
        print("="*80 + "\n")
        return
    
    print(f"\n📊 Data Summary:")
    print(f"   Symbol: {symbol}")
    print(f"   Period: {data['date'].min()} to {data['date'].max()}")
    print(f"   Candles: {len(data)}")
    print(f"   Price Range: ₹{data['close'].min():.2f} - ₹{data['close'].max():.2f}\n")
    
    # Initialize backtesting engine
    print("⚙️  Initializing backtest engine...")
    engine = BacktestingEngine(
        initial_capital=100000,      # ₹1 Lakh
        commission_percent=0.03,     # 0.03% commission
        slippage_percent=0.05,       # 0.05% slippage
        position_size_percent=10,    # 10% per trade
        max_positions=3              # Max 3 concurrent positions
    )
    
    # Define strategies to test
    strategies_to_test = {
        '1. MA Crossover (Original)': {
            'function': moving_average_crossover_strategy,
            'params': {
                'fast_period': 20,
                'slow_period': 50,
                'stop_loss_percent': 3.0,
                'target_percent': 8.0
            },
            'description': 'Classic 20/50 MA crossover'
        },
        '2. MA Crossover (Optimized)': {
            'function': moving_average_crossover_strategy,
            'params': {
                'fast_period': 10,
                'slow_period': 30,
                'stop_loss_percent': 2.5,
                'target_percent': 6.0
            },
            'description': 'Faster MAs (10/30) for quicker signals'
        },
        '3. RSI Mean Reversion': {
            'function': rsi_mean_reversion_strategy,
            'params': {
                'rsi_period': 14,
                'oversold_threshold': 30,
                'overbought_threshold': 70,
                'stop_loss_percent': 3.0,
                'target_percent': 6.0
            },
            'description': 'Buy oversold, sell overbought'
        },
        '4. Bollinger Breakout': {
            'function': bollinger_breakout_strategy,
            'params': {
                'bb_period': 20,
                'bb_std': 2,
                'stop_loss_atr_multiplier': 1.5,
                'target_atr_multiplier': 3.0,
                'trailing_stop_percent': 1.5
            },
            'description': 'Breakout above upper band with volume'
        },
        '5. MACD Momentum': {
            'function': macd_momentum_strategy,
            'params': {
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'stop_loss_percent': 2.5,
                'target_percent': 5.0
            },
            'description': 'MACD crossover with positive MACD'
        },
        '6. Momentum Breakout': {
            'function': momentum_breakout_strategy,
            'params': {
                'lookback_period': 20,
                'volume_multiplier': 1.5,
                'stop_loss_percent': 2.0,
                'trailing_stop_percent': 2.0
            },
            'description': 'Price breakout + volume surge + RSI filter'
        },
        '7. Combined Strategy (MA+RSI+Volume)': {
            'function': combined_ma_rsi_volume_strategy,
            'params': {
                'fast_ma': 10,
                'slow_ma': 30,
                'rsi_period': 14,
                'rsi_threshold': 50,
                'volume_multiplier': 1.3,
                'stop_loss_percent': 2.5,
                'target_percent': 6.0
            },
            'description': 'Multi-filter: MA crossover + RSI > 50 + Volume surge'
        }
    }
    
    print(f"\n🔬 Testing {len(strategies_to_test)} strategies...\n")
    print("="*80 + "\n")
    
    results = {}
    
    # Run backtest for each strategy
    for strategy_name, strategy_config in strategies_to_test.items():
        print(f"\n{'='*80}")
        print(f"📊 TESTING: {strategy_name}")
        print(f"📝 Description: {strategy_config['description']}")
        print(f"{'='*80}\n")
        
        try:
            trades, equity_curve = engine.run_backtest(
                data=data,
                strategy_function=strategy_config['function'],
                strategy_name=strategy_name,
                **strategy_config['params']
            )
            
            metrics = engine.calculate_metrics(trades, equity_curve)
            
            results[strategy_name] = {
                'trades': trades,
                'equity_curve': equity_curve,
                'metrics': metrics,
                'description': strategy_config['description']
            }
            
            # Print results
            print_metrics(metrics, strategy_name)
            
            # Show sample trades
            if len(trades) > 0:
                print(f"\n📋 Sample Trades (First 3):")
                print("-" * 80)
                for i, trade in enumerate(trades[:3]):
                    print(f"\nTrade #{i+1}:")
                    print(f"  Entry: {trade.entry_date.date()} @ ₹{trade.entry_price:.2f}")
                    print(f"  Exit:  {trade.exit_date.date()} @ ₹{trade.exit_price:.2f}")
                    print(f"  P&L:   ₹{trade.pnl:.2f} ({trade.pnl_percent:+.2f}%)")
                    print(f"  Reason: {trade.exit_reason}")
            
        except Exception as e:
            print(f"❌ Error testing {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comparison summary
    print("\n\n" + "="*80)
    print("📊 STRATEGY COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    summary_data = []
    for strategy_name, result in results.items():
        metrics = result['metrics']
        summary_data.append({
            'Strategy': strategy_name,
            'Trades': metrics.total_trades,
            'Win Rate': f"{metrics.win_rate:.1f}%",
            'Sharpe': f"{metrics.sharpe_ratio:.2f}",
            'Sortino': f"{metrics.sortino_ratio:.2f}",
            'Max DD': f"{metrics.max_drawdown_percent:.1f}%",
            'Net P&L': f"₹{metrics.net_pnl:,.0f}",
            'Return': f"{metrics.total_pnl_percent:.1f}%",
            'Profit Factor': f"{metrics.profit_factor:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("\n" + "="*80)
    
    # Recommendations
    print("\n💡 STRATEGY RECOMMENDATIONS:")
    print("-" * 80)
    
    # Find best strategy by different metrics
    best_sharpe = max(results.items(), key=lambda x: x[1]['metrics'].sharpe_ratio)
    best_return = max(results.items(), key=lambda x: x[1]['metrics'].total_pnl_percent)
    best_winrate = max(results.items(), key=lambda x: x[1]['metrics'].win_rate)
    safest = min(results.items(), key=lambda x: abs(x[1]['metrics'].max_drawdown_percent))
    
    print(f"\n🏆 Best Risk-Adjusted Returns (Sharpe Ratio):")
    print(f"   {best_sharpe[0]}")
    print(f"   Sharpe: {best_sharpe[1]['metrics'].sharpe_ratio:.2f}")
    print(f"   → Best for consistent, risk-adjusted profits")
    
    print(f"\n💰 Best Absolute Returns:")
    print(f"   {best_return[0]}")
    print(f"   Return: {best_return[1]['metrics'].total_pnl_percent:.1f}%")
    print(f"   → Best for maximum profit potential")
    
    print(f"\n🎯 Best Win Rate:")
    print(f"   {best_winrate[0]}")
    print(f"   Win Rate: {best_winrate[1]['metrics'].win_rate:.1f}%")
    print(f"   → Best for psychological comfort (more winning trades)")
    
    print(f"\n🛡️  Safest (Lowest Drawdown):")
    print(f"   {safest[0]}")
    print(f"   Max DD: {safest[1]['metrics'].max_drawdown_percent:.1f}%")
    print(f"   → Best for capital preservation")
    
    print("\n" + "-" * 80)
    print("📊 STRATEGY INSIGHTS:")
    print("-" * 80)
    print("\n1. MA Crossover (Original 20/50):")
    print("   ✓ Pros: Simple, reliable in trending markets")
    print("   ✗ Cons: Slow to react, many whipsaws in sideways markets")
    
    print("\n2. MA Crossover (Optimized 10/30):")
    print("   ✓ Pros: Faster signals, catches trends earlier")
    print("   ✗ Cons: More false signals, higher trading frequency")
    
    print("\n3. RSI Mean Reversion:")
    print("   ✓ Pros: Works well in ranging markets, high win rate")
    print("   ✗ Cons: Can get caught in strong trends, missing big moves")
    
    print("\n4. Bollinger Breakout:")
    print("   ✓ Pros: Catches explosive moves, volume confirmation")
    print("   ✗ Cons: Rare signals, can give back profits without trailing stop")
    
    print("\n5. MACD Momentum:")
    print("   ✓ Pros: Good trend following, uses both trend & momentum")
    print("   ✗ Cons: Lagging indicator, late entries")
    
    print("\n6. Momentum Breakout:")
    print("   ✓ Pros: Multi-filter reduces false signals, good R:R")
    print("   ✗ Cons: Very selective, few trades")
    
    print("\n7. Combined Strategy (MA+RSI+Volume):")
    print("   ✓ Pros: Multiple confirmations, filters noise effectively")
    print("   ✗ Cons: May miss some good trades due to strict filters")
    print("   → Recommended for live trading (lower risk, high quality signals)")
    print("\n" + "-" * 80)
    
    # Prepare output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('backtest_results')
    output_dir.mkdir(exist_ok=True)
    
    # Generate and save visualizations
    print("\n\n📊 Generating and saving comparison charts...")
    try:
        visualizer = BacktestVisualizer()
        
        # Save chart to file
        chart_filename = f'{symbol}_comparison_{timestamp}.png'
        chart_path = output_dir / chart_filename
        
        # Plot strategy comparison (will display and save)
        visualizer.plot_strategy_comparison(
            results,
            title=f'Strategy Performance Comparison - {symbol}',
            save_path=str(chart_path)
        )
        
        print(f"   ✅ Chart saved: {chart_path}")
        print("   ✅ Chart displayed!")
        
    except Exception as e:
        print(f"⚠️  Could not generate charts: {e}")
        import traceback
        traceback.print_exc()
    
    # Save all results to files
    print("\n📁 Saving backtest results...")
    try:
        saved_files = save_backtest_results(results, symbol, output_dir=str(output_dir))
        
        print("\n" + "="*80)
        print("📦 SAVED FILES SUMMARY")
        print("="*80)
        print(f"📁 Output Directory: {saved_files['output_dir']}")
        print(f"📊 Comparison CSV: {Path(saved_files['csv_path']).name}")
        print(f"📋 JSON Report: {Path(saved_files['json_path']).name}")
        print(f"📝 Text Summary: {Path(saved_files['txt_path']).name}")
        print(f"📈 Chart PNG: {chart_filename}")
        print("="*80)
        
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print(f"✅ COMPREHENSIVE BACKTEST COMPLETE - {symbol}")
    print("="*80 + "\n")
    
    return results


def run_multi_stock_comparison(symbols, days_back=730, best_strategy=None):
    """Run comparison across multiple stocks to find best stock-strategy combinations"""
    print("\n" + "="*80)
    print(f"🔬 MULTI-STOCK COMPARISON - Testing {len(symbols)} Stocks")
    print("="*80)
    print(f"\n📋 Stocks to test: {', '.join(symbols)}")
    
    if best_strategy:
        print(f"🎯 Testing best strategy: {best_strategy}")
    
    print(f"\nThis will take approximately {len(symbols) * 20} seconds...")
    input("\nPress Enter to continue...")
    
    all_results = {}
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n\n{'='*80}")
        print(f"📊 [{i}/{len(symbols)}] TESTING: {symbol}")
        print(f"{'='*80}\n")
        
        try:
            # Fetch data
            data = fetch_real_data(symbol, days_back=days_back)
            
            if data is None:
                print(f"⚠️  Skipping {symbol} - Could not fetch data")
                continue
            
            print(f"📊 Data: {len(data)} candles from {data['date'].min().date()} to {data['date'].max().date()}")
            
            # Initialize engine
            engine = BacktestingEngine(
                initial_capital=100000,
                commission_percent=0.03,
                slippage_percent=0.05,
                position_size_percent=10,
                max_positions=3
            )
            
            # Test all strategies
            stock_results = {}
            
            strategies_to_test = {
                'MA Crossover (20/50)': (moving_average_crossover_strategy, {'fast_period': 20, 'slow_period': 50, 'stop_loss_percent': 3.0, 'target_percent': 8.0}),
                'MA Crossover (10/30)': (moving_average_crossover_strategy, {'fast_period': 10, 'slow_period': 30, 'stop_loss_percent': 2.5, 'target_percent': 6.0}),
                'RSI Mean Reversion': (rsi_mean_reversion_strategy, {'rsi_period': 14, 'oversold_threshold': 30, 'overbought_threshold': 70, 'stop_loss_percent': 3.0, 'target_percent': 6.0}),
                'Bollinger Breakout': (bollinger_breakout_strategy, {'bb_period': 20, 'bb_std': 2, 'stop_loss_atr_multiplier': 1.5, 'target_atr_multiplier': 3.0, 'trailing_stop_percent': 1.5}),
                'MACD Momentum': (macd_momentum_strategy, {'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9, 'stop_loss_percent': 2.5, 'target_percent': 5.0}),
                'Momentum Breakout': (momentum_breakout_strategy, {'lookback_period': 20, 'volume_multiplier': 1.5, 'stop_loss_percent': 2.0, 'trailing_stop_percent': 2.0}),
                'Combined (MA+RSI+Vol)': (combined_ma_rsi_volume_strategy, {'fast_ma': 10, 'slow_ma': 30, 'rsi_period': 14, 'rsi_threshold': 50, 'volume_multiplier': 1.3, 'stop_loss_percent': 2.5, 'target_percent': 6.0})
            }
            
            for strategy_name, (strategy_func, params) in strategies_to_test.items():
                try:
                    trades, equity_curve = engine.run_backtest(
                        data=data,
                        strategy_function=strategy_func,
                        strategy_name=strategy_name,
                        **params
                    )
                    
                    metrics = engine.calculate_metrics(trades, equity_curve)
                    
                    stock_results[strategy_name] = {
                        'metrics': metrics,
                        'trades_count': len(trades)
                    }
                    
                except Exception as e:
                    print(f"  ⚠️  {strategy_name}: Error - {e}")
            
            all_results[symbol] = stock_results
            
            # Show quick summary for this stock
            if stock_results:
                best = max(stock_results.items(), key=lambda x: x[1]['metrics'].sharpe_ratio)
                print(f"\n✅ {symbol} Best Strategy: {best[0]}")
                print(f"   Sharpe: {best[1]['metrics'].sharpe_ratio:.2f} | Return: {best[1]['metrics'].total_pnl_percent:.1f}% | Win Rate: {best[1]['metrics'].win_rate:.1f}%")
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate cross-stock comparison
    print("\n\n" + "="*80)
    print("📊 CROSS-STOCK STRATEGY COMPARISON")
    print("="*80 + "\n")
    
    # Create comparison table
    comparison_data = []
    
    for symbol, strategies in all_results.items():
        for strategy_name, result in strategies.items():
            m = result['metrics']
            comparison_data.append({
                'Stock': symbol,
                'Strategy': strategy_name,
                'Trades': result['trades_count'],
                'Win%': f"{m.win_rate:.1f}%",
                'Sharpe': f"{m.sharpe_ratio:.2f}",
                'Return%': f"{m.total_pnl_percent:.1f}%",
                'Max DD%': f"{m.max_drawdown_percent:.1f}%",
                'P&L': f"₹{m.net_pnl:,.0f}"
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Sort by Sharpe ratio (descending)
        df_sorted = df.copy()
        df_sorted['Sharpe_num'] = df_sorted['Sharpe'].astype(float)
        df_sorted = df_sorted.sort_values('Sharpe_num', ascending=False)
        df_sorted = df_sorted.drop('Sharpe_num', axis=1)
        
        print("🏆 TOP 10 STOCK-STRATEGY COMBINATIONS (by Sharpe Ratio):")
        print("-" * 120)
        print(df_sorted.head(10).to_string(index=False))
        
        print("\n\n📈 STOCK-WISE BEST STRATEGIES:")
        print("-" * 120)
        
        for symbol in symbols:
            if symbol in all_results:
                stock_data = all_results[symbol]
                if stock_data:
                    best_strategy = max(stock_data.items(), key=lambda x: x[1]['metrics'].sharpe_ratio)
                    m = best_strategy[1]['metrics']
                    print(f"\n{symbol:12} → {best_strategy[0]:25} | Sharpe: {m.sharpe_ratio:5.2f} | Return: {m.total_pnl_percent:6.1f}% | Win: {m.win_rate:5.1f}%")
        
        print("\n" + "="*80)
        
        # Strategy-wise performance across stocks
        print("\n\n📊 STRATEGY PERFORMANCE ACROSS STOCKS:")
        print("-" * 120)
        
        strategy_names = list(strategies_to_test.keys())
        for strategy_name in strategy_names:
            print(f"\n{strategy_name}:")
            strategy_results = []
            
            for symbol in symbols:
                if symbol in all_results and strategy_name in all_results[symbol]:
                    m = all_results[symbol][strategy_name]['metrics']
                    strategy_results.append({
                        'Stock': symbol,
                        'Sharpe': m.sharpe_ratio,
                        'Return': m.total_pnl_percent,
                        'Win%': m.win_rate
                    })
            
            if strategy_results:
                avg_sharpe = sum(r['Sharpe'] for r in strategy_results) / len(strategy_results)
                avg_return = sum(r['Return'] for r in strategy_results) / len(strategy_results)
                avg_win = sum(r['Win%'] for r in strategy_results) / len(strategy_results)
                
                print(f"  Avg Sharpe: {avg_sharpe:5.2f} | Avg Return: {avg_return:6.1f}% | Avg Win Rate: {avg_win:5.1f}%")
                print(f"  Best on: {max(strategy_results, key=lambda x: x['Sharpe'])['Stock']}")
    
    # Save multi-stock comparison results
    print("\n📁 Saving multi-stock comparison results...")
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('backtest_results')
        output_dir.mkdir(exist_ok=True)
        
        # Save full comparison table as CSV
        if comparison_data:
            csv_filename = f'multi_stock_comparison_{timestamp}.csv'
            csv_path = output_dir / csv_filename
            df_sorted.to_csv(csv_path, index=False)
            print(f"   ✅ Multi-stock CSV saved: {csv_path}")
            
            # Save summary report
            summary_filename = f'multi_stock_summary_{timestamp}.txt'
            summary_path = output_dir / summary_filename
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write(f"MULTI-STOCK COMPARISON REPORT\n")
                f.write("="*80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Stocks Tested: {', '.join(symbols)}\n")
                f.write(f"Total Combinations: {len(comparison_data)}\n")
                f.write("="*80 + "\n\n")
                
                f.write("TOP 10 STOCK-STRATEGY COMBINATIONS (by Sharpe Ratio)\n")
                f.write("-"*120 + "\n")
                f.write(df_sorted.head(10).to_string(index=False))
                f.write("\n\n")
                
                f.write("="*80 + "\n")
                f.write("STOCK-WISE BEST STRATEGIES\n")
                f.write("="*80 + "\n")
                for symbol in symbols:
                    if symbol in all_results:
                        stock_data = all_results[symbol]
                        if stock_data:
                            best_strategy = max(stock_data.items(), key=lambda x: x[1]['metrics'].sharpe_ratio)
                            m = best_strategy[1]['metrics']
                            f.write(f"\n{symbol:12} → {best_strategy[0]:25} | Sharpe: {m.sharpe_ratio:5.2f} | Return: {m.total_pnl_percent:6.1f}% | Win: {m.win_rate:5.1f}%\n")
            
            print(f"   ✅ Summary report saved: {summary_path}")
            
            print("\n" + "="*80)
            print("📦 MULTI-STOCK FILES SAVED")
            print("="*80)
            print(f"📁 Output Directory: {output_dir}")
            print(f"📊 Comparison CSV: {csv_filename}")
            print(f"📝 Summary Report: {summary_filename}")
            print("="*80)
    
    except Exception as e:
        print(f"⚠️  Could not save multi-stock results: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ MULTI-STOCK COMPARISON COMPLETE")
    print("="*80 + "\n")
    
    return all_results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE BACKTESTING WITH REAL KITE DATA")
    print("="*80 + "\n")
    
    print("This script will:")
    print("  1. Connect to Kite API")
    print("  2. Fetch 2 years of historical data for your chosen stock")
    print("  3. Test 7 different strategies:")
    print("      • MA Crossover (Original 20/50)")
    print("      • MA Crossover (Optimized 10/30 - Faster signals)")
    print("      • RSI Mean Reversion (Buy oversold)")
    print("      • Bollinger Breakout (Volume confirmed)")
    print("      • MACD Momentum (Trend following)")
    print("      • Momentum Breakout (Multi-filter approach)")
    print("      • Combined Strategy (MA + RSI + Volume filters)")
    print("  4. Compare performance metrics")
    print("  5. Recommend best strategy")
    print("  6. Display comparison charts")
    
    input("\nPress Enter to continue...")
    
    # Get stock symbol from user
    print("\nCommon stocks: SBIN, RELIANCE, TCS, INFY, HDFCBANK, WIPRO, ITC, HDFC")
    symbol = input("Enter stock symbol to test (default: SBIN): ").strip().upper()
    
    if not symbol:
        symbol = 'SBIN'
        print(f"Using default: {symbol}")
    
    # Run initial backtest on selected stock
    results = run_comprehensive_backtest(symbol=symbol)
    
    if results is None:
        print("\n❌ Initial backtest failed. Exiting.")
        exit(1)
    
    # Ask if user wants to compare with competitors/index
    print("\n\n" + "="*80)
    print("🔄 MULTI-STOCK COMPARISON OPTIONS")
    print("="*80)
    
    compare_option = input("\nDo you want to compare with competitors/index stocks? (y/n): ").strip().lower()
    
    if compare_option == 'y':
        # Check if stock has known competitors
        stock_info = STOCK_SECTOR_MAP.get(symbol)
        
        if stock_info:
            print(f"\n📊 {symbol} belongs to: {stock_info['sector']} sector ({stock_info['index']})")
            print(f"Competitors: {', '.join(stock_info['competitors'][:5])}")
            print(f"\nOptions:")
            print(f"  1. Compare with direct competitors ({len(stock_info['competitors'])} stocks)")
            print(f"  2. Compare with full {stock_info['index']} index ({len(INDEX_CONSTITUENTS.get(stock_info['index'], []))} stocks)")
            print(f"  3. Enter custom stock list")
            print(f"  4. Skip comparison")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                # Compare with competitors
                stocks_to_test = [symbol] + stock_info['competitors']
                print(f"\n✅ Will test {len(stocks_to_test)} stocks: {', '.join(stocks_to_test)}")
                run_multi_stock_comparison(stocks_to_test, days_back=730)
                
            elif choice == '2':
                # Compare with full index
                index_name = stock_info['index']
                stocks_to_test = INDEX_CONSTITUENTS.get(index_name, [])
                print(f"\n✅ Will test {len(stocks_to_test)} {index_name} stocks: {', '.join(stocks_to_test)}")
                run_multi_stock_comparison(stocks_to_test, days_back=730)
                
            elif choice == '3':
                # Custom stock list
                print("\nEnter stock symbols separated by commas (e.g., SBIN,HDFC,ICICI):")
                custom_input = input("Stocks: ").strip().upper()
                stocks_to_test = [s.strip() for s in custom_input.split(',') if s.strip()]
                
                if stocks_to_test:
                    print(f"\n✅ Will test {len(stocks_to_test)} stocks: {', '.join(stocks_to_test)}")
                    run_multi_stock_comparison(stocks_to_test, days_back=730)
                else:
                    print("❌ No valid stocks entered.")
            else:
                print("\n✅ Skipping multi-stock comparison.")
        
        else:
            # Stock not in our mapping - offer custom comparison
            print(f"\n⚠️  {symbol} not found in sector mapping.")
            print("Available options:")
            print("  1. Enter custom competitor stocks")
            print("  2. Select an index to compare")
            print("  3. Skip comparison")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                print("\nEnter stock symbols separated by commas (e.g., SBIN,HDFC,ICICI):")
                custom_input = input("Stocks: ").strip().upper()
                stocks_to_test = [s.strip() for s in custom_input.split(',') if s.strip()]
                
                if stocks_to_test:
                    # Add original stock if not in list
                    if symbol not in stocks_to_test:
                        stocks_to_test.insert(0, symbol)
                    print(f"\n✅ Will test {len(stocks_to_test)} stocks: {', '.join(stocks_to_test)}")
                    run_multi_stock_comparison(stocks_to_test, days_back=730)
                else:
                    print("❌ No valid stocks entered.")
                    
            elif choice == '2':
                print("\nAvailable indices:")
                for i, (index_name, stocks) in enumerate(INDEX_CONSTITUENTS.items(), 1):
                    print(f"  {i}. {index_name} ({len(stocks)} stocks)")
                
                index_choice = input("\nSelect index (1-7): ").strip()
                
                try:
                    index_num = int(index_choice)
                    if 1 <= index_num <= len(INDEX_CONSTITUENTS):
                        index_name = list(INDEX_CONSTITUENTS.keys())[index_num - 1]
                        stocks_to_test = INDEX_CONSTITUENTS[index_name]
                        print(f"\n✅ Will test {len(stocks_to_test)} {index_name} stocks: {', '.join(stocks_to_test)}")
                        run_multi_stock_comparison(stocks_to_test, days_back=730)
                    else:
                        print("❌ Invalid selection.")
                except ValueError:
                    print("❌ Invalid input.")
            else:
                print("\n✅ Skipping multi-stock comparison.")
    
    print("\n" + "="*80)
    print("🎉 BACKTESTING SESSION COMPLETE!")
    print("="*80 + "\n")
