"""
Closing Momentum to Gap Scanner
Analyzes closing strength and next-day gap behavior
"""

from kite_stock_scanner import KiteStockScanner
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClosingMomentumScanner(KiteStockScanner):
    """
    Scanner for identifying stocks with strong closing momentum leading to gap-up openings
    """
    
    def scan_stocks_for_momentum(self, stocks, lookback_days=100):
        """
        Custom scan for closing momentum (requires less historical data than long-term strategies)
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=lookback_days)
        
        all_data = []
        
        for idx, stock in enumerate(stocks):
            try:
                logger.info(f"Processing {idx+1}/{len(stocks)}: {stock['tradingsymbol']}")
                
                # Fetch historical data
                df = self.get_historical_data(
                    stock['instrument_token'],
                    from_date,
                    to_date,
                    interval='day'
                )
                
                # Skip if insufficient data (only need 25+ days for closing momentum)
                if df.empty or len(df) < 25:
                    logger.warning(f"Skipping {stock['tradingsymbol']}: insufficient data ({len(df)} days)")
                    continue
                
                # Add technical indicators
                df = self.add_technical_indicators(df)
                
                # Add symbol and token to every row
                df['symbol'] = stock['tradingsymbol']
                df['token'] = stock['instrument_token']
                
                # Append this stock's full history
                all_data.append(df)
                
            except Exception as e:
                logger.error(f"Error processing {stock['tradingsymbol']}: {e}")
                continue
        
        # Combine all stocks' historical data into one DataFrame
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Collected {len(combined_df)} total data points from {len(all_data)} stocks")
            return combined_df
        else:
            logger.warning("No stock data collected")
            return pd.DataFrame()
    
    def strategy_closing_momentum_gap(self, df):
        """
        Closing Momentum to Gap Strategy
        
        CONCEPT:
        Analyzes last 15-min equivalent (closing strength) and next day opening gap
        This strategy identifies stocks where:
        - Strong closing momentum (close near high of day)
        - High volume in closing session
        - Price gaps up on next open
        - Momentum continues into next session
        
        INDICATORS USED:
        1. Close Position: How close the close is to the day's high (closing strength)
        2. Gap %: Difference between today's open and yesterday's close
        3. Volume Trend: Increasing volume suggesting institutional interest
        4. RSI: Momentum confirmation
        5. MACD: Trend strength
        
        ENTRY SETUP:
        - Previous day closed in top 20% of its range (strong close)
        - Gap up opening (current open > previous close)
        - Volume increasing (current > previous day)
        - RSI > 50 (bullish momentum)
        - Price sustaining above open (not gap-fill selloff)
        
        EXIT RULE:
        - If price closes below gap-fill level
        - RSI drops below 40
        - Volume dries up significantly
        """
        if len(df) < 3:
            return pd.DataFrame()
        
        # Calculate closing strength (where price closed in day's range)
        # 100 = closed at high, 0 = closed at low
        day_range = df['high'] - df['low']
        # Avoid division by zero
        day_range = day_range.replace(0, np.nan)
        close_position = ((df['close'] - df['low']) / day_range) * 100
        prev_close_position = close_position.shift(1)
        
        # Calculate gap % (today's open vs yesterday's close)
        gap_pct = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * 100
        
        # Calculate if price held above open (didn't gap fill)
        held_above_open = df['close'] > df['open']
        
        # Volume trend (today vs yesterday)
        volume_increase = df['volume'] > df['volume'].shift(1)
        
        conditions = (
            # Previous day had strong close (top 30% of range)
            (prev_close_position > 70) &
            
            # Gapped up today (at least 0.3%)
            (gap_pct > 0.3) &
            
            # Price held above open (no gap fill)
            (held_above_open) &
            
            # Volume increasing
            (volume_increase) &
            (df['volume_ratio'] > 1.0) &
            
            # Momentum confirmation
            (df['rsi_14'] > 50) &
            (df['rsi_14'] < 80) &  # Not overbought
            
            # Trend confirmation
            (df['macd'] > df['macd_signal']) &
            
            # Price above short-term MA
            (df['close'] > df['sma_20'])
        )
        
        result = df[conditions].copy()
        
        # Add custom columns for analysis
        if not result.empty:
            result['prev_close_strength'] = prev_close_position[conditions]
            result['gap_pct'] = gap_pct[conditions]
            result['held_gap'] = held_above_open[conditions]
            result['close_vs_open_pct'] = ((result['close'] - result['open']) / result['open'] * 100)
        
        return result
    
    def strategy_late_day_surge(self, df):
        """
        Late Day Surge Strategy
        
        Identifies stocks that show strong late-day buying (proxy for last 15-min)
        using close position relative to day's range
        """
        if len(df) < 3:
            return pd.DataFrame()
        
        # Close position in day's range
        day_range = df['high'] - df['low']
        day_range = day_range.replace(0, np.nan)
        close_position = ((df['close'] - df['low']) / day_range) * 100
        
        # Change from open to close
        intraday_change_pct = ((df['close'] - df['open']) / df['open']) * 100
        
        conditions = (
            # Closed in top 25% of range (strong close)
            (close_position > 75) &
            
            # Positive intraday move
            (intraday_change_pct > 0.5) &
            
            # High volume
            (df['volume_ratio'] > 1.2) &
            
            # Momentum strong
            (df['rsi_14'] > 55) &
            (df['rsi_14'] < 75) &
            
            # Uptrend
            (df['close'] > df['sma_20'])
        )
        
        result = df[conditions].copy()
        
        if not result.empty:
            result['close_strength'] = close_position[conditions]
            result['intraday_gain_pct'] = intraday_change_pct[conditions]
        
        return result
    
    def strategy_volume_surge_close(self, df):
        """
        Volume Surge at Close Strategy
        
        Identifies stocks with significantly higher volume, suggesting
        institutional accumulation in closing session
        """
        if len(df) < 3:
            return pd.DataFrame()
        
        # Day range position
        day_range = df['high'] - df['low']
        day_range = day_range.replace(0, np.nan)
        close_position = ((df['close'] - df['low']) / day_range) * 100
        
        conditions = (
            # Very high volume (2x average)
            (df['volume_ratio'] > 2.0) &
            
            # Closed strong
            (close_position > 70) &
            
            # Price up for the day
            (df['close'] > df['open']) &
            
            # RSI bullish
            (df['rsi_14'] > 50) &
            
            # Above key MAs
            (df['close'] > df['sma_20']) &
            (df['sma_20'] > df['sma_50'])
        )
        
        result = df[conditions].copy()
        
        if not result.empty:
            result['close_strength'] = close_position[conditions]
        
        return result
    
    def calculate_risk_reward(self, df):
        """Calculate risk-reward metrics"""
        df['stop_loss'] = df['close'] - (2 * df['atr'])
        df['target_1'] = df['close'] + (2 * df['atr'])
        df['target_2'] = df['close'] + (3 * df['atr'])
        
        df['risk_pct'] = ((df['close'] - df['stop_loss']) / df['close']) * 100
        df['reward_1_pct'] = ((df['target_1'] - df['close']) / df['close']) * 100
        df['reward_2_pct'] = ((df['target_2'] - df['close']) / df['close']) * 100
        
        df['rr_ratio_1'] = df['reward_1_pct'] / df['risk_pct']
        df['rr_ratio_2'] = df['reward_2_pct'] / df['risk_pct']
        
        return df
    
    def scan_with_strategies(self, stock_data):
        """Apply all closing momentum strategies"""
        results = {}
        
        strategies = {
            'Closing Momentum Gap': self.strategy_closing_momentum_gap,
            'Late Day Surge': self.strategy_late_day_surge,
            'Volume Surge Close': self.strategy_volume_surge_close
        }
        
        # Group by symbol and apply strategies
        all_strategy_matches = {name: [] for name in strategies}
        
        symbol_count = 0
        for symbol, symbol_data in stock_data.groupby('symbol'):
            symbol_count += 1
            symbol_data = symbol_data.reset_index(drop=True)
            
            for strategy_name, strategy_func in strategies.items():
                try:
                    filtered = strategy_func(symbol_data)
                    
                    if filtered is None or filtered.empty:
                        continue
                    
                    # Keep last matching row
                    if not filtered.empty:
                        last_row = filtered.iloc[[-1]].copy()
                        if 'symbol' not in last_row.columns:
                            last_row['symbol'] = symbol
                        all_strategy_matches[strategy_name].append(last_row)
                        
                except Exception as e:
                    logger.error(f"{strategy_name} on {symbol}: Error - {str(e)}")
                    continue
        
        logger.info(f"Processed {symbol_count} symbols for closing momentum strategies")
        
        # Combine results
        for strategy_name, matches in all_strategy_matches.items():
            if matches:
                combined = pd.concat(matches, ignore_index=True)
                
                if not combined.empty:
                    # Deduplicate
                    if 'date' in combined.columns:
                        combined = combined.sort_values('date', ascending=False).drop_duplicates(
                            subset=['symbol'], keep='first').sort_values('symbol').reset_index(drop=True)
                    else:
                        combined = combined.drop_duplicates(
                            subset=['symbol'], keep='first').sort_values('symbol').reset_index(drop=True)
                    
                    # Add risk-reward
                    combined = self.calculate_risk_reward(combined)
                    combined['scan_date'] = datetime.now().strftime('%Y-%m-%d')
                    combined['strategy'] = strategy_name
                    results[strategy_name] = combined
                    logger.info(f"{strategy_name}: Found {len(combined)} stocks")
            else:
                logger.warning(f"{strategy_name}: No matches found")
        
        return results


def main():
    """Main function for closing momentum scanning"""
    
    # Configuration
    API_KEY = os.environ.get('KITE_API_KEY', "u664cda77q2cf7ft")
    ACCESS_TOKEN = os.environ.get('KITE_ACCESS_TOKEN')
    
    # Try reading saved credentials
    cred_file = os.path.join(os.path.dirname(__file__), 'kite_credentials.txt')
    if not ACCESS_TOKEN and os.path.exists(cred_file):
        try:
            with open(cred_file, 'r') as f:
                for line in f:
                    if line.startswith('ACCESS_TOKEN='):
                        ACCESS_TOKEN = line.strip().split('=', 1)[1]
                    if line.startswith('API_KEY=') and (not API_KEY):
                        API_KEY = line.strip().split('=', 1)[1]
        except Exception:
            pass
    
    if not ACCESS_TOKEN:
        print('\nERROR: ACCESS_TOKEN not found. Run _exchange_request_token.py first.')
        return
    
    # Initialize scanner
    scanner = ClosingMomentumScanner(API_KEY, ACCESS_TOKEN)
    
    # Get stocks
    logger.info("Fetching NSE stocks...")
    all_stocks = scanner.get_nse_stocks('NSE')
    
    # Nifty 50 stocks
    nifty50_symbols = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR',
        'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK',
        'ASIANPAINT', 'MARUTI', 'SUNPHARMA', 'TITAN', 'BAJFINANCE',
        'WIPRO', 'ULTRACEMCO', 'NESTLEIND', 'ONGC', 'NTPC', 'POWERGRID',
        'HCLTECH', 'COALINDIA', 'BAJAJFINSV', 'M&M', 'ADANIPORTS',
        'TATASTEEL', 'GRASIM', 'TECHM', 'INDUSINDBK', 'DIVISLAB',
        'DRREDDY', 'CIPLA', 'EICHERMOT', 'HINDALCO', 'BRITANNIA',
        'JSWSTEEL', 'APOLLOHOSP', 'HEROMOTOCO', 'SHREECEM', 'BPCL',
        'TATACONSUM', 'UPL', 'SBILIFE', 'BAJAJ-AUTO', 'ADANIENT', 'TATAMOTORS'
    ]
    
    def _normalize(sym):
        return ''.join(ch for ch in str(sym).upper() if ch.isalnum())
    
    nifty_set = set(_normalize(s) for s in nifty50_symbols)
    stocks_to_scan = [s for s in all_stocks if _normalize(s.get('tradingsymbol', '')) in nifty_set]
    
    if not stocks_to_scan:
        logger.warning('No Nifty50 symbols matched; using first 50 instruments')
        stocks_to_scan = all_stocks[:50]
    
    # Scan stocks
    logger.info("Scanning stocks for closing momentum patterns...")
    stock_data = scanner.scan_stocks_for_momentum(stocks_to_scan, lookback_days=150)  # Custom method with lower min requirement
    
    if stock_data.empty:
        print("No stock data retrieved. Please check your API credentials.")
        return
    
    print(f"\nRetrieved data for {stock_data['symbol'].nunique()} stocks")
    print(f"Total data points: {len(stock_data)}")
    print("Analyzing closing momentum patterns...\n")
    
    # Apply strategies
    strategy_results = scanner.scan_with_strategies(stock_data)
    
    # Display results
    print("\n" + "="*100)
    print("CLOSING MOMENTUM SCANNER - RESULTS")
    print("="*100)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = f"closing_momentum_{timestamp}.csv"
    
    # Collect all results
    all_results = [stocks for stocks in strategy_results.values() if not stocks.empty]
    
    if all_results:
        combined_all = pd.concat(all_results, ignore_index=True)
        
        # Display each strategy
        for strategy_name, stocks in strategy_results.items():
            if not stocks.empty:
                print(f"\n{'='*100}")
                print(f"STRATEGY: {strategy_name} ({len(stocks)} stocks)")
                print(f"{'='*100}")
                
                # Select display columns based on available columns
                base_cols = ['symbol', 'close', 'rsi_14', 'volume_ratio']
                
                if 'prev_close_strength' in stocks.columns:
                    display_cols = base_cols + ['prev_close_strength', 'gap_pct', 'close_vs_open_pct', 'rr_ratio_1']
                elif 'close_strength' in stocks.columns:
                    display_cols = base_cols + ['close_strength', 'intraday_gain_pct', 'rr_ratio_1']
                else:
                    display_cols = base_cols + ['stop_loss', 'target_1', 'rr_ratio_1']
                
                available_cols = [col for col in display_cols if col in stocks.columns]
                print(stocks[available_cols].head(10).to_string(index=False))
        
        # Save to CSV
        combined_all.to_csv(csv_file, index=False)
        print(f"\n{'='*100}")
        print(f"✓ CSV file created: {csv_file}")
        print(f"✓ Total stocks found: {len(combined_all)}")
        print(f"{'='*100}")
        
        # Summary analysis
        if len(combined_all) > 0:
            print(f"\nSTRATEGY DISTRIBUTION:")
            print(combined_all['strategy'].value_counts().to_string())
            
            # Stocks in multiple strategies
            symbol_counts = combined_all['symbol'].value_counts()
            multi_strategy = symbol_counts[symbol_counts > 1]
            
            if not multi_strategy.empty:
                print(f"\n{'='*100}")
                print("STOCKS MATCHING MULTIPLE CLOSING MOMENTUM STRATEGIES")
                print(f"{'='*100}")
                for symbol in multi_strategy.index:
                    strategies = combined_all[combined_all['symbol'] == symbol]['strategy'].tolist()
                    print(f"{symbol}: {', '.join(strategies)}")
    else:
        print("\n❌ No closing momentum patterns found.")
    
    print("\n" + "="*100)
    print("SCAN COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
