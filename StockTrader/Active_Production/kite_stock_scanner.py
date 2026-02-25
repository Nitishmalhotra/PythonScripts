"""
NSE Stock Scanner using Kite Connect API
Fetches current day's technical data and identifies tradable stocks
"""

from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KiteStockScanner:
    """
    Stock scanner class to fetch NSE data and calculate technical indicators
    """
    
    def __init__(self, api_key, access_token):
        """
        Initialize Kite Connect
        
        Args:
            api_key (str): Your Kite Connect API key
            access_token (str): Your access token (generated after login)
        """
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        logger.info("Kite Connect initialized successfully")
    
    def get_nse_stocks(self, segment='NSE'):
        """
        Get list of NSE stocks
        
        Args:
            segment (str): Market segment (NSE, NFO, BSE, etc.)
            
        Returns:
            list: List of instruments
        """
        try:
            instruments = self.kite.instruments(segment)
            # Filter only equity stocks (not indices, futures, etc.)
            stocks = [inst for inst in instruments if inst['segment'] == segment and 
                     inst['instrument_type'] == 'EQ']
            logger.info(f"Found {len(stocks)} stocks in {segment}")
            return stocks
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            return []
    
    def get_historical_data(self, instrument_token, from_date, to_date, interval='day'):
        """
        Fetch historical data for a stock
        
        Args:
            instrument_token (int): Instrument token
            from_date (datetime): Start date
            to_date (datetime): End date
            interval (str): Candle interval (minute, day, 3minute, 5minute, etc.)
            
        Returns:
            pd.DataFrame: Historical data
        """
        try:
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {instrument_token}: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self, data, period=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            data (pd.Series): Price data
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_sma(self, data, period):
        """
        Calculate Simple Moving Average
        
        Args:
            data (pd.Series): Price data
            period (int): SMA period
            
        Returns:
            pd.Series: SMA values
        """
        return data.rolling(window=period).mean()
    
    def calculate_ema(self, data, period):
        """
        Calculate Exponential Moving Average
        
        Args:
            data (pd.Series): Price data
            period (int): EMA period
            
        Returns:
            pd.Series: EMA values
        """
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data (pd.Series): Price data
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line period
            
        Returns:
            tuple: (macd_line, signal_line, histogram)
        """
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """
        Calculate Bollinger Bands
        
        Args:
            data (pd.Series): Price data
            period (int): Moving average period
            std_dev (float): Standard deviation multiplier
            
        Returns:
            tuple: (upper_band, middle_band, lower_band)
        """
        middle_band = self.calculate_sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return upper_band, middle_band, lower_band
    
    def calculate_atr(self, df, period=14):
        """
        Calculate Average True Range (ATR)
        
        Args:
            df (pd.DataFrame): DataFrame with high, low, close columns
            period (int): ATR period
            
        Returns:
            pd.Series: ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_volume_indicators(self, df):
        """
        Calculate volume-based indicators
        
        Args:
            df (pd.DataFrame): DataFrame with volume column
            
        Returns:
            dict: Volume indicators
        """
        volume_sma_20 = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'] / volume_sma_20
        
        return {
            'volume_sma_20': volume_sma_20.iloc[-1] if len(volume_sma_20) > 0 else 0,
            'volume_ratio': volume_ratio.iloc[-1] if len(volume_ratio) > 0 else 0
        }
    
    def add_technical_indicators(self, df):
        """
        Add all technical indicators to the dataframe
        
        Args:
            df (pd.DataFrame): Historical data
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        if df.empty or len(df) < 50:
            return df
        
        # Price-based indicators
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        df['sma_20'] = self.calculate_sma(df['close'], 20)
        df['sma_50'] = self.calculate_sma(df['close'], 50)
        df['ema_12'] = self.calculate_ema(df['close'], 12)
        df['ema_26'] = self.calculate_ema(df['close'], 26)
        
        # MACD
        macd, signal, hist = self.calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # Bollinger Bands
        upper, middle, lower = self.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # ATR
        df['atr'] = self.calculate_atr(df)
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Price change percentage (day-over-day)
        df['change_pct'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * 100
        
        return df
    
    def scan_stocks(self, stocks, lookback_days=50):
        """
        Scan stocks and calculate technical indicators
        
        Args:
            stocks (list): List of stock instruments
            lookback_days (int): Number of days to fetch historical data
            
        Returns:
            pd.DataFrame: DataFrame with stocks and their technical data
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=lookback_days)
        
        results = []
        
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
                
                if df.empty or len(df) < 50:
                    continue
                
                # Add technical indicators
                df = self.add_technical_indicators(df)
                
                # Get latest values
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                
                stock_data = {
                    'symbol': stock['tradingsymbol'],
                    'token': stock['instrument_token'],
                    'close': latest['close'],
                    'open': latest['open'],
                    'high': latest['high'],
                    'low': latest['low'],
                    'volume': latest['volume'],
                    'change_pct': ((latest['close'] - prev['close']) / prev['close']) * 100,
                    'rsi_14': latest['rsi_14'],
                    'sma_20': latest['sma_20'],
                    'sma_50': latest['sma_50'],
                    'ema_12': latest['ema_12'],
                    'ema_26': latest['ema_26'],
                    'macd': latest['macd'],
                    'macd_signal': latest['macd_signal'],
                    'macd_hist': latest['macd_hist'],
                    'bb_upper': latest['bb_upper'],
                    'bb_middle': latest['bb_middle'],
                    'bb_lower': latest['bb_lower'],
                    'atr': latest['atr'],
                    'volume_ratio': latest['volume_ratio']
                }
                
                results.append(stock_data)
                
            except Exception as e:
                logger.error(f"Error processing {stock['tradingsymbol']}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def scan_stocks_for_strategies(self, stocks, lookback_days=300):
        """
        Scan stocks and return FULL historical data for strategy analysis
        (unlike scan_stocks which returns only latest values)
        
        This method is used by strategies that need historical time series data
        for calculations like moving averages, pattern detection, etc.
        
        Args:
            stocks (list): List of stock instruments
            lookback_days (int): Number of days to fetch (default 300 for 200-day EMA)
            
        Returns:
            pd.DataFrame: DataFrame with full historical data for all stocks
                         Each row is a date-stock combination with all indicators
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
                
                # Skip if insufficient data
                if df.empty or len(df) < 50:
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
    
    def filter_tradable_stocks(self, df, quality_mode='standard'):
        """
        Filter stocks based on trading conditions
        
        Conditions:
        1. RSI between 30-70 (not overbought/oversold)
        2. Price above SMA 20 (uptrend)
        3. MACD histogram positive (bullish momentum)
        4. Volume above average (good liquidity)
        5. Price near middle Bollinger Band (not at extremes)
        
        Args:
            df (pd.DataFrame): DataFrame with stock data
            quality_mode (str): 'basic', 'standard', or 'strict'
            
        Returns:
            pd.DataFrame: Filtered tradable stocks
        """
        if df.empty:
            return df
        
        # Base technical conditions (always applied)
        conditions = (
            (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) &  # Not overbought/oversold
            (df['close'] > df['sma_20']) &  # Price above 20 SMA (uptrend)
            (df['macd_hist'] > 0) &  # Positive MACD histogram (bullish momentum)
            (df['volume_ratio'] > 1.2) &  # Volume 20% above average
            (df['close'] > df['bb_lower']) & (df['close'] < df['bb_upper'])  # Not at BB extremes
        )
        
        # Apply quality filters based on mode
        if quality_mode == 'standard':
            # Exclude low-quality stocks
            quality_conditions = (
                (df['close'] >= 20) &  # Minimum price ₹20
                (~df['symbol'].str.contains('DVR', na=False)) &  # Exclude DVR stocks
                (~df['symbol'].str.contains('-BE', na=False)) &  # Exclude BE series
                (df['volume'] >= 5000)  # Minimum volume
            )
            conditions = conditions & quality_conditions
            
        elif quality_mode == 'strict':
            # High-quality stocks only
            quality_conditions = (
                (df['close'] >= 50) &  # Minimum price ₹50
                (~df['symbol'].str.contains('DVR', na=False)) &  # Exclude DVR stocks
                (~df['symbol'].str.contains('-BE', na=False)) &  # Exclude BE series
                (~df['symbol'].str.contains('-SM', na=False)) &  # Exclude SM series
                (df['macd_hist'] > 0.5) &  # Stronger momentum required
                (df['volume'] >= 10000) &  # Higher minimum volume
                (df['volume_ratio'] > 1.5)  # Higher volume requirement
            )
            conditions = conditions & quality_conditions
        
        filtered_df = df[conditions].copy()
        
        # Sort by multiple criteria
        if not filtered_df.empty:
            filtered_df['score'] = (
                (filtered_df['volume_ratio'] * 0.3) +
                ((70 - abs(50 - filtered_df['rsi_14'])) * 0.3) +  # Prefer RSI near 50
                (filtered_df['macd_hist'] * 0.2) +
                (filtered_df['change_pct'] * 0.2)
            )
            filtered_df = filtered_df.sort_values('score', ascending=False)
        
        return filtered_df


def main():
    """
    Main function to run the stock scanner
    """
    # ============================================
    # CONFIGURATION - Your API credentials
    # ============================================
    API_KEY = "u664cda77q2cf7ft"
    ACCESS_TOKEN = "CaFhMymwgB9benK3htP3T0Lt3Yy1dpGm"  # Run generate_token.py to get this
    
    # ============================================
    # Initialize scanner
    # ============================================
    scanner = KiteStockScanner(API_KEY, ACCESS_TOKEN)
    
    # ============================================
    # Get NSE stocks (you can filter specific stocks here)
    # ============================================
    logger.info("Fetching NSE stocks...")
    all_stocks = scanner.get_nse_stocks('NSE')
    
    # Optional: Filter only liquid stocks (you can add your own filter)
    # For example, only scan Nifty 50 stocks or stocks you're interested in
    # filtered_stocks = [s for s in all_stocks if s['tradingsymbol'] in ['RELIANCE', 'TCS', 'INFY', ...]]
    
    # For demo, let's scan first 50 stocks (remove this limit for full scan)
    stocks_to_scan = all_stocks[:50]
    
    # ============================================
    # Scan stocks and calculate technical indicators
    # ============================================
    logger.info("Scanning stocks and calculating technical indicators...")
    stock_data = scanner.scan_stocks(stocks_to_scan, lookback_days=60)
    
    # ============================================
    # Filter tradable stocks with quality control
    # ============================================
    logger.info("Filtering tradable stocks...")
    
    # Choose quality mode: 'basic', 'standard', or 'strict'
    # - basic: Only technical filters (may include DVR, penny stocks)
    # - standard: Excludes DVR, very low price stocks (recommended)
    # - strict: High-quality stocks only, stronger requirements
    
    quality_mode = 'standard'  # Change to 'basic' or 'strict' as needed
    tradable_stocks = scanner.filter_tradable_stocks(stock_data, quality_mode=quality_mode)
    
    logger.info(f"Quality mode: {quality_mode.upper()}")
    
    # ============================================
    # Display results
    # ============================================
    print("\n" + "="*100)
    print("TRADABLE STOCKS - TECHNICAL ANALYSIS")
    print("="*100)
    
    if not tradable_stocks.empty:
        display_cols = ['symbol', 'close', 'change_pct', 'rsi_14', 'macd_hist', 
                       'volume_ratio', 'score']
        print(tradable_stocks[display_cols].to_string(index=False))
        print(f"\nTotal tradable stocks found: {len(tradable_stocks)}")
        
        # Save to CSV
        output_file = f"tradable_stocks_{datetime.now().strftime('%Y%m%d')}.csv"
        tradable_stocks.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    else:
        print("No tradable stocks found matching the criteria.")
    
    print("="*100)


if __name__ == "__main__":
    main()