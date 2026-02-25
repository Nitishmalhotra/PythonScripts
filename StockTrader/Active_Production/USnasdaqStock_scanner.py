"""
Enhanced NASDAQ-100 Stock Swing Trading Screener
Analyzes NASDAQ-100 stocks with comprehensive technical & fundamental indicators
Version 1.0 - Complete Edition
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Try importing technical analysis library
try:
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False
    print("Warning: ta library not installed. Install with: pip install ta")


class NasdaqStockScreener:
    """Enhanced NASDAQ-100 Stock Swing Trading Screener with Advanced Analytics"""
    
    def __init__(self, stock_list, period='6mo', interval='1d', benchmark='QQQ'):
        self.stock_list = stock_list
        self.period = period
        self.interval = interval
        self.benchmark = benchmark
        self.results = []
        self.benchmark_data = None
        
    # ==================== BASIC INDICATORS ====================
    
    def calculate_sma(self, data, window):
        """Calculate Simple Moving Average"""
        try:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            
            if TA_LIB_AVAILABLE:
                return SMAIndicator(data, window=window).sma_indicator()
            else:
                return data.rolling(window=window).mean()
        except:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            return data.rolling(window=window).mean()
    
    def calculate_ema(self, data, window):
        """Calculate Exponential Moving Average"""
        try:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            
            if TA_LIB_AVAILABLE:
                return EMAIndicator(data, window=window).ema_indicator()
            else:
                return data.ewm(span=window, adjust=False).mean()
        except:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            return data.ewm(span=window, adjust=False).mean()
    
    def calculate_rsi(self, data, window=14):
        """Calculate Relative Strength Index"""
        try:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            
            if TA_LIB_AVAILABLE:
                return RSIIndicator(data, window=window).rsi()
            else:
                delta = data.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
        except:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        try:
            if TA_LIB_AVAILABLE:
                close_data = pd.Series(data) if not isinstance(data, pd.Series) else data
                macd_obj = MACD(close_data, window_fast=fast, window_slow=slow, window_sign=signal)
                return macd_obj.macd(), macd_obj.macd_signal(), macd_obj.macd_diff()
            else:
                ema_fast = data.ewm(span=fast, adjust=False).mean()
                ema_slow = data.ewm(span=slow, adjust=False).mean()
                macd = ema_fast - ema_slow
                macd_signal = macd.ewm(span=signal, adjust=False).mean()
                macd_hist = macd - macd_signal
                return macd, macd_signal, macd_hist
        except:
            ema_fast = data.ewm(span=fast, adjust=False).mean()
            ema_slow = data.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            macd_hist = macd - macd_signal
            return macd, macd_signal, macd_hist
    
    def calculate_adx(self, high, low, close, window=14):
        """Calculate Average Directional Index"""
        try:
            if TA_LIB_AVAILABLE:
                high_data = pd.Series(high) if not isinstance(high, pd.Series) else high
                low_data = pd.Series(low) if not isinstance(low, pd.Series) else low
                close_data = pd.Series(close) if not isinstance(close, pd.Series) else close
                adx_obj = ADXIndicator(high_data, low_data, close_data, window=window)
                return adx_obj.adx()
            else:
                tr = np.maximum(
                    np.maximum(high - low, np.abs(high - close.shift(1))),
                    np.abs(low - close.shift(1))
                )
                atr = tr.rolling(window=window).mean()
                
                plus_dm = np.where(high - high.shift(1) > low.shift(1) - low, high - high.shift(1), 0)
                minus_dm = np.where(low.shift(1) - low > high - high.shift(1), low.shift(1) - low, 0)
                
                plus_di = 100 * (pd.Series(plus_dm).rolling(window=window).mean() / atr)
                minus_di = 100 * (pd.Series(minus_dm).rolling(window=window).mean() / atr)
                
                dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
                adx = dx.rolling(window=window).mean()
                
                return adx
        except:
            tr = np.maximum(
                np.maximum(high - low, np.abs(high - close.shift(1))),
                np.abs(low - close.shift(1))
            )
            atr = tr.rolling(window=window).mean()
            
            plus_dm = np.where(high - high.shift(1) > low.shift(1) - low, high - high.shift(1), 0)
            minus_dm = np.where(low.shift(1) - low > high - high.shift(1), low.shift(1) - low, 0)
            
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=window).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=window).mean() / atr)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=window).mean()
            
            return adx
    
    # ==================== ENHANCED INDICATORS ====================
    
    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        try:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            
            if TA_LIB_AVAILABLE:
                bb = BollingerBands(data, window=window, window_dev=num_std)
                return bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband()
            else:
                sma = data.rolling(window=window).mean()
                std = data.rolling(window=window).std()
                upper = sma + (std * num_std)
                lower = sma - (std * num_std)
                return sma, upper, lower
        except:
            if not isinstance(data, pd.Series):
                data = pd.Series(data)
            sma = data.rolling(window=window).mean()
            std = data.rolling(window=window).std()
            upper = sma + (std * num_std)
            lower = sma - (std * num_std)
            return sma, upper, lower
    
    def calculate_atr(self, high, low, close, window=14):
        """Calculate Average True Range"""
        try:
            if TA_LIB_AVAILABLE:
                high_data = pd.Series(high) if not isinstance(high, pd.Series) else high
                low_data = pd.Series(low) if not isinstance(low, pd.Series) else low
                close_data = pd.Series(close) if not isinstance(close, pd.Series) else close
                return AverageTrueRange(high_data, low_data, close_data, window=window).average_true_range()
            else:
                tr1 = high - low
                tr2 = np.abs(high - close.shift(1))
                tr3 = np.abs(low - close.shift(1))
                tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
                return tr.rolling(window=window).mean()
        except:
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            return tr.rolling(window=window).mean()
    
    def calculate_stochastic(self, high, low, close, window=14):
        """Calculate Stochastic Oscillator"""
        try:
            if TA_LIB_AVAILABLE:
                high_data = pd.Series(high) if not isinstance(high, pd.Series) else high
                low_data = pd.Series(low) if not isinstance(low, pd.Series) else low
                close_data = pd.Series(close) if not isinstance(close, pd.Series) else close
                stoch = StochasticOscillator(high_data, low_data, close_data, window=window)
                return stoch.stoch(), stoch.stoch_signal()
            else:
                lowest_low = low.rolling(window=window).min()
                highest_high = high.rolling(window=window).max()
                k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
                d = k.rolling(window=3).mean()
                return k, d
        except:
            lowest_low = low.rolling(window=window).min()
            highest_high = high.rolling(window=window).max()
            k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d = k.rolling(window=3).mean()
            return k, d
    
    # ==================== PATTERN DETECTION ====================
    
    def detect_price_patterns(self, data):
        """Detect price patterns"""
        patterns = {
            'higher_highs': False,
            'higher_lows': False,
            'lower_highs': False,
            'lower_lows': False,
            'support_level': None,
            'resistance_level': None
        }
        
        try:
            recent = data.tail(20)
            highs = recent['high'].tail(10)
            lows = recent['low'].tail(10)
            
            if len(highs) >= 3 and len(lows) >= 3:
                patterns['higher_highs'] = highs.iloc[-1] > highs.iloc[-3] > highs.iloc[-5]
                patterns['higher_lows'] = lows.iloc[-1] > lows.iloc[-3] > lows.iloc[-5]
                patterns['lower_highs'] = highs.iloc[-1] < highs.iloc[-3] < highs.iloc[-5]
                patterns['lower_lows'] = lows.iloc[-1] < lows.iloc[-3] < lows.iloc[-5]
            
            if len(data) >= 50:
                last_50 = data.tail(50)
                patterns['support_level'] = last_50['low'].min()
                patterns['resistance_level'] = last_50['high'].max()
        except:
            pass
        
        return patterns
    
    # ==================== VOLUME ANALYSIS ====================
    
    def analyze_volume(self, data):
        """Analyze volume patterns"""
        volume_metrics = {
            'avg_volume_20': 0,
            'volume_ratio': 1.0,
            'volume_trend': 'neutral',
            'volume_breakout': False
        }
        
        try:
            volume_metrics['avg_volume_20'] = data['volume'].tail(20).mean()
            current_vol = data['volume'].iloc[-1]
            avg_vol = volume_metrics['avg_volume_20']
            volume_metrics['volume_ratio'] = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            recent_vol = data['volume'].tail(10).mean()
            prev_vol = data['volume'].tail(20).head(10).mean()
            
            if recent_vol > prev_vol * 1.2:
                volume_metrics['volume_trend'] = 'increasing'
            elif recent_vol < prev_vol * 0.8:
                volume_metrics['volume_trend'] = 'decreasing'
            else:
                volume_metrics['volume_trend'] = 'stable'
            
            volume_metrics['volume_breakout'] = volume_metrics['volume_ratio'] > 2.0
        except:
            pass
        
        return volume_metrics
    
    # ==================== FUNDAMENTAL ANALYSIS ====================
    
    def get_stock_fundamentals(self, ticker):
        """Get stock fundamental data"""
        fundamentals = {
            'name': 'N/A',
            'sector': 'N/A',
            'market_cap': 'N/A',
            'pe_ratio': 'N/A',
            'eps': 'N/A',
            'pb_ratio': 'N/A',
            'dividend_yield': 'N/A',
            'beta': 'N/A',
            'ytd_return': 'N/A',
            '52w_high': 'N/A',
            '52w_low': 'N/A',
            'avg_volume': 'N/A'
        }
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            fundamentals['name'] = info.get('longName', 'N/A')
            fundamentals['sector'] = info.get('sector', 'N/A')
            
            if 'marketCap' in info and info['marketCap']:
                market_cap = info['marketCap'] / 1e9
                fundamentals['market_cap'] = f"${market_cap:.2f}B"
            
            if 'trailingPE' in info and info['trailingPE']:
                fundamentals['pe_ratio'] = f"{info['trailingPE']:.2f}"
            
            if 'trailingEps' in info and info['trailingEps']:
                fundamentals['eps'] = f"${info['trailingEps']:.2f}"
            
            if 'priceToBook' in info and info['priceToBook']:
                fundamentals['pb_ratio'] = f"{info['priceToBook']:.2f}"
            
            if 'trailingAnnualDividendYield' in info and info['trailingAnnualDividendYield']:
                fundamentals['dividend_yield'] = f"{info['trailingAnnualDividendYield']*100:.2f}%"
            
            if 'beta' in info and info['beta']:
                fundamentals['beta'] = f"{info['beta']:.2f}"
            
            if 'ytdReturn' in info and info['ytdReturn']:
                fundamentals['ytd_return'] = f"{info['ytdReturn']*100:.2f}%"
            
            if 'fiftyTwoWeekHigh' in info and info['fiftyTwoWeekHigh']:
                fundamentals['52w_high'] = f"${info['fiftyTwoWeekHigh']:.2f}"
            
            if 'fiftyTwoWeekLow' in info and info['fiftyTwoWeekLow']:
                fundamentals['52w_low'] = f"${info['fiftyTwoWeekLow']:.2f}"
            
            if 'averageVolume' in info and info['averageVolume']:
                avg_vol = info['averageVolume'] / 1e6
                fundamentals['avg_volume'] = f"{avg_vol:.2f}M"
        except:
            pass
        
        return fundamentals
    
    # ==================== RISK METRICS ====================
    
    def calculate_risk_metrics(self, data):
        """Calculate risk and performance metrics"""
        metrics = {
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'return_1m': 0,
            'return_3m': 0,
            'return_6m': 0
        }
        
        try:
            returns = data['close'].pct_change().dropna()
            metrics['volatility'] = returns.std() * np.sqrt(252) * 100
            
            risk_free_rate = 0.02 / 252
            excess_returns = returns - risk_free_rate
            if returns.std() != 0:
                metrics['sharpe_ratio'] = (excess_returns.mean() / returns.std()) * np.sqrt(252)
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min() * 100
            
            if len(data) >= 21:
                metrics['return_1m'] = ((data['close'].iloc[-1] / data['close'].iloc[-21]) - 1) * 100
            if len(data) >= 63:
                metrics['return_3m'] = ((data['close'].iloc[-1] / data['close'].iloc[-63]) - 1) * 100
            if len(data) >= 126:
                metrics['return_6m'] = ((data['close'].iloc[-1] / data['close'].iloc[-126]) - 1) * 100
        except:
            pass
        
        return metrics
    
    # ==================== RELATIVE STRENGTH ====================
    
    def calculate_relative_strength(self, data, benchmark_data):
        """Calculate relative strength vs benchmark"""
        try:
            if benchmark_data is None or len(benchmark_data) == 0:
                return 0
            
            if len(data) > len(benchmark_data):
                data = data.tail(len(benchmark_data))
            elif len(benchmark_data) > len(data):
                benchmark_data = benchmark_data.tail(len(data))
            
            stock_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
            benchmark_return = (benchmark_data['close'].iloc[-1] / benchmark_data['close'].iloc[0] - 1) * 100
            
            return stock_return - benchmark_return
        except:
            return 0
    
    # ==================== SCORING SYSTEM ====================
    
    def calculate_composite_score(self, signals):
        """Calculate composite score based on multiple factors"""
        score = 0
        
        if signals.get('trend') == 'bullish':
            score += 10
        elif signals.get('trend') == 'bearish':
            score -= 10
        
        if signals.get('macd_signal') == 'bullish':
            score += 10
        elif signals.get('macd_signal') == 'bearish':
            score -= 10
        
        if signals.get('rsi_zone') == 'oversold':
            score += 5
        elif signals.get('rsi_zone') == 'overbought':
            score -= 5
        elif signals.get('rsi_zone') == 'bullish':
            score += 5
        
        if signals.get('stoch_signal') == 'oversold':
            score += 5
        elif signals.get('stoch_signal') == 'overbought':
            score -= 5
        
        if signals.get('adx_strength') == 'strong':
            score += 10
        elif signals.get('adx_strength') == 'weak':
            score -= 5
        
        if signals.get('price_pattern') in ['higher_highs_lows', 'breakout']:
            score += 10
        elif signals.get('price_pattern') == 'lower_highs_lows':
            score -= 10
        
        if signals.get('volume_confirmation'):
            score += 10
        
        if signals.get('volume_breakout'):
            score += 10
        
        rel_strength = signals.get('relative_strength', 0)
        if rel_strength > 5:
            score += 10
        elif rel_strength > 0:
            score += 5
        elif rel_strength < -5:
            score -= 10
        elif rel_strength < 0:
            score -= 5
        
        if signals.get('bollinger_position') == 'oversold':
            score += 5
        elif signals.get('bollinger_position') == 'overbought':
            score -= 5
        
        score = max(0, min(100, 50 + score))
        return score
    
    # ==================== MAIN SCAN FUNCTION ====================
    
    def scan(self):
        """Scan all stocks for trading signals"""
        
        try:
            print(f"Downloading benchmark ({self.benchmark}) data...")
            self.benchmark_data = yf.download(
                self.benchmark,
                period=self.period,
                interval=self.interval,
                progress=False
            )
            if isinstance(self.benchmark_data.columns, pd.MultiIndex):
                self.benchmark_data.columns = self.benchmark_data.columns.get_level_values(0)
            self.benchmark_data.columns = [str(col).lower() for col in self.benchmark_data.columns]
        except:
            print("Warning: Could not download benchmark data")
            self.benchmark_data = None
        
        for i, ticker in enumerate(self.stock_list, 1):
            try:
                print(f"[{i}/{len(self.stock_list)}] Scanning {ticker}...", end=' ')
                
                data = yf.download(
                    ticker,
                    period=self.period,
                    interval=self.interval,
                    progress=False
                )
                
                if isinstance(data, pd.Series):
                    data = data.to_frame()
                
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                data.columns = [str(col).lower() for col in data.columns]
                
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    print(f"❌ (Missing: {missing_cols})")
                    continue
                
                if len(data) < 50:
                    print(f"❌ (Insufficient data)")
                    continue
                
                data['SMA_20'] = self.calculate_sma(data['close'], 20)
                data['SMA_50'] = self.calculate_sma(data['close'], 50)
                data['SMA_200'] = self.calculate_sma(data['close'], 200)
                data['EMA_12'] = self.calculate_ema(data['close'], 12)
                data['EMA_26'] = self.calculate_ema(data['close'], 26)
                data['RSI_14'] = self.calculate_rsi(data['close'], 14)
                
                macd, macd_signal, macd_hist = self.calculate_macd(data['close'])
                data['MACD'] = macd
                data['MACD_Signal'] = macd_signal
                data['MACD_Hist'] = macd_hist
                
                data['ADX_14'] = self.calculate_adx(data['high'], data['low'], data['close'], 14)
                
                bb_mid, bb_upper, bb_lower = self.calculate_bollinger_bands(data['close'], 20, 2)
                data['BB_Mid'] = bb_mid
                data['BB_Upper'] = bb_upper
                data['BB_Lower'] = bb_lower
                
                data['ATR_14'] = self.calculate_atr(data['high'], data['low'], data['close'], 14)
                
                stoch_k, stoch_d = self.calculate_stochastic(data['high'], data['low'], data['close'], 14)
                data['Stoch_K'] = stoch_k
                data['Stoch_D'] = stoch_d
                
                latest = data.iloc[-1]
                prev = data.iloc[-2]
                
                if pd.isna(latest['SMA_20']) or pd.isna(latest['RSI_14']) or pd.isna(latest['MACD']):
                    print(f"❌ (Incomplete indicators)")
                    continue
                
                signals = {}
                
                signals['trend'] = 'bullish' if latest['SMA_20'] > latest['SMA_50'] else 'bearish'
                
                macd_cross = (latest['MACD'] > latest['MACD_Signal']) and (prev['MACD'] <= prev['MACD_Signal'])
                signals['macd_signal'] = 'bullish' if macd_cross else 'bearish' if latest['MACD'] < latest['MACD_Signal'] else 'neutral'
                
                if latest['RSI_14'] < 30:
                    signals['rsi_zone'] = 'oversold'
                elif latest['RSI_14'] > 70:
                    signals['rsi_zone'] = 'overbought'
                elif 50 < latest['RSI_14'] < 70:
                    signals['rsi_zone'] = 'bullish'
                elif 30 < latest['RSI_14'] < 50:
                    signals['rsi_zone'] = 'bearish'
                else:
                    signals['rsi_zone'] = 'neutral'
                
                if not pd.isna(latest['Stoch_K']):
                    if latest['Stoch_K'] < 20:
                        signals['stoch_signal'] = 'oversold'
                    elif latest['Stoch_K'] > 80:
                        signals['stoch_signal'] = 'overbought'
                    else:
                        signals['stoch_signal'] = 'neutral'
                else:
                    signals['stoch_signal'] = 'neutral'
                
                signals['adx_strength'] = 'strong' if latest['ADX_14'] > 25 else 'weak' if latest['ADX_14'] < 20 else 'moderate'
                
                bb_position = (latest['close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']) * 100
                if bb_position < 20:
                    signals['bollinger_position'] = 'oversold'
                elif bb_position > 80:
                    signals['bollinger_position'] = 'overbought'
                else:
                    signals['bollinger_position'] = 'neutral'
                
                volume_metrics = self.analyze_volume(data)
                signals['volume_confirmation'] = volume_metrics['volume_ratio'] > 1.2
                signals['volume_breakout'] = volume_metrics['volume_breakout']
                
                patterns = self.detect_price_patterns(data)
                if patterns['higher_highs'] and patterns['higher_lows']:
                    signals['price_pattern'] = 'higher_highs_lows'
                elif patterns['lower_highs'] and patterns['lower_lows']:
                    signals['price_pattern'] = 'lower_highs_lows'
                else:
                    signals['price_pattern'] = 'consolidation'
                
                signals['relative_strength'] = self.calculate_relative_strength(data, self.benchmark_data)
                risk_metrics = self.calculate_risk_metrics(data)
                fundamentals = self.get_stock_fundamentals(ticker)
                composite_score = self.calculate_composite_score(signals)
                
                passes_checklist, checks, confidence_count = self.validate_trading_setup(data, latest, signals, volume_metrics)
                
                if composite_score >= 70:
                    final_signal = "STRONG BUY"
                    signal_strength = 5
                elif composite_score >= 60:
                    final_signal = "BUY"
                    signal_strength = 4
                elif composite_score >= 55:
                    final_signal = "WEAK BUY"
                    signal_strength = 3
                elif composite_score <= 30:
                    final_signal = "STRONG SELL"
                    signal_strength = -5
                elif composite_score <= 40:
                    final_signal = "SELL"
                    signal_strength = -4
                elif composite_score <= 45:
                    final_signal = "WEAK SELL"
                    signal_strength = -3
                else:
                    final_signal = "HOLD"
                    signal_strength = 0
                
                self.results.append({
                    'Ticker': ticker,
                    'Name': fundamentals['name'][:25] if len(fundamentals['name']) > 25 else fundamentals['name'],
                    'Sector': fundamentals['sector'][:20],
                    'Price': f"${latest['close']:.2f}",
                    'EdgeScore': self.calculate_edge_score(
                        checklist_count=confidence_count,
                        composite_score=composite_score,
                        sharpe_ratio=risk_metrics['sharpe_ratio'],
                        relative_strength=signals['relative_strength'],
                        max_dd=risk_metrics['max_drawdown'],
                        return_1m=risk_metrics['return_1m'],
                        return_3m=risk_metrics['return_3m'],
                        return_6m=risk_metrics['return_6m']
                    ),
                    'Score': composite_score,
                    'Signal': final_signal,
                    'Checklist': f"{confidence_count}/7",
                    'ChecksPassed': passes_checklist,
                    'RSI': f"{latest['RSI_14']:.1f}",
                    'MACD': 'Bull' if signals['macd_signal'] == 'bullish' else 'Bear',
                    'ADX': f"{latest['ADX_14']:.1f}",
                    'Stoch': f"{latest['Stoch_K']:.1f}" if not pd.isna(latest['Stoch_K']) else 'N/A',
                    'Vol_Ratio': f"{volume_metrics['volume_ratio']:.2f}x",
                    'Rel_Str': f"{signals['relative_strength']:+.1f}%",
                    'Volatility': f"{risk_metrics['volatility']:.1f}%",
                    'Sharpe': f"{risk_metrics['sharpe_ratio']:.2f}",
                    'MaxDD': f"{risk_metrics['max_drawdown']:.1f}%",
                    '1M': f"{risk_metrics['return_1m']:+.1f}%",
                    '3M': f"{risk_metrics['return_3m']:+.1f}%",
                    '6M': f"{risk_metrics['return_6m']:+.1f}%",
                    'MarketCap': fundamentals['market_cap'],
                    'PE': fundamentals['pe_ratio'],
                    'EPS': fundamentals['eps'],
                    'Dividend': fundamentals['dividend_yield'],
                    'Beta': fundamentals['beta']
                })
                
                print(f"✓ Score: {composite_score}/100 | {final_signal}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)[:50]}")
                continue
    
    def display_results(self):
        """Display scanning results"""
        if not self.results:
            print("\n❌ No results found")
            return
    
        df = pd.DataFrame(self.results)
        
        # Extract numeric checklist count from string (e.g., "5/7" -> 5)
        df['_ChecklistNum'] = df['Checklist'].str.split('/').str[0].astype(int)
        
        # Set dtypes to prevent auto-conversion
        df['Checklist'] = df['Checklist'].astype(str)
        df['Score'] = df['Score'].astype(int)
        
        # Apply classification and action columns with volatility cap
        df[['Classification', 'Action']] = df.apply(
            lambda row: pd.Series(classify_edge(row['EdgeScore'], row['Volatility'])), 
            axis=1
        )
        
        df = df.sort_values(by='EdgeScore', ascending=False)  # Sort by EdgeScore instead of Score
        
        print("\n" + "="*220)
        print("NASDAQ-100 STOCK SWING TRADING SCREENER - COMPREHENSIVE ANALYSIS")
        print(f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Benchmark: {self.benchmark}")
        print("="*220)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 20)
        
        # High-confidence trades (EdgeScore >= 70)
        high_confidence = df[(df['EdgeScore'] >= 70)]
        
        # Update top displays to include EdgeScore:
        top_setups = high_confidence.head(10)[['Ticker', 'Name', 'EdgeScore', 'Classification', 'Action', 'Score', 'Signal', 'Checklist', 
                                              'RSI', 'ADX', 'Vol_Ratio', 'PE', 'MarketCap']]
        
        top_20 = df.head(20)[['Ticker', 'Name', 'Sector', 'EdgeScore', 'Classification', 'Action', 'Score', 'Signal', 'Checklist',
                           'RSI', 'MACD', 'Vol_Ratio', 'Rel_Str', '1M', 'PE', 'MarketCap']]
        print(top_setups.to_string(index=False))
        
        print("\n" + "-"*220)
        print("SUMMARY:")
        print("="*220)
        strong_buy = len(df[df['Signal'] == 'STRONG BUY'])
        buy = len(df[df['Signal'] == 'BUY'])
        weak_buy = len(df[df['Signal'] == 'WEAK BUY'])
        hold = len(df[df['Signal'] == 'HOLD'])
        weak_sell = len(df[df['Signal'] == 'WEAK SELL'])
        sell = len(df[df['Signal'] == 'SELL'])
        strong_sell = len(df[df['Signal'] == 'STRONG SELL'])
        
        passed_checks = len(df[df['ChecksPassed'] == True])
        
        print(f"  🟢 STRONG BUY: {strong_buy}")
        print(f"  🟢 BUY: {buy}")
        print(f"  🟡 WEAK BUY: {weak_buy}")
        print(f"  ⚪ HOLD: {hold}")
        print(f"  🟡 WEAK SELL: {weak_sell}")
        print(f"  🔴 SELL: {sell}")
        print(f"  🔴 STRONG SELL: {strong_sell}")
        print(f"\n  ✅ Passed 7-Point Checklist (4+/7): {passed_checks}")
        print(f"  📊 Average Score: {df['Score'].mean():.1f}/100")
        print(f"  🏆 Best Performer: {df.iloc[0]['Ticker']} ({df.iloc[0]['Score']}/100)")
        print(f"  📈 Total Scanned: {len(df)}")
        print("="*220 + "\n")
    
    def save_results(self, filename=None):
        """Save results to CSV"""
        if not self.results:
            print("No results to save")
            return
        
        if filename is None:
            filename = f"nasdaq100_stock_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame(self.results)
        
        # Apply classification and action columns with volatility cap
        df[['Classification', 'Action']] = df.apply(
            lambda row: pd.Series(classify_edge(row['EdgeScore'], row['Volatility'])), 
            axis=1
        )
        
        # Sort by EdgeScore
        df = df.sort_values(by='EdgeScore', ascending=False)
        
        df.to_csv(filename, index=False)
        print(f"✓ Results saved to: {filename}")


# ==================== VALIDATION CHECKLIST ====================
    
    def validate_trading_setup(self, data, latest, signals, volume_metrics):
        """
        Validate 7-point checklist for high-confidence trades
        Returns: (passes_checklist, checked_items, confidence_count)
        """
        checks = {
            'price_vs_sma': False,
            'vwap': False,
            'rsi_structure': False,
            'stochastic': False,
            'adx_trend': False,
            'volume': False,
            'bollinger': False
        }
        
        try:
            # 1. Price vs 50 SMA - Price above 50 SMA
            if 'SMA_50' in data.columns:
                checks['price_vs_sma'] = latest['close'] > latest['SMA_50']
            
            # 2. VWAP - Trading above intraday VWAP
            try:
                vwap = self.calculate_vwap(data)
                if not pd.isna(vwap.iloc[-1]):
                    checks['vwap'] = latest['close'] > vwap.iloc[-1]
            except:
                pass
            
            # 3. RSI Structure - Above 50 or bullish divergence
            if 'RSI_14' in data.columns:
                rsi = latest['RSI_14']
                if rsi > 50:
                    checks['rsi_structure'] = True
                else:
                    # Check for bullish divergence (RSI rising while price stable/down)
                    if len(data) >= 5:
                        rsi_5bars_ago = data['RSI_14'].iloc[-5]
                        if rsi > rsi_5bars_ago and rsi > 40:
                            checks['rsi_structure'] = True
            
            # 4. Stochastic - Turning up from oversold OR strong above 80
            if 'Stoch_K' in data.columns:
                stoch = latest['Stoch_K']
                if not pd.isna(stoch):
                    if stoch > 80:
                        checks['stochastic'] = True
                    elif stoch < 20:
                        # Check if turning up from oversold
                        if len(data) >= 3:
                            prev_stoch = data['Stoch_K'].iloc[-2]
                            if stoch > prev_stoch:
                                checks['stochastic'] = True
            
            # 5. ADX - Above 25 (trend exists)
            if 'ADX_14' in data.columns:
                checks['adx_trend'] = latest['ADX_14'] > 25
            
            # 6. Volume - Up-day volume > down-day volume or OBV rising
            if len(data) >= 2:
                current_up = latest['close'] > data['close'].iloc[-2]
                current_vol = latest['volume']
                prev_vol = data['volume'].iloc[-2]
                
                if current_up and current_vol > prev_vol:
                    checks['volume'] = True
                else:
                    # Check OBV trend
                    try:
                        obv = self.calculate_obv(data['close'], data['volume'])
                        if len(obv) >= 2 and obv.iloc[-1] > obv.iloc[-5]:
                            checks['volume'] = True
                    except:
                        pass
            
            # 7. Bollinger - Squeeze breakout or band walk
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                bb_width = latest['BB_Upper'] - latest['BB_Lower']
                bb_sma_width = data['BB_Upper'] - data['BB_Lower']
                
                # Squeeze breakout: current BB width < avg width (breakout imminent)
                if bb_width < bb_sma_width.mean() * 1.2:
                    checks['bollinger'] = True
                
                # Band walk: price near upper band
                elif (latest['close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']) > 0.7:
                    checks['bollinger'] = True
        
        except Exception as e:
            pass
        
        # Count passing checks
        confidence = sum(1 for v in checks.values() if v)
        passes = confidence >= 4
        
        return passes, checks, confidence
    
    def calculate_vwap(self, data):
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap = (typical_price * data['volume']).rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
            return vwap
        except:
            return pd.Series([np.nan] * len(data))
    
    def calculate_obv(self, close, volume):
        """Calculate On-Balance Volume"""
        try:
            if not isinstance(close, pd.Series):
                close = pd.Series(close)
            if not isinstance(volume, pd.Series):
                volume = pd.Series(volume)
            
            if TA_LIB_AVAILABLE:
                return OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            else:
                obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
                return obv
        except:
            if not isinstance(close, pd.Series):
                close = pd.Series(close)
            if not isinstance(volume, pd.Series):
                volume = pd.Series(volume)
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
            return obv


# ==================== EDGE SCORE CALCULATION ====================
    
    def calculate_edge_score(self, checklist_count, composite_score, sharpe_ratio, relative_strength, max_dd, return_1m, return_3m, return_6m):
        """
        Edge Score calculation with parameters directly
        """
        # --- Pillar 1: Checklist Quality (weight: 25%) ---
        checklist_score = (checklist_count / 7) * 100
    
        # --- Pillar 2: Composite Score (weight: 30%) ---
        composite_score_val = composite_score
    
        # --- Pillar 3: Risk-Adjusted Performance (weight: 45%) ---
        sharpe_normalized = min(sharpe_ratio / 4.0, 1.0) * 100
        rel_str_normalized = min(max(relative_strength / 100, 0), 1.0) * 100
        dd_score = max(0, 100 + (max_dd * 2))
        
        momentum_score = 0
        if return_1m > 0:
            momentum_score += 33
        if return_3m > 0:
            momentum_score += 33
        if return_6m > 0:
            momentum_score += 34
        
        if return_3m > 0 and return_1m > (return_3m / 3):
            momentum_score = min(momentum_score + 15, 100)
        
        risk_adjusted_score = (
            sharpe_normalized * 0.35
            + rel_str_normalized * 0.25
            + dd_score * 0.20
            + momentum_score * 0.20
        )
        
        edge_score = (
            checklist_score * 0.25
            + composite_score_val * 0.30
            + risk_adjusted_score * 0.45
        )
        
        return round(edge_score, 2)


# ==================== EDGE SCORE CLASSIFICATION ====================

def classify_edge(edge_score, volatility=None):
    """
    Translate edge score into actionable signal with recommendation.
    Returns dict with classification and action for each tier.
    
    With volatility cap: High volatility stocks (>50%) cannot be HIGHEST CONVICTION
    """
    # Extract volatility from string if needed (e.g., "45.3%")
    if volatility is not None and isinstance(volatility, str):
        try:
            volatility = float(volatility.replace('%', ''))
        except:
            volatility = 0
    
    # High volatility cap: no HIGHEST CONVICTION for vol > 50%
    if volatility and volatility > 50:
        if edge_score >= 75:
            return {
                'Classification': "STRONG EDGE",
                'Action': "Standard position size"
            }
        elif edge_score >= 65:
            return {
                'Classification': "MODERATE EDGE",
                'Action': "Smaller position, tighter stop"
            }
        elif edge_score >= 55:
            return {
                'Classification': "WEAK EDGE",
                'Action': "Paper trade or skip"
            }
        else:
            return {
                'Classification': "NO EDGE",
                'Action': "Do not trade"
            }
    
    # Normal classification (volatility <= 50% or not provided)
    if edge_score >= 85:
        return {
            'Classification': "HIGHEST CONVICTION",
            'Action': "Size up, trade aggressively"
        }
    elif edge_score >= 75:
        return {
            'Classification': "STRONG EDGE",
            'Action': "Standard position size"
        }
    elif edge_score >= 65:
        return {
            'Classification': "MODERATE EDGE",
            'Action': "Smaller position, tighter stop"
        }
    elif edge_score >= 55:
        return {
            'Classification': "WEAK EDGE",
            'Action': "Paper trade or skip"
        }
    else:
        return {
            'Classification': "NO EDGE",
            'Action': "Do not trade"
        }


def passes_gate(stock):
    """
    Stricter gate conditions to filter out false signals.
    Stock must pass ALL gates to proceed.
    Includes logging for debugging.
    """
    # Validate and clean data first
    stock = validate_and_clean(stock)
    
    reasons = []
    
    # Extract values safely
    checks_passed = stock.get('ChecksPassed', False)
    score = stock.get('Score', 0)
    signal = stock.get('Signal', 'HOLD')
    rsi = float(stock.get('RSI', '50').replace('%', '')) if isinstance(stock.get('RSI'), str) else stock.get('RSI', 50)
    adx = float(stock.get('ADX', '15').replace('%', '')) if isinstance(stock.get('ADX'), str) else stock.get('ADX', 15)
    sharpe = float(stock.get('Sharpe', '0').replace('%', '')) if isinstance(stock.get('Sharpe'), str) else stock.get('Sharpe', 0)
    rel_str = float(stock.get('Rel_Str', '0').replace('%', '+').replace('%', '')) if isinstance(stock.get('Rel_Str'), str) else stock.get('Rel_Str', 0)
    max_dd = float(stock.get('MaxDD', '0').replace('%', '')) if isinstance(stock.get('MaxDD'), str) else stock.get('MaxDD', 0)
    
    # Gate 1: Checklist >= 5/7
    if not checks_passed:
        reasons.append("Checklist <4/7")
    
    # Gate 2: Score >= 65
    if score < 65:
        reasons.append(f"Score {score} < 65")
    
    # Gate 3: Signal must be BUY or STRONG BUY
    if signal not in ['STRONG BUY', 'BUY']:
        reasons.append(f"Signal '{signal}' not allowed")
    
    # Gate 4: RSI between 40-75 (not overbought/oversold)
    if not (40 <= rsi <= 75):
        reasons.append(f"RSI {rsi:.1f} outside [40-75]")
    
    # Gate 5: ADX >= 20 (trend exists)
    if adx < 20:
        reasons.append(f"ADX {adx:.1f} < 20")
    
    # Gate 6: Sharpe >= 0.5
    if sharpe < 0.5:
        reasons.append(f"Sharpe {sharpe:.2f} < 0.5")
    
    # Gate 7: Rel_Str >= 0 (outperforming benchmark)
    if rel_str < 0:
        reasons.append(f"Rel_Str {rel_str:.1f}% < 0%")
    
    # Gate 8: MaxDD >= -30% (acceptable drawdown)
    if max_dd < -30:
        reasons.append(f"MaxDD {max_dd:.1f}% < -30%")
    
    if reasons:
        print(f"  ⛔ BLOCKED {stock.get('Ticker', 'UNKNOWN')}: {'; '.join(reasons)}")
        return False
    
    return True


def validate_and_clean(stock):
    """Clean and validate stock data"""
    if not isinstance(stock, dict):
        return {}
    return {k: v for k, v in stock.items() if v is not None}


def apply_modifiers(row):
    """
    Apply bonuses/penalties to edge score based on secondary factors.
    
    Bonuses:
    - Volume confirmation: +2-5 points
    - Oversold bounce setup: +3-5 points
    - Defensive sector: +2-3 points
    - Value trade (PE < 25): +2-3 points
    - Positive EPS: +1-2 points
    
    Penalties:
    - Overbought (RSI > 70): -5 points
    - Low volume: -2-3 points
    - Negative EPS: -5 points
    - Weak trend (ADX < 25): -2-3 points
    """
    modifier = 0
    
    # Volume bonuses
    vol_ratio = row.get('volume_ratio', 1.0)
    if vol_ratio >= 1.5:
        modifier += 5
    elif vol_ratio >= 1.2:
        modifier += 3
    elif vol_ratio < 0.8:
        modifier -= 3
    
    # RSI-based modifiers
    rsi = row.get('rsi', 50)
    if rsi < 35:  # Oversold bounce
        modifier += 4
    elif rsi > 70:  # Overbought
        modifier -= 5
    
    # ADX trend strength
    adx = row.get('adx', 25)
    if adx >= 30:
        modifier += 2
    elif adx < 20:
        modifier -= 3
    
    # Earnings quality
    eps = row.get('eps', 0)
    pe = row.get('pe_ratio', 25)
    if eps > 0:
        modifier += 2
        if pe < 20:
            modifier += 3  # Value + positive earnings
    else:
        modifier -= 5
    
    # Momentum quality
    return_6m = row.get('return_6m', 0)
    return_3m = row.get('return_3m', 0)
    if return_6m > 0 and return_3m > 0:
        modifier += 2  # Sustained momentum
    
    return round(modifier, 1)


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # NASDAQ-100 Stock List (100 stocks)
    nasdaq_100_stocks = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'GOOGL', 'GOOG', 'META', 'AVGO', 'CDNS',
        'CHTR', 'CMCS', 'CSCO', 'COST', 'CRWD', 'DDOG', 'DXCM', 'FANG', 'FAST', 'FTNT',
        'GILD', 'GS', 'HOLX', 'HSAI', 'IDXX', 'ILMN', 'INTC', 'INTU', 'ISRG', 'JD',
        'KDP', 'KEYS', 'KLAC', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDB', 'MDLZ', 'MELI',
        'MNST', 'MRNA', 'MSTR', 'MTCH', 'MU', 'NBLES', 'NFLX', 'NTES', 'NVAX', 'NXPI',
        'ODFL', 'ON', 'ORLY', 'PCAR', 'PDD', 'PAYC', 'PAYX', 'PCRX', 'PEGEN', 'PSTG',
        'PYPL', 'QCOM', 'REGN', 'ROST', 'RULE', 'SBUX', 'SGEN', 'SETR', 'SHAK', 'SIRI',
        'SNPS', 'STZ', 'SWKS', 'TEAM', 'TCOM', 'TECH', 'TMDX', 'TME', 'TMUS', 'TRNSL',
        'TRIP', 'TSM', 'TTD', 'TTWO', 'TWKS', 'UPLD', 'VRSN', 'VRSK', 'VRTX', 'WDAY',
        'WERN', 'WFG', 'WKHS', 'XMVU', 'XLNX', 'YY', 'ZK', 'ZM', 'ZS', 'ZUMZ'
    ]
    
    # Remove duplicates and invalid tickers
    nasdaq_100_stocks = list(set([s.upper() for s in nasdaq_100_stocks if len(s) <= 4]))
    nasdaq_100_stocks = sorted(nasdaq_100_stocks)
    
    # Settings
    period = "6mo"       # 6 months of data
    interval = "1d"      # Daily candles
    benchmark = "QQQ"    # NASDAQ-100 ETF as benchmark
    
    # Create screener and run scan
    print("=" * 200)
    print("INITIALIZING NASDAQ-100 STOCK SCREENER...")
    print(f"Total stocks to scan: {len(nasdaq_100_stocks)}")
    print("=" * 200)
    
    screener = NasdaqStockScreener(nasdaq_100_stocks, period=period, interval=interval, benchmark=benchmark)
    screener.scan()
    screener.display_results()
    screener.save_results()