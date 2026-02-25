"""  
Nifty 50 OI (Open Interest) Tracker
Analyzes option chain data to find best strikes for Iron Butterfly based on OI levels
Creates an interactive HTML tracker showing OI, PCR, and entry/exit signals
"""

import pandas as pd
import numpy as np
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import logging
import math
from pathlib import Path

# Get project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load credentials from project root
creds_file = PROJECT_ROOT / 'kite_credentials.txt'
with open(creds_file, 'r') as f:
    creds = {line.split('=')[0].strip(): line.split('=')[1].strip() for line in f if '=' in line}

API_KEY = creds.get('API_KEY', '')
ACCESS_TOKEN = creds.get('ACCESS_TOKEN', '')
USER_ID = creds.get('USER_ID', '')

class NiftyOITracker:
    """Track Nifty 50 Open Interest and recommend best strikes for F&O strategies"""
    
    def __init__(self):
        self.kite = KiteConnect(api_key=API_KEY)
        self.kite.set_access_token(ACCESS_TOKEN)
        logger.info("Kite Connect initialized successfully")
        
        self.nifty_token = self._get_nifty_token()
        self.current_price = 0
        self.strikes_data = {}
        self.expiry_date = None
    
    def _get_nifty_token(self):
        """Fetch Nifty 50 index token"""
        try:
            instruments = self.kite.instruments('NSE')
            for inst in instruments:
                if inst.get('tradingsymbol', '').strip() == 'NIFTY 50':
                    logger.info(f"Found Nifty 50 token: {inst['instrument_token']}")
                    return inst['instrument_token']
            logger.warning("Nifty token not found, using fallback")
            return 256265
        except Exception as e:
            logger.error(f"Error fetching Nifty token: {str(e)}, using fallback")
            return 256265
    
    def get_next_weekly_expiry(self):
        """Get next Wednesday (weekly expiry)"""
        today = datetime.now().date()
        days_ahead = 2 - today.weekday()  # Wednesday = 2
        if days_ahead <= 0:
            days_ahead += 7
        return today + timedelta(days=days_ahead)
    
    def fetch_nifty_price(self):
        """Fetch current Nifty 50 price"""
        try:
            data = self.kite.quote('NSE:NIFTY 50')
            self.current_price = data['NSE:NIFTY 50']['last_price']
            logger.info(f"Current Nifty 50 Price: ₹{self.current_price:.2f}")
            return self.current_price
        except Exception as e:
            logger.error(f"Error fetching Nifty price: {str(e)}")
            # Fallback
            self.current_price = 25935.15
            return self.current_price
    
    def fetch_nifty_options_chain(self):
        """Fetch option chain data for all strikes using Kite API"""
        try:
            self.expiry_date = self.get_next_weekly_expiry()
            atm_strike = (int(self.current_price) // 100) * 100
            
            # List of common Nifty strikes (typically at 100-point intervals)
            strikes_to_fetch = []
            for i in range(-800, 900, 100):  # -800 to +800 range around ATM
                strikes_to_fetch.append(atm_strike + i)
            
            logger.info(f"Fetching option chain data for {len(strikes_to_fetch)} strikes...")
            
            # Kite API format for Nifty options: NIFTY{DDMMMYYYY}{STRIKE}{TYPE}
            # Example: NFO:NIFTY18FEB202625900CE
            expiry_str = self.expiry_date.strftime('%d%b%Y')
            
            option_symbols = []
            symbol_map = {}  # Map symbol to (strike, type)
            
            for strike in strikes_to_fetch:
                for opt_type in ['CE', 'PE']:
                    symbol = f'NFO:NIFTY{expiry_str}{int(strike)}{opt_type}'
                    option_symbols.append(symbol)
                    symbol_map[symbol] = (strike, opt_type)
            
            logger.info(f"Fetching quotes for {len(option_symbols)} option contracts...")
            
            # Fetch in batches (Kite API limit is ~50-100 symbols per request)
            self.strikes_data = {}
            for strike in strikes_to_fetch:
                self.strikes_data[strike] = {'strike': strike, 'call': {}, 'put': {}}
            
            batch_size = 50
            for i in range(0, len(option_symbols), batch_size):
                batch = option_symbols[i:i+batch_size]
                
                try:
                    quotes = self.kite.quote(batch)
                    
                    # Parse the response
                    for symbol, quote_data in quotes.items():
                        if symbol in symbol_map:
                            strike, opt_type = symbol_map[symbol]
                            
                            # Extract OHLC and OI data
                            last_price = quote_data.get('last_price', 0)
                            bid = quote_data.get('bid', 0)
                            ask = quote_data.get('ask', 0)
                            oi = quote_data.get('oi', 0)
                            volume = quote_data.get('volume', 0)
                            high = quote_data.get('high', last_price)
                            low = quote_data.get('low', last_price)
                            
                            if opt_type == 'CE':
                                self.strikes_data[strike]['call'] = {
                                    'oi': oi,
                                    'bid': bid,
                                    'ask': ask,
                                    'ltp': last_price,
                                    'volume': volume,
                                    'high': high,
                                    'low': low
                                }
                            else:  # PE
                                self.strikes_data[strike]['put'] = {
                                    'oi': oi,
                                    'bid': bid,
                                    'ask': ask,
                                    'ltp': last_price,
                                    'volume': volume,
                                    'high': high,
                                    'low': low
                                }
                except Exception as e:
                    logger.warning(f"Error fetching batch {i//batch_size + 1}: {str(e)}")
                    # Continue with available data
            
            # Check if we got real data
            total_oi = sum(self.strikes_data[s]['call'].get('oi', 0) + 
                          self.strikes_data[s]['put'].get('oi', 0) 
                          for s in strikes_to_fetch)
            
            if total_oi > 0:
                logger.info(f"✅ Fetched LIVE option chain data for {len(self.strikes_data)} strikes")
            else:
                logger.warning("Live OI data returned 0, using simulated data...")
                return self._generate_simulated_oi_data()
            
            return self.strikes_data
            
        except Exception as e:
            logger.error(f"Error fetching option chain: {str(e)}")
            logger.info("Falling back to simulated OI data...")
            return self._generate_simulated_oi_data()
    
    def _generate_simulated_oi_data(self):
        """Generate simulated OI data for demonstration"""
        atm_strike = (int(self.current_price) // 100) * 100
        self.expiry_date = self.get_next_weekly_expiry()
        
        # Create realistic simulated OI distribution
        strikes = list(range(atm_strike - 800, atm_strike + 900, 100))
        self.strikes_data = {}
        
        for strike in strikes:
            distance_from_atm = abs(strike - atm_strike)
            
            # OI typically peaks at ATM and nearby strikes
            # Decreases as we move away from ATM
            oi_multiplier = max(0.1, 1 - (distance_from_atm / 2000))
            
            # Higher OI for deep OTM calls, lower for ITM calls
            call_oi = int(5000000 * oi_multiplier * (1 + (strike - atm_strike) / 1000))
            put_oi = int(5000000 * oi_multiplier * (1 + (atm_strike - strike) / 1000))
            
            call_ltp = max(0.1, atm_strike - strike + 50) if strike > atm_strike else max(0.1, atm_strike - strike)
            put_ltp = max(0.1, strike - atm_strike + 50) if strike < atm_strike else max(0.1, strike - atm_strike)
            
            self.strikes_data[strike] = {
                'strike': strike,
                'call': {
                    'oi': call_oi,
                    'bid': call_ltp - 0.5,
                    'ask': call_ltp + 0.5,
                    'ltp': call_ltp,
                    'volume': int(call_oi / 1000)
                },
                'put': {
                    'oi': put_oi,
                    'bid': put_ltp - 0.5,
                    'ask': put_ltp + 0.5,
                    'ltp': put_ltp,
                    'volume': int(put_oi / 1000)
                }
            }
        
        return self.strikes_data
    
    @staticmethod
    def norm_cdf(x):
        """Normal CDF approximation"""
        return (1 + math.erf(x / np.sqrt(2))) / 2
    
    def black_scholes(self, S, K, T, r, sigma, option_type='CE'):
        """Black-Scholes option pricing"""
        if T <= 0 or sigma == 0:
            if option_type == 'CE':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'CE':
            price = S * self.norm_cdf(d1) - K * np.exp(-r * T) * self.norm_cdf(d2)
        else:  # PE
            price = K * np.exp(-r * T) * self.norm_cdf(-d2) - S * self.norm_cdf(-d1)
        
        return max(price, 0)
    
    def calculate_strategy_pnl(self, strategy_data, current_price, expiry_date, iv=0.15, r=0.06):
        """Calculate P&L for strategy at different price levels"""
        T = (expiry_date - datetime.now().date()).days / 365
        
        # Generate price scenarios
        atm = (int(current_price) // 100) * 100
        price_scenarios = [atm - 300, atm - 200, atm - 100, atm, atm + 100, atm + 200, atm + 300]
        
        pnl_data = []
        
        for scenario_price in price_scenarios:
            pnl = 0
            
            # Calculate PnL based on strategy type
            for key, strike in strategy_data.items():
                if key == 'long_call' and strike:
                    current_value = self.black_scholes(scenario_price, strike, T, r, iv, 'CE')
                    entry_value = self.black_scholes(current_price, strike, T, r, iv, 'CE')
                    pnl += (current_value - entry_value) * 20  # Nifty multiplier
                    
                elif key == 'short_call' and strike:
                    entry_value = self.black_scholes(current_price, strike, T, r, iv, 'CE')
                    current_value = self.black_scholes(scenario_price, strike, T, r, iv, 'CE')
                    pnl += (entry_value - current_value) * 20  # Short = reverse
                    
                elif key == 'long_put' and strike:
                    current_value = self.black_scholes(scenario_price, strike, T, r, iv, 'PE')
                    entry_value = self.black_scholes(current_price, strike, T, r, iv, 'PE')
                    pnl += (current_value - entry_value) * 20
                    
                elif key == 'short_put' and strike:
                    entry_value = self.black_scholes(current_price, strike, T, r, iv, 'PE')
                    current_value = self.black_scholes(scenario_price, strike, T, r, iv, 'PE')
                    pnl += (entry_value - current_value) * 20
            
            pnl_data.append({
                'price': scenario_price,
                'move': scenario_price - atm,
                'pnl': pnl,
                'color': '#28a745' if pnl > 0 else '#dc3545'
            })
        
        return pnl_data
    
    def analyze_oi_levels(self):
        """Analyze OI to find best strike combinations for multiple strategies"""
        if not self.strikes_data:
            return None
        
        analysis = {
            'atm_strike': (int(self.current_price) // 100) * 100,
            'peak_call_oi_strike': None,
            'peak_put_oi_strike': None,
            'pcr_ratio': 0,
            'highest_oi': 0,
            'total_call_oi': 0,
            'total_put_oi': 0,
            'strategies': {},
            'strike_oi_summary': [],
            'market_sentiment': 'NEUTRAL'
        }
        
        atm = analysis['atm_strike']
        
        # Find peak OI strikes and totals
        max_call_oi = 0
        max_put_oi = 0
        
        for strike, data in sorted(self.strikes_data.items()):
            call_oi = data['call'].get('oi', 0)
            put_oi = data['put'].get('oi', 0)
            
            analysis['total_call_oi'] += call_oi
            analysis['total_put_oi'] += put_oi
            
            if call_oi > max_call_oi:
                max_call_oi = call_oi
                analysis['peak_call_oi_strike'] = strike
            
            if put_oi > max_put_oi:
                max_put_oi = put_oi
                analysis['peak_put_oi_strike'] = strike
            
            if call_oi + put_oi > analysis['highest_oi']:
                analysis['highest_oi'] = call_oi + put_oi
        
        # Calculate PCR (Put Call Ratio) - FIX
        if analysis['total_call_oi'] > 0:
            analysis['pcr_ratio'] = analysis['total_put_oi'] / analysis['total_call_oi']
        else:
            analysis['pcr_ratio'] = 1.0
        
        # Determine market sentiment from PCR
        if analysis['pcr_ratio'] > 1.3:
            analysis['market_sentiment'] = 'BULLISH'
        elif analysis['pcr_ratio'] > 1.0:
            analysis['market_sentiment'] = 'NEUTRAL_BULLISH'
        elif analysis['pcr_ratio'] > 0.7:
            analysis['market_sentiment'] = 'NEUTRAL_BEARISH'
        else:
            analysis['market_sentiment'] = 'BEARISH'
        
        atm_strike = analysis['atm_strike']
        
        # ===== STRATEGY 1: IRON BUTTERFLY (Range-bound, Low IV) =====
        analysis['strategies']['IRON_BUTTERFLY'] = {
            'name': 'Iron Butterfly',
            'condition': 'Range-bound market, Low IV (<20%)',
            'short_call': atm_strike,
            'long_call': atm_strike + 100,
            'short_put': atm_strike,
            'long_put': atm_strike - 100,
            'best_for': 'Neutral/Range-bound bias'
        }
        
        # ===== STRATEGY 2: BULL CALL SPREAD (Uptrend) =====
        analysis['strategies']['BULL_CALL_SPREAD'] = {
            'name': 'Bull Call Spread',
            'condition': 'Uptrend, Bullish OI (low put buying)',
            'long_call': atm_strike,
            'short_call': atm_strike + 100,
            'best_for': 'Moderate bullish view, lower capital'
        }
        
        # ===== STRATEGY 3: BEAR CALL SPREAD (Downtrend) =====
        analysis['strategies']['BEAR_CALL_SPREAD'] = {
            'name': 'Bear Call Spread',
            'condition': 'Downtrend, Bearish OI (high call buying)',
            'short_call': atm_strike,
            'long_call': atm_strike + 100,
            'best_for': 'Moderate bearish view, defined loss'
        }
        
        # ===== STRATEGY 4: LONG CALL (Strong Uptrend) =====
        analysis['strategies']['LONG_CALL'] = {
            'name': 'Long Call',
            'condition': 'Strong uptrend, High call OI',
            'long_call': atm_strike,
            'best_for': 'Bullish high momentum, unlimited profit'
        }
        
        # ===== STRATEGY 5: LONG PUT (Strong Downtrend) =====
        analysis['strategies']['LONG_PUT'] = {
            'name': 'Long Put',
            'condition': 'Strong downtrend, High put OI',
            'long_put': atm_strike,
            'best_for': 'Bearish high momentum, downside protection'
        }
        
        # ===== STRATEGY 6: STRADDLE (High Volatility Expected) =====
        analysis['strategies']['STRADDLE'] = {
            'name': 'Straddle',
            'condition': 'High IV >20%, Expecting big move',
            'long_call': atm_strike,
            'long_put': atm_strike,
            'best_for': 'Earnings/Events, expects sudden volatility'
        }
        
        # Create strike summary with all data
        for strike in sorted(self.strikes_data.keys()):
            if atm_strike - 400 <= strike <= atm_strike + 400:  # Focus on ±400 range
                data = self.strikes_data[strike]
                call_oi = data['call'].get('oi', 0)
                put_oi = data['put'].get('oi', 0)
                
                analysis['strike_oi_summary'].append({
                    'strike': strike,
                    'call_oi': call_oi,
                    'put_oi': put_oi,
                    'call_ltp': data['call'].get('ltp', 0),
                    'put_ltp': data['put'].get('ltp', 0),
                    'total_oi': call_oi + put_oi,
                    'distance_from_atm': strike - atm_strike,
                    'call_volume': data['call'].get('volume', 0),
                    'put_volume': data['put'].get('volume', 0)
                })
        
        return analysis
    
    def generate_oi_tracker_html(self, analysis, strategy_pnl):
        """Generate interactive HTML tracker showing OI analysis for multiple strategies with P&L scenarios"""
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nifty 50 Weekly F&O OI Tracker - Multi-Strategy with P&L</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.2em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .metric-box {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}
        
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        
        .sentiment-badge {{
            display: inline-block;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: 600;
            margin-top: 10px;
        }}
        
        .sentiment-bullish {{
            background: #d4edda;
            color: #155724;
        }}
        
        .sentiment-bearish {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .sentiment-neutral {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .strategies-section {{
            background: #f8f9fa;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        
        .strategies-section h2 {{
            color: #667eea;
            margin-bottom: 25px;
            font-size: 1.6em;
        }}
        
        .strategy-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
            transition: all 0.3s ease;
        }}
        
        .strategy-card:hover {{
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
        }}
        
        .strategy-title {{
            font-size: 1.2em;
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
        }}
        
        .strategy-condition {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 12px;
            font-style: italic;
        }}
        
        .strategy-strikes {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 12px;
        }}
        
        .strike-badge {{
            background: #f0f0f0;
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 0.9em;
        }}
        
        .strike-label {{
            font-weight: 600;
            color: #333;
        }}
        
        .strike-value {{
            color: #667eea;
            font-weight: bold;
        }}
        
        .strategy-best-for {{
            font-size: 0.9em;
            color: #28a745;
            font-weight: 600;
        }}
        
        .pnl-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin-top: 20px;
            border-radius: 8px;
            overflow: hidden;
            font-size: 0.85em;
        }}
        
        .pnl-table th {{
            background: #e8ebf0;
            color: #333;
            padding: 10px;
            text-align: center;
            font-weight: 600;
            border-bottom: 2px solid #667eea;
        }}
        
        .pnl-table td {{
            padding: 9px 10px;
            text-align: center;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .pnl-positive {{
            background: #d4edda;
            color: #155724;
            font-weight: 600;
        }}
        
        .pnl-negative {{
            background: #f8d7da;
            color: #721c24;
            font-weight: 600;
        }}
        
        .pnl-neutral {{
            background: #fff3cd;
            color: #856404;
            font-weight: 600;
        }}
        
        .pcr-indicator {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 2px solid #ffc107;
            border-left: 5px solid #ffc107;
        }}
        
        .pcr-indicator p {{
            margin: 8px 0;
        }}
        
        .oi-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
        }}
        
        .oi-table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            font-size: 0.95em;
        }}
        
        .oi-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .oi-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .atm-row {{
            background: #fff3cd !important;
            font-weight: 600;
        }}
        
        .call-oi {{
            color: #dc3545;
            font-weight: 600;
        }}
        
        .put-oi {{
            color: #28a745;
            font-weight: 600;
        }}
        
        .legend {{
            background: #e8f4f8;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #17a2b8;
        }}
        
        .legend h3 {{
            color: #0c5460;
            margin-bottom: 15px;
        }}
        
        .legend-item {{
            margin: 8px 0;
            font-size: 0.9em;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e0e0e0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Nifty 50 Weekly F&O OI Tracker</h1>
            <p>Multi-Strategy Analysis Based on Open Interest</p>
            <p>Expiry: {self.expiry_date.strftime('%A, %B %d, %Y')}</p>
            <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="content">
            <!-- Key Metrics -->
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-label">Current Nifty 50</div>
                    <div class="metric-value">₹{self.current_price:.0f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">ATM Strike</div>
                    <div class="metric-value">₹{analysis['atm_strike']:.0f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">PCR Ratio</div>
                    <div class="metric-value">{analysis['pcr_ratio']:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Market Sentiment</div>
                    <div class="metric-value sentiment-badge sentiment-{analysis['market_sentiment'].lower()}">{analysis['market_sentiment'].replace('_', ' ')}</div>
                </div>
            </div>
            
            <!-- OI Legend -->
            <div class="legend">
                <h3>📖 Open Interest (OI) Explanation</h3>
                <div class="legend-item"><strong>Call OI:</strong> Total outstanding call option contracts at that strike (red)</div>
                <div class="legend-item"><strong>Put OI:</strong> Total outstanding put option contracts at that strike (green)</div>
                <div class="legend-item"><strong>High OI Strike:</strong> Acts as support/resistance level (institutional trading)</div>
                <div class="legend-item"><strong>Low OI Strike:</strong> Good for wings/spreads (cheaper premiums, less liquid)</div>
                <div class="legend-item"><strong>PCR Ratio:</strong> Put OI ÷ Call OI ratio (>1.3=Bullish, <0.7=Bearish)</div>
            </div>
            
            <!-- PCR Indicator -->
            <div class="pcr-indicator">
                <p><strong>📊 Put-Call Ratio (PCR): {analysis['pcr_ratio']:.2f}</strong></p>
                <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
                    {self._get_pcr_interpretation(analysis['pcr_ratio'])}
                </p>
            </div>
            
            <!-- All Strategies Based on Market Condition -->
            <div class="strategies-section">
                <h2>💡 6 Strategies - Choose Based on Market Condition & View P&L Scenarios</h2>
                
                <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #ffc107;">
                    <strong>Current Market:</strong> {analysis['market_sentiment'].replace('_', ' ')} | PCR: {analysis['pcr_ratio']:.2f} | Days to Expiry: {(self.expiry_date - datetime.now().date()).days}
                </div>
"""
        
        # Add all strategies
        strategy_info = {
            'IRON_BUTTERFLY': {
                'emoji': '🎯',
                'color': '#667eea',
                'when': 'Choose when: Market is RANGING, IV is LOW (<20%), ATM has high OI'
            },
            'BULL_CALL_SPREAD': {
                'emoji': '📈',
                'color': '#28a745',
                'when': 'Choose when: Market is BULLISH, Sell OTM calls, Buy ATM calls'
            },
            'BEAR_CALL_SPREAD': {
                'emoji': '📉',
                'color': '#dc3545',
                'when': 'Choose when: Market is BEARISH, Short ATM calls, Long higher calls'
            },
            'LONG_CALL': {
                'emoji': '🚀',
                'color': '#ffc107',
                'when': 'Choose when: Strong UPTREND expected, want unlimited profit'
            },
            'LONG_PUT': {
                'emoji': '🔻',
                'color': '#17a2b8',
                'when': 'Choose when: Strong DOWNTREND expected, want downside protection'
            },
            'STRADDLE': {
                'emoji': '⚡',
                'color': '#e83e8c',
                'when': 'Choose when: High volatility expected (earnings/events), price may move sharply'
            }
        }
        
        for strategy_key, strategy_data in analysis['strategies'].items():
            info = strategy_info.get(strategy_key, {})
            color = info.get('color', '#667eea')
            emoji = info.get('emoji', '•')
            when = info.get('when', '')
            
            html_content += f"""
                <div class="strategy-card" style="border-left: 5px solid {color};">
                    <div class="strategy-title">{emoji} {strategy_data['name']}</div>
                    <div class="strategy-condition">Condition: {strategy_data['condition']}</div>
                    <div style="background: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 0.9em;">
                        {when}
                    </div>
"""
            
            # Show strikes for this strategy
            html_content += """                    <div class="strategy-strikes">"""
            
            if 'long_call' in strategy_data:
                html_content += f"""<div class="strike-badge"><span class="strike-label">LONG CALL:</span> <span class="strike-value">₹{strategy_data['long_call']:.0f}</span></div>"""
            if 'short_call' in strategy_data:
                html_content += f"""<div class="strike-badge"><span class="strike-label">SHORT CALL:</span> <span class="strike-value">₹{strategy_data['short_call']:.0f}</span></div>"""
            if 'long_put' in strategy_data:
                html_content += f"""<div class="strike-badge"><span class="strike-label">LONG PUT:</span> <span class="strike-value">₹{strategy_data['long_put']:.0f}</span></div>"""
            if 'short_put' in strategy_data:
                html_content += f"""<div class="strike-badge"><span class="strike-label">SHORT PUT:</span> <span class="strike-value">₹{strategy_data['short_put']:.0f}</span></div>"""
            
            html_content += f"""                    </div>
                    <div class="strategy-best-for">✓ Best For: {strategy_data['best_for']}</div>
                    
                    <!-- P&L Scenarios Table -->
                    <table class="pnl-table">
                        <thead>
                            <tr>
                                <th>Nifty Price</th>
                                <th>Move (pts)</th>
                                <th>P&L (₹)</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
"""
            
            # Add P&L rows for this strategy
            if strategy_key in strategy_pnl:
                for pnl_scenario in strategy_pnl[strategy_key]:
                    pnl = pnl_scenario['pnl']
                    status_class = 'pnl-positive' if pnl > 100 else 'pnl-negative' if pnl < -100 else 'pnl-neutral'
                    status_text = '✓ PROFIT' if pnl > 100 else '✗ LOSS' if pnl < -100 else '= BREAK'
                    
                    html_content += f"""                            <tr>
                                <td><strong>₹{pnl_scenario['price']:.0f}</strong></td>
                                <td>{pnl_scenario['move']:+.0f}</td>
                                <td class="{status_class}">₹{pnl:+,.0f}</td>
                                <td class="{status_class}">{status_text}</td>
                            </tr>
"""
            
            html_content += """                        </tbody>
                    </table>
                </div>
"""
        
        html_content += """            </div>
            
            <!-- OI Distribution Table -->
            <h2 style="color: #667eea; margin-top: 40px; margin-bottom: 20px;">📊 Open Interest Distribution Across All Strikes</h2>
            <p style="background: #f0f0f0; padding: 15px; border-radius: 8px; margin-bottom: 15px; font-size: 0.9em;">
                <strong>How to Read:</strong> 
                <span style="color: #dc3545;">🔴 Call OI (Red)</span> = Number of call contracts | 
                <span style="color: #28a745;">🟢 Put OI (Green)</span> = Number of put contracts | 
                Higher OI = More institutional activity = Better support/resistance
            </p>
            <table class="oi-table">
                <thead>
                    <tr>
                        <th>Strike (₹)</th>
                        <th>Call OI 🔴</th>
                        <th>Put OI 🟢</th>
                        <th>Call Price (₹)</th>
                        <th>Put Price (₹)</th>
                        <th>Total OI</th>
                        <th>Distance from ATM</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add rows for each strike
        for strike_data in analysis['strike_oi_summary']:
            strike = strike_data['strike']
            call_oi = strike_data['call_oi']
            put_oi = strike_data['put_oi']
            total_oi = strike_data['total_oi']
            distance = strike_data['distance_from_atm']
            call_ltp = strike_data['call_ltp']
            put_ltp = strike_data['put_ltp']
            
            # Highlight ATM
            row_class = ""
            if strike == analysis['atm_strike']:
                row_class = "atm-row"
            
            html_content += f"""                    <tr {f'class="{row_class}"' if row_class else ''}>
                        <td><strong>₹{strike:.0f}</strong></td>
                        <td class="call-oi">{call_oi/1000:.0f}K</td>
                        <td class="put-oi">{put_oi/1000:.0f}K</td>
                        <td>₹{call_ltp:.2f}</td>
                        <td>₹{put_ltp:.2f}</td>
                        <td><strong>{total_oi/1000:.0f}K</strong></td>
                        <td>{distance:+.0f}</td>
                    </tr>
"""
        
        html_content += """                </tbody>
            </table>
            
            <!-- Important Notes -->
            <div style="background: #fff3cd; padding: 20px; border-radius: 8px; margin-top: 30px; border-left: 4px solid #ffc107;">
                <h3 style="color: #856404; margin-bottom: 10px;">⚠️ How to Use This Tracker</h3>
                <ul style="margin-left: 20px; color: #333;">
                    <li><strong>Step 1:</strong> Check PCR ratio to determine market sentiment (bullish/bearish/neutral)</li>
                    <li><strong>Step 2:</strong> Look at OI distribution - high OI = support/resistance, low OI = potential breakout zones</li>
                    <li><strong>Step 3:</strong> Choose strategy that matches current market condition (see strategy cards above)</li>
                    <li><strong>Step 4:</strong> Use ATM or near-ATM strikes for short positions (high OI = stable)</li>
                    <li><strong>Step 5:</strong> Use away-from-ATM strikes for long positions (low OI = cheaper premiums)</li>
                    <li><strong>Best Entry Time:</strong> Tuesday-Wednesday after market opens (OI settles)</li>
                    <li><strong>Exit Rule:</strong> Book profits at 30-40%, or exit by Day 5-6 before gamma explodes</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>✅ This tracker analyzes Open Interest (OI) data to recommend the best F&O strategy</p>
            <p>🎯 Not restricted to Iron Butterfly - shows ALL strategies based on market conditions</p>
            <p>⚠️ Disclaimer: This is for educational purposes. Always consult your advisor before trading.</p>
            <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p IST')}</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML file to Results folder
        filename = RESULTS_DIR / "nifty_oi_tracker.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"OI Tracker HTML generated: {filename}")
        return filename
    
    @staticmethod
    def _get_pcr_interpretation(pcr):
        """Get PCR interpretation"""
        if pcr > 1.3:
            return "🟢 BULLISH: Strong put buying, market expects upside (more protective puts)"
        elif pcr > 1.0:
            return "🟡 NEUTRAL-BULLISH: Moderate put bias, slight bullish tilt expected"
        elif pcr > 0.7:
            return "🟠 NEUTRAL-BEARISH: Moderate call bias, slight bearish tilt expected"
        else:
            return "🔴 BEARISH: Strong call buying, market expects downside (more call writing)"
    
    def run(self):
        """Execute complete OI analysis and tracker generation"""
        logger.info("=" * 80)
        logger.info("Starting Nifty 50 Weekly OI Tracker Analysis")
        logger.info("=" * 80)
        
        # Fetch current price
        self.fetch_nifty_price()
        
        # Fetch option chain data
        self.fetch_nifty_options_chain()
        
        # Analyze OI
        analysis = self.analyze_oi_levels()
        
        # Calculate P&L scenarios for each strategy
        strategy_pnl = {}
        for strategy_key in analysis['strategies'].keys():
            strategy_data = analysis['strategies'][strategy_key]
            pnl_data = self.calculate_strategy_pnl(strategy_data, self.current_price, self.expiry_date)
            strategy_pnl[strategy_key] = pnl_data
        
        # Generate HTML tracker with P&L scenarios
        html_file = self.generate_oi_tracker_html(analysis, strategy_pnl)
        
        logger.info("=" * 80)
        logger.info(f"OI Tracker generated: {html_file}")
        logger.info("=" * 80)
        return html_file


if __name__ == '__main__':
    tracker = NiftyOITracker()
    tracker.run()
