# NSE Stock Scanner - Kite Connect API

A Python-based stock scanner that fetches NSE stock data from Kite Connect API and identifies tradable stocks based on technical indicators.

## Features

- Fetches real-time and historical data from Kite Connect API
- Calculates multiple technical indicators:
  - RSI (Relative Strength Index)
  - Moving Averages (SMA, EMA)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - ATR (Average True Range)
  - Volume indicators
- Filters stocks based on predefined trading conditions
- Generates a scored list of tradable stocks
- Exports results to CSV

## Prerequisites

1. **Kite Connect Account**: You need a developer account with Zerodha
   - Sign up at: https://developers.kite.trade/
   - Create an app to get your API credentials

2. **Python 3.7+** installed on your system

## Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Setup

### Step 1: Get API Credentials

1. Go to https://developers.kite.trade/
2. Login with your Zerodha credentials
3. Create a new app
4. Note down your **API Key** and **API Secret**

### Step 2: Generate Access Token

The Kite Connect API requires authentication. You have two options:

#### Option A: Manual Login (Recommended for testing)
```python
from kiteconnect import KiteConnect

api_key = "your_api_key"
api_secret = "your_api_secret"

kite = KiteConnect(api_key=api_key)

# Generate login URL
print(kite.login_url())
# Visit this URL in browser, login, and you'll get a request token in the URL

# Use the request token to generate access token
data = kite.generate_session("request_token_from_url", api_secret=api_secret)
print(data["access_token"])  # Save this token
```

#### Option B: Automated Login (For production)
You'll need to implement automated browser login using Selenium. See Kite Connect documentation.

### Step 3: Update Configuration

Edit `kite_stock_scanner.py` and update:
```python
API_KEY = "your_api_key_here"
ACCESS_TOKEN = "your_access_token_here"
```

## Usage

### Basic Usage

Run the scanner:
```bash
python kite_stock_scanner.py
```

### Customizing Stock Selection

By default, the script scans the first 50 NSE stocks. To customize:

```python
# Scan specific stocks
stocks_to_scan = [s for s in all_stocks if s['tradingsymbol'] in 
                 ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']]

# Scan all NSE stocks (warning: takes longer)
stocks_to_scan = all_stocks

# Scan Nifty 50 stocks (you'll need to maintain a list)
nifty50_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', ...]  # Add all 50
stocks_to_scan = [s for s in all_stocks if s['tradingsymbol'] in nifty50_symbols]
```

### Customizing Filter Conditions

Modify the `filter_tradable_stocks` method to adjust trading criteria:

```python
conditions = (
    (df['rsi_14'] >= 30) & (df['rsi_14'] <= 70) &  # RSI range
    (df['close'] > df['sma_20']) &  # Uptrend
    (df['macd_hist'] > 0) &  # Bullish momentum
    (df['volume_ratio'] > 1.2) &  # Volume above average
    (df['close'] > df['bb_lower']) & (df['close'] < df['bb_upper'])  # BB position
)
```

## Technical Indicators Explained

### 1. **RSI (Relative Strength Index)**
- Range: 0-100
- < 30: Oversold (potential buy)
- > 70: Overbought (potential sell)
- 40-60: Neutral zone

### 2. **Moving Averages**
- **SMA 20**: 20-day simple moving average
- **SMA 50**: 50-day simple moving average
- Price > SMA: Uptrend
- Price < SMA: Downtrend

### 3. **MACD**
- MACD Line crossing above Signal Line: Bullish
- MACD Line crossing below Signal Line: Bearish
- Positive Histogram: Bullish momentum

### 4. **Bollinger Bands**
- Upper Band: Resistance level
- Lower Band: Support level
- Price touching upper band: Overbought
- Price touching lower band: Oversold

### 5. **Volume Ratio**
- Ratio of current volume to 20-day average
- > 1: Above average volume
- > 1.5: Significantly high volume

## Default Filter Criteria

The scanner identifies tradable stocks using:

1. **RSI between 30-70**: Not in extreme zones
2. **Price above SMA 20**: Uptrend confirmation
3. **Positive MACD Histogram**: Bullish momentum
4. **Volume > 1.2x average**: Good liquidity
5. **Price within Bollinger Bands**: Not at extremes

Stocks are then scored and ranked based on:
- Volume ratio (30%)
- RSI proximity to 50 (30%)
- MACD histogram strength (20%)
- Price change percentage (20%)

## Output

The script generates:
1. Console output showing filtered stocks
2. CSV file: `tradable_stocks_YYYYMMDD.csv`

### Sample Output:
```
symbol      close  change_pct  rsi_14  macd_hist  volume_ratio  score
RELIANCE   2456.30      1.25   58.42      0.45          1.85   45.32
TCS        3678.90      0.87   52.18      0.32          1.62   42.15
INFY       1543.20      1.10   55.67      0.28          1.54   40.88
```

## Important Notes

1. **Rate Limits**: Kite Connect has API rate limits. The script includes delays to respect these limits.

2. **Market Hours**: Best to run during market hours (9:15 AM - 3:30 PM IST) for real-time data.

3. **Access Token Validity**: Access tokens expire daily. You'll need to regenerate them.

4. **Paper Trading First**: Test your strategy with paper trading before using real money.

5. **Not Financial Advice**: This is a technical analysis tool. Always do your own research.

## Troubleshooting

### "Invalid Access Token"
- Generate a new access token
- Ensure you're using the correct API key

### "No data found"
- Check if markets are open
- Verify instrument tokens are correct
- Increase lookback_days parameter

### "Rate limit exceeded"
- Add delays between API calls
- Reduce number of stocks being scanned

## Extending the Scanner

### Add More Indicators

```python
def calculate_stochastic(self, df, period=14):
    """Calculate Stochastic Oscillator"""
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    return k
```

### Add Intraday Scanning

Change interval from 'day' to '5minute', '15minute', etc.:
```python
df = self.get_historical_data(
    stock['instrument_token'],
    from_date,
    to_date,
    interval='5minute'  # For 5-minute candles
)
```

## License

This project is for educational purposes. Use at your own risk.

## Disclaimer

This tool is for educational and informational purposes only. It does not constitute financial advice. Trading in stocks involves risk, and you should only trade with money you can afford to lose. Always consult with a qualified financial advisor before making investment decisions.
