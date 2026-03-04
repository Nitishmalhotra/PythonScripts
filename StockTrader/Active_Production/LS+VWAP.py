import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -----------------------------
# SETTINGS
# -----------------------------
LOOKBACK_SWEEP = 20
VWAP_LOOKBACK = 20
DAYS_HISTORY = 120

# Nifty 50 list (can update dynamically later)
nifty50 = [
   'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
                    'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BHARTIARTL', 'BPCL',
                    'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY',
                    'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HEROMOTOCO',
                    'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK', 'ITC',
                    'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI',
                    'NESTLEIND', 'NTPC', 'ONGC', 'POWERGRID', 'RELIANCE',
                    'SBILIFE', 'SBIN', 'SHREECEM', 'SUNPHARMA', 'TATASTEEL',
                    'TATACONSUM', 'TATAMOTORS', 'TCS', 'TECHM', 'TITAN',
                    'ULTRACEMCO', 'UPL', 'WIPRO', 'INFY','BEL'
]


def to_yfinance_symbol(symbol):
    if '.' in symbol:
        return symbol
    return f"{symbol}.NS"


def normalize_ohlcv_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_cols.issubset(set(df.columns)):
        return pd.DataFrame()

    return df

# -----------------------------
# FUNCTIONS
# -----------------------------

def calculate_vwap(df):
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['CumVol'] = df['Volume'].cumsum()
    df['CumTPV'] = (df['TP'] * df['Volume']).cumsum()
    df['VWAP'] = df['CumTPV'] / df['CumVol']
    return df


def detect_liquidity_sweep(df):
    df['Prev_High_Max'] = df['High'].rolling(LOOKBACK_SWEEP).max().shift(1)
    df['Prev_Low_Min'] = df['Low'].rolling(LOOKBACK_SWEEP).min().shift(1)

    df['Bearish_Sweep'] = np.where(
        (df['High'] > df['Prev_High_Max']) &
        (df['Close'] < df['Prev_High_Max']),
        True, False
    )

    df['Bullish_Sweep'] = np.where(
        (df['Low'] < df['Prev_Low_Min']) &
        (df['Close'] > df['Prev_Low_Min']),
        True, False
    )

    return df


def anchored_vwap_from_extreme(df):
    if len(df) < VWAP_LOOKBACK:
        return df

    # Anchor from most recent 20-day low
    recent_low_idx = df['Low'].rolling(VWAP_LOOKBACK).apply(lambda x: np.argmin(x), raw=True).iloc[-1]
    if pd.isna(recent_low_idx):
        return df

    anchor_index = int(len(df) - VWAP_LOOKBACK + recent_low_idx)

    anchor_df = df.iloc[anchor_index:].copy()
    anchor_df['TP'] = (anchor_df['High'] + anchor_df['Low'] + anchor_df['Close']) / 3
    anchor_df['CumVol'] = anchor_df['Volume'].cumsum()
    anchor_df['CumTPV'] = (anchor_df['TP'] * anchor_df['Volume']).cumsum()
    anchor_df['Anchored_VWAP'] = anchor_df['CumTPV'] / anchor_df['CumVol']

    df.loc[anchor_df.index, 'Anchored_VWAP'] = anchor_df['Anchored_VWAP']
    return df


# -----------------------------
# MAIN SCANNER
# -----------------------------

results = []

end_date = datetime.today()
start_date = end_date - timedelta(days=DAYS_HISTORY)

total_stocks = len(nifty50)

for index, stock in enumerate(nifty50, start=1):
    try:
        print(f"[{index}/{total_stocks}] Processing {stock}...", flush=True)
        yf_symbol = to_yfinance_symbol(stock)
        df = yf.download(yf_symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
        df = normalize_ohlcv_columns(df)

        if df.empty or len(df) < max(LOOKBACK_SWEEP, VWAP_LOOKBACK) + 1:
            print(f"[{index}/{total_stocks}] Skipped {stock} (insufficient data)", flush=True)
            continue

        df = calculate_vwap(df)
        df = detect_liquidity_sweep(df)
        df = anchored_vwap_from_extreme(df)

        latest = df.iloc[-1]

        signal = None

        if latest['Bullish_Sweep'] and latest['Close'] > latest['Anchored_VWAP']:
            signal = "Bullish Setup"

        elif latest['Bearish_Sweep'] and latest['Close'] < latest['Anchored_VWAP']:
            signal = "Bearish Setup"

        if signal:
            results.append({
                "Stock": stock,
                "Close": round(latest['Close'],2),
                "Anchored_VWAP": round(latest['Anchored_VWAP'],2),
                "Signal": signal
            })
            print(f"[{index}/{total_stocks}] Signal for {stock}: {signal}", flush=True)
        else:
            print(f"[{index}/{total_stocks}] No setup for {stock}", flush=True)

    except Exception as e:
        print(f"[{index}/{total_stocks}] Error in {stock}: {e}")

# -----------------------------
# OUTPUT
# -----------------------------

if results:
    print("\nLiquidity Sweep + Anchored VWAP Signals:\n")
    print(pd.DataFrame(results))
else:
    print("\nNo active setups found today.")