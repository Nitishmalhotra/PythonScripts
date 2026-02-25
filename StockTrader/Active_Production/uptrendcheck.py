"""
NSE Stock Screener
Identifies stocks that are:
1. Down 40%+ from their 52-week high
2. Showing an uptrend based on EMA10

Dependencies: pip install yfinance pandas numpy requests
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. Fetch NSE stock list
# ──────────────────────────────────────────────

def get_nse_symbols(max_stocks: int = None) -> list[str]:
    """
    Load NSE equity symbol list from local CSV or online source.
    Falls back to a small sample list if both fail.
    """
    # First try: local file path
    csv_path = r"C:\Users\ankit\OneDrive\Desktop\Personal\Nitish\stock_dashboard\StockTrader\Results\nse_stocks_list.csv"
    try:
        print(f"Loading NSE equity list from: {csv_path}")
        df = pd.read_csv(csv_path)
        # Column is usually 'SYMBOL' or 'symbol'
        symbol_col = [c for c in df.columns if "SYMBOL" in c.upper()][0]
        symbols = df[symbol_col].dropna().str.strip().tolist()
        print(f"  [OK] {len(symbols)} symbols loaded from CSV.")
    except FileNotFoundError:
        print(f"  [ERR] CSV file not found at {csv_path}. Trying online NSE source...")
        try:
            url = "https://www1.nseindia.com/content/equities/EQUITY_L.csv"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.nseindia.com/",
            }
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            symbol_col = [c for c in df.columns if "SYMBOL" in c.upper()][0]
            symbols = df[symbol_col].dropna().str.strip().tolist()
            print(f"  [OK] {len(symbols)} symbols loaded from NSE.")
        except Exception as e:
            print(f"  [ERR] Could not fetch NSE list online ({e}). Using fallback sample.")
            # Fallback: Common NSE symbols
            symbols = [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
                "ICICIBANK", "KOTAKBANK", "SBIN", "BAJFINANCE", "BHARTIARTL",
                "ITC", "ASIANPAINT", "AXISBANK", "LT", "DMART",
                "SUNPHARMA", "TITAN", "NESTLEIND", "WIPRO", "ULTRACEMCO",
                "POWERGRID", "NTPC", "TECHM", "HCLTECH", "MARUTI",
                "ONGC", "COALINDIA", "TATASTEEL", "JSWSTEEL", "BAJAJ-AUTO",
            ]
    except Exception as e:
        print(f"  [ERR] Error reading CSV ({e}). Using fallback sample.")
        # Fallback: Common NSE symbols
        symbols = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
            "ICICIBANK", "KOTAKBANK", "SBIN", "BAJFINANCE", "BHARTIARTL",
            "ITC", "ASIANPAINT", "AXISBANK", "LT", "DMART",
            "SUNPHARMA", "TITAN", "NESTLEIND", "WIPRO", "ULTRACEMCO",
            "POWERGRID", "NTPC", "TECHM", "HCLTECH", "MARUTI",
            "ONGC", "COALINDIA", "TATASTEEL", "JSWSTEEL", "BAJAJ-AUTO",
        ]

    if max_stocks:
        symbols = symbols[:max_stocks]

    # Convert to Yahoo Finance format (append .NS)
    return [f"{s}.NS" for s in symbols]


# ──────────────────────────────────────────────
# 2. Screening logic
# ──────────────────────────────────────────────

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def is_ema10_uptrend(close: pd.Series, lookback: int = 5) -> bool:
    """
    Returns True if EMA10 is showing an uptrend.
    An uptrend is defined as: EMA10 slope over the last
    `lookback` bars is > 0 (i.e., rising).
    """
    if len(close) < 15:
        return False  # not enough data → treat as no uptrend
    ema = calculate_ema(close, 10)
    recent_ema = ema.iloc[-lookback:]
    # Linear regression slope sign
    x = np.arange(len(recent_ema))
    slope = np.polyfit(x, recent_ema.values, 1)[0]
    return slope > 0  # True ⟹ IS uptrend


def screen_stock(ticker: str) -> dict | None:
    """
    Downloads ~1 year of daily data and applies the two filters.
    Returns a result dict or None if the stock passes (i.e., does not meet criteria).
    """
    try:
        df = yf.download(ticker, period="1y", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 20:
            return None

        close = df["Close"].squeeze()
        high_52w = df["High"].squeeze().max()
        current_price = close.iloc[-1]

        # Filter 1: >= 40% below 52-week high
        drop_pct = (high_52w - current_price) / high_52w * 100
        if drop_pct < 40:
            return None

        # Filter 2: EMA10 showing uptrend
        if not is_ema10_uptrend(close):
            return None  # skip stocks that are NOT in uptrend

        ema10_latest = calculate_ema(close, 10).iloc[-1]

        return {
            "Ticker": ticker.replace(".NS", ""),
            "Current Price (₹)": round(float(current_price), 2),
            "52W High (₹)": round(float(high_52w), 2),
            "Drop from 52W High (%)": round(float(drop_pct), 2),
            "EMA10": round(float(ema10_latest), 2),
            "Price vs EMA10": "Below" if current_price < ema10_latest else "Above",
        }
    except Exception:
        return None


# ──────────────────────────────────────────────
# 3. Main runner
# ──────────────────────────────────────────────

def run_screener(max_stocks: int = None, delay: float = 0.3):
    """
    Run the full screener.

    Parameters
    ----------
    max_stocks : int, optional
        Limit the number of stocks to scan (useful for quick testing).
        Pass None to scan all NSE stocks.
    delay : float
        Seconds to wait between API calls to avoid rate-limiting.
    """
    symbols = get_nse_symbols(max_stocks=max_stocks)
    total = len(symbols)
    print(f"\nScreening {total} stocks …\n")

    results = []
    for i, ticker in enumerate(symbols, 1):
        if i % 50 == 0 or i == 1:
            print(f"  Progress: {i}/{total} …")
        result = screen_stock(ticker)
        if result:
            results.append(result)
            print(f"  [MATCH] {result['Ticker']}  "
                  f"Drop: {result['Drop from 52W High (%)']:.1f}%  "
                  f"EMA10: {result['EMA10']:.2f}")
        time.sleep(delay)

    print(f"\n{'='*60}")
    print(f"Screening complete. {len(results)} stocks matched the criteria.")
    print(f"{'='*60}\n")

    if results:
        df_result = pd.DataFrame(results).sort_values(
            "Drop from 52W High (%)", ascending=False
        ).reset_index(drop=True)
        df_result.index += 1  # 1-based index

        # Rename columns to avoid Unicode rupee symbol
        df_result.columns = [col.replace(" (₹)", " (INR)") for col in df_result.columns]
        
        try:
            print(df_result.to_string())
        except UnicodeEncodeError:
            # Fallback: print as CSV if Unicode fails
            print(df_result.to_csv(index_label="Rank"))

        # Save to CSV
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nse_screener_results_{ts}.csv"
        df_result.to_csv(filename, index_label="Rank")
        print(f"\nResults saved to: {filename}")
        return df_result
    else:
        print("No stocks matched the criteria.")
        return pd.DataFrame()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NSE Stock Screener – 40% down from 52W high + EMA10 in uptrend"
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help="Limit number of stocks to scan (default: all NSE stocks). "
             "Use a small number like 100 for quick testing.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay in seconds between API calls (default: 0.3).",
    )
    args = parser.parse_args()

    run_screener(max_stocks=args.max_stocks, delay=args.delay)