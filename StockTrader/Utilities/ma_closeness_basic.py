import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: yfinance. Install with: pip install yfinance pandas"
    ) from exc


def calculate_ma_metrics(symbol: str, period: str = "2y", ma_type: str = "ema") -> dict | None:
    ticker = f"{symbol}.NS"
    data = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)

    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            return None
        close_data = data.xs("Close", axis=1, level=0)
    else:
        if "Close" not in data.columns:
            return None
        close_data = data["Close"]

    if isinstance(close_data, pd.DataFrame):
        if close_data.shape[1] == 0:
            return None
        close_data = close_data.iloc[:, 0]

    close = pd.to_numeric(close_data, errors="coerce").dropna()
    if len(close) < 200:
        return None

    if ma_type == "ema":
        ma20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ma50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        ma100 = float(close.ewm(span=100, adjust=False).mean().iloc[-1])
        ma200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
    else:
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma100 = float(close.rolling(100).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
    last_close = float(close.iloc[-1])

    if pd.isna(ma20) or pd.isna(ma50) or pd.isna(ma100) or pd.isna(ma200) or last_close <= 0:
        return None

    ma_values = [ma20, ma50, ma100, ma200]

    ma_band_pct = ((max(ma_values) - min(ma_values)) / last_close) * 100

    pairwise = [
        abs(ma20 - ma50),
        abs(ma20 - ma100),
        abs(ma20 - ma200),
        abs(ma50 - ma100),
        abs(ma50 - ma200),
        abs(ma100 - ma200),
    ]
    avg_pair_diff_pct = (sum(pairwise) / len(pairwise) / last_close) * 100

    return {
        "symbol": symbol,
        "last_close": round(last_close, 2),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "ma100": round(ma100, 2),
        "ma200": round(ma200, 2),
        "ma_band_pct": round(ma_band_pct, 3),
        "avg_pair_diff_pct": round(avg_pair_diff_pct, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read NSE stock list CSV and compute MA20/50/100/200 plus closeness metrics."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=r"C:\Users\ankit\OneDrive\Desktop\Personal\Nitish\stock_dashboard\StockTrader\Results\nse_stocks_list.csv",
        help="Path to input CSV containing tradingsymbol column",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output CSV path. If not provided, auto-saves into Results folder.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of symbols to process from the CSV (default: 50 for faster run).",
    )
    parser.add_argument(
        "--ma-type",
        type=str,
        default="ema",
        choices=["ema", "sma"],
        help="Moving average type: ema or sma (default: ema)",
    )
    parser.add_argument(
        "--history-period",
        type=str,
        default="2y",
        help="Price history period to fetch from Yahoo (e.g., 1y, 2y, 5y, max).",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="",
        help="Optional single tradingsymbol to process (e.g., ANGELONE).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    stocks_df = pd.read_csv(input_path)
    if "tradingsymbol" not in stocks_df.columns:
        raise SystemExit("CSV must contain 'tradingsymbol' column")

    symbols = (
        stocks_df["tradingsymbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("")]
        .tolist()
    )

    if args.symbol:
        symbols = [args.symbol.strip().upper()]
    elif args.limit > 0:
        symbols = symbols[: args.limit]

    results = []
    for i, symbol in enumerate(symbols, start=1):
        row = calculate_ma_metrics(symbol, period=args.history_period, ma_type=args.ma_type)
        if row:
            results.append(row)
        if i % 10 == 0:
            print(f"Processed {i}/{len(symbols)}")

    if not results:
        raise SystemExit("No valid stocks processed. Check symbols/internet connection.")

    output_df = pd.DataFrame(results).sort_values("ma_band_pct", ascending=True)

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = input_path.parent / f"ma_closeness_report_{timestamp}.csv"

    output_df.to_csv(output_path, index=False)

    print(f"\nTop 10 stocks where {args.ma_type.upper()}20/50/100/200 are closest:")
    print(
        output_df[
            [
                "symbol",
                "last_close",
                "ma20",
                "ma50",
                "ma100",
                "ma200",
                "ma_band_pct",
                "avg_pair_diff_pct",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
