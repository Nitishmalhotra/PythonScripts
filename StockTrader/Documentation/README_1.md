# NSE Stock Screener

A Python script that scans NSE-listed stocks and identifies potential **value traps or recovery candidates** — stocks that are significantly beaten down from their 52-week highs but have not yet shown a recovery trend via EMA10.

---

## What It Does

The screener applies **two filters** to every NSE stock:

| Filter | Criteria |
|--------|----------|
| **Price Drop** | Current price is >= 40% below the 52-week high |
| **EMA10 Trend** | EMA10 slope over the last 5 bars is **rising** (positive slope) |

Stocks that satisfy **both** conditions are flagged as matches and saved to a CSV report.

---

## Why These Filters?

- **40% drop from 52W high** highlights stocks in significant drawdown — often due to sector headwinds, earnings misses, or broader market sell-offs.
- **EMA10 in uptrend** ensures the stock is beginning to show early momentum recovery despite being deeply discounted — a classic setup for potential mean-reversion or turnaround plays.

---

## Installation

**Requirements:** Python 3.10+

Install dependencies:

```bash
pip install yfinance pandas numpy requests
```

---

## Usage

### Scan all NSE stocks (full run, ~15-30 min)
```bash
python nse_screener.py
```

### Quick test with first 100 stocks
```bash
python nse_screener.py --max-stocks 100
```

### Reduce delay between API calls
```bash
python nse_screener.py --delay 0.1
```

### All options
```bash
python nse_screener.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-stocks` | None (all) | Limit number of stocks to scan |
| `--delay` | 0.3 | Seconds between API calls (increase if rate-limited) |

---

## Stock List Source

The script loads the NSE symbol list in the following priority order:

1. **Local CSV file** — path configured inside `get_nse_symbols()`:
   ```
   C:\Users\ankit\...\nse_stocks_list.csv
   ```
   Update this path to point to your own local copy. The CSV must have a column named `SYMBOL` (case-insensitive).

2. **NSE online source** — fetched from `nseindia.com` if the local file is not found.

3. **Fallback list** — 30 common NSE blue-chip symbols used if both above fail.

---

## Output

### Console
Matching stocks are printed in real-time as they are found:

```
[MATCH] EXAMPLECO    | Current: 134.50 | 52W High: 310.00 | 52W Low: 120.00 | Drop: 56.6% | EMA10: 138.20
```

### CSV File
Results are saved automatically to the working directory:

```
nse_screener_results_YYYYMMDD_HHMMSS.csv
```

**Output columns:**

| Column | Description |
|--------|-------------|
| Rank | Sorted rank (1 = largest drop) |
| Ticker | NSE trading symbol |
| Current Price (INR) | Latest closing price |
| 52W High (INR) | Highest price in the last 52 weeks |
| 52W Low (INR) | Lowest price in the last 52 weeks |
| Drop from 52W High (%) | Percentage decline from 52-week high |
| EMA10 | Current 10-period Exponential Moving Average value |
| Price vs EMA10 | Whether current price is Above or Below EMA10 |

Results are sorted by **Drop from 52W High (%) descending** — most beaten-down stocks appear first.

---

## How EMA10 Trend Is Calculated

A linear regression slope is computed on the last 5 EMA10 values using `numpy.polyfit`:

- **Slope > 0** means EMA10 is rising → stock IS in uptrend → **included** in results
- **Slope <= 0** means EMA10 is flat or falling → stock is NOT in uptrend → **excluded** from results

A minimum of 15 data points is required; stocks with insufficient history are skipped.

---

## Disclaimer

> This tool is for **informational and research purposes only**. It does not constitute financial advice. Always conduct your own due diligence before making any investment decisions. Past price performance is not indicative of future results.

---

## Project Structure

```
.
├── nse_screener.py               # Main screener script
├── nse_stocks_list.csv           # (Optional) Local NSE symbol list
├── README.md                     # This file
└── nse_screener_results_*.csv    # Auto-generated output files
```
