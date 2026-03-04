"""
NSE Stock Screener with Quantitative Analysis & Game Theory
============================================================
Original Strategy: Down 40%+ from 52W high + EMA10 uptrend
Enhanced with:
- Quantitative Metrics: Sharpe Ratio, Volatility, Risk-Reward Scoring
- Game Theory: Nash Equilibrium modeling, Optimal Entry/Exit strategies
- Statistical Analysis: Monte Carlo simulations, Win/Loss probabilities

Dependencies: pip install yfinance pandas numpy requests scipy
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import warnings
from datetime import datetime, timedelta
from scipy import stats

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. Fetch NSE stock list
# ──────────────────────────────────────────────

def get_nse_symbols(max_stocks: int = None) -> list[str]:
    """Load NSE equity symbol list from local CSV or online source."""
    csv_path = r"C:\Users\ankit\OneDrive\Desktop\Personal\Nitish\stock_dashboard\StockTrader\Results\nse_stocks_list.csv"
    try:
        print(f"Loading NSE equity list from: {csv_path}")
        df = pd.read_csv(csv_path)
        symbol_col = [c for c in df.columns if "SYMBOL" in c.upper()][0]
        symbols = df[symbol_col].dropna().str.strip().tolist()
        print(f"  [OK] {len(symbols)} symbols loaded from CSV.")
    except FileNotFoundError:
        print(f"  [ERR] CSV file not found. Using fallback sample.")
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
        symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR"]

    if max_stocks:
        symbols = symbols[:max_stocks]

    return [f"{s}.NS" for s in symbols]


# ──────────────────────────────────────────────
# 2. Technical Analysis & Quantitative Metrics
# ──────────────────────────────────────────────

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """
    Calculate Sharpe Ratio (annualized).
    Measures risk-adjusted return.
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    return np.sqrt(252) * (excess_returns.mean() / returns.std())


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """
    Calculate Sortino Ratio - focuses only on downside volatility.
    Better measure for asymmetric returns.
    """
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - (risk_free_rate / 252)
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return np.sqrt(252) * (excess_returns.mean() / downside.std())


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate Maximum Drawdown percentage."""
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak * 100
    return drawdown.min()


def calculate_volatility(returns: pd.Series) -> float:
    """Calculate annualized volatility."""
    if len(returns) < 2:
        return 0.0
    return returns.std() * np.sqrt(252) * 100


def is_ema10_uptrend(close: pd.Series, lookback: int = 5) -> bool:
    """Check if EMA10 shows uptrend using linear regression."""
    if len(close) < 15:
        return False
    ema = calculate_ema(close, 10)
    recent_ema = ema.iloc[-lookback:]
    x = np.arange(len(recent_ema))
    slope = np.polyfit(x, recent_ema.values, 1)[0]
    return slope > 0


# ──────────────────────────────────────────────
# 3. Game Theory Analysis
# ──────────────────────────────────────────────

def calculate_nash_equilibrium_score(current_price: float, 
                                      ema10: float, 
                                      high_52w: float, 
                                      low_52w: float) -> float:
    """
    Nash Equilibrium Score based on Game Theory.
    Models the optimal strategy when buyers and sellers are rational actors.
    
    Score components:
    - Distance from 52W low (support strength)
    - Distance from EMA10 (trend alignment)
    - Risk-reward asymmetry (upside vs downside potential)
    
    Returns: Score 0-100 (higher = better Nash equilibrium for buyers)
    """
    # Position in 52-week range (0 = at low, 1 = at high)
    range_52w = high_52w - low_52w
    if range_52w == 0:
        return 0.0
    
    position_in_range = (current_price - low_52w) / range_52w
    
    # Distance from EMA10 (proximity to trend support)
    ema_proximity = abs(current_price - ema10) / current_price
    
    # Upside potential (to 52W high) vs Downside risk (to 52W low)
    upside = (high_52w - current_price) / current_price
    downside = (current_price - low_52w) / current_price if current_price > low_52w else 0.01
    risk_reward_ratio = upside / downside if downside > 0 else 0
    
    # Nash Score: weighted combination
    # Near 52W low + above EMA10 + high risk-reward = optimal entry
    support_score = (1 - position_in_range) * 40  # 40% weight (lower in range = better)
    trend_score = (1 - min(ema_proximity, 0.15) / 0.15) * 30  # 30% weight
    rr_score = min(risk_reward_ratio / 3, 1) * 30  # 30% weight (cap at 3:1)
    
    return support_score + trend_score + rr_score


def calculate_kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly Criterion - optimal position sizing.
    f* = (p * b - q) / b
    where p = win probability, q = loss probability, b = win/loss ratio
    """
    if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    b = avg_win / avg_loss  # Win/loss ratio
    q = 1 - win_rate
    kelly_pct = (win_rate * b - q) / b
    
    # Apply half-Kelly for safety
    return max(0, min(kelly_pct * 0.5, 0.25))  # Cap at 25% of portfolio


def monte_carlo_simulation(returns: pd.Series, days_forward: int = 30, 
                           simulations: int = 1000) -> dict:
    """
    Monte Carlo simulation for future price movements.
    Returns probability distribution of outcomes.
    """
    if len(returns) < 10:
        return {"prob_profit": 0.5, "expected_return": 0.0, "var_95": 0.0}
    
    mu = returns.mean()
    sigma = returns.std()
    
    final_returns = []
    for _ in range(simulations):
        daily_returns = np.random.normal(mu, sigma, days_forward)
        cumulative_return = (1 + daily_returns).prod() - 1
        final_returns.append(cumulative_return)
    
    final_returns = np.array(final_returns)
    
    return {
        "prob_profit": (final_returns > 0).mean(),  # Probability of profit
        "expected_return": final_returns.mean() * 100,  # Expected return %
        "var_95": np.percentile(final_returns, 5) * 100,  # 95% VaR
        "median_return": np.median(final_returns) * 100,
    }


# ──────────────────────────────────────────────
# 4. Enhanced Screening with Quant + Game Theory
# ──────────────────────────────────────────────

def screen_stock_quant(ticker: str) -> dict | None:
    """
    Enhanced screening with quantitative and game theory analysis.
    """
    try:
        df = yf.download(ticker, period="1y", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:  # Need more data for quant analysis
            return None

        close = df["Close"].squeeze()
        high_52w = df["High"].squeeze().max()
        low_52w = df["Low"].squeeze().min()
        current_price = close.iloc[-1]
        
        # Calculate returns
        returns = close.pct_change().dropna()

        # Filter 1: >= 40% below 52-week high
        drop_pct = (high_52w - current_price) / high_52w * 100
        if drop_pct < 40:
            return None

        # Filter 2: EMA10 showing uptrend
        if not is_ema10_uptrend(close):
            return None

        # ---- Quantitative Metrics ----
        ema10_latest = calculate_ema(close, 10).iloc[-1]
        ema20_latest = calculate_ema(close, 20).iloc[-1]
        ema50_latest = calculate_ema(close, 50).iloc[-1]
        
        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)
        volatility = calculate_volatility(returns)
        max_dd = calculate_max_drawdown(close)
        
        # ---- Game Theory Analysis ----
        nash_score = calculate_nash_equilibrium_score(
            current_price, ema10_latest, high_52w, low_52w
        )
        
        # Monte Carlo simulation
        mc_results = monte_carlo_simulation(returns, days_forward=30, simulations=1000)
        
        # Kelly Criterion (using historical win rate approximation)
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.5
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0.01
        avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0.01
        kelly_pct = calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        
        # Risk-Reward Ratio
        upside_to_52w = ((high_52w - current_price) / current_price) * 100
        downside_to_52wlow = ((current_price - low_52w) / current_price) * 100
        risk_reward = upside_to_52w / downside_to_52wlow if downside_to_52wlow > 0 else 0
        
        # Overall Quant Score (0-100)
        quant_score = (
            min(sharpe / 2, 1) * 20 +  # Sharpe component
            min(sortino / 2, 1) * 20 +  # Sortino component
            mc_results["prob_profit"] * 30 +  # Monte Carlo probability
            (nash_score / 100) * 30  # Nash equilibrium
        )

        return {
            "Ticker": ticker.replace(".NS", ""),
            "Current Price (INR)": round(float(current_price), 2),
            "52W High (INR)": round(float(high_52w), 2),
            "52W Low (INR)": round(float(low_52w), 2),
            "Drop from 52W High (%)": round(float(drop_pct), 2),
            "EMA10": round(float(ema10_latest), 2),
            "EMA20": round(float(ema20_latest), 2),
            "EMA50": round(float(ema50_latest), 2),
            # Quant Metrics
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2),
            "Volatility (%)": round(volatility, 2),
            "Max Drawdown (%)": round(max_dd, 2),
            # Game Theory Metrics
            "Nash Score": round(nash_score, 2),
            "MC Profit Prob (%)": round(mc_results["prob_profit"] * 100, 2),
            "MC Expected Return (%)": round(mc_results["expected_return"], 2),
            "MC VaR 95% (%)": round(mc_results["var_95"], 2),
            "Kelly % (Position Size)": round(kelly_pct * 100, 2),
            "Risk-Reward Ratio": round(risk_reward, 2),
            # Combined Score
            "Quant Score (0-100)": round(quant_score, 2),
            "Price vs EMA10": "Below" if current_price < ema10_latest else "Above",
        }
    except Exception as e:
        return None


# ──────────────────────────────────────────────
# 5. Main Runner
# ──────────────────────────────────────────────

def run_quant_screener(max_stocks: int = None, delay: float = 0.3):
    """
    Run quantitative & game theory enhanced screener.
    """
    symbols = get_nse_symbols(max_stocks=max_stocks)
    total = len(symbols)
    print(f"\n{'='*80}")
    print(f"QUANTITATIVE & GAME THEORY STOCK SCREENER")
    print(f"{'='*80}")
    print(f"Screening {total} stocks with advanced analytics...\n")

    results = []
    for i, ticker in enumerate(symbols, 1):
        if i % 50 == 0 or i == 1:
            print(f"  Progress: {i}/{total} ...")
        result = screen_stock_quant(ticker)
        if result:
            results.append(result)
            print(f"\n  [MATCH] {result['Ticker']:<12}")
            print(f"    Price: {result['Current Price (INR)']:.2f} | 52W High: {result['52W High (INR)']:.2f} | 52W Low: {result['52W Low (INR)']:.2f}")
            print(f"    Drop: {result['Drop from 52W High (%)']:.1f}% | Quant Score: {result['Quant Score (0-100)']:.1f}")
            print(f"    Sharpe: {result['Sharpe Ratio']:.2f} | Sortino: {result['Sortino Ratio']:.2f} | Nash: {result['Nash Score']:.1f}")
            print(f"    MC Profit Prob: {result['MC Profit Prob (%)']:.1f}% | Kelly Size: {result['Kelly % (Position Size)']:.1f}%")
            print(f"    Risk-Reward: {result['Risk-Reward Ratio']:.2f}:1")
        time.sleep(delay)

    print(f"\n{'='*80}")
    print(f"Screening complete. {len(results)} stocks matched the criteria.")
    print(f"{'='*80}\n")

    if results:
        df_result = pd.DataFrame(results).sort_values(
            "Quant Score (0-100)", ascending=False
        ).reset_index(drop=True)
        df_result.index += 1

        try:
            print(df_result.to_string())
        except UnicodeEncodeError:
            print(df_result.to_csv(index_label="Rank"))

        # Save to CSV
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quant_gametheory_results_{ts}.csv"
        df_result.to_csv(filename, index_label="Rank")
        print(f"\nResults saved to: {filename}")
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("PORTFOLIO SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"Average Quant Score: {df_result['Quant Score (0-100)'].mean():.2f}")
        print(f"Average Sharpe Ratio: {df_result['Sharpe Ratio'].mean():.2f}")
        print(f"Average Nash Score: {df_result['Nash Score'].mean():.2f}")
        print(f"Average MC Profit Probability: {df_result['MC Profit Prob (%)'].mean():.1f}%")
        print(f"Average Risk-Reward Ratio: {df_result['Risk-Reward Ratio'].mean():.2f}:1")
        print(f"Recommended Kelly Position Size: {df_result['Kelly % (Position Size)'].mean():.1f}%")
        
        return df_result
    else:
        print("No stocks matched the criteria.")
        return pd.DataFrame()


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NSE Stock Screener with Quantitative Analysis & Game Theory"
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=None,
        help="Limit number of stocks to scan (default: all NSE stocks)."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay in seconds between API calls (default: 0.3)."
    )
    args = parser.parse_args()

    run_quant_screener(max_stocks=args.max_stocks, delay=args.delay)
