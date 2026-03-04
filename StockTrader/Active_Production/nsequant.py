"""
NSE Stock Screener — Enhanced with Quantitative Analysis & Game Theory
=======================================================================
Identifies stocks that are:
  1. Down 40%+ from their 52-week high
  2. Showing an uptrend in EMA10

Then scores each matched stock using:
  A. Quantitative Analysis  — momentum, volatility, mean-reversion signals
  B. Game Theory            — Nash Equilibrium player-behaviour modelling,
                              institutional vs retail signal, prisoner's
                              dilemma payoff matrix for buy/hold/sell

Dependencies:
    pip install yfinance pandas numpy requests scipy scikit-learn
"""

import warnings
warnings.filterwarnings("ignore")

import time
import requests
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

import yfinance as yf


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — NSE Symbol Loader
# ══════════════════════════════════════════════════════════════════════════════

def get_nse_symbols(max_stocks: int = None, nifty50_only: bool = False) -> list[str]:
    # NIFTY50 symbols
    nifty50 = {'RELIANCE':'Reliance Industries','TCS':'TCS','HDFCBANK':'HDFC Bank',
    'INFY':'Infosys','ICICIBANK':'ICICI Bank','HINDUNILVR':'Hindustan Unilever',
    'ITC':'ITC','SBIN':'State Bank of India','BHARTIARTL':'Bharti Airtel',
    'KOTAKBANK':'Kotak Mahindra Bank','LT':'Larsen & Toubro','AXISBANK':'Axis Bank',
    'ASIANPAINT':'Asian Paints','MARUTI':'Maruti Suzuki','SUNPHARMA':'Sun Pharma',
    'TITAN':'Titan Company','BAJFINANCE':'Bajaj Finance','WIPRO':'Wipro',
    'ULTRACEMCO':'UltraTech Cement','NESTLEIND':'Nestle India','ONGC':'ONGC',
    'NTPC':'NTPC','POWERGRID':'Power Grid','HCLTECH':'HCL Technologies',
    'COALINDIA':'Coal India','BAJAJFINSV':'Bajaj Finserv','M&M':'Mahindra & Mahindra',
    'ADANIPORTS':'Adani Ports','TATASTEEL':'Tata Steel','GRASIM':'Grasim Industries',
    'TECHM':'Tech Mahindra','INDUSINDBK':'IndusInd Bank','DIVISLAB':"Divi's Labs",
    'DRREDDY':"Dr. Reddy's",'CIPLA':'Cipla','EICHERMOT':'Eicher Motors',
    'HINDALCO':'Hindalco','BRITANNIA':'Britannia','JSWSTEEL':'JSW Steel',
    'APOLLOHOSP':'Apollo Hospitals','HEROMOTOCO':'Hero MotoCorp','SHREECEM':'Shree Cement',
    'BPCL':'BPCL','TATACONSUM':'Tata Consumer','UPL':'UPL',
    'SBILIFE':'SBI Life Insurance','BAJAJ-AUTO':'Bajaj Auto',
    'ADANIENT':'Adani Enterprises',
    }
    
    # Always use NIFTY50 for now (CSV/online loading temporarily disabled)
    print("Loading NIFTY50 stocks only...")
    symbols = list(nifty50.keys())
    # Remove duplicates while preserving order
    symbols = list(dict.fromkeys(symbols))
    print(f"  [OK] {len(symbols)} NIFTY50 symbols loaded.")
    
    # ── TEMPORARILY DISABLED: CSV and online NSE loading ──────────────────────
    # else:
    #     csv_path = r"C:\Users\ankit\OneDrive\Desktop\Personal\Nitish\stock_dashboard\StockTrader\Results\nse_stocks_list.csv"
    #     try:
    #         print(f"Loading NSE equity list from: {csv_path}")
    #         df = pd.read_csv(csv_path)
    #         symbol_col = [c for c in df.columns if "SYMBOL" in c.upper()][0]
    #         symbols = df[symbol_col].dropna().str.strip().tolist()
    #         print(f"  [OK] {len(symbols)} symbols loaded from CSV.")
    #     except FileNotFoundError:
    #         print(f"  [WARN] Local CSV not found. Trying NSE online source...")
    #         try:
    #             url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    #             headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"}
    #             resp = requests.get(url, headers=headers, timeout=15)
    #             resp.raise_for_status()
    #             df = pd.read_csv(StringIO(resp.text))
    #             symbol_col = [c for c in df.columns if "SYMBOL" in c.upper()][0]
    #             symbols = df[symbol_col].dropna().str.strip().tolist()
    #             print(f"  [OK] {len(symbols)} symbols loaded from NSE.")
    #         except Exception as e:
    #             print(f"  [WARN] Online fetch failed ({e}). Using fallback list.")
    #             symbols = [
    #                 "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR",
    #                 "ICICIBANK", "KOTAKBANK", "SBIN", "BAJFINANCE", "BHARTIARTL",
    #                 "ITC", "ASIANPAINT", "AXISBANK", "LT", "DMART",
    #                 "SUNPHARMA", "TITAN", "NESTLEIND", "WIPRO", "ULTRACEMCO",
    #                 "POWERGRID", "NTPC", "TECHM", "HCLTECH", "MARUTI",
    #                 "ONGC", "COALINDIA", "TATASTEEL", "JSWSTEEL", "BAJAJ-AUTO",
    #             ]
    #     except Exception as e:
    #         print(f"  [WARN] CSV read error ({e}). Using fallback list.")
    #         symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]
    # ──────────────────────────────────────────────────────────────────────────

    if max_stocks:
        symbols = symbols[:max_stocks]
    return [f"{s}.NS" for s in symbols]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Technical Helpers
# ══════════════════════════════════════════════════════════════════════════════

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return float(100 - 100 / (1 + rs.iloc[-1]))

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def macd_signal(close: pd.Series) -> dict:
    macd_line   = ema(close, 12) - ema(close, 26)
    signal_line = ema(macd_line, 9)
    histogram   = macd_line - signal_line
    return {
        "macd":      float(macd_line.iloc[-1]),
        "signal":    float(signal_line.iloc[-1]),
        "histogram": float(histogram.iloc[-1]),
        "bullish":   bool(macd_line.iloc[-1] > signal_line.iloc[-1]),
    }

def bollinger_bands(close: pd.Series, period: int = 20) -> dict:
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    price = close.iloc[-1]
    band_width = float((upper.iloc[-1] - lower.iloc[-1]) / mid.iloc[-1])
    pct_b = float((price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]))
    return {"upper": float(upper.iloc[-1]), "lower": float(lower.iloc[-1]),
            "mid": float(mid.iloc[-1]), "pct_b": pct_b, "band_width": band_width}

def ema10_slope(close: pd.Series, lookback: int = 5) -> float:
    e = ema(close, 10)
    x = np.arange(lookback)
    slope, *_ = np.polyfit(x, e.iloc[-lookback:].values, 1)
    return float(slope)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Quantitative Analysis Module
# ══════════════════════════════════════════════════════════════════════════════

class QuantAnalyser:
    """
    Computes a multi-factor quant score for a single stock.

    Factors
    -------
    1. Momentum Score       - price & EMA momentum across multiple timeframes
    2. Mean-Reversion Score - how far price has deviated from statistical mean
    3. Volatility Score     - ATR-normalised, Bollinger Band width
    4. Volume Confirmation  - OBV trend, volume spike detection
    5. Statistical Edge     - Z-score of returns, Sharpe-like ratio

    Weights: Momentum 30% | Mean-Reversion 25% | Volume 20% | Volatility 15% | Statistical 10%
    """

    def __init__(self, df: pd.DataFrame):
        self.close  = df["Close"].squeeze()
        self.high   = df["High"].squeeze()
        self.low    = df["Low"].squeeze()
        self.volume = df["Volume"].squeeze()
        self.returns = self.close.pct_change().dropna()

    # ── 3a. Momentum ──────────────────────────────────────────────────────────
    def momentum_score(self) -> dict:
        c     = self.close
        price = float(c.iloc[-1])

        roc = {
            "roc_5":  float((c.iloc[-1] / c.iloc[-6]  - 1) * 100) if len(c) > 6  else 0,
            "roc_10": float((c.iloc[-1] / c.iloc[-11] - 1) * 100) if len(c) > 11 else 0,
            "roc_20": float((c.iloc[-1] / c.iloc[-21] - 1) * 100) if len(c) > 21 else 0,
        }

        e10, e20, e50 = ema(c, 10), ema(c, 20), ema(c, 50)
        ema_stack = (int(price > e10.iloc[-1]) +
                     int(e10.iloc[-1] > e20.iloc[-1]) +
                     int(e20.iloc[-1] > e50.iloc[-1]))  # 0-3

        rsi_val   = rsi(c)
        rsi_score = 1.0 if 40 < rsi_val < 70 else (0.5 if rsi_val <= 40 else 0.0)

        mc         = macd_signal(c)
        macd_score = 1.0 if mc["bullish"] and mc["histogram"] > 0 else 0.0

        raw = (
            (roc["roc_5"] + roc["roc_10"] + roc["roc_20"]) / 3 * 0.3 +
            ema_stack / 3 * 40 * 0.3 +
            rsi_score  * 40 * 0.2 +
            macd_score * 40 * 0.2
        )
        return {
            **roc,
            "rsi": round(rsi_val, 2),
            "ema_stack": ema_stack,
            "macd_bullish": mc["bullish"],
            "macd_histogram": round(mc["histogram"], 4),
            "momentum_score": round(min(max(raw, 0), 100), 2),
        }

    # ── 3b. Mean Reversion ────────────────────────────────────────────────────
    def mean_reversion_score(self) -> dict:
        c     = self.close
        price = float(c.iloc[-1])

        mean_52w = float(c.rolling(252).mean().iloc[-1]) if len(c) >= 252 else float(c.mean())
        std_52w  = float(c.rolling(252).std().iloc[-1])  if len(c) >= 252 else float(c.std())
        z_price  = (price - mean_52w) / std_52w if std_52w else 0

        bb    = bollinger_bands(c)
        pct_b = bb["pct_b"]

        mr_raw = min(max((-z_price + 2) / 4 * 100, 0), 100)

        return {
            "z_price": round(z_price, 3),
            "bb_pct_b": round(pct_b, 3),
            "bb_width": round(bb["band_width"], 4),
            "mean_reversion_score": round(mr_raw, 2),
        }

    # ── 3c. Volatility ────────────────────────────────────────────────────────
    def volatility_score(self) -> dict:
        atr_val = atr(self.high, self.low, self.close)
        price   = float(self.close.iloc[-1])
        atr_pct = atr_val / price * 100
        hv      = float(self.returns.std() * np.sqrt(252) * 100)
        vol_score = max(0, 100 - hv)
        return {
            "atr_pct": round(atr_pct, 2),
            "hist_vol_pct": round(hv, 2),
            "volatility_score": round(vol_score, 2),
        }

    # ── 3d. Volume Confirmation ───────────────────────────────────────────────
    def volume_score(self) -> dict:
        v, c = self.volume, self.close

        obv           = (np.sign(c.diff()) * v).fillna(0).cumsum()
        obv_slope, *_ = np.polyfit(np.arange(10), obv.iloc[-10:].values, 1)

        vol_20_avg = float(v.rolling(20).mean().iloc[-1])
        vol_ratio  = float(v.iloc[-1] / vol_20_avg) if vol_20_avg else 1.0

        vol_score = min((obv_slope > 0) * 50 + min(vol_ratio, 2) / 2 * 50, 100)
        return {
            "obv_rising": bool(obv_slope > 0),
            "volume_ratio_vs_avg": round(vol_ratio, 2),
            "volume_score": round(vol_score, 2),
        }

    # ── 3e. Statistical Edge ──────────────────────────────────────────────────
    def statistical_edge(self) -> dict:
        r       = self.returns
        mean_r  = float(r.mean())
        std_r   = float(r.std())
        sharpe  = mean_r / std_r * np.sqrt(252) if std_r else 0
        skew    = float(stats.skew(r.dropna()))
        win_rate = float((r.iloc[-20:] > 0).mean() * 100)
        t_stat, p_val = stats.ttest_1samp(r.iloc[-20:].dropna(), 0)

        edge_score = min(max(
            (sharpe + 1) / 4 * 40 +
            min(max(skew, -1), 1) / 2 * 20 + 10 +
            win_rate / 100 * 30 +
            (1 - p_val) * 10,
            0), 100)

        return {
            "sharpe_approx": round(sharpe, 3),
            "return_skew": round(skew, 3),
            "win_rate_20d": round(win_rate, 1),
            "t_pvalue": round(float(p_val), 4),
            "statistical_edge_score": round(edge_score, 2),
        }

    # ── 3f. Composite Quant Score ─────────────────────────────────────────────
    def composite_score(self) -> dict:
        m   = self.momentum_score()
        mr  = self.mean_reversion_score()
        v   = self.volatility_score()
        vc  = self.volume_score()
        se  = self.statistical_edge()

        quant_total = (
            m["momentum_score"]          * 0.30 +
            mr["mean_reversion_score"]   * 0.25 +
            v["volatility_score"]        * 0.15 +
            vc["volume_score"]           * 0.20 +
            se["statistical_edge_score"] * 0.10
        )
        return {**m, **mr, **v, **vc, **se, "quant_score": round(quant_total, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Game Theory Module
# ══════════════════════════════════════════════════════════════════════════════

class GameTheoryAnalyser:
    """
    Models market participants as strategic players and computes:

    1. Nash Equilibrium Signal
       Players  : Institutional Investors  vs  Retail Traders
       Actions  : BUY | HOLD | SELL
       Payoffs  : Derived from price, volume, and volatility proxies.
       The Nash Equilibrium is the action profile where no player
       benefits from deviating unilaterally given the other's strategy.

    2. Prisoner's Dilemma Payoff
       Models the BUY vs SELL tension when price is deeply depressed.
       Mutual cooperation (both BUY) yields the highest collective payoff.
       Defection (retail SELL into institutional BUY) yields a short-term
       gain for the defector but systemic loss for the market.

    3. Information Asymmetry Score
       KL-divergence between institutional and retail probability
       distributions. High divergence signals institutions are acting
       very differently from retail — a classic "smart money" setup.
    """

    # Payoff matrix — calibrated for a deeply discounted stock context
    # (institutional_action, retail_action): (inst_payoff, retail_payoff)
    PAYOFF = {
        ("BUY",  "BUY"):  (8, 6),   # mutual accumulation — best collective outcome
        ("BUY",  "HOLD"): (5, 3),
        ("BUY",  "SELL"): (3, 4),   # retail exits cheap, inst. accumulates
        ("HOLD", "BUY"):  (4, 5),
        ("HOLD", "HOLD"): (2, 2),   # stagnation
        ("HOLD", "SELL"): (1, 3),
        ("SELL", "BUY"):  (4, 1),   # inst. distributes into retail buying
        ("SELL", "HOLD"): (3, 1),
        ("SELL", "SELL"): (0, 0),   # full capitulation — maximum pain
    }
    ACTIONS = ["BUY", "HOLD", "SELL"]

    def __init__(self, close: pd.Series, volume: pd.Series,
                 high: pd.Series, low: pd.Series):
        self.close  = close
        self.volume = volume
        self.high   = high
        self.low    = low

    # ── 4a. Proxy signals for institutional vs retail behaviour ───────────────
    def _player_signals(self) -> dict:
        c, v = self.close, self.volume
        returns  = c.pct_change()

        # Institutional proxy: large-volume up-days (smart money accumulation)
        big_vol  = v > v.rolling(20).mean() * 1.5
        up_days  = returns > 0
        inst_acc = float((big_vol & up_days).iloc[-20:].mean())  # 0-1

        # Retail proxy: elevated daily range (panic / fear)
        daily_range      = (self.high - self.low) / c
        retail_fear      = float(daily_range.iloc[-10:].mean())
        retail_fear_norm = min(retail_fear / 0.05, 1.0)  # normalise 0-1

        return {"inst_acc": inst_acc, "retail_fear": retail_fear_norm}

    # ── 4b. Construct mixed-strategy probability distributions ────────────────
    def _action_probs(self, signals: dict) -> tuple[dict, dict]:
        ia = signals["inst_acc"]    # 0 = no accumulation, 1 = heavy
        rf = signals["retail_fear"] # 0 = calm, 1 = fearful

        inst_probs = {
            "BUY":  min(ia * 1.2, 0.80),
            "HOLD": 0.15,
            "SELL": max(1 - min(ia * 1.2, 0.80) - 0.15, 0.05),
        }
        retail_probs = {
            "SELL": min(rf * 1.1, 0.75),
            "HOLD": 0.20,
            "BUY":  max(1 - min(rf * 1.1, 0.75) - 0.20, 0.05),
        }

        for probs in (inst_probs, retail_probs):
            total = sum(probs.values())
            for k in probs:
                probs[k] = round(probs[k] / total, 4)

        return inst_probs, retail_probs

    # ── 4c. Nash Equilibrium finder (pure strategy) ───────────────────────────
    def _nash_equilibrium(self, inst_probs: dict, retail_probs: dict) -> dict:
        """
        Each player chooses the action that maximises their expected payoff
        given the opponent's mixed strategy. Returns dominant action per player.
        """
        best_inst = max(
            self.ACTIONS,
            key=lambda a_i: sum(
                retail_probs[a_r] * self.PAYOFF.get((a_i, a_r), (0, 0))[0]
                for a_r in self.ACTIONS))

        best_retail = max(
            self.ACTIONS,
            key=lambda a_r: sum(
                inst_probs[a_i] * self.PAYOFF.get((a_i, a_r), (0, 0))[1]
                for a_i in self.ACTIONS))

        nash_payoff = self.PAYOFF.get((best_inst, best_retail), (0, 0))

        return {
            "nash_inst_action":   best_inst,
            "nash_retail_action": best_retail,
            "nash_inst_payoff":   nash_payoff[0],
            "nash_retail_payoff": nash_payoff[1],
        }

    # ── 4d. Prisoner's Dilemma — cooperation score ────────────────────────────
    def _prisoners_dilemma_score(self, nash: dict) -> float:
        """
        Measures how close the equilibrium is to mutual cooperation (BUY/BUY).
        Score of 100 = both players at Nash BUY. Score of 0 = mutual SELL.
        """
        actual_payoff = self.PAYOFF.get(
            (nash["nash_inst_action"], nash["nash_retail_action"]), (0, 0))
        coop_payoff   = self.PAYOFF[("BUY", "BUY")]
        max_combined  = coop_payoff[0] + coop_payoff[1]
        actual_combined = actual_payoff[0] + actual_payoff[1]
        return round(actual_combined / max_combined * 100, 2)

    # ── 4e. Information Asymmetry (KL Divergence) ─────────────────────────────
    def _info_asymmetry(self, inst_probs: dict, retail_probs: dict) -> float:
        """
        KL-divergence between institutional and retail distributions.
        High divergence = institutions and retail are acting very differently
        = institutions have an information edge = stronger signal.
        """
        eps = 1e-9
        kl  = sum(
            inst_probs[a] * np.log((inst_probs[a] + eps) / (retail_probs[a] + eps))
            for a in self.ACTIONS)
        return round(min(kl * 50, 100), 2)

    # ── 4f. Composite Game Theory Score ───────────────────────────────────────
    def composite_score(self) -> dict:
        signals          = self._player_signals()
        inst_p, retail_p = self._action_probs(signals)
        nash             = self._nash_equilibrium(inst_p, retail_p)
        pd_score         = self._prisoners_dilemma_score(nash)
        asym             = self._info_asymmetry(inst_p, retail_p)

        # Score composition:
        # Nash BUY for institutions       -> +40 pts (strong signal)
        # High retail fear (contrarian)   -> +30 pts (sell panic = buy opportunity)
        # High cooperation score          -> +20 pts (alignment towards recovery)
        # High info asymmetry             -> +10 pts (institutions ahead of retail)
        nash_bonus = 40 if nash["nash_inst_action"] == "BUY" else \
                    (20 if nash["nash_inst_action"] == "HOLD" else 0)
        fear_bonus = signals["retail_fear"] * 30
        coop_bonus = pd_score / 100 * 20
        asym_bonus = asym / 100 * 10

        gt_score = round(min(nash_bonus + fear_bonus + coop_bonus + asym_bonus, 100), 2)

        return {
            "inst_buy_prob":           round(inst_p["BUY"], 3),
            "inst_sell_prob":          round(inst_p["SELL"], 3),
            "retail_fear_index":       round(signals["retail_fear"], 3),
            "retail_sell_prob":        round(retail_p["SELL"], 3),
            "nash_inst_action":        nash["nash_inst_action"],
            "nash_retail_action":      nash["nash_retail_action"],
            "nash_inst_payoff":        nash["nash_inst_payoff"],
            "prisoners_dilemma_score": pd_score,
            "info_asymmetry_score":    asym,
            "game_theory_score":       gt_score,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Final Conviction Rating
# ══════════════════════════════════════════════════════════════════════════════

def conviction_rating(quant_score: float, gt_score: float) -> dict:
    """
    Combines quant (60%) + game theory (40%) into a 0-100 CONVICTION score
    and maps it to a human-readable signal and star rating.
    """
    combined = round(quant_score * 0.60 + gt_score * 0.40, 2)

    if combined >= 75:
        signal, rating = "STRONG BUY", "5/5"
    elif combined >= 60:
        signal, rating = "BUY",        "4/5"
    elif combined >= 45:
        signal, rating = "WATCH",      "3/5"
    elif combined >= 30:
        signal, rating = "WEAK",       "2/5"
    else:
        signal, rating = "AVOID",      "1/5"

    return {"conviction_score": combined, "signal": signal, "rating": rating}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Per-stock Analysis Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def analyse_stock(ticker: str) -> dict | None:
    try:
        df = yf.download(ticker, period="1y", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            return None

        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        high_52w      = float(high.max())
        low_52w       = float(low.min())
        current_price = float(close.iloc[-1])

        # ── Primary filter 1: >= 40% below 52-week high ────────────────────
        drop_pct = (high_52w - current_price) / high_52w * 100
        if drop_pct < 10:
            return None

        # ── Primary filter 2: EMA10 in uptrend ────────────────────────────
        slope = ema10_slope(close)
        if slope <= 0:
            return None

        # ── Quant analysis ─────────────────────────────────────────────────
        qa     = QuantAnalyser(df)
        q_data = qa.composite_score()

        # ── Game theory analysis ───────────────────────────────────────────
        gt      = GameTheoryAnalyser(close, volume, high, low)
        gt_data = gt.composite_score()

        # ── Conviction rating ──────────────────────────────────────────────
        conv = conviction_rating(q_data["quant_score"], gt_data["game_theory_score"])

        return {
            # Base Info
            "Ticker":                   ticker.replace(".NS", ""),
            "Current Price (INR)":      round(current_price, 2),
            "52W High (INR)":           round(high_52w, 2),
            "52W Low (INR)":            round(low_52w, 2),
            "Drop from 52W High (%)":   round(drop_pct, 2),
            "EMA10":                    round(float(ema(close, 10).iloc[-1]), 2),
            "EMA10 Slope":              round(slope, 4),

            # Quant Scores
            "RSI":                      q_data["rsi"],
            "MACD Bullish":             q_data["macd_bullish"],
            "Momentum Score":           q_data["momentum_score"],
            "Mean Reversion Score":     q_data["mean_reversion_score"],
            "Volatility Score":         q_data["volatility_score"],
            "Volume Score":             q_data["volume_score"],
            "Statistical Edge Score":   q_data["statistical_edge_score"],
            "Quant Score":              q_data["quant_score"],

            # Game Theory
            "Inst Buy Prob":            gt_data["inst_buy_prob"],
            "Retail Fear Index":        gt_data["retail_fear_index"],
            "Nash Inst Action":         gt_data["nash_inst_action"],
            "Nash Retail Action":       gt_data["nash_retail_action"],
            "Prisoners Dilemma Score":  gt_data["prisoners_dilemma_score"],
            "Info Asymmetry Score":     gt_data["info_asymmetry_score"],
            "Game Theory Score":        gt_data["game_theory_score"],

            # Final Signal
            "Conviction Score":         conv["conviction_score"],
            "Signal":                   conv["signal"],
            "Rating":                   conv["rating"],
        }

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Main Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_screener(max_stocks: int = None, delay: float = 0.3,
                 min_conviction: float = 0.0, nifty50_only: bool = False):
    """
    Parameters
    ----------
    max_stocks      : cap on symbols to scan (None = all NSE stocks)
    delay           : seconds between yfinance API calls
    min_conviction  : filter final results to conviction_score >= this value
    nifty50_only    : if True, scan only NIFTY50 stocks
    """
    symbols = get_nse_symbols(max_stocks=max_stocks, nifty50_only=nifty50_only)
    total   = len(symbols)
    print(f"\nScanning {total} stocks — Quant Analysis + Game Theory\n")

    results = []
    for i, ticker in enumerate(symbols, 1):
        if i % 50 == 0 or i == 1:
            print(f"  Progress: {i}/{total} ...")

        result = analyse_stock(ticker)
        if result and result["Conviction Score"] >= min_conviction:
            results.append(result)
            print(
                f"  [MATCH] {result['Ticker']:<12} | "
                f"Drop: {result['Drop from 52W High (%)']:.1f}% | "
                f"Quant: {result['Quant Score']:.1f} | "
                f"GT: {result['Game Theory Score']:.1f} | "
                f"Conviction: {result['Conviction Score']:.1f} | "
                f"{result['Signal']} ({result['Rating']})"
            )
        time.sleep(delay)

    print(f"\n{'='*70}")
    print(f"Scan complete. {len(results)} stocks matched all criteria.")
    print(f"{'='*70}\n")

    if not results:
        print("No stocks matched. Try lowering --min-conviction or scanning more stocks.")
        return pd.DataFrame()

    df_out = (pd.DataFrame(results)
                .sort_values("Conviction Score", ascending=False)
                .reset_index(drop=True))
    df_out.index += 1

    display_cols = [
        "Ticker", "Current Price (INR)", "Drop from 52W High (%)",
        "RSI", "Quant Score", "Game Theory Score", "Conviction Score",
        "Signal", "Rating"
    ]
    try:
        print(df_out[display_cols].to_string())
    except UnicodeEncodeError:
        print(df_out[display_cols].to_csv(index_label="Rank"))

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nse_screener_quant_gt_{ts}.csv"
    df_out.to_csv(filename, index_label="Rank")
    print(f"\nFull results saved to: {filename}")
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "NSE Stock Screener — Quant Analysis + Game Theory\n"
            "---------------------------------------------------\n"
            "Screens for stocks 40%+ below 52W high with EMA10 uptrend,\n"
            "then ranks them using multi-factor quant models and Nash\n"
            "Equilibrium game theory to produce a CONVICTION score."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--max-stocks",     type=int,   default=None,
                        help="Max stocks to scan (default: all NSE stocks)")
    parser.add_argument("--delay",          type=float, default=0.3,
                        help="Delay between API calls in seconds (default: 0.3)")
    parser.add_argument("--min-conviction", type=float, default=0.0,
                        help="Only show stocks with conviction score >= N (default: 0)")
    parser.add_argument("--nifty50",        action="store_true",
                        help="Scan NIFTY50 stocks only")
    args = parser.parse_args()

    run_screener(
        max_stocks=args.max_stocks,
        delay=args.delay,
        min_conviction=args.min_conviction,
        nifty50_only=args.nifty50,
    )