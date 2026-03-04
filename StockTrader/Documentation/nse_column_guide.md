# NSE Stock Screener — Complete Column Guide

## 📋 CSV Output Reference

When you run the screener, all results are saved to a CSV file with the following columns:

---

## **SECTION 1: Basic Stock Information**

### Rank
- **Type:** Integer
- **Example:** 1, 2, 3...
- **Meaning:** Position in sorted results (highest conviction first)

### Ticker
- **Type:** Text (Stock Symbol)
- **Example:** ITC, RELIANCE, HINDUNILVR
- **Meaning:** NSE stock identifier without ".NS" suffix
- **Usage:** Use this for broker orders

---

## **SECTION 2: Price Data (52-Week Analysis)**

### Current Price (INR)
- **Type:** Currency (₹)
- **Example:** 415.50
- **Meaning:** Latest closing price in Indian Rupees
- **Significance:** Entry point reference for trading

### 52W High (INR)
- **Type:** Currency (₹)
- **Example:** 553.75
- **Meaning:** Highest price in last 52 weeks
- **Calculation:** max(price data from last 252 trading days)
- **Usage:** Identifies how far stock has fallen from recent peaks

### 52W Low (INR)
- **Type:** Currency (₹)
- **Example:** 285.20
- **Meaning:** Lowest price in last 52 weeks
- **Calculation:** min(price data from last 252 trading days)
- **Usage:** Shows support level; how much room for further downside

### Drop from 52W High (%)
- **Type:** Percentage (0-100)
- **Example:** 25.1%
- **Formula:** `(52W High - Current Price) / 52W High × 100`
- **Interpretation:**
  - 10-25% = Moderate drawdown (mean-reversion candidate)
  - 25-40% = Significant drawdown (high reversal potential)
  - 40%+ = Deep drawdown (severe oversold, high risk/reward)
- **Primary Filter:** ≥10% required for screening

---

## **SECTION 3: Technical Indicators**

### EMA10
- **Type:** Currency (₹)
- **Full Name:** Exponential Moving Average (10-day)
- **Example:** 405.32
- **Calculation:** Weighted average of last 10 closing prices (more weight to recent)
- **Interpretation:**
  - Price > EMA10 = Positive short-term trend
  - Price < EMA10 = Negative short-term trend
  - Distance from EMA10 = Deviation magnitude

### EMA10 Slope
- **Type:** Decimal (positive or negative)
- **Example:** 0.0234 or -0.0145
- **Meaning:** Rate of change of EMA10 (is it rising or falling?)
- **Formula:** Slope of linear regression fit across last 5 EMA10 values
- **Interpretation:**
  - \> 0 = EMA10 rising (uptrend confirmed ✓)
  - < 0 = EMA10 falling (downtrend, fails filter ✗)
- **Primary Filter:** Must be > 0 to pass screening

---

## **SECTION 4: Momentum Analysis**

### RSI (Relative Strength Index)
- **Type:** Number 0-100
- **Example:** 52.3
- **Range Interpretation:**
  - 0-30 = Oversold (potential buy)
  - 30-70 = Neutral zone
  - 70-100 = Overbought (potential sell)
- **Quality Zones:**
  - 40-60 = Optimal for reversal trades (stock neither too weak nor too strong)
  - 50-70 = Momentum building (trend continuation)
- **Calculation:** 100 - [100 / (1 + RS)] where RS = avg gain / avg loss over 14 days

### MACD Bullish
- **Type:** Boolean (True/False)
- **Example:** True or False
- **Meaning:** Has MACD line crossed above signal line?
- **True = Bullish:** Momentum shifting positive ✓
- **False = Bearish:** Momentum shifting negative ✗
- **How It Works:** Compares 12-day EMA with 26-day EMA and their 9-day signal line

---

## **SECTION 5: Quantitative Scores (0-100 Scale)**

### Momentum Score
- **Type:** 0-100
- **Weight in Quant:** 30%
- **Example:** 56.3
- **Measures:** Price acceleration across multiple timeframes
- **Components:**
  - 5-day, 10-day, 20-day Rate of Change (ROC)
  - EMA alignment (price > EMA10 > EMA20 > EMA50 = max 3 points)
  - RSI bullish zone confirmation
  - MACD bullish signal
- **High Score (75+):** Strong recent uptrend, good momentum
- **Medium Score (50-74):** Mixed signals, some momentum
- **Low Score (<50):** Weakening momentum, poor trend

### Mean Reversion Score
- **Type:** 0-100
- **Weight in Quant:** 25%
- **Example:** 62.8
- **Measures:** How oversold is the stock vs statistical mean?
- **Components:**
  - Z-score (standard deviations from 52-week mean)
  - Bollinger Band position (% between upper and lower bands)
  - Band width (volatility compression)
- **High Score (75+):** Extremely oversold, strong reversal potential
- **Medium Score (50-74):** Moderately oversold, good reversal opportunity
- **Low Score (<50):** Fair valued or overbought, limited reversal

### Volatility Score
- **Type:** 0-100
- **Weight in Quant:** 15%
- **Example:** 45.2
- **Measures:** Price stability and risk level
- **Components:**
  - ATR % (Average True Range as % of price)
  - Historical Volatility (annualized standard deviation)
- **High Score (75+):** Low volatility (stable, lower risk)
- **Medium Score (50-74):** Normal volatility
- **Low Score (<50):** High volatility (risky, but higher potential returns)

### Volume Score
- **Type:** 0-100
- **Weight in Quant:** 20%
- **Example:** 68.5
- **Measures:** Trading activity confirmation of price moves
- **Components:**
  - OBV trend (On-Balance Volume slope - positive = accumulation)
  - Volume ratio (current volume vs 20-day average)
- **High Score (75+):** Strong volume confirmation (credible move)
- **Medium Score (50-74):** Moderate volume support
- **Low Score (<50):** Weak volume (price move less credible)

### Statistical Edge Score
- **Type:** 0-100
- **Weight in Quant:** 10%
- **Example:** 54.1
- **Measures:** Mathematical probability of future success
- **Components:**
  - Sharpe Ratio (risk-adjusted returns)
  - Return Skewness (distribution shape, positive better)
  - 20-day Win Rate (% of up days)
  - T-test P-value (statistical significance of recent performance)
- **High Score (75+):** Strong statistical edge, good forward-looking probability
- **Medium Score (50-74):** Moderate edge
- **Low Score (<50):** Poor edge, weak forward probability

### Quant Score
- **Type:** 0-100
- **Example:** 56.3
- **Meaning:** Composite quantitative analysis score
- **Formula:**
  ```
  Quant Score = (Momentum × 0.30) + (MeanReversion × 0.25) 
              + (Volatility × 0.15) + (Volume × 0.20) 
              + (Statistical Edge × 0.10)
  ```
- **Interpretation:**
  - 75+ = Excellent quantitative setup
  - 60-74 = Good setup
  - 45-59 = Fair setup (needs game theory confirmation)
  - <45 = Weak setup

---

## **SECTION 6: Game Theory Scores**

### Inst Buy Prob (Institutional Buy Probability)
- **Type:** Decimal 0.000-1.000
- **Example:** 0.580
- **Meaning:** Probability (0-100%) that institutional investors are actively buying
- **Calculation:** Based on large-volume up-days (proxy for smart money accumulation)
- **0.0-0.3 = Institutions not interested**
- **0.3-0.6 = Moderate institutional interest**
- **0.6-1.0 = Strong institutional buying (excellent signal)**

### Inst Sell Prob (Institutional Sell Probability)
- **Type:** Decimal 0.000-1.000
- **Example:** 0.125
- **Meaning:** Probability (0-100%) that institutional investors are selling
- **Calculation:** 1 - Buy Prob (approximately)
- **0.0-0.2 = Institutions not selling (bullish)**
- **0.2-0.5 = Some institutional selling (caution)**
- **0.5-1.0 = Heavy institutional selling (bearish)**

### Retail Fear Index
- **Type:** Decimal 0.000-1.000
- **Example:** 0.680
- **Meaning:** Retail trader panic level (based on daily price range spike)
- **Calculation:** Average daily range (high-low)/close, normalized to 0-1
- **0.0-0.3 = Low fear, traders calm (bad entry timing)**
- **0.3-0.7 = Moderate fear (good contrarian entry, retail panicking)**
- **0.7-1.0 = High fear, panic selling (classic reversal setup)**

### Retail Sell Prob (Retail Sell Probability)
- **Type:** Decimal 0.000-1.000
- **Example:** 0.650
- **Meaning:** Probability (0-100%) that retail traders are selling (capitulating)
- **Calculation:** Based on fear index and elevated daily range
- **0.0-0.3 = Retail holding/buying (bearish)**
- **0.3-0.6 = Some retail selling (mixed)**
- **0.6-1.0 = Heavy retail selling (excellent contrarian signal)**

### Nash Inst Action (Institutional Nash Equilibrium Action)
- **Type:** Text (BUY / HOLD / SELL)
- **Example:** BUY
- **Meaning:** What is the optimal action for institutional investors given retail behavior?
- **BUY = Best response is to accumulate (bullish)**
- **HOLD = Neutral position taken (mixed signal)**
- **SELL = Best response is to distribute (bearish)**
- **Key Signal:** When Institutions = BUY and Retail = SELL = Classic reversal setup

### Nash Retail Action (Retail Nash Equilibrium Action)
- **Type:** Text (BUY / HOLD / SELL)
- **Example:** SELL
- **Meaning:** What is the optimal action for retail traders given institutional behavior?
- **BUY = Retail buying (bullish sentiment)**
- **HOLD = Retail neutral**
- **SELL = Retail in capitulation (bearish short-term, bullish medium-term)**

### Nash Inst Payoff (Institutional Player Payoff Value)
- **Type:** Integer 0-8
- **Example:** 5
- **Meaning:** How satisfied is the institutional player with this Nash equilibrium?
- **Scale:**
  - 0 = Terrible outcome
  - 4 = Neutral outcome
  - 8 = Excellent outcome
- **Higher values = Better equilibrium for institutions**

### Prisoners Dilemma Score
- **Type:** 0-100
- **Example:** 72.4
- **Meaning:** Cooperation score - how close is the outcome to mutual BUY scenario?
- **Calculation:** (Actual Combined Payoff / Maximum Collective Payoff) × 100
- **0-33 = Low cooperation (conflict, one player wins, other loses)**
- **34-66 = Medium cooperation (mixed signals)**
- **67-100 = High cooperation (alignment, both players benefit)**
- **Interpretation:** Higher score = Both institutions and retail aligned toward buying = Strongest reversal signal

### Info Asymmetry Score
- **Type:** 0-100
- **Example:** 68.3
- **Meaning:** Information advantage score - how different are institutional vs retail behaviors?
- **Calculation:** KL-Divergence between probability distributions (0-50 range, scaled to 0-100)
- **0-33 = Low asymmetry (institutions + retail thinking similarly, no edge)**
- **34-66 = Medium asymmetry (some deviation)**
- **67-100 = High asymmetry (institutions clearly acting differently, strong smart money edge)**
- **Investment Implication:** High score = Institutions have information advantage = Follow them

### Game Theory Score
- **Type:** 0-100
- **Example:** 72.2
- **Meaning:** Composite game theory score combining all behavioral signals
- **Formula:**
  ```
  GT Score = min(
    (Nash Inst Action Bonus: 40 if BUY, 20 if HOLD, 0 if SELL)
    + (Retail Fear × 30)
    + (Prisoners Dilemma ÷ 100 × 20)
    + (Info Asymmetry ÷ 100 × 10),
    100
  )
  ```
- **Interpretation:**
  - 75+ = Institutions clearly buying while retail panics (excellent setup)
  - 60-74 = Bullish behavioral setup
  - 45-59 = Mixed behavioral signals
  - 30-44 = Bearish behavioral signals
  - <30 = Institutions selling or unclear

---

## **SECTION 7: Final Investment Rating**

### Conviction Score
- **Type:** 0-100
- **Example:** 62.7
- **Meaning:** Final combined recommendation strength (0-100 scale)
- **Formula:**
  ```
  Conviction Score = (Quant Score × 0.60) + (Game Theory Score × 0.40)
  ```
- **Why 60/40?**
  - Quantitative analysis is more analytical/factual (60% weight)
  - Game theory is more interpretive/behavioral (40% weight)
- **Interpretation:**
  - 75-100 = STRONG BUY (multiple factors aligned)
  - 60-74 = BUY (most factors favorable)
  - 45-59 = WATCH (mixed signals, monitor)
  - 30-44 = WEAK (few positive factors)
  - 0-29 = AVOID (most factors unfavorable)

### Signal
- **Type:** Text (Investment Recommendation)
- **Example:** BUY
- **Options:**
  - **STRONG BUY** (Conviction 75+)
  - **BUY** (Conviction 60-74)
  - **WATCH** (Conviction 45-59)
  - **WEAK** (Conviction 30-44)
  - **AVOID** (Conviction <30)

### Rating
- **Type:** Text (Star Rating)
- **Example:** 4/5
- **Options:**
  - **5/5 ⭐⭐⭐⭐⭐** (STRONG BUY) - Exceptional opportunity
  - **4/5 ⭐⭐⭐⭐** (BUY) - Good opportunity
  - **3/5 ⭐⭐⭐** (WATCH) - Monitor carefully
  - **2/5 ⭐⭐** (WEAK) - Marginal opportunity
  - **1/5 ⭐** (AVOID) - Not recommended

---

## 📊 FULL EXAMPLE INTERPRETATION

```
Rank: 1
Ticker: ITC
Current Price (INR): 415.50
52W High (INR): 553.75
52W Low (INR): 285.20
Drop from 52W High (%): 25.1
EMA10: 405.32
EMA10 Slope: 0.0234

RSI: 52.3 (bullish - in optimal 40-60 zone)
MACD Bullish: True (momentum confirming)
Momentum Score: 56.3 (moderate momentum building)
Mean Reversion Score: 62.8 (oversold, good reversal)
Volatility Score: 45.2 (elevated, but acceptable)
Volume Score: 68.5 (decent volume confirmation)
Statistical Edge Score: 54.1 (moderate edge)
Quant Score: 56.3 (OVERALL: Good quantitative setup)

Inst Buy Prob: 0.580 (moderate institutional buying)
Inst Sell Prob: 0.125 (low institutional selling)
Retail Fear Index: 0.680 (high retail panic)
Retail Sell Prob: 0.650 (retail capitulating)
Nash Inst Action: BUY (institutions best response)
Nash Retail Action: SELL (retail best response)
Nash Inst Payoff: 5 (good institutional outcome)
Prisoners Dilemma Score: 72.4 (allies aligned toward recovery)
Info Asymmetry Score: 68.3 (institutions have clear edge)
Game Theory Score: 72.2 (OVERALL: Bullish behavioral setup)

Conviction Score: 62.7
Signal: BUY ✓
Rating: 4/5 ⭐⭐⭐⭐
```

### **What This Means:**
- ITC is down 25% from recent highs (oversold)
- Technical indicators show early reversal (uptrend starting)
- Quantitative analysis = Decent opportunity (score 56.3)
- Game theory = Institutions buying into retail panic (score 72.2)
- **Combined conviction = 62.7 = BUY with 4/5 confidence**
- **Trading interpretation:** Consider entry on next 5-10% dip; upside target: 500-550 (recovery toward highs)
- **Risk management:** Stop-loss below 390 (52W low support + buffer)

---

## 🎯 QUICK REFERENCE THRESHOLDS

| Metric | Poor | Fair | Good | Excellent |
|--------|------|------|------|-----------|
| Drop % | <10% | 10-20% | 20-35% | 35%+ |
| EMA10 Slope | <0 | 0-0.01 | 0.01-0.05 | >0.05 |
| RSI | <30 or >70 | 30-40 or 60-70 | 40-60 | 50-60 |
| Quant Score | <30 | 30-50 | 50-70 | 70-100 |
| GT Score | <30 | 30-50 | 50-70 | 70-100 |
| Conviction | <30 | 30-45 | 45-65 | 65-100 |
| Retail Fear | <0.3 | 0.3-0.5 | 0.5-0.8 | >0.8 |

---

## ❓ COMMON QUESTIONS

**Q: Which column should I focus on most?**
A: Start with "Conviction Score" (final recommendation), then drill into "Signal" and "Rating". Verify with "Quant Score" and "Game Theory Score" for confidence.

**Q: What's a good entry point?**
A: When "Drop from 52W High" is 20-40% AND "EMA10 Slope" > 0 AND "Conviction Score" > 55.

**Q: When should I exit?**
A: When "EMA10 Slope" turns negative OR "Conviction Score" drops below 40 OR "RSI" exceeds 70.

**Q: Why does "Retail Sell Prob" being HIGH mean it's a GOOD signal?**
A: Contrarian principle: When retail traders panic-sell (capitulation), institutions step in to buy cheap. High retail fear = institutions accumulating = price recovery likely.

**Q: Should I follow "Nash Inst Action" or "Conviction Score"?**
A: Use "Conviction Score" for overall decision (combines both quant + game theory). Use "Nash Inst Action" to verify institutional alignment with quant setup.

**Q: When do I see 0 matches?**
A: When no stocks meet ALL criteria simultaneously. This is normal (markets are efficient). Lower "Drop from 52W High" filter or reduce "Conviction Score" threshold.

---

**Document Version:** 2.0  
**Last Updated:** February 26, 2026  
**Compatible With:** nsequant.py v1.0+
