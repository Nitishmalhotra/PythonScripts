# 🟢 Active Production Files

## **Core System Components - DO NOT DELETE**

These are the **essential files** that power the automated stock scanning system.

### **Files in this folder:**

1. **automated_scanner.py** ⭐  
   - **Main entry point** for the entire system
   - Orchestrates all scanning operations
   - Manages automation and scheduling
   - Coordinates output generation

2. **advanced_scanner.py** ⭐  
   - Core strategy engine with 10+ trading strategies
   - Strategies: Momentum Breakout, RSI Bounce, MACD Crossover, Bollinger Squeeze, Volume Surge, EMA Crossover, etc.
   - Extends KiteStockScanner base class

3. **kite_stock_scanner.py** ⭐  
   - Foundation class for all scanners
   - Kite Connect API wrapper
   - Fetches market data from NSE
   - Calculates technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)

4. **telegram_notifier.py** ⭐  
   - Telegram bot integration
   - Sends real-time trading alerts
   - Shares CSV reports and analysis
   - Notification management

5. **enhanced_html_generator.py** ⭐  
   - Generates interactive HTML dashboards
   - Dark mode interface
   - Market sentiment analysis
   - Risk/reward visualization
   - Export functionality

6. **nifty_oi_tracker.py** 🎯  
   - Options chain analyzer
   - Open Interest (OI) analysis
   - Put-Call Ratio (PCR) calculations
   - Iron Butterfly strategy recommendations
   - Generates separate HTML tracker

---

## **Execution Flow:**

```
automated_scanner.py (MAIN)
    ├─> kite_stock_scanner.py (Initialize API, fetch data)
    ├─> advanced_scanner.py (Run strategies)
    ├─> enhanced_html_generator.py (Generate reports)
    └─> telegram_notifier.py (Send alerts)

nifty_oi_tracker.py (Independent module for options)
```

---

## **To Run the System:**

```powershell
cd Active_Production
python automated_scanner.py
```

For options analysis:
```powershell
python nifty_oi_tracker.py
```

---

**⚠️ CRITICAL:** These files are interdependent. Do not delete or move individually.
