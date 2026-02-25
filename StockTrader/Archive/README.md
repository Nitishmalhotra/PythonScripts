# 📦 Archived Alternative Scanners

## **Alternative Implementations - Not in Active Use**

These files are alternative scanner implementations that are **not currently used** in production. They have been archived because their functionality is either redundant or uses different data sources.

### **Files in this folder:**

1. **Profitable_strategy_scanner.py**  
   - Alternative premium scanner with top 10 profitable strategies
   - Uses weighted scoring system
   - **Redundant:** `advanced_scanner.py` already implements these strategies
   - Extends KiteStockScanner

2. **closing_momentum_scanner.py**  
   - Analyzes closing strength and next-day gap behavior
   - Focuses on momentum leading to gap-ups
   - **Redundant:** Functionality covered by `advanced_scanner.py`
   - Useful if you want specialized momentum-only scanning

3. **eth_swing_screener.py**  
   - ETF/iShares swing trading screener
   - **Different data source:** Uses yfinance, not Kite Connect
   - Analyzes US market ETFs (SPY, QQQ, IWM, etc.)
   - Standalone use case for US market analysis

4. **check_tomorrow.py**  
   - Purpose unclear (needs review)
   - Likely a one-time analysis script
   - Kept for reference

---

## **When to Use These Files:**

### **If you want to:**
- **Test alternative strategies** → Run `Profitable_strategy_scanner.py`
- **Focus only on gap strategies** → Run `closing_momentum_scanner.py`
- **Analyze US ETFs** → Run `eth_swing_screener.py` (requires yfinance)

### **Current Status:**
These scanners are **functional** but **not integrated** into the main `automated_scanner.py` workflow.

---

## **Recommendation:**

📂 **Keep archived** for potential future use  
⚠️ Can be deleted if you're confident you won't need alternative implementations  
🔄 Can be re-integrated if needed (they extend KiteStockScanner)

---

**Note:** If you want to use any of these, they may need updates to work with current dependencies and API structure.
