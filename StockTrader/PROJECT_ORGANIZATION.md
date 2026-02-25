# 📂 StockTrader - Organized Project Structure

**Project reorganized on:** February 11, 2026

---

## **📊 Folder Organization Summary**

| Folder | Files | Purpose | Action |
|--------|-------|---------|--------|
| **Active_Production/** | 6 files (29%) | Core system - actively used | ⭐ **KEEP** |
| **Archive/** | 4 files (19%) | Alternative scanners - not in use | 📦 Keep for reference |
| **Utilities/** | 5 files (24%) | Setup & maintenance tools | 🔧 Keep for maintenance |
| **Debug_ToRemove/** | 5 files (24%) | Debug/test files | 🗑️ **Safe to delete** |
| **Root folder** | Config & docs | Configuration files & documentation | ✅ Essential |

**Total Python files:** 20 scripts

---

## **🚀 Quick Start**

### **Run the main scanner:**
```powershell
cd Active_Production
python automated_scanner.py
```

### **Run options analyzer:**
```powershell
cd Active_Production
python nifty_oi_tracker.py
```

### **Generate new Kite token:**
```powershell
cd Utilities
python generate_token.py
```

---

## **📁 Detailed Folder Structure**

### **🟢 Active_Production/** - Core System
**Status:** ⭐ **PRODUCTION USE - DO NOT DELETE**

Main automated scanner system with all active components:
- `automated_scanner.py` - Main entry point
- `advanced_scanner.py` - Strategy engine (10+ strategies)
- `kite_stock_scanner.py` - Base scanner class
- `telegram_notifier.py` - Telegram notifications
- `enhanced_html_generator.py` - HTML dashboard generator
- `nifty_oi_tracker.py` - Options chain analyzer

**See:** [Active_Production/README.md](Active_Production/README.md)

---

### **📦 Archive/** - Alternative Scanners
**Status:** 📂 **ARCHIVED - Not in active use**

Alternative scanner implementations kept for reference:
- `Profitable_strategy_scanner.py` - Premium strategies (redundant)
- `closing_momentum_scanner.py` - Momentum/gap scanner (redundant)
- `eth_swing_screener.py` - ETF screener (uses yfinance)
- `check_tomorrow.py` - Analysis script

**See:** [Archive/README.md](Archive/README.md)

---

### **🔧 Utilities/** - Setup & Tools
**Status:** 🔧 **KEEP - Needed for maintenance**

Setup, configuration, and standalone analysis tools:
- `generate_token.py` - Kite token generation (run daily/weekly)
- `setup_telegram.py` - Telegram bot setup (run once)
- `quick_token.py` - Quick token generator
- `analyze_trade.py` - Iron Butterfly calculator
- `_exchange_request_token.py` - Token exchange utility

**See:** [Utilities/README.md](Utilities/README.md)

---

### **🗑️ Debug_ToRemove/** - Debug Files
**Status:** ❌ **SAFE TO DELETE**

Development/debugging tools no longer needed:
- `debug_indicators.py` - Indicator testing
- `debug_yfinance.py` - yfinance debugging
- `diagnosetoken.py` - Token diagnostics
- `check_nifty_token.py` - Token verification
- `test_excel.py` - Excel testing

**See:** [Debug_ToRemove/README.md](Debug_ToRemove/README.md)

---

### **📄 Root Folder** - Configuration & Results
**Status:** ✅ **ESSENTIAL - Keep in root**

#### Configuration Files:
- `.env` - Main environment variables
- `.env.telegram` - Telegram bot credentials
- `kite_credentials.txt` - Kite API credentials
- `requirements.txt` - Python dependencies
- `load_env.ps1` - Environment loader

#### Results & Output:
- `scanner_results.html` - Latest scan results
- `nifty_oi_tracker.html` - Options analysis
- `strategies_*.csv` - Historical results (7 files)
- `scanner_automation.log` - System logs
- `Results/` - Results archive folder

#### Documentation:
- `README.md` - This file
- `AUTOMATION_FLOW.txt` - Automation workflow
- `QUICK_START.md` - Getting started guide
- `TELEGRAM_SETUP_GUIDE.md` - Telegram setup
- `TRADING_CONCEPTS_GUIDE.md` - Trading concepts
- `REFRESH_FEATURE_GUIDE.md` - Auto-refresh feature
- `AUTO_REFRESH_FEATURE_SUMMARY.md` - Feature summary
- `OI_TRACKER_UPDATE.md` - OI tracker updates

#### Scripts:
- `run_scanner.bat` - Windows batch launcher

---

## **🔄 System Architecture**

```
┌─────────────────────────────────────────────────────────┐
│                  STOCKTRADER SYSTEM                     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │   Active_Production/               │
        │   automated_scanner.py (MAIN)     │
        └───────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    ┌────────┐  ┌────────┐  ┌──────────┐
    │  Kite  │  │Scanner │  │ HTML Gen │
    │  API   │  │Engine  │  │          │
    └────────┘  └────────┘  └──────────┘
         │           │            │
         └───────────┴────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Telegram Notifier   │
         │   + CSV/HTML Output   │
         └───────────────────────┘
```

---

## **⚙️ Dependencies**

Install required packages:
```powershell
pip install -r requirements.txt
```

Key dependencies:
- `kiteconnect` - Zerodha Kite API
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `requests` - HTTP requests (Telegram)
- `ta` - Technical analysis library

---

## **🔐 Setup Checklist**

- [ ] Install Python dependencies: `pip install -r requirements.txt`
- [ ] Configure Kite API credentials in `kite_credentials.txt`
- [ ] Generate access token: `cd Utilities && python generate_token.py`
- [ ] Setup Telegram bot: `cd Utilities && python setup_telegram.py`
- [ ] Configure `.env` and `.env.telegram` files
- [ ] Test run: `cd Active_Production && python automated_scanner.py`

---

## **📞 Support & Documentation**

Each folder contains a detailed README explaining its contents:
- [Active_Production/README.md](Active_Production/README.md) - Production files
- [Archive/README.md](Archive/README.md) - Archived scanners
- [Utilities/README.md](Utilities/README.md) - Setup utilities
- [Debug_ToRemove/README.md](Debug_ToRemove/README.md) - Debug files

---

## **🧹 Cleanup Guide**

### **Immediately Safe to Delete:**
```powershell
# Delete debug files (recommended)
Remove-Item -Recurse -Force "Debug_ToRemove"
```

### **Can Delete After Review:**
```powershell
# Delete archived scanners if you won't use them
Remove-Item -Recurse -Force "Archive"

# Delete old CSV results (keep last 2-3 for reference)
Remove-Item "strategies_20260211_*.csv"
```

### **Never Delete:**
- `Active_Production/` folder
- Configuration files (`.env`, `kite_credentials.txt`)
- `requirements.txt`
- Documentation markdown files

---

**Last Updated:** February 11, 2026  
**Organization Version:** 1.0  
**Total Python Scripts:** 20 files organized into 4 categories
