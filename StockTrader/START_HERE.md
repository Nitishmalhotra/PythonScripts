# 🚀 START HERE - StockTrader Quick Setup

**Last Updated:** February 11, 2026

---

## **📂 Clean Folder Structure**

```
StockTrader/
├── 🟢 Active_Production/    ← Main system files (6 files)
├── 🔧 Utilities/            ← Setup tools (5 files)
├── 📊 Results/              ← Output files (auto-generated)
├── 📚 Documentation/        ← All guides & docs
├── 📦 Archive/              ← Old scanners (not used)
├── 🗑️  Debug_ToRemove/      ← Can delete
│
├── .env                     ← Environment variables
├── .env.telegram           ← Telegram credentials
├── kite_credentials.txt    ← Kite API keys
├── requirements.txt        ← Python dependencies
└── run_scanner.bat         ← Quick launcher
```

---

## **⚡ Quick Start (3 Steps)**

### **Step 1: Install Dependencies**
```powershell
pip install -r requirements.txt
```

### **Step 2: Configure API Credentials**

Edit `kite_credentials.txt`:
```
API_KEY=your_api_key
ACCESS_TOKEN=your_access_token
USER_ID=your_user_id
```

Edit `.env.telegram`:
```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### **Step 3: Run Scanner**

**Option A - Simple (Windows):**
```powershell
.\run_scanner.bat
```

**Option B - Direct (Any OS):**
```powershell
cd Active_Production
python automated_scanner.py
```

---

## **🔑 Generate New Kite Token (Daily/Weekly)**

```powershell
cd Utilities
python generate_token.py
```

Follow the prompts to get a new access token.

---

## **📊 View Results**

After running, check:
- **HTML Dashboard:** `Results/scanner_results.html`
- **CSV Export:** `Results/strategies_YYYYMMDD_HHMMSS.csv`
- **Logs:** `Results/scanner_automation.log`
- **Telegram:** Check your bot for notifications

---

## **📚 Need Help?**

| Topic | File |
|-------|------|
| **Quick commands** | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |
| **Full organization** | [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md) |
| **Setup guides** | [Documentation/](Documentation/) |
| **Telegram setup** | [Documentation/TELEGRAM_SETUP_GUIDE.md](Documentation/TELEGRAM_SETUP_GUIDE.md) |
| **Trading concepts** | [Documentation/TRADING_CONCEPTS_GUIDE.md](Documentation/TRADING_CONCEPTS_GUIDE.md) |

---

## **🧹 Optional Cleanup**

**Delete debug files:**
```powershell
Remove-Item -Recurse -Force Debug_ToRemove
```

**Delete archived scanners:**
```powershell
Remove-Item -Recurse -Force Archive
```

---

## **🔧 Troubleshooting**

### **Token expired?**
```powershell
cd Utilities
python generate_token.py
```

### **Module not found?**
```powershell
pip install -r requirements.txt
```

### **Telegram not working?**
```powershell
cd Utilities
python setup_telegram.py
```

---

**That's it! You're ready to scan. 🎯**
