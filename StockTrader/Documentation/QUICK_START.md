# 🚀 Quick Start - Telegram Automation

## ⚡ 5-Minute Setup

### 1️⃣ Install Package
```powershell
pip install requests
```

### 2️⃣ Run Setup Wizard
```powershell
python setup_telegram.py
```

The wizard will guide you through:
- Creating a Telegram bot
- Getting your chat ID
- Testing the connection
- Saving credentials

### 3️⃣ Test Automation
```powershell
python automated_scanner.py
```

You'll receive Telegram notifications with:
- 📊 Scan summary
- 🔥 High-priority signals
- 📈 Strategy breakdown
- 📄 CSV file (optional)

---

## 📱 Telegram Bot Commands

Send these to @BotFather to create your bot:

```
/newbot
My Stock Scanner Bot
mystock_scanner_bot
```

---

## 🕐 Schedule Automation

### Option 1: Quick Schedule (PowerShell)

```powershell
# Run this to create daily tasks at 9 AM, 12:30 PM, and 3:30 PM
$scriptPath = Get-Location
$batFile = Join-Path $scriptPath "run_scanner.bat"

# Pre-Market (9:00 AM)
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$batFile`"" -WorkingDirectory $scriptPath
$trigger = New-ScheduledTaskTrigger -Daily -At "09:00AM"
Register-ScheduledTask -TaskName "Stock Scanner - Pre Market" -Action $action -Trigger $trigger

# Mid-Day (12:30 PM)
$trigger2 = New-ScheduledTaskTrigger -Daily -At "12:30PM"
Register-ScheduledTask -TaskName "Stock Scanner - Mid Day" -Action $action -Trigger $trigger2

# End-of-Day (3:30 PM)
$trigger3 = New-ScheduledTaskTrigger -Daily -At "3:30PM"
Register-ScheduledTask -TaskName "Stock Scanner - End of Day" -Action $action -Trigger $trigger3

Write-Host "✅ Scheduled tasks created!"
```

### Option 2: Manual (Task Scheduler GUI)

1. Open Task Scheduler (`Win + R` → `taskschd.msc`)
2. Create Basic Task
3. Name: "Stock Scanner"
4. Trigger: Daily at 9:00 AM
5. Action: Start a program → `run_scanner.bat`
6. Done!

---

## 📋 Daily Checklist

- [ ] Check morning alerts (9 AM scan)
- [ ] Review high-priority signals
- [ ] Monitor mid-day updates (12:30 PM)
- [ ] Review end-of-day summary (3:30 PM)
- [ ] Check `scanner_automation.log` if needed

---

## 🎯 What You'll Receive

### Every Scan:
1. **Summary** - Total stocks, sentiment, top strategy
2. **High-Priority Alerts** - Best setups (low risk, high R:R)
3. **Strategy Breakdown** - Signals per strategy
4. **Top 3 Strategies** - Best 3-5 signals each
5. **CSV File** - Complete data (optional)

### Example Notification:
```
📊 Stock Scanner Alert
⏰ February 11, 2026 at 09:00 AM

📈 Scan Results:
• Total Stocks: 46
• Active Strategies: 10
• Market Sentiment: 🟢 BULLISH 76%
• Top Strategy: Swing Trading (17 stocks)

🔗 Check dashboard for details
```

---

## 🔧 Troubleshooting

### Bot not sending messages?
```powershell
# Test individually:
python telegram_notifier.py
```

### Scan failing?
```powershell
# Check logs:
Get-Content scanner_automation.log -Tail 20
```

### Task not running?
```powershell
# Check Task Scheduler:
# 1. Open Task Scheduler
# 2. Find your task
# 3. Right-click → Run
# 4. Check "Last Run Result"
```

---

## ⚙️ Customization

### Change notification frequency:
Edit `automated_scanner.py`:
```python
# Line ~180
scanner.run_full_automation(
    send_csv=False,     # Don't send CSV every time
    generate_html=True
)
```

### Change alert filters:
Edit `telegram_notifier.py`:
```python
# Line ~110 - Only super high-quality setups
priority_signals = signals_df[
    (signals_df['risk_level'] == 'LOW') &
    (signals_df['risk_reward'] >= 2.0) &  # Higher R:R
    (signals_df['volume_ratio'] > 2.5)    # Much higher volume
].head(3)  # Top 3 only
```

---

## 📞 Get Help

**Check files created:**
- `telegram_notifier.py` - Telegram integration
- `automated_scanner.py` - Main automation script
- `setup_telegram.py` - Setup wizard
- `run_scanner.bat` - Windows batch file
- `TELEGRAM_SETUP_GUIDE.md` - Full guide
- `.env` - Your credentials (created by wizard)

**Missing packages?**
```powershell
pip install -r requirements.txt
```

---

## 🎉 You're Ready!

Your automated stock scanner will:
- ✅ Run on schedule
- ✅ Scan Nifty 50 stocks
- ✅ Apply 13 trading strategies
- ✅ Send you Telegram alerts
- ✅ Generate HTML dashboard
- ✅ Save results to CSV

**Happy Trading! 📈**
