# 🤖 Telegram Bot Setup & Automation Guide

Complete guide to set up automated stock scanner with Telegram notifications.

---

## 📋 Part 1: Create Telegram Bot

### Step 1: Get Bot Token

1. **Open Telegram** on your phone or desktop
2. **Search for** `@BotFather` (the official bot creation bot)
3. **Start a chat** and send `/newbot`
4. **Follow the prompts:**
   - Choose a name for your bot (e.g., "Stock Scanner Bot")
   - Choose a username (must end in 'bot', e.g., "mystock_scanner_bot")
5. **Copy the bot token** - looks like this:
   ```
   1234567890:ABCdefGHIjklMNOpqrsTUVwxyz123456789
   ```
6. **Save this token** - you'll need it for configuration

### Step 2: Get Your Chat ID

**Method 1: Using Python Script**
```powershell
# 1. Start a chat with your bot (search for it in Telegram)
# 2. Send any message to your bot (like "hello")
# 3. Run this command (replace YOUR_BOT_TOKEN):

python -c "import requests; r=requests.get('https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates'); print(r.json())"
```

**Method 2: Using Browser**
1. Start a chat with your bot
2. Send any message
3. Open browser and go to:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
4. Look for `"chat":{"id": YOUR_CHAT_ID}`

**Method 3: Using IDBot**
1. Search for `@myidbot` on Telegram
2. Send `/getid` command
3. Copy the chat ID

### Step 3: Test Your Bot

```powershell
# Set environment variables (PowerShell)
$env:TELEGRAM_BOT_TOKEN = "your_bot_token_here"
$env:TELEGRAM_CHAT_ID = "your_chat_id_here"

# Test the bot
python telegram_notifier.py
```

You should receive a test message on Telegram!

---

## ⚙️ Part 2: Configure Automation

### Step 1: Create .env File

1. **Copy the template:**
   ```powershell
   Copy-Item .env.telegram .env
   ```

2. **Edit .env file** and fill in your credentials:
   ```bash
   # Telegram Configuration
   TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz123456789
   TELEGRAM_CHAT_ID=123456789
   
   # Kite Connect (use your existing credentials)
   KITE_API_KEY=your_api_key
   KITE_ACCESS_TOKEN=your_access_token
   ```

### Step 2: Install Required Package

```powershell
pip install requests
```

### Step 3: Test Automated Scanner

```powershell
# Run once manually to test
python automated_scanner.py
```

You should receive multiple notifications:
- ✅ Scan started
- 📊 Scan summary
- 🔥 High-priority signals
- 📈 Strategy breakdown
- 📄 Top signals per strategy
- ✅ Completion message

---

## 🕐 Part 3: Schedule Automation (Windows Task Scheduler)

### Schedule scanner to run automatically at market hours

### Option A: Using PowerShell (Quick Setup)

Run this script to create scheduled tasks:

```powershell
# Create 3 scheduled tasks: Pre-market, Mid-day, End-of-day

# 1. Pre-Market Scan (9:00 AM)
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"cd /d C:\Users\ankit\OneDrive\Desktop\Personal\Nitish\stock_dashboard\StockTrader && run_scanner.bat`""
$trigger = New-ScheduledTaskTrigger -Daily -At "09:00AM"
Register-ScheduledTask -TaskName "Stock Scanner - Pre Market" -Action $action -Trigger $trigger -Description "Run stock scanner before market opens"

# 2. Mid-Day Scan (12:30 PM)
$trigger2 = New-ScheduledTaskTrigger -Daily -At "12:30PM"
Register-ScheduledTask -TaskName "Stock Scanner - Mid Day" -Action $action -Trigger $trigger2 -Description "Run stock scanner mid-day"

# 3. End-of-Day Scan (3:30 PM)
$trigger3 = New-ScheduledTaskTrigger -Daily -At "3:30PM"
Register-ScheduledTask -TaskName "Stock Scanner - End of Day" -Action $action -Trigger $trigger3 -Description "Run stock scanner after market closes"

Write-Host "✅ Scheduled tasks created successfully!"
Write-Host "   - Stock Scanner - Pre Market (9:00 AM)"
Write-Host "   - Stock Scanner - Mid Day (12:30 PM)"
Write-Host "   - Stock Scanner - End of Day (3:30 PM)"
```

### Option B: Manual Setup via GUI

1. **Press** `Win + R`, type `taskschd.msc`, press Enter
2. **Click** "Create Basic Task" in the right panel
3. **Task Name:** "Stock Scanner - Daily"
4. **Trigger:** Daily, choose time (e.g., 9:00 AM for pre-market)
5. **Action:** Start a program
6. **Program/script:** Browse to `run_scanner.bat`
7. **Start in:** Browse to the StockTrader folder
8. **Finish** and enable "Run whether user is logged on or not"

### Multiple Scans Per Day

Recommended schedule for trading days:
- **9:00 AM** - Pre-market scan
- **12:30 PM** - Mid-day scan  
- **3:30 PM** - End-of-day scan

Create 3 separate tasks with different triggers.

---

## 📱 Part 4: Customize Notifications

### Edit `automated_scanner.py` to customize:

```python
# Line ~100: Choose what to send
scanner.run_full_automation(
    send_csv=True,      # Send CSV file? (True/False)
    generate_html=True  # Generate dashboard? (True/False)
)
```

### Filter Notifications

Edit `telegram_notifier.py` to customize filters:

```python
# Line ~110: High-priority filter criteria
priority_signals = signals_df[
    (signals_df['risk_level'] == 'LOW') &      # Only LOW risk
    (signals_df['risk_reward'] >= 2.0) &       # R:R >= 2.0
    (signals_df['volume_ratio'] > 2.0)         # High volume
].head(5)  # Top 5 signals only
```

---

## 🧪 Testing & Troubleshooting

### Test Individual Components

```powershell
# 1. Test Telegram bot
python telegram_notifier.py

# 2. Test scanner without notifications
python advanced_scanner.py

# 3. Test full automation
python automated_scanner.py

# 4. Test batch file
run_scanner.bat
```

### Check Logs

```powershell
# View automation log
Get-Content scanner_automation.log -Tail 50
```

### Common Issues

**Issue:** "No module named 'requests'"
```powershell
pip install requests
```

**Issue:** "Telegram bot not responding"
- Check bot token is correct
- Make sure you've sent at least one message to the bot
- Verify chat ID is correct

**Issue:** "Kite API rate limiting"
- Scanner has 3-second timeout per stock
- Reduce number of stocks or increase delays

**Issue:** "Task Scheduler not running"
- Check "Run whether user is logged on or not" is enabled
- Make sure .env file has correct credentials
- Check scanner_automation.log for errors

---

## 📊 What You'll Receive

### Telegram Notifications Include:

1. **Scan Summary** 📊
   - Total stocks scanned
   - Active strategies
   - Market sentiment
   - Top strategy

2. **High-Priority Alerts** 🔥
   - Low-risk setups
   - High reward:risk ratio
   - Strong volume confirmation

3. **Strategy Breakdown** 📈
   - Count per strategy
   - Top 3 strategies with signals

4. **CSV File** (optional) 📄
   - Complete scan results
   - All technical indicators

---

## 🎯 Advanced: On-Demand Scans

### Trigger scan manually via Telegram command

Add to `automated_scanner.py`:

```python
# At the end of file
def check_telegram_commands(bot_token, chat_id):
    """Check for /scan command from Telegram"""
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    response = requests.get(url)
    
    if response.status_code == 200:
        for update in response.json()['result']:
            if 'message' in update and 'text' in update['message']:
                if update['message']['text'] == '/scan':
                    return True
    return False

# In main():
if len(sys.argv) > 1 and sys.argv[1] == '--listen':
    # Listen for Telegram commands
    while True:
        if check_telegram_commands(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID):
            scanner.run_full_automation()
        time.sleep(60)  # Check every minute
```

Then schedule:
```powershell
# Run listener in background
python automated_scanner.py --listen
```

---

## 📅 Recommended Schedule

### For Active Traders:
- **9:00 AM** - Pre-market scan (prepare for day)
- **12:30 PM** - Mid-day scan (catch lunch-time moves)
- **3:30 PM** - End-of-day scan (review and plan)

### For Swing Traders:
- **3:45 PM** - Daily scan after market close

### For Long-Term Investors:
- **Weekly** - Every Sunday at 8:00 PM

---

## ✅ Quick Start Checklist

- [ ] Created Telegram bot via @BotFather
- [ ] Got bot token
- [ ] Found chat ID
- [ ] Created .env file with credentials
- [ ] Installed `requests` package
- [ ] Tested telegram_notifier.py
- [ ] Tested automated_scanner.py
- [ ] Created scheduled task(s)
- [ ] Verified notifications received
- [ ] Checked scanner_automation.log

---

## 🆘 Support

If you encounter issues:

1. Check `scanner_automation.log` file
2. Verify .env file has correct credentials
3. Test components individually
4. Ensure Python and packages are installed
5. Check Windows Task Scheduler history

---

**🎉 You're all set! Your automated stock scanner will now run on schedule and send you Telegram alerts!**
