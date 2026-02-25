# 📊 Results & Output Files

All scanner outputs, logs, and generated reports.

## **Contents:**

### **HTML Reports:**
- `scanner_results.html` - Latest stock scanner results
- `nifty_oi_tracker.html` - Options OI analysis dashboard

### **CSV Exports:**
- `strategies_*.csv` - Historical strategy results with timestamps

### **Logs:**
- `scanner_automation.log` - System activity and error logs

---

## **Auto-Generated:**

These files are automatically created/updated when you run:
- `Active_Production/automated_scanner.py` → Creates CSV + HTML + logs
- `Active_Production/nifty_oi_tracker.py` → Creates OI HTML report

## **Cleanup:**

Old CSV files can be deleted periodically:
```powershell
# Keep only last 3 results
Get-ChildItem "strategies_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -Skip 3 | Remove-Item
```

---

**Location:** New outputs will be saved here automatically.
