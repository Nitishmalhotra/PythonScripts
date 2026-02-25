# 🔧 Utilities & Setup Scripts

## **One-Time Setup and Standalone Tools**

These are utility scripts for **setup, configuration, and standalone analysis**. They are not part of the daily automated workflow but are needed for system maintenance.

### **Files in this folder:**

#### **Setup & Configuration:**

1. **generate_token.py** 🔑  
   - Generates Kite Connect access token
   - **Run:** Daily or when token expires
   - Interactive script guides you through token generation
   - Updates `kite_credentials.txt`

2. **setup_telegram.py** 📱  
   - Initial Telegram bot setup and configuration
   - **Run:** Once during initial setup
   - Tests bot connectivity
   - Helps configure `.env.telegram`

3. **quick_token.py** ⚡  
   - Quick alternative to `generate_token.py`
   - Faster token generation process
   - Same functionality, streamlined interface

#### **Analysis & Testing:**

4. **analyze_trade.py** 📊  
   - Standalone Iron Butterfly premium calculator
   - Uses Black-Scholes model
   - Manual options trade analysis
   - Independent of main scanner system

5. **_exchange_request_token.py** 🔄  
   - Token exchange utility
   - Converts request token to access token
   - Troubleshooting tool for authentication issues

---

## **Usage Guide:**

### **Daily Maintenance:**
```powershell
# If access token expires, run:
python generate_token.py
# or
python quick_token.py
```

### **Initial Setup:**
```powershell
# First time Telegram setup:
python setup_telegram.py
```

### **Manual Options Analysis:**
```powershell
# Analyze Iron Butterfly trades:
python analyze_trade.py
```

---

## **Recommendation:**

✅ **KEEP THESE FILES** - They are needed for system maintenance  
🔧 Use as needed for setup and troubleshooting  
📌 Not part of automated daily workflow  

---

**Note:** These utilities should be run manually when needed, not automated.
