# 🛠️ Debug & Diagnostic Files

## **Development/Testing Tools - SAFE TO DELETE**

These files were used during development for debugging and testing. They are **not required** for production operation.

### **Files in this folder:**

1. **debug_indicators.py**  
   - Tests technical indicator calculations
   - Validates RSI, MACD, Bollinger Bands formulas
   - Development artifact

2. **debug_yfinance.py**  
   - Debugs yfinance library connectivity
   - Not used in production (we use Kite API)
   - Testing tool

3. **diagnosetoken.py**  
   - Diagnoses Kite Connect token issues
   - Troubleshooting utility
   - One-time use

4. **check_nifty_token.py**  
   - Verifies Nifty 50 instrument token
   - One-time validation
   - No longer needed

5. **test_excel.py**  
   - Tests Excel export functionality
   - Development testing file
   - Not part of production system

---

## **Recommendation:**

✅ **SAFE TO DELETE ALL FILES IN THIS FOLDER**

These files served their purpose during development and are no longer needed for production operations.

If you encounter issues in the future, you can always recover them from version control or backups.

---

**Action:** Delete this entire folder when ready to clean up the project.
