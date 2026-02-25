import yfinance as yf
import pandas as pd
from eth_swing_screener import ETFSwingScreener

# Test the screener directly
screener = ETFSwingScreener(['IVV'], period='6mo', interval='1d')

# Download data for IVV
data = yf.download('IVV', period='6mo', interval='1d', progress=False)

# Handle MultiIndex columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Standardize column names to lowercase
data.columns = [str(col).lower() for col in data.columns]

print(f"Data columns: {list(data.columns)}")
print(f"Data shape: {data.shape}")
print(f"Data length: {len(data)}")

# Calculate indicators - using updated method
print("\nCalculating indicators...")
sma20_result = screener.calculate_sma(data['close'], 20)
data['SMA_20'] = sma20_result.values if hasattr(sma20_result, 'values') else sma20_result

sma50_result = screener.calculate_sma(data['close'], 50)
data['SMA_50'] = sma50_result.values if hasattr(sma50_result, 'values') else sma50_result

rsi_result = screener.calculate_rsi(data['close'], 14)
data['RSI_14'] = rsi_result.values if hasattr(rsi_result, 'values') else rsi_result

macd, macd_signal, macd_hist = screener.calculate_macd(data['close'])
data['MACD'] = macd.values if hasattr(macd, 'values') else macd
data['MACD_Signal'] = macd_signal.values if hasattr(macd_signal, 'values') else macd_signal

adx_result = screener.calculate_adx(data['high'], data['low'], data['close'], 14)
data['ADX_14'] = adx_result.values if hasattr(adx_result, 'values') else adx_result

print(f"\nLast 5 rows of indicators:")
print(data[['SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'ADX_14']].tail())

print(f"\nNaN counts:")
print(data[['SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'ADX_14']].isna().sum())

print(f"\nFirst non-NaN index for each column:")
for col in ['SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'ADX_14']:
    first_valid = data[col].first_valid_index()
    print(f"  {col}: {first_valid}")
