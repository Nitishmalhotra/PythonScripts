import yfinance as yf
import pandas as pd

# Test single ticker download
print("Testing single ticker download...")
data = yf.download('IVV', period='6mo', interval='1d', progress=False)

print(f"Data type: {type(data)}")
print(f"Columns type: {type(data.columns)}")
print(f"Is MultiIndex: {isinstance(data.columns, pd.MultiIndex)}")
print(f"Raw columns: {data.columns.tolist()}")
print(f"Column names: {list(data.columns)}")
print(f"Column names lowercase: {[str(col).lower() for col in data.columns]}")
print(f"\nFirst few rows:")
print(data.head())
print(f"\nData dtypes:")
print(data.dtypes)
