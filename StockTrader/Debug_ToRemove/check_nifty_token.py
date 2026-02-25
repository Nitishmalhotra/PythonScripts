import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_file = Path.cwd() / '.env'
print(f"Current directory: {Path.cwd()}")
print(f"Looking for .env at: {env_file}")
print(f"File exists: {env_file.exists()}")

if env_file.exists():
    load_dotenv(env_file)
    
api_key = os.getenv('KITE_API_KEY')
access_token = os.getenv('KITE_ACCESS_TOKEN')

print(f"\nLoaded credentials:")
print(f"API_KEY: {api_key}")
print(f"ACCESS_TOKEN: {access_token}")
print(f"Both set: {bool(api_key and access_token)}")

from kiteconnect import KiteConnect

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

print("Searching for NIFTY tokens...")
instruments = kite.instruments('NSE')
nifty_tokens = [inst for inst in instruments if 'NIFTY' in inst.get('tradingsymbol', '').upper()][:10]
for inst in nifty_tokens:
    print(f"{inst['tradingsymbol']}: {inst['instrument_token']}")
