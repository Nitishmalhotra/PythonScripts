import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR if SCRIPT_DIR.name == 'StockTrader' else SCRIPT_DIR.parent

def load_env_file(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"✅ Loading from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                print(f"   {key} = {value[:20]}...")

load_env_file(PROJECT_ROOT / '.env')
print(f"\nLoaded KITE_ACCESS_TOKEN: {os.getenv('KITE_ACCESS_TOKEN')}")

from kiteconnect import KiteConnect

api_key = "u664cda77q2cf7ft"
access_token = "FYoTEgNWQY7r4TG3vEd3ebriT3WZDENv"

try:
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    
    # Try to fetch instruments
    instruments = kite.instruments('NSE')
    print(f"✅ SUCCESS! Fetched {len(instruments)} instruments")
except Exception as e:
    print(f"❌ ERROR: {e}")
    print(f"   API Key: {api_key}")
    print(f"   Access Token: {access_token}")
