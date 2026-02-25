"""
Kite Connect Token Diagnostic & Fix
Checks token validity and guides you to fix it
"""

import os
import sys
from pathlib import Path
import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from kiteconnect import KiteConnect

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
env_file = PROJECT_ROOT / '.env'

print(f"Loading from: {env_file}")
print(f"File exists: {env_file.exists()}")

if env_file.exists():
    with open(env_file) as f:
        print("\n.env contents:")
        print(f.read())

load_dotenv(env_file)

api_key = os.getenv('KITE_API_KEY')
access_token = os.getenv('KITE_ACCESS_TOKEN')

print(f"\nLoaded API_KEY: {api_key}")
print(f"Loaded ACCESS_TOKEN: {access_token}")

if not api_key or not access_token:
    print("\n❌ Credentials not loaded!")
    sys.exit(1)

print("\nTesting Kite connection...")
try:
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    
    instruments = kite.instruments('NSE')
    print(f"✅ SUCCESS! Fetched {len(instruments)} instruments")
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

# Test the token
print("="*80)
print("🧪 TESTING ACCESS TOKEN")
print("="*80)
print()

try:
    kite = KiteConnect(api_key="u664cda77q2cf7ft")
    kite.set_access_token("CaFhMymwgB9benK3htP3T0Lt3Yy1dpGm")
    
    print("Testing token by fetching profile...")
    profile = kite.profile()
    
    print()
    print("="*80)
    print("✅ TOKEN IS VALID!")
    print("="*80)
    print()
    print(f"User ID: {profile['user_id']}")
    print(f"User Name: {profile['user_name']}")
    print(f"Email: {profile['email']}")
    print(f"Broker: {profile['broker']}")
    print()
    print("Your token is working correctly.")
    print("The error might be temporary or rate-limiting.")
    print()
    print("="*80)
    print("💡 SUGGESTIONS:")
    print("="*80)
    print("1. Wait 1-2 minutes and try again (rate limit)")
    print("2. Reduce number of stocks being scanned")
    print("3. Add delays between API calls")
    print()
    
except Exception as e:
    error_msg = str(e).lower()
    
    print()
    print("="*80)
    print("❌ TOKEN TEST FAILED")
    print("="*80)
    print()
    print(f"Error: {e}")
    print()
    
    if 'incorrect' in error_msg or 'invalid' in error_msg or 'token' in error_msg:
        print("="*80)
        print("🔄 TOKEN EXPIRED OR INVALID")
        print("="*80)
        print()
        print("Your access token has expired or is invalid.")
        print()
        print("Kite Connect access tokens expire:")
        print("  • Daily at 3:30 PM IST")
        print("  • After being unused for some time")
        print()
        print("="*80)
        print("🔧 FIX: Generate New Token")
        print("="*80)
        print()
        print("Step 1: Run the token generator")
        print("  python simple_token_generator.py")
        print()
        print("Step 2: Update the scanner")
        print("  python update_scanner_token.py")
        print()
        print("Step 3: Run your scanner again")
        print("  python kite_stock_scanner.py")
        print()
        
    elif 'network' in error_msg or 'connection' in error_msg:
        print("="*80)
        print("🌐 NETWORK ERROR")
        print("="*80)
        print()
        print("Check your internet connection and try again.")
        print()
        
    else:
        print("="*80)
        print("❓ UNKNOWN ERROR")
        print("="*80)
        print()
        print("Try generating a fresh token:")
        print("  python simple_token_generator.py")
        print()

print("="*80)
print()

# Check current time
now = datetime.now()
ist_hour = (now.hour + 5) % 24  # Rough IST conversion
ist_minute = (now.minute + 30) % 60

print("Current time (approximate IST):", f"{ist_hour:02d}:{ist_minute:02d}")
print()

if ist_hour >= 15 and ist_minute >= 30:
    print("⚠️  NOTE: Market hours are over (after 3:30 PM IST)")
    print("   Access tokens expire at 3:30 PM IST daily")
    print("   Generate a new token tomorrow morning")
elif ist_hour < 9:
    print("⚠️  NOTE: Markets haven't opened yet (before 9:15 AM IST)")
    print("   You can generate token, but limited data available")

print()
print("="*80)