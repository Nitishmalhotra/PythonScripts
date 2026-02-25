import numpy as np
import math

# Current market data
current_price = 25935.15
iv = 0.94 / 100  # 0.94%
r = 0.06  # risk-free rate
time_to_expiry = 6 / 365  # 6 days

def norm_cdf(x):
    return (1 + math.erf(x / np.sqrt(2))) / 2

def black_scholes(S, K, T, r, sigma):
    if T <= 0 or sigma == 0:
        return max(S - K, 0), max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    put = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    
    return call, put

# Iron Butterfly strikes
atm_strike = 25900
upper_strike = 26000
lower_strike = 25800

# Calculate premiums - FIXED EXTRACTION
short_call, short_put = black_scholes(current_price, atm_strike, time_to_expiry, r, iv)
long_call, long_put_upper = black_scholes(current_price, upper_strike, time_to_expiry, r, iv)
_, long_put_lower = black_scholes(current_price, lower_strike, time_to_expiry, r, iv)  # FIX: Extract PUT (2nd return)

print("=" * 70)
print("🔧 IRON BUTTERFLY ANALYSIS - FIXED & VERIFIED")
print("=" * 70)
print(f"\nCurrent Nifty Price: ₹{current_price:.2f}")
print(f"Time to Expiry: 6 days")
print(f"IV: 0.94%")
print(f"\n✅ CORRECTED STRIKES & PREMIUMS:")
print("-" * 70)
print(f"SHORT CALL @ ₹{atm_strike}: Premium = ₹{short_call:.2f}")
print(f"SHORT PUT @ ₹{atm_strike}: Premium = ₹{short_put:.2f}")
print(f"LONG CALL @ ₹{upper_strike}: Premium = ₹{long_call:.2f}")
print(f"LONG PUT @ ₹{lower_strike}: Premium = ₹{long_put_lower:.2f} (NOW CORRECT!)")

# Net credit calculation
wing_width = 100
net_credit = short_call + short_put - long_call - long_put_lower

print(f"\n✅ CORRECTED CALCULATIONS:")
print("-" * 70)
print(f"Net Credit = (₹{short_call:.2f} + ₹{short_put:.2f}) - (₹{long_call:.2f} + ₹{long_put_lower:.2f})")
print(f"Net Credit = ₹{net_credit:.2f} (in points) ✓")
print(f"\nWing Width = ₹{wing_width} points")
print(f"Max Loss per lot (in points) = ₹{wing_width:.2f} - ₹{net_credit:.2f} = ₹{wing_width - net_credit:.2f}")

# Per point value for Nifty
point_value = 20

print(f"\n✅ RUPEE VALUES (per lot):")
print("-" * 70)
net_credit_rupees = net_credit * point_value
max_loss_per_lot = (wing_width - net_credit) * point_value
max_profit_per_lot = net_credit * point_value

print(f"Net Credit in Rupees = ₹{net_credit:.2f} × {point_value} = ₹{net_credit_rupees:.2f}")
print(f"Max Loss per lot = ₹{wing_width - net_credit:.2f} × {point_value} = ₹{max_loss_per_lot:.2f}")
print(f"Max Profit per lot = ₹{max_profit_per_lot:.2f}")

# Verify with user's numbers
capital_required = 87863
profit = 5105
loss = 744

print(f"\n📊 USER PROVIDED NUMBERS - VERIFICATION:")
print("-" * 70)
print(f"Capital Required: ₹{capital_required}")
print(f"Expected Profit: ₹{profit}")
print(f"Expected Loss: ₹{loss}")

# Calculate implied lots
if max_loss_per_lot > 0:
    implied_lots_from_capital = capital_required / max_loss_per_lot
    actual_lots = round(implied_lots_from_capital)
    
    print(f"\nCalculated from Capital:")
    print(f"  Implied lots = ₹{capital_required} / ₹{max_loss_per_lot:.2f} = {implied_lots_from_capital:.2f}")
    print(f"  → Rounded to {actual_lots} lots")
    
    # Check if profit/loss match calculation
    expected_profit_total = net_credit_rupees * actual_lots
    expected_loss_total = max_loss_per_lot * actual_lots
    
    implied_profit_per_lot = profit / actual_lots if actual_lots > 0 else 0
    implied_loss_per_lot = loss / actual_lots if actual_lots > 0 else 0
    
    print(f"\n💰 PROFIT/LOSS ANALYSIS:")
    print(f"  {'Per Lot':<30} {'User Provided':<20} {'Calculated':<20}")
    print(f"  {'-' * 70}")
    print(f"  {'Profit':<30} ₹{implied_profit_per_lot:<18.2f} ₹{max_profit_per_lot:<18.2f}")
    print(f"  {'Loss':<30} ₹{implied_loss_per_lot:<18.2f} ₹{max_loss_per_lot:<18.2f}")
    
    print(f"\n  {'For {0} lots:':<30}")
    print(f"  {'-' * 70}")
    print(f"  {'Total Profit':<30} ₹{profit:<18.0f} ₹{expected_profit_total:<18.2f}")
    print(f"  {'Total Loss':<30} ₹{loss:<18.0f} ₹{expected_loss_total:<18.2f}")

print("\n" + "=" * 70)
print("✅ ACCURACY ASSESSMENT:")
print("=" * 70)

# Calculate match accuracy
if max_loss_per_lot > 0 and actual_lots > 0:
    actual_profit_per_lot = profit / actual_lots
    actual_loss_per_lot = loss / actual_lots
    
    profit_match = abs(actual_profit_per_lot - max_profit_per_lot) < 1
    loss_match = abs(actual_loss_per_lot - max_loss_per_lot) < 10
    
    if profit_match:
        print(f"✓ Profit matches: YES (both ≈ ₹{max_profit_per_lot:.2f}/lot)")
    else:
        print(f"⚠ Profit mismatch: Expected ₹{max_profit_per_lot:.2f}/lot, User shows ₹{actual_profit_per_lot:.2f}/lot")
    
    if loss_match:
        print(f"✓ Loss matches: YES (within range)")
    else:
        print(f"⚠ Loss mismatch: User shows ₹{actual_loss_per_lot:.2f}/lot vs expected ₹{max_loss_per_lot:.2f}/lot")
    
    print(f"\n📌 INTERPRETATION:")
    print("-" * 70)
    if net_credit_rupees > 0:
        print(f"✓ This is a CREDIT SPREAD (receive ₹{net_credit_rupees:.2f} per lot initially)")
        print(f"✓ Max Profit: ₹{expected_profit_total:.2f} (if Nifty stays in ₹{lower_strike}-₹{upper_strike})")
        print(f"✓ Max Loss: ₹{expected_loss_total:.2f} (if Nifty breaks out ±₹{wing_width} points)")
        print(f"✓ Risk:Reward Ratio: 1:{expected_profit_total/expected_loss_total:.2f}")
    else:
        print(f"⚠ This is a DEBIT SPREAD (must pay ₹{abs(net_credit_rupees):.2f} per lot)")

print("\n" + "=" * 70)

