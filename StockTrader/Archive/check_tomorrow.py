from datetime import datetime, timedelta

today = datetime.now().date()
weekday = today.weekday()  # 0=Mon, 1=Tue, 2=Wed
days_ahead = 2 - weekday
if days_ahead <= 0:
    days_ahead += 7

expiry = today + timedelta(days=days_ahead)

print(f'Today: {today.strftime("%A, %B %d, %Y")}')
print(f'Tomorrow: {(today + timedelta(days=1)).strftime("%A, %B %d, %Y")}')
print(f'Weekly Expiry: {expiry.strftime("%A, %B %d, %Y")}')
print(f'Days to Expiry: {(expiry - today).days}')
print()
print("TOMORROW SCENARIO:")
print(f"- Expiry is in {(expiry - (today + timedelta(days=1))).days} day(s) from tomorrow")
