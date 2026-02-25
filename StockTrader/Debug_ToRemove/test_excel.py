import pandas as pd
from datetime import datetime

# Create test data
test_data = {
    'symbol': ['RELIANCE', 'TCS', 'INFY'],
    'close': [2500.50, 3400.25, 1450.75],
    'rsi_14': [55.3, 62.1, 58.9]
}

df = pd.DataFrame(test_data)

print(f"Test DataFrame:\n{df}")
print(f"\nShape: {df.shape}")
print(f"Empty: {df.empty}")

# Try writing to Excel
excel_file = f"test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

try:
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        print(f"\nWriting to {excel_file}...")
        df.to_excel(writer, sheet_name='Test_Sheet', index=False)
        print("✓ Write successful")
    
    # Try reading back
    df_read = pd.read_excel(excel_file, sheet_name='Test_Sheet')
    print(f"\nRead back:\n{df_read}")
    print(f"✓ Excel file created successfully: {excel_file}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
