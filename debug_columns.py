import pandas as pd
import os

# 1. FIND THE FILE
# This uses the same logic as your other scripts to find the data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'delhi_ncr_aqi_dataset.csv')

print(f"🕵️ INVESTIGATING: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print("❌ ERROR: File not found at this location.")
else:
    # 2. READ THE HEADER
    try:
        df = pd.read_csv(DATA_PATH)
        print("\n✅ FILE FOUND! Here are your EXACT column names:")
        print("--------------------------------------------------")
        print(df.columns.tolist())
        print("--------------------------------------------------")
        print("Please copy and paste the list above into the chat!")
    except Exception as e:
        print(f"❌ Error reading file: {e}")