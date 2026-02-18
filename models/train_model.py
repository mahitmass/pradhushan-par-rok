import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# 1. SETUP PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'delhi_ncr_aqi_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'pollution_model.pkl')

def train():
    print(f"🔄 Loading data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print("❌ ERROR: Data file not found.")
        return

    # 2. LOAD DATA
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data Loaded. Shape: {df.shape}")

    # 3. SMART FEATURE ENGINEERING
    target = 'aqi'
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
    
    # Features (Wind, Temp, Humidity, Visibility)
    features = ['hour', 'month', 'day_of_week', 'temperature', 'humidity', 'wind_speed', 'visibility']
    
    # Validate Columns
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing columns: {missing_cols}")
        return

    # Clean Data
    df = df.dropna(subset=[target])
    X = df[features].fillna(df[features].mean())
    y = df[target]

    # 4. TRAIN OPTIMIZED MODEL (The "Diet" Plan)
    print(f"🧠 Training Optimized Model on {len(df)} rows...")
    print("   (Limiting tree depth to prevent 2GB file size...)")
    
    model = RandomForestRegressor(
        n_estimators=100,      # Reduced from 150 to 100 (Still plenty)
        max_depth=20,          # Limit depth (Prevents infinite growth)
        min_samples_leaf=4,    # Don't split for tiny groups (Reduces noise)
        max_features='sqrt',   # More efficient splitting
        n_jobs=-1,
        random_state=42
    )
    model.fit(X, y)

    # 5. SAVE WITH COMPRESSION (Crucial!)
    # compress=3 shrinks the file significantly
    joblib.dump(model, MODEL_PATH, compress=3)
    
    # Check Size
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✅ SUCCESS! Model saved to: {MODEL_PATH}")
    print(f"📦 New File Size: {size_mb:.2f} MB")
    
    if size_mb > 100:
        print("⚠️ WARNING: Still over 100MB. GitHub might reject it.")
    else:
        print("🚀 PERFECT! Ready for GitHub push.")

if __name__ == "__main__":
    train()