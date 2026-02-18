import pandas as pd
import numpy as np
import datetime

def get_historical_data():
    """
    Simulates fetching 2 years of Delhi AQI data for training.
    In real life, you would load 'data/delhi.csv' here.
    """
    # Create date range
    dates = pd.date_range(start="2024-01-01", end=datetime.date.today(), freq='H')
    
    # Generate synthetic pollution patterns (Higher in winter, lower in monsoon)
    df = pd.DataFrame(index=dates)
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    
    # Base AQI + Seasonality + Random Noise
    # Winter (Nov-Jan) is high, Monsoon (July-Sep) is low
    df['pm25'] = 100 + (np.cos((df['month'] - 1) / 12 * 2 * np.pi) * -50) + np.random.normal(0, 20, len(dates))
    
    # Add daily rush hour spikes (8am-10am, 6pm-9pm)
    df['pm25'] += df['hour'].apply(lambda x: 40 if x in [8,9,18,19,20] else 0)
    
    # Clip to realistic values
    df['pm25'] = df['pm25'].clip(lower=20, upper=999)
    
    return df

def prepare_training_data(df):
    """
    Prepares data for the AI Model (Features vs Target).
    Target: PM2.5 levels
    Features: Hour of day, Month, Day of week (Traffic patterns)
    """
    df['day_of_week'] = df.index.dayofweek
    features = ['hour', 'month', 'day_of_week']
    target = 'pm25'
    return df[features], df[target]
  
