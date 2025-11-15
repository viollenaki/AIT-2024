import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample stock dataset
def create_sample_dataset(filename='stock_prices.csv'):
    np.random.seed(42)
    
    # Generate 3 years of daily data
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Remove weekends (keep only weekdays)
    dates = dates[dates.dayofweek < 5]
    
    # Generate realistic price data with trends and volatility
    prices = [180.0]  # Starting price
    
    for i in range(1, len(dates)):
        # Base parameters
        trend = 0.0003  # Small upward trend
        volatility = 0.018  # Daily volatility
        
        # Add some market cycles
        cycle = 0.002 * np.sin(2 * np.pi * i / 63)  # Quarterly cycles
        
        # Random shock
        shock = np.random.normal(0, volatility)
        
        # Combine components
        daily_return = trend + cycle + shock
        new_price = prices[-1] * (1 + daily_return)
        
        # Ensure price doesn't go negative
        new_price = max(new_price, 1.0)
        prices.append(new_price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 50000000, len(dates))
    })
    
    # Add some missing values to simulate real data
    mask = np.random.random(len(df)) < 0.01  # 1% missing values
    df.loc[mask, 'Close'] = np.nan
    df = df.dropna()
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Sample dataset created: {filename}")
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df

# Create the sample file
create_sample_dataset('stock_prices.csv')