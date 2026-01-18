import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def markov_chain_predict_btc(csv_path, current_date="2024-01-01", simulations=5000):
    # Load data
    df = pd.read_csv(csv_path)
    if np.issubdtype(df['Timestamp'].dtype, np.number):
        if df['Timestamp'].max() > 1e10:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        else:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    else:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df = df.dropna(subset=['Close'])

    # Compute hourly price change
    df['Change'] = df['Close'].diff()
    mu = df['Change'].mean()
    sigma = df['Change'].std()

    # Find current and next points
    cutoff = pd.to_datetime(current_date)
    last_idx = df[df['Timestamp'] <= cutoff].index.max()
    if last_idx is None or np.isnan(last_idx):
        raise ValueError("No data before or at the given date.")

    last_price = df.loc[last_idx, 'Close']
    next_price = df.loc[last_idx + 1, 'Close'] if last_idx + 1 < len(df) else None
    next_time = df.loc[last_idx + 1, 'Timestamp'] if last_idx + 1 < len(df) else None

    # Simulate next prices via Markov Chain
    next_prices = last_price + np.random.normal(mu, sigma, size=simulations)
    pred_mean = np.mean(next_prices)
    pred_ci = np.percentile(next_prices, [2.5, 97.5])

    # Output
    print(f"Last known price (at {df.loc[last_idx, 'Timestamp']}): {last_price:.2f}")
    print(f"Predicted next price (Markov Chain): {pred_mean:.2f}")
    print(f"95% confidence interval: [{pred_ci[0]:.2f}, {pred_ci[1]:.2f}]")
    if next_price is not None:
        print(f"Actual next price (at {next_time}): {next_price:.2f}")
        diff = pred_mean - next_price
        pct_error = 100 * diff / next_price
        print(f"Prediction error: {diff:.2f} ({pct_error:.2f}%)")

    # Plot distribution
    plt.figure(figsize=(8,5))
    plt.hist(next_prices, bins=50, color='skyblue', alpha=0.7, density=True)
    plt.axvline(pred_mean, color='black', linestyle='--', label=f'Predicted mean {pred_mean:.2f}')
    if next_price is not None:
        plt.axvline(next_price, color='red', label=f'Actual {next_price:.2f}')
    plt.title('Markov Chain Monte Carlo Prediction for Next BTC Price')
    plt.xlabel('BTC Price')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Example usage:
# markov_chain_predict_btc("btc_hourly.csv", "2024-01-01")
