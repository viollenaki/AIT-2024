# save as predict_btc_mcmc.py
# Requirements:
# pip install pandas numpy pymc3 arviz matplotlib
# (use conda if you prefer; PyMC3 requires Theano/PyMC dependencies)

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

csv_path = "btc_hourly.csv"   # change if needed
predict_cutoff = pd.to_datetime("2024-01-01")  # "current date" as requested

# --- 1. Load data ----------------------------------------------------------
df = pd.read_csv(csv_path)
# Ensure Timestamp column parsed (common column name "Timestamp")
# If Timestamp is unix ms, detect and convert
if np.issubdtype(df['Timestamp'].dtype, np.number):
    # Try ms -> convert
    # Heuristic: if values larger than 1e10 treat as ms
    if df['Timestamp'].max() > 1e10:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    else:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
else:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

df = df.sort_values('Timestamp').reset_index(drop=True)

# Ensure Close is numeric
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

# Drop rows with NaN timestamp or Close
df = df.dropna(subset=['Timestamp', 'Close']).reset_index(drop=True)

# --- 2. Choose last observed timestamp <= predict_cutoff  ------------------
last_obs_idx = df[df['Timestamp'] <= predict_cutoff].index
if len(last_obs_idx) == 0:
    raise SystemExit(f"No rows found with Timestamp <= {predict_cutoff.isoformat()}")

last_idx = last_obs_idx.max()
last_time = df.loc[last_idx, 'Timestamp']
last_close = df.loc[last_idx, 'Close']

# find next actual row (for verifying prediction)
if last_idx + 1 < len(df):
    next_idx = last_idx + 1
    next_time = df.loc[next_idx, 'Timestamp']
    next_actual_close = df.loc[next_idx, 'Close']
else:
    next_idx = None
    next_time = None
    next_actual_close = None

print(f"Using last observed timestamp <= {predict_cutoff.date()}:")
print(f"  last_time = {last_time} (index {last_idx}), last_close = {last_close}")
if next_time is not None:
    print(f"  next_time in data = {next_time} (index {next_idx}), actual next Close = {next_actual_close}")
else:
    print("  No later timestamp in dataset; cannot show actual next price.")

# --- 3. Prepare series for AR(1) model (use Close series up to last_idx) ---
series = df.loc[:last_idx, 'Close'].values
T = len(series)
print(f"Training on {T} hourly observations (up to and including {last_time}).")

# Optionally standardize for numerical stability
mean_close = series.mean()
std_close = series.std(ddof=0) if series.std(ddof=0) > 0 else 1.0
series_std = (series - mean_close) / std_close

# Build lagged arrays: y_t and y_{t-1}
y = series_std[1:]
y_prev = series_std[:-1]
N = len(y)

# --- 4. Bayesian AR(1) using PyMC3 ----------------------------------------
with pm.Model() as ar1_model:
    sigma = pm.HalfNormal("sigma", sigma=1.0)
    alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
    beta = pm.Normal("beta", mu=0.0, sigma=1.0)  # AR(1) coefficient

    mu = alpha + beta * y_prev
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    # Use NUTS
    trace = pm.sample(draws=2000, tune=1000, target_accept=0.9, random_seed=RANDOM_SEED, return_inferencedata=True)

# Summarize posterior
summary = az.summary(trace, var_names=["alpha", "beta", "sigma"], round_to=4)
print("\nPosterior summary (alpha, beta, sigma):")
print(summary)

# --- 5. Posterior predictive for the next time --------------------------------
# We want to predict Close_{t+1} given the last observed Close (last_close).
# Transform last_close using same standardization
last_close_std = (last_close - mean_close) / std_close

# Draw posterior samples and compute predictive distribution for next std value:
posterior = trace.posterior  # xarray
alpha_samples = posterior['alpha'].stack(draws=("chain", "draw")).values
beta_samples = posterior['beta'].stack(draws=("chain", "draw")).values
sigma_samples = posterior['sigma'].stack(draws=("chain", "draw")).values

n_samples = alpha_samples.size
# compute predictive draws: y_next_std ~ Normal(alpha + beta * last_close_std, sigma)
y_next_std_draws = np.random.normal(loc=(alpha_samples + beta_samples * last_close_std),
                                   scale=sigma_samples,
                                   size=n_samples)

# Transform back to price scale
y_next_draws = y_next_std_draws * std_close + mean_close

pred_mean = np.mean(y_next_draws)
pred_median = np.median(y_next_draws)
pred_std = np.std(y_next_draws)
ci_lower, ci_upper = np.percentile(y_next_draws, [2.5, 97.5])

print("\nPrediction for the next timestamp after", last_time)
print(f" Predicted mean Close: {pred_mean:.2f}")
print(f" Predicted median Close: {pred_median:.2f}")
print(f" 95% credible interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f" Predicted std (posterior predictive): {pred_std:.2f}")

if next_actual_close is not None:
    print(f"\nActual next Close at {next_time}: {next_actual_close:.2f}")
    error = pred_mean - next_actual_close
    pct_error = 100.0 * error / next_actual_close
    print(f" Prediction error (mean - actual): {error:.2f} ({pct_error:.2f}%)")
else:
    print("\nNo actual next price available in dataset to compare.")

# --- 6. Optional: plot posterior predictive histogram vs actual --------------
plt.figure(figsize=(8,5))
plt.hist(y_next_draws, bins=80, density=True, alpha=0.6)
plt.axvline(pred_mean, linestyle='--', label=f'Pred mean {pred_mean:.2f}')
plt.axvline(ci_lower, color='gray', linestyle=':', label=f'95% CI [{ci_lower:.2f}, {ci_upper:.2f}]')
plt.axvline(ci_upper, color='gray', linestyle=':')
if next_actual_close is not None:
    plt.axvline(next_actual_close, color='red', linewidth=2, label=f'Actual next {next_actual_close:.2f}')
plt.legend()
plt.title(f"Posterior predictive for next Close after {last_time}")
plt.xlabel("BTC Close price")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
