import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# ----------------------------
# 1. Helper Functions
# ----------------------------

def shannon_entropy(data, bins=50):
    hist, _ = np.histogram(data, bins=bins, density=False)
    probs = hist / np.sum(hist)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def gbm_trajectory(S0, mu, sigma, n_steps, dt=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    T = n_steps * dt
    t = np.linspace(0, T, n_steps + 1)
    dW = np.random.normal(0, np.sqrt(dt), size=n_steps)
    W = np.concatenate([[0], np.cumsum(dW)])
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)
    return S  # length n_steps + 1

def fit_gbm_params(data, dt=1.0):
    log_returns = np.diff(np.log(data))
    mu = np.mean(log_returns) / dt
    sigma = np.std(log_returns) / np.sqrt(dt)
    return mu, sigma

# ----------------------------
# 2. Simulate Time Series (Eq. 5: upward trend)
# ----------------------------

np.random.seed(42)
T_total = 100
t_full = np.arange(T_total)
X0 = 100
m = 0.5
delta = np.random.normal(0, 1, size=T_total)
X_true = X0 + m * (t_full + np.sin(2 * t_full)) + delta

# Split: 75% historical, 25% future
split_idx = int(0.75 * T_total)  # 75
historical = X_true[:split_idx]   # first 75 points
future_true = X_true[split_idx:]  # last 25 points
n_forecast = len(future_true)

print(f"Using {len(historical)} historical points to forecast {n_forecast} future points.")

# ----------------------------
# 3. Fit GBM parameters from historical data
# ----------------------------

mu, sigma = fit_gbm_params(historical)
S0 = historical[-1]
dt = 1.0

# ----------------------------
# 4. Generate GBM forecasts (Monte Carlo)
# ----------------------------

n_trajectories = 10000
gbm_forecasts = []
for i in range(n_trajectories):
    path = gbm_trajectory(S0, mu, sigma, n_forecast, dt=dt)
    gbm_forecasts.append(path[1:])  # exclude S0

gbm_forecasts = np.array(gbm_forecasts)  # shape: (N, n_forecast)
gbm_mean = np.mean(gbm_forecasts, axis=0)

# ----------------------------
# 5. Apply EC-GBM filtering
# ----------------------------

H_ref = shannon_entropy(historical, bins=30)
entropy_changes = []
all_paths = []

for i in range(n_trajectories):
    future_vals = gbm_forecasts[i]
    combined = np.concatenate([historical, future_vals])
    H_new = shannon_entropy(combined, bins=30)
    delta_H = H_ref - H_new
    entropy_changes.append(delta_H)
    all_paths.append(future_vals)

entropy_changes = np.array(entropy_changes)
max_delta_H = np.max(entropy_changes[entropy_changes > 0]) if np.any(entropy_changes > 0) else 0.0

# Accept trajectories with entropy reduction >= 75% of max
epsilon = 0.75 * max_delta_H
accepted_mask = entropy_changes >= epsilon

if not np.any(accepted_mask):
    # Fallback: use all that reduce entropy
    accepted_mask = entropy_changes > 0
    if not np.any(accepted_mask):
        raise RuntimeError("No trajectory reduced entropy. Try increasing bins or n_trajectories.")

ec_gbm_forecasts = gbm_forecasts[accepted_mask]
ec_gbm_mean = np.mean(ec_gbm_forecasts, axis=0)

print(f"Accepted {len(ec_gbm_forecasts)} / {n_trajectories} trajectories in EC-GBM.")

# ----------------------------
# 6. Evaluation Metrics
# ----------------------------

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

rmse_gbm = rmse(future_true, gbm_mean)
rmse_ec_gbm = rmse(future_true, ec_gbm_mean)

print(f"RMSE - GBM: {rmse_gbm:.3f}")
print(f"RMSE - EC-GBM: {rmse_ec_gbm:.3f}")

# ----------------------------
# 7. Plot Results
# ----------------------------

plt.figure(figsize=(12, 6))
t_hist = np.arange(len(historical))
t_future = np.arange(len(historical), len(historical) + n_forecast)

# Historical
plt.plot(t_hist, historical, 'o-', color='blue', label='Historical (75%)', markersize=3)

# True future
plt.plot(t_future, future_true, 's-', color='black', label='True Future (25%)', markersize=3)

# GBM forecast
plt.plot(t_future, gbm_mean, '--', color='red', label=f'GBM Forecast (RMSE={rmse_gbm:.2f})')

# EC-GBM forecast
plt.plot(t_future, ec_gbm_mean, '--', color='green', label=f'EC-GBM Forecast (RMSE={rmse_ec_gbm:.2f})')

plt.axvline(x=split_idx - 0.5, color='gray', linestyle=':', linewidth=1)
plt.title("GBM vs EC-GBM Forecast on Simulated Time Series")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()