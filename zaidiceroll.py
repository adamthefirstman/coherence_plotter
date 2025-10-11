import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
K = 5000  # Number of historical dice rolls
n_simulations = 5000  # Number of Monte Carlo simulations
n_forecast_steps = 100  # Number of steps to forecast

# Define the biased dice probabilities (favoring 3 and 4 to create a normal-like distribution)
dice_probabilities = np.array([0.05, 0.15, 0.30, 0.30, 0.15, 0.05])
dice_faces = np.array([1, 2, 3, 4, 5, 6])

# Generate K biased dice rolls
historical_rolls = np.random.choice(dice_faces, size=K, p=dice_probabilities)

# Calculate the historical distribution
historical_counts = np.bincount(historical_rolls, minlength=7)[1:7]  # Ignore 0 index
historical_probabilities = historical_counts / K

# Create a time series from the dice rolls (normalized to [0,1])
time_series = historical_rolls / 6.0

# Calculate log returns for GBM
log_returns = np.diff(np.log(time_series + 1e-10))  # Add small value to avoid log(0)

# Estimate drift (μ) and volatility (σ) for GBM
mu = np.mean(log_returns)
sigma = np.std(log_returns)

# Time step (dt)
dt = 1

# Function to simulate GBM
def simulate_gbm(S0, mu, sigma, dt, n_steps):
    """
    Simulate a Geometric Brownian Motion path.
    
    Parameters:
    S0 (float): Initial value
    mu (float): Drift coefficient
    sigma (float): Volatility coefficient
    dt (float): Time step
    n_steps (int): Number of steps to simulate
    
    Returns:
    numpy.ndarray: Simulated path
    """
    # Generate random shocks
    shocks = np.random.normal(0, 1, n_steps)
    
    # Initialize the path
    path = np.zeros(n_steps + 1)
    path[0] = S0
    
    # Simulate the path
    for t in range(1, n_steps + 1):
        path[t] = path[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks[t-1])
    
    return path

# Function to convert GBM values to dice rolls with bias
def gbm_to_dice(gbm_values, dice_probabilities):
    """
    Convert GBM values to dice rolls with bias.
    
    Parameters:
    gbm_values (numpy.ndarray): GBM values
    dice_probabilities (numpy.ndarray): Desired dice probabilities
    
    Returns:
    numpy.ndarray: Dice rolls
    """
    # Normalize GBM values to [0, 1]
    min_val = np.min(gbm_values)
    max_val = np.max(gbm_values)
    
    # Avoid division by zero
    if max_val == min_val:
        normalized = np.ones_like(gbm_values) * 0.5
    else:
        normalized = (gbm_values - min_val) / (max_val - min_val)
    
    # Convert to dice rolls using the biased probabilities
    dice_rolls = np.zeros_like(normalized, dtype=int)
    for i, value in enumerate(normalized):
        # Create a cumulative distribution from the probabilities
        cum_prob = np.cumsum(dice_probabilities)
        
        # Find where the normalized value falls in the cumulative distribution
        idx = np.searchsorted(cum_prob, value)
        
        # Ensure idx is within [0, 5]
        idx = max(0, min(5, idx))
        
        # Assign the dice face
        dice_rolls[i] = idx + 1
    
    return dice_rolls

# Monte Carlo simulations using GBM
all_forecasts = []
for _ in range(n_simulations):
    # Initial value for GBM is the last value of the time series
    S0 = time_series[-1]
    
    # Simulate GBM path
    gbm_path = simulate_gbm(S0, mu, sigma, dt, n_forecast_steps)
    
    # Convert to dice rolls
    dice_forecast = gbm_to_dice(gbm_path, dice_probabilities)
    
    all_forecasts.extend(dice_forecast)

# Calculate the forecast distribution
forecast_counts = np.bincount(all_forecasts, minlength=7)[1:7]  # Ignore 0 index
forecast_probabilities = forecast_counts / len(all_forecasts)

# Plot the comparison
plt.figure(figsize=(12, 6))

# Historical distribution
plt.subplot(1, 2, 1)
plt.bar(dice_faces, historical_probabilities, alpha=0.7, label='Historical')
plt.title('Historical Dice Roll Distribution')
plt.xlabel('Dice Face')
plt.ylabel('Probability')
plt.xticks(dice_faces)
plt.legend()

# Forecast distribution
plt.subplot(1, 2, 2)
plt.bar(dice_faces, forecast_probabilities, alpha=0.7, label='Forecast', color='orange')
plt.title('GBM Forecasted Dice Roll Distribution')
plt.xlabel('Dice Face')
plt.ylabel('Probability')
plt.xticks(dice_faces)
plt.legend()

plt.tight_layout()
plt.show()

# Print the probability distributions
print("Historical Probabilities:")
for face, prob in zip(dice_faces, historical_probabilities):
    print(f"P({face}) = {prob:.4f}")

print("\nGBM Forecasted Probabilities:")
for face, prob in zip(dice_faces, forecast_probabilities):
    print(f"P({face}) = {prob:.4f}")

# Statistical comparison
print("\nStatistical Comparison:")
print(f"Kullback-Leibler Divergence (Historical || Forecast): {stats.entropy(historical_probabilities, forecast_probabilities):.4f}")
print(f"Kullback-Leibler Divergence (Forecast || Historical): {stats.entropy(forecast_probabilities, historical_probabilities):.4f}")
print(f"Bhattacharyya Coefficient: {np.sum(np.sqrt(historical_probabilities * forecast_probabilities)):.4f}")