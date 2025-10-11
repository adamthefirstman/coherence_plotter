import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

class ProperECGBM:
    """
    Proper implementation of EC-GBM based on the paper's methodology
    """
    
    def __init__(self):
        self.mu = 0.0
        self.sigma = 1.0
        self.S0 = 1.0
    
    def shannon_entropy(self, probabilities):
        """Calculate Shannon entropy for a probability distribution"""
        probs = probabilities[probabilities > 0]
        if len(probs) == 0:
            return 0
        return -np.sum(probs * np.log(probs))
    
    def compute_probability_distribution(self, data, bins=50, data_range=None):
        """Compute probability distribution from continuous data"""
        if data_range is None:
            data_range = (np.min(data), np.max(data))
        
        counts, _ = np.histogram(data, bins=bins, range=data_range, density=True)
        probabilities = counts / np.sum(counts)
        return probabilities
    
    def kl_divergence(self, p, q):
        """Calculate KL divergence D_KL(P || Q)"""
        epsilon = 1e-10
        p_safe = np.clip(p, epsilon, 1)
        q_safe = np.clip(q, epsilon, 1)
        return np.sum(p_safe * np.log(p_safe / q_safe))
    
    def fit_gbm_parameters(self, data):
        """Estimate GBM parameters from data assuming geometric returns"""
        # For dice experiment, we need to transform to continuous space
        # Use log of cumulative sum to get something that resembles geometric growth
        cumulative_data = np.cumsum(data)
        # Add small constant to avoid log(0)
        log_data = np.log(cumulative_data + 1e-6)
        log_returns = np.diff(log_data)
        
        if len(log_returns) > 0:
            self.mu = np.mean(log_returns)
            self.sigma = np.std(log_returns)
            self.S0 = cumulative_data[-1]
        else:
            self.mu = 0.001
            self.sigma = 0.1
            self.S0 = np.mean(data)
    
    def gbm_trajectory(self, n_steps, dt=1):
        """Generate a single GBM trajectory"""
        # Generate Wiener process
        dW = np.random.standard_normal(size=n_steps) * np.sqrt(dt)
        W = np.cumsum(dW)
        
        # Time array
        t = np.arange(1, n_steps + 1) * dt
        
        # GBM formula: S(t) = S0 * exp((μ - σ²/2)t + σW(t))
        drift = (self.mu - 0.5 * self.sigma**2) * t
        diffusion = self.sigma * W
        
        trajectory = self.S0 * np.exp(drift + diffusion)
        return trajectory
    
    def monte_carlo_ecgbm(self, reference_data, n_trajectories=1000, 
                         forecast_steps=10, epsilon=0.1, bins=50):
        """Proper EC-GBM implementation following paper methodology"""
        
        # Fit GBM parameters to reference data
        self.fit_gbm_parameters(reference_data)
        
        # Compute reference distribution and entropy
        data_range = (np.min(reference_data), np.max(reference_data))
        ref_probs = self.compute_probability_distribution(reference_data, bins=bins, data_range=data_range)
        H_ref = self.shannon_entropy(ref_probs)
        
        print(f"Reference entropy: {H_ref:.4f}, mu: {self.mu:.4f}, sigma: {self.sigma:.4f}")
        
        accepted_trajectories = []
        entropy_changes = []
        all_trajectories = []
        
        for i in range(n_trajectories):
            # Generate GBM trajectory (continuous)
            trajectory = self.gbm_trajectory(forecast_steps)
            all_trajectories.append(trajectory)
            
            # Combine with reference data
            combined_data = np.concatenate([reference_data, trajectory])
            
            # Compute new distribution and entropy
            new_probs = self.compute_probability_distribution(combined_data, bins=bins, data_range=data_range)
            H_new = self.shannon_entropy(new_probs)
            
            entropy_change = H_ref - H_new
            
            # Accept trajectory if entropy decrease exceeds threshold
            if entropy_change > epsilon:
                accepted_trajectories.append(trajectory)
                entropy_changes.append(entropy_change)
        
        return (np.array(accepted_trajectories), np.array(entropy_changes), 
                np.array(all_trajectories), H_ref)

def proper_dice_experiment():
    """Proper implementation following paper methodology"""
    
    # Define biased dice probabilities
    dice_faces = np.array([1, 2, 3, 4, 5, 6])
    biased_probs = np.array([0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    print("Biased Dice Probabilities:")
    for face, prob in zip(dice_faces, biased_probs):
        print(f"Face {face}: {prob:.3f}")
    
    theoretical_entropy = -np.sum(biased_probs * np.log(biased_probs))
    print(f"\nTheoretical Shannon Entropy: {theoretical_entropy:.3f}")
    
    sample_sizes = [50, 500, 5000]
    forecast_lengths = [10, 50, 100]
    
    results = {}
    metrics_summary = []
    
    for K, forecast_steps in zip(sample_sizes, forecast_lengths):
        print(f"\n{'='*60}")
        print(f"Number of rolls: {K}")
        print(f"Forecast steps: {forecast_steps}")
        print(f"{'='*60}")
        
        # Generate biased dice rolls
        np.random.seed(42)
        rolls = np.random.choice(dice_faces, size=K, p=biased_probs)
        
        # Transform to continuous space for GBM
        # Use cumulative sum to create a time series that GBM can model
        continuous_data = np.cumsum(rolls).astype(float)
        
        # Initialize EC-GBM
        ec_gbm = ProperECGBM()
        
        # Run EC-GBM
        epsilon = 0.05  # Adjusted threshold
        accepted_trajs, entropy_changes, all_trajs, H_ref = ec_gbm.monte_carlo_ecgbm(
            continuous_data, n_trajectories=500,
            forecast_steps=forecast_steps, epsilon=epsilon
        )
        
        if len(accepted_trajs) > 0:
            # Compute distributions for evaluation
            data_range = (np.min(continuous_data), np.max(continuous_data))
            ref_probs = ec_gbm.compute_probability_distribution(continuous_data, data_range=data_range)
            
            # EC-GBM distribution (using accepted trajectories)
            ecgbm_forecast = accepted_trajs.flatten()
            ecgbm_probs = ec_gbm.compute_probability_distribution(
                np.concatenate([continuous_data, ecgbm_forecast]), data_range=data_range
            )
            
            # Traditional GBM distribution (using all trajectories)
            gbm_forecast = all_trajs.flatten()
            gbm_probs = ec_gbm.compute_probability_distribution(
                np.concatenate([continuous_data, gbm_forecast]), data_range=data_range
            )
            
            # Calculate metrics
            ecgbm_kl = ec_gbm.kl_divergence(ref_probs, ecgbm_probs)
            gbm_kl = ec_gbm.kl_divergence(ref_probs, gbm_probs)
            
            acceptance_rate = len(accepted_trajs) / len(all_trajs)
            
            # Store results
            results[K] = {
                'rolls': rolls,
                'continuous_data': continuous_data,
                'ref_probs': ref_probs,
                'ecgbm_probs': ecgbm_probs,
                'gbm_probs': gbm_probs,
                'acceptance_rate': acceptance_rate,
                'entropy_changes': entropy_changes,
                'H_ref': H_ref,
                'H_ecgbm': ec_gbm.shannon_entropy(ecgbm_probs),
                'H_gbm': ec_gbm.shannon_entropy(gbm_probs)
            }
            
            metrics_summary.append({
                'K': K,
                'ECGBM_KL': ecgbm_kl,
                'GBM_KL': gbm_kl,
                'Acceptance_Rate': acceptance_rate,
                'Reference_Entropy': H_ref,
                'ECGBM_Entropy': ec_gbm.shannon_entropy(ecgbm_probs),
                'GBM_Entropy': ec_gbm.shannon_entropy(gbm_probs),
                'Avg_Entropy_Change': np.mean(entropy_changes) if len(entropy_changes) > 0 else 0,
                'N_Accepted': len(accepted_trajs)
            })
            
            print(f"Acceptance rate: {acceptance_rate:.3f}")
            print(f"EC-GBM KL: {ecgbm_kl:.4f}, GBM KL: {gbm_kl:.4f}")
            print(f"Reference entropy: {H_ref:.4f}")
            print(f"EC-GBM entropy: {ec_gbm.shannon_entropy(ecgbm_probs):.4f}")
            print(f"GBM entropy: {ec_gbm.shannon_entropy(gbm_probs):.4f}")
            print(f"Average entropy change: {np.mean(entropy_changes):.6f}")
        else:
            print("No trajectories accepted - try decreasing epsilon")
    
    return results, dice_faces, biased_probs, metrics_summary

def analyze_proper_results(results, metrics_summary):
    """Analyze the proper EC-GBM implementation"""
    
    if not results:
        print("No results to analyze")
        return
    
    # Create summary table
    df = pd.DataFrame(metrics_summary)
    print("\n" + "="*80)
    print("PROPER EC-GBM IMPLEMENTATION RESULTS")
    print("="*80)
    print(df.round(4).to_string(index=False))
    
    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (K, result) in enumerate(results.items()):
        ax = axes[idx]
        
        # Plot probability distributions
        x = np.linspace(0, 1, len(result['ref_probs']))
        ax.plot(x, result['ref_probs'], 'b-', linewidth=3, label='Reference', alpha=0.8)
        ax.plot(x, result['ecgbm_probs'], 'g-', linewidth=2, label='EC-GBM', alpha=0.8)
        ax.plot(x, result['gbm_probs'], 'r--', linewidth=2, label='GBM', alpha=0.8)
        
        ax.set_title(f'K = {K} | Acceptance: {result["acceptance_rate"]:.3f}', fontsize=12)
        ax.set_xlabel('Normalized Value')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add entropy information
        ax.text(0.02, 0.95, f'H_ref: {result["H_ref"]:.3f}', transform=ax.transAxes, 
                fontsize=9, bbox=dict(boxstyle="round", facecolor="blue", alpha=0.3))
        ax.text(0.02, 0.88, f'H_ECGBM: {result["H_ecgbm"]:.3f}', transform=ax.transAxes, 
                fontsize=9, bbox=dict(boxstyle="round", facecolor="green", alpha=0.3))
        ax.text(0.02, 0.81, f'H_GBM: {result["H_gbm"]:.3f}', transform=ax.transAxes, 
                fontsize=9, bbox=dict(boxstyle="round", facecolor="red", alpha=0.3))
    
    plt.tight_layout()
    plt.show()
    
    # Performance comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    for _, row in df.iterrows():
        improvement = ((row['GBM_KL'] - row['ECGBM_KL']) / row['GBM_KL']) * 100
        entropy_reduction = ((row['Reference_Entropy'] - row['ECGBM_Entropy']) / row['Reference_Entropy']) * 100
        
        print(f"K={row['K']}:")
        print(f"  EC-GBM shows {improvement:+.1f}% improvement in KL divergence")
        print(f"  EC-GBM reduces entropy by {entropy_reduction:+.1f}%")
        print(f"  {row['N_Accepted']} trajectories accepted")

# Run the proper experiment
if __name__ == "__main__":
    print("PROPER EC-GBM IMPLEMENTATION")
    print("Based on Gupta et al. (2024) - Entropy corrected geometric Brownian motion")
    print("="*80)
    
    results, dice_faces, biased_probs, metrics_summary = proper_dice_experiment()
    analyze_proper_results(results, metrics_summary)