def detailed_distribution_analysis(historical_data, test_data, n_bins=50):
    """
    Detailed probability distribution analysis as in paper Figures 2 & 3
    """
    forecast_horizon = len(test_data)
    
    # Initialize models
    ec_gbm = EntropyCorrectedGBM(n_bins=n_bins, mc_simulations=2000, random_state=42)
    ec_gbm.estimate_gbm_parameters(historical_data)
    
    # Generate trajectories
    gbm_trajectories = ec_gbm.gbm_simulation(forecast_horizon)
    ec_results = ec_gbm.ec_gbm_forecast(historical_data, forecast_horizon, epsilon_frac=0.75)
    
    # Create detailed distribution comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel 1: Distribution reconstruction (as in paper Figure 2)
    x_min = min(historical_data.min(), gbm_trajectories.min(), ec_results['accepted_trajectories'].min())
    x_max = max(historical_data.max(), gbm_trajectories.max(), ec_results['accepted_trajectories'].max())
    x_range = np.linspace(x_min, x_max, 1000)
    
    # Kernel Density Estimates for smooth distributions
    kde_historical = stats.gaussian_kde(historical_data)
    kde_gbm = stats.gaussian_kde(gbm_trajectories.flatten())
    kde_ec_gbm = stats.gaussian_kde(ec_results['accepted_trajectories'].flatten())
    
    axes[0].plot(x_range, kde_historical(x_range), 'orange', linewidth=3, 
                label='Reference PD', alpha=0.8)
    axes[0].plot(x_range, kde_gbm(x_range), 'green', linewidth=2, 
                label='GBM Reconstructed PD', alpha=0.7, linestyle='--')
    axes[0].plot(x_range, kde_ec_gbm(x_range), 'blue', linewidth=2, 
                label='EC-GBM Reconstructed PD', alpha=0.7, linestyle='--')
    
    axes[0].fill_between(x_range, kde_historical(x_range), alpha=0.3, color='orange')
    axes[0].fill_between(x_range, kde_gbm(x_range), alpha=0.2, color='green')
    axes[0].fill_between(x_range, kde_ec_gbm(x_range), alpha=0.2, color='blue')
    
    axes[0].set_xlabel('Value', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Probability Distribution Reconstruction\n(Kernel Density Estimation)', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Statistical moments comparison
    moments_data = {
        'Mean': [np.mean(historical_data), np.mean(gbm_trajectories), np.mean(ec_results['accepted_trajectories'])],
        'Std Dev': [np.std(historical_data), np.std(gbm_trajectories), np.std(ec_results['accepted_trajectories'])],
        'Skewness': [stats.skew(historical_data), stats.skew(gbm_trajectories.flatten()), 
                    stats.skew(ec_results['accepted_trajectories'].flatten())],
        'Kurtosis': [stats.kurtosis(historical_data), stats.kurtosis(gbm_trajectories.flatten()), 
                    stats.kurtosis(ec_results['accepted_trajectories'].flatten())]
    }
    
    moments_df = pd.DataFrame(moments_data, 
                             index=['Reference', 'GBM', 'EC-GBM'])
    
    # Plot moments comparison
    x_moments = np.arange(len(moments_df.columns))
    width = 0.25
    
    for i, (idx, row) in enumerate(moments_df.iterrows()):
        axes[1].bar(x_moments + i*width, row.values, width, label=idx, alpha=0.8)
    
    axes[1].set_xlabel('Statistical Moment', fontsize=12)
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('Comparison of Statistical Moments', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_moments + width)
    axes[1].set_xticklabels(moments_df.columns)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(moments_df.iterrows()):
        for j, value in enumerate(row.values):
            axes[1].text(j + i*width, value + 0.01 * max(row.values), 
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print KL divergences
    def calculate_kl_divergence(p, q):
        """Calculate KL divergence between two distributions"""
        p = np.where(p == 0, 1e-10, p)
        q = np.where(q == 0, 1e-10, q)
        return np.sum(p * np.log(p / q))
    
    # Calculate KL divergence using histograms
    hist_ref, bins = np.histogram(historical_data, bins=50, density=True)
    hist_gbm, _ = np.histogram(gbm_trajectories.flatten(), bins=bins, density=True)
    hist_ec_gbm, _ = np.histogram(ec_results['accepted_trajectories'].flatten(), bins=bins, density=True)
    
    kl_gbm = calculate_kl_divergence(hist_ref, hist_gbm)
    kl_ec_gbm = calculate_kl_divergence(hist_ref, hist_ec_gbm)
    
    print(f"\n--- Distribution Similarity (KL Divergence) ---")
    print(f"KL Divergence (Reference vs GBM): {kl_gbm:.6f}")
    print(f"KL Divergence (Reference vs EC-GBM): {kl_ec_gbm:.6f}")
    print(f"Improvement: {((kl_gbm - kl_ec_gbm) / kl_gbm * 100):.1f}%")
    
    return moments_df

# Run detailed distribution analysis
print("RUNNING DETAILED DISTRIBUTION ANALYSIS")
synthetic_data, t = analyzer.generate_synthetic_data(trend_type='upward')
train_data, test_data = analyzer.train_test_split(synthetic_data, test_size=0.2)
moments_df = detailed_distribution_analysis(train_data, test_data)
print("\nStatistical Moments Comparison:")
print(moments_df)