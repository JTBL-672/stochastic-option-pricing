import numpy as np


def bs_monte_carlo(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N_sims: int = 100_000,
    option_type: str = "call",
    use_antithetic: bool = True,
    seed: int = 42,
) -> tuple[float, float]:
    """Prices a European option using Monte Carlo simulation under Geometric Brownian Motion (GBM).

    Supports variance reduction via Antithetic Variates.

    Parameters:
        S0: Current stock price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free interest rate
        sigma: Volatility
        N_sims: Total number of simulated price paths
        option_type: "call" or "put"
        use_antithetic: Whether to apply antithetic variates variance reduction
        seed: Random seed for reproducibility

    Returns:
        option_price: Discounted mean payoff
        standard_error: Standard error of the estimate
    """
    np.random.seed(seed)

    # 1. Generate Standard Normal Random Variables Z ~ N(0, 1)
    if use_antithetic:
        # Draw half the samples and concatenate their negative mirrors
        half_sims = N_sims // 2
        Z_half = np.random.standard_normal(half_sims)
        Z = np.concatenate([Z_half, -Z_half])
    else:
        Z = np.random.standard_normal(N_sims)

    # 2. Simulate Terminal Stock Prices S_T using exact GBM solution
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    S_T = S0 * np.exp(drift + diffusion)

    # 3. Compute Terminal Option Payoffs
    option_type = option_type.lower()
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError(
            f"Invalid option_type '{option_type}'. Expected 'call' or 'put'."
        )

    # 4. Discount Payoffs to Present Value (t = 0)
    discounted_payoffs = np.exp(-r * T) * payoffs

    # 5. Compute Option Price and Standard Error
    if use_antithetic:
        # Pairwise average of antithetic paths for accurate standard error calculation
        paired_payoffs = 0.5 * (
            discounted_payoffs[:half_sims] + discounted_payoffs[half_sims:]
        )
        option_price = float(np.mean(paired_payoffs))
        standard_error = float(
            np.std(paired_payoffs, ddof=1) / np.sqrt(half_sims)
        )
    else:
        option_price = float(np.mean(discounted_payoffs))
        standard_error = float(
            np.std(discounted_payoffs, ddof=1) / np.sqrt(N_sims)
        )

    return option_price, standard_error


if __name__ == "__main__":
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    N_sims = 100_000

    # Standard Monte Carlo
    c_std, se_std = bs_monte_carlo(
        S0, K, T, r, sigma, N_sims=N_sims, use_antithetic=False
    )

    # Antithetic Variates Monte Carlo
    c_anti, se_anti = bs_monte_carlo(
        S0, K, T, r, sigma, N_sims=N_sims, use_antithetic=True
    )

    print("=== Monte Carlo Simulation Results (100k paths) ===")
    print(f"Standard MC Call Price:   £{c_std:.4f} +/- £{se_std:.4f}")
    print(f"Antithetic MC Call Price: £{c_anti:.4f} +/- £{se_anti:.4f}")
    print(f"Standard Error Reduction: {((se_std - se_anti) / se_std) * 100:.2f}%")
