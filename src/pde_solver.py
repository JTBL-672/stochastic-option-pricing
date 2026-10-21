import numpy as np

def bs_pde_implicit_fdm(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N_S: int = 200,
    N_t: int = 1000,
    S_max_mult: float = 3.0,
    option_type: str = "call",
) -> float:
    """Prices a European option using an Implicit Finite Difference Method on the Black-Scholes PDE.

    Parameters:
        S0: Current asset price
        K: Strike price
        T: Time to maturity (years)
        r: Risk-free rate
        sigma: Volatility
        N_S: Number of stock price grid intervals
        N_t: Number of time steps
        S_max_mult: Multiplier for upper stock boundary (S_max = K * S_max_mult)
        option_type: "call" or "put"

    Returns:
        option_price: Interpolated option price at S0
    """
    # 1. Grid Discretization
    S_max = K * S_max_mult
    dS = S_max / N_S
    dt = T / N_t
    S_grid = np.linspace(0, S_max, N_S + 1)

    # 2. Terminal Payoff Condition (at t = T)
    option_type = option_type.lower()
    if option_type == "call":
        V = np.maximum(S_grid - K, 0.0)
    elif option_type == "put":
        V = np.maximum(K - S_grid, 0.0)
    else:
        raise ValueError(
            f"Invalid option_type '{option_type}'. Expected 'call' or 'put'."
        )

    # 3. Tridiagonal Coefficients for Interior Nodes (i = 1 to N_S - 1)
    i_indices = np.arange(1, N_S)
    A = 0.5 * dt * (r * i_indices - (sigma**2) * (i_indices**2))
    B = 1.0 + dt * (r + (sigma**2) * (i_indices**2))
    C = -0.5 * dt * (r * i_indices + (sigma**2) * (i_indices**2))

    # 4. Construct Tridiagonal System Matrix M
    M = np.diag(B, 0) + np.diag(A[1:], -1) + np.diag(C[:-1], 1)

    # 5. Backward Time-Stepping Loop (t = T -> 0)
    V_current = V.copy()
    for step in range(1, N_t + 1):
        tau = step * dt

        # Evaluate exact boundary conditions at current time step
        if option_type == "call":
            V_lower = 0.0
            V_upper = S_max - K * np.exp(-r * tau)
        else:
            V_lower = K * np.exp(-r * tau)
            V_upper = 0.0

        # Formulate Right-Hand Side (RHS) vector
        rhs = V_current[1:-1].copy()
        rhs[0] -= A[0] * V_lower
        rhs[-1] -= C[-1] * V_upper

        # Solve linear system M * V_interior = rhs
        V_interior = np.linalg.solve(M, rhs)
        V_current = np.concatenate([[V_lower], V_interior, [V_upper]])

    # 6. Interpolate Option Price at S0
    return float(np.interp(S0, S_grid, V_current))


if __name__ == "__main__":
    # Test execution against analytical benchmarks
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    call_pde = bs_pde_implicit_fdm(
        S0, K, T, r, sigma, N_S=200, N_t=1000, option_type="call"
    )
    put_pde = bs_pde_implicit_fdm(
        S0, K, T, r, sigma, N_S=200, N_t=1000, option_type="put"
    )

    print(f"Implicit FDM Call Price: £{call_pde:.4f}")
    print(f"Implicit FDM Put Price:  £{put_pde:.4f}")
