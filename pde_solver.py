import numpy as np
def bs_pde_implicit_fdm(
    S0, K, T, r, sigma, N_S=200, N_t=1000, S_max_mult=3, option_type="call"
):
    # 1. Grid Setup
    S_max = K*S_max_mult
    dS = S_max/N_S
    dt = T/N_t
    S_grid = np.linspace(0, S_max, N_S+1)

    # 2. Terminal Payoff Condition
    option_type = option_type.lower()
    if option_type=="call":
        V = np.maximum(S_grid - K, 0.0)
    elif option_type=="put":
        V = np.maximum(K - S_grid, 0.0)
    else:
        raise ValueError(
            f"Invalid option_type: '{option_type}'. Expected 'call' or 'put'."
        )

    # 3. Interior Indices and Coefficient Vectors
    i_indices = np.arange(1, N_S)
    A = 0.5*dt*(r*i_indices - sigma**2*i_indices**2)
    B = 1 + dt*(r + i_indices**2*sigma**2)
    C = -0.5*dt*(r*i_indices + sigma**2*i_indices**2)

    # 4. Constructing tridiagonal matrix M using A, B, and C
    M_B = np.diag(B,0)
    M_A = np.diag(A[1:], -1)
    M_C = np.diag(C[:-1], 1)
    M = M_A + M_B + M_C

    # 5. Backward Time Stepping Loop 
    V_current = V.copy()
    for step in range(1,N_t + 1):
        tau = step * dt # time remaining to maturity

        if option_type=="call":
            V_lower = 0.0
            V_upper = S_max - K * np.exp(-r * tau)
        else: # put
            V_lower = K * np.exp(-r * tau)
            V_upper = 0.0


        # Construct Right-Hand Side (RHS)
        rhs = V_current[1:-1].copy()
        rhs[0] = rhs[0] - A[0] * V_lower
        rhs[-1] = rhs[-1] - C[-1] * V_upper

        # Solve linear system for interior nodes
        V_interior = np.linalg.solve(M, rhs)
        V_current = np.concatenate([[V_lower], V_interior, [V_upper]])

    # 6. Interpolate Option Price at S0
    option_price = np.interp(S0, S_grid, V_current)
    return option_price, S_grid, V_current

# Test execution
if __name__ == "__main__":
    # S_grid, V, i_indices, A, B, C, M, option_price = bs_pde_implicit_fdm(100, 100, 1, 0.05, 0.2, option_type="call")
    # print("S_grid shape:", S_grid.shape)
    # print("First 5 grid values:", S_grid[:5])
    # print("First 5 payoff values:", V[:5])
    # print("Last 5 payoff values:", V[-5:])

    # print("i_indices shape:", i_indices.shape)
    # print("A shape:", A.shape)
    # print("B shape:", B.shape)
    # print("C shape:", C.shape)
    # print("M shape:", M.shape)


    import matplotlib.pyplot as plt

    S0, K, T, r, sigma = 100, 120, 1.0, 0.05, 0.2

    # Run the PDE solver to get price and the solved array V_today
    price, S_grid, V_today = bs_pde_implicit_fdm(
        S0, K, T, r, sigma, N_S=200, N_t=1000, option_type="call"
    )

    V_maturity = np.maximum(S_grid - K, 0.0)

    plt.figure(figsize=(10, 6))

    # Payoff at T (Red dashed)
    plt.plot(
        S_grid,
        V_maturity,
        label="Payoff at Maturity (t = T)",
        linestyle="--",
        color="red",
    )

    # Option Value Today (Blue solid)
    plt.plot(
        S_grid,
        V_today,
        label="PDE Option Value Today (t = 0)",
        color="blue",
        linewidth=2,
    )

    # Key Price Reference Lines
    plt.axvline(
        x=S0,
        color="grey",
        linestyle=":",
        label=f"Current Stock / Strike (S0 = K = {S0})",
    )
    plt.axhline(
        y=price,
        color="green",
        linestyle=":",
        label=f"Option Price at S0 = £{price:.2f}",
    )

    plt.title("Black-Scholes PDE Option Pricing Surface Evolution")
    plt.xlabel("Stock Price (S)")
    plt.ylabel("Option Value (V)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    # print(option_price)
    





