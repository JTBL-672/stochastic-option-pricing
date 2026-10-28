import numpy as np
from scipy.linalg import solve_banded


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
    # 1. Grid Setup
    S_max = K * S_max_mult
    dt = T / N_t
    S_grid = np.linspace(0, S_max, N_S + 1)

    # 2. Terminal Payoff
    option_type = option_type.lower()
    if option_type == "call":
        V = np.maximum(S_grid - K, 0.0)
    elif option_type == "put":
        V = np.maximum(K - S_grid, 0.0)
    else:
        raise ValueError(
            f"Invalid option_type '{option_type}'. Expected 'call' or 'put'."
        )

    # 3. Tridiagonal Coefficients
    i_indices = np.arange(1, N_S)
    A = 0.5 * dt * (r * i_indices - (sigma**2) * (i_indices**2))
    B = 1.0 + dt * (r + (sigma**2) * (i_indices**2))
    C = -0.5 * dt * (r * i_indices + (sigma**2) * (i_indices**2))

    # 4. Form Compact Banded Matrix Representation for SciPy
    # SciPy solve_banded expects matrix shape (l + u + 1, N) where:
    # Row 0: Upper diagonal (C) shifted right
    # Row 1: Main diagonal (B)
    # Row 2: Lower diagonal (A) shifted left
    ab = np.zeros((3, N_S - 1))
    ab[0, 1:] = C[:-1]  # Super-diagonal
    ab[1, :] = B  # Main diagonal
    ab[2, :-1] = A[1:]  # Sub-diagonal

    # 5. Backward Time-Stepping Loop
    V_current = V.copy()
    for step in range(1, N_t + 1):
        tau = step * dt

        if option_type == "call":
            V_lower = 0.0
            V_upper = S_max - K * np.exp(-r * tau)
        else:
            V_lower = K * np.exp(-r * tau)
            V_upper = 0.0

        rhs = V_current[1:-1].copy()
        rhs[0] -= A[0] * V_lower
        rhs[-1] -= C[-1] * V_upper

        # Fast O(N) Banded Solver
        V_interior = solve_banded((1, 1), ab, rhs)
        V_current = np.concatenate([[V_lower], V_interior, [V_upper]])

    return float(np.interp(S0, S_grid, V_current))
