import time
from src.analytical import black_scholes_analytical
from src.monte_carlo import bs_monte_carlo
from src.pde_solver import bs_pde_implicit_fdm


def run_benchmark():
    # Parameters
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    print("==========================================================")
    print("      STOCHASTIC OPTION PRICING ENGINE BENCHMARK          ")
    print("==========================================================\n")

    for option_type in ["call", "put"]:
        print(f"--- Option Type: {option_type.upper()} ---")

        # 1. Analytical Benchmark
        t0 = time.perf_counter()
        analytical_price = black_scholes_analytical(
            S0, K, T, r, sigma, option_type=option_type
        )
        t_analytical = (time.perf_counter() - t0) * 1000

        # 2. Implicit PDE Solver
        t0 = time.perf_counter()
        pde_price = bs_pde_implicit_fdm(
            S0,
            K,
            T,
            r,
            sigma,
            N_S=200,
            N_t=1000,
            option_type=option_type,
        )
        t_pde = (time.perf_counter() - t0) * 1000
        pde_error = (
            abs(pde_price - analytical_price) / analytical_price
        ) * 100

        # 3. Antithetic Monte Carlo
        t0 = time.perf_counter()
        mc_price, mc_se = bs_monte_carlo(
            S0,
            K,
            T,
            r,
            sigma,
            N_sims=100_000,
            use_antithetic=True,
            option_type=option_type,
        )
        t_mc = (time.perf_counter() - t0) * 1000
        mc_error = abs(mc_price - analytical_price) / analytical_price * 100

        # Print Comparative Results
        print(
            f"Analytical Price : £{analytical_price:.4f}  | Time: {t_analytical:.3f} ms"
        )
        print(
            f"Implicit PDE Price: £{pde_price:.4f}  | Time: {t_pde:.3f} ms | Error: {pde_error:.3f}%"
        )
        print(
            f"Monte Carlo Price : £{mc_price:.4f}  | Time: {t_mc:.3f} ms | Error: {mc_error:.3f}% (SE: +/- £{mc_se:.4f})\n"
        )


if __name__ == "__main__":
    run_benchmark()

