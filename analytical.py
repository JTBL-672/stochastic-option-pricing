import numpy as np
from scipy.stats import norm

def black_scholes_analytical(S0, K, T, r, sigma, option_type="call"):
    # 1. Calculate d1 using S0, K, T, r, and sigma
    d1 = (np.log(S0/K) + (r+(sigma**2)/2)*T)/(sigma*np.sqrt(T))

    # 2. Calculate d2 using d1, sigma, and T
    d2 = d1 - (sigma*np.sqrt(T))

    # 3. Calculate the Call or Put price based on option_type
    if option_type == "call":
        price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return price
    elif option_type == "put":
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
        return price

    # 4. Return the calculated price
    else:
        return "Error, invalid option type. Try again."
    

print(black_scholes_analytical(100, 100, 1, 0.05, 0.2, "put"))