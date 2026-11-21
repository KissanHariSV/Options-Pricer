"""Demo: price one option four ways and compare.

Run:  python run.py
"""
import numpy as np
import matplotlib.pyplot as plt

from options import (black_scholes, greeks, binomial_tree, monte_carlo,
                     implied_vol)

# An at-the-money 1-year call: S = K = 100, r = 5%, vol = 20%.
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

print(f"Option: {S=}, {K=}, {T=}y, {r=}, {sigma=}\n")

# 1) Black-Scholes closed form + Greeks
bs_call = black_scholes(S, K, T, r, sigma, "call")
bs_put = black_scholes(S, K, T, r, sigma, "put")
print(f"Black-Scholes call: {bs_call:.4f}   put: {bs_put:.4f}")
print("Put-call parity check: C - P should equal S - K*exp(-rT)")
print(f"  C - P = {bs_call - bs_put:.4f},  S - K*exp(-rT) = "
      f"{S - K * np.exp(-r * T):.4f}\n")

g = greeks(S, K, T, r, sigma, "call")
print("Call Greeks:")
for name, val in g.items():
    print(f"  {name:6s} {val: .4f}")
print("  (theta is per year; per day = theta/365 = "
      f"{g['theta'] / 365:.4f})\n")

# 2) Binomial tree convergence to Black-Scholes
print("Binomial tree (CRR) convergence:")
steps_list = [10, 50, 100, 500, 1000]
bin_prices = []
for n in steps_list:
    price = binomial_tree(S, K, T, r, sigma, n, "call")
    bin_prices.append(price)
    print(f"  N={n:5d}  price={price:.4f}  error={price - bs_call:+.4f}")
am_put = binomial_tree(S, K, T, r, sigma, 500, "put", american=True)
print(f"American put (N=500): {am_put:.4f} vs European {bs_put:.4f} "
      f"-> early-exercise premium = {am_put - bs_put:.4f}\n")

# 3) Monte Carlo with standard error
mc_price, mc_se = monte_carlo(S, K, T, r, sigma, 200_000, "call")
print(f"Monte Carlo (200k paths, antithetic): {mc_price:.4f} "
      f"+/- {1.96 * mc_se:.4f} (95% CI)")
print(f"  True (BS) value {bs_call:.4f} inside CI: "
      f"{abs(mc_price - bs_call) < 1.96 * mc_se}\n")

# 4) Implied volatility round-trip
iv = implied_vol(bs_call, S, K, T, r, "call")
print(f"Implied vol of the BS price: {iv:.4f} (true sigma = {sigma})\n")

# Charts: convergence of the two numerical methods.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(steps_list, bin_prices, "o-")
axes[0].axhline(bs_call, color="grey", ls="--", label="Black-Scholes")
axes[0].set_xscale("log")
axes[0].set_xlabel("Binomial steps (N, log scale)")
axes[0].set_ylabel("Call price")
axes[0].set_title("Binomial convergence")
axes[0].legend()

paths_list = [1_000, 5_000, 20_000, 100_000, 500_000]
mc_vals, mc_errs = [], []
for n in paths_list:
    p, se = monte_carlo(S, K, T, r, sigma, n, "call")
    mc_vals.append(p)
    mc_errs.append(1.96 * se)
axes[1].errorbar(paths_list, mc_vals, yerr=mc_errs, fmt="o-", capsize=3)
axes[1].axhline(bs_call, color="grey", ls="--", label="Black-Scholes")
axes[1].set_xscale("log")
axes[1].set_xlabel("Monte Carlo paths (log scale)")
axes[1].set_ylabel("Call price")
axes[1].set_title("Monte Carlo convergence (95% CI)")
axes[1].legend()

plt.tight_layout()
plt.savefig("convergence.png", dpi=120)
print("Saved convergence.png")
plt.show()  # opens a chart window; close it to end the script
