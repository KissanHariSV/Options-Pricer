"""Options pricing from scratch: Black-Scholes, Greeks, binomial tree,
Monte Carlo, and implied volatility.

Conventions
-----------
- S: spot price, K: strike, T: time to expiry in YEARS, r: risk-free rate
  (continuously compounded), sigma: volatility (annualised).
- All prices are for European options unless `american=True` is passed to
  the binomial pricer.
- Theta is returned per YEAR; traders usually quote it per day (divide by 365).
- Vega is returned per 1.0 of volatility (i.e. per 100 vol points);
  per 1 vol point, divide by 100.

The key idea behind everything here is risk-neutral pricing: in a
no-arbitrage market, an option is worth the discounted expected payoff
under the risk-neutral measure, where the stock grows at the risk-free
rate r (not at its real-world expected return).
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ---------------------------------------------------------------- Black-Scholes
def _d1_d2(S, K, T, r, sigma):
    """The two standardised moneyness terms in Black-Scholes."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes(S, K, T, r, sigma, kind="call"):
    """Closed-form European option price.

    Call = S*N(d1) - K*exp(-rT)*N(d2)
    Put  = K*exp(-rT)*N(-d2) - S*N(-d1)
    """
    if T <= 0:  # expired: worth its intrinsic value
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if kind == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


def greeks(S, K, T, r, sigma, kind="call"):
    """First-order sensitivities of the option price.

    delta: dPrice/dS        gamma: dDelta/dS = d2Price/dS2
    theta: dPrice/dT        vega:  dPrice/dSigma
    rho:   dPrice/dr
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T)
    if kind == "call":
        delta = norm.cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif kind == "put":
        delta = norm.cdf(d1) - 1
        theta = (-(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("kind must be 'call' or 'put'")
    return {"delta": delta, "gamma": gamma, "theta": theta,
            "vega": vega, "rho": rho}


# ------------------------------------------------------------- Binomial (CRR)
def binomial_tree(S, K, T, r, sigma, n_steps=100, kind="call", american=False):
    """Cox-Ross-Rubinstein binomial tree.

    At each step the stock moves up by u = exp(sigma*sqrt(dt)) or down by
    d = 1/u. The risk-neutral probability of an up move is
    p = (exp(r*dt) - d) / (u - d). We fill in the payoff at expiry and
    work backwards, discounting by exp(-r*dt) each step. For American
    options we also allow early exercise at every node.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Stock prices at expiry, then option payoff at expiry.
    j = np.arange(n_steps + 1)  # number of up moves
    stock = S * (u ** j) * (d ** (n_steps - j))
    value = np.maximum(stock - K, 0.0) if kind == "call" \
        else np.maximum(K - stock, 0.0)

    # Backward induction.
    for _ in range(n_steps, 0, -1):
        value = disc * (p * value[1:] + (1 - p) * value[:-1])
        if american:
            # Recompute stock prices at this step for early exercise.
            n = len(value)
            j = np.arange(n)
            stock = S * (u ** j) * (d ** ((n - 1) - j))
            exercise = np.maximum(stock - K, 0.0) if kind == "call" \
                else np.maximum(K - stock, 0.0)
            value = np.maximum(value, exercise)
    return float(value[0])


# ---------------------------------------------------------------- Monte Carlo
def monte_carlo(S, K, T, r, sigma, n_paths=100_000, kind="call",
                seed=42, antithetic=True):
    """Price by simulating terminal stock prices under risk-neutral GBM.

    S_T = S * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z), Z ~ N(0,1).
    Antithetic variates (using +Z and -Z together) cut the variance of the
    estimate roughly in half for free. Returns (price, std_error).
    """
    rng = np.random.default_rng(seed)
    n = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal(n)
    if antithetic:
        z = np.concatenate([z, -z])
    s_t = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(s_t - K, 0.0) if kind == "call" \
        else np.maximum(K - s_t, 0.0)
    discounted = np.exp(-r * T) * payoff
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return float(price), float(std_error)


# --------------------------------------------------------- Implied volatility
def implied_vol(market_price, S, K, T, r, kind="call"):
    """The volatility that makes the Black-Scholes price equal the market
    price, found by Brent's method (a robust root-finder).

    This is the market's consensus forecast of future volatility, and it
    is how traders actually quote options. No closed form exists, so every
    trading system solves this numerically, thousands of times a day.
    """
    intrinsic = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    if market_price < intrinsic - 1e-8:
        raise ValueError("market price below intrinsic value: no solution")
    f = lambda sig: black_scholes(S, K, T, r, sig, kind) - market_price
    return float(brentq(f, 1e-6, 5.0))
