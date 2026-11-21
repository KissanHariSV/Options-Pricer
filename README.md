# Project 1 — Options Pricer

Price a European option four ways and prove the methods agree: the
Black–Scholes closed form, the Greeks, a Cox–Ross–Rubinstein binomial
tree, Monte Carlo simulation, and implied volatility by root-finding.

## What you'll learn

- **Risk-neutral pricing** — the single idea underneath all derivatives
  pricing: an option is worth its discounted expected payoff in a world
  where every asset grows at the risk-free rate.
- **Black–Scholes** — the formula, what d1 and d2 mean, and why the
  "real" expected return of the stock never appears in it.
- **The Greeks** — delta, gamma, theta, vega, rho: what each one
  measures, its sign, and how a trading desk uses it to hedge.
- **Numerical methods** — binomial trees (which also handle American
  early exercise) and Monte Carlo with variance reduction (antithetic
  variates), including how to report a standard error with a price.
- **Implied volatility** — how the market's vol forecast is backed out
  of a price, and why traders quote vol instead of dollars.

## How to run

```bash
pip install -r requirements.txt
python run.py            # prices one option four ways, saves convergence.png
python test_options.py   # 11 sanity tests (parity, bounds, convergence, SVI)
python smile.py               # real SPY vol smile, 4 expiries (needs internet), saves smile_SPY.png
python smile.py AAPL 14 60    # any ticker, your own target days to expiry
python surface.py             # 3D SPY vol surface, all expiries up to 1 year, saves surface_SPY.png
python surface.py AAPL 180    # any ticker, expiries up to 180 days
python svi_surface.py         # fit SVI to every expiry, check for arbitrage, saves svi_SPY.png
python vol_premium.py         # implied vs realised vol (today + VIX since 1993), saves vol_premium_SPY.png
```

## Try this next

1. **Volatility smile:** price the same option at strikes 80–120, back
   out implied vol for each, and plot vol vs. strike. Real markets show
   a smile/skew — Black–Scholes assumes flat vol. Why?
2. **Delta hedging simulation:** simulate a stock path, delta-hedge a
   short call daily, and see how close the hedged P&L comes to the
   Black–Scholes price. Re-hedge weekly instead — what breaks?
3. **Dividends:** add a continuous dividend yield `q` to Black–Scholes
   and check put–call parity still holds with the new formula.
4. **Barrier option:** price a knock-out call with Monte Carlo (path
   dependent — no closed form needed) and compare against the
   analytical formula.
