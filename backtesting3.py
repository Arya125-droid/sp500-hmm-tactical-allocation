import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

np.random.seed(42)

# --- 1. Fetch Real Market Data ---
print("Downloading S&P 500 (^GSPC) historical data...")
ticker = yf.Ticker("^GSPC")
df = ticker.history(period="10y")
df['Return'] = df['Close'].pct_change()
df = df.dropna()

# Discretization helper
def discretize_returns(returns, threshold=0.01):
    states = []
    for r in returns:
        if r < -threshold:
            states.append(0) # Bear
        elif r <= threshold:
            states.append(1) # Sideways
        else:
            states.append(2) # Bull
    return np.array(states)

# Helper function for max drawdown
def max_drawdown(wealth_series):
    peak = np.maximum.accumulate(wealth_series)
    drawdown = (wealth_series - peak) / peak
    return drawdown.min()

# Helper function for annualized Sharpe Ratio
def annualized_sharpe(returns_series, periods_per_year=252):
    mean_ret = np.mean(returns_series)
    std_ret = np.std(returns_series)
    if std_ret == 0:
        return 0.0
    return (mean_ret / std_ret) * np.sqrt(periods_per_year)

# --- Baseline Strategy: Buy & Hold ---
bh_returns = df['Return'].values
bh_wealth = 1000 * (1 + bh_returns).cumprod()

# --- Balanced Markov Strategy ---
daily_states = discretize_returns(bh_returns, threshold=0.01)
df['Regime'] = daily_states

weights = []
for regime in df['Regime'][:-1]:
    if regime == 2:   # Bull: Fully invested
        weights.append(1.0)
    elif regime == 1: # Sideways: Mild buffer
        weights.append(0.8)
    else:             # Bear: Partial defense (keep 50% to catch recoveries!)
        weights.append(0.5)

df_strat = df.iloc[1:].copy()
strat_returns = weights * df_strat['Return'].values
strat_wealth = 1000 * (1 + strat_returns).cumprod()

# --- Calculate Metrics ---
bh_final = bh_wealth[-1]
strat_final = strat_wealth[-1]

bh_mdd = max_drawdown(bh_wealth)
strat_mdd = max_drawdown(strat_wealth)

bh_sharpe = annualized_sharpe(bh_returns)
strat_sharpe = annualized_sharpe(strat_returns)

# --- Print Formatted Results ---
print("\n" + "="*50)
print(f"{'Performance Metric':<25} {'Buy & Hold':<12} {'Balanced Markov':<15}")
print("="*50)
print(f"{'Final Portfolio Value':<25} ${bh_final:<11,.2f} ${strat_final:<14,.2f}")
print(f"{'Maximum Drawdown':<25} {bh_mdd*100:<11.2f}% {strat_mdd*100:<14.2f}%")
print(f"{'Sharpe Ratio':<25} {bh_sharpe:<11.2f} {strat_sharpe:<14.2f}")
print("="*50)

# --- Plotting the Results ---
plt.figure(figsize=(12, 6))
plt.plot(df.index, bh_wealth, label='Buy & Hold (S&P 500)', color='black', alpha=0.7, linewidth=1.5)
plt.plot(df_strat.index, strat_wealth, label='Balanced Markov Strategy', color='royalblue', linewidth=1.8)

plt.title('Balanced Regime-Switching Strategy vs. Buy & Hold (10-Year Backtest)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Portfolio Value ($)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

print("\nGenerating performance comparison chart...")
plt.show()