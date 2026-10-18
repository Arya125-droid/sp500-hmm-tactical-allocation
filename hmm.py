import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

np.random.seed(42)

# --- 1. Fetch Real Market Data ---
print("Downloading S&P 500 (^GSPC) historical data...")
ticker = yf.Ticker("^GSPC")
df = ticker.history(period="10y")
df['Return'] = df['Close'].pct_change()
# Add rolling volatility as a second feature to help the HMM distinguish regimes
df['Volatility'] = df['Return'].rolling(window=21).std()
df = df.dropna()

# Prepare features for HMM (returns and volatility)
X = np.column_stack([df['Return'].values, df['Volatility'].values])

# --- 2. Fit Hidden Markov Model (2 States) ---
print("Fitting Gaussian Hidden Markov Model...")
# 2 States: State A (Low Vol / Bull) and State B (High Vol / Bear)
hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
hmm_model.fit(X)
hidden_states = hmm_model.predict(X)

# Automatically identify which state is the 'Bear' state (the one with the lower mean return)
state_means = hmm_model.means_
bear_state_idx = np.argmin(state_means[:, 0])
print(f"Identified Hidden State {bear_state_idx} as the Bear/High-Risk Regime.")

# --- 3. Run Strategy Backtest ---
# Allocation rule: If HMM predicts Bear state tomorrow, drop to 50% equity. Otherwise 100%.
weights = []
for state in hidden_states[:-1]:
    if state == bear_state_idx:
        weights.append(0.5)  # Defensive stance during HMM-detected bear regime
    else:
        weights.append(1.0)  # Fully invested during bull regime

df_strat = df.iloc[1:].copy()
strat_returns = weights * df_strat['Return'].values
strat_wealth = 1000 * (1 + strat_returns).cumprod()

bh_returns = df['Return'].values
bh_wealth = 1000 * (1 + bh_returns).cumprod()

# --- 4. Performance Metrics Helpers ---
def max_drawdown(wealth_series):
    peak = np.maximum.accumulate(wealth_series)
    drawdown = (wealth_series - peak) / peak
    return drawdown.min()

def annualized_sharpe(returns_series, periods_per_year=252):
    mean_ret = np.mean(returns_series)
    std_ret = np.std(returns_series)
    if std_ret == 0: return 0.0
    return (mean_ret / std_ret) * np.sqrt(periods_per_year)

bh_final = bh_wealth[-1]
strat_final = strat_wealth[-1]

bh_mdd = max_drawdown(bh_wealth)
strat_mdd = max_drawdown(strat_wealth)

bh_sharpe = annualized_sharpe(bh_returns)
strat_sharpe = annualized_sharpe(strat_returns)

# --- Print Results ---
print("\n" + "="*55)
print(f"{'Performance Metric':<25} {'Buy & Hold':<12} {'HMM Strategy':<15}")
print("="*55)
print(f"{'Final Portfolio Value':<25} ${bh_final:<11,.2f} ${strat_final:<14,.2f}")
print(f"{'Maximum Drawdown':<25} {bh_mdd*100:<11.2f}% {strat_mdd*100:<14.2f}%")
print(f"{'Sharpe Ratio':<25} {bh_sharpe:<11.2f} {strat_sharpe:<14.2f}")
print("="*55)

# --- 5. Plotting the Results ---
plt.figure(figsize=(12, 6))
plt.plot(df.index, bh_wealth, label='Buy & Hold (S&P 500)', color='black', alpha=0.7, linewidth=1.5)
plt.plot(df_strat.index, strat_wealth, label='Hidden Markov Model (HMM) Strategy', color='purple', linewidth=1.8)

plt.title('Hidden Markov Model Regime-Switching vs. Buy & Hold (10-Year Backtest)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Portfolio Value ($)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

print("\nGenerating HMM performance comparison chart...")
plt.show()