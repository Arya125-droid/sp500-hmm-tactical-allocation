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
df['Volatility'] = df['Return'].rolling(window=21).std()
df = df.dropna()

X = np.column_stack([df['Return'].values, df['Volatility'].values])

# --- 2. Fit 3-State Hidden Markov Model ---
print("Fitting 3-State Gaussian Hidden Markov Model...")
hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, tol=1e-4, random_state=42)
hmm_model.fit(X)
hidden_states = hmm_model.predict(X)

state_means = hmm_model.means_[:, 0]
sorted_state_indices = np.argsort(state_means)

bear_state = sorted_state_indices[0]
sideways_state = sorted_state_indices[1]
bull_state = sorted_state_indices[2]

# --- 3. Run Strategy Backtest with Weights & Transaction Fees ---
weights = []
for state in hidden_states[:-1]:
    if state == bull_state:
        weights.append(1.0)
    elif state == sideways_state:
        weights.append(0.9)
    else:
        weights.append(0.5)

df_strat = df.iloc[1:].copy()
raw_strat_returns = np.array(weights) * df_strat['Return'].values

# Transaction fee logic: 5 bps (0.0005) per unit of portfolio rebalanced
fee_rate = 0.0005
strat_returns_net = []
for i in range(len(raw_strat_returns)):
    r = raw_strat_returns[i]
    if i > 0:
        # Cost is proportional to the absolute change in equity weight
        rebalance_cost = abs(weights[i] - weights[i-1]) * fee_rate
        r -= rebalance_cost
    strat_returns_net.append(r)

strat_returns_net = np.array(strat_returns_net)
strat_wealth = 1000 * (1 + strat_returns_net).cumprod()

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
strat_sharpe = annualized_sharpe(strat_returns_net)

# --- Print Results ---
print("\n" + "="*60)
print(f"{'Performance Metric':<25} {'Buy & Hold':<12} {'HMM + Fees (Net)':<20}")
print("="*60)
print(f"{'Final Portfolio Value':<25} ${bh_final:<11,.2f} ${strat_final:<19,.2f}")
print(f"{'Maximum Drawdown':<25} {bh_mdd*100:<11.2f}% {strat_mdd*100:<19.2f}%")
print(f"{'Sharpe Ratio':<25} {bh_sharpe:<11.2f} {strat_sharpe:<19.2f}")
print("="*60)

# --- 5. Plotting the Results ---
plt.figure(figsize=(12, 6))
plt.plot(df.index, bh_wealth, label='Buy & Hold (S&P 500)', color='black', alpha=0.7, linewidth=1.5)
plt.plot(df_strat.index, strat_wealth, label='Optimized 3-State HMM (Net of Fees)', color='purple', linewidth=1.8)

plt.title('HMM Strategy with Transaction Fees vs. Buy & Hold (10-Year Backtest)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Portfolio Value ($)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

print("\nGenerating performance chart with transaction fees...")
plt.show()