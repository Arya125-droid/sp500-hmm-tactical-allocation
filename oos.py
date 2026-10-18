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

df['Year'] = df.index.year
years = sorted(df['Year'].unique())

warmup_years = 3
train_years_pool = years[:warmup_years]
test_years_pool = years[warmup_years:]

print(f"Initial Training Window: {train_years_pool[0]} - {train_years_pool[-1]}")
print(f"Walk-Forward OOS Testing Window: {test_years_pool[0]} - {test_years_pool[-1]}")

# --- 2. Walk-Forward Loop ---
oos_dates = []
oos_returns = []
oos_weights = []

for test_year in test_years_pool:
    print(f"Walk-forward step: Predicting year {test_year}...")
    
    train_df = df[df['Year'] < test_year]
    test_df = df[df['Year'] == test_year]
    
    if len(test_df) == 0:
        continue
        
    X_train = np.column_stack([train_df['Return'].values, train_df['Volatility'].values])
    X_test = np.column_stack([test_df['Return'].values, test_df['Volatility'].values])
    
    # Use covariance_type="diag" to prevent matrix singularity during extreme volatility shocks
    hmm_model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, tol=1e-4, random_state=42)
    hmm_model.fit(X_train)
    
    state_means = hmm_model.means_[:, 0]
    sorted_state_indices = np.argsort(state_means)
    bear_state = sorted_state_indices[0]
    sideways_state = sorted_state_indices[1]
    bull_state = sorted_state_indices[2]
    
    test_states = hmm_model.predict(X_test)
    
    year_weights = []
    for state in test_states:
        if state == bull_state:
            year_weights.append(1.0)
        elif state == sideways_state:
            year_weights.append(0.9)
        else:
            year_weights.append(0.5)
            
    oos_dates.extend(test_df.index)
    oos_returns.extend(test_df['Return'].values)
    oos_weights.extend(year_weights)

# --- 3. Assemble OOS Backtest & Performance Metrics ---
oos_df = pd.DataFrame({
    'Return': oos_returns,
    'Weight': oos_weights
}, index=oos_dates)

oos_df['Exec_Weight'] = oos_df['Weight'].shift(1).fillna(1.0)
oos_df['Strategy_Return'] = oos_df['Exec_Weight'] * oos_df['Return']

fee_rate = 0.0005
net_strat_returns = []
for i in range(len(oos_df)):
    r = oos_df['Strategy_Return'].iloc[i]
    if i > 0:
        weight_change = abs(oos_df['Exec_Weight'].iloc[i] - oos_df['Exec_Weight'].iloc[i-1])
        r -= weight_change * fee_rate
    net_strat_returns.append(r)

oos_df['Net_Return'] = net_strat_returns

benchmark_slice = df.loc[oos_df.index]
oos_df['BH_Wealth'] = 1000 * (1 + benchmark_slice['Return']).cumprod()
oos_df['OOS_Wealth'] = 1000 * (1 + oos_df['Net_Return']).cumprod()

def max_drawdown(wealth_series):
    peak = np.maximum.accumulate(wealth_series)
    drawdown = (wealth_series - peak) / peak
    return drawdown.min()

def annualized_sharpe(returns_series, periods_per_year=252):
    mean_ret = np.mean(returns_series)
    std_ret = np.std(returns_series)
    if std_ret == 0: return 0.0
    return (mean_ret / std_ret) * np.sqrt(periods_per_year)

bh_final = oos_df['BH_Wealth'].iloc[-1]
strat_final = oos_df['OOS_Wealth'].iloc[-1]

bh_mdd = max_drawdown(oos_df['BH_Wealth'].values)
strat_mdd = max_drawdown(oos_df['OOS_Wealth'].values)

bh_sharpe = annualized_sharpe(benchmark_slice['Return'].values)
strat_sharpe = annualized_sharpe(oos_df['Net_Return'].values)

print("\n" + "="*65)
print(f"{'Performance Metric (OOS Period)':<30} {'Buy & Hold':<12} {'Walk-Forward HMM':<20}")
print("="*65)
print(f"{'Final Portfolio Value':<30} ${bh_final:<11,.2f} ${strat_final:<19,.2f}")
print(f"{'Maximum Drawdown':<30} {bh_mdd*100:<11.2f}% {strat_mdd*100:<19.2f}%")
print(f"{'Sharpe Ratio':<30} {bh_sharpe:<11.2f} {strat_sharpe:<19.2f}")
print("="*65)

plt.figure(figsize=(12, 6))
plt.plot(oos_df.index, oos_df['BH_Wealth'], label='Buy & Hold (OOS)', color='black', alpha=0.7, linewidth=1.5)
plt.plot(oos_df.index, oos_df['OOS_Wealth'], label='Walk-Forward 3-State HMM (OOS)', color='crimson', linewidth=1.8)
plt.title('Out-of-Sample Walk-Forward HMM Strategy vs. Buy & Hold', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Portfolio Value ($)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

print("\nGenerating walk-forward OOS performance chart...")
plt.show()