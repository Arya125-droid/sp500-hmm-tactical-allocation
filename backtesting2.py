import numpy as np
import pandas as pd
import yfinance as yf

np.random.seed(42)

class MarketRegimeModel:
    def __init__(self, threshold=0.0075):
        self.state_names = ["Bear", "Sideways", "Bull"]
        self.transition_matrix = None
        self.num_states = len(self.state_names)
        self.threshold = threshold

    def discretize_returns(self, returns):
        states = []
        for r in returns:
            if r < -self.threshold:
                states.append(0)
            elif r <= self.threshold:
                states.append(1)
            else:
                states.append(2)
        return np.array(states)

    def fit(self, returns):
        states = self.discretize_returns(returns)
        counts = np.zeros((self.num_states, self.num_states))
        
        for (current_state, next_state) in zip(states[:-1], states[1:]):
            counts[current_state, next_state] += 1
            
        row_sums = counts.sum(axis=1, keepdims=True)
        self.transition_matrix = np.divide(
            counts, row_sums, 
            where=row_sums != 0, 
            out=np.zeros_like(counts)
        )
        return self.transition_matrix

# --- 1. Fetch Real Market Data Once ---
print("Downloading S&P 500 (^GSPC) historical data...")
ticker = yf.Ticker("^GSPC")
df = ticker.history(period="10y")
df['Return'] = df['Close'].pct_change()
df = df.dropna()
returns_array = df['Return'].values

# Helper function compatible with NumPy arrays
def max_drawdown(wealth_series):
    peak = np.maximum.accumulate(wealth_series)
    drawdown = (wealth_series - peak) / peak
    return drawdown.min()

# --- 2. Test Different Thresholds ---
thresholds_to_test = [0.0050, 0.0075, 0.0125]  # 0.50%, 0.75%, 1.25%

print("\n--- Comparative Backtest Results ($1 Initial Investment) ---")
print(f"{'Threshold':<12} {'Final Wealth':<15} {'Max Drawdown':<15}")
print("-" * 44)

# Baseline Buy & Hold
bh_wealth = 1 * (1 + df['Return']).cumprod()
bh_final = bh_wealth.iloc[-1]
bh_mdd = max_drawdown(bh_wealth.values)
print(f"{'Buy & Hold':<12} ${bh_final:<14,.2f} {bh_mdd*100:<14.2f}%")

for th in thresholds_to_test:
    model = MarketRegimeModel(threshold=th)
    model.fit(returns_array)
    states_array = model.discretize_returns(returns_array)
    
    df_temp = df.iloc[1:].copy()
    regimes_today = states_array[:-1]
    
    # Strategy allocation rule based on regime
    weights = np.where(regimes_today == 2, 1.0, np.where(regimes_today == 1, 0.5, 0.0))
    
    strategy_returns = weights * df_temp['Return'].values
    strategy_wealth = 1 * (1 + strategy_returns).cumprod()
    
    strat_final = strategy_wealth[-1]
    strat_mdd = max_drawdown(strategy_wealth)
    
    label = f"±{th*100:.2f}%"
    print(f"{label:<12} ${strat_final:<14,.2f} {strat_mdd*100:<14.2f}%")