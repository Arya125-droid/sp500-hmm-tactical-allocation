import numpy as np
import pandas as pd
import yfinance as yf

# Set random seed for reproducibility
np.random.seed(42)

class MarketRegimeModel:
    def __init__(self):
        self.state_names = ["Bear", "Sideways", "Bull"]
        self.transition_matrix = None
        self.num_states = len(self.state_names)

    def discretize_returns(self, returns):
        """
        Convert continuous daily returns into discrete market regimes:
        State 0: Bear (Return < -0.75%)
        State 1: Sideways (-0.75% <= Return <= 0.75%)
        State 2: Bull (Return > 0.75%)
        """
        states = []
        for r in returns:
            if r < -0.0075:
                states.append(0)
            elif r <= 0.0075:
                states.append(1)
            else:
                states.append(2)
        return np.array(states)

    def fit(self, returns):
        """Estimate the transition probability matrix from historical returns data."""
        states = self.discretize_returns(returns)
        
        # Initialize transition count matrix
        counts = np.zeros((self.num_states, self.num_states))
        
        # Count transitions from state i to state j
        for (current_state, next_state) in zip(states[:-1], states[1:]):
            counts[current_state, next_state] += 1
            
        # Normalize rows to turn counts into probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        self.transition_matrix = np.divide(
            counts, row_sums, 
            where=row_sums != 0, 
            out=np.zeros_like(counts)
        )
        return self.transition_matrix

    def stationary_distribution(self):
        """Calculate long-term steady-state probabilities using eigenvalues."""
        eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
        idx = np.argmin(np.abs(eigenvals - 1.0))
        stat_dist = np.real(eigenvecs[:, idx])
        return stat_dist / stat_dist.sum()

# --- 1. Fetch Real Market Data ---
print("Downloading S&P 500 (^GSPC) historical data...")
ticker = yf.Ticker("^GSPC")
df = ticker.history(period="10y")

# Calculate daily percentage returns
df['Return'] = df['Close'].pct_change()
df = df.dropna()

# --- 2. Initialize and Fit Model ---
model = MarketRegimeModel()
returns_array = df['Return'].values
t_matrix = model.fit(returns_array)
states_array = model.discretize_returns(returns_array)

# Add states back to dataframe for backtesting alignment
df['Regime'] = states_array

# --- 3. Run Strategy Backtest ---
# Strategy rule:
# If regime today is Bull (2) -> weight tomorrow = 1.0
# If regime today is Sideways (1) -> weight tomorrow = 0.5
# If regime today is Bear (0) -> weight tomorrow = 0.0 (Cash, 0% return)

weights = []
for regime in df['Regime'][:-1]: # exclude last row since we need next day's return
    if regime == 2:
        weights.append(1.0)
    elif regime == 1:
        weights.append(0.5)
    else:
        weights.append(0.0)

# Align weights with returns starting from index 1
strategy_returns = weights * df['Return'].values[1:]

# Calculate cumulative growth of $1,000 initial investment
df_backtest = df.iloc[1:].copy()
df_backtest['Strategy_Return'] = strategy_returns

df_backtest['Buy_Hold_Wealth'] = 1000 * (1 + df_backtest['Return']).cumprod()
df_backtest['Strategy_Wealth'] = 1000 * (1 + df_backtest['Strategy_Return']).cumprod()

# --- 4. Print Results ---
print("\n--- 10-Year Backtest Results ($1,000 Initial Investment) ---")
final_bh = df_backtest['Buy_Hold_Wealth'].iloc[-1]
final_strat = df_backtest['Strategy_Wealth'].iloc[-1]

print(f"Buy & Hold Final Value: ${final_bh:,.2f}")
print(f"Markov Strategy Final Value: ${final_strat:,.2f}")

# Calculate max drawdown function for comparison
def max_drawdown(wealth_series):
    peak = wealth_series.cummax()
    drawdown = (wealth_series - peak) / peak
    return drawdown.min()

bh_mdd = max_drawdown(df_backtest['Buy_Hold_Wealth'])
strat_mdd = max_drawdown(df_backtest['Strategy_Wealth'])

print(f"\nBuy & Hold Max Drawdown: {bh_mdd * 100:.2f}%")
print(f"Markov Strategy Max Drawdown: {strat_mdd * 100:.2f}%")