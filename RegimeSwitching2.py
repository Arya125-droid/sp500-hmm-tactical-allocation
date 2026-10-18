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

    def simulate(self, start_state_idx, steps=30):
        """Simulate future market regimes step-by-step."""
        current = start_state_idx
        path = [current]
        
        for _ in range(steps):
            next_state = np.random.choice(
                self.num_states, 
                p=self.transition_matrix[current]
            )
            path.append(next_state)
            current = next_state
            
        return [self.state_names[i] for i in path]

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
print(df.tail())

# Calculate daily percentage returns and drop missing values
daily_returns = df['Close'].pct_change().dropna().values

# --- 2. Initialize and Fit Model ---
model = MarketRegimeModel()
t_matrix = model.fit(daily_returns)

print("\n--- Estimated Transition Matrix (S&P 500 Real Data) ---")
print(f"{'':<10} {'Bear':<10} {'Sideways':<10} {'Bull':<10}")
for i, name in enumerate(model.state_names):
    print(f"{name:<10} {t_matrix[i][0]:<10.4f} {t_matrix[i][1]:<10.4f} {t_matrix[i][2]:<10.4f}")

# --- 3. Compute Stationary Distribution ---
steady_state = model.stationary_distribution()
print("\n--- Long-Term Stationary Distribution ---")
for name, prob in zip(model.state_names, steady_state):
    print(f"{name}: {prob * 100:.2f}%")

# --- 4. Simulate ---
future_path = model.simulate(start_state_idx=2, steps=15) # Starting from 'Bull'
print("\n--- 15-Day Market Simulation (Starting in Bull Regime) ---")
print(" -> ".join(future_path))