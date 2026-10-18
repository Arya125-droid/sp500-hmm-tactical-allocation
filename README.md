# Comprehensive Quantitative Research Report: Market Regime-Switching Strategies

## Executive Summary

This report documents the end-to-end development, iteration, and out-of-sample validation of quantitative market regime-switching strategies applied to the S&P 500 (`^GSPC`) over a 10-year historical dataset. The research transitions from rigid, discrete threshold-based Markov chains to sophisticated, multi-dimensional **Gaussian Hidden Markov Models (HMM)**, culminating in rigorous **Out-of-Sample (OOS) Walk-Forward Testing** with realistic transaction friction.

---

## Codebase Architecture & File Overview

The research project is structured across several sequential scripts:

* **`RegimeSwitching.py` & `RegimeSwitching2.py`**: Foundation scripts implementing discrete Markov chains using synthetic and real S&P 500 daily returns, calculating transition matrices, stationary distributions, and forward regime simulations.


* **`backtesting.py`**: The initial backtesting engine utilizing a discrete threshold model ($\pm 0.75\%$) with binary equity/cash allocations.


* **`backtesting2.py`**: Explores threshold sensitivity by testing multiple discretization boundaries ($\pm 0.50\%$, $\pm 0.75\%$, and $\pm 1.25\%$).


* **`backtesting3.py`**: Introduces "balanced" asset weights (raising Bear allocations to 50% and Sideways to 80%) to mitigate cash drag, incorporating annualized Sharpe ratios.


* **`hmm.py`**: Upgrades the system to a continuous **2-State Gaussian Hidden Markov Model**, incorporating multi-dimensional features (daily returns + 21-day rolling volatility).


* **`hmm2.py`**: Expands to a **3-State Gaussian HMM** and introduces realistic transaction friction (5 basis points per unit of portfolio rebalance) and optimization tolerance tuning (`tol=1e-4`).


* **`oos.py`**: Implements a rigorous **Out-of-Sample Walk-Forward Testing framework** with expanding training windows and diagonal covariance matrices (`covariance_type="diag"`) to eliminate look-ahead bias and handle extreme volatility shocks.



---

## Iterative Findings & Strategy Evolution

### 1. Discrete Markov Chains & The Cash Drag Trap (`backtesting.py`, `backtesting2.py`)

* **Methodology:** Returns were classified into Bear ($< -0.75\%$), Sideways, and Bull ($> 0.75\%$) regimes.


* **Finding:** Assigning a 0% equity weight (100% cash) during Bear regimes successfully insulated the portfolio during deep drawdowns, but resulted in severe **cash drag** during secular bull runs, causing total strategy wealth to lag significantly behind the Buy & Hold benchmark. Threshold sensitivity tests confirmed that static percentage boundaries react too slowly to changing market dynamics.

### 2. Balanced Weights & In-Sample Gaussian HMM (`backtesting3.py`, `hmm.py`)

* **Methodology:** Strategy rules were modified to retain a 50% equity buffer during Bear regimes to capture rapid market recoveries. Furthermore, replacing manual thresholds with a **2-State Gaussian HMM** utilizing both returns and 21-day rolling volatility allowed the model to identify hidden market states probabilistically.


* **Result:** In-sample testing on the 2-state HMM achieved a final wealth of **$3,835.40**, crushed the Buy & Hold maximum drawdown down from $-33.92\%$ to **$-20.03\%$**, and delivered an exceptional Sharpe ratio of **1.13** (vs. 0.80 for Buy & Hold).



### 3. Out-of-Sample Walk-Forward Validation (`oos.py`)

To prove the models were not overfitted to historical data, an expanding-window walk-forward validation framework was deployed, retraining the model year-by-year from 2019 to 2026 without future foresight.

* **3-State Walk-Forward HMM:** Achieved an OOS Sharpe ratio of $0.94$ and a maximum drawdown of $-25.08\%$.
* **2-State Walk-Forward HMM:** Stripped out intermediate buffer drag, closely matching the benchmark's final compounding wealth while retaining elite risk control.

---

## Final Empirical Comparison Summary

Below is the performance summary of the ultimate **Out-of-Sample Walk-Forward 2-State Gaussian HMM** (tested across the unseen 2019–2026 period with 5 bps transaction friction) compared against the benchmark:

| Performance Metric (OOS Period) | Buy & Hold Benchmark | 2-State Walk-Forward HMM (OOS) |
| --- | --- | --- |
| **Final Portfolio Value** | **$3,055.40** | **$3,021.88** |
| **Maximum Drawdown** | **-33.92%** | **-20.05%** |
| **Sharpe Ratio** | **0.84** | **1.06** |

---

## Other Useful Outputs:
| Performance Metric (OOS Period) | Buy & Hold Benchmark | 3-State Walk-Forward HMM (OOS) |
| --- | --- | --- |
| **Final Portfolio Value** | **$3,055.40** | **$2,828.06** |
| **Maximum Drawdown** | **-33.92%** | **-25.08%** |
| **Sharpe Ratio** | **0.84** | **0.94** |


| Performance Metric (Gaussian HMM) | Buy & Hold Benchmark | 3-State Gaussian Hidden Markov Model + Fees (Net) |
| --- | --- | --- |
| **Final Portfolio Value** | **$3,539.40** | **$3,344.03** |
| **Maximum Drawdown** | **-33.92%** | **-19.65%** |
| **Sharpe Ratio** | **0.80** | **1.08** |


| Performance Metric (Gaussian HMM) | Buy & Hold Benchmark | 2-State Gaussian Hidden Markov Model |
| --- | --- | --- |
| **Final Portfolio Value** | **$3,539.40** | **$3,835.40** |
| **Maximum Drawdown** | **-33.92%** | **-20.03%** |
| **Sharpe Ratio** | **0.80** | **1.13** |
---

## Conclusion & Verdict: Which Model is the Best?

Among all models and iterations tested across this project, the **Out-of-Sample Walk-Forward 2-State Gaussian Hidden Markov Model (`hmm.py` / `oos.py`)** is unequivocally the **best-performing and most robust model**.

### Why It Wins:

1. **Uncompromised Compounding:** Unlike the 3-state variants or threshold models which introduced structural lag or cash drag, the 2-state OOS model captured nearly 99% of the Buy & Hold final wealth ($3,021.88 vs $3,055.40).


2. **Elite Drawdown Control:** It successfully slashed the maximum drawdown from a catastrophic $-33.92\%$ down to **$-20.05\%$**, proving its capacity to defend capital during severe macro shocks (such as the 2020 COVID crash and 2022 bear market).


3. **True Out-of-Sample Validity:** Because these results were generated via strict walk-forward testing (zero look-ahead bias) net of real-world transaction friction (5 bps), it represents an institutional-grade trading system ready for live deployment.