# Quantitative Market Regime Switching & Risk Mitigation

An institutional-grade quantitative framework designed to model, detect, and adapt to shifting S&P 500 market regimes using First-Order Markov Chains and Gaussian Hidden Markov Models (HMM). 

This research tracks the project's evolution from synthetic Markov models to a multi-feature, walk-forward out-of-sample (OOS) backtesting engine net of transaction fees.

---

## Executive Summary & Core Results

The primary goal of this framework is **downside risk mitigation without sacrificing risk-adjusted returns**. Through walk-forward out-of-sample validation (2019–2026), the strategy successfully mitigated market tail-risk while improving overall Sharpe efficiency compared to a passive Buy & Hold benchmark.

### Out-of-Sample (OOS) Performance Summary
* **Testing Window:** 2019 – 2026 (Walk-Forward Rolling Window)
* **Transaction Friction:** 5 bps (0.0005) per unit of portfolio rebalanced
* **Initial Capital:** $1,000.00

| Performance Metric | Buy & Hold (S&P 500) | Walk-Forward 3-State HMM (OOS) | Impact / Delta |
| :--- | :---: | :---: | :---: |
| **Final Portfolio Value** | **$3,055.40** | **$2,828.06** | -$227.34 (-7.4%) |
| **Maximum Drawdown** | **-33.92%** | **-25.08%** | **+8.84% Reduction** |
| **Annualized Sharpe Ratio** | **0.84** | **0.94** | **+0.10 Improvement** |

> **Key Takeaway:** By dynamically paring equity exposure during high-volatility/bear regimes, the model shaved **~8.84 percentage points off maximum drawdown** (spanning the COVID-19 crash and 2022 rate hike cycles) while producing superior risk-adjusted returns (**0.94 Sharpe** vs **0.84 Buy & Hold**).

---

## Architectural Evolution

### Phase 1: Synthetic First-Order Markov Chain (`RegimeSwitching.py`)
* Built foundational transition matrix estimators ($\mathbf{P}_{ij} = P(S_{t+1} = j \mid S_t = i)$) using discrete return bins (Bear: $r < -0.75\%$, Sideways: $|r| \le 0.75\%$, Bull: $r > 0.75\%$).
* Solved for long-term steady-state probabilities using left-eigenvector decomposition ($\boldsymbol{\pi} \mathbf{P} = \boldsymbol{\pi}$).

### Phase 2: Empirical Calibration & Backtesting (`backtesting.py`, `backtesting2.py`)
* Applied discrete thresholding to 10 years of historical S&P 500 data (`^GSPC`).
* Evaluated return threshold sensitivity across $\pm 0.50\%$, $\pm 0.75\%$, and $\pm 1.25\%$.
* **Key Finding:** Hard cash allocations during Bear states created severe "cash drag" by missing sharp market rebound days.

### Phase 3: Risk-Optimized Allocation Rules (`backtesting3.py`)
* Transitioned from binary cash rules to a balanced weight structure:
  * **Bull Regime:** 100% Equity Exposure
  * **Sideways Regime:** 80% Equity Exposure
  * **Bear Regime:** 50% Equity Exposure (retaining partial exposure to capture rapid mean-reversions).

### Phase 4: Hidden Markov Models with Multi-Features (`hmm.py`, `hmm2.py`)
* Upgraded from discrete return bins to continuous multivariate Gaussian Hidden Markov Models via `hmmlearn`.
* Incorporated **21-day rolling volatility** alongside daily percentage returns to capture volatility clustering.
* Extended model to 3 latent components (Bull, Sideways, Bear) and incorporated **5 bps transaction costs** per rebalancing action.

### Phase 5: Walk-Forward Out-of-Sample Backtesting (`oos.py`)
* Formulated an expanding walk-forward engine (3-year initial warmup) predicting test years entirely out-of-sample.
* Resolved covariance matrix singularities during extreme volatility shocks (e.g., March 2020) by shifting to diagonal covariance structures (`covariance_type="diag"`).

---

## Code Base Map

```
├── RegimeSwitching.py   # Basic discrete Markov chain solver on synthetic data
├── RegimeSwitching2.py  # Discrete Markov chain model fitted to 10yr S&P 500 data
├── backtesting.py       # Initial backtest with binary cash-switching rules
├── backtesting2.py      # Threshold sensitivity analysis (0.50%, 0.75%, 1.25%)
├── backtesting3.py      # Balanced regime-switching strategy with partial equity buffer
├── hmm.py               # 2-State Gaussian HMM using Returns + Rolling Volatility
├── hmm2.py              # 3-State Gaussian HMM + 5 bps rebalancing transaction fees
└── oos.py               # Production Walk-Forward Out-of-Sample (OOS) engine
```

---

## Setup & Running the Framework

### Dependencies
Install the required quantitative packages:
```bash
pip install numpy pandas yfinance matplotlib hmmlearn
```

### Executing the Production Out-of-Sample Model
To run the full walk-forward out-of-sample pipeline:
```bash
python oos.py
```

---

## Mathematical Formulation

1. **State Transition Matrix:**
   $$P_{ij} = \frac{N_{ij}}{\sum_{k} N_{ik}}$$

2. **Stationary Distribution:**
   $$\boldsymbol{\pi} = \lim_{k \to \infty} \mathbf{P}^k \quad \text{where} \quad \boldsymbol{\pi} \mathbf{P} = \boldsymbol{\pi}, \quad \sum_i \pi_i = 1$$

3. **Gaussian HMM Emission Likelihood:**
   $$X_t \mid (S_t = k) \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$
   Where $X_t = \begin{bmatrix} r_t & \sigma_{21, t} \end{bmatrix}^T$ represents the joint return and rolling volatility vector.