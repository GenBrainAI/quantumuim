# Quantum Finance

## 1. Overview

Financial services is one of the most promising near-term application domains for quantum computing. Key problems — option pricing, risk analysis, portfolio optimization — map naturally to quantum algorithms and offer clear economic value if quantum speedup is achieved.

## 2. Option Pricing via Quantum Amplitude Estimation

### Classical Approach: Monte Carlo
- Simulate asset price paths under risk-neutral measure
- Average discounted payoff over paths
- Convergence: O(1/√N) — need 10⁶ paths for 3-digit precision, 10¹⁰ for 5-digit

### Quantum Approach: Amplitude Estimation
Replace Monte Carlo sampling with quantum amplitude estimation for quadratic speedup.

**Steps:**
1. **Distribution loading:** Encode probability distribution of asset prices into quantum state
   - Log-normal for Black-Scholes, or empirical distributions
   - Requires O(n) qubits for n-bit price discretization
2. **Payoff computation:** Apply comparator + controlled rotation to encode payoff in qubit amplitude
   - European call: max(S - K, 0) where S is spot price, K is strike
3. **Amplitude estimation:** QPE on Grover operator to extract expected payoff
   - M evaluation queries → O(1/M) error (vs O(1/√M) classical)

### Error Analysis
| Method | Queries for ε error |
|--------|-------------------|
| Classical MC | O(1/ε²) |
| Quantum AE (canonical) | O(1/ε) |
| Iterative QAE | O(1/ε) (no QPE needed) |

**Quadratic speedup** — a problem requiring 10⁶ classical samples needs ~10³ quantum queries.

### Practical Considerations
- Distribution loading circuit depth is the bottleneck
- Grover oracle for payoff must be efficiently implementable
- Current hardware: proof-of-concept with 4-8 qubits, <1% error vs Black-Scholes analytical

## 3. Risk Analysis

### Value at Risk (VaR)
The maximum loss at a given confidence level (e.g., 99% VaR = loss exceeded only 1% of the time).

**Quantum approach:**
1. Load portfolio loss distribution into quantum state
2. Use quantum bisection search to find the α-quantile
3. Speedup: O(1/ε) vs O(1/ε²) classical

### Conditional Value at Risk (CVaR)
Expected loss given that loss exceeds VaR — captures tail risk.

**Quantum approach:**
1. Encode loss distribution
2. Apply indicator function for losses > VaR threshold
3. Amplitude estimation on tail expectation
4. More informative than VaR for fat-tailed distributions

### Credit Risk
- Portfolio of N correlated assets, each with default probability pᵢ
- Correlations via Gaussian copula or factor model
- Quantum loading of joint default distribution
- Amplitude estimation for expected loss and loss percentiles

## 4. Portfolio Optimization

### Classical Formulation (Markowitz Mean-Variance)
Minimize: w'Σw - μ'w (risk - λ × return)
Subject to: Σwᵢ = 1, wᵢ ∈ {0,1} (or continuous)

### QUBO Mapping
- Binary variables xᵢ ∈ {0,1}: include asset i or not
- Objective: min Σᵢⱼ Σᵢⱼ xᵢxⱼ - λ Σᵢ μᵢxᵢ + penalty × (Σxᵢ - K)²
- K = target number of assets, penalty enforces cardinality constraint

### Quantum Approaches
1. **QAOA:** Encode portfolio QUBO as cost Hamiltonian, run p-layer QAOA
2. **VQE:** Variational optimization with problem-specific ansatz
3. **Quantum annealing:** Direct mapping to Ising model on D-Wave
4. **Grover adaptive search:** Exact solution with quadratic speedup (fault-tolerant)

### Results
- Classical solvers (Gurobi, CPLEX) solve portfolios with ~1000 assets in seconds
- Quantum advantage requires either very large portfolios or additional constraints that make classical solvers struggle
- Near-term value: hybrid quantum-classical warm-starting

## 5. Quantum Monte Carlo for Derivatives

Beyond vanilla options:
- **Path-dependent options:** Asian, barrier, lookback — require multi-step price simulation
- **Multi-asset derivatives:** Basket options, spread options — high-dimensional distributions
- **Interest rate derivatives:** Swaptions, caps/floors — term structure modeling

Each benefits from the quadratic speedup of quantum amplitude estimation, with the speedup advantage growing with problem dimension.

## 6. Machine Learning in Finance

Quantum ML applied to financial problems:
- **Fraud detection:** Quantum kernel methods on transaction features
- **Credit scoring:** VQC for binary classification of default risk
- **Time series prediction:** Quantum reservoir computing for market dynamics
- **Anomaly detection:** Quantum autoencoders for unusual trading patterns

## 7. Timeline to Practical Quantum Advantage

| Application | Qubits (logical) | Estimated Timeline | Speedup |
|------------|------------------|-------------------|---------|
| Option pricing (simple) | 50-100 | 2027-2030 | Quadratic |
| Portfolio optimization | 100-1000 | 2030-2035 | Problem-dependent |
| Risk analysis (full) | 100-500 | 2028-2032 | Quadratic |
| Credit risk (large portfolio) | 500-2000 | 2032-2040 | Quadratic |

## 8. Relation to Quantumium SDK

Our SDK provides:
- **European option pricing:** Amplitude estimation with Black-Scholes benchmark
- **Risk analysis:** VaR and CVaR via quantum distribution loading
- **Portfolio optimization:** QAOA on QUBO formulation with mean-variance objective
- See `quantumium.finance` package — 41 tests, runnable demo example
