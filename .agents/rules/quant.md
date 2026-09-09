---
trigger:
  - on_label: ["quant"]
  - on_file_path_regex: "src/.*(daily|backfill|processing|data|sync|api).*"
  - on_file_path_glob: ["src/**/daily/**/*.py", "src/**/backfill/**/*.py", "src/**/processing/**/*.py"]
priority: 10
---

# Quant & Financial Engineering Principles

> **Never leak future information, preserve the reality of capital flows and execution viability, guard against validation leakage and overfitting, and prioritize economic correctness over specific implementation mechanics.**

## 1. Temporal Integrity & Information Availability (PIT & Leakage)
- **Information Availability:** Use strictly data that was realistically known and released at the decision timestamp. Never apply `.shift(1)` blindly without causal verification.
- **Point-in-Time (PIT) & Survivorship:** Ensure universes, index constituents, and financial/filing disclosures contain no look-ahead restatements or survivorship bias.
- **ML & Validation Leakage:** Prevent feature/label overlap across splits. Apply purging/embargoing for overlapping windows, and fit all learned preprocessing (scalers, encoders, feature selection) strictly on train folds.

## 2. Execution Realism & Portfolio Accounting
- **Execution Viability & Friction:** Differentiate signal prices from realistically executable fill prices (spread, tick size, auction dynamics, slippage). Account for transaction taxes, fees, and funding/borrow costs.
- **Portfolio Accounting Consistency:** Accurately reconcile cash, positions, fees, P&L, settlement cycles (e.g., T+2), and external cash flows to avoid misrepresenting returns (TWR, MWR, CAGR).
- **Research-to-Production Parity:** Maintain consistent universe definitions, sizing logic, timing semantics, and feature engineering across research/backtesting and live execution.

## 3. Numerical Integrity & Economic Correctness
- **Numerical Edge Cases:** Handle division by zero, NaNs, and infinities according to their genuine economic/market meaning (e.g., halted trading, zero volume, unfillable) rather than silently coercing them into arbitrary normal values.
- **Metric Significance vs. Overfitting:** Avoid blindly tuning parameters against isolated metrics (Sharpe, Accuracy); guard against selection bias and evaluate economic viability aligned with the investment objective.
- **Principles Over Mechanics:** Prioritize sound financial and statistical meaning over rigid dogma around specific functions or recipes.
