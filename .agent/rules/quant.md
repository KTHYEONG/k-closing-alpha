---
trigger: glob
---

trigger:

- "src/daily/\*_/_.py"
- "src/processing/\*_/_.py"
- "src/data/\*_/_.py"
- "src/sync/\*_/_.py"
- "src/api/\*_/_.py"
- on*file_path_regex: "src/.*(AI|preprocessor|loader|sync|client|data|scale).\_"
- on_label: ["quant"]

---

# Quant & Financial Engineering Directives (Equity Focused)

Priority: 1.Correctness > 2.No Look-Ahead & Bias Control > 3.Numerical Stability > 4.Reproducibility > 5.Market Realism (Taxes/Friction) > 6.Efficiency

## 1. Hard Stop (Fail-Fast)

If critical parameters for 'Correctness', 'Time-Series Safety', or 'Equity-Specific Biases' are missing: DO NOT generate code. Output "Task Classification" and explicitly list missing params under "Needs confirmation". Ask for clarification.

## 2. Core Constraints

- **Data Integrity & Schema:** Validate shape, dtype, and strict chronological order. Explicitly perform assertion of column types and dimension sizes before data input.
- **Corporate Actions:** Explicitly handle Adjusted vs. Unadjusted price series (Stock splits, dividends, mergers). Never mix them silently.
- **Time-Series (No Leakage):** Strict chronological alignment. Default backtest: Signal generated at $t$ (using data up to $t$ close) executes at $t+1$ (Open or Close).
- **Math & Numerical Stability:** Block division by zero and prevent uncontrolled NaN/Inf propagation. Justify `epsilon` choices or handling mechanisms (e.g., IQR Scaling, Rank transformation) for fat-tailed asset returns.
- **Trading Realism:** Parameterize equity-specific friction: broker commissions, Stock Transaction Tax (STT), exchange fees, slippage, and borrow costs for short-selling.
- **Reproducibility:** Fix random seeds for all data splits, random sampling, and ML model training. No hidden global states.

## 3. Performance & Code Quality

- **Zero-Loop Policy:** Strictly prohibit pure Python `for` or `while` loops **ONLY when performing mathematical/statistical operations on price time-series or multi-asset matrices.** (Allowed for API payload parsing, I/O operations, and network synchronization).
- **Vectorization First:** Utilize `NumPy`, `Pandas`, or `Polars` vectorization and broadcasting for all primary calculations.
- **Numba (JIT Compilation):**
    - Apply `@njit(nopython=True, cache=True)` ONLY for recursive, path-dependent logic, or proven bottlenecks.
    - NEVER pass DataFrames/Series to Numba; explicitly extract arrays via `.to_numpy()`.
    - `fastmath=True` requires explicit justification in comments.
- **Memory & Latency Management:** Default to pre-allocation via `np.zeros()`. Use `Polars` Lazy Evaluation, `Pandas` `chunksize`, or Generators for large-scale tick/order book data.
- **Streaming Feeds:** Use fixed-size `Ring Buffers` (`collections.deque` or fixed numpy arrays) instead of variable-length lists when processing real-time exchange feeds to minimize allocation latency.
- **Quality:** Enforce explicit type hints/signatures and clear validation steps. Separate mathematical indicators from execution logic.

## 4. Anti-Patterns (Do NOT use unless explicitly justified)

- **Time-Series CV:** Random K-Fold cross-validation (Use Walk-forward or Purged/Embargoed CV instead).
- **Machine Learning Overkill:** Purged/Embargoed CV or Combinatorial Purged CV (CPCV) unless overlapping labels are explicitly present.
- **Advanced Labeling:** Triple-Barrier Method unless building path-dependent intraday execution/stop-loss models.
- **Dimensionality Reduction:** PCA or cross-sectional Z-score normalization unless managing large feature sets or broad multi-asset universes.
- **Mathematical Derivations:** Writing extensive mathematical proofs or architecture-level documentation for simple, isolated indicator tasks.

## 5. Context-Specific Checks (Apply only if relevant)

- **Backtest:** Is the signal properly shifted to prevent look-ahead? Are transaction taxes applied on the sell side? Are long/short constraints and bounded exposures enforced?
- **ML Modeling:** Ensure train/test splits maintain strict temporal order. Feature scalers must fit ONLY on the training data to prevent data leakage.
- **Equity (Stock Market Dynamics):**
    - **Market Hours:** Account for specific market hours, exchange holidays, and overnight gap risks (Close-to-Open jumps).
    - **Survivorship Bias:** Historical universe construction must include delisted/merged tickers to avoid upward performance bias.
    - **Liquidity & Impact:** Check for volume constraints; simulate market impact using ADV (Average Daily Volume) metrics for large sizing.
    - **Short Constraints:** Validate borrow availability and locate fees before executing short signals.

## 6. Output Modes & Templates

Determine task scope and output STRICTLY using the matching template.

[Mode: Micro] (Indicators, Utils, Snippets)
Task: [Type]
Assumptions: [Minimal Math/Statistical assumptions]
Code: [High-performance vectorized implementation with complexity comments]
Checks: [Edge cases, NaN/Inf handling]

[Mode: Standard] (Backtests, Signals, Features)
Task: [Type]
Data Shape & Alignment: [Define explicit input/output shapes (e.g., Dataframe of Date x Tickers) and boundary alignments before coding]
Code: [Clean, production-grade implementation]
Verification: [Main stability checks, execution shift validation]

[Mode: Full] (ML, Portfolio, Execution)
Task: [Type, Asset Universe, Objective]
Pipeline Logic Plan: [Describe how corporate actions, missing data, and time-order constraints are preserved in the pipeline steps]
Method Choice: [Method, Stylized Facts awareness & Justification]
Code: [Production-grade scalable logic with Numba/Vectorization optimization]
Verification: [Leakage checklist, Survivorship bias control, Transaction tax/friction realism, Performance benchmarks]
