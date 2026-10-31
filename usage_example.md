## **Usage Example**

This guide demonstrates the study interface `run_task_34` of the Python reproduction of *$L_0$ Feature Selection via Monte Carlo Tree Search in High-Dimensional Financial Regression* (Cameron Mouangue and Tristan Cazenave). The method is a Generalized Nested Rollout Policy Adaptation (GNRPA) search over feature subsets, with a lexicographic reward on validation $R^2$, sparsity and multicollinearity, followed by a soft-voting ensemble of unshrunk linear models. Python implementation author: **CS Chirinda**.

The example has four steps. Step 1 synthesises the four raw panels the pipeline ingests: `YAHOO_FINANCE_RAW`, `FRED_MACRO_RAW`, `FACTOR_LIBRARY_RAW` and `TRADING_CALENDAR`. Step 2 reads the existing `config.yaml` into the plain dictionary `config`. Step 3 composes and validates every argument of `run_task_34`. Step 4 calls the interface and reads back what it returns. Every executed code fence below was run, in the order shown, against `feature_selection_via_monte_carlo_tree_search_in_high_dimensional_financial_regression_draft.ipynb`. The section **Verification record** states what ran and what did not.

**Assumptions and execution requirements.**

1. Every research callable and helper already lives in the **single Jupyter notebook** `feature_selection_via_monte_carlo_tree_search_in_high_dimensional_financial_regression_draft.ipynb`. Execute its import cell under the heading *Import Essential Modules*, then the code cells from *Task 1* through *Task 36*. Do **not** execute the notebook's *Study Inputs* and *Study Execution* cells: they only record a stale `blocked` status, and the pipeline does not use them. Then append the example cells below. **No Python module folder (`.py` package) is imported or required**; `run_task_34` and its collaborators are resolved from the notebook's global namespace.
2. The working directory is the notebook's folder, and `config.yaml` already exists there. The example reads it and never modifies it. Its JSON twin, `study_config.json`, carries the same parameters; the notebook reads neither file by itself, so the dictionary is always passed in.
3. The verification interpreter was Python 3.13.9 with NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.7.2, Optuna 5.0.0, matplotlib 3.10.6, PyYAML 6.0.3 and pandas_market_calendars 5.5.0. The example itself imports NumPy, pandas, pandas_market_calendars and PyYAML.
4. **Faker is not used.** It is not installed in the verification environment, and it generates names and addresses rather than market data. The panels come from a private, seeded NumPy `PCG64` data-generating process (seed `20261031`), which is the functionally similar alternative the brief permits. Every run is bit-for-bit reproducible.
5. **Every value is synthetic.** Levels, rates, factors and volumes are artificial, and only the NYSE session dates are real. A completed run demonstrates the input contract and the wiring of the pipeline. It does not reproduce the paper's empirical results, and no figure in this document should be read as one.
6. On Windows, keep the output directory short (the example uses `~/l0_runs`). Stage artifacts are nested inside it, and Windows refuses paths longer than 260 characters unless long paths are enabled.
7. All date columns are `datetime64[ns]` and timezone-naive.

### **The study interface: `run_task_34`**

The actual signature, read from the notebook, is:

```text
run_task_34(
    raw_inputs: Mapping[str, Any],
    config: Mapping[str, Any],
    configuration_name: str,
    mode: str = "full",
    *,
    seed: Optional[int] = None,
    use_walk_forward_fold: bool = False,
    audit_log_path: Any = None,
    output_dir: Any = None,
    run_index: int = 0,
    enable_tuning: bool = False,
    ensemble_runs: Optional[int] = None,
    baseline_cv: bool = False,
    run_baselines: bool = True,
    strict_reporting: bool = False,
    allow_missing_calendar_oracle: bool = False,
    policy_norm_bound: Optional[float] = None,
    prepared: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]
```

Three arguments are required. Everything after `*` is keyword-only.

| Argument | Default | Contract |
| --- | --- | --- |
| `raw_inputs` | required | The four raw panels, keyed `YAHOO_FINANCE_RAW`, `FRED_MACRO_RAW`, `FACTOR_LIBRARY_RAW` and `TRADING_CALENDAR`. The block names of `config["raw_input_schemas"]` are accepted as alternative keys. |
| `config` | required | The study configuration. It is never mutated. |
| `configuration_name` | required | `"fixed"`, `"tuned"` or `"endogenous"`: the three columns of Table 2. |
| `mode` | `"full"` | `"full"` runs Tasks 1 to 33. `"smoke"` sets $I_1 = I_2 = I_3 = 1$, runs the preparation and the search on pre-2024 data, checks completion, policy convergence, the information barrier and the lexicographic ordering of two rewards, and never evaluates the test block. |
| `seed` | `None` | Run seed. `None` uses the register's global seed, `compute_and_reproducibility.global_random_seeds.numpy_seed`, which is 42. |
| `use_walk_forward_fold` | `False` | Fold-scoring flag of the search stage. |
| `audit_log_path` | `None` | JSON Lines audit file. `None` uses `<output_dir>/task_34_audit.jsonl`. |
| `output_dir` | `None` | Directory for the stage artifacts. It is required unless `audit_log_path` is given. |
| `run_index` | `0` | Run index recorded in the diagnostics. |
| `enable_tuning` | `False` | Runs the Optuna stages, Tasks 28 and 29, in full mode, and applies the best vector. With `False`, the frozen vectors of Table 2 are used. |
| `ensemble_runs` | `None` | Overrides the run count of Table 3. `None` uses `table3_ensemble_diagnostics.runs_executed`: 18, 18 and 12. |
| `baseline_cv` | `False` | Re-runs the baselines' cross-validated selection on the pre-2024 history instead of using the register's selected values. |
| `run_baselines` | `True` | Runs Tasks 25 to 27 in full mode. |
| `strict_reporting` | `False` | Raises when Task 32 or Task 33 fails. By default the failure is recorded in the stage log and in `reporting_failures`. |
| `allow_missing_calendar_oracle` | `False` | Skips Task 3's comparison with the exchange calendar when the calendar package is missing. |
| `policy_norm_bound` | `None` | Frobenius bound of the smoke convergence check. `None` lets the notebook derive it. |
| `prepared` | `None` | A Tasks 1 to 8 bundle for the same configuration, which skips those tasks. It can come from `prepare_design_matrix_task_34` or from the `prepared` key of an earlier full-mode result. |

**Tasks outside this interface.** Task 34 is the orchestrator itself. Task 35 (robustness analyses) and Task 36 (final verification against the paper's targets) have their own entry points, `run_task_35` and `run_task_36`, and run after it. They are not called in this example.

**Return value.** A dictionary. Both modes return `mode`, `search_artifact`, `coefficients`, `validation_r2_trajectory`, `run_diagnostics`, `oos_r2`, `oos_dispersion`, `information_barrier` and `stage_log`; smoke mode adds `smoke`, and `oos_r2` and `oos_dispersion` are `None` there because the test block is never touched. Full mode adds `single_run_oos_r2`, `selected_features`, `subset_sizes`, `configuration_record`, `baselines`, `greedy`, `sparsity_stress`, `sparsity_stress_vs_measured_mean`, `tuning`, `ensemble`, `oos`, `inference`, `interpretability`, `reporting_failures` and `prepared`. In full mode with an ensemble, `oos_r2` is the ensemble's headline figure and `single_run_oos_r2` is the seed run's.

**Exceptions.** `ValueError` for an invalid mode, configuration or input. `RuntimeError` when the budget preflight, the information barrier or a smoke check fails.

### **Step 1: Synthetic Data Generation (the four raw panels)**

The panels follow `config["raw_input_schemas"]`. The data-generating process is built to pass the validators a vendor extract must pass, so the structural properties that the stages rely on are reproduced deliberately:

* **One latent market.** All panels draw on one simulated market, so the cross-panel co-movement survives: the leverage effect links equity returns to same-day volatility and credit moves, and the factors keep their cross-correlations.
* **Real session dates.** The session axis comes from the NYSE calendar, because Task 3 compares `TRADING_CALENDAR` with that same calendar. This gives 6,600 sessions; see **Where the brief and the notebook differ** for the 6,260 stated in the schema.
* **Burn-in history.** The market and factor panels start on 1999-11-01 and the credit spread on 1996-12-31, ahead of the first calendar session, so the rolling features are defined at the start of the study window.
* **Vendor artefacts.** `YAHOO_FINANCE_RAW` has 33 rows on exchange holidays, with a missing `ndx_close` and zero volume, as the real extract has. Task 2 removes them.
* **Monthly stamping.** CPI, unemployment and payrolls are stamped on the first of the month, as FRED dates them, so some rows fall on weekends and the daily series are missing there. Release lags are not applied in the panel; Task 7 applies them at the join.
* **Factor units.** Factors are in percent per session, the unit of the library's daily files and of the study's input panel.

| Panel | Shape | Date key | Columns |
| --- | --- | --- | --- |
| `YAHOO_FINANCE_RAW` | 6,676 × 7 | `trading_date` | `ndx_close` (float64), `ndx_volume` (int64), `vix_close` (float64), `usd_index_close` (float64), `source_fetched_at_utc` (datetime64[ns]), `raw_adjustment_flag` (str) |
| `FRED_MACRO_RAW` | 7,723 × 8 | `observation_date` | `rate_short` (DGS2), `rate_long` (DGS10), `spread_2s10s` (T10Y2Y), `spread_hy` (BAMLH0A0HYM2), `inflation_cpi` (CPIAUCSL), `labor_unrate` (UNRATE), `labor_payems` (PAYEMS), all float64 |
| `FACTOR_LIBRARY_RAW` | 6,643 × 8 | `date` | `MKT`, `SMB`, `HML`, `RMW`, `CMA`, `MOM` (float64), `vendor_vintage_id` (str) |
| `TRADING_CALENDAR` | 6,600 × 6 | `trading_date` | `session_open_flag` (bool), `half_day_flag` (bool), `exchange` (category), plus the provenance columns `calendar_source` and `calendar_derivation_rule` |

#### **Step 1.1: Imports and data-generating constants**

```python
# =============================================================================
# Step 1: synthetic raw panels for run_task_34
# =============================================================================
# Import postponed evaluation of annotations, as the notebook does.
from __future__ import annotations

# Import time to measure each run.
import time

# Import Path for the configuration file and the output directories.
from pathlib import Path

# Import the typing names used in the signatures below.
from typing import Any, Dict, Mapping, Tuple

# Import NumPy for the seeded PCG64 data-generating process.
import numpy as np

# Import pandas for the four raw DataFrames.
import pandas as pd

# Import the exchange calendar package that Task 3 also uses as its oracle.
import pandas_market_calendars as mcal

# Import PyYAML to read config.yaml.
import yaml

# Fix the seed of the data-generating process.
DGP_SEED: int = 20261031
# Fix the first vendor date of the market and factor panels.
FIRST_VENDOR_DATE: str = "1999-11-01"
# Fix the first session of the trading calendar.
FIRST_CALENDAR_DATE: str = "2000-01-03"
# Fix the first date of the high-yield spread history.
FIRST_CREDIT_DATE: str = "1996-12-31"
# Fix the last date of every panel (Section 3.1 of the paper).
LAST_DATE: str = "2026-03-31"
# Fix the synthetic retrieval timestamp recorded for provenance.
FETCHED_AT_UTC: pd.Timestamp = pd.Timestamp("2026-10-31 06:00:00")
# Fix the synthetic factor-library vintage identifier.
VENDOR_VINTAGE_ID: str = (
    "F-F_Research_Data_5_Factors_2x3_daily_vintage_20261031"
)
# Fix the calendar construction rule named in raw_input_schemas.
CALENDAR_SOURCE: str = "union_of_market_sessions"
# Fix the human-readable statement of that rule.
CALENDAR_DERIVATION_RULE: str = (
    "NYSE sessions over 2000-01-03 to 2026-03-31, exclusive of full "
    "holidays, inclusive of half-days, indexed by session close date"
)
# Fix the number of vendor dates that are not NYSE sessions.
N_VENDOR_ONLY_DATES: int = 33

# Fix the scripted index turning points as (date, level) pairs.
NDX_ANCHORS: Tuple[Tuple[str, float], ...] = (
    ("1999-11-01", 2616.0),
    ("2000-03-27", 4704.0),
    ("2002-10-07", 804.0),
    ("2007-10-31", 2238.0),
    ("2008-11-20", 1036.0),
    ("2020-02-19", 9718.0),
    ("2020-03-20", 6994.0),
    ("2021-11-19", 16573.0),
    ("2022-12-28", 10679.0),
    ("2026-03-31", 23740.0),
)

# Fix the daily factor volatilities, in percent per session.
FACTOR_VOLATILITY: Dict[str, float] = {
    "MKT": 1.234,
    "SMB": 0.646,
    "HML": 0.779,
    "RMW": 0.553,
    "CMA": 0.451,
    "MOM": 1.066,
}

# Fix the factor correlation matrix (order: MKT, SMB, HML, RMW, CMA, MOM).
FACTOR_CORRELATION: np.ndarray = np.array(
    [
        [1.00, 0.15, 0.01, -0.36, -0.26, -0.24],
        [0.15, 1.00, 0.17, -0.29, 0.05, -0.09],
        [0.01, 0.17, 1.00, 0.18, 0.49, -0.32],
        [-0.36, -0.29, 0.18, 1.00, 0.28, 0.08],
        [-0.26, 0.05, 0.49, 0.28, 1.00, 0.08],
        [-0.24, -0.09, -0.32, 0.08, 0.08, 1.00],
    ]
)
```

#### **Step 1.2: The session axes**

```python
def build_session_axes() -> Dict[str, pd.DatetimeIndex]:
    """
    Build the NYSE session axes shared by the four synthetic panels.

    Purpose:
        Fixes one date axis for every panel, so the synthetic extract
        passes the same calendar checks as a vendor extract. Sessions are
        public exchange facts, so they are taken from the calendar package
        rather than simulated.

    Inputs:
        None. The span comes from FIRST_VENDOR_DATE and LAST_DATE.

    Processes:
        1. Reads the NYSE schedule over the vendor span.
        2. Reads the early closes of that schedule.
        3. Lists the weekdays that are not sessions (exchange holidays).

    Outputs:
        Dict[str, pd.DatetimeIndex]: Keys "sessions", "half_days" and
        "holidays", each timezone-naive at datetime64[ns].

    Raises:
        ValueError: If the schedule is empty.
    """
    # Read the NYSE calendar object.
    nyse = mcal.get_calendar("NYSE")
    # Read the session schedule over the vendor span.
    schedule = nyse.schedule(FIRST_VENDOR_DATE, LAST_DATE)
    # Refuse an empty schedule.
    if schedule.empty:
        # Name the failing span.
        raise ValueError("the NYSE schedule is empty over the vendor span.")
    # Convert the schedule index to timezone-naive nanosecond dates.
    sessions = pd.DatetimeIndex(schedule.index).as_unit("ns")
    # Read the early-close sessions of the schedule.
    half_days = pd.DatetimeIndex(nyse.early_closes(schedule).index)
    # List every weekday in the span.
    weekdays = pd.bdate_range(FIRST_VENDOR_DATE, LAST_DATE)
    # Keep the weekdays on which the exchange was closed.
    holidays = weekdays.difference(sessions)
    # Return the three axes.
    return {
        "sessions": sessions,
        "half_days": half_days.as_unit("ns"),
        "holidays": holidays.as_unit("ns"),
    }
```

#### **Step 1.3: The latent market**

```python
def simulate_market_states(
    sessions: pd.DatetimeIndex, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Simulate the latent daily market states behind the synthetic panels.

    Purpose:
        Produces one coherent market so the panels share the co-movement
        the study relies on: the leverage effect (Black, 1976) links equity
        returns to same-day volatility and credit moves, and the factors
        keep their published cross-correlations.

    Inputs:
        sessions (pd.DatetimeIndex): The NYSE sessions.
        rng (np.random.Generator): The seeded PCG64 generator.

    Processes:
        1. Draws correlated Gaussian factors x_t ~ N(0, C) and a Student-t
           scale w_t = sqrt(3 / chi2_5), which gives heavy tails.
        2. Forms the equity shock z_t = (0.90 x_{MKT,t} + 0.436 u_t) w_t.
        3. Iterates the log-variance of a stochastic-volatility model:
           h_t = (1 - phi) h_bar + phi h_{t-1} + nu e_t, with
           e_t = rho z_t + sqrt(1 - rho^2) v_t and rho < 0.
        4. Sets the raw return to r_t = sigma_{t-1} z_t.
        5. Adds a constant to the log returns of each segment between
           NDX_ANCHORS, so the level path hits every anchor exactly.

    Outputs:
        pd.DataFrame: Indexed by session, with columns "ndx_close",
        "sigma", "equity_shock", "scale" and the six factor shocks.

    Raises:
        ValueError: If there are too few sessions or the anchors are not
        strictly increasing in time.
    """
    # Count the sessions.
    n_sessions = len(sessions)
    # Refuse a span too short for the rolling features.
    if n_sessions < 1000:
        # Name the short span.
        raise ValueError(f"too few sessions: {n_sessions}.")
    # Read the anchors as timestamps.
    anchor_dates = pd.DatetimeIndex([day for day, _ in NDX_ANCHORS])
    # Refuse anchors that are not strictly increasing.
    if not anchor_dates.is_monotonic_increasing:
        # Name the malformed anchors.
        raise ValueError("NDX_ANCHORS must be increasing in time.")
    # Factor the correlation matrix: C = L L^T.
    cholesky = np.linalg.cholesky(FACTOR_CORRELATION)
    # Draw the correlated Gaussian factors.
    factors = rng.standard_normal((n_sessions, 6)) @ cholesky.T
    # Draw the heavy-tailed scale of a unit-variance Student-t(5).
    scale = np.sqrt(3.0 / rng.chisquare(5.0, n_sessions))
    # Draw the equity-specific Gaussian component.
    equity_noise = rng.standard_normal(n_sessions)
    # Form z_t, the equity shock with unit variance.
    shock = (0.90 * factors[:, 0] + 0.436 * equity_noise) * scale
    # Draw the independent part of the volatility shock.
    vol_noise = rng.standard_normal(n_sessions)
    # Fix the leverage correlation between returns and volatility.
    rho = -0.70
    # Fix the persistence of the log variance.
    phi = 0.985
    # Fix the volatility of the log variance.
    nu = 0.145
    # Fix the long-run log variance (1.35% daily volatility).
    h_bar = 2.0 * np.log(0.0135)
    # Allocate the log-variance path.
    log_variance = np.empty(n_sessions)
    # Start the path at its long-run level.
    log_variance[0] = h_bar
    # Iterate the stochastic-volatility recursion.
    for t in range(1, n_sessions):
        # Combine the leverage and independent shocks into e_t.
        vol_shock = rho * shock[t] + np.sqrt(1.0 - rho**2) * vol_noise[t]
        # Update h_t = (1 - phi) h_bar + phi h_{t-1} + nu e_t.
        log_variance[t] = (
            (1.0 - phi) * h_bar + phi * log_variance[t - 1] + nu * vol_shock
        )
    # Convert the log variance to volatility, sigma_t = exp(h_t / 2).
    sigma = np.exp(0.5 * log_variance)
    # Lag the volatility, so r_t depends on sigma_{t-1} only.
    lagged_sigma = np.r_[sigma[0], sigma[:-1]]
    # Form the raw simple returns r_t = sigma_{t-1} z_t.
    raw_returns = lagged_sigma * shock
    # Convert the simple returns to log returns.
    log_returns = np.log1p(np.clip(raw_returns, -0.20, 0.25))
    # Locate each anchor on the session axis.
    positions = sessions.searchsorted(anchor_dates)
    # Keep every anchor inside the axis.
    positions = np.minimum(positions, n_sessions - 1)
    # Read the anchor levels in logs.
    anchor_logs = np.log([level for _, level in NDX_ANCHORS])
    # Start the log-level path at the first anchor.
    log_level = np.full(n_sessions, anchor_logs[0])
    # Walk consecutive anchors.
    for i in range(len(positions) - 1):
        # Read the segment's first and last positions.
        start, end = int(positions[i]), int(positions[i + 1])
        # Read the unconstrained log increments of the segment.
        steps = log_returns[(start + 1):(end + 1)]
        # Compute the constant that closes the segment on its anchor.
        gap = anchor_logs[i + 1] - anchor_logs[i] - steps.sum()
        # Spread the gap evenly over the segment.
        drift = gap / len(steps)
        # Accumulate the corrected log level over the segment.
        log_level[(start + 1):(end + 1)] = anchor_logs[i] + np.cumsum(
            steps + drift
        )
    # Assemble the latent states.
    states = pd.DataFrame(
        {
            "ndx_close": np.exp(log_level),
            "sigma": sigma,
            "equity_shock": shock,
            "scale": scale,
        },
        index=sessions,
    )
    # Walk the six factors in the matrix order.
    for position, name in enumerate(FACTOR_VOLATILITY):
        # Store the factor's unit shock, scaled by w_t.
        states[f"shock_{name}"] = factors[:, position] * scale
    # Return the states.
    return states
```

#### **Step 1.4: `YAHOO_FINANCE_RAW`**

```python
def generate_yahoo_finance_raw(
    states: pd.DataFrame,
    holidays: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate YAHOO_FINANCE_RAW, the raw daily market panel.

    Purpose:
        Supplies the NASDAQ-100 level behind r_t = (P_t - P_{t-1}) / P_{t-1}
        and Y_t = sum_{k=0}^{4} r_{t-k} of Equation (3), plus the VIX and
        US dollar levels that enter the feature universe.

    Inputs:
        states (pd.DataFrame): Output of simulate_market_states.
        holidays (pd.DatetimeIndex): Weekdays that are not NYSE sessions.
        rng (np.random.Generator): The seeded PCG64 generator.

    Processes:
        1. Maps volatility to the VIX: VIX_t = 80 sqrt(252) sigma_t + 2,
           plus noise, clipped to [9, 83].
        2. Simulates a mean-reverting log US dollar index.
        3. Simulates volume as a rising trend with lognormal noise.
        4. Adds vendor-only dates: the dollar index and VIX print on them,
           the NASDAQ-100 does not, so ndx_close is missing and
           ndx_volume is zero. Task 2 removes these rows.

    Outputs:
        pd.DataFrame: Columns trading_date, ndx_close, ndx_volume,
        vix_close, usd_index_close, source_fetched_at_utc and
        raw_adjustment_flag, typed as config.yaml declares.

    Raises:
        ValueError: If the state frame lacks a required column.
    """
    # Name the state columns the generator reads.
    required = {"ndx_close", "sigma"}
    # Refuse a state frame that lacks them.
    if not required.issubset(states.columns):
        # Name the missing columns.
        raise ValueError(f"states lacks {sorted(required - set(states))}.")
    # Count the sessions.
    n_sessions = len(states)
    # Map daily volatility to annualised points, with a variance premium.
    vix = 80.0 * np.sqrt(252.0) * states["sigma"].to_numpy() + 2.0
    # Add measurement noise and keep the level in its historical range.
    vix = np.clip(vix + rng.normal(0.0, 0.35, n_sessions), 9.0, 83.0)
    # Fix the long-run log level of the dollar index.
    usd_mean = np.log(93.0)
    # Allocate the log dollar index path.
    usd_log = np.empty(n_sessions)
    # Start near the 1999 level.
    usd_log[0] = np.log(99.3)
    # Iterate the mean-reverting log dollar index.
    for t in range(1, n_sessions):
        # Pull weakly towards the long-run level and add a daily shock.
        usd_log[t] = (
            usd_log[t - 1]
            + 0.002 * (usd_mean - usd_log[t - 1])
            + rng.normal(0.0, 0.0042)
        )
    # Build the rising volume trend, 1.0bn to 6.0bn shares.
    volume_trend = np.linspace(np.log(1.0e9), np.log(6.0e9), n_sessions)
    # Add lognormal noise and round to whole shares.
    volume = np.exp(volume_trend + rng.normal(0.0, 0.25, n_sessions)).round()
    # Assemble the session rows.
    sessions_frame = pd.DataFrame(
        {
            "ndx_close": states["ndx_close"].to_numpy(),
            "ndx_volume": volume,
            "vix_close": np.round(vix, 2),
            "usd_index_close": np.round(np.exp(usd_log), 2),
        },
        index=states.index,
    )
    # Draw the vendor-only dates from the exchange holidays.
    extra_dates = pd.DatetimeIndex(
        rng.choice(holidays, N_VENDOR_ONLY_DATES, replace=False)
    ).sort_values()
    # Merge the session and vendor-only dates.
    axis = states.index.union(extra_dates)
    # Reindex the panel onto the vendor axis.
    panel = sessions_frame.reindex(axis)
    # Carry the VIX and dollar levels onto the vendor-only dates.
    panel[["vix_close", "usd_index_close"]] = panel[
        ["vix_close", "usd_index_close"]
    ].ffill()
    # Record zero volume on the dates without a NASDAQ-100 print.
    panel["ndx_volume"] = panel["ndx_volume"].fillna(0.0)
    # Move the date index into the primary-key column.
    panel = panel.rename_axis("trading_date").reset_index()
    # Record the retrieval timestamp (provenance).
    panel["source_fetched_at_utc"] = FETCHED_AT_UTC
    # Record that the vendor returned adjusted closes.
    panel["raw_adjustment_flag"] = "adjusted"
    # Return the panel typed as raw_input_schemas declares.
    return panel.astype(
        {
            "trading_date": "datetime64[ns]",
            "ndx_close": "float64",
            "ndx_volume": "int64",
            "vix_close": "float64",
            "usd_index_close": "float64",
            "source_fetched_at_utc": "datetime64[ns]",
            "raw_adjustment_flag": "object",
        }
    )
```

#### **Step 1.5: `FRED_MACRO_RAW`**

```python
def _bounded_path(
    knots: Mapping[str, float],
    axis: pd.DatetimeIndex,
    noise_sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Interpolate a slow path through dated knots and add AR(1) noise.

    Purpose:
        Gives each macro series a realistic history without copying any
        published observation.

    Inputs:
        knots (Mapping[str, float]): Date-to-level pairs, at least two.
        axis (pd.DatetimeIndex): The dates to evaluate.
        noise_sd (float): Standard deviation of the AR(1) innovations.
        rng (np.random.Generator): The seeded PCG64 generator.

    Processes:
        1. Interpolates the knots linearly in time.
        2. Adds AR(1) noise with coefficient 0.98.

    Outputs:
        np.ndarray: The path on the axis.

    Raises:
        ValueError: If fewer than two knots are given or noise_sd < 0.
    """
    # Refuse a path without two knots.
    if len(knots) < 2:
        # Name the malformed input.
        raise ValueError("at least two knots are required.")
    # Refuse a negative noise scale.
    if noise_sd < 0.0:
        # Name the malformed input.
        raise ValueError(f"noise_sd must be non-negative; got {noise_sd}.")
    # Read the knot dates as nanosecond integers.
    knot_x = pd.DatetimeIndex(list(knots)).asi8.astype("float64")
    # Interpolate the knot levels on the axis.
    level = np.interp(
        axis.as_unit("ns").asi8.astype("float64"),
        knot_x,
        np.fromiter(knots.values(), dtype="float64"),
    )
    # Allocate the AR(1) noise.
    noise = np.empty(len(axis))
    # Start the noise at zero.
    noise[0] = 0.0
    # Draw the innovations in one call.
    innovations = rng.normal(0.0, noise_sd, len(axis))
    # Iterate the AR(1) recursion.
    for t in range(1, len(axis)):
        # Apply n_t = 0.98 n_{t-1} + e_t.
        noise[t] = 0.98 * noise[t - 1] + innovations[t]
    # Return the noisy path.
    return level + noise


def generate_fred_macro_raw(
    states: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate FRED_MACRO_RAW, the raw macroeconomic panel.

    Purpose:
        Supplies the daily rates, the 2s10s spread and the high-yield OAS,
        whose differences carry the credit-spread drag of Table 4, plus the
        monthly CPI, unemployment and payroll series. Release lags are NOT
        applied here: Task 7 applies them at the join.

    Inputs:
        states (pd.DataFrame): Output of simulate_market_states.
        rng (np.random.Generator): The seeded PCG64 generator.

    Processes:
        1. Builds weekdays from FIRST_CREDIT_DATE, as FRED publishes them.
        2. Simulates the rates from FIRST_VENDOR_DATE, and sets
           spread_2s10s to rate_long - rate_short, which may be negative.
        3. Simulates the log high-yield OAS with an AR(1) pull and a shock
           loaded negatively on the equity shock.
        4. Stamps the monthly series on the first of each month, as FRED
           dates them, and adds weekend rows where the first falls on a
           weekend. Daily series are missing on those rows and on exchange
           holidays.

    Outputs:
        pd.DataFrame: Columns observation_date, rate_short, rate_long,
        spread_2s10s, spread_hy, inflation_cpi, labor_unrate, labor_payems.

    Raises:
        ValueError: If the state frame lacks equity_shock.
    """
    # Refuse a state frame without the equity shock.
    if "equity_shock" not in states.columns:
        # Name the missing column.
        raise ValueError("states lacks equity_shock.")
    # Build the weekday axis of the credit history.
    weekdays = pd.bdate_range(FIRST_CREDIT_DATE, LAST_DATE)
    # Align the equity shock to the weekdays; other days have no shock.
    equity = states["equity_shock"].reindex(weekdays).fillna(0.0).to_numpy()
    # Fix the long-run log OAS (about 4.6 percentage points).
    hy_mean = np.log(4.6)
    # Allocate the log OAS path.
    hy_log = np.empty(len(weekdays))
    # Start at the 1996 level.
    hy_log[0] = np.log(3.1)
    # Draw the idiosyncratic credit shocks.
    hy_noise = rng.normal(0.0, 0.012, len(weekdays))
    # Iterate the log OAS: a pull to its mean and a leverage-style shock.
    for t in range(1, len(weekdays)):
        # Widen the spread when equities fall.
        hy_log[t] = (
            hy_log[t - 1]
            + 0.004 * (hy_mean - hy_log[t - 1])
            - 0.032 * equity[t]
            + hy_noise[t]
        )
    # Build the daily frame.
    daily = pd.DataFrame(
        {"spread_hy": np.round(np.exp(hy_log), 2)}, index=weekdays
    )
    # Locate the rate history.
    rate_axis = weekdays[weekdays >= FIRST_VENDOR_DATE]
    # Simulate the short rate through dated knots.
    short = _bounded_path(
        {
            "1999-11-01": 5.8,
            "2003-06-30": 1.2,
            "2006-12-31": 4.9,
            "2011-06-30": 0.4,
            "2018-12-31": 2.6,
            "2020-08-01": 0.15,
            "2023-10-01": 5.0,
            "2026-03-31": 3.8,
        },
        rate_axis,
        0.012,
        rng,
    )
    # Simulate the long rate through dated knots.
    long = _bounded_path(
        {
            "1999-11-01": 6.1,
            "2003-06-30": 3.5,
            "2007-06-30": 5.1,
            "2012-07-31": 1.5,
            "2018-10-31": 3.2,
            "2020-08-01": 0.6,
            "2023-10-15": 4.9,
            "2026-03-31": 4.3,
        },
        rate_axis,
        0.012,
        rng,
    )
    # Quote the short rate to two decimals, above its historical floor.
    daily.loc[rate_axis, "rate_short"] = np.round(np.maximum(short, 0.09), 2)
    # Quote the long rate to two decimals, above its historical floor.
    daily.loc[rate_axis, "rate_long"] = np.round(np.maximum(long, 0.52), 2)
    # Form the spread without clipping it at zero.
    daily["spread_2s10s"] = (daily["rate_long"] - daily["rate_short"]).round(2)
    # Blank the Treasury series on exchange holidays.
    closed = rate_axis.difference(states.index)
    # Treasury series do not print on those days.
    daily.loc[closed, ["rate_short", "rate_long", "spread_2s10s"]] = np.nan
    # List the monthly reference dates (first of month, as FRED dates them).
    months = pd.date_range(FIRST_VENDOR_DATE, LAST_DATE, freq="MS")
    # Simulate monthly log CPI growth of about 0.21% a month.
    cpi = 168.4 * np.exp(np.cumsum(rng.normal(0.0021, 0.0025, len(months))))
    # Simulate unemployment through recession knots.
    unemployment = _bounded_path(
        {
            "1999-11-01": 4.1,
            "2003-06-01": 6.3,
            "2007-05-01": 4.4,
            "2009-10-01": 10.0,
            "2020-02-01": 3.5,
            "2020-04-01": 14.8,
            "2022-01-01": 4.0,
            "2026-03-01": 4.3,
        },
        months,
        0.06,
        rng,
    )
    # Simulate payrolls through dated knots, in thousands of persons.
    payrolls = _bounded_path(
        {
            "1999-11-01": 130466.0,
            "2008-01-01": 138400.0,
            "2010-02-01": 129700.0,
            "2020-02-01": 152500.0,
            "2020-04-01": 130400.0,
            "2023-01-01": 155000.0,
            "2026-03-01": 158600.0,
        },
        months,
        45.0,
        rng,
    )
    # Build the monthly frame.
    monthly = pd.DataFrame(
        {
            "inflation_cpi": np.round(cpi, 3),
            "labor_unrate": np.round(unemployment, 1),
            "labor_payems": np.round(payrolls, 0),
        },
        index=months,
    )
    # Outer-join on the reference date, which adds weekend first-of-months.
    panel = daily.join(monthly, how="outer").sort_index()
    # Move the date index into the primary-key column.
    panel = panel.rename_axis("observation_date").reset_index()
    # Fix the column order of raw_input_schemas.
    columns = [
        "observation_date",
        "rate_short",
        "rate_long",
        "spread_2s10s",
        "spread_hy",
        "inflation_cpi",
        "labor_unrate",
        "labor_payems",
    ]
    # Return the panel at the declared dtypes.
    return panel[columns].astype(
        {"observation_date": "datetime64[ns]"}
        | {column: "float64" for column in columns[1:]}
    )
```

#### **Step 1.6: `FACTOR_LIBRARY_RAW`**

```python
def generate_factor_library_raw(states: pd.DataFrame) -> pd.DataFrame:
    """
    Generate FACTOR_LIBRARY_RAW, the raw daily factor panel.

    Purpose:
        Supplies the Fama-French five factors and the Carhart momentum
        factor, whose rolling means and deviations enter Table 4.

    Inputs:
        states (pd.DataFrame): Output of simulate_market_states.

    Processes:
        1. Scales each unit shock by its daily volatility and by the
           current volatility regime, sigma_{t-1} / mean(sigma).
        2. Rounds to two decimals, as the data library publishes them.

    Outputs:
        pd.DataFrame: Columns date, MKT, SMB, HML, RMW, CMA, MOM and
        vendor_vintage_id. Factors are in PERCENT per session, the unit of
        the library's daily files and of the study's input panel.

    Raises:
        ValueError: If a factor shock column is missing.
    """
    # Name the shock columns the generator reads.
    required = {f"shock_{name}" for name in FACTOR_VOLATILITY}
    # Refuse a state frame that lacks them.
    if not required.issubset(states.columns):
        # Name the missing columns.
        raise ValueError(f"states lacks {sorted(required - set(states))}.")
    # Read the volatility path.
    sigma = states["sigma"].to_numpy()
    # Lag it and normalise it into a volatility regime.
    regime = np.r_[sigma[0], sigma[:-1]] / sigma.mean()
    # Start the panel from the session dates.
    panel = pd.DataFrame({"date": states.index})
    # Walk the factors in the declared order.
    for name, volatility in FACTOR_VOLATILITY.items():
        # Scale the shock and round to the library's two decimals.
        panel[name] = np.round(
            volatility * regime * states[f"shock_{name}"].to_numpy(), 2
        )
    # Record the library vintage (provenance).
    panel["vendor_vintage_id"] = VENDOR_VINTAGE_ID
    # Return the panel at the declared dtypes.
    return panel.astype(
        {"date": "datetime64[ns]", "vendor_vintage_id": "object"}
    )
```

#### **Step 1.7: `TRADING_CALENDAR`**

```python
def generate_trading_calendar(
    sessions: pd.DatetimeIndex, half_days: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Generate TRADING_CALENDAR, the canonical session calendar.

    Purpose:
        Fixes the observation axis of the study. Half-days are kept: they
        are valid closing-price dates.

    Inputs:
        sessions (pd.DatetimeIndex): The NYSE sessions of the vendor span.
        half_days (pd.DatetimeIndex): The early-close sessions.

    Processes:
        1. Keeps the sessions from FIRST_CALENDAR_DATE onwards.
        2. Flags every kept row open, and flags the early closes.
        3. Types the exchange code as a category.

    Outputs:
        pd.DataFrame: Columns trading_date, session_open_flag,
        half_day_flag, exchange, calendar_source and
        calendar_derivation_rule. Tasks 1 to 3 read the first four; the
        last two are provenance columns outside raw_input_schemas.

    Raises:
        ValueError: If the sessions are not unique and increasing.
    """
    # Refuse an axis that is not unique and increasing.
    if not (sessions.is_unique and sessions.is_monotonic_increasing):
        # Name the malformed axis.
        raise ValueError("sessions must be unique and increasing.")
    # Keep the sessions of the study span.
    kept = sessions[sessions >= FIRST_CALENDAR_DATE]
    # Assemble the calendar.
    calendar = pd.DataFrame(
        {
            "trading_date": kept,
            "session_open_flag": True,
            "half_day_flag": kept.isin(half_days),
            "exchange": "XNAS",
            "calendar_source": CALENDAR_SOURCE,
            "calendar_derivation_rule": CALENDAR_DERIVATION_RULE,
        }
    )
    # Return the calendar at the declared dtypes.
    return calendar.astype(
        {
            "trading_date": "datetime64[ns]",
            "session_open_flag": "bool",
            "half_day_flag": "bool",
            "exchange": "category",
        }
    )
```

#### **Step 1.8: The `raw_inputs` mapping**

```python
def generate_raw_inputs(seed: int = DGP_SEED) -> Dict[str, pd.DataFrame]:
    """
    Generate the four raw panels that run_task_34 ingests.

    Purpose:
        Single entry point of Step 1. Every panel draws on one latent
        market, so cross-panel co-movement is preserved.

    Inputs:
        seed (int): Seed of the PCG64 generator. Default is DGP_SEED.

    Processes:
        1. Builds the session axes.
        2. Simulates the latent market states.
        3. Generates the panels in a fixed order.

    Outputs:
        Dict[str, pd.DataFrame]: Keys YAHOO_FINANCE_RAW, FRED_MACRO_RAW,
        FACTOR_LIBRARY_RAW and TRADING_CALENDAR.

    Raises:
        TypeError: If seed is not an integer.
    """
    # Refuse a seed that is not an integer.
    if isinstance(seed, bool) or not isinstance(seed, int):
        # Name the malformed seed.
        raise TypeError(f"seed must be an integer; got {type(seed)}.")
    # Create the seeded PCG64 generator.
    rng = np.random.Generator(np.random.PCG64(seed))
    # Build the session axes.
    axes = build_session_axes()
    # Simulate the latent market.
    states = simulate_market_states(axes["sessions"], rng)
    # Return the four panels under the driver's keys.
    return {
        "YAHOO_FINANCE_RAW": generate_yahoo_finance_raw(
            states, axes["holidays"], rng
        ),
        "FRED_MACRO_RAW": generate_fred_macro_raw(states, rng),
        "FACTOR_LIBRARY_RAW": generate_factor_library_raw(states),
        "TRADING_CALENDAR": generate_trading_calendar(
            axes["sessions"], axes["half_days"]
        ),
    }


# Generate the four raw panels.
raw_inputs = generate_raw_inputs()

# Report each panel's shape and date span.
for panel_name, frame in raw_inputs.items():
    # Read the panel's first column, which is its date key.
    dates = frame.iloc[:, 0]
    # Print the panel name, shape and span.
    print(
        f"{panel_name:>18}: {str(frame.shape):>11} "
        f"{dates.min().date()} to {dates.max().date()}"
    )
```

Output:

```text
 YAHOO_FINANCE_RAW:   (6676, 7) 1999-11-01 to 2026-03-31
    FRED_MACRO_RAW:   (7723, 8) 1996-12-31 to 2026-03-31
FACTOR_LIBRARY_RAW:   (6643, 8) 1999-11-01 to 2026-03-31
  TRADING_CALENDAR:   (6600, 6) 2000-01-03 to 2026-03-31
```

### **Step 2: Loading `config.yaml`**

The file has 34 top-level blocks: the study metadata, the raw input schemas, the feature generation and preprocessing policies, the equations (`eq1_policy` through `eq12_ensemble`), Algorithm 1, Tables 2 to 4, the baselines, the outer tuning, the compute and reproducibility settings, and the documented gaps and verification targets. The configurations the `configuration_name` argument selects are stored in `table2_configs`:

| Parameter | `fixed` | `tuned` | `endogenous` |
| --- | --- | --- | --- |
| Training window (sessions) | 1,890 | 1,890 | 1,000 |
| Target features $K$ | 60 | 15 | endogenous |
| Redundancy metric | mean | rms | top-3-rms |
| $\gamma$ | 0.666 | 0.772 | 0.811 |
| $\beta_{mag}$ | 24.82 | 14.97 | 13.72 |
| $\beta_{stop}$ | none | none | 0.909 |
| $\tau$ | 1.52 | 0.868 | 1.515 |
| $\alpha_{L1}$ | 1.514 | 2.003 | 0.177 |
| $\alpha_{high}$ | 1.514 | 2.003 | 2.212 |
| Iterations $I_1, I_2, I_3$ | 84, 3, 6 | 18, 28, 18 | var., 30, 15 |
| Level-1 patience | none | none | 63 |
| Ensemble runs (Table 3) | 18 | 18 | 12 |

```python
def load_study_configuration(
    filepath: str | Path = "config.yaml",
) -> Dict[str, Any]:
    """
    Read config.yaml from the working directory into a dictionary.

    Purpose:
        Separates the study parameters from the code. The dictionary is
        passed to run_task_34 unchanged.

    Inputs:
        filepath (str | Path): Path of the file. Default is "config.yaml".

    Processes:
        1. Refuses a missing file.
        2. Parses the file with yaml.safe_load, which cannot run code.
        3. Requires a mapping that holds the blocks run_task_34 reads.

    Outputs:
        Dict[str, Any]: The nested study configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML is invalid, is not a mapping, or lacks a
        required block.
    """
    # Normalise the path.
    path = Path(filepath)
    # Refuse a missing file, rather than continuing with an empty mapping.
    if not path.is_file():
        # Name the file and the directory searched.
        raise FileNotFoundError(f"{path} not found in {Path.cwd()}.")
    # Parse the file, translating a syntax error into a clear message.
    try:
        # Read the text as UTF-8 and parse it safely.
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    # Catch any YAML parsing failure.
    except yaml.YAMLError as error:
        # Name the file and keep the original error as the cause.
        raise ValueError(f"{path} is not valid YAML: {error}") from error
    # Refuse a document that is not a mapping.
    if not isinstance(config, dict):
        # Name the unexpected type.
        raise ValueError(f"{path} must hold a mapping; got {type(config)}.")
    # Name the blocks the pipeline cannot run without.
    required = {
        "raw_input_schemas",
        "feature_generation_policy",
        "eq3_target_and_partition",
        "algorithm_1_controller",
        "table2_configs",
        "baselines",
    }
    # Find any absent block.
    missing = required - set(config)
    # Refuse an incomplete configuration.
    if missing:
        # Name the absent blocks.
        raise ValueError(f"{path} lacks {sorted(missing)}.")
    # Return the configuration.
    return config


# Read config.yaml from the working directory as the dictionary `config`.
config = load_study_configuration("config.yaml")

# Report the size of the configuration.
print(f"{len(config)} top-level blocks loaded from config.yaml")
# Show that the stabiliser of Equation (6) parsed as a number.
print("Equation (6) lambda:", config["eq6_fit"]["stabilizer_value"])
# Show the three realised training windows of Table 2.
print("Table 2 windows:", config["table2_configs"]["training_window_days"])
```

Output:

```text
34 top-level blocks loaded from config.yaml
Equation (6) lambda: 1e-05
Table 2 windows: {'fixed': 1890, 'tuned': 1890, 'endogenous': 1000}
```

`yaml.safe_load` returns the stabiliser as the float `1e-05`, because the file writes it with a decimal point. A form such as `1e-5` would be read as a string by PyYAML.

### **Step 3: Composing and validating the arguments**

`prepare_run_arguments` fails before any computation when an argument is malformed, and it writes out every value the call will use. `RUN_TASK_34_DEFAULTS` restates the original defaults of the signature, so an override is always visible as a departure from them. The seed is read from `config.yaml`, and it equals the value that `seed=None` would resolve to.

```python
# Fix a short output root: Windows refuses paths over 260 characters.
OUTPUT_ROOT: Path = Path.home() / "l0_runs"

# Fix the three configurations of Table 2.
CONFIGURATION_NAMES: Tuple[str, ...] = ("fixed", "tuned", "endogenous")

# Fix the two modes of run_task_34.
RUN_MODES: Tuple[str, ...] = ("full", "smoke")

# Fix the panel keys that run_task_34 requires.
RAW_INPUT_KEYS: Tuple[str, ...] = (
    "YAHOO_FINANCE_RAW",
    "FRED_MACRO_RAW",
    "FACTOR_LIBRARY_RAW",
    "TRADING_CALENDAR",
)

# Restate the signature's keyword defaults, for reference and validation.
RUN_TASK_34_DEFAULTS: Dict[str, Any] = {
    "seed": None,
    "use_walk_forward_fold": False,
    "audit_log_path": None,
    "output_dir": None,
    "run_index": 0,
    "enable_tuning": False,
    "ensemble_runs": None,
    "baseline_cv": False,
    "run_baselines": True,
    "strict_reporting": False,
    "allow_missing_calendar_oracle": False,
    "policy_norm_bound": None,
    "prepared": None,
}


def prepare_run_arguments(
    raw_inputs: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    configuration_name: str,
    mode: str,
    output_dir: str | Path,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Compose and validate the arguments of one run_task_34 call.

    Purpose:
        Fails before any computation when an argument is malformed, and
        states every value the call will use.

    Inputs:
        raw_inputs (Mapping[str, pd.DataFrame]): The four raw panels.
        config (Mapping[str, Any]): The dictionary read from config.yaml.
        configuration_name (str): "fixed", "tuned" or "endogenous".
        mode (str): "full" or "smoke".
        output_dir (str | Path): The run's artifact directory.
        **overrides (Any): Keyword arguments of run_task_34 that differ
            from RUN_TASK_34_DEFAULTS.

    Processes:
        1. Requires the four panels, the configuration name and the mode.
        2. Reads the run seed from config.yaml.
        3. Merges the defaults and the overrides, refusing unknown names.
        4. Creates the output directory.

    Outputs:
        Dict[str, Any]: Keyword arguments for run_task_34.

    Raises:
        ValueError: If a panel, name, mode or keyword is invalid.
    """
    # Find any absent panel.
    missing = set(RAW_INPUT_KEYS) - set(raw_inputs)
    # Refuse an incomplete input mapping.
    if missing:
        # Name the absent panels.
        raise ValueError(f"raw_inputs lacks {sorted(missing)}.")
    # Refuse an unknown configuration.
    if configuration_name not in CONFIGURATION_NAMES:
        # Name the valid configurations.
        raise ValueError(f"configuration_name: {CONFIGURATION_NAMES}.")
    # Refuse an unknown mode.
    if mode not in RUN_MODES:
        # Name the valid modes.
        raise ValueError(f"mode must be one of {RUN_MODES}.")
    # Find any keyword that run_task_34 does not accept.
    unknown = set(overrides) - set(RUN_TASK_34_DEFAULTS)
    # Refuse those keywords.
    if unknown:
        # Name them.
        raise ValueError(f"unknown run_task_34 keywords: {sorted(unknown)}.")
    # Normalise the output directory.
    directory = Path(output_dir)
    # Create it, including its parents.
    directory.mkdir(parents=True, exist_ok=True)
    # Read the register's numpy seed from config.yaml.
    seeds = config["compute_and_reproducibility"]["global_random_seeds"]
    # Merge the defaults, the stated values and the overrides.
    arguments = {
        **RUN_TASK_34_DEFAULTS,
        "seed": seeds["numpy_seed"],
        "output_dir": directory,
        **overrides,
    }
    # Add the two arguments that precede the keyword-only marker.
    arguments.update(configuration_name=configuration_name, mode=mode)
    # Return the composed keyword arguments.
    return arguments
```

```python
def execute_study(
    raw_inputs: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Call run_task_34 and report what happened.

    Purpose:
        The single point at which the pipeline is invoked, so a failure
        names its configuration and mode.

    Inputs:
        raw_inputs (Mapping[str, pd.DataFrame]): The four raw panels.
        config (Mapping[str, Any]): The dictionary read from config.yaml.
        arguments (Mapping[str, Any]): Output of prepare_run_arguments.

    Processes:
        1. Calls run_task_34, the top-level orchestrator of Tasks 1 to 33.
        2. Times the call.
        3. Reports any reporting stage that failed without stopping the
           run.

    Outputs:
        Dict[str, Any]: The mapping returned by run_task_34.

    Raises:
        ValueError: If run_task_34 rejects an input.
        RuntimeError: If the budget preflight, the barrier or a smoke
        check fails.
    """
    # Copy the arguments, so the caller's mapping is untouched.
    keywords = dict(arguments)
    # Remove the configuration name, passed positionally.
    configuration_name = keywords.pop("configuration_name")
    # Remove the mode, passed positionally.
    mode = keywords.pop("mode")
    # Start the timer.
    started = time.perf_counter()
    # Run the pipeline, adding context to any failure.
    try:
        # Invoke the study interface.
        result = run_task_34(
            raw_inputs, config, configuration_name, mode, **keywords
        )
    # Catch the two exception types that run_task_34 documents.
    except (ValueError, RuntimeError) as error:
        # Name the run and keep the original error as the cause.
        raise type(error)(
            f"run_task_34({configuration_name!r}, {mode!r}) failed: {error}"
        ) from error
    # Measure the elapsed time.
    elapsed = time.perf_counter() - started
    # Report it.
    print(f"{configuration_name}/{mode}: {elapsed:.1f} seconds")
    # Report any recorded reporting failure.
    for failure in result.get("reporting_failures") or []:
        # Print the failure record.
        print("reporting failure:", failure)
    # Return the result.
    return result


def stage_table(result: Mapping[str, Any]) -> pd.DataFrame:
    """
    Tabulate the stage log of a run_task_34 result.

    Purpose:
        Shows which stages ran, their status and their duration.

    Inputs:
        result (Mapping[str, Any]): The mapping returned by run_task_34.

    Processes:
        Converts the list of stage records to a DataFrame.

    Outputs:
        pd.DataFrame: One row per stage, with stage, status and seconds.

    Raises:
        KeyError: If the result has no stage_log.
    """
    # Refuse a result without a stage log.
    if "stage_log" not in result:
        # Name the missing key.
        raise KeyError("the result has no stage_log.")
    # Build the table and round the durations.
    return pd.DataFrame(result["stage_log"]).round({"seconds": 2})
```

```python
def summarize_run(result: Mapping[str, Any]) -> pd.DataFrame:
    """
    Tabulate the headline figures of a full-mode run_task_34 result.

    Purpose:
        Collects, in one table, the figures that the result stores in
        several nested records.

    Inputs:
        result (Mapping[str, Any]): The mapping returned by run_task_34
            with mode="full".

    Processes:
        1. Refuses a smoke result, which has no test-block figures.
        2. Reads the single-run, ensemble and per-run R^2 values.
        3. Reads the baseline, greedy and sparsity-stress records.
        4. Reads the one-sided t-test of Task 32.

    Outputs:
        pd.DataFrame: Columns "quantity" and "value".

    Raises:
        ValueError: If the result does not come from a full-mode run.
    """
    # Refuse a result from a smoke run.
    if result.get("mode") != "full":
        # Name the required mode.
        raise ValueError("summarize_run requires a full-mode result.")
    # Read the per-run dispersion of Task 31.
    dispersion = result["oos_dispersion"]
    # Read the three penalised baselines of Task 25.
    baselines = result["baselines"]
    # Read the sparsity-stress record of Task 27.
    stress = result["sparsity_stress"]
    # Read the t-test record of Task 32, which a failed stage leaves empty.
    ttest = (result.get("inference") or {}).get("ttest") or {}
    # List the quantity and value pairs.
    rows = [
        ("single-run test R^2 (seed run)", result["single_run_oos_r2"]),
        ("ensemble test R^2 (headline)", result["oos_r2"]),
        ("mean of per-run test R^2", dispersion["mean"]),
        ("std of per-run test R^2", dispersion["std"]),
        ("runs in the ensemble", dispersion["n_runs"]),
        ("features in the ensemble", len(result["selected_features"])),
        ("Ridge test R^2", baselines["ridge_L2"]["oos_r2"]),
        ("Lasso test R^2", baselines["lasso_L1"]["oos_r2"]),
        ("Lasso features", baselines["lasso_L1"]["n_selected"]),
        ("Elastic Net test R^2", baselines["elastic_net"]["oos_r2"]),
        ("greedy test R^2", result["greedy"]["oos_r2"]),
        ("greedy features", result["greedy"]["n_features"]),
        ("sparsity-stress Lasso features", stress["n_features"]),
        ("sparsity-stress Lasso test R^2", stress["oos_r2"]),
        ("t statistic against the baseline", ttest.get("t_statistic")),
        ("one-sided p-value", ttest.get("p_value")),
    ]
    # Return the table.
    return pd.DataFrame(rows, columns=["quantity", "value"])


def list_artifacts(directory: str | Path) -> pd.DataFrame:
    """
    Count the files that a run wrote into its output directory.

    Purpose:
        Shows where each stage left its audit and result files.

    Inputs:
        directory (str | Path): The run's output_dir.

    Processes:
        1. Walks the directory recursively.
        2. Groups the files by their first path component.

    Outputs:
        pd.DataFrame: Columns "entry" and "files".

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    # Normalise the path.
    root = Path(directory)
    # Refuse a missing directory.
    if not root.is_dir():
        # Name the directory.
        raise FileNotFoundError(f"{root} does not exist.")
    # List every file under the directory.
    files = [path for path in root.rglob("*") if path.is_file()]
    # Read each file's first component below the root.
    entries = [path.relative_to(root).parts[0] for path in files]
    # Count the files per entry.
    counts = pd.Series(entries, dtype="object").value_counts().sort_index()
    # Return the counts as a table.
    return counts.rename_axis("entry").reset_index(name="files")
```

### **Step 4: Running the pipeline**

#### **Step 4.1: The smoke call, for each configuration**

Smoke mode is the first thing to run on any new data. It executes Tasks 1 to 23 at $I_1 = I_2 = I_3 = 1$ and never reads the test block.

```python
# Hold the smoke result of each configuration.
smoke_results: Dict[str, Dict[str, Any]] = {}

# Walk the three configurations of Table 2.
for configuration in CONFIGURATION_NAMES:
    # Compose the arguments of the smoke call.
    smoke_arguments = prepare_run_arguments(
        raw_inputs,
        config,
        configuration,
        "smoke",
        OUTPUT_ROOT / f"smoke_{configuration}",
    )
    # Run the smoke test and keep its result.
    smoke_results[configuration] = execute_study(
        raw_inputs, config, smoke_arguments
    )

# Show the stage log of the first configuration.
print(stage_table(smoke_results["fixed"]).to_string())

# Walk the three smoke records.
for configuration, result in smoke_results.items():
    # Read the record of the smoke checks.
    checks = result["smoke"]
    # Print the selected subset size and the barrier's overlap count.
    print(
        configuration,
        checks["completion"]["n_features"],
        checks["policy_convergence"]["converged"],
        checks["information_barrier"]["overlap_count"],
    )

# Print the barrier record, which is the same for every configuration.
print(smoke_results["fixed"]["information_barrier"])
```

Output (standard output only). The run also writes warnings to standard error: SciPy's `LinAlgWarning` for ill-conditioned candidate subsets, and `UserWarning` lines that record the documented differences from the paper, 6,600 against 6,260 sessions and 877 against 1,072 features.

```text
fixed/smoke: 122.8 seconds
tuned/smoke: 122.2 seconds
endogenous/smoke: 121.9 seconds
                         stage     status  seconds
0                       task_1  completed     0.09
1                       task_3  completed     4.16
2                       task_2  completed     0.20
3       task_7_macro_alignment  completed     0.14
4      task_7_factor_alignment  completed     0.08
5                       task_4  completed     0.01
6                       task_5  completed     1.64
7                       task_6  completed     1.36
8   task_6_information_barrier  completed   113.02
9                       task_7  completed     0.60
10                      task_8  completed     0.37
11           tasks_9_23_search  completed     0.64
12                smoke_checks  completed     0.03
fixed 60 True 0
tuned 15 True 0
endogenous 29 True 0
{'search_max_date': '2023-12-29', 'test_start': '2024-01-02', 'n_search_dates': 2452, 'n_test_dates': 563, 'overlap_count': 0, 'payload_has_test_key': False, 'test_fingerprint_unchanged': True}
```

The feature count of 60 under `fixed` is the target cardinality enforced by the hard stop mask. The stage names show the order in which the pipeline prepares the data: the calendar comes before cleansing, and the macro and factor alignment come before the target and feature stages.

#### **Step 4.2: The time-boxed full call**

This call runs every stage of the interface except tuning. The only departure from the defaults is `ensemble_runs=6` instead of 18. It took about 44 minutes on the verification machine, a CPU-only Windows workstation.

```python
# Compose the time-boxed full call: six ensemble runs instead of 18.
full_arguments = prepare_run_arguments(
    raw_inputs,
    config,
    "fixed",
    "full",
    OUTPUT_ROOT / "full_fixed",
    ensemble_runs=6,
)

# Run Tasks 1 to 33 through the study interface.
full_result = execute_study(raw_inputs, config, full_arguments)

# Show the stage log of the full run.
print(stage_table(full_result).to_string())
```

Output:

```text
fixed/full: 2633.9 seconds
                               stage     status  seconds
0                             task_1  completed     0.10
1                             task_3  completed     4.16
2                             task_2  completed     0.19
3             task_7_macro_alignment  completed     0.17
4            task_7_factor_alignment  completed     0.07
5                             task_4  completed     0.01
6                             task_5  completed     1.64
7                             task_6  completed     1.26
8         task_6_information_barrier  completed   112.24
9                             task_7  completed     0.82
10                            task_8  completed     0.42
11  tasks_9_23_search_and_evaluation  completed   214.93
12                           task_24  completed     0.00
13                           task_25  completed     5.46
14                           task_26  completed   445.53
15                           task_27  completed   410.92
16                           task_30  completed  1432.06
17                           task_31  completed     0.03
18                           task_32  completed     0.02
19                           task_33  completed     3.56
```

#### **Step 4.3: Reading the result**

`summarize_run` collects the headline figures. `list_artifacts` shows what the run left on disk.

```python
# Show the headline figures of the run.
print(summarize_run(full_result).to_string())

# Show the first five names of the seed run's subset.
print(full_result["search_artifact"]["selected_names"][:5])

# Show the number of nonzero coefficients of the seed run.
print(int(np.count_nonzero(full_result["coefficients"])))

# Show the validation R^2 of the terminal subset.
print(round(full_result["search_artifact"]["validation_r2"], 4))

# Show the subset size of each ensemble member.
print(full_result["subset_sizes"])

# Show the barrier record, now with the fingerprint comparison.
print(full_result["information_barrier"])

# Show whether any reporting stage failed.
print(full_result["reporting_failures"])

# Show what Task 32 concluded from the checks it ran.
print(full_result["inference"]["all_checks_passed"])

# Show the files the run wrote, by stage.
print(list_artifacts(OUTPUT_ROOT / "full_fixed").to_string())
```

Output:

```text
                            quantity       value
0     single-run test R^2 (seed run)    0.635368
1       ensemble test R^2 (headline)    0.678159
2           mean of per-run test R^2    0.624378
3            std of per-run test R^2    0.022238
4               runs in the ensemble    6.000000
5           features in the ensemble  214.000000
6                     Ridge test R^2    0.351052
7                     Lasso test R^2    0.691110
8                     Lasso features   24.000000
9               Elastic Net test R^2    0.692350
10                   greedy test R^2    0.653367
11                   greedy features    3.000000
12    sparsity-stress Lasso features   23.000000
13    sparsity-stress Lasso test R^2    0.691066
14  t statistic against the baseline   -0.762488
15                 one-sided p-value    0.759913
['ndx_close__sum__126', 'ndx_close__lag_level__2', 'ndx_volume__sum__252', 'ndx_volume__lag_diff__1', 'vix_close__diff__d1']
60
0.7891
[60, 60, 60, 60, 60, 60]
{'search_max_date': '2023-12-29', 'test_start': '2024-01-02', 'n_search_dates': 2452, 'n_test_dates': 563, 'overlap_count': 0, 'payload_has_test_key': False, 'test_fingerprint_unchanged': True}
[]
False
                     entry  files
0                  prepare     12
1                  task_24      1
2                  task_25      1
3                  task_26      1
4                  task_27      1
5                  task_30      1
6             task_31.json      1
7      task_31_freeze.json      1
8                  task_32      1
9                  task_33      4
10     task_34_audit.jsonl      1
11  task_34_stage_log.json      1
```

Reading these figures correctly matters more than the figures themselves:

* **They are properties of the synthetic panel.** The ranking of the methods here says nothing about the ranking in the paper.
* **The t-test uses the manuscript's Lasso.** Task 32 tests the per-run mean against the manuscript's 0.6313, not against the Lasso measured in this run. `all_checks_passed` is therefore `False` here, and it would be meaningless to expect otherwise on synthetic data.
* **The interpretability audit compares against Table 4.** Task 33 records `reference_features_missing` as `True` because a synthetic market has no reason to select the paper's exemplar features. It is a recorded audit outcome, not a failed stage, so `reporting_failures` stays empty.
* **The sparsity stress missed its band here.** Task 27 aims for 14 to 16 Lasso features. On this panel its bracket ends at 23, so `target_attained` is `False`, and the stress Lasso is close to the unstressed one.
* **The Lasso penalty is the register's.** With `baseline_cv=False`, Lasso and Elastic Net use the alpha of the manuscript's register, and only Ridge reports a walk-forward selection in this run.

#### **Step 4.4: Reusing the preparation**

Tasks 1 to 8 are the slowest part of a short run. The `prepared` key of a full-mode result can be passed back for the same configuration, so that a new seed runs at once.

```python
# Reuse the Tasks 1 to 8 bundle of the full run for a new smoke seed.
reuse_arguments = prepare_run_arguments(
    raw_inputs,
    config,
    "fixed",
    "smoke",
    OUTPUT_ROOT / "reuse_fixed",
    seed=7,
    prepared=full_result["prepared"],
)

# Run the search stage only; Tasks 1 to 8 are not repeated.
reuse_result = execute_study(raw_inputs, config, reuse_arguments)

# Show that the preparation stage was reused.
print(stage_table(reuse_result).to_string())
```

Output:

```text
fixed/smoke: 0.9 seconds
               stage     status  seconds
0          tasks_1_8     reused     0.00
1  tasks_9_23_search  completed     0.69
2       smoke_checks  completed     0.02
```

Smoke mode does not return `prepared`, so the first bundle must come from a full-mode result or from `prepare_design_matrix_task_34(raw_inputs, config, configuration_name, output_dir=...)`.

#### **Step 4.5: The study-scale call (not executed here)**

With every default in place, the call below is the study run: Table 3's run counts and the frozen Table 2 vectors. It was **not** executed on these synthetic panels. On the real panels, before the 2026-10-31 search correction, the pipeline took 3,839 seconds for `fixed`, 5,161 seconds for `tuned` and 22,813 seconds for `endogenous`.

```python
# Not executed in the verification: this is the study-scale call.
# Walk the three configurations of Table 2.
for configuration in CONFIGURATION_NAMES:
    # Compose the call with every default of the signature.
    study_arguments = prepare_run_arguments(
        raw_inputs,
        config,
        configuration,
        "full",
        OUTPUT_ROOT / f"study_{configuration}",
    )
    # Run Tasks 1 to 33 at the budgets of Tables 2 and 3.
    study_result = execute_study(raw_inputs, config, study_arguments)
    # Report the ensemble's test-block R^2.
    print(configuration, study_result["oos_r2"])
```

Adding `enable_tuning=True` would also run the Optuna stages. This example does not exercise them.

### **Where the brief and the notebook differ**

The study brief, `config.yaml` and the notebook disagree in a few places. The example follows the notebook, because the notebook is what runs. Each point is stated here rather than resolved silently.

| Point | Brief or `config.yaml` | Notebook behaviour used here |
| --- | --- | --- |
| `original_release_date` in the FRED panel | Described as optional. | A date column in this panel makes Task 7 fail, because it cannot be converted to float64. The column is omitted. Release lags come from `release_lag_days_by_series`. |
| Factor units | The brief describes decimals. | The study's real input panel is in percent, as the library publishes it. The synthetic panel follows it. |
| `target_row_count` of the calendar | 6,260. | The NYSE has 6,600 sessions from 2000-01-03 to 2026-03-31. The difference is recorded, not raised. |
| `calendar_source` and `calendar_derivation_rule` | Named by the brief as calendar metadata. | They are not columns of `raw_input_schemas`. Task 1 checks declared columns only, so these two are ignored. They are kept as provenance. |
| Feature count | The paper reports 1,072 features. | The feature rules in `config.yaml` produce 877, both on these panels and on the real panels. |
| Smoke result | The docstring lists `prepared` in the return value. | Smoke mode does not return it. Step 4.4 therefore takes the bundle from the full-mode result. |

### **Verification record**

- **Method.** A script extracted every `python` fence from this file and executed them in order, in one namespace, after the notebook's import cell and its Task 1 to Task 36 cells. The *Study Inputs* and *Study Execution* cells were left out. The namespace was built by executing the notebook's cell sources in one Python process, not through a Jupyter kernel. Nothing was imported from a `.py` module.
- **Executed.** Every fence except the one in Step 4.5, with exit code 0. The `text` blocks are the standard output of that run. It took about 50 minutes: 366.8 seconds for the three smoke calls, 2,633.9 seconds for the full call and 0.9 seconds for the reuse call.
- **Stages.** All 13 smoke stages completed in each configuration, and all 20 full-mode stages completed. `reporting_failures` is empty.
- **Barrier.** Searched and test dates do not overlap: 2,452 search dates to 2023-12-29, 563 test dates from 2024-01-02, and the test fingerprint is unchanged.
- **Determinism.** A separate run of the same smoke and full calls, with the Tasks 1 to 8 bundle prepared once, gave the same figures: ensemble $R^2$ 0.678159 and seed-run $R^2$ 0.635368.
- **Faulty first run.** An earlier run of this check reported a Task 32 failure. The cause was the checking script, which assigned a variable named `stats` in the shared namespace and so hid the notebook's `scipy.stats`. No example cell binds that name; the script was corrected and everything was run again. The failure does show how `reporting_failures` behaves. It also carries a warning: do not rebind a notebook name such as `stats` in an appended cell.
- **Synthetic market.** NASDAQ-100 daily return standard deviation 1.64% (real panel 1.71%), excess kurtosis 8.8, correlation with the VIX change -0.64 (real -0.665), correlation with `MKT` 0.89 (real 0.88). These are checks on the generator, not study results.
- **Code standard.** `flake8` at 79 columns reports only the undefined name `run_task_34`, which is expected, because that name comes from the notebook. A script confirmed that every statement in every function has a preceding comment, and that every function has type hints and a docstring with Purpose, Inputs, Processes, Outputs and Raises.
- **Not executed.** The Step 4.5 study-scale call on these panels, the Optuna stages (`enable_tuning=True`), `baseline_cv=True`, and `run_task_35` and `run_task_36`. Rendering was checked locally, not on GitHub itself: fences, tables and anchors by script, and the 45 math expressions of the README with MathJax 3, with no errors. No figure in this document is a result of the study.

### **Summary of the Execution Flow**

1. **Notebook namespace.** Execute the import cell and the Task 1 to Task 36 cells of `feature_selection_via_monte_carlo_tree_search_in_high_dimensional_financial_regression_draft.ipynb`, and append the example cells below them. Nothing is imported from a `.py` folder.
2. **Raw panels.** `generate_raw_inputs` uses a seeded `PCG64` process, in Faker's place, to produce the four panels from one latent market: `YAHOO_FINANCE_RAW`, `FRED_MACRO_RAW`, `FACTOR_LIBRARY_RAW` and `TRADING_CALENDAR`.
3. **Configuration.** `load_study_configuration` reads `config.yaml` into `config` and requires the blocks the pipeline cannot run without.
4. **Arguments.** `prepare_run_arguments` validates the panels, the configuration name, the mode and the keywords, merges them with the signature's defaults and creates the output directory.
5. **Interface call.** `execute_study` calls `run_task_34(raw_inputs, config, configuration_name, mode, **keywords)`, times it and reports any reporting failure.
6. **Smoke.** `mode="smoke"` runs Tasks 1 to 23 at unit budgets for each configuration, and checks completion, policy convergence, the information barrier and the lexicographic ordering. The test block is never evaluated.
7. **Full.** `mode="full"` adds the baselines (Tasks 25 to 27), the ensemble (Task 30), the out-of-sample evaluation (Task 31), inference (Task 32) and interpretability (Task 33). Tuning (Tasks 28 and 29) runs only with `enable_tuning=True`.
8. **Reading.** `stage_table`, `summarize_run` and `list_artifacts` tabulate the stage log, the headline figures and the files written. `prepared` lets a later call skip Tasks 1 to 8.
9. **Scale and honesty.** The defaults give the study-scale run, which is not executed here. Task 35 and Task 36 run separately. Synthetic panels verify wiring and the input contract, never the paper's results.
