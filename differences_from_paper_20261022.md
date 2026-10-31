# Differences from the paper

Written 2026-10-22, after all three configurations had been run in full mode. This supersedes the
"Not run" and tuned sections of `paper_vs_pipeline_trace_20261022.md`. That file still holds the
step-by-step trace for the fixed run: the data audit, the call trace, the baseline λ path and the
Table 4 defects. Nothing here called a paid service.

**Availability.** The raw data, the `e2e_runs/` run folders, the `tools/` scripts and the trace file named in this document are not in the public repository, for privacy and data-licensing reasons. They are cited as the provenance of the figures, so the figures cannot be re-derived from the repository alone.

**Update of 2026-11-01.** §1 now reports the corrected runs of 2026-10-31. The runs dated 20261022 were all made with one policy table shared across nesting levels, whereas Algorithm 1 passes the policy by value. `gnrpa_nested_search_task_22` now gives each child level its own copy. §2 was not re-run after the correction, so its figures are from the shared-policy module.

**Runs.** All runs use `data_raw_20261022_fred` and call the module's `run_task_34` through `tools/run_end_to_end.py`.

**Module versions.** The tuned, endogenous and two ablation runs used the module as it stands now (`fe23b8a0…`). The only difference is the header comment that records the notebook SHA, `f7756e08…`. The fixed run `final_e2e_20261022` used `1fdf6eb8…` for Tasks 1–33, and its Task 33 was then re-run with the corrected audit. The current module differs from `1fdf6eb8…` only in two places:
- the Task 33 audit fix;
- the ablation switch in Task 6, which does nothing when the flag is off.

Each run is in `e2e_runs/`:

| Run | Configuration | p | Runs |
|---|---|---|---|
| `final_e2e_20261022` | fixed | 877 | 18 |
| `tuned18_20261022` | tuned | 877 | 18 |
| `endogenous_full_20261022` | endogenous | 877 | 12 |
| `ablate_fixed_20261022` | fixed, [U] diff columns withheld | 707 | 18 |
| `ablate_tuned18_20261022` | tuned, [U] diff columns withheld | 707 | 18 |
| `policy_copy_fixed_20261031` | fixed, corrected search | 877 | 18 |
| `policy_copy_tuned18_20261031` | tuned, corrected search | 877 | 18 |
| `policy_copy_endogenous_20261031` | endogenous, corrected search | 877 | 12 |

Summary: `e2e_runs/summary_all_20261022.json`. Task 36 tables: `e2e_runs/task36_20261022/`.

Corrected runs: module `a2735065…`, the same seeds as the earlier runs, summary `e2e_runs/summary_policy_copy_20261101.json`, Task 36 tables `e2e_runs/task36_20261101/`. Their searches ran first, in parallel, and were cached in `e2e_runs/policy_copy_cache_20261031/`. The real driver then assembled each run from that cache and computed no search live. Their run times and ensemble build times are therefore not comparable with the earlier runs. Together the three took about six hours on seven shared workers.

Labels as in the trace:
- **DATA:** the raw inputs.
- **UNIVERSE:** the reconstructed feature set.
- **CONFIG:** a like-for-like mismatch.
- **SEARCH:** the selection behaviour.
- **CONVENTION:** a choice the paper does not state.
- **NOT RUN:** a step that was not executed.

---

## 1. Headline results (Tables 1 and 3)

| | Fixed (K = 60) | Tuned (K = 15) | Endogenous |
|---|---|---|---|
| Per-run OOS R², paper | 0.643 ± 0.030 | 0.665 ± 0.039 | 0.668 ± 0.009 |
| Per-run OOS R², pipeline | 0.480 ± 0.337 | 0.637 ± 0.046 | 0.614 ± 0.074 |
| Per-run OOS R², pipeline, shared policy | 0.584 ± 0.076 | 0.590 ± 0.053 | 0.610 ± 0.065 |
| Per-run median, pipeline | 0.592 | 0.653 | 0.634 |
| Per-run range, paper | [0.575, 0.686] | [0.519, 0.698] | [0.650, 0.682] |
| Per-run range, pipeline | [−0.786, 0.676] | [0.544, 0.687] | [0.461, 0.695] |
| Per-run range, pipeline, shared policy | [0.401, 0.675] | [0.521, 0.683] | [0.486, 0.679] |
| **Ensemble OOS R², paper** | **0.6855** | **0.6908** | **0.6961** |
| **Ensemble OOS R², pipeline** | **0.6614** | **0.6789** | **0.6793** |
| Ensemble OOS R², pipeline, shared policy | 0.6823 | 0.6306 | 0.6723 |
| Validation R², paper | 0.702 | 0.723 | 0.755 |
| Validation R², pipeline | 0.769 | 0.752 | 0.789 |
| Validation R², pipeline, shared policy | 0.769 | 0.712 | 0.798 |
| Model size, paper | 60 | 15 | 18.6 |
| Model size, pipeline | 60 | 15 | 29.5 (26–34) |
| Model size, pipeline, shared policy | 60 | 15 | 32.9 (23–36) |
| Global features, paper | 341 | 74 | 59 |
| Global features, pipeline | 445 | 116 | 183 |
| Global features, pipeline, shared policy | 429 | 144 | 198 |

Effect of the correction, paired by seed on per-run OOS R²:

| Configuration | Mean change | Seeds better | Wilcoxon p |
|---|---|---|---|
| Tuned | +0.047 | 15 of 18 | 0.008 |
| Endogenous | +0.005 | 4 of 12 | 0.79 |
| Fixed | −0.104 | 7 of 18 | 0.14 |

What reproduces:
- The ensemble ordering, endogenous > tuned > fixed, although endogenous and tuned differ by only 0.0004.
- All three ensembles within 0.024: tuned 0.012 short, endogenous 0.017 short, fixed 0.024 short.
- All 30 Table 2 hyperparameters, used exactly as published.
- The rollout budgets.

What does not reproduce:
- **Per-run accuracy:** 0.028 below the paper for tuned, 0.054 for endogenous and 0.163 for fixed.
- **Per-run ordering:** tuned > endogenous > fixed here, and endogenous > tuned > fixed in the paper.
- **Per-run dispersion:** tuned is close, 0.046 against 0.039. Endogenous is 0.074 against 0.009, and fixed is 0.337 against 0.030.
- **Agreement across runs:** the union of selected features is 1.3–3× the paper's.
- **Endogenous model size:** 29.5 features instead of 18.6.

**Fixed run 3.** Seed 45 has OOS −0.786 with a validation reward of 0.750, in line with the other runs. Recomputing its predictions from the recorded support and coefficients gives the same R², so the figure is real. Its largest test-window contributions come from `ndx_close__EMA__21`, `ndx_close__lag_level__3` and `ndx_close__x__rate_long`. The first two average about 21,000 in the test window against about 7,000 in training, so the unshrunk coefficients extrapolate. Seed 51, at 0.217, shows the same pattern with `inflation_cpi__x__rate_long` and `rate_long__x__vix_close`. Nothing in the search penalises non-stationary level columns.

**Cause: the shared policy, then SEARCH on a different UNIVERSE.**
- The shared policy was a real defect, and it explained most of the tuned shortfall. The earlier 0.003 match on the fixed ensemble came from the defective search, and it did not survive the correction.
- Validation R² stays above the paper's in all three configurations, while OOS stays below it. The search fits the 2021-10 to 2023-12 validation regime better than the paper's did, and that extra fit does not carry over to the test window. Details are in the trace, §3.
- The universe gives the search more ways to over-fit. The [U] reconstructed columns such as `vix_close__diff_mean__5` (18 of 18 runs in fixed and tuned, 12 of 12 in endogenous) and `usd_index_close__diff_mean__5` (12 of 12 in endogenous) let each run reach a similar validation fit through different column sets. That is why the union of selected features is large.
- Endogenous stopping also stops later than the paper's. Its validation reward keeps improving on this universe, so the stop rule (β_stop = 0.909, patience 63) is reached at about 30 features.

## 2. Is the feature universe the cause? The ablation test

The paper does not disclose 403 of its 1,072 features. The register reconstructs 221 of them, including 170 [U] `diff_mean`/`diff_std` columns. To test whether these columns cause the gap, I removed all 170 (`--exclude-extension-families=diff_mean,diff_std`, leaving p = 707) and re-ran fixed and tuned.

| | Fixed, 877 | Fixed, 707 | Tuned, 877 | Tuned, 707 | Paper |
|---|---|---|---|---|---|
| Per-run OOS mean | 0.584 | 0.416 | 0.590 | 0.316 | 0.643 / 0.665 |
| Per-run OOS median | 0.606 | 0.555 | 0.570 | 0.378 | — |
| Runs with OOS below 0 | 0 | 3 (worst −0.56) | 0 | 2 (worst −0.49) | 0 |
| Ensemble OOS | 0.682 | 0.635 | 0.631 | 0.482 | 0.686 / 0.691 |
| Lasso baseline | 0.581 | 0.085 | same | same | 0.631 |
| Greedy baseline | 0.552 | −0.180 | same | same | 0.249 |
| Ridge baseline | 0.228 | 0.260 | same | same | 0.525 |
| Exemplar Table 4 features matched | 8 / 15 | **10 / 15** | 5 / 15 | 4 / 15 | 15 / 15 |

How often the paper's Table 4 features were picked across the 18 fixed runs:

| Feature | 877 columns | 707 columns |
|---|---|---|
| VIX_{t−5} | 0 | **16** |
| ΔVIX_t | 14 | **18** |
| ΔVIX_{t−2} | 11 | 14 |
| ΔHY_{t−2} | 16 | **18** |
| CMA_t^{(5d mean)} | 17 | **18** |
| VIX_{t−1}^{(5d mean)} | 0 | 0 |

**Finding:**
- **The selection does change.** Removing the reconstructed diff columns brings back the paper's VIX-spread and ΔHY lag mechanisms. They had been crowded out by single columns that encode the same 5-day changes.
- **Accuracy gets much worse.** Removing them also destroys accuracy and stability: Lasso falls from 0.58 to 0.09, and some MCTS runs fit the test window worse than its mean.
- **What this says about the paper's universe.** The paper reports Lasso 0.631 on its own universe, so that universe must contain smoothed-change information that our disclosed 707 features lack. Our [U] columns supply it, but in a form that displaces the paper's named lags.
- **Conclusion.** The feature universe is the leading explanation of the remaining gap, but neither of our universes is the paper's, and the manuscript does not allow it to be rebuilt. Closing the gap needs the authors' feature list or code. The request is drafted in `author_data_request_draft_20261022.md` and has not been sent.

## 3. Baselines (Table 1)

| | Paper | Pipeline (877) | Cause |
|---|---|---|---|
| Ridge | 0.525 | 0.228 | **UNIVERSE.** Even the oracle α reaches only 0.393 on this universe. |
| Lasso | 0.631, 60 features | 0.581, 11 features | **UNIVERSE.** One dominant [U] column absorbs the signal, so the model is much sparser. |
| Elastic Net | 0.624, 55 features | 0.581, 11 features | As for Lasso. |
| Greedy L0 | 0.249, k = 5 | 0.552, k = 3 | **UNIVERSE.** It picks `vix_close__diff_mean__5` first; on the 707 columns it gives −0.180. |

Each baseline swings by up to 0.73 R² depending on which undisclosed families are included (§2). This confirms the trace's conclusion that every baseline difference is a universe difference first.

## 4. Table 4 and the §4.4 mechanisms

- **The data carries the paper's model.**
  - The paper's own 15 features, fitted with Eq. (6) on our training block in per-SD units (Task 36 `table4_comparison.csv`), give coefficients close to the paper's: 14 of 15 signs agree, and only CMA_{t−1}^{(5d mean)} differs.
  - VIX_t² is the largest miss, 0.0002 against 0.0109.
  - The same subset gives OOS 0.674, or 0.689 with an intercept, against the paper's 0.698 (trace, step 1.5).
- **The search reaches the signal through other columns.** Matched features in the best run of each configuration:

  | Configuration | Best-run OOS | Table 4 features matched |
  |---|---|---|
  | Tuned | 0.687 | 4 / 15, all 4 with the paper's sign |
  | Fixed | 0.676 | 9 / 15, 8 with the paper's sign |
  | Endogenous | 0.695 | 6 / 15, 5 with the paper's sign |

- **Two audit defects were fixed earlier on 2026-10-22:** name matching and the units of comparison (trace, step 4.7).
- **Tolerance decision (delegated by the user).** The 10% relative tolerance on Table 4 coefficients is informational, not a pass/fail gate. The paper's own subset fitted on our data passes it for only 1 of 15 coefficients, so the gate would fail even a faithful replica. This is recorded as a comment in the register; the executable configuration is unchanged.

## 5. Task 36 verdict

`overall_pass = False`, for reasons that are documented and expected. These are the 2026-11-01 results on the corrected runs:

- **Leakage passes.**
  - The search and Optuna domains are disjoint from the 2024-01-02 to 2026-03-31 test window.
  - The information-barrier re-run matches the recorded audit: 877 features, maximum univariate correlation with the target 0.10.
- **Table 1 fails tolerance:**
  - all three MCTS per-run means;
  - all three ensembles;
  - every baseline;
  - endogenous model size.

  The "features" values for fixed and tuned are within tolerance.
- **Table 2 fails only on `best_cv_r2`.** Optuna (Tasks 28–29) was not re-run, by design, because the runs use the published parameters. Every published hyperparameter matches exactly.
- **Table 3 build times have no meaning for the corrected runs,** because the searches came from the cache. The earlier full runs took 9–16× the paper's times, which reflects hardware and Python, not a result difference.

## 6. Smaller differences (unchanged from the trace)

| Item | Paper | Pipeline | Label | Effect |
|---|---|---|---|---|
| T | 6,260 | 6,600 (the exact NYSE count for the span) | DATA, paper unexplained | None; recorded, not raised. |
| p | 1,072 | 877 | UNIVERSE | §2. |
| Test days | 562 | 563 | DATA | ±0.0003 on ensemble R². |
| OHLC indicators | presumably OHLC | close and volume only | DATA MODEL | 13 indicator families use close-only proxies. Small, because NDX features are weak behind the h = 5 barrier. |
| Intercept and standardisation | not stated; coefficients look z-scored | no intercept, raw, as in Eq. (6) | CONVENTION | About +0.015 on the paper's own subset. Not re-searched. |
| Rates and spreads | kept at date t | `rate_short`, `rate_long`, `spread_2s10s` and `spread_hy` lagged one session | CONVENTION | On the paper's 15 features, removing the lag raises OOS from 0.674 to 0.692, or 0.689 to 0.709 with an intercept, and sign matches go from 14 to 15 of 15. Not re-searched. |
| Tuning | Optuna, 2,000 trials | Table 2 values used | NOT RUN | No best-CV R² counterpart. |
| Task 35 sensitivity | reported | not run | NOT RUN | — |

## 7. What would close the gap

1. **The authors' 1,072 feature names or code.** This is the decisive item: every remaining difference in §1–§3 runs through the universe.
2. **Their Eq. (6) intercept and standardisation convention**, then a re-search under it.
3. **Their HY data vendor and pull date, and the seeds or per-run results.** These allow a run-by-run comparison.

All three are in the drafted request, `author_data_request_draft_20261022.md`. It has not been sent.
