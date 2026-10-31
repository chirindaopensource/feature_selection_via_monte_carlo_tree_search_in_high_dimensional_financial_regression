# Paper Summary

Below is a granular, section-by-section summary of Mouangue and Cazenave, *“$L_0$ Feature Selection via Monte Carlo Tree Search in High-Dimensional Financial Regression”* (ACM ICAIF ’26).

## 1. Bibliographic identification and high-level thesis

- **Title:** $L_0$ Feature Selection via Monte Carlo Tree Search in High-Dimensional Financial Regression.
- **Authors:** Cameron Mouangue and Tristan Cazenave, LAMSADE, Université Paris Dauphine-PSL, CNRS.
- **Venue:** ACM International Conference on AI in Finance (ICAIF ’26), Milan, Italy, November 14–17, 2026.
- **Keywords:** Monte Carlo Tree Search, Feature Selection, Quantitative Finance, Reinforcement Learning, Combinatorial Optimization.
- **Core thesis:** Sparse feature selection in high-dimensional financial regression can be formulated as a sequential decision problem and solved approximately by Monte Carlo Tree Search with Generalized Nested Rollout Policy Adaptation (GNRPA), using validation performance rather than training error, and a lexicographic reward that controls subset size and multicollinearity.
- **Headline empirical claim:** On a contemporaneous explanatory regression task for weekly NASDAQ-100 returns using 1,072 engineered features, the method achieves an out-of-sample $R^2$ of 0.6961 versus 0.6313 for an optimized Lasso baseline.

## 2. Motivation and problem statement

1. **High-dimensional financial regression.** Modern financial datasets contain hundreds or thousands of engineered predictors from prices, macro variables, technical signals, and factor data. Many are highly correlated, so feature selection is needed for interpretability, variance reduction, and overfitting control.
2. **Lasso limitations.** Lasso ($L_1$) produces sparse solutions but shrinks coefficients. In financial settings, economically meaningful signals such as spreads, distributed lags, and momentum measures often require combinations of correlated variables. Lasso may discard relevant variables or over-shrink coefficients, attenuating explanatory power and introducing bias.
3. **Exact sparse regression.** $L_0$-constrained subset selection avoids shrinkage but is NP-hard. Existing exact or approximate solvers typically minimize penalized training error, which can overfit noisy financial data; the paper instead targets out-of-sample explanatory performance.
4. **Methodological gap.** The paper positions itself between convex regularization, which is fast but shrinkage-biased, and exact combinatorial methods, which are exact but often in-sample-focused and computationally prohibitive in high dimensions.
5. **Application focus.** The empirical task is deliberately framed as a contemporaneous explanatory regression—explaining, not forecasting, the weekly NASDAQ-100 return—so that the value of subset selection under multicollinearity is isolated from the separate problem of predictive return forecasting.

## 3. Main contributions

The paper lists three contributions:

1. **Sequential decision formulation.** Sparse feature selection is formulated as a sequential decision problem, and GNRPA is applied to approximate $L_0$ optimization using validation-guided tree search.
2. **Lexicographic search framework.** The framework supports a lexicographic objective including sparsity, multicollinearity, and endogenous stopping.
3. **High-dimensional empirical evaluation.** The method is evaluated on a high-dimensional financial regression problem and shown to improve out-of-sample explanatory performance relative to optimized Lasso and related baselines.

## 4. Dataset and target construction

1. **Time span.** Daily financial data from January 2000 through March 2026, with $T = 6{,}260$ observations.
2. **Target variable.** The response is the contemporaneous weekly return of the NASDAQ-100 index, defined as the five-day rolling sum of daily returns:

   $$ Y_t = \sum_{k=0}^{4} r_{t-k}, \qquad r_t = \frac{P_t - P_{t-1}}{P_{t-1}}. $$

3. **Predictor sources.**
   - **Market data:** price, trading volume, and volatility indices (NASDAQ-100, CBOE VIX, USD Index) via Yahoo Finance.
   - **Macroeconomic data:** interest rates, yield spreads, inflation, labor metrics via FRED.
   - **Factor data:** Fama–French five factors and momentum.
4. **Feature engineering.** The final design matrix contains $p = 1{,}072$ engineered features, including technical indicators, interaction terms, and rolling statistics computed over lookback windows $W \in \lbrace 5, 21, 63, 126, 252 \rbrace$ trading days.
5. **Leakage control.** Technical indicators derived from the NASDAQ-100 with lookback periods shorter than five trading days were excluded to avoid information leakage.
6. **Contemporaneous exogenous variables.** Macroeconomic indicators, volatility measures, yield spreads, and factor returns were retained at time $t$ because they are treated as contemporaneously observed explanatory variables.
7. **Important interpretive caveat.** The paper explicitly notes that this contemporaneous alignment is deliberate rather than an oversight. $Y_t$ aggregates returns over a five-day window, while several exogenous predictors—e.g., $\Delta\text{VIX}_t$, $\Delta\text{HY Spread}_t$—are single-day changes. The model therefore exploits same-day and near-term co-movement between volatility/credit markets and equity returns, consistent with the leverage effect, rather than a genuine lead–lag forecasting relationship. The reported $R^2$ values are therefore explanatory, not predictive, in interpretation.

## 5. Train/validation/test splitting and validation logic

1. **Chronological split.** The dataset was partitioned chronologically at December 31, 2023.
2. **Out-of-sample test set.** The final 562 trading days—January 2024 to March 2026—were held out entirely as an out-of-sample test set.
3. **Training and validation windows.**
   - The search environment used a continuous historical block comprising a training window immediately followed by a 562-day internal validation window, positioned immediately before the 2024–2026 test period.
   - For the fixed- and tuned-cardinality configurations, the training window is 1,890 days, selected by cross-validating the Lasso baseline and reused unchanged.
   - For the endogenous-stopping configuration, the training window itself was included in the Optuna search space; a shorter 1,000-day window was found best jointly with other hyperparameters.
4. **Regime-similarity trade-off.** The paper states that validating on the period immediately preceding the test period is deliberate: because market regimes shift, the search is validated against the period most similar in time—and presumably regime—to the eventual test period. This trades some protection against regime change for a validation signal more representative of the final evaluation conditions.
5. **Hyperparameter tuning isolation.** Hyperparameter selection never accesses the post-2024 hold-out period. Optuna’s objective is computed by averaging validation $R^2$ across three non-overlapping walk-forward folds constructed entirely from pre-2024 data, with zero temporal overlap between folds.
6. **Optimizer’s curse.** The best-trial cross-validation objective reaches up to 0.761, exceeding the true test $R^2$ range of 0.643–0.668. The paper explains this as the expected signature of the optimizer’s curse when hyperparameters are chosen as the argmax over many noisy trials; it does not imply leakage given the temporal separation of the test period.
7. **Out-of-sample $R^2$ definition.** All out-of-sample $R^2$ values are computed using the standard coefficient of determination, with residual variance normalized by the variance of the test sample itself.

## 6. Problem formulation as cardinality-constrained sparse regression

Sparse linear regression seeks a parameter vector $\theta \in \mathbb{R}^p$ minimizing residual sum of squares subject to a cardinality constraint:

$$ \min_{\theta \in \mathbb{R}^p} \frac{1}{2N_{\text{train}}} \lVert Y_{\text{train}} - X_{\text{train},S}\theta \rVert_2^2, \quad \text{s.t. } \lVert \theta \rVert_0 \le K, $$

where $X_{\text{train},S}$ is the design matrix restricted to the selected feature subset $S$. Candidate subsets are evaluated on a validation set rather than the training set.

The method supports two search modes:

1. **Fixed-cardinality search:** limits subset size to a predefined $K$.
2. **Adaptive search:** uses an endogenous stopping mechanism and a lexicographic reward to determine subset size automatically.

Feature selection is formulated as an episodic Markov Decision Process (MDP) defined by $(\mathcal{S}, \mathcal{A}, \mathcal{R})$.

## 7. MDP formulation

1. **State.** Let $\mathcal{F} = \lbrace f_1, \dots, f_p \rbrace$ denote candidate features. Although the selected subset is unordered, the search constructs it sequentially. The state is

   $$ s_t = (f_{\text{last}}, S_t), $$

   where $S_t \subset \mathcal{F}$ is the current subset and $f_{\text{last}}$ is the most recently selected feature. Tracking $f_{\text{last}}$ allows the $p \times p$ policy matrix to capture pairwise feature synergies, such as variables forming spread signals. The initial state is $s_0 = (\emptyset, \emptyset)$.
2. **Action.** At each state, the agent may select any feature not already in the current subset or terminate the search:

   $$ \mathcal{A}(s_t) = (\mathcal{F} \setminus S_t) \cup \lbrace a_{\text{stop}} \rbrace. $$

3. **Reward.** Rewards are evaluated only at terminal states. For a feature subset $S$, regression coefficients are estimated by ordinary least squares using a ridge-stabilized matrix inversion:

   $$ \hat{\theta}_S = \left(X_{\text{train},S}^\top X_{\text{train},S} + \lambda I\right)^{-1} X_{\text{train},S}^\top Y_{\text{train}}, $$

   where $\lambda = 10^{-5}$ provides numerical stability. The resulting model is evaluated on a separate validation set, and validation performance defines the reward used during search.

## 8. Search heuristics: mRMR bias and redundancy control

1. **Search-space size.** The complete search space contains

   $$ 2^{1072} \approx 10^{322} $$

   possible subsets. Even with a maximum capacity of $K = 60$, the valid subset space exceeds $10^{99}$.
2. **mRMR heuristic.** The method incorporates a Minimum Redundancy Maximum Relevance (mRMR) heuristic into the GNRPA bias term $\beta$.
3. **Relevance.** For candidate feature $f_i$, relevance is the absolute correlation with the target:

   $$ \text{Rel}(f_i) = |\rho(f_i, Y)|. $$

4. **Redundancy.** To penalize localized correlation clusters rather than average redundancy across the whole subset, the paper uses a top-$k$ RMS metric. Let

   $$ \mathcal{C}_{i,S} = \lbrace |\rho(f_i, f_j)| : f_j \in S \rbrace $$

   be the absolute correlations between candidate $f_i$ and already selected features. Denote $c_{(m)}$ as the $m$-th largest order statistic. Then

   $$ \text{Red}_{\text{top-}k\text{-rms}}(f_i, S) = \left( \frac{1}{\min(k, |S|)} \sum_{m=1}^{\min(k, |S|)} c_{(m)}^2 \right)^{1/2}. $$

5. **Redundancy variants.** The search treats the redundancy metric as a categorical hyperparameter with three settings:
   - **Mean:** averages $c_{(m)}^2$ over all of $S$.
   - **RMS:** root-mean-square over the full subset.
   - **Top-$k$-RMS:** restricts the sum to the $k$ most correlated features.
6. **Configuration-specific choices.** The fixed- and tuned-cardinality configurations use the mean and RMS variants, respectively. The endogenous-stopping configuration uses top-$k$-RMS with $k = 3$.
7. **Heuristic score.** The score combines relevance and redundancy using trade-off parameter $\gamma \in [0,1]$, scaled by magnitude parameter $\beta_{\text{mag}}$:

   $$ \beta(f_i \mid S) = \beta_{\text{mag}} \left[ \gamma \cdot \text{Rel}(f_i) - (1-\gamma) \cdot \text{Red}_{\text{top-}k\text{-rms}}(f_i, S) \right]. $$

## 9. Adaptive search and endogenous stopping

1. **Problem with fixed $K$.** Standard $L_0$ implementations require pre-specifying target cardinality $K = |S|$. The paper addresses this by allowing the search policy to determine stopping depth automatically, making sparsity endogenous to the search.
2. **Stop-action bias.** To balance feature addition against termination during Gibbs sampling, the heuristic bias for $a_{\text{stop}}$ is piecewise:

   $$ \beta(a_{\text{stop}} \mid S) = \begin{cases} -\infty & \text{if } |S| < K_{\min}, \\ \beta_{\text{mag}} \gamma \beta_{\text{stop}} \left( \frac{|S| - K_{\min}}{\max(1, K_{\max} - |S|)} \right)^{3/2} & \text{if } |S| \ge K_{\min}. \end{cases} $$

   - $\beta_{\text{stop}}$ scales the stopping bias.
   - The exponent $3/2$ is an empirically chosen constant.
   - The function increases bias toward stopping as subset size approaches $K_{\max}$, while allowing exploration near $K_{\min}$.
3. **Dual learning rates.** The Montparnasse algorithm introduces a dual-learning-rate mechanism:
   - At the base level, a conservative learning rate $\alpha_{L1}$ prevents stochastic single-rollout evaluations from prematurely skewing the policy.
   - At higher nesting levels, where trajectories are filtered using repeated simulations, a larger learning rate $\alpha_{\text{high}}$ reinforces proven high-quality sequences.
   - The fixed- and tuned-cardinality configurations are the special case $\alpha_{L1} = \alpha_{\text{high}}$, i.e., a single global learning rate.
4. **Level-1 patience.** The level-1 loop is terminated early once the best lexicographic score fails to improve for $P$ consecutive iterations. $P$ is the level-1 patience hyperparameter. Consequently, the root-level iteration count $I_1$ is reported as variable rather than fixed.

## 10. Lexicographic reward and MaxVIF

1. **Scalar vs. lexicographic reward.** To improve model stability, the terminal scalar reward is replaced by a lexicographic tuple:

   $$ \mathcal{R}(S) = \left( \lfloor R^2_{\text{val}}(S) \rceil_3, -|S|, -\text{MaxVIF}(S) \right), $$

   where $\lfloor R^2_{\text{val}} \rceil_3$ denotes validation $R^2$ rounded to three decimal places.
2. **Tie-breaking.** Ties are broken first by prioritizing smaller subsets, then by minimizing maximum Variance Inflation Factor (MaxVIF).
3. **MaxVIF definition.** MaxVIF is computed from the empirical correlation matrix $C_S \in \mathbb{R}^{|S| \times |S|}$ of variables in subset $S$:

   $$ \text{MaxVIF}(S) = \max_{1 \le j \le |S|} \left(C_S^{-1}\right)_{jj}. $$

4. **Purpose.** By evaluating VIF as a tie-breaker inside the search tree, the algorithm penalizes highly collinear subsets that convex penalties might fail to separate.

## 11. GNRPA policy and adaptation equations

Given state $s$ and available actions $\mathcal{A}(s)$, the probability of selecting action $a_k$ is

$$ \pi(a_k \mid s) = \frac{ \exp\left(\frac{w_{s,a_k}}{\tau} + \beta_{s,a_k}\right) }{ \sum_{a_j \in \mathcal{A}(s)} \exp\left(\frac{w_{s,a_j}}{\tau} + \beta_{s,a_j}\right) }, $$

where $W$ denotes policy weights, $\tau$ is temperature, and $\beta$ represents heuristic domain knowledge.

During policy adaptation, weights are updated as

$$ w_{s,a_j} \leftarrow w_{s,a_j} - \frac{\alpha}{\tau} \left( \pi(a_j \mid s) - \mathbb{I}_{\lbrace a_j = a^* \rbrace} \right), $$

where $\alpha$ is the learning rate and $a^*$ denotes the action selected in the best rollout sequence.

## 12. Algorithmic procedure

Algorithm 1, “GNRPA with Endogenous Stopping,” proceeds as follows:

1. **Inputs:** level $L$, state $s$, policy $W$, max iterations $I_L$, patience $P$.
2. **Global hyperparameters:** learning rates $\alpha_{L1}, \alpha_{\text{high}}$, temperature $\tau$.
3. **Terminal playout phase:** if $L = 0$, call Rollout$(s, W)$.
4. **Learning-rate selection:** $\alpha \leftarrow \alpha_{L1}$ if $L = 1$, else $\alpha_{\text{high}}$.
5. **Initialization:** best score $(-\infty, -\infty, -\infty)$, best sequence $\emptyset$, no-improvement counter $0$, iteration counter $n \leftarrow 0$.
6. **Main loop:**
   - Call GNRPA$(L-1, s, W)$ to obtain $seq, score$.
   - If $score > best\_score$ lexicographically, update best score, best sequence, and reset no-improvement counter.
   - Else increment no-improvement counter.
   - Increment $n$.
   - If $L = 1$ and no-improvement $\ge P$, break for early stopping.
   - If $L > 1$ and $n \ge I_L$, break for fixed iterations.
   - Update weights $W$ toward $best\_seq$ via the adaptation equation using $\alpha$.
7. **Return:** $best\_seq, best\_score$.
8. **Rollout:** while $a_{\text{stop}}$ not selected and $|s| < K_{\max}$:
   - Compute $\beta$ via heuristics.
   - Sample action $a \sim \pi(a \mid s)$ using $W, \beta, \tau$.
   - Update $s \leftarrow s \cup \lbrace a \rbrace$, updating both $f_{\text{last}}$ and $S_t$.

   Return $s$ and EvaluateState$(s, X_{\text{train}}, Y_{\text{train}})$.

## 13. Experimental design and baselines

The paper evaluates three objectives:

1. Compare GNRPA-based $L_0$ selection against convex penalties to highlight shrinkage limitations.
2. Assess trade-offs between fixed, tuned, and endogenous-stopping configurations under varying search budgets.
3. Examine subset interpretability and ensemble stability.

Baselines:

- **Ridge ($L_2$):** stabilizes coefficient estimation but does not select features; retains all 1,072 predictors; OOS $R^2 = 0.5250$.
- **Lasso ($L_1$):** cross-validation selected a 1,890-day lookback window and $\lambda = 0.0745$, yielding OOS $R^2 = 0.6313$ with 60 selected features. Increasing $\lambda$ to obtain a GNRPA-comparable subset (14–16 features) reduced OOS $R^2$ to 0.39–0.43, compared with 0.6648 for GNRPA.
- **Elastic Net:** evaluated over 180 hyperparameter combinations. Cross-validation selected an $L_1$ ratio of 0.990 (the maximum grid bound prior to pure Lasso), giving 55 features and OOS $R^2 = 0.6243$. This suggests the $L_2$ penalty’s tendency to distribute weights across correlated variables may be suboptimal for this dataset, possibly because it restricts the large, opposing coefficients used to capture spread signals.
- **Forward stepwise selection (greedy $L_0$):** using the same 3-fold cross-validation procedure, stopped at $K = 5$ features, giving OOS $R^2 = 0.2489$.

## 14. Hyperparameter optimization and ensembles

1. **Optuna tuning.** Hyperparameters were tuned with Optuna. Optimal values are reported in Table 2.
2. **Budgets.** Optuna budgets were 15, 50, and 20 trials for fixed-, tuned-, and endogenous-stopping configurations, respectively. Total hyperparameter-search wall-clock times were 0.41, 4.08, and 11.27 hours on a 16-core workstation, in addition to ensemble-building time.
3. **Optimal hyperparameters (Table 2):**
   - **Fixed cardinality:** training window 1,890 days (from Lasso CV); target $K = 60$; redundancy mean; $\gamma = 0.666$; $\beta_{\text{mag}} = 24.82$; $\tau = 1.520$; learning rate 1.514 (global); tree iterations $84 \times 3 \times 6$; best CV validation $R^2 = 0.7089$.
   - **Tuned cardinality:** training window 1,890 days (from Lasso CV); target $K = 15$; redundancy RMS; $\gamma = 0.772$; $\beta_{\text{mag}} = 14.97$; $\tau = 0.868$; learning rate 2.003 (global); tree iterations $18 \times 28 \times 18$; best CV validation $R^2 = 0.7491$.
   - **Endogenous stopping:** training window 1,000 days (GNRPA-tuned); target endogenous (mean 18.6); redundancy top-3-RMS; $\gamma = 0.811$; $\beta_{\text{mag}} = 13.72$; $\beta_{\text{stop}} = 0.909$; $\tau = 1.515$; learning rates $\alpha_{L1} = 0.177$, $\alpha_{\text{high}} = 2.212$; tree iterations $\text{var.} \times 30 \times 15$; level-1 patience 63; best CV validation $R^2 = 0.7607$.
4. **Ensemble construction.** Runs were executed in batches of six and combined by averaging (soft voting). The ensemble was extended until aggregate validation $R^2$ stabilized, defined as change below 0.01 across two consecutive batches.
5. **Deterministic baselines.** Lasso and Elastic Net are deterministic given fixed regularization strength, so they admit no equivalent ensemble notion. Table 1 therefore compares these single models against both single stochastic-search runs and the combined soft-voted ensemble.
6. **Ensemble diagnostics (Table 3):**
   - **Fixed cardinality:** 18 runs; ensemble build time 166 s; validation $R^2 = 0.702$; individual OOS $R^2 = 0.643 \pm 0.030$ [0.575, 0.686]; global features 341.
   - **Tuned cardinality:** 18 runs; ensemble build time 343 s; validation $R^2 = 0.723$; individual OOS $R^2 = 0.665 \pm 0.039$ [0.519, 0.698]; global features 74.
   - **Endogenous stopping:** 12 runs; ensemble build time 2,127 s; validation $R^2 = 0.755$; individual OOS $R^2 = 0.668 \pm 0.009$ [0.650, 0.682]; global features 59.

## 15. Main out-of-sample results

Table 1 reports out-of-sample explanatory performance, execution time, and sparsity over the 562-day test window:

| Model | Features | Time (s) | Mean Individual $R^2$ | Ensemble $R^2$ |
|---|---:|---:|---:|---:|
| Ridge ($L_2$) | 1,072 | <1.0 | 0.5250 | — |
| Lasso ($L_1$) | 60 | <1.0 | 0.6313 | — |
| Elastic Net | 55 | <1.0 | 0.6243 | — |
| Greedy ($L_0$) | 5 | 17.1 | 0.2489 | — |
| Fixed cardinality | 60 | 52.9 | 0.6428 | 0.6855 |
| Tuned cardinality | 15 | 106.9 | 0.6648 | 0.6908 |
| Endogenous stopping | 18.6 | 1,671.6 | 0.6682 | 0.6961 |

Key observations:

1. **Fixed-cardinality configuration.** Search constrained to exactly 60 features per run; 341 unique features across the ensemble. Mean individual OOS $R^2 = 0.6428$, exceeding Lasso’s 0.6313.
2. **Tuned-cardinality configuration.** Selected 15 features per run; 74 across the ensemble. Mean individual OOS $R^2 = 0.6648$, with statistical significance $p < 0.01$ versus Lasso via one-sample $t$-test. Ensemble $R^2 = 0.6908$. Its worst run drops to 0.519.
3. **Endogenous-stopping configuration.** Removes cardinality constraint; individual runs converged to an average subset size of 18.6 features; 59 unique features across ensemble. Mean individual OOS $R^2 = 0.6682$, with $p < 0.001$ versus Lasso. Even its worst run (0.650) beats Lasso. Ensemble $R^2 = 0.6961$.
4. **Variance reduction.** Endogenous stopping, combined with additional exploration budget and the lexicographic reward, reduced between-run variance relative to tuned cardinality: $\sigma = 0.009$ versus $0.039$, and range [0.650, 0.682] versus [0.519, 0.698].
5. **Computational cost.** Most additional search time for endogenous stopping appears to come from the level-1 patience hyperparameter rather than the stopping mechanism itself. The results illustrate a trade-off between computational cost, subset size, and run-to-run variation.
6. **Ensemble trade-off.** A single endogenous-stopping run gives a sparse model (18.6 features on average) that already outperforms Lasso, while combining runs into a soft-voted ensemble gives the highest OOS $R^2 = 0.6961$ at the cost of a larger combined feature set (59 variables).

## 16. Interpretability and selected feature mechanisms

1. **No shrinkage.** Because $L_0$ selection limits only the number of features and applies no shrinkage, OLS can assign large, opposing weights to correlated variables to construct spread signals. Convex penalties can struggle to preserve such interactions because they penalize coefficient size.
2. **Best individual run.** The paper inspects the best-performing individual run from the tuned-cardinality ensemble: 15 features, OOS $R^2 = 0.6980$. The authors emphasize this is qualitative illustration, not the headline result; treating a post hoc best draw as the primary metric conflates expected performance with selection bias. The ensemble-averaged $R^2$ remains the primary evaluation metric.
3. **Economic mechanisms in the 15-feature subset.**
   - **Volatility spread construction:** The subset pairs $\text{VIX}_{t-1}^{(5\text{d mean})}$ ($\beta = -0.0621$) with a longer lag $\text{VIX}_{t-5}$ ($\beta = +0.0513$), consistent with short-term volatility persistence and mean reversion. Continuous penalties shrink these opposing coefficients, attenuating the pattern; unshrunk OLS preserves it.
   - **Credit spreads:** The daily change in the High Yield spread, $\Delta\text{HY Spread}_t$, is selected alongside its first three lags. All four coefficients are negative and similarly sized ($\beta \in [-0.0052, -0.0041]$), capturing a cumulative drag where credit spread changes affect returns over consecutive days.
   - **Factor exposures:** The Conservative-Minus-Aggressive factor, $\text{CMA}_t^{(5\text{d mean})}$ ($\beta = -0.0056$), is included alongside its one-day lag ($\beta = +0.0005$). The search retains both, with the lag appearing to act as a temporal smoothing term.
4. **Table 4 composition.** The 15-feature subset includes:
   - Volatility metrics: $\text{VIX}_{t-5}$, $\text{VIX}_{t-1}^{(5\text{d mean})}$, $\text{VIX}_t^2$, $\Delta\text{VIX}_t$, $\Delta\text{VIX}_{t-1}$, $\Delta\text{VIX}_{t-2}$.
   - Credit stress: $\Delta\text{HY Spread}_t$, $\Delta\text{HY Spread}_{t-1}$, $\Delta\text{HY Spread}_{t-2}$, $\Delta\text{HY Spread}_{t-3}$.
   - Factor exposures: $\text{CMA}_t^{(5\text{d mean})}$, $\text{CMA}_{t-1}^{(5\text{d mean})}$, $\text{HML}_t^{(5\text{d mean})}$, $\text{HML}_{t-2}^{(5\text{d std})}$, $\text{SMB}_t^{(21\text{d mean})}$.
5. **Figure 2.** Shows the correlation matrix of the best-performing individual 15-feature GNRPA subset.

## 17. Ensemble stability and linear attribution

1. **Soft voting.** To reduce variance of the stochastic search, the paper averages predictions of independent OLS regressions. Unlike a tree-based ensemble, the average of $M$ linear regressions is itself a single linear equation:

   $$ \hat{y}_{\text{ens}} = \frac{1}{M} \sum_{m=1}^M X \hat{\theta}^{(m)} = X \left( \frac{1}{M} \sum_{m=1}^M \hat{\theta}^{(m)} \right) = X \theta^*. $$

   This preserves linear performance attribution.
2. **Feature consistency.** The endogenous-stopping ensemble used 59 unique variables in total out of 1,072 candidate features, fewer than the 74 unique features used by the tuned-cardinality ensemble, even though its individual runs selected slightly larger subsets. This suggests endogenous stopping leads independent runs to converge on a more consistent set of features.
3. **Stability metrics.** Future work could formalize consistency using stability selection metrics such as selection frequency and inter-run Jaccard similarity.

## 18. Discussion, limitations, and future work

1. **Summary of contributions.** The paper presents a reinforcement-learning-based feature selection framework for high-dimensional financial regression. By formulating validation-guided $L_0$ subset selection as a sequential MDP, the approach avoids the shrinkage bias of convex regularization methods such as Lasso while retaining the interpretability of linear models.
2. **Threefold contribution restated.**
   - Introduces an adaptive search framework that optimizes discrete subsets for out-of-sample performance, avoiding convex shrinkage.
   - Demonstrates that MCTS readily accommodates domain-specific feature selection objectives, using MaxVIF to favor subsets with lower internal multicollinearity when explanatory performance is similar.
   - Shows the method finds sparse, interpretable feature subsets that outperform convex baselines out of sample on this dataset, with a linear ensemble reaching OOS $R^2 = 0.6961$ while preserving additive performance attribution.
3. **Computational cost.** Improved out-of-sample fit comes at higher computational cost and tuning burden than convex baselines. Reported Python runtimes on a single workstation are conservative and could be reduced via compiled languages, parallelized ensembles, and ablating the root-level patience parameter $P$.
4. **Limitations.** The evaluation uses a single index and a single out-of-sample window (January 2024–March 2026). The paper notes that extending the benchmark across rolling test windows, additional asset classes, and non-convex penalties such as SCAD or MCP remains important future work.
5. **Natural extension.** Applying MCTS to other combinatorial problems in finance, such as cardinality-constrained index tracking. Unlike traditional metaheuristics such as genetic algorithms or simulated annealing, which mutate complete candidate solutions, GNRPA builds a solution incrementally and adapts its policy during the search. Comparing these approaches under long-only and fully invested portfolio constraints would help clarify the practical value of reinforcement learning for this class of problem in quantitative finance.
6. **Conclusion.** Monte Carlo Tree Search is presented as a practical alternative to convex regularization for sparse explanatory feature selection in financial regression, with potential value for factor attribution and risk decomposition, where explanatory transparency matters as much as out-of-sample fit.

## 19. Compact final takeaway

The paper’s central methodological move is to replace training-error minimization in $L_0$ subset selection with validation-guided sequential search driven by GNRPA/Montparnasse, augmented by mRMR heuristics, endogenous stopping, and a lexicographic reward that prioritizes validation $R^2$, then sparsity, then low multicollinearity. Empirically, on a contemporaneous NASDAQ-100 explanatory regression with 1,072 engineered features and a 562-day out-of-sample test window, the method’s soft-voted ensemble achieves OOS $R^2 = 0.6961$ versus 0.6313 for optimized Lasso, while using only 59 unique features in the endogenous-stopping ensemble. The results are explanatory rather than predictive, and the paper is explicit that the contemporaneous alignment exploits same-day and near-term co-movement—consistent with the leverage effect—not a genuine lead–lag forecasting relationship.
