# Case 4 feasibility: Why Does the Model Think This Shopper Will Buy?

Audit date: 2026-09-28. Scope: environment, raw-data preservation, focused audit, provenance review, and two provisional fits. No teaching notebook, final feature set, final preprocessing strategy, or final model has been selected.

## 1. Case framing

An e-commerce/product analytics team wants to score purchase intention during browsing and inspect individual high/low predictions. Potential uses include UX analysis, personalization or assistance hypotheses, and investigating model scores. The chapter focus is model-agnostic local explanations with LIME and SHAP, treated as objects to audit rather than ground truth.

**Assessment:** the data supports an offline explanation-auditing case. Active-session prediction remains conditional on an explicit observation cutoff and defensible feature provenance. The recorded purchase outcome is a proxy for intention, not a measurement of shopper psychology. Attributions explain a fitted model under particular assumptions; they do not establish causal effects or actionable real-world recourse.

## 2. Dataset snapshot

The repository has `Data/raw/`, `Data/processed/`, `Data/metadata/`, `notebooks/`, `outputs/`, `src/`, root preparation scripts, and an existing `.venv`. Root `AGENTS.md` was read; no nested `AGENTS.md` was found. Existing cases were left untouched.

### Environment readiness

Python 3.13.7, using `.venv/bin/python`. Versions were checked with `importlib.metadata` and installed modules were actually imported.

| Package | Version | Import status |
| --- | --- | --- |
| pandas | 3.0.6 | Passed |
| numpy | 2.5.3 | Passed |
| scikit-learn | 1.9.1 | Passed |
| matplotlib | 3.11.2 | Passed; used a temporary cache because the default user cache was not writable |
| lime | Missing | Not executed |
| shap | Missing | Not executed |
| ucimlrepo | 0.0.7 | Passed |

No packages were installed or changed. LIME/SHAP compatibility and runtime have therefore **not** been tested. A writable temporary `MPLCONFIGDIR` can avoid the cache warning in future runs.

### Acquisition and preservation

Dataset ID 468, [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset), DOI [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q), credited to Sakar and Kastro (2018), CC BY 4.0. UCI describes different users sampled over a one-year period. [UCI documentation](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

No matching dataset was present. Downloaded the official ZIP with:

```sh
curl -fL --output /tmp/case4_uci468.zip 'https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip'
```

The sandbox attempt failed DNS resolution; the approved network retry succeeded. At 2026-09-28T07:48:39Z, Python `zipfile.ZipFile.read('online_shoppers_intention.csv')` extracted the sole archive member. Its bytes were written with exclusive creation (`open('xb')`) to `Data/raw/online_shoppers_intention.csv`, preventing overwrite. No pandas round-trip, normalization, deduplication, or editing was applied to this file. The temporary ZIP is not a repository artifact.

- Raw CSV size: **1,072,063 bytes**.
- Raw CSV SHA-256: `b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9`.
- Downloaded ZIP SHA-256: `2972e6184d3ad7beaaa831d9fc2b059dc3ee29df69d1ec593c466a5cd8485d14`.
- Extracted bytes were compared with the archive member: identical.

Observed shape: **12,330 rows × 18 columns**, comprising 17 predictors and the target `Revenue`. There are 10 semantically numeric predictors and 7 categorical predictors (including Boolean `Weekend`). Numeric category codes are not quantities.

## 3. Target and imbalance

| Revenue | Meaning | Count | Percentage |
| --- | --- | --- | --- |
| False / 0 | No recorded purchase | 10,422 | 84.5255% |
| True / 1 | Recorded purchase | 1,908 | 15.4745% |

The negative-to-positive ratio is approximately **5.46:1**. Always predicting no purchase gives 84.53% accuracy while detecting no purchases. Report ROC-AUC alongside precision-recall performance, plus precision/recall at explicitly stated thresholds and probability-quality checks when interpreting scores. Do not equate an uncalibrated score with a reliable purchase probability.

## 4. Data-quality findings

All statistics below were computed directly from the preserved CSV. Full-data summaries are descriptive audit findings, not learned preprocessing rules or target-guided feature selection.

### Columns, inferred types, and distinct values

| Column | pandas dtype | Role / semantic type | Unique | Missing | Most common value (%) |
| --- | --- | --- | --- | --- | --- |
| Administrative | int64 | Numeric | 27 | 0 | 0 (46.78%) |
| Administrative_Duration | float64 | Numeric | 3335 | 0 | 0.0 (47.88%) |
| Informational | int64 | Numeric | 17 | 0 | 0 (78.66%) |
| Informational_Duration | float64 | Numeric | 1258 | 0 | 0.0 (80.49%) |
| ProductRelated | int64 | Numeric | 311 | 0 | 1 (5.04%) |
| ProductRelated_Duration | float64 | Numeric | 9551 | 0 | 0.0 (6.12%) |
| BounceRates | float64 | Numeric | 1872 | 0 | 0.0 (44.75%) |
| ExitRates | float64 | Numeric | 4777 | 0 | 0.2 (5.76%) |
| PageValues | float64 | Numeric | 2704 | 0 | 0.0 (77.86%) |
| SpecialDay | float64 | Numeric | 6 | 0 | 0.0 (89.85%) |
| Month | str | Categorical | 10 | 0 | May (27.28%) |
| OperatingSystems | int64 | Categorical | 8 | 0 | 2 (53.54%) |
| Browser | int64 | Categorical | 13 | 0 | 2 (64.57%) |
| Region | int64 | Categorical | 9 | 0 | 1 (38.77%) |
| TrafficType | int64 | Categorical | 20 | 0 | 2 (31.74%) |
| VisitorType | str | Categorical | 3 | 0 | Returning_Visitor (85.57%) |
| Weekend | bool | Categorical | 2 | 0 | False (76.74%) |
| Revenue | bool | Target | 2 | 0 | False (84.53%) |

No missing values, nonfinite numeric values, or negative numeric values were found. There are no constant variables. Using an explicit ≥95% modal-frequency screening rule, no predictor is near-constant; this is an audit convention, not a removal rule. `SpecialDay` is nevertheless 89.85% zero, `Informational_Duration` 80.49% zero, and `PageValues` 77.86% zero. Zero inflation can collapse quantile bins and distort perturbation neighborhoods.

### Categorical cardinality and rare levels

| Feature | Levels | Observed counts |
| --- | --- | --- |
| Month | 10 | May: 3364; Nov: 2998; Mar: 1907; Dec: 1727; Oct: 549; Sep: 448; Aug: 433; Jul: 432; June: 288; Feb: 184 |
| OperatingSystems | 8 | 2: 6601; 1: 2585; 3: 2555; 4: 478; 8: 79; 6: 19; 7: 7; 5: 6 |
| Browser | 13 | 2: 7961; 1: 2462; 4: 736; 5: 467; 6: 174; 10: 163; 8: 135; 3: 105; 13: 61; 7: 49; 12: 10; 11: 6; 9: 1 |
| Region | 9 | 1: 4780; 3: 2403; 4: 1182; 2: 1136; 6: 805; 7: 761; 9: 511; 8: 434; 5: 318 |
| TrafficType | 20 | 2: 3913; 1: 2451; 3: 2052; 4: 1069; 13: 738; 10: 450; 6: 444; 8: 343; 5: 260; 11: 247; 20: 198; 9: 42; 7: 40; 15: 38; 19: 17; 14: 13; 18: 10; 16: 3; 12: 1; 17: 1 |
| VisitorType | 3 | Returning_Visitor: 10551; New_Visitor: 1694; Other: 85 |
| Weekend | 2 | False: 9462; True: 2868 |

There are 65 categorical levels overall. `Month` omits January and April; `June` uses a different abbreviation length. Several browser/traffic codes occur only once. Preserve literal labels; do not invent code meanings absent a codebook. Rare levels can be absent from training and cannot support strong local conclusions.

### Numeric ranges relevant to perturbations

Durations are shown in stored units; the inspected UCI variable table does not establish their units. Extreme observations are candidates for inspection, not automatic errors or deletion.

| Feature | Min | Median | 95th percentile | 99th percentile | Max | Zero % |
| --- | --- | --- | --- | --- | --- | --- |
| Administrative | 0 | 1 | 9 | 14 | 27 | 46.78 |
| Administrative_Duration | 0 | 7.5 | 348.266 | 830.587 | 3398.75 | 47.88 |
| Informational | 0 | 0 | 3 | 6 | 24 | 78.66 |
| Informational_Duration | 0 | 0 | 195 | 716.39 | 2549.38 | 80.49 |
| ProductRelated | 0 | 18 | 109 | 221 | 705 | 0.31 |
| ProductRelated_Duration | 0 | 598.937 | 4300.29 | 8701.14 | 63973.5 | 6.12 |
| BounceRates | 0 | 0.00311247 | 0.2 | 0.2 | 0.2 | 44.75 |
| ExitRates | 0 | 0.0251564 | 0.2 | 0.2 | 0.2 | 0.62 |
| PageValues | 0 | 0 | 38.1605 | 85.4985 | 361.764 | 77.86 |
| SpecialDay | 0 | 0 | 0.6 | 1 | 1 | 89.85 |

`ProductRelated_Duration` reaches 63,973.52 against a median of 598.94, and `ProductRelated` reaches 705 against 18: strong right tails affect distance-based neighborhoods. Both rates stop at 0.2 (700 `BounceRates` and 710 `ExitRates` observations at that ceiling). The audit cannot distinguish source scaling, clipping, or natural support; do not rescale them to 0–1 or assert a mechanism. `SpecialDay` takes only {0, .2, .4, .6, .8, 1}.

No row has zero page count with positive matching duration. Positive count with zero duration occurs for Administrative (135), Informational (226), and ProductRelated (717); possible measurement/recording behavior needs clarification, not automatic imputation.

### Duplicates, identifiers, and split implications

There are **125 extra identical rows**, involving 201 rows in 76 duplicate groups; all are non-purchases. Predictor-only duplicates are also 125, with no conflicting targets. There is no obvious user/session ID, full timestamp, or page-sequence column. A DataFrame index is only a row locator and must not become a predictor. High uniqueness of continuous durations does not make them identifiers.

A standard 80/20 stratified split (`random_state=42`) puts 22 identical-predictor groups across partitions, with 29 test rows matching training predictors. Identical aggregate rows might represent distinct people rather than duplicated collection records. Keep raw rows and resolve the policy explicitly. For the provisional fits, identical predictor vectors were grouped so none crossed the split; no rows were removed.

Stratification is useful given imbalance, but does not guarantee category coverage, temporal generalization, or valid feature timing. Purchase rates differ descriptively by month (February 1.63%, November 25.35%); a random split mixes these populations. Month alone cannot establish exact chronology or within-month cutoffs. No user IDs exist to verify the documented distinct-user sampling. A future deployment claim requires stronger temporal/provenance evidence; the present split is an offline feasibility split.

## 5. Feature provenance and timing

Documentation facts: UCI describes page-category counts/durations updated with navigation; Bounce/Exit/Page Value as page-level Google Analytics metrics; SpecialDay as calendar proximity adjusted for commerce timing; VisitorType as visitor status; and Month as visit month. Its definitions do not specify a reproducible as-of aggregation pipeline. [UCI variable descriptions](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

The original paper's accessible abstract describes using aggregates tracked during visits. That supports the intended setting, but does not establish the observation cutoff of each released CSV row; the full collection implementation was not verified. [Original paper, Sakar et al.](https://doi.org/10.1007/s00521-018-3523-0)

The classifications below are this audit's cautious judgments, not labels supplied by UCI.

| Variable | Documented meaning | Classification | Assumption / implication |
| --- | --- | --- | --- |
| PageValues | Average page value associated with transactions after page visits | **Suspicious for the intended prediction use** | Establish when the underlying values were computed and whether current-session or future outcomes contributed. Suspicion is not proof of leakage. |
| BounceRates | Page entry visits ending without another analytics request, as a proportion | **Timing/provenance-sensitive** | Historical page-level rates could be available while browsing; the current session's final bounce status is not known then. The aggregation window and page-to-session mapping are unverified. |
| ExitRates | Proportion of pageviews that are last in a session | **Timing/provenance-sensitive** | Historical rates can be used prospectively; a current session's final exit is future information before it happens. Do not confuse these interpretations. |
| SpecialDay | Proximity to a special date | **Clearly available before the outcome**, if its calendar rule is fixed | Calendar information can be computed in advance. Whether the supplied commerce-specific rule was informed by later outcomes is undocumented. |
| VisitorType | New/returning visitor status; CSV also contains Other | **Likely available during the browsing session** | Requires pre-existing tracking/history. Identity resolution and the meaning of Other are unspecified. |
| Month | Month of visit | **Clearly available before the outcome** | Assumes session-start/current date rather than retrospectively assigning a cross-month session. Seasonal association is not a causal effect. |

### Why PageValues needs a human decision

Google's legacy Analytics explanation computes Page Value from transaction revenue and goal value divided by unique pageviews, allocating value using subsequent conversions. This is historical Universal Analytics context, not an assumption that modern GA4 metrics are identical. [Google's Page Value explanation](https://analytics.googleblog.com/2012/07/understanding-and-using-page-value.html) and [Google's forward-looking attribution description](https://support.google.com/analytics/answer/1033861?hl=en).

Three scenarios have different implications:

1. **Historical, frozen page values:** computed from completed earlier sessions, then aggregated only over pages already visited at the prediction cutoff. Potentially legitimate predictors, provided historical computation excludes held-out/future outcomes.
2. **Retrospective values:** include the current session's eventual transaction or outcomes from a later reporting window. These would contaminate a prospective prediction; a train/test split applied afterward cannot repair precomputed outcome contamination.
3. **Unknown window, as in the inspected release:** suitability is unresolved. Label the assumption and consider an exclusion sensitivity experiment; do not announce confirmed leakage.

Observed association is large but not deterministic: PageValues = 0 has 370 purchases / 9,600 sessions (3.85%); PageValues > 0 has 1,538 / 2,730 (56.34%). This does not identify its computation window or prove leakage.

The six page-count/duration predictors also need a cutoff. Incremental values could be available during browsing, but a released whole-session total is not interchangeable with the value after, for example, the fifth pageview. This CSV has no sequence or snapshot timestamps to reconstruct that experiment. Excluding PageValues alone does **not** certify a prospective feature set.

## 6. Relevant behavioral feature relationships

Correlations below use the ten semantic numeric predictors, excluding arbitrary category codes and the target. They describe this file, not causation. Both Pearson and Spearman are useful because tails, zeros, and nonlinear monotonic relationships differ.

### Strongest Pearson correlations (absolute ranking)

| Predictor A | Predictor B | Correlation |
| --- | --- | --- |
| ExitRates | BounceRates | 0.9130 |
| ProductRelated_Duration | ProductRelated | 0.8609 |
| Informational_Duration | Informational | 0.6190 |
| Administrative_Duration | Administrative | 0.6016 |
| ProductRelated | Administrative | 0.4311 |
| ProductRelated_Duration | Informational | 0.3875 |
### Strongest Spearman correlations (absolute ranking)

| Predictor A | Predictor B | Correlation |
| --- | --- | --- |
| Informational_Duration | Informational | 0.9510 |
| Administrative_Duration | Administrative | 0.9407 |
| ProductRelated_Duration | ProductRelated | 0.8827 |
| ExitRates | BounceRates | 0.6023 |
| ExitRates | ProductRelated | -0.5189 |
| ExitRates | ProductRelated_Duration | -0.4769 |

BounceRates/ExitRates and each page count/duration pair are candidates for attribution redistribution, not automatic removal. Administrative count/duration has Spearman 0.9407 despite Pearson 0.6016; Informational is 0.9510 versus 0.6190. Redundancy depends on the relationship being measured. Independent perturbations can leave these observed relationships.

A categorical dependency also matters: every nonzero SpecialDay appears in February or May in this file. Independently sampling Month and SpecialDay creates unsupported combinations. Neither pairwise correlation nor grouped importance alone captures all such constraints.

## 7. Preprocessing implications

All options remain provisional. Fit encoders, scaling, binning, imputation if later needed, and any target-guided steps using training/development data only. The test set must not supply explanation backgrounds or perturbation statistics.

| Choice | Benefits | Costs / safeguards |
| --- | --- | --- |
| One-hot categorical inputs | Avoids treating nominal codes as ordered measurements; straightforward sklearn pipeline | 10 numeric + 65 categorical levels would give 75 columns if fitted to the full inventory; this run learned 74 because one traffic level was absent. Expanded explanations can fragment one concept. Handle unknown levels explicitly. |
| Native categorical model support | Potentially retains the 17-feature representation | Depends on the eventual estimator and version; still requires correct categorical declarations and an explainer-compatible prediction wrapper. Not tested here. |
| Ordinal codes | Compact transport representation for explainers | Nominal categories must not silently become numeric distances or ordered split semantics. Decode before calling a pipeline that treats them categorically. |
| Numeric passthrough | Adequate for the provisional random forest | Does not solve distance/perturbation problems in LIME. |
| Training-fitted scaling or log transforms | May help other model families or skewed neighborhood geometry | Not needed for tree splits; explanations must recover original units. Log transforms require a clear zero convention and do not fix invalid feature combinations. |

For presentation, keep a mapping from original columns and encoder `categories_` to `ColumnTransformer.get_feature_names_out()` and output slices; do not guess names from string splitting. Check transformed feature count and order. Prefer an explainer interface in original-feature space that decodes categories and calls the entire fitted pipeline. Verify its batch probabilities against the pipeline on unmodified rows, including positive-class indexing.

If explanations are computed after one-hot encoding, summing **signed** SHAP values over a categorical block preserves the total attribution for that computed game. It is not generally identical to SHAP computed with the original category as one coalition player; summing absolute values instead inflates importance through cancellation loss. Independently perturbing dummy columns can produce multiple/no active categories.

LIME requires categorical column indices and consistent integer/category-name mappings. Its documented settings include discretization, random state, kernel width, and continuous sampling around the instance versus the training mean. Categorical sampling uses training frequencies. [LIME tabular API](https://lime-ml.readthedocs.io/en/latest/lime.html) and [author's categorical tutorial](https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20continuous%20and%20categorical%20features.html).

Audit generated samples: negative/fractional counts, rates outside observed support, positive duration with zero pages, inconsistent count/duration pairs, and Month/SpecialDay combinations. Even valid individual categories may form unsupported joint configurations. Clipping or rounding may repair support but changes the explanation neighborhood and must be reported. Training-only, support-aware or empirical joint sampling are options, not a selected implementation. Zero-heavy quantile bins need inspection; entropy/label-guided binning must never see evaluation labels.

## 8. Lightweight modeling feasibility

Two fits of the **same** provisional nonlinear estimator were run: `RandomForestClassifier(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=2)`, otherwise defaults, with numeric passthrough and training-fitted one-hot encoding (`handle_unknown='ignore', sparse_output=False`). One uses all supplied predictors; the other omits only PageValues as a provenance sensitivity probe. This is neither a model competition nor final feature selection. No tuning, resampling, calibration, or learned threshold was applied.

Split: first fold of `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, grouping identical full predictor vectors. Training: 9,864 rows (1,526 purchases); test: 2,466 rows (382 purchases). All rows retained; no identical vector crosses partitions. Encoders were fit only on training rows. Test-only TrafficType code 12 was handled as unknown. No models or processed datasets were saved.

| Predictor scenario | ROC-AUC | PR-AUC (trapezoid) | Average precision | Precision @ .5 | Recall @ .5 | Brier |
| --- | --- | --- | --- | --- | --- | --- |
| all_supplied_predictors | 0.9326 | 0.7609 | 0.7612 | 0.7899 | 0.4921 | 0.0714 |
| without_PageValues | 0.7752 | 0.3853 | 0.3867 | 0.5854 | 0.0628 | 0.1133 |

Average precision (AP) uses recall-increment weighting and is not the same numerical summary as trapezoidal PR-AUC; both are reported explicitly. [scikit-learn AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). The no-skill precision reference is test prevalence, **0.1549**; a constant score has ROC-AUC 0.5. The constant training-prevalence probability gives Brier **0.1309**. Brier measures probability error including both calibration and discrimination; these values do not by themselves demonstrate good calibration.

Confusion matrices at the illustrative fixed 0.5 threshold, rows = actual [no purchase, purchase], columns = predicted [no purchase, purchase]:

```text
All predictors:       [[2034, 50], [194, 188]]
Without PageValues:   [[2067, 17], [358,  24]]
```

**Answer to the feasibility question:** yes, there is useful offline ranking signal to explain. Without PageValues, ROC-AUC remains 0.7752 and AP 0.3867, but recall at 0.5 is only 0.0628. This does not establish usefulness for any particular assistance policy. The large performance change is a reason to investigate provenance, not evidence that PageValues is necessarily invalid. Neither fit demonstrates active-session performance or causal value of an intervention.

Only one approximately stratified grouped holdout was evaluated, with no uncertainty intervals. Results are preliminary. Because these results have now been inspected, treat this partition as development evidence if it informs later choices; arrange a fresh final evaluation protocol rather than presenting it as untouched final testing.

For the requested session types, observed score bands were:

| Scenario | Score ≤ .10 | .45 ≤ score ≤ .55 | Score ≥ .90 |
| --- | --- | --- | --- |
| all_supplied_predictors | 1562 | 88 | 0 |
| without_PageValues | 969 | 75 | 0 |

Borderline and low-score examples exist. **Neither provisional fit produced a ≥0.90 purchase score** on this holdout. “High confidence” needs a human-approved operational definition and calibration evidence. A highest-score/upper-quantile example can be called relatively high-scoring, not silently relabeled ≥90% confidence. No examples were selected for a final narrative.

## 9. LIME feasibility

LIME fits an interpretable model around a prediction; local fidelity is a property to measure, not presume. [Original LIME paper](https://arxiv.org/abs/1602.04938). **A–C are feasible in principle but unexecuted because lime is missing.**

| Experiment | Feasibility and complications |
| --- | --- |
| A. Explain one prediction | Use a held-out row, frozen fitted pipeline, purchase-class probability, and training-derived neighborhood statistics. Check the surrogate's weighted local fit and error at the original row. Correlation, skew, and rare categories can make a visually plausible explanation unreliable. |
| B. Repeat seeds/settings | Hold row/model/background/preprocessing fixed. Recreate the explainer with each explicit seed; separately vary sample count, kernel width, or discretization one factor at a time. Changing bins changes the objects being compared, so align original features and record predicate labels. |
| C. Quantify stability | On a proposed small grid (e.g. 20 seeds, fixed top-5), record feature selection frequency, pairwise top-k Jaccard, absolute-weight rank agreement, sign agreement, weight dispersion, local fidelity, original-row error, and invalid-sample rate. Report distributions, not just one average. |

Proposed definitions: Jaccard = size of top-k intersection / size of union; compute rank agreement on a common feature universe with an explicit tie/omitted-feature rule. Sign agreement should condition on both runs selecting the feature and on an explicit near-zero tolerance; absence is not negative contribution. Keep bins fixed for seed-only comparisons. Compare directions for the same predicate/value, not merely a shared column name. A stable but poorly fitting surrogate is not satisfactory. These are planned audit measures; no stability numbers are claimed here.

## 10. SHAP feasibility

SHAP assigns additive feature attributions to individual model outputs. [Original SHAP paper](https://arxiv.org/abs/1705.07874). **D–H are feasible subject to the qualifications below, but shap is missing and no attribution computation was run.**

For a model-agnostic chapter, Kernel SHAP or permutation SHAP can call a frozen pipeline. Kernel SHAP integrates masked features using background rows and supports identity or logit links; its output units must be stated. [KernelExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html). PermutationExplainer is another model-agnostic option with a configurable evaluation budget. [PermutationExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html). An automatic explainer choice must be inspected: a tree-specific method would change the teaching emphasis.

| Experiment | Feasibility and complications |
| --- | --- |
| D. Explain the exact same session | Preserve raw row locator/hash, pipeline, class index, and output scale. Check `base_value + sum(attributions)` against that same model output within a declared numerical tolerance. Test wrappers and output shapes after installation. |
| E. Compare LIME and SHAP | Align original-feature groups and purchase output. Compare top-k overlap, ranks, and signs, but explain that a local surrogate coefficient and a reference-relative Shapley attribution answer different questions. Their raw magnitudes and signs need not agree. Disagreement does not identify which is correct. |
| F. Vary background/reference | Use equal-size training-only samples representing overall training prevalence, returning visitors, or a specified calendar subgroup. Keep model and explained session fixed. Record background membership/seed, expected output, attributions, and reconstruction error. Repeat samples within a population to separate sampling noise from population change. A class-balanced reference intentionally changes the question and must be labeled. |
| G. Correlated behavior | Inspect count/duration pairs and BounceRates/ExitRates as individual and grouped contributions. Standard marginal masking can create implausible hybrids; conditional or constrained-coalition alternatives change the estimand. Group sums are descriptive, not proof of causal credit or equivalence to grouped-game SHAP. An ablation/refit would also change the model and must be distinguished from explainer sensitivity. |
| H. High, low, borderline | Freeze a score-based selection rule on development data; use identical rows for LIME and SHAP. Display score, threshold, observed label, and uncertainty/fidelity caveats, including incorrect predictions. Strict ≥.90 purchase examples were absent here, so that part is conditional rather than demonstrated. Borderline means close to an explicit decision threshold, not necessarily 0.5. |

A modest training background (e.g. 50–100 actual rows) and a few sessions are a reasonable first runtime probe, not a proven budget. There are 17 original players versus up to 75 expanded columns; exact enumeration is costly, especially after encoding. Actual-row references preserve category validity better than arbitrary averaged centroids, but masking still mixes the explained row with reference rows and can break dependence.

Changing background changes the expected prediction and hence the comparison being explained; attribution changes can be appropriate rather than algorithmic failure. Holding the model fixed, the reconstructed prediction should remain the same. Feature dependence changes how credit is allocated; methods that account for it need explicit assumptions and computational checks. [Primary research on dependent-feature SHAP](https://arxiv.org/abs/1903.10464).

## 11. Explanation-specific risks

- **Timing dominates interpretation:** an explanation of a retrospective metric can be faithful to the model yet unsuitable for an active-session claim. Standard splitting cannot fix upstream contamination.
- **Association is not psychology or causation:** “the model assigns higher purchase probability” does not mean “this feature caused buying.” A model-level counterfactual is not validated shopper recourse or an effective personalization action.
- **Neighborhood/reference artifacts:** invalid perturbed sessions, rare categories, zero-heavy bins, and correlated variables can drive unstable or misleading local explanations.
- **False comparability:** output scales, class orientation, bins, categorical grouping, and baseline choices must align before comparing methods. LIME's intercept is not automatically SHAP's expected prediction.
- **Confidence and threshold ambiguity:** high model score, correct classification, calibrated probability, explanation fidelity, and explanation stability are different properties. A borderline decision can flip even while its probability changes little.
- **Selection and evaluation reuse:** choosing only attractive explanations or repeatedly tuning against this holdout would overstate evidence. Report difficult cases and reserve a final evaluation protocol.

## 12. Important open questions

Human decisions required before teaching implementation:

1. What exact point in browsing is being scored: an early fixed cutoff, any current snapshot, or a retrospective session record? The CSV cannot reconstruct arbitrary early snapshots.
2. What evidence or explicitly labeled assumption is acceptable for PageValues, BounceRates, ExitRates, and accumulated counts/durations? Should the teaching case retain PageValues as a sensitivity scenario, exclude it from the main analysis, or seek better provenance? No final feature decision is made here.
3. What final research question/narrative and dataset suitability claim should students work with? Offline model auditing is supported; prospective operational validation is not yet supported.
4. What defines high/low confidence and a borderline score? Is calibration needed, and what development-only threshold procedure matches the intended use?
5. Which duplicate/split policy, final model, preprocessing representation, SHAP reference population, and explanation comparison protocol should be adopted? These remain choices, not conclusions from the two fits.
6. When further implementation is authorized, may lime and shap be installed into the existing `.venv` and their compatibility tested? No installation has occurred.

Additional provenance gaps: the meaning of Other and numeric category codes, rate ceilings/aggregation, duration units and recording conventions, exact sampling dates, and the page-value reporting window.

## 13. Recommended next experiments

After the human framing/timing decisions, proposed work is:

1. Specify the prediction contract and document an as-of assumption for every retained predictor; seek original collection/codebook evidence where possible.
2. Establish training/development/final evaluation roles with duplicate handling. Keep reference populations, bins, calibration, thresholds, and any target-guided selection inside training/development.
3. Install only the authorized missing explanation packages, verify imports, and smoke-test categorical decoding, positive-class probabilities, feature names, and explanation reconstruction on one row.
4. Run a small LIME seed-only stability experiment, followed by one-factor neighborhood sensitivity, measuring both stability and fidelity/support validity.
5. Explain the same development sessions with model-agnostic SHAP. Hold the model fixed while testing a small set of justified training reference populations; examine both individual and grouped correlated-feature attribution.
6. Compare high-scoring, low-scoring, and borderline sessions under an agreed rule. Assess agreement, disagreement, and assumption sensitivity before deciding what belongs in one final teaching notebook.

**Execution status:** raw acquisition, imports for available packages, audit, and both model fits succeeded. LIME/SHAP experiments were assessed, not executed. No generic plots, teaching notebook, saved model, processed data, extensive tuning, or changes to existing cases were made. Stop here pending human decisions.

### Commands and reproducibility record

Read-only inspection used `cat AGENTS.md`, `rg --files` (including a search for nested AGENTS.md), `ls -la`, and `git status --short`. Environment checks used an inline `.venv/bin/python` script with `importlib.metadata.version` and `importlib.import_module`. Dataset acquisition used the curl command above and an inline Python `zipfile`/`hashlib` extraction check. Official UCI, Google Analytics, LIME, SHAP, scikit-learn documentation and primary-paper pages were inspected through web browsing; relevant links are placed with the claims they support.

Numerical work ran as `.venv/bin/python /tmp/case4_audit.py`. A supplemental inline pandas check counted rate ceilings, listed SpecialDay values, cross-tabulated Month/SpecialDay, and counted categorical levels. The report was assembled with `.venv/bin/python /tmp/case4_report.py`; final checks verified report sections, raw-byte equality/SHA-256, and numerical invariants. Temporary working files are outside the repository. The only repository additions are this report and the raw CSV.

For durable reproducibility without another repository artifact, the exact numerical audit/model script used is preserved below. Save this block to a temporary file and run it from the repository root using `.venv/bin/python`. It reads raw data and writes only `/tmp/case4_audit.json`; it does not install anything or modify the raw CSV. The supplemental checks are reproducible with `pd.crosstab(df.Month, df.SpecialDay)` and equality/value-count operations described above.

```python
import json, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, precision_score, recall_score, confusion_matrix, brier_score_loss

path=Path('Data/raw/online_shoppers_intention.csv')
df=pd.read_csv(path)
X=df.drop(columns='Revenue'); y=df.Revenue.astype(int)
cat=['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend']
num=[c for c in X if c not in cat]
result={'shape':list(df.shape),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'target':y.value_counts().to_dict(),'purchase_rate':float(y.mean()),'numeric':num,'categorical':cat}
result['columns']=[{'name':c,'dtype':str(df[c].dtype),'unique':int(df[c].nunique()),'missing':int(df[c].isna().sum()),'mode':str(df[c].mode().iloc[0]),'mode_pct':float(df[c].value_counts(normalize=True).iloc[0]*100)} for c in df]
result['categories']={c:{str(k):int(v) for k,v in X[c].value_counts().items()} for c in cat}
result['numeric_stats']=X[num].describe(percentiles=[.5,.95,.99]).to_dict()
result['zero_pct']=(X[num].eq(0).mean()*100).to_dict()
result['nonfinite']=int((~np.isfinite(X[num])).sum().sum())
result['negative']=(X[num]<0).sum().to_dict()
result['duplicates']={'extra_full_rows':int(df.duplicated().sum()),'all_full_duplicate_members':int(df.duplicated(keep=False).sum()),'extra_predictor_rows':int(X.duplicated().sum()),'duplicate_labels':df.loc[df.duplicated(keep=False),'Revenue'].value_counts().to_dict()}
groups=pd.util.hash_pandas_object(X,index=False)
result['duplicates']['conflicting_predictor_groups']=int(pd.DataFrame({'g':groups,'y':y}).groupby('g').y.nunique().gt(1).sum())
a,b=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=42)
result['duplicates']['ordinary_split_shared_groups']=len(set(groups.iloc[a])&set(groups.iloc[b]))
result['duplicates']['ordinary_split_test_rows_seen']=int(groups.iloc[b].isin(set(groups.iloc[a])).sum())
result['correlations']={}
for method in ['pearson','spearman']:
 corr=X[num].corr(method=method)
 pairs=[{'a':num[i],'b':num[j],'r':float(corr.iloc[i,j])} for i in range(len(num)) for j in range(i)]
 result['correlations'][method]=sorted(pairs,key=lambda p:abs(p['r']),reverse=True)[:10]
result['count_duration_checks']={}
for c in ['Administrative','Informational','ProductRelated']:
 result['count_duration_checks'][c]={'zero_count_positive_duration':int(((X[c]==0)&(X[c+'_Duration']>0)).sum()),'positive_count_zero_duration':int(((X[c]>0)&(X[c+'_Duration']==0)).sum())}
result['month_target']=df.groupby('Month').Revenue.agg(['count','mean']).to_dict('index')
result['pagevalues_target']=df.groupby(df.PageValues.gt(0)).Revenue.agg(['count','sum','mean']).to_dict('index')
train,test=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(X,y,groups))
assert not set(groups.iloc[train])&set(groups.iloc[test])
result['split']={'method':'first fold of StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), groups=hash of all 17 predictors','train_n':len(train),'test_n':len(test),'train_counts':y.iloc[train].value_counts().to_dict(),'test_counts':y.iloc[test].value_counts().to_dict()}
result['models']={}
for name,drop in [('all_supplied_predictors',[]),('without_PageValues',['PageValues'])]:
 cols=[c for c in num if c not in drop]
 prep=ColumnTransformer([('numeric','passthrough',cols),('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
 model=make_pipeline(prep,RandomForestClassifier(n_estimators=150,min_samples_leaf=3,random_state=42,n_jobs=2))
 model.fit(X.iloc[train],y.iloc[train])
 p=model.predict_proba(X.iloc[test])[:,list(model.classes_).index(1)]; pred=p>=.5
 precision,recall,_=precision_recall_curve(y.iloc[test],p)
 result['models'][name]={'ROC_AUC':roc_auc_score(y.iloc[test],p),'AP':average_precision_score(y.iloc[test],p),'PR_AUC_trapezoid':auc(recall,precision),'precision_at_0.5':precision_score(y.iloc[test],pred,zero_division=0),'recall_at_0.5':recall_score(y.iloc[test],pred),'confusion_matrix_TN_FP_FN_TP':confusion_matrix(y.iloc[test],pred).tolist(),'Brier':brier_score_loss(y.iloc[test],p),'encoded_features':len(model[0].get_feature_names_out()),'score_bands':{'p<=0.1':int((p<=.1).sum()),'0.45<=p<=0.55':int(((p>=.45)&(p<=.55)).sum()),'p>=0.9':int((p>=.9).sum())},'unseen_test_categories':{c:sorted(map(str,set(X.iloc[test][c])-set(X.iloc[train][c]))) for c in cat}}
result['baseline']={'test_prevalence':float(y.iloc[test].mean()),'constant_train_prevalence_brier':brier_score_loss(y.iloc[test],np.repeat(y.iloc[train].mean(),len(test)))}
Path('/tmp/case4_audit.json').write_text(json.dumps(result,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else str(x)))
print(json.dumps(result,indent=2))
```
