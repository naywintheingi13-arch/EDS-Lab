"""Render Case 4 numerical artifacts; never fit models or choose experiments."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def table(frame, columns=None):
    if columns is not None:
        frame = frame[columns]
    def cell(v):
        if isinstance(v, float):
            return '—' if pd.isna(v) else f'{v:.5f}'
        return str(v).replace('|', '/')
    rows = ['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows += ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(rows) + '\n\n'


def render(root, out):
    def read(name):
        return json.loads((out / name).read_text())
    def csv(name):
        return pd.read_csv(out / name)
    config, environment = read('config.json'), read('environment.json')
    split, validation = read('split_metadata.json'), read('validation.json')
    metrics = read('predictive_metrics.json')
    cases, lf, lr = csv('selected_cases.csv'), csv('lime_features.csv'), csv('lime_runs.csv')
    sf, sr = csv('shap_values.csv'), csv('shap_runs.csv')
    freq, stability = csv('lime_selection_summary.csv'), read('lime_stability_summary.json')
    support, bg = csv('perturbation_support.csv'), csv('shap_background_comparison.csv')
    comparisons = read('lime_vs_shap.json')
    primary_lime = lf[(lf.case == 'near_threshold') & (lf.seed == 0) & (lf.samples == 5000)].sort_values('rank')
    primary_run = lr[(lr.case == 'near_threshold') & (lr.seed == 0) & (lr.samples == 5000)].iloc[0]
    primary_shap = sf[(sf.case == 'near_threshold') & (sf.background == 'A_overall_train')].sort_values('rank')
    primary_shap_run = sr[(sr.case == 'near_threshold') & (sr.background == 'A_overall_train')].iloc[0]
    bg_runs = sr[sr.case == 'near_threshold'].set_index('background')
    shift = bg_runs.loc['B_nonpurchase_train', 'base_value'] - bg_runs.loc['A_overall_train', 'base_value']
    cross = next(r for r in comparisons if r['case'] == 'near_threshold')
    bgo = read('shap_background_overlap.json')
    figs = out / 'figures'; figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    chosen = freq[freq.selected_runs > 0].sort_values('frequency')
    axes[0].barh(chosen.feature, chosen.frequency, color='#286a91')
    axes[0].set(xlim=(0, 1.05), xlabel='Fraction of 20 seeds selecting feature', title='Near-threshold session: top-5 selection')
    seeds = lr[lr.experiment == 'seed_stability'].sort_values('seed')
    axes[1].plot(seeds.seed, seeds.weighted_r2, 'o-', label='Weighted neighborhood R²')
    axes[1].plot(seeds.seed, seeds.absolute_point_error, 's-', label='Absolute error at explained row')
    axes[1].set(xlabel='LIME random seed', title='Stability must be read alongside fidelity')
    axes[1].set_xticks([0, 5, 10, 15, 19])
    axes[1].legend(loc='best', fontsize=9)
    fig.savefig(figs / 'lime_stability.png', dpi=160); plt.close(fig)

    plot = bg.copy()
    plot['strength'] = np.maximum(abs(plot.A_value), abs(plot.B_value))
    plot = plot.sort_values('strength').tail(10)
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    pos = np.arange(len(plot))
    ax.barh(pos - .19, plot.A_value, .38, label='A: overall training sample', color='#286a91')
    ax.barh(pos + .19, plot.B_value, .38, label='B: non-purchasing training sample', color='#ce7040')
    ax.set_yticks(pos, plot.feature)
    ax.axvline(0, color='black', linewidth=.6)
    ax.set(xlabel='Kernel SHAP contribution (purchase-score units)', title='Same near-threshold session, different reference populations')
    ax.legend(fontsize=9)
    fig.savefig(figs / 'shap_background_sensitivity.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout='constrained')
    for i, case in enumerate(['high_scoring_purchaser', 'low_scoring_non_purchaser', 'near_threshold']):
        lc = lf[(lf.case == case) & (lf.seed == 0) & (lf.samples == 5000)].sort_values('rank', ascending=False)
        sc = sf[(sf.case == case) & (sf.background == 'A_overall_train')].nsmallest(5, 'rank').sort_values('rank', ascending=False)
        for ax, rows, label in [(axes[i, 0], lc, 'LIME surrogate weights'), (axes[i, 1], sc, 'Kernel SHAP contributions')]:
            ax.barh(rows.feature, rows.weight, color=['#286a91' if w >= 0 else '#ce7040' for w in rows.weight])
            ax.axvline(0, color='black', linewidth=.6)
            ax.set_title(case.replace('_', ' ') + '\n' + label, fontsize=10)
            ax.set_xlabel('Surrogate coefficient' if label.startswith('LIME') else 'Purchase-score contribution')
    fig.suptitle('Different explanation quantities: do not compare raw bar magnitudes across methods', fontsize=12)
    fig.savefig(figs / 'three_session_explanations.png', dpi=160); plt.close(fig)

    sections = []
    def add(title, text):
        sections.append(f'## {len(sections) + 1}. {title}\n\n{text}\n\n')
    add('Frozen case definition', '''**Why Does the Model Think This Shopper Will Buy?** Chapter: model-agnostic local explanations using LIME and SHAP.

Primary question: **Can we trust the explanation of an individual online-shopping prediction?** This is an explanation audit. The human-frozen design was executed without model competition, new post-result experiments, or a final teaching notebook. Numerical artifacts and figures are in [case4_online_shopping/](case4_online_shopping/); working entry point: [research script](../run_case4_online_shopping_experiments.py).''')
    add('Prediction contract and limitations', '''Target: `Revenue`, whether a completed session ended in a recorded purchase. The outcome is a behavioral proxy, not shopper psychology or true purchase intention. This is **offline session-level prediction and explanation auditing**.

An operational analogue might support analytics, UX analysis, personalization, or assistance, but would need genuinely decision-time inputs, timestamped feature provenance, a reproducible within-session cutoff, prospective evaluation, and evidence about intervention effects. These session aggregates establish none of those deployment claims. Removing PageValues does not make the remaining totals valid early-session measurements.''')
    add('Feature policy', '''PageValues is excluded because its as-of validity cannot be established sufficiently. This is not a declaration of confirmed leakage. There is no PageValues model or attribution experiment here.

Nine numeric predictors: Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, SpecialDay.

Seven nominal categorical predictors: Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend. All 16 are retained; correlated pairs are intentional audit material. No target-guided automatic predictive feature selection was performed. LIME's selection of five surrogate terms is explanation sparsification, not predictive feature selection.''')
    prep = read('preprocessing.json')
    add('Split and preprocessing', table(pd.DataFrame(split['summary'])) + f'''Fresh split: first fold of 5-fold StratifiedGroupKFold (seed 2026) reserves final test; first fold of 4-fold StratifiedGroupKFold (seed 2027) splits the remaining rows into training/development. Grouping uses SHA-256 of the **final 16 predictors**, without PageValues. All 12,330 rows remain; {split['extra_duplicate_final_vectors']} extra identical final vectors and {split['mixed_label_groups']} groups with mixed outcomes were found. No group crosses partitions. This differs from the feasibility holdout. It is a fresh partition of the same previously audited dataset, not an independent new data collection.

Row index is zero-based CSV record position; CSV line = row index + 2. Persisted hashes use ordered JSON feature values, with nominal columns represented as strings. Membership, hashes, and targets are in `split_assignments.csv`; these identifiers are never model inputs.

OneHotEncoder(handle_unknown='ignore') is fit on training only; numeric inputs pass through. The fitted representation has {len(prep['encoded_names'])} columns. The original-to-encoded mapping is saved in `preprocessing.json`.

Both explainers use 16 conceptual features. Categorical numeric transport codes are training-fitted and explicitly declared to LIME, then decoded to nominal labels before the entire pipeline predicts. A reserved unseen-category code decodes to an unknown label and therefore produces the pipeline's all-zero unknown-category block. No nominal code becomes a quantitative model predictor. Unknown development levels: `{prep['unknown_development_levels']}`.

Wrapper comparisons on every training and development row passed: maximum absolute errors `{prep['wrapper_max_abs_error']}`. Positive class index is {prep['positive_class_index']}. LIME receives both class scores; Kernel SHAP receives that same wrapper's positive-class output. The final-test features are not sent to explainers.''')
    add('Predictive performance', '''Frozen RandomForestClassifier: n_estimators=150, min_samples_leaf=3, random_state=2026, n_jobs=2; other defaults. No calibration, threshold learning, resampling, or hyperparameter tuning. Scores are **uncalibrated model scores**. The fixed threshold is 0.5.

Development predictive results (baseline always emits training prevalence):

''' + table(pd.DataFrame([{'model': 'Random Forest', **metrics['development']}, {'model': 'constant train prevalence', **metrics['baseline_development']}])) + '''ROC-AUC and Average Precision describe ranking; AP is not trapezoidal PR-AUC. Brier is probability-score error, not a standalone calibration guarantee. Confusion matrices use rows actual [0,1], columns predicted [0,1]. Final-test results were obtained only after development explanations and checks completed; see section 16.''')
    add('Selected development sessions', table(cases[['case','row_index','true_outcome','model_score','predicted_class','correct']]) + '''Cases were saved before explanations. High-scoring purchaser = maximum development score among purchases; low-scoring non-purchaser = minimum among non-purchases; near-threshold = minimum absolute distance from 0.5. Ties use the smallest original row index. Exact hashes and feature values are saved in `selected_cases.csv` and `selected_case_features.csv`. These exact rows are shared between methods.

“High-scoring” is relative to observed scores; it is not a claim of 90% confidence. The near-threshold row remains primary regardless of correctness or explanation appearance.''')
    add('Primary LIME explanation', f'''Near-threshold session, positive class, seed 0, 5,000 samples, top-5. Training-only quartile discretization, Euclidean distance, kernel width 3.0 (= 0.75√16), feature_selection='auto', default ridge surrogate. Bins and category mappings stay fixed across all runs. Bin names and bounds are recorded in `lime_bins.json`.

Black-box score **{primary_run.black_box_score:.6f}**; surrogate prediction **{primary_run.surrogate_prediction:.6f}**; absolute point error **{primary_run.absolute_point_error:.6f}**; weighted neighborhood R² **{primary_run.weighted_r2:.6f}**; intercept **{primary_run.intercept:.6f}**.

''' + table(primary_lime[['rank','feature','predicate','weight','direction']]) + '''These weights describe a weighted local surrogate in discretized predicate space. They are not causal effects, direct psychological explanations, or validated effects of changing a feature. Weighted R² is in-sample neighborhood fidelity; the point error checks fidelity at this particular row. Neither validates the black box.''')
    add('LIME stability', f'''Exactly 20 seeds (0–19), 5,000 samples each, top-5, one fixed near-threshold row. Only random seed changes. The primary run is seed 0, reused rather than rerun. Bin definitions were checked identical each time.

Across 190 seed pairs, mean top-5 Jaccard **{stability['pairwise_jaccard']['mean']:.4f}**, range **{stability['pairwise_jaccard']['min']:.4f}–{stability['pairwise_jaccard']['max']:.4f}**. Mean union-rank Spearman **{stability['pairwise_union_rank_spearman']['mean']:.4f}**. Rank comparisons use absolute weights, with omitted terms tied at rank 6 on the pairwise union. These are descriptive comparisons of truncated rankings, not full-model importance rankings.

Weighted R² mean **{stability['weighted_r2']['mean']:.4f}**, range **{stability['weighted_r2']['min']:.4f}–{stability['weighted_r2']['max']:.4f}**. Absolute point-error mean **{stability['absolute_point_error']['mean']:.4f}**, range **{stability['absolute_point_error']['min']:.4f}–{stability['absolute_point_error']['max']:.4f}**.

''' + table(freq[freq.selected_runs > 0][['feature','selected_runs','frequency','positive_runs','negative_runs','sign_consistency_given_selection_nonzero','mean_rank_selected','mean_weight_selected','std_weight_selected','min_weight_selected','max_weight_selected']]) + '''Selection frequency covers all 20 runs. Weight/rank summaries condition on selection. Sign consistency is the majority sign fraction among selected nonzero weights (tolerance 1e-8); an omitted feature is not a negative or zero finding. All 16 features, including never-selected ones, are in the CSV. Pairwise comparisons, fidelity distributions, and per-run predicates/weights are retained. No composite stability score is constructed.

VisitorType, ProductRelated_Duration, and BounceRates were selected in every seed; the remaining slots varied. Selected nonzero signs were consistent, but the surrogate explained only about 27% of weighted neighborhood score variation and missed the original score by about 0.20 on average. **These results do not support calling the local surrogate a trustworthy approximation of this prediction**, despite a stable core of selected terms.

![LIME stability and fidelity](case4_online_shopping/figures/lime_stability.png)''')
    budget_runs = lr[(lr.case == 'near_threshold') & (lr.seed == 0)].sort_values('samples')
    budget_comps = read('lime_budget_comparison.json')
    budget_features = lf[(lf.case == 'near_threshold') & (lf.seed == 0)].sort_values(['samples','rank'])
    add('LIME sample-budget sensitivity', '''Only perturbation sample count changes: 1,000 / 5,000 / 10,000, with seed 0, same model, row, bins, category mappings, kernel, and top-5. The 5,000 run is reused. Same seed does not imply nested identical neighborhoods because the library draws feature-wise arrays of different sizes.

''' + table(budget_runs[['samples','weighted_r2','surrogate_prediction','absolute_point_error']]) + table(pd.DataFrame(budget_comps)[['samples','shared_n','jaccard','union_rank_spearman','sign_agreement','sign_disagreements']]) + table(budget_features[['samples','rank','feature','weight','direction']]) + '''Comparisons use the 5,000-sample explanation as the anchor. Both other budgets replace one of its five terms, with no sign reversal among shared terms. At 10,000 samples the original-point error rises to about 0.224 rather than improving on 0.194 at 5,000. The core signal persists, but feature ranking and the fifth term are budget-sensitive; increasing budget does not resolve poor fidelity here. This is a single-seed budget sensitivity check, not a convergence proof or basis for retrospectively choosing the most attractive budget. More samples can reduce Monte Carlo variation without fixing a misspecified neighborhood or linear surrogate.''')
    add('Perturbation-support observations', '''The public prediction callback was instrumented during the standard 5,000-sample LIME run. It observed the actual decoded neighborhood sent to the black box, excluding the original first row (4,999 generated rows). No private neighborhood extraction, silent rounding, clipping, or repair was used.

''' + table(support) + '''Fractional-count checks use NumPy isclose with atol=1e-8 and its default rtol=1e-5; zero-count checks use absolute tolerance 1e-8. Rate limits and duration/count p99 are derived only from training. The duration/count flag is a tail heuristic, not proof of impossibility. Rare category means fewer than 20 training rows; rarity is not invalidity. Month/SpecialDay absence includes off-grid SpecialDay values and should not be mistaken for a pure calendar contradiction count. Rounded pair matching uses 10 decimal places; the separate grid check uses numerical tolerance.

Quartile discretization does not guarantee integer counts or preserve joint behavior. Unsupported combinations mean the surrogate may summarize behavior away from observed sessions. Eight examples are saved in `perturbation_examples.csv`, not a huge neighborhood dump. This is one neighborhood audit, not a population estimate across every seed.''')
    add('Primary SHAP explanation', f'''Model-agnostic **KernelExplainer**, identity link, scalar positive-class output, 16 original features, nsamples=2,080, l1_reg=0.0, seed 2026 reset for each call. Explicitly disabling L1 sparsification retains all 16 attribution players; no TreeSHAP substitution occurred. The output units are changes in the uncalibrated purchase score, not log odds or causal effects.

Background A: 50 actual training rows sampled without replacement, seed 2026. Composition and row identities are saved. Expected score **{primary_shap_run.base_value:.6f}**; explained score **{primary_shap_run.model_score:.6f}**; sum of contributions **{primary_shap_run.sum_contributions:.6f}**; absolute reconstruction error **{primary_shap_run.reconstruction_error:.3g}**.

''' + table(primary_shap[['rank','feature','weight','direction']]) + '''Strongest positive contributors:

''' + table(primary_shap[primary_shap.weight > 0].nlargest(3,'weight')[['feature','weight']]) + '''Strongest negative contributors:

''' + table(primary_shap[primary_shap.weight < 0].nsmallest(3,'weight')[['feature','weight']]) + '''Reconstruction tolerance is 1e-8 for every explanation. Additivity is checked numerically, but it does not show that finite-sample attributions have converged or that the masking distribution is realistic.''')
    add('LIME vs SHAP', f'''Same saved near-threshold session, same frozen pipeline, purchase output, and original-feature names. Top-5 overlap **{cross['shared_n']}/5**, Jaccard **{cross['jaccard']:.4f}**, union-rank Spearman **{cross['union_rank_spearman']:.4f}**. Shared features: {', '.join(cross['shared_features'])}. Sign-comparable shared features: {cross['sign_comparable_n']}; direction agreement **{cross['sign_agreement']}**; disagreements: `{cross['sign_disagreements']}`.

''' + table(pd.DataFrame({'LIME_top5': primary_lime.feature.tolist(), 'SHAP_top5': primary_shap.head(5).feature.tolist()})) + '''LIME approximates the model locally with a weighted surrogate; SHAP attributes a prediction relative to a reference/background expectation. A LIME predicate weight and SHAP's reference-relative contribution are different quantities. Sign comparison is therefore descriptive even for the same feature; raw coefficient magnitudes must not be compared as though they share an estimand. Disagreement alone does not prove either explanation wrong.''')
    background_meta = read('shap_backgrounds.json')
    add('SHAP background sensitivity', table(pd.DataFrame([{'background': k, **{p: v[p] for p in ['sample_seed','size','purchases','prevalence','mean_model_score','target_informed']}} for k,v in background_meta.items()])) + f'''Background A samples the overall training population; Background B samples 50 non-purchasing training sessions, seed 2026. B is explicitly **target-informed using training labels only**. It asks: “How does this session differ, in model-attribution terms, from typical non-purchasing training sessions?” Neither background uses development or final-test rows.

The model, explained row, preprocessing, wrapper, link, coalition budget/seed, regularization and sample size remain fixed. Baseline shift B − A = **{shift:.6f}**. Top-5 overlap **{bgo['shared_n']}/5** (Jaccard **{bgo['jaccard']:.4f}**); {int(bg.sign_change.sum())} direction changes under the 1e-8 sign convention. Every original-feature value, rank, signed change and reconstruction error is recorded below/in CSV.

''' + table(bg.sort_values('A_rank')) + table(sr[sr.case == 'near_threshold']) + '''The top-five set stays unchanged, with ExitRates and ProductRelated_Duration exchanging ranks 2 and 3. VisitorType remains the largest positive contribution. Three lower-ranked terms (Administrative, Month, OperatingSystems) change from small negative to small positive contributions. This is reference sensitivity in magnitudes and weak terms, not a wholesale reversal of the dominant explanation.

Changes reflect a different reference comparison, not SHAP failure. A small sampled “overall” background is only an approximation to the training population: A includes 9 purchases (18%) and 22 November sessions (44%), whereas B includes 8 November sessions (16%). Those realized composition differences can contribute to the comparison, so it is not an isolated causal contrast of purchase status. This single A/B sample comparison combines population choice with finite-background sampling variation; there is no additional background-resampling experiment in this frozen design. Some sign changes can concern near-zero contributions; inspect magnitude rather than counting signs alone.

![SHAP background sensitivity](case4_online_shopping/figures/shap_background_sensitivity.png)''')
    add('Correlated-feature attribution', table(csv('correlated_attributions.csv')) + '''Correlations are computed on training data. The table shows both members' SHAP values under each reference and their **signed sum**; it also records LIME weights if selected (blank means omitted, not zero attribution). Opposite signs can cancel, and contributions can move between related features when the reference changes.

Under A, BounceRates/ExitRates jointly contribute about +0.0659 and ProductRelated/duration +0.0501. Under B these become +0.0699 and +0.0741. Both members of each pair receive positive SHAP attribution; LIME selects both rates but only the product duration. Administrative count/duration have opposing signs under A and both positive signs under B. Informational count/duration nearly cancel.

This fixed-model comparison cannot isolate correlation as the cause of redistribution. Grouped signed sums are descriptive totals of the existing attribution game, not recomputed group-player SHAP values or causal credit. No predictors were removed and no model was refit. Independent masking can splice correlated values into unsupported combinations.''')
    comparison_rows = []
    for case in cases.case:
        run = lr[(lr.case == case) & (lr.seed == 0) & (lr.samples == 5000)].iloc[0]
        match = next(c for c in comparisons if c['case'] == case)
        comparison_rows.append({'case': case, 'weighted_r2': run.weighted_r2, 'point_error': run.absolute_point_error,
                                'top5_overlap': match['shared_n'], 'rank_spearman': match['union_rank_spearman'],
                                'shared_sign_agreement': match['sign_agreement']})
    add('Comparison across the three sessions', table(pd.DataFrame(comparison_rows)) + '''All three use LIME seed 0 / 5,000 samples / top-5 and SHAP Background A / 2,080 coalitions. The near-threshold explanation is reused. Separate numerical tables preserve predicates, all SHAP values, and exact rows.

All three have four-of-five method overlap and matching directions on those shared terms, but ranks differ: the high-scoring and low-scoring cases have approximately zero and negative union-rank agreement, respectively. For the high-scoring purchaser, LIME ranks VisitorType first with a negative weight, whereas SHAP ranks Month first positively and VisitorType falls outside its top five. LIME original-point error is about 0.304 for that purchaser versus 0.027 for the zero-score non-purchaser. This is a fidelity difference, not evidence of a stability difference.

The high/low cases have only one LIME seed each: **their explanation stability cannot be compared empirically with the 20-seed near-threshold result**. We can compare observed feature sets, method agreement, and surrogate fidelity, not claim one session type is inherently more stable. Prediction correctness, explanation stability, and fidelity remain distinct.

![Three session explanations](case4_online_shopping/figures/three_session_explanations.png)''')
    add('Final-test performance', table(pd.DataFrame([{'model': 'Random Forest', **metrics['final_test']}, {'model': 'constant train prevalence', **metrics['baseline_final_test']}])) + '''Final predictions were made only after all 24 LIME runs, four Kernel SHAP explanations, background comparisons, and numerical checks completed successfully. The model was not changed afterward. The final set supplied no explanation examples, background rows, preprocessing estimates, or tuning criteria. `development_experiments_complete.json` records the pre-evaluation stage and `validation.json` records the checks.

Ranking exceeds the trivial baseline (AP 0.3314 versus prevalence 0.1545), but recall at 0.5 is only 3.94%: 15 of 381 purchases are detected, and 366 are missed. This is useful signal for an explanation audit, not evidence of a useful purchase-detection policy at that threshold. The human-frozen model and threshold remain unchanged.

Metrics describe this one offline split without confidence intervals. The fixed 0.5 threshold is illustrative, not an optimized assistance policy. No calibration claim follows from Brier alone. This fresh split shares the previously audited dataset, so it is untouched during this experiment but is not independent of all earlier dataset-level exploration.''')
    add('Findings suitable for teaching', f'''1. A concrete prediction can be explained with both methods, but neither attribution is ground truth about the shopper.
2. The seed experiment quantifies feature-selection variation (mean top-5 Jaccard {stability['pairwise_jaccard']['mean']:.3f}) while separately checking fidelity. A repeatable explanation can still fit its neighborhood or its original point poorly.
3. The sample-budget comparison separates one computational assumption from seed variation. Larger sample count does not repair unsupported behavioral combinations.
4. LIME and SHAP share {cross['shared_n']} of five leading features for the primary row; different comparison questions can produce disagreement without an implementation error.
5. Changing only the reference population shifts the SHAP baseline by {shift:+.4f} and can change individual attributions while reconstructing the same model score.

Students should notice the selected row's true outcome, score and correctness, inspect the fidelity/support diagnostics, and explain which assumptions each method uses before declaring an explanation trustworthy.''')
    add('Important limitations', '''Uncalibrated scores are not demonstrated calibrated purchase probabilities. Prediction is not explanation; explanation is not ground truth; attribution is not causality or shopper psychology. Stable explanations need not accompany correct predictions. Local surrogate fidelity does not establish black-box validity. Model-level associations do not validate effective interventions or real-world recourse. Retrospective performance does not establish real-time deployment validity.

LIME's discretized independent perturbations and Kernel SHAP's marginal masking can break feature dependence. Twenty seeds quantify one row/setting, not universal reliability. One-seed budget sensitivity and one sample per background do not establish convergence or sampling uncertainty. There is no repeated stability audit of the other two sessions. No adaptive repair, conditional SHAP, new model, or extra experiment was added after inspecting results.

Kernel SHAP's numerical additivity is not an accuracy certificate for each attribution. Small changes near zero deserve restraint. Unknown category handling keeps prediction calls valid but does not create training evidence for unseen levels. Numeric category codes retain readable code labels because the source lacks semantic codebooks.

Human interpretation remains necessary when deciding which findings to emphasize, whether the surrogate fidelity is educationally acceptable, and how strongly to describe disagreement or reference sensitivity. The frozen design and offline framing were not redesigned.''')
    nonzero_support = support[support.flagged_n > 0][['check','flagged_n','fraction']]
    add('Unexpected findings', '''Observed departures worth discussing, without adding experiments:

''' + table(nonzero_support) + f'''The primary surrogate predicts {primary_run.surrogate_prediction:.4f} for a black-box score of {primary_run.black_box_score:.4f}; its absolute error is {primary_run.absolute_point_error:.4f}. Reference B changes the baseline by {shift:+.4f}; that change is a consequence to interpret, not a failed reconstruction.

Python 3.13 compatibility succeeded with lime 0.2.0.1 and shap 0.52.0. No installed package was upgraded. The existing environment uses SciPy {environment['packages'].get('scipy')}, while the pre-existing requirements file pins 1.18.1; that unrelated discrepancy was not edited. The actual execution environment is fully recorded in `environment.json`.

### Reproduction and validation

Run from the repository root:

```sh
.venv/bin/python run_case4_online_shopping_experiments.py
.venv/bin/python run_case4_online_shopping_experiments.py --validate-only
.venv/bin/python run_case4_online_shopping_experiments.py --report-only
```

The default command regenerates experiment artifacts using fixed settings; it does not install packages or write raw data. Floating-point summation with two forest workers can differ at machine precision; numerical equivalence, rather than bitwise identity, is expected for recomputed scores. The latter two commands validate saved artifacts and render reports/figures without model fitting or new evaluation. `installation.json` is the one-time environment-change record, not an experiment product.

Other commands run: read-only `cat`, `rg`, `sed`, `ls` inspection; `.venv/bin/python -m pip list --format=json`; pip dry-run and installation of `lime==0.2.0.1 shap==0.52.0` with JSON installation reports; import smoke tests; `.venv/bin/python -m pip check`; `.venv/bin/python -m py_compile run_case4_online_shopping_experiments.py src/case4_reporting.py`; and checksum/manifest checks. The first sandbox PyPI request failed DNS resolution; the approved network retry and installation succeeded. Required additions, with no changes to existing installed packages, are recorded in `installation.json` and pinned in requirements.txt.

 LIME/SHAP source/API behavior was checked against [LIME documentation](https://lime-ml.readthedocs.io/en/latest/lime.html) and [KernelExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html); installed versions and explicit parameters determine this run.

Validation: raw SHA-256 unchanged; all rows preserved; identical final vectors partition-disjoint; preprocessing/transport/bins fit on train only; all selected rows development-only; both backgrounds training-only; positive-class index correct; both methods use the verified wrapper; maximum SHAP reconstruction error {validation['max_shap_reconstruction_error']:.3g}; fixed seeds/settings saved. Saved metrics are recomputed from saved scores during validation, and SHAP reconstruction is checked again from saved contributions.

Output inventory and content hashes are saved in `artifact_manifest.json`. No model object or large perturbation array is saved; the fitted pipeline is deterministically regenerated. Only the new research code, dependency additions, and Case 4 results were written. No final teaching notebook or completed case was modified.''')
    report = '# Case 4 experiment results: Why Does the Model Think This Shopper Will Buy?\n\n' + ''.join(sections)
    (root / 'outputs/case4_online_shopping_experiment_results.md').write_text(report)
    manifest = {}
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name != 'artifact_manifest.json':
            manifest[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for path in [root / 'run_case4_online_shopping_experiments.py', root / 'src/case4_reporting.py', root / 'outputs/case4_online_shopping_experiment_results.md']:
        manifest[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
