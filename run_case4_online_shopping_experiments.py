"""Research only: execute the human-frozen Case 4 design; no teaching notebook.

Run: .venv/bin/python run_case4_online_shopping_experiments.py
Render existing results without experiments: add --report-only
Validate saved artifacts without fitting/evaluating: add --validate-only
All outputs are restricted to the Case 4 output directory and results report.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import itertools
import json
import os
from pathlib import Path
import platform
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'case4-matplotlib'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lime.lime_tabular import LimeTabularExplainer
import shap

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/case4_online_shopping'
RAW = ROOT / 'Data/raw/online_shoppers_intention.csv'
RAW_SHA = 'b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9'
NUM = ['Administrative', 'Administrative_Duration', 'Informational',
       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
       'BounceRates', 'ExitRates', 'SpecialDay']
CAT = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
FEATURES = NUM + CAT
PAIRS = [('BounceRates', 'ExitRates'), ('ProductRelated', 'ProductRelated_Duration'),
         ('Administrative', 'Administrative_Duration'), ('Informational', 'Informational_Duration')]
CONFIG = {
    'outer_seed': 2026, 'outer_folds': 5, 'inner_seed': 2027, 'inner_folds': 4,
    'fold_selection': 'first yielded fold, no search',
    'model': {'n_estimators': 150, 'min_samples_leaf': 3, 'random_state': 2026, 'n_jobs': 2},
    'lime_seeds': list(range(20)), 'lime_primary_seed': 0, 'lime_top_k': 5,
    'lime_samples': 5000, 'lime_budgets': [1000, 5000, 10000],
    'lime_discretizer': 'quartile', 'lime_kernel_width': 3.0,
    'lime_feature_selection': 'auto', 'lime_distance': 'euclidean',
    'shap_algorithm': 'KernelExplainer', 'shap_link': 'identity',
    'shap_nsamples': 2080, 'shap_l1_reg': 0.0, 'shap_seed': 2026,
    'background_size': 50, 'background_seed': 2026,
    'threshold': 0.5, 'sign_tolerance': 1e-8, 'reconstruction_tolerance': 1e-8,
    'rank_convention': 'absolute magnitude; LIME omissions tied at rank 6 in union comparisons',
}


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(name, value):
    (OUT / name).write_text(json.dumps(native(value), indent=2, allow_nan=False) + '\n')


def save_csv(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / name, index=False, float_format='%.12g')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_hashes(frame):
    # SHA-256 over a canonical JSON list in the declared 16-feature order.
    return pd.Series([hashlib.sha256(json.dumps(native(row), separators=(',', ':'),
                      ensure_ascii=True).encode()).hexdigest()
                      for row in frame.itertuples(index=False, name=None)], index=frame.index)


def metrics(y, p):
    pred = np.asarray(p) >= CONFIG['threshold']
    return {'roc_auc': roc_auc_score(y, p), 'average_precision': average_precision_score(y, p),
            'brier': brier_score_loss(y, p), 'precision_at_0.5': precision_score(y, pred, zero_division=0),
            'recall_at_0.5': recall_score(y, pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def direction(weight):
    return 'positive' if weight > CONFIG['sign_tolerance'] else (
        'negative' if weight < -CONFIG['sign_tolerance'] else 'near_zero')


class Transport:
    """Train-fitted categorical transport; decode before the one-hot pipeline.

    Unseen levels map to a reserved transport code and decode to an unknown token;
    the predictive OneHotEncoder then emits the same all-zero unknown block.
    Numeric perturbations are intentionally never clipped, rounded, or repaired.
    """
    def __init__(self, train, pipeline):
        self.pipeline = pipeline
        self.levels = {c: sorted(train[c].unique().tolist()) for c in CAT}
        self.unknown = '__UNSEEN_IN_TRAIN__'
        assert all(self.unknown not in v for v in self.levels.values())
        self.names = {FEATURES.index(c): values + [self.unknown] for c, values in self.levels.items()}
        self.positive_index = list(pipeline.classes_).index(1)
        assert list(pipeline.classes_) == [0, 1]

    def encode(self, frame):
        result = frame[NUM].to_numpy(dtype=float)
        for c in CAT:
            mapping = {value: i for i, value in enumerate(self.levels[c])}
            code = frame[c].map(mapping).fillna(len(mapping)).to_numpy(dtype=float)
            result = np.column_stack([result, code])
        return result

    def decode(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        assert values.shape[1] == len(FEATURES) and np.isfinite(values).all()
        result = pd.DataFrame(values[:, :len(NUM)], columns=NUM)
        for i, c in enumerate(CAT, start=len(NUM)):
            codes = values[:, i]
            assert np.allclose(codes, np.rint(codes), atol=1e-10, rtol=0), 'Nonintegral category code'
            codes = np.rint(codes).astype(int)
            labels = self.levels[c] + [self.unknown]
            assert codes.min() >= 0 and codes.max() < len(labels)
            result[c] = np.asarray(labels)[codes]
        return result[FEATURES]

    def proba(self, values):
        # Chunking controls Kernel SHAP memory without changing model semantics.
        values = np.atleast_2d(values)
        return np.vstack([self.pipeline.predict_proba(self.decode(values[i:i + 10000]))
                          for i in range(0, len(values), 10000)])

    def positive(self, values):
        return self.proba(values)[:, self.positive_index]


def support_audit(values, train, transport):
    """Summarize actual inverse samples observed at the public prediction callback."""
    generated = transport.decode(values[1:])  # exclude original row, kept at position zero by LIME
    flags = {}
    for c in ['Administrative', 'Informational', 'ProductRelated']:
        flags[c + ': fractional count'] = ~np.isclose(generated[c], np.rint(generated[c]), atol=1e-8)
        flags[c + ': negative count'] = generated[c] < -1e-8
        flags[c + ': zero count, positive duration'] = (generated[c].abs() < 1e-8) & (generated[c + '_Duration'] > 1e-8)
        valid = train[c] > 0
        upper = (train.loc[valid, c + '_Duration'] / train.loc[valid, c]).quantile(.99)
        # A descriptive tail flag; not a declaration that a session is impossible.
        flags[c + ': duration/count above train p99'] = (generated[c] > 1e-8) & (generated[c + '_Duration'] / generated[c].clip(lower=1e-8) > upper)
    for c in ['BounceRates', 'ExitRates']:
        flags[c + ': outside train range'] = (generated[c] < train[c].min() - 1e-8) | (generated[c] > train[c].max() + 1e-8)
    observed = set(zip(train.Month, train.SpecialDay.round(10)))
    flags['Month/SpecialDay: pair absent from train'] = np.array([(m, round(d, 10)) not in observed for m, d in zip(generated.Month, generated.SpecialDay)])
    flags['SpecialDay: off observed grid'] = ~np.isclose(generated.SpecialDay.to_numpy()[:, None], train.SpecialDay.unique()[None, :], atol=1e-8).any(axis=1)
    rare = np.zeros(len(generated), dtype=bool)
    for c in CAT:
        rare |= generated[c].map(train[c].value_counts()).fillna(0).to_numpy() < 20
    flags['Any category with <20 training rows'] = rare
    records = [{'check': k, 'flagged_n': int(np.sum(v)), 'generated_n': len(generated),
                'fraction': float(np.mean(v))} for k, v in flags.items()]
    # Retain only eight diagnostic examples, not the generated neighborhood.
    examples = []
    for i in np.flatnonzero(np.any(np.column_stack(list(flags.values())), axis=1))[:8]:
        examples.append({'generated_row': int(i + 1), **generated.iloc[i].to_dict(),
                         'flags': '; '.join(k for k, v in flags.items() if np.asarray(v)[i])})
    return records, examples


def overlap(left, right):
    """Compare top-five sets/ranks; sign agreement only for shared nonzero features."""
    a, b = set(left), set(right)
    union = sorted(a | b)
    ra = {k: i + 1 for i, k in enumerate(left)}
    rb = {k: i + 1 for i, k in enumerate(right)}
    av, bv = [ra.get(k, 6) for k in union], [rb.get(k, 6) for k in union]
    rho = float(spearmanr(av, bv).statistic) if len(set(av)) > 1 and len(set(bv)) > 1 else None
    shared = sorted(a & b)
    nonzero = [k for k in shared if direction(left[k]) != 'near_zero' and direction(right[k]) != 'near_zero']
    return {'shared_n': len(shared), 'jaccard': len(shared) / len(union), 'union_rank_spearman': rho,
            'shared_features': shared, 'sign_comparable_n': len(nonzero),
            'sign_agreement': np.mean([direction(left[k]) == direction(right[k]) for k in nonzero]).item() if nonzero else None,
            'sign_disagreements': [k for k in nonzero if direction(left[k]) != direction(right[k])]}


def top_map(rows):
    return {r['feature']: r['weight'] for r in sorted(rows, key=lambda r: r['rank'])[:5]}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    assert digest(RAW) == RAW_SHA, 'Raw data changed'
    save_json('config.json', CONFIG)
    versions = {d.metadata['Name']: d.version for d in metadata.distributions()}
    save_json('environment.json', {'python': platform.python_version(), 'packages': versions,
              'note': 'SciPy 1.16.3 pre-existed; repository requirements lists 1.18.1. No existing package upgraded.'})
    frame = pd.read_csv(RAW)
    X, y = frame[FEATURES].copy(), frame.Revenue.astype(int)
    for c in CAT:
        X[c] = X[c].astype(str)  # nominal labels, including originally integer codes
    assert not X.isna().any().any() and len(X) == 12330
    groups = row_hashes(X)
    assert groups.nunique() == X.drop_duplicates().shape[0], 'Hash collision or inconsistent canonicalization'
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    train_dev, test = next(outer.split(X, y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=2027)
    tr, dv = next(inner.split(X.iloc[train_dev], y.iloc[train_dev], groups.iloc[train_dev]))
    train, dev = train_dev[tr], train_dev[dv]
    partitions = {'train': train, 'development': dev, 'final_test': test}
    assignments = pd.DataFrame({'row_index': X.index, 'csv_line': X.index + 2,
                                'feature_sha256': groups, 'Revenue': y, 'partition': ''})
    for name, idx in partitions.items():
        assignments.loc[idx, 'partition'] = name
    assert assignments.groupby('feature_sha256').partition.nunique().max() == 1
    assert sum(map(len, partitions.values())) == len(X)
    save_csv('split_assignments.csv', assignments)
    split_summary = [{'partition': name, 'rows': len(idx), 'purchases': int(y.iloc[idx].sum()),
                      'prevalence': float(y.iloc[idx].mean()), 'unique_groups': groups.iloc[idx].nunique()}
                     for name, idx in partitions.items()]
    save_json('split_metadata.json', {'summary': split_summary, 'raw_sha256': RAW_SHA,
              'hash_definition': 'SHA256 canonical JSON list, NUM + CAT order, nominal columns strings',
              'features': FEATURES, 'extra_duplicate_final_vectors': int(X.duplicated().sum()),
              'mixed_label_groups': int(pd.DataFrame({'g': groups, 'y': y}).groupby('g').y.nunique().gt(1).sum())})
    print('Partitions:', split_summary, flush=True)
    pipeline = Pipeline([('preprocessing', ColumnTransformer([
        ('numeric', 'passthrough', NUM),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT)])),
        ('model', RandomForestClassifier(**CONFIG['model']))])
    pipeline.fit(X.iloc[train], y.iloc[train])
    transport = Transport(X.iloc[train], pipeline)
    encoded_train = transport.encode(X.iloc[train])
    encoded_dev = transport.encode(X.iloc[dev])
    wrapper_errors = {}
    for name, idx, encoded in [('train', train, encoded_train), ('development', dev, encoded_dev)]:
        direct = pipeline.predict_proba(X.iloc[idx])
        wrapper_errors[name] = float(np.max(np.abs(direct - transport.proba(encoded))))
        assert wrapper_errors[name] < 1e-12
    encoder = pipeline.named_steps['preprocessing'].named_transformers_['categorical']
    assert all(list(levels) == transport.levels[c] for c, levels in zip(CAT, encoder.categories_))
    names = pipeline.named_steps['preprocessing'].get_feature_names_out().tolist()
    mapping, position = {}, 0
    for c in NUM:
        mapping[c] = [names[position]]; position += 1
    for c, levels in zip(CAT, encoder.categories_):
        mapping[c] = names[position:position + len(levels)]; position += len(levels)
    assert position == len(names)
    save_json('preprocessing.json', {'original_features': FEATURES, 'encoded_names': names,
              'original_to_encoded': mapping, 'transport_categories': transport.names,
              'category_levels_train': transport.levels, 'positive_class_index': transport.positive_index,
              'wrapper_max_abs_error': wrapper_errors,
              'unknown_development_levels': {c: sorted(set(X.iloc[dev][c]) - set(transport.levels[c])) for c in CAT}})
    dev_scores = pipeline.predict_proba(X.iloc[dev])[:, transport.positive_index]
    predictions = pd.DataFrame({'row_index': dev, 'Revenue': y.iloc[dev].to_numpy(), 'score': dev_scores})
    predictions['distance_to_0.5'] = abs(predictions.score - .5)
    # Deterministic tie break: smallest original CSV row index.
    selected = {
        'near_threshold': int(predictions.sort_values(['distance_to_0.5', 'row_index']).iloc[0].row_index),
        'high_scoring_purchaser': int(predictions.loc[predictions.Revenue == 1].sort_values(['score', 'row_index'], ascending=[False, True]).iloc[0].row_index),
        'low_scoring_non_purchaser': int(predictions.loc[predictions.Revenue == 0].sort_values(['score', 'row_index']).iloc[0].row_index),
    }
    case_rows = []
    for case, idx in selected.items():
        assert idx in set(dev)
        score = float(predictions.set_index('row_index').loc[idx, 'score'])
        case_rows.append({'case': case, 'row_index': idx, 'csv_line': idx + 2,
                          'feature_sha256': groups.loc[idx], 'true_outcome': int(y.loc[idx]),
                          'model_score': score, 'predicted_class': int(score >= .5),
                          'correct': bool((score >= .5) == y.loc[idx])})
    save_csv('selected_cases.csv', case_rows)
    save_csv('selected_case_features.csv', [{'case': c, 'row_index': i, **X.loc[i].to_dict()} for c, i in selected.items()])
    save_csv('development_predictions.csv', predictions)
    predictive = {'development': metrics(y.iloc[dev], dev_scores),
                  'baseline_development': metrics(y.iloc[dev], np.repeat(y.iloc[train].mean(), len(dev)))}
    print('Development cases selected before explanations:', case_rows, flush=True)

    # Standard primary run doubles as seed zero of the frozen stability experiment.
    lime_features, lime_runs, bins_reference = [], [], None
    support_records, support_examples = [], []

    def explain_lime(case, seed, samples, experiment, inspect_support=False):
        nonlocal bins_reference, support_records, support_examples
        explainer = LimeTabularExplainer(encoded_train, mode='classification',
            feature_names=FEATURES, categorical_features=list(range(len(NUM), len(FEATURES))),
            categorical_names=transport.names, class_names=['no purchase', 'purchase'],
            discretize_continuous=True, discretizer='quartile', kernel_width=3.0,
            feature_selection='auto', sample_around_instance=False, random_state=seed)
        bins = {FEATURES[k]: {'predicates': v, 'mins': explainer.discretizer.mins[k],
                             'maxs': explainer.discretizer.maxs[k]} for k, v in explainer.discretizer.names.items()}
        if bins_reference is None:
            bins_reference = native(bins); save_json('lime_bins.json', bins_reference)
        assert bins_reference == native(bins), 'Discretization changed between runs'
        idx = selected[case]
        point = transport.encode(X.loc[[idx]])[0]

        def callback(values):
            nonlocal support_records, support_examples
            if inspect_support:
                assert np.allclose(values[0], point)
                support_records, support_examples = support_audit(values, X.iloc[train], transport)
            return transport.proba(values)

        exp = explainer.explain_instance(point, callback, labels=(1,), top_labels=None,
                                        num_features=5, num_samples=samples, distance_metric='euclidean')
        score = float(transport.positive(point)[0])
        local = float(np.asarray(exp.local_pred).ravel()[0])
        record = {'case': case, 'experiment': experiment, 'seed': seed, 'samples': samples,
                  'weighted_r2': float(exp.score), 'surrogate_prediction': local,
                  'black_box_score': score, 'absolute_point_error': abs(local - score),
                  'intercept': float(exp.intercept[1])}
        pairs = exp.local_exp[1]
        # map_exp_ids returns (predicate, weight); zip retains original feature IDs.
        rows = []
        for rank, ((feature_idx, weight), (predicate, _)) in enumerate(zip(pairs, exp.domain_mapper.map_exp_ids(pairs)), start=1):
            rows.append({'case': case, 'experiment': experiment, 'seed': seed, 'samples': samples,
                         'feature': FEATURES[feature_idx], 'predicate': predicate, 'rank': rank,
                         'weight': float(weight), 'direction': direction(weight)})
        assert len(rows) == 5 and abs(exp.intercept[1] + sum(v for _, v in pairs) - local) < 1e-8
        lime_runs.append(record); lime_features.extend(rows)
        return rows

    for seed in CONFIG['lime_seeds']:
        explain_lime('near_threshold', seed, 5000, 'seed_stability', inspect_support=seed == 0)
        print(f'LIME seed {seed + 1}/20 complete', flush=True)
    for budget in [1000, 10000]:
        explain_lime('near_threshold', 0, budget, 'sample_budget')
    for case in ['high_scoring_purchaser', 'low_scoring_non_purchaser']:
        explain_lime(case, 0, 5000, 'session_comparison')
    save_csv('lime_runs.csv', lime_runs)
    save_csv('lime_features.csv', lime_features)
    save_csv('perturbation_support.csv', support_records)
    save_csv('perturbation_examples.csv', support_examples)
    lf, lr = pd.DataFrame(lime_features), pd.DataFrame(lime_runs)
    stability_features = lf[lf.experiment == 'seed_stability']
    stability_runs = lr[lr.experiment == 'seed_stability']
    feature_summary = []
    for feature in FEATURES:
        sub = stability_features[stability_features.feature == feature]
        n = len(sub)
        nonzero = sub[sub.direction != 'near_zero']
        sign_counts = nonzero.direction.value_counts()
        feature_summary.append({'feature': feature, 'selected_runs': n, 'frequency': n / 20,
            'positive_runs': int((sub.direction == 'positive').sum()), 'negative_runs': int((sub.direction == 'negative').sum()),
            'sign_consistency_given_selection_nonzero': float(sign_counts.max() / len(nonzero)) if len(nonzero) else None,
            'mean_rank_selected': float(sub['rank'].mean()) if n else None,
            'mean_weight_selected': float(sub.weight.mean()) if n else None,
            'std_weight_selected': float(sub.weight.std(ddof=1)) if n > 1 else None,
            'min_weight_selected': float(sub.weight.min()) if n else None,
            'max_weight_selected': float(sub.weight.max()) if n else None})
    pairwise = []
    for a, b in itertools.combinations(range(20), 2):
        result = overlap(top_map(stability_features[stability_features.seed == a].to_dict('records')),
                         top_map(stability_features[stability_features.seed == b].to_dict('records')))
        pairwise.append({'seed_a': a, 'seed_b': b, **result})
    save_csv('lime_selection_summary.csv', feature_summary)
    save_json('lime_pairwise_stability.json', pairwise)
    summary = {'pairwise_jaccard': pd.Series([p['jaccard'] for p in pairwise]).describe().to_dict(),
               'pairwise_union_rank_spearman': pd.Series([p['union_rank_spearman'] for p in pairwise]).describe().to_dict(),
               'weighted_r2': stability_runs.weighted_r2.describe().to_dict(),
               'absolute_point_error': stability_runs.absolute_point_error.describe().to_dict()}
    save_json('lime_stability_summary.json', summary)
    budget_comparisons = []
    primary_lime = top_map(stability_features[stability_features.seed == 0].to_dict('records'))
    for budget in [1000, 5000, 10000]:
        rows = lf[(lf.case == 'near_threshold') & (lf.seed == 0) & (lf.samples == budget)]
        budget_comparisons.append({'samples': budget, **overlap(primary_lime, top_map(rows.to_dict('records')))})
    save_json('lime_budget_comparison.json', budget_comparisons)

    # Both reference samples are actual TRAIN rows; B intentionally uses TRAIN labels.
    backgrounds = {
        'A_overall_train': X.iloc[train].sample(n=50, random_state=2026).index.to_numpy(),
        'B_nonpurchase_train': X.loc[train[y.iloc[train].to_numpy() == 0]].sample(n=50, random_state=2026).index.to_numpy(),
    }
    background_metadata, background_rows = {}, []
    for name, indices in backgrounds.items():
        assert set(indices).issubset(set(train)) and not set(indices) & set(test)
        background_metadata[name] = {'sample_seed': 2026, 'size': len(indices),
            'purchases': int(y.loc[indices].sum()), 'prevalence': float(y.loc[indices].mean()),
            'mean_model_score': float(transport.positive(transport.encode(X.loc[indices])).mean()),
            'target_informed': name.startswith('B'),
            'month_composition': X.loc[indices, 'Month'].value_counts().to_dict(),
            'visitor_composition': X.loc[indices, 'VisitorType'].value_counts().to_dict()}
        background_rows.extend([{'background': name, 'row_index': int(i), 'feature_sha256': groups.loc[i],
                                 'Revenue': int(y.loc[i])} for i in indices])
    save_json('shap_backgrounds.json', background_metadata)
    save_csv('shap_background_rows.csv', background_rows)
    shap_rows, shap_runs = [], []
    for bg, indices in backgrounds.items():
        background = transport.encode(X.loc[indices])
        explainer = shap.KernelExplainer(transport.positive, background, feature_names=FEATURES, link='identity')
        cases = list(selected) if bg == 'A_overall_train' else ['near_threshold']
        for case in cases:
            # Fixed coalition sampling RNG for each call, including A vs B.
            np.random.seed(CONFIG['shap_seed'])
            point = transport.encode(X.loc[[selected[case]]])
            values = np.asarray(explainer.shap_values(point, nsamples=2080, l1_reg=0.0, silent=True)).reshape(-1)
            assert len(values) == len(FEATURES)
            base = float(np.asarray(explainer.expected_value).ravel()[0])
            score = float(transport.positive(point)[0])
            error = abs(base + values.sum() - score)
            assert error < CONFIG['reconstruction_tolerance']
            assert abs(base - background_metadata[bg]['mean_model_score']) < 1e-12
            order = np.argsort(-np.abs(values), kind='stable')
            for rank, j in enumerate(order, 1):
                shap_rows.append({'case': case, 'background': bg, 'feature': FEATURES[j], 'rank': rank,
                                  'weight': float(values[j]), 'direction': direction(values[j])})
            shap_runs.append({'case': case, 'background': bg, 'base_value': base, 'model_score': score,
                              'sum_contributions': float(values.sum()), 'reconstruction_error': error})
            print(f'Kernel SHAP {case} / {bg}: reconstruction error {error:.3g}', flush=True)
    save_csv('shap_values.csv', shap_rows)
    save_csv('shap_runs.csv', shap_runs)
    sf = pd.DataFrame(shap_rows)
    comparisons = []
    for case in selected:
        l = top_map(lf[(lf.case == case) & (lf.seed == 0) & (lf.samples == 5000)].to_dict('records'))
        s = top_map(sf[(sf.case == case) & (sf.background == 'A_overall_train')].to_dict('records'))
        comparisons.append({'case': case, **overlap(l, s)})
    save_json('lime_vs_shap.json', comparisons)
    a = sf[(sf.case == 'near_threshold') & (sf.background == 'A_overall_train')].set_index('feature')
    b = sf[(sf.case == 'near_threshold') & (sf.background == 'B_nonpurchase_train')].set_index('feature')
    bg_comparison = [{'feature': f, 'A_value': a.loc[f, 'weight'], 'B_value': b.loc[f, 'weight'],
                     'delta_B_minus_A': b.loc[f, 'weight'] - a.loc[f, 'weight'],
                     'A_rank': a.loc[f, 'rank'], 'B_rank': b.loc[f, 'rank'],
                     'sign_change': a.loc[f, 'direction'] != b.loc[f, 'direction']} for f in FEATURES]
    save_csv('shap_background_comparison.csv', bg_comparison)
    save_json('shap_background_overlap.json', overlap(top_map(a.reset_index().to_dict('records')), top_map(b.reset_index().to_dict('records'))))
    correlated = []
    for first, second in PAIRS:
        for bg in backgrounds:
            sub = sf[(sf.case == 'near_threshold') & (sf.background == bg)].set_index('feature')
            correlated.append({'pair': first + ' / ' + second, 'background': bg,
                'pearson_train': X.iloc[train][[first, second]].corr().iloc[0, 1],
                'spearman_train': X.iloc[train][[first, second]].corr(method='spearman').iloc[0, 1],
                'first_SHAP': sub.loc[first, 'weight'], 'second_SHAP': sub.loc[second, 'weight'],
                'grouped_signed_SHAP': sub.loc[first, 'weight'] + sub.loc[second, 'weight'],
                'first_LIME_if_selected': primary_lime.get(first),
                'second_LIME_if_selected': primary_lime.get(second)})
    save_csv('correlated_attributions.csv', correlated)
    assert digest(RAW) == RAW_SHA
    assert len(lime_runs) == 24 and len(shap_runs) == 4
    save_json('development_experiments_complete.json', {'success': True, 'lime_runs': 24,
        'shap_runs': 4, 'final_test_evaluated_at_this_stage': False})
    # Final-test firewall: no test predictions, case selection, or references above.
    # Dataset-wide hashing/stratification and partition prevalence are split bookkeeping only.
    print('All development experiments passed. Evaluating the frozen final-test partition now.', flush=True)
    p_test = pipeline.predict_proba(X.iloc[test])[:, transport.positive_index]
    predictive['final_test'] = metrics(y.iloc[test], p_test)
    predictive['baseline_final_test'] = metrics(y.iloc[test], np.repeat(y.iloc[train].mean(), len(test)))
    save_json('predictive_metrics.json', predictive)
    save_csv('final_test_predictions.csv', pd.DataFrame({'row_index': test, 'Revenue': y.iloc[test].to_numpy(), 'score': p_test}))
    save_json('validation.json', {'raw_sha256_matches': digest(RAW) == RAW_SHA,
        'all_rows_preserved': len(assignments) == len(frame), 'group_disjoint': True,
        'preprocessing_and_transport_fit_partition': 'train', 'explanation_cases_partition': 'development',
        'background_partition': 'train', 'positive_class_index': transport.positive_index,
        'wrapper_max_abs_error': wrapper_errors, 'bins_identical_across_runs': True,
        'max_shap_reconstruction_error': max(r['reconstruction_error'] for r in shap_runs),
        'development_experiments_finished_before_final_test_prediction': True,
        'model_changed_after_final_test': False})
    print('Final test:', predictive['final_test'], flush=True)


def validate_saved():
    assert digest(RAW) == RAW_SHA
    split = pd.read_csv(OUT / 'split_assignments.csv')
    assert len(split) == 12330 and split.row_index.nunique() == 12330
    assert split.groupby('feature_sha256').partition.nunique().max() == 1
    raw = pd.read_csv(RAW)
    features = raw[FEATURES].copy()
    for c in CAT:
        features[c] = features[c].astype(str)
    hashes = row_hashes(features)
    assert hashes.tolist() == split.feature_sha256.tolist()
    assert raw.Revenue.astype(int).tolist() == split.Revenue.tolist()
    # Recreate only the split (no fitting or new final-test prediction).
    td, te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026).split(
        features, raw.Revenue, hashes))
    tr, dv = next(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=2027).split(
        features.iloc[td], raw.Revenue.iloc[td], hashes.iloc[td]))
    for name, ids in [('train', td[tr]), ('development', td[dv]), ('final_test', te)]:
        assert set(ids) == set(split.loc[split.partition == name, 'row_index'])
    partition = split.set_index('row_index').partition
    cases = pd.read_csv(OUT / 'selected_cases.csv')
    assert cases.row_index.map(partition).eq('development').all()
    assert cases.feature_sha256.tolist() == cases.row_index.map(hashes).tolist()
    predictions = pd.read_csv(OUT / 'development_predictions.csv')
    expected_cases = {
        'near_threshold': int(predictions.sort_values(['distance_to_0.5', 'row_index']).iloc[0].row_index),
        'high_scoring_purchaser': int(predictions.loc[predictions.Revenue == 1].sort_values(
            ['score', 'row_index'], ascending=[False, True]).iloc[0].row_index),
        'low_scoring_non_purchaser': int(predictions.loc[predictions.Revenue == 0].sort_values(
            ['score', 'row_index']).iloc[0].row_index),
    }
    assert cases.set_index('case').row_index.to_dict() == expected_cases
    prep = json.loads((OUT / 'preprocessing.json').read_text())
    train_ids = split.loc[split.partition == 'train', 'row_index']
    assert prep['original_features'] == FEATURES and 'PageValues' not in FEATURES
    assert all(prep['category_levels_train'][c] == sorted(features.loc[train_ids, c].unique()) for c in CAT)
    backgrounds = pd.read_csv(OUT / 'shap_background_rows.csv')
    assert backgrounds.row_index.map(partition).eq('train').all()
    assert backgrounds.groupby('background').size().eq(50).all()
    assert backgrounds.feature_sha256.tolist() == backgrounds.row_index.map(hashes).tolist()
    assert backgrounds.loc[backgrounds.background == 'B_nonpurchase_train', 'Revenue'].eq(0).all()
    runs = pd.read_csv(OUT / 'lime_runs.csv')
    assert len(runs) == 24 and set(runs[runs.experiment == 'seed_stability'].seed) == set(range(20))
    sh = pd.read_csv(OUT / 'shap_runs.csv')
    assert len(sh) == 4 and sh.reconstruction_error.max() < 1e-8
    values = pd.read_csv(OUT / 'shap_values.csv')
    for row in sh.itertuples():
        contributions = values[(values.case == row.case) & (values.background == row.background)].weight.sum()
        assert abs(row.base_value + contributions - row.model_score) < 1e-8
    # Validate persisted predictive metrics from persisted scores, not another model run.
    saved_metrics = json.loads((OUT / 'predictive_metrics.json').read_text())
    for name, file in [('development', 'development_predictions.csv'), ('final_test', 'final_test_predictions.csv')]:
        pred = pd.read_csv(OUT / file)
        actual = metrics(pred.Revenue, pred.score)
        for key, value in actual.items():
            if key == 'confusion_matrix':
                assert value == saved_metrics[name][key]
            else:
                assert abs(value - saved_metrics[name][key]) < 1e-9
    print('Saved-output validation passed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--report-only', action='store_true')
    mode.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if not args.report_only and not args.validate_only:
        run()
    validate_saved()
    if not args.validate_only:
        from src.case4_reporting import render
        render(ROOT, OUT)
        print('Results report and figures generated.', flush=True)


if __name__ == '__main__':
    main()
