import importlib.util
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / '.'
REV = P / 'results'
OUT = REV / 'journal_check'
table = pd.read_csv(OUT / 'review_inference_14_tests.csv')
wide = pd.read_csv(REV / 'compiled/curve_level_seed_averaged_errors.csv')
pred = pd.read_csv(REV / 'neural_models/c100_shape_ic_20seeds/neural_predictions.csv')
pred = pred[pred.split.eq('terminal_test')].copy()
pred['ae'] = (pred.predicted_expansion_percent - pred.observed_expansion_percent).abs()
pred['sq'] = (pred.predicted_expansion_percent - pred.observed_expansion_percent) ** 2
per_seed = pred.groupby(['curve_id', 'seed']).agg(mae=('ae', 'mean'), mse=('sq', 'mean'))
per_seed['rmse'] = np.sqrt(per_seed.mse)
shape = per_seed.groupby('curve_id')[['mae', 'rmse']].mean()
for metric in ['mae', 'rmse']:
    wide[f'{metric}_percent_shape_regularized'] = wide.curve_id.map(shape[metric])
checks = []
for row in table.itertuples():
    a = wide[f'{row.metric}_{row.candidate}'].to_numpy()
    b = wide[f'{row.metric}_{row.comparator}'].to_numpy()
    delta = b - a
    nonzero = delta[delta != 0]
    ranks = rankdata(np.abs(nonzero))
    rb = (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
    source = wide.assign(d=delta).groupby('record_id').d.mean().to_numpy()
    signs = np.array(list(itertools.product([-1, 1], repeat=len(source))))
    source_p = np.mean(np.abs(signs @ source) >= abs(source.sum()) - 1e-15)
    assert np.isclose(wilcoxon(delta).pvalue, row.raw_two_sided_p, atol=1e-12)
    assert np.isclose(rb, row.rank_biserial_correlation, atol=1e-12)
    assert np.isclose(100 * (1 - a.mean() / b.mean()), row.relative_reduction_percent, atol=1e-10)
    assert int((delta > 0).sum()) == row.curve_wins
    assert np.isclose(source_p, row.source_exact_sign_flip_p)
    checks.append({'comparison': row.comparison, 'metric': row.metric, 'passed': True})
p = table.raw_two_sided_p.to_numpy()
order = np.argsort(p)
adjusted = np.empty(len(p))
adjusted[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
assert np.allclose(adjusted, table.holm_adjusted_p, atol=1e-12)
spec = importlib.util.spec_from_file_location('trainer', P / 'src/train_def_pinn_models_v3.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
import argparse
run = json.loads((REV / 'neural_models/review_curve_scale_dfo/neural_model_run_summary.json').read_text())
args = argparse.Namespace(**run)
prepared = trainer.prepare_data(args)
data = prepared['data']
for code in range(len(prepared['curve_scales'])):
    curve_mask = data.curve_code_v3.eq(code).to_numpy()
    scale = max(float(data.loc[prepared['train_mask'] & curve_mask, 'expansion_percent'].max()), 0.015)
    assert np.isclose(scale, prepared['curve_scales'][code], rtol=1e-06)
assert np.allclose(prepared['targets'] * prepared['curve_scales'][prepared['curve_indices']], data.expansion_percent, atol=1e-06)
cohort = pd.read_csv(REV / 'c100_revision/all_terminal_predictions.csv')
rows = []
for subset, frame in [('all', cohort), ('excluding_negligible', cohort[cohort.response_class.ne('negligible')])]:
    for model, group in frame.groupby('model'):
        rows.append({'subset': subset, 'model': model, 'curves': group.curve_id.nunique(), 'pooled_ensemble_or_deterministic_mae_percent': group.absolute_error_percent.mean(), 'mean_curve_ensemble_or_deterministic_mae_percent': group.groupby('curve_id').absolute_error_percent.mean().mean()})
pd.DataFrame(rows).to_csv(OUT / 'review_without_negligible.csv', index=False)
report = {'14_test_inference_independently_recomputed': all((r['passed'] for r in checks)), 'holm_matches': True, 'source_sign_tests_match': True, 'per_curve_scaling_training_only': True, 'comparisons': checks, 'scientific_review_complete': False}
(OUT / 'current_review_numerical_verification.json').write_text(json.dumps(report, indent=2))
trunc = pd.read_csv(OUT / 'review_all_comparator_truncation.csv')
lines = ['# Computational Supplement', '', 'This supplement records exploratory diagnostics, not independent prospective validation.', '', '## Shorter Training Histories', '', 'Fractions are elapsed time within the original preterminal history, with a minimum of five points. Neural entries are 10-seed means; other entries are deterministic. Errors are mean curve MAE in % strain.', '', '| Method | 50% time prefix | 75% time prefix |', '|---|---:|---:|']
for model, group in trunc.groupby('model'):
    vals = group.set_index('fraction_elapsed_preterminal_time').mean_curve_mae_percent
    lines.append(f'| {model} | {vals.loc[0.5]:.6f} | {vals.loc[0.75]:.6f} |')
lines += ['', '## Residual and Scaling Diagnostics', '', 'Raw derivative, right-hand-side, and residual RMS values are in `results/journal_check/review_residual_component_summary.csv`. Parameter ranges and the explicit 1%-of-support near-bound rule are in `review_head_bound_summary.csv`. Near-bound labels are not proof of exact numerical bound contact.', '', 'New five-seed domain, gate-aware sampling, and per-curve scaling results are in `review_new_sensitivity_metrics.csv`. The gate-aware sampling measure differs from uniform collocation. Per-curve scaling also changes dimensional head capacity support. These tests are not independent confirmations of the selected primary configuration.', '', '## Conformal Diagnostic', '', 'For sorted calibration absolute errors $e_{(h)}$, the finite-sample index is $h=\\lceil(n_{cal}+1)(1-\\delta)\\rceil$ and the half-width is $e_{(h)}$; if $h>n_{cal}$, use an infinite half-width. The 302 preterminal calibration residuals were reused for selection and are serially/source dependent. Empirical coverage therefore does not establish exchangeable or prospective coverage guarantees. The archived coverage files are computational diagnostics only.', '', '## Remaining Scientific Checks', '', 'Equal optimizer-search budgets, onset-stratified rolling-origin evaluation, independent digitization error in strain units, and neutral treatment of ambiguous marker groups remain incomplete. The package must not be described as submission-ready.']
(ROOT / 'src/computational_supplement.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('Verified 14 contrasts, source tests, training-only per-curve scales; saved supplementary tables.')
