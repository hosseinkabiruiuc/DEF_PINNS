import importlib.util
import itertools
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / '.'
REV = P / 'results'
OUT = REV / 'journal_check'
spec = importlib.util.spec_from_file_location('compile_results', P / 'src/compile_v3_results.py')
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)

def metric_rows(pred):
    pred = pred[pred.split.eq('terminal_test')].copy()
    pred['absolute_error'] = (pred.predicted_expansion_percent - pred.observed_expansion_percent).abs()
    return pred.groupby(['curve_id', 'record_id', 'source_short', 'seed'], as_index=False).absolute_error.mean()

def inference():
    primary = REV / 'neural_models/primary_selected_20seeds'
    shape = REV / 'neural_models/c100_shape_ic_20seeds'
    shape_run = json.loads((shape / 'neural_model_run_summary.json').read_text())
    main_run = json.loads((primary / 'neural_model_run_summary.json').read_text())
    settings = {key: shape_run.get(key) for key in ['epochs', 'learning_rate', 'weight_decay', 'monotonicity_weight', 'initial_weight', 'collocation_count', 'gate_slope']}
    assert shape_run['seeds'] == main_run['seeds']
    shape_predictions = pd.read_csv(shape / 'neural_predictions.csv')
    shape_curve = stats.curve_metrics(shape_predictions[shape_predictions.split.eq('terminal_test')])
    shape_means = shape_curve.groupby('curve_id')[['mae_percent', 'rmse_percent']].mean()
    wide = pd.read_csv(REV / 'compiled/curve_level_seed_averaged_errors.csv')
    for metric in ['mae_percent', 'rmse_percent']:
        wide[f'{metric}_shape_regularized'] = wide.curve_id.map(shape_means[metric])
    tests = pd.read_csv(REV / 'compiled/curve_level_statistical_tests.csv')
    tests = tests[~tests.comparison.eq('direct threshold fit vs threshold PINN')].copy()
    rows = []
    for comparator, label in [('shape_regularized', 'threshold PINN vs shape + initial condition'), ('shifted_first_order', 'threshold PINN vs direct threshold fit')]:
        for metric in ['mae_percent', 'rmse_percent']:
            a = wide[f'{metric}_pinn_shifted_first_order'].to_numpy()
            b = wide[f'{metric}_{comparator}'].to_numpy()
            diff = b - a
            low, high, rlow, rhigh, sparse = stats.stratified_curve_bootstrap(wide, 'pinn_shifted_first_order', comparator, metric)
            src = wide.assign(difference=diff).groupby('record_id').difference.mean()
            signs = np.array(list(itertools.product([-1, 1], repeat=len(src))))
            rows.append({'comparison': label, 'candidate': 'pinn_shifted_first_order', 'comparator': comparator, 'metric': metric, 'n_curves': len(a), 'candidate_mean': a.mean(), 'comparator_mean': b.mean(), 'mean_paired_improvement': diff.mean(), 'relative_reduction_percent': 100 * (1 - a.mean() / b.mean()), 'rank_biserial_correlation': stats.rank_biserial(diff), 'raw_two_sided_p': wilcoxon(diff).pvalue, 'bootstrap_relative_reduction_ci_low_percent': rlow, 'bootstrap_relative_reduction_ci_high_percent': rhigh, 'source_sign_concordance': f'{int((src > 0).sum())}/{len(src)}', 'source_exact_sign_flip_p': np.mean(np.abs(signs @ src.to_numpy()) >= abs(src.sum()) - 1e-15)})
    tests = pd.concat([tests, pd.DataFrame(rows)], ignore_index=True)
    tests['holm_adjusted_p'] = stats.holm_adjust(tests.raw_two_sided_p.tolist())
    for idx, row in tests.iterrows():
        diff = wide[f'{row.metric}_{row.comparator}'] - wide[f'{row.metric}_{row.candidate}']
        tests.loc[idx, 'curve_wins'] = int((diff > 0).sum())
    tests.to_csv(OUT / 'review_inference_14_tests.csv', index=False)
    shape_metrics = pd.read_csv(shape / 'neural_seed_metrics.csv')
    shape_metrics = shape_metrics[shape_metrics.split.eq('terminal_test')]
    summary = {'shape_configuration': settings, 'seeds_identical': True, 'shape_mean_curve_mae': float(shape_metrics.mean_curve_mae_percent.mean()), 'holm_family_size': len(tests), 'remaining_tuning_limit': 'The optimizer search budget differs between the primary data-only and physics models; this fixed-optimizer ablation does not remove that limitation.'}
    frames = []
    for name, path in [('dfo', primary), ('shape', shape)]:
        frame = metric_rows(pd.read_csv(path / 'neural_predictions.csv')).groupby(['curve_id', 'record_id', 'source_short'], as_index=False).absolute_error.mean()
        frame['method'] = name
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)
    long['log_mae'] = np.log(np.maximum(long.absolute_error, 1e-09))
    long['is_dfo'] = (long.method == 'dfo').astype(int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit = smf.mixedlm('log_mae ~ is_dfo', long, groups=long.record_id, re_formula='1', vc_formula={'curve': '0+C(curve_id)'}).fit(reml=True, method='lbfgs', maxiter=1000)
        summary['exploratory_log_curve_mae_mixed_model'] = {'coefficient': float(fit.params['is_dfo']), 'se': float(fit.bse['is_dfo']), 'p': float(fit.pvalues['is_dfo']), 'converged': bool(fit.converged), 'warnings': [str(w.message) for w in caught], 'limitations': 'Five source clusters; mix identifiers unavailable; random-intercept assumptions are not source-level confirmation.'}
    long.to_csv(OUT / 'review_mixed_model_inputs.csv', index=False)
    (OUT / 'review_ablation_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

def residuals():
    rows = []
    heads = []
    runnames = ['primary_selected_20seeds', 'compliance_lambda_0_5seeds', 'compliance_lambda_0p00003_5seeds', 'compliance_lambda_0p0001_5seeds', 'compliance_lambda_0p0003_5seeds', 'compliance_lambda_0p001_5seeds', 'compliance_lambda_0p01_5seeds', 'compliance_lambda_0p1_5seeds']
    for name in runnames:
        path = REV / 'neural_models' / name
        df = pd.read_csv(path / 'neural_residual_diagnostics.csv')
        df = df[df.model.eq('pinn_shifted_first_order')]
        for (curve, seed, region), g in df.groupby(['curve_id', 'seed', 'region']):
            derivative = g.state_derivative.to_numpy()
            rhs = g.kinetic_rhs.to_numpy()
            residual = g.residual.to_numpy()
            dr = np.sqrt(np.mean(derivative ** 2))
            rr = np.sqrt(np.mean(rhs ** 2))
            er = np.sqrt(np.mean(residual ** 2))
            rows.append({'run': name, 'curve_id': curve, 'seed': seed, 'region': region, 'rms_derivative': dr, 'rms_rhs': rr, 'rms_residual': er, 'normalized_residual': er / max(dr, rr, 1e-12), 'rhs_to_derivative_rms': rr / max(dr, 1e-12), 'derivative_rhs_correlation': np.corrcoef(derivative, rhs)[0, 1] if np.std(rhs) > 1e-12 and np.std(derivative) > 1e-12 else np.nan})
        params = pd.read_csv(path / 'neural_inferred_parameters.csv')
        params = params[params.model.eq('pinn_shifted_first_order')]
        run = json.loads((path / 'neural_model_run_summary.json').read_text())
        bounds = {'rate_normalized': (0.02, 30.02), 'amplitude_percent': (0, 1.5 * run['target_scale_percent']), 'onset_normalized': (0, params.onset_upper_normalized)}
        for key, (low, high) in bounds.items():
            pos = (params[key] - low) / (high - low)
            median_by_curve = params.groupby('curve_id')[key].median()
            heads.append({'run': name, 'parameter': key, 'rows': len(params), 'near_bound_tolerance_fraction': 0.01, 'near_lower_bound': int((pos <= 0.01).sum()), 'near_upper_bound': int((pos >= 0.99).sum()), 'curve_median_min': float(median_by_curve.min()), 'curve_median_median': float(median_by_curve.median()), 'curve_median_max': float(median_by_curve.max())})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'review_raw_residuals.csv', index=False)
    summary = frame.groupby(['run', 'region'], as_index=False)[['rms_derivative', 'rms_rhs', 'rms_residual', 'normalized_residual', 'rhs_to_derivative_rms', 'derivative_rhs_correlation']].median()
    summary.to_csv(OUT / 'review_residual_component_summary.csv', index=False)
    pd.DataFrame(heads).to_csv(OUT / 'review_head_bound_summary.csv', index=False)
    print(summary.to_string(index=False), flush=True)

def factual():
    data = pd.read_csv(P / 'data/modeling/def_kinetics_observations_v2.csv')
    counts = data.groupby('curve_id').agg(n=('split', 'size'), test=('split', lambda s: int(s.eq('terminal_test').sum())))
    out = {'curve_n_min': int(counts.n.min()), 'curve_n_max': int(counts.n.max()), 'actual_holdout_fraction_min': float((counts.test / counts.n).min()), 'actual_holdout_fraction_max': float((counts.test / counts.n).max())}
    for name in ['c100_physics_ic_20seeds', 'c100_physics_only_20seeds']:
        metrics = pd.read_csv(REV / 'neural_models' / name / 'neural_seed_metrics.csv')
        out[name] = metrics[metrics.split.eq('terminal_test')][['mean_curve_mae_percent', 'mae_percent']].mean().to_dict()
    (OUT / 'review_factual_checks.json').write_text(json.dumps(out, indent=2))
    print(out, flush=True)
if __name__ == '__main__':
    inference()
    residuals()
    factual()
