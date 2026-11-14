from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / 'results'
NEURAL = REV / 'neural_models' / 'primary_selected_20seeds'
DIRECT = REV / 'direct_kinetic_fits' / 'primary_training_only'
DATA = ROOT / 'data/modeling' / 'def_kinetics_observations_v2.csv'
OUT = REV / 'compiled'
RNG_SEED = 20260815
BOOTSTRAP_ITERATIONS = 50000

def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = observed - predicted
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    return {'mae_percent': float(np.mean(np.abs(residual))), 'rmse_percent': float(np.sqrt(np.mean(residual ** 2))), 'r2': 1.0 - float(np.sum(residual ** 2)) / denominator if denominator > 0 else math.nan}

def baseline_predictions(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for curve_id, group in data.groupby('curve_id', sort=True):
        group = group.sort_values('time_days')
        train = group[group['split'] == 'train']
        test = group[group['split'] == 'terminal_test']
        last_value = float(train.iloc[-1]['expansion_percent'])
        tail = train.tail(min(5, len(train)))
        if tail['time_days'].nunique() >= 2:
            slope, intercept = np.polyfit(tail['time_days'].to_numpy(float), tail['expansion_percent'].to_numpy(float), 1)
            linear = np.maximum(0.0, intercept + slope * test['time_days'].to_numpy(float))
        else:
            linear = np.full(len(test), last_value)
        for model, predicted in (('persistence', np.full(len(test), last_value)), ('linear_last5', linear)):
            for (_, source_row), prediction in zip(test.iterrows(), predicted):
                rows.append({'observation_id': source_row['observation_id'], 'curve_id': curve_id, 'record_id': source_row['record_id'], 'source_short': source_row['source_short'], 'model': model, 'seed': -1, 'split': 'terminal_test', 'time_days': source_row['time_days'], 'observed_expansion_percent': source_row['expansion_percent'], 'predicted_expansion_percent': float(prediction)})
    return pd.DataFrame(rows)

def curve_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dimensions = ['model', 'seed', 'curve_id', 'record_id', 'source_short']
    for keys, group in predictions.groupby(dimensions, sort=True):
        model, seed, curve_id, record_id, source_short = keys
        rows.append({'model': model, 'seed': int(seed), 'curve_id': curve_id, 'record_id': record_id, 'source_short': source_short, 'observations': len(group), **metrics(group['observed_expansion_percent'].to_numpy(float), group['predicted_expansion_percent'].to_numpy(float))})
    return pd.DataFrame(rows)

def rank_biserial(improvement: np.ndarray) -> float:
    values = improvement[np.isfinite(improvement) & (improvement != 0)]
    if len(values) == 0:
        return 0.0
    ranks = rankdata(np.abs(values), method='average')
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    return (positive - negative) / (positive + negative)

def stratified_curve_bootstrap(frame: pd.DataFrame, candidate: str, comparator: str, metric: str) -> tuple[float, float, float, float, int]:
    rng = np.random.default_rng(RNG_SEED)
    groups = {key: (group[f'{metric}_{candidate}'].to_numpy(float), group[f'{metric}_{comparator}'].to_numpy(float)) for key, group in frame.groupby('record_id', sort=True)}
    source_ids = np.asarray(sorted(groups))
    differences = np.empty(BOOTSTRAP_ITERATIONS)
    reductions = np.empty(BOOTSTRAP_ITERATIONS)
    fewer_than_three_sources = 0
    for iteration in range(BOOTSTRAP_ITERATIONS):
        sampled_sources = rng.choice(source_ids, size=len(source_ids), replace=True)
        if len(set(sampled_sources)) < 3:
            fewer_than_three_sources += 1
        candidate_sum = 0.0
        comparator_sum = 0.0
        count = 0
        for source_id in sampled_sources:
            candidate_values, comparator_values = groups[source_id]
            indices = rng.integers(0, len(candidate_values), size=len(candidate_values))
            candidate_sum += float(candidate_values[indices].sum())
            comparator_sum += float(comparator_values[indices].sum())
            count += len(indices)
        candidate_mean = candidate_sum / count
        comparator_mean = comparator_sum / count
        differences[iteration] = comparator_mean - candidate_mean
        reductions[iteration] = 100.0 * (1.0 - candidate_mean / comparator_mean) if comparator_mean != 0 else math.nan
    diff_ci = np.quantile(differences, [0.025, 0.975])
    reduction_ci = np.nanquantile(reductions, [0.025, 0.975])
    return (float(diff_ci[0]), float(diff_ci[1]), float(reduction_ci[0]), float(reduction_ci[1]), fewer_than_three_sources)

def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()

def aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed), group in predictions.groupby(['model', 'seed']):
        rows.append({'model': model, 'seed': seed, 'observations': len(group), **metrics(group['observed_expansion_percent'].to_numpy(float), group['predicted_expansion_percent'].to_numpy(float))})
    return pd.DataFrame(rows)

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    neural_predictions = pd.read_csv(NEURAL / 'neural_predictions.csv')
    neural_terminal = neural_predictions[neural_predictions['split'] == 'terminal_test'].copy()
    direct_predictions = pd.read_csv(DIRECT / 'direct_fit_predictions.csv')
    direct_terminal = direct_predictions[direct_predictions['split'] == 'terminal_test'].copy()
    direct_terminal['seed'] = -1
    baselines = baseline_predictions(data)
    baselines.to_csv(OUT / 'baseline_predictions.csv', index=False)
    all_predictions = pd.concat([neural_terminal, direct_terminal, baselines], ignore_index=True, sort=False)
    all_curve_metrics = curve_metrics(all_predictions)
    all_curve_metrics.to_csv(OUT / 'terminal_curve_metrics_all.csv', index=False)
    aggregate = aggregate_predictions(all_predictions)
    aggregate.to_csv(OUT / 'terminal_aggregate_metrics_all.csv', index=False)
    neural_curve = all_curve_metrics[all_curve_metrics['seed'] >= 0]
    neural_mean = neural_curve.groupby(['model', 'curve_id', 'record_id', 'source_short'], as_index=False).agg(mae_percent=('mae_percent', 'mean'), rmse_percent=('rmse_percent', 'mean'))
    single_curve = all_curve_metrics[all_curve_metrics['seed'] == -1][['model', 'curve_id', 'record_id', 'source_short', 'mae_percent', 'rmse_percent']]
    inference_curve = pd.concat([neural_mean, single_curve], ignore_index=True)
    wide = inference_curve.pivot(index=['curve_id', 'record_id', 'source_short'], columns='model', values=['mae_percent', 'rmse_percent'])
    wide.columns = [f'{metric}_{model}' for metric, model in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(OUT / 'curve_level_seed_averaged_errors.csv', index=False)
    comparisons = [('pinn_shifted_first_order', 'data_only', 'threshold PINN vs data-only'), ('pinn_avrami', 'data_only', 'Avrami PINN vs data-only'), ('pinn_shifted_first_order', 'pinn_avrami', 'threshold PINN vs Avrami PINN'), ('shifted_first_order', 'pinn_shifted_first_order', 'direct threshold fit vs threshold PINN'), ('pinn_shifted_first_order', 'persistence', 'threshold PINN vs persistence'), ('pinn_shifted_first_order', 'linear_last5', 'threshold PINN vs linear extrapolation')]
    test_rows: list[dict[str, object]] = []
    for candidate, comparator, label in comparisons:
        for metric in ['mae_percent', 'rmse_percent']:
            required = [f'{metric}_{candidate}', f'{metric}_{comparator}']
            frame = wide.dropna(subset=required).copy()
            candidate_values = frame[required[0]].to_numpy(float)
            comparator_values = frame[required[1]].to_numpy(float)
            improvement = comparator_values - candidate_values
            try:
                statistic, p_value = wilcoxon(improvement, alternative='two-sided', zero_method='wilcox', method='auto')
            except ValueError:
                statistic, p_value = (math.nan, 1.0)
            ci_low, ci_high, reduction_low, reduction_high, sparse_bootstraps = stratified_curve_bootstrap(frame, candidate, comparator, metric)
            frame = frame.assign(improvement=improvement)
            source_means = frame.groupby('record_id')['improvement'].mean()
            concordant_sources = int((source_means > 0).sum())
            signs = np.asarray(np.meshgrid(*[[-1, 1]] * len(source_means))).T.reshape(-1, len(source_means))
            source_exact_p = float(np.mean(np.abs(signs @ source_means.to_numpy()) >= abs(source_means.sum()) - 1e-15))
            delete_one = np.asarray([frame.loc[~frame['record_id'].eq(source), 'improvement'].mean() for source in source_means.index])
            test_rows.append({'comparison': label, 'candidate': candidate, 'comparator': comparator, 'metric': metric, 'n_curves': len(frame), 'candidate_mean': float(np.mean(candidate_values)), 'comparator_mean': float(np.mean(comparator_values)), 'mean_paired_improvement': float(np.mean(improvement)), 'relative_reduction_percent': 100.0 * (1.0 - float(np.mean(candidate_values)) / float(np.mean(comparator_values))), 'rank_biserial_correlation': rank_biserial(improvement), 'wilcoxon_statistic': float(statistic), 'ties': int(np.isclose(improvement, 0.0, atol=1e-12).sum()), 'raw_two_sided_p': float(p_value), 'bootstrap_improvement_ci_low': ci_low, 'bootstrap_improvement_ci_high': ci_high, 'bootstrap_relative_reduction_ci_low_percent': reduction_low, 'bootstrap_relative_reduction_ci_high_percent': reduction_high, 'bootstrap_design': '50000 two-stage source/curve resamples', 'bootstrap_replicates_with_fewer_than_three_distinct_sources': sparse_bootstraps, 'source_sign_concordance': f'{concordant_sources}/{len(source_means)}', 'source_exact_sign_flip_p': source_exact_p, 'delete_one_source_min': float(delete_one.min()), 'delete_one_source_max': float(delete_one.max())})
    tests = pd.DataFrame(test_rows)
    tests['holm_adjusted_p'] = holm_adjust(tests['raw_two_sided_p'].tolist())
    tests.to_csv(OUT / 'curve_level_statistical_tests.csv', index=False)
    seed_summary_rows = []
    neural_aggregate = aggregate[aggregate['seed'] >= 0]
    for model, group in neural_aggregate.groupby('model'):
        seed_summary_rows.append({'model': model, 'seeds': group['seed'].nunique(), 'mae_mean_percent': group['mae_percent'].mean(), 'mae_sd_percent': group['mae_percent'].std(ddof=1), 'mae_cv_percent': 100.0 * group['mae_percent'].std(ddof=1) / group['mae_percent'].mean(), 'rmse_mean_percent': group['rmse_percent'].mean(), 'rmse_sd_percent': group['rmse_percent'].std(ddof=1), 'rmse_cv_percent': 100.0 * group['rmse_percent'].std(ddof=1) / group['rmse_percent'].mean(), 'r2_mean': group['r2'].mean(), 'r2_sd': group['r2'].std(ddof=1)})
    seed_summary = pd.DataFrame(seed_summary_rows)
    seed_summary.to_csv(OUT / 'seed_stability_summary.csv', index=False)
    source_rows = []
    for (source, model), group in inference_curve.groupby(['source_short', 'model']):
        source_rows.append({'source_short': source, 'model': model, 'curves': group['curve_id'].nunique(), 'mean_curve_mae_percent': group['mae_percent'].mean(), 'median_curve_mae_percent': group['mae_percent'].median(), 'mean_curve_rmse_percent': group['rmse_percent'].mean(), 'median_curve_rmse_percent': group['rmse_percent'].median()})
    pd.DataFrame(source_rows).to_csv(OUT / 'source_stratified_curve_metrics.csv', index=False)
    direct_parameters = pd.read_csv(DIRECT / 'direct_fit_parameters.csv')
    identifiability = direct_parameters.groupby('model', as_index=False).agg(curves=('curve_id', 'nunique'), median_information_condition_number=('jacobian_information_condition_number', 'median'), maximum_information_condition_number=('jacobian_information_condition_number', 'max'), median_maximum_abs_parameter_correlation=('maximum_absolute_parameter_correlation', 'median'), curves_with_any_bound_hit=('bound_hit_count', lambda x: int((x > 0).sum())), total_bound_hits=('bound_hit_count', 'sum'), converged_fits=('converged', 'sum'))
    identifiability.to_csv(OUT / 'direct_fit_identifiability_summary.csv', index=False)
    direct_curve = all_curve_metrics[(all_curve_metrics['seed'] == -1) & all_curve_metrics['model'].isin((MODELS := list(pd.read_csv(DIRECT / 'direct_fit_aggregate_metrics.csv')['model'].unique())))]
    catastrophe_rows = []
    for model, group in direct_curve.groupby('model'):
        median_rmse = float(group['rmse_percent'].median())
        threshold = max(0.25, 5.0 * median_rmse)
        severe = group[group['rmse_percent'] > threshold]
        catastrophe_rows.append({'model': model, 'definition': 'curve terminal RMSE > max(0.25% strain, 5 x model median)', 'threshold_rmse_percent': threshold, 'severe_curve_count': len(severe), 'maximum_curve_rmse_percent': group['rmse_percent'].max()})
    pd.DataFrame(catastrophe_rows).to_csv(OUT / 'direct_fit_catastrophic_extrapolations.csv', index=False)
    summary = {'curves': int(data['curve_id'].nunique()), 'terminal_observations': int((data['split'] == 'terminal_test').sum()), 'neural_seeds': int(neural_predictions['seed'].nunique()), 'statistical_unit': 'curve; neural errors averaged across seeds before inference', 'test_family_size': len(tests), 'multiplicity': 'Holm family-wise adjustment', 'bootstrap': f'{BOOTSTRAP_ITERATIONS} two-stage source/curve resamples', 'headline_seed_summary': seed_summary.to_dict(orient='records'), 'headline_tests': tests.to_dict(orient='records')}
    (OUT / 'compiled_v3_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(seed_summary.to_string(index=False))
    print(tests[['comparison', 'metric', 'n_curves', 'relative_reduction_percent', 'rank_biserial_correlation', 'raw_two_sided_p', 'holm_adjusted_p']].to_string(index=False))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
