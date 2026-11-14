from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / '.'
REV = PROJECT / 'results'
OUT = REV / 'reviewer_diagnostics'
OUT.mkdir(parents=True, exist_ok=True)
OBS_PATH = PROJECT / 'data/modeling' / 'def_kinetics_observations_v2.csv'
CURVE_PATH = PROJECT / 'data/modeling' / 'def_kinetics_curves_v2.csv'
PRIMARY = REV / 'neural_models' / 'primary_selected_20seeds'
CALIBRATION = REV / 'neural_models' / 'reviewer_v5_lambda_cv_0p0003_5seeds'
C100 = REV / 'c100_revision'
COMPILED = REV / 'compiled'

def finite_quantile(values: pd.Series, probabilities=(0, 0.25, 0.5, 0.75, 1)) -> list[float]:
    return [float(x) for x in np.quantile(values.dropna().to_numpy(float), probabilities)]

def seed_mean_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ['observation_id', 'curve_id', 'record_id', 'source_short', 'model', 'split', 'time_days']
    return frame.groupby(keys, as_index=False).agg(observed_expansion_percent=('observed_expansion_percent', 'first'), predicted_expansion_percent=('predicted_expansion_percent', 'mean'))

def main() -> None:
    obs = pd.read_csv(OBS_PATH).sort_values(['curve_id', 'time_days'])
    curves = pd.read_csv(CURVE_PATH)
    neural_raw = pd.read_csv(PRIMARY / 'neural_predictions.csv')
    primary = seed_mean_predictions(neural_raw)
    terminal = primary[primary['split'].eq('terminal_test')]
    all_predictions = pd.read_csv(C100 / 'all_terminal_predictions.csv')
    curve_errors = pd.read_csv(COMPILED / 'curve_level_seed_averaged_errors.csv')
    duration = obs.groupby('curve_id')['time_days'].agg(lambda values: values.max() - values.min())
    tau_max = neural_raw[neural_raw['split'].eq('terminal_test')].groupby('curve_id')['tau'].max().rename('tau_max')
    task = pd.concat([duration.rename('duration_days'), tau_max], axis=1).reset_index()
    task['tau_tercile'] = pd.qcut(task['tau_max'], 3, labels=['short', 'middle', 'long'], duplicates='drop')
    task.to_csv(OUT / 'r2_forecast_horizon_by_curve.csv', index=False)
    terciles = curve_errors.merge(task[['curve_id', 'tau_max', 'tau_tercile']], on='curve_id')
    models = ['pinn_shifted_first_order', 'data_only', 'shifted_first_order', 'persistence']
    tercile_rows = []
    for level, group in terciles.groupby('tau_tercile', observed=True):
        for model in models:
            values = group[f'mae_percent_{model}']
            tercile_rows.append({'tau_tercile': str(level), 'model': model, 'curves': len(group), 'tau_min': group.tau_max.min(), 'tau_max': group.tau_max.max(), 'mean_curve_mae_percent': values.mean(), 'median_curve_mae_percent': values.median()})
    pd.DataFrame(tercile_rows).to_csv(OUT / 'r2_metrics_by_tau_tercile.csv', index=False)
    curve_metric = pd.read_csv(C100 / 'curve_level_metrics.csv')
    large_rows = []
    for model, group in curve_metric.groupby('model'):
        large_rows.append({'model': model, 'curves': len(group), 'curve_mae_gt_0p10': int(group.mae_percent.gt(0.1).sum()), 'curve_mae_gt_0p20': int(group.mae_percent.gt(0.2).sum()), 'curve_mae_five_number': finite_quantile(group.mae_percent)})
    pd.DataFrame(large_rows).assign(curve_mae_five_number=lambda frame: frame.curve_mae_five_number.map(json.dumps)).to_csv(OUT / 'r2_large_error_counts.csv', index=False)
    wide = all_predictions[all_predictions.model.isin(['pinn_shifted_first_order', 'data_only'])].pivot(index=['curve_id', 'observation_id'], columns='model', values=['observed_expansion_percent', 'predicted_expansion_percent'])
    observed = wide['observed_expansion_percent']['pinn_shifted_first_order']
    squared_advantage = (wide['predicted_expansion_percent']['data_only'] - observed) ** 2 - (wide['predicted_expansion_percent']['pinn_shifted_first_order'] - observed) ** 2
    contributors = squared_advantage.groupby(level='curve_id').sum().sort_values(ascending=False).rename('squared_error_advantage').reset_index()
    contributors.to_csv(OUT / 'r2_pooled_squared_error_contributors.csv', index=False)
    calibration = seed_mean_predictions(pd.read_csv(CALIBRATION / 'neural_predictions.csv'))
    calibration = calibration[calibration.model.eq('pinn_shifted_first_order') & calibration.split.eq('internal_validation')]
    calibration = calibration.merge(curves[['curve_id', 'response_class']], on='curve_id', how='left')
    calibration['absolute_error'] = (calibration.observed_expansion_percent - calibration.predicted_expansion_percent).abs()
    test = terminal[terminal.model.eq('pinn_shifted_first_order')].merge(curves[['curve_id', 'response_class']], on='curve_id', how='left')
    test['absolute_error'] = (test.observed_expansion_percent - test.predicted_expansion_percent).abs()
    conformal_rows = []
    for response_class, cal_group in calibration.groupby('response_class'):
        test_group = test[test.response_class.eq(response_class)]
        for nominal in (0.8, 0.9):
            n = len(cal_group)
            rank = math.ceil((n + 1) * nominal)
            half = float(np.sort(cal_group.absolute_error)[rank - 1]) if rank <= n else math.inf
            conformal_rows.append({'response_class': response_class, 'nominal': nominal, 'calibration_n': n, 'test_n': len(test_group), 'half_width_percent': half, 'coverage': float(test_group.absolute_error.le(half).mean())})
    pd.DataFrame(conformal_rows).to_csv(OUT / 'r2_conformal_by_response_class.csv', index=False)
    raw_truth = obs.set_index('observation_id')['expansion_percent_raw']
    clipped = obs.set_index('observation_id')['expansion_percent']
    locations = obs.loc[obs.expansion_percent_raw.lt(0), ['observation_id', 'curve_id', 'record_id', 'source_short', 'time_days', 'expansion_percent_raw', 'expansion_percent']]
    locations.to_csv(OUT / 'r2_negative_observation_locations.csv', index=False)
    raw_rows = []
    selected_predictions = all_predictions[all_predictions.model.isin(models)].copy()
    selected_predictions['raw_observed'] = selected_predictions.observation_id.map(raw_truth).fillna(selected_predictions.observation_id.map(clipped))
    selected_predictions['raw_absolute_error'] = (selected_predictions.predicted_expansion_percent - selected_predictions.raw_observed).abs()
    for model, group in selected_predictions.groupby('model'):
        per_curve = group.groupby('curve_id').agg(clipped_mae=('absolute_error_percent', 'mean'), raw_mae=('raw_absolute_error', 'mean'))
        raw_rows.append({'model': model, 'curves': len(per_curve), 'mean_curve_mae_clipped': per_curve.clipped_mae.mean(), 'mean_curve_mae_raw': per_curve.raw_mae.mean(), 'difference': per_curve.raw_mae.mean() - per_curve.clipped_mae.mean()})
    pd.DataFrame(raw_rows).to_csv(OUT / 'r2_raw_negative_target_sensitivity.csv', index=False)
    params = pd.read_csv(PRIMARY / 'neural_inferred_parameters.csv')
    params = params[params.model.eq('pinn_shifted_first_order')]
    params['threshold_fraction'] = 1 - np.exp(-params.rate_normalized * params.onset_normalized)
    param_curve = params.groupby(['curve_id', 'source_short'], as_index=False).threshold_fraction.median()
    resolution = {'Aubert et al. 2013': 0.009654957140467355, 'Brunetaud et al. 2007': 2.7245581360746626e-05, 'Kawabata et al. 2023': 0.005520919012214786, 'Martin et al. 2023': 0.03370197276826354, 'Tiecher et al. 2021': 0.018916245127827625}
    train_max = obs[obs.split.eq('train')].groupby('curve_id').expansion_percent.max()
    param_curve['resolution_floor'] = param_curve.source_short.map(resolution).clip(lower=0.02)
    param_curve['training_exceeds_floor'] = param_curve.curve_id.map(train_max).ge(param_curve.resolution_floor)
    restricted = param_curve[param_curve.training_exceeds_floor].threshold_fraction
    param_curve.to_csv(OUT / 'r2_threshold_resolution_screen.csv', index=False)
    inventory = curves.copy()
    counts = obs.groupby(['curve_id', 'split']).size().unstack(fill_value=0)
    inventory = inventory.merge(counts, on='curve_id', how='left').merge(task, on='curve_id', how='left')
    inventory.to_csv(OUT / 'r2_curve_inventory_61.csv', index=False)
    summary = {'duration_days_five_number': finite_quantile(duration), 'tau_max_five_number': finite_quantile(tau_max), 'threshold_resolution_restricted_n': int(len(restricted)), 'threshold_resolution_restricted_five_number': finite_quantile(restricted), 'top_five_squared_error_contributors': contributors.head(5).to_dict(orient='records')}
    (OUT / 'r2_supplementary_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
