from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/modeling' / 'def_kinetics_observations_v2.csv'
OUT = ROOT / 'results' / 'direct_kinetic_fits'
ACTIVE_GATE_SLOPE = 200.0

def smooth_clock(t: np.ndarray, onset: float) -> np.ndarray:
    slope = ACTIVE_GATE_SLOPE
    return (np.logaddexp(0.0, slope * (t - onset)) - np.logaddexp(0.0, -slope * onset)) / slope

def first_order(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    amplitude, rate = p
    return amplitude * (1.0 - np.exp(-rate * np.maximum(t, 0.0)))

def shifted_first_order(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    amplitude, rate, onset = p
    return amplitude * (1.0 - np.exp(-rate * smooth_clock(t, onset)))

def shifted_avrami(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    amplitude, rate, onset, exponent = p
    return amplitude * (1.0 - np.exp(-np.power(rate * smooth_clock(t, onset), exponent)))

def gompertz(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    amplitude, rate, midpoint = p
    return amplitude * np.exp(-np.exp(-rate * (t - midpoint)))

def larive(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    amplitude, characteristic_time, latency = p
    characteristic_time = max(float(characteristic_time), 1e-08)
    numerator = 1.0 - np.exp(-np.maximum(t, 0.0) / characteristic_time)
    denominator = 1.0 + np.exp(np.clip(-(t - latency) / characteristic_time, -700.0, 700.0))
    return amplitude * numerator / denominator
MODELS = {'first_order': {'function': first_order, 'lower': np.array([0.0, 0.0001]), 'upper': np.array([5.0, 100.0]), 'parameters': ['amplitude_percent', 'rate_normalized']}, 'shifted_first_order': {'function': shifted_first_order, 'lower': np.array([0.0, 0.0001, 0.0]), 'upper': np.array([5.0, 100.0, 3.0]), 'parameters': ['amplitude_percent', 'rate_normalized', 'onset_normalized']}, 'shifted_avrami': {'function': shifted_avrami, 'lower': np.array([0.0, 0.0001, 0.0, 1.0]), 'upper': np.array([5.0, 100.0, 3.0, 4.0]), 'parameters': ['amplitude_percent', 'rate_normalized', 'onset_normalized', 'avrami_exponent']}, 'gompertz': {'function': gompertz, 'lower': np.array([0.0, 0.0001, -0.5]), 'upper': np.array([5.0, 100.0, 3.0]), 'parameters': ['amplitude_percent', 'rate_normalized', 'midpoint_normalized']}, 'larive': {'function': larive, 'lower': np.array([0.0, 0.0001, 0.0]), 'upper': np.array([5.0, 10.0, 3.0]), 'parameters': ['amplitude_percent', 'characteristic_time_normalized', 'latency_normalized']}}

def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = observed - predicted
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    return {'mae_percent': float(np.mean(np.abs(residual))), 'rmse_percent': float(np.sqrt(np.mean(residual ** 2))), 'r2': 1.0 - float(np.sum(residual ** 2)) / denominator if denominator > 0 else math.nan, 'holdout_variance': float(np.var(observed, ddof=0))}

def initial_points(model_name: str, ymax: float) -> list[np.ndarray]:
    amplitudes = [max(ymax, 0.02), min(5.0, max(1.25 * ymax, 0.05))]
    rates = [0.3, 1.0, 3.0, 10.0]
    if model_name == 'first_order':
        return [np.array([a, k]) for a in amplitudes for k in rates]
    if model_name == 'shifted_first_order':
        return [np.array([a, k, t0]) for a in amplitudes for k in rates for t0 in [0.05, 0.4, 0.9, 1.5]]
    if model_name == 'shifted_avrami':
        return [np.array([a, k, t0, n]) for a in amplitudes for k in [0.5, 2.0, 8.0] for t0 in [0.05, 0.5, 1.0, 1.5] for n in [1.1, 2.0, 3.5]]
    if model_name == 'larive':
        return [np.array([a, tc, tl]) for a in amplitudes for tc in [0.05, 0.2, 0.7, 2.0] for tl in [0.05, 0.4, 0.9, 1.5]]
    return [np.array([a, k, tm]) for a in amplitudes for k in rates for tm in [0.1, 0.5, 1.0, 1.5]]

def fit_one(model_name: str, train_t: np.ndarray, train_y: np.ndarray, onset_upper: float | None=None) -> tuple[np.ndarray, bool, float, float, float, int, int]:
    spec = MODELS[model_name]
    lower = spec['lower'].copy()
    upper = spec['upper'].copy()
    if onset_upper is not None and 'onset_normalized' in spec['parameters']:
        onset_index = spec['parameters'].index('onset_normalized')
        upper[onset_index] = min(float(upper[onset_index]), float(onset_upper))
    ymax = float(max(np.max(train_y), 0.01))
    residual_scale = max(float(np.std(train_y)), 0.015)

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (spec['function'](train_t, parameters) - train_y) / residual_scale
    best = None
    successful = 0
    for initial in initial_points(model_name, ymax):
        initial = np.minimum(np.maximum(initial, lower + 1e-08), upper - 1e-08)
        result = least_squares(residual, x0=initial, bounds=(lower, upper), loss='soft_l1', f_scale=1.0, xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=30000)
        successful += int(result.success)
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    raw_residual = spec['function'](train_t, best.x) - train_y
    rss = float(np.sum(raw_residual ** 2))
    jacobian = np.asarray(best.jac, dtype=float)
    information = jacobian.T @ jacobian
    condition_number = float(np.linalg.cond(information)) if information.size else math.nan
    try:
        covariance = np.linalg.pinv(information)
        standard = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        correlation = covariance / np.outer(standard, standard)
        np.fill_diagonal(correlation, 0.0)
        max_abs_correlation = float(np.nanmax(np.abs(correlation)))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        max_abs_correlation = math.nan
    tolerance = 0.0001 * np.maximum(upper - lower, 1.0)
    bound_hits = int(np.sum((best.x - lower <= tolerance) | (upper - best.x <= tolerance)))
    return (best.x, bool(best.success), rss, condition_number, max_abs_correlation, bound_hits, successful)

def training_mask(group: pd.DataFrame, fraction: float) -> np.ndarray:
    base = group['split'].eq('train').to_numpy()
    if fraction >= 1.0:
        return base
    train_group = group[base].sort_values('time_days')
    t_min = float(train_group['time_days'].min())
    t_max = float(train_group['time_days'].max())
    cutoff = t_min + fraction * (t_max - t_min)
    selected_indices = train_group[train_group['time_days'] <= cutoff].index
    if len(selected_indices) < min(5, len(train_group)):
        selected_indices = train_group.iloc[:min(5, len(train_group))].index
    return group.index.isin(selected_indices)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='primary_training_only')
    parser.add_argument('--time-scaling', choices=['training_only', 'full_series'], default='training_only')
    parser.add_argument('--train-observation-fraction', type=float, default=1.0)
    parser.add_argument('--gate-slope', type=float, default=200.0)
    parser.add_argument('--response-derived-onset-bound', action='store_true')
    parser.add_argument('--activation-detection-percent', type=float, default=0.02)
    parser.add_argument('--activation-margin-tau', type=float, default=0.1)
    return parser.parse_args()

def main() -> int:
    global ACTIVE_GATE_SLOPE
    args = parse_args()
    ACTIVE_GATE_SLOPE = args.gate_slope
    if not 0.0 < args.train_observation_fraction <= 1.0:
        raise ValueError('train_observation_fraction must be in (0, 1]')
    out = OUT / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    for curve_id, original_group in data.groupby('curve_id', sort=True):
        group = original_group.sort_values('time_days').copy()
        train_mask = training_mask(group, args.train_observation_fraction)
        training_horizon = max(float(group.loc[train_mask, 'time_days'].max()), 1.0)
        full_horizon = max(float(group['time_days'].max()), 1.0)
        time_scale = full_horizon if args.time_scaling == 'full_series' else training_horizon
        normalized_time = group['time_days'].to_numpy(float) / time_scale
        observed = group['expansion_percent'].to_numpy(float)
        train_t = normalized_time[train_mask]
        train_y = observed[train_mask]
        onset_upper = 3.0
        if args.response_derived_onset_bound:
            activated = np.flatnonzero(train_mask & (observed >= args.activation_detection_percent))
            if len(activated):
                onset_upper = float(np.clip(normalized_time[int(activated[0])] + args.activation_margin_tau, 0.15, 3.0))
        evaluation_split = group['split'].astype(str).to_numpy()
        evaluation_split[group['split'].eq('train').to_numpy() & ~train_mask] = 'preterminal_unobserved'
        for model_name, spec in MODELS.items():
            parameters, converged, rss, condition_number, max_abs_correlation, bound_hits, successful_starts = fit_one(model_name, train_t, train_y, onset_upper=onset_upper if args.response_derived_onset_bound else None)
            predicted = spec['function'](normalized_time, parameters)
            n_train = int(train_mask.sum())
            parameter_count = len(parameters)
            variance = max(rss / max(n_train, 1), 1e-16)
            gaussian_bic = n_train * math.log(variance) + parameter_count * math.log(n_train)
            parameter_row: dict[str, object] = {'curve_id': curve_id, 'record_id': group['record_id'].iloc[0], 'source_short': group['source_short'].iloc[0], 'model': model_name, 'converged': converged, 'time_scaling': args.time_scaling, 'time_scale_days': time_scale, 'training_horizon_days': training_horizon, 'full_horizon_days': full_horizon, 'train_observation_fraction': args.train_observation_fraction, 'response_derived_onset_bound': args.response_derived_onset_bound, 'onset_upper_normalized': onset_upper if 'onset_normalized' in spec['parameters'] else math.nan, 'n_train': n_train, 'train_rss': rss, 'gaussian_bic_from_raw_rss': gaussian_bic, 'jacobian_information_condition_number': condition_number, 'maximum_absolute_parameter_correlation': max_abs_correlation, 'bound_hit_count': bound_hits, 'successful_multistarts': successful_starts, 'total_multistarts': len(initial_points(model_name, float(max(np.max(train_y), 0.01))))}
            for name, value in zip(spec['parameters'], parameters):
                parameter_row[name] = float(value)
            if 'rate_normalized' in parameter_row:
                parameter_row['rate_per_day'] = float(parameter_row['rate_normalized']) / time_scale
            if 'onset_normalized' in parameter_row:
                parameter_row['onset_days'] = float(parameter_row['onset_normalized']) * time_scale
            if 'midpoint_normalized' in parameter_row:
                parameter_row['midpoint_days'] = float(parameter_row['midpoint_normalized']) * time_scale
            if 'characteristic_time_normalized' in parameter_row:
                parameter_row['characteristic_time_days'] = float(parameter_row['characteristic_time_normalized']) * time_scale
            if 'latency_normalized' in parameter_row:
                parameter_row['latency_days'] = float(parameter_row['latency_normalized']) * time_scale
            parameter_rows.append(parameter_row)
            for split_name in sorted(set(evaluation_split)):
                mask = evaluation_split == split_name
                if not mask.any():
                    continue
                metric_rows.append({'curve_id': curve_id, 'record_id': group['record_id'].iloc[0], 'source_short': group['source_short'].iloc[0], 'model': model_name, 'split': split_name, 'observations': int(mask.sum()), **metrics(observed[mask], predicted[mask])})
            for row_index, (_, source_row) in enumerate(group.iterrows()):
                prediction_rows.append({'observation_id': source_row['observation_id'], 'curve_id': curve_id, 'record_id': source_row['record_id'], 'source_short': source_row['source_short'], 'model': model_name, 'split': evaluation_split[row_index], 'time_days': source_row['time_days'], 'tau': float(normalized_time[row_index]), 'observed_expansion_percent': observed[row_index], 'predicted_expansion_percent': float(predicted[row_index])})
    predictions = pd.DataFrame(prediction_rows)
    per_curve_metrics = pd.DataFrame(metric_rows)
    parameters = pd.DataFrame(parameter_rows)
    aggregate_rows = []
    for (model_name, split_name), group in predictions.groupby(['model', 'split']):
        aggregate = metrics(group['observed_expansion_percent'].to_numpy(), group['predicted_expansion_percent'].to_numpy())
        curve_group = per_curve_metrics[(per_curve_metrics['model'] == model_name) & (per_curve_metrics['split'] == split_name)]
        aggregate_rows.append({'model': model_name, 'split': split_name, 'observations': len(group), **aggregate, 'mean_curve_mae_percent': float(curve_group['mae_percent'].mean()), 'median_curve_mae_percent': float(curve_group['mae_percent'].median()), 'mean_curve_rmse_percent': float(curve_group['rmse_percent'].mean()), 'median_curve_rmse_percent': float(curve_group['rmse_percent'].median())})
    aggregate_metrics = pd.DataFrame(aggregate_rows).sort_values(['split', 'rmse_percent'])
    predictions.to_csv(out / 'direct_fit_predictions.csv', index=False)
    per_curve_metrics.to_csv(out / 'direct_fit_per_curve_metrics.csv', index=False)
    parameters.to_csv(out / 'direct_fit_parameters.csv', index=False)
    aggregate_metrics.to_csv(out / 'direct_fit_aggregate_metrics.csv', index=False)
    terminal = aggregate_metrics[aggregate_metrics['split'] == 'terminal_test'].sort_values('rmse_percent')
    summary = {'curves': int(data['curve_id'].nunique()), 'time_scaling': args.time_scaling, 'train_observation_fraction': args.train_observation_fraction, 'gate_slope': args.gate_slope, 'activation_transition_width_2_to_98_percent_tau': 2.0 * math.log(49.0) / args.gate_slope, 'response_derived_onset_bound': args.response_derived_onset_bound, 'activation_detection_percent': args.activation_detection_percent, 'activation_margin_tau': args.activation_margin_tau, 'candidate_models': list(MODELS), 'selection_criterion': 'Each fit minimizes training soft-L1 objective. The revised reporting endpoint is seed-averaged mean curve terminal MAE; pooled terminal RMSE is secondary. Reporting endpoint was chosen during exploratory development, not preregistered.', 'best_terminal_model': terminal.iloc[0]['model'], 'terminal_metrics': terminal.to_dict(orient='records')}
    (out / 'direct_fit_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
