from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr, t as student_t
import statsmodels.formula.api as smf
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / '.'
REV = PROJECT / 'results'
OUT = REV / 'c100_revision'
OUT.mkdir(parents=True, exist_ok=True)
OBS_PATH = PROJECT / 'data/modeling' / 'def_kinetics_observations_v2.csv'
CURVE_PATH = PROJECT / 'data/modeling' / 'def_kinetics_curves_v2.csv'
NEURAL_PATH = REV / 'neural_models' / 'primary_selected_20seeds' / 'neural_predictions.csv'
PARAM_PATH = REV / 'neural_models' / 'primary_selected_20seeds' / 'neural_inferred_parameters.csv'
RUN_PATH = REV / 'neural_models' / 'primary_selected_20seeds' / 'neural_model_run_summary.json'
DIRECT_DIR = REV / 'direct_kinetic_fits' / 'c100_primary_with_larive'
BASELINE_PATH = REV / 'compiled' / 'baseline_predictions.csv'
CALIBRATION_PATH = REV / 'neural_models' / 'reviewer_v5_lambda_cv_0p0003_5seeds' / 'neural_predictions.csv'
MODEL_LABELS = {'pinn_shifted_first_order': 'Delayed first-order PINN', 'data_only': 'Data-only NN', 'pinn_avrami': 'Avrami-residual PINN', 'shifted_first_order': 'Direct delayed first order', 'larive': 'Direct Larive', 'first_order': 'Direct first order', 'shifted_avrami': 'Direct Avrami', 'gompertz': 'Direct Gompertz', 'persistence': 'Persistence', 'linear_last5': 'Last-five linear', 'nlme_dfo': 'Partially pooled DFO'}

def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = observed - predicted
    denominator = np.sum((observed - observed.mean()) ** 2)
    return {'mae_percent': float(np.mean(np.abs(residual))), 'rmse_percent': float(np.sqrt(np.mean(residual ** 2))), 'r2': float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 0 else math.nan}

def seed_mean_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ['observation_id', 'curve_id', 'record_id', 'source_short', 'model', 'split', 'time_days']
    out = frame.groupby(keys, as_index=False).agg(observed_expansion_percent=('observed_expansion_percent', 'first'), predicted_expansion_percent=('predicted_expansion_percent', 'mean'), prediction_sd_percent=('predicted_expansion_percent', 'std'))
    return out

def assemble_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(OBS_PATH).sort_values(['curve_id', 'time_days'])
    curves = pd.read_csv(CURVE_PATH)
    neural = seed_mean_predictions(pd.read_csv(NEURAL_PATH))
    direct = pd.read_csv(DIRECT_DIR / 'direct_fit_predictions.csv')
    direct['prediction_sd_percent'] = np.nan
    baseline = pd.read_csv(BASELINE_PATH)
    baseline['prediction_sd_percent'] = np.nan
    common = ['observation_id', 'curve_id', 'record_id', 'source_short', 'model', 'split', 'time_days', 'observed_expansion_percent', 'predicted_expansion_percent', 'prediction_sd_percent']
    predictions = pd.concat([neural[common], direct[common], baseline[common]], ignore_index=True)
    predictions = predictions[predictions['split'].eq('terminal_test')].copy()
    anchors = obs[obs['split'].eq('train')].sort_values(['curve_id', 'time_days']).groupby('curve_id', as_index=False).tail(1)[['curve_id', 'time_days', 'expansion_percent']].rename(columns={'time_days': 'training_horizon_days', 'expansion_percent': 'boundary_expansion_percent'})
    train_scale = obs[obs['split'].eq('train')].groupby('curve_id')['expansion_percent'].apply(lambda s: float(np.mean(np.abs(np.diff(s.to_numpy(float))))) if len(s) > 1 else np.nan).rename('mase_scale_percent').reset_index()
    available = obs.groupby('curve_id').agg(descriptor_reported=('heat_temperature_C', 'size')).reset_index()
    descriptor_cols = ['heat_temperature_C', 'heat_duration_days', 'storage_temperature_C_std', 'water_cement_ratio_std', 'SO3_Al2O3_ratio_std', 'alkali_percent_std', 'scm_percent_std', 'moisture_index', 'restraint_index']
    curve_desc = obs.groupby('curve_id', as_index=False)[descriptor_cols].first()
    curve_desc['descriptor_completeness'] = curve_desc[descriptor_cols].notna().mean(axis=1)
    predictions = predictions.merge(anchors, on='curve_id', how='left').merge(train_scale, on='curve_id', how='left')
    predictions = predictions.merge(curves[['curve_id', 'response_class', 'terminal_expansion_percent']], on='curve_id', how='left')
    predictions = predictions.merge(curve_desc[['curve_id', 'descriptor_completeness']], on='curve_id', how='left')
    predictions['lead_days'] = predictions['time_days'] - predictions['training_horizon_days']
    predictions['lead_tau'] = predictions['lead_days'] / predictions['training_horizon_days']
    predictions['observed_increment_percent'] = predictions['observed_expansion_percent'] - predictions['boundary_expansion_percent']
    predictions['predicted_increment_percent'] = predictions['predicted_expansion_percent'] - predictions['boundary_expansion_percent']
    predictions['absolute_error_percent'] = np.abs(predictions['observed_expansion_percent'] - predictions['predicted_expansion_percent'])
    predictions['increment_absolute_error_percent'] = np.abs(predictions['observed_increment_percent'] - predictions['predicted_increment_percent'])
    predictions['scaled_absolute_error'] = predictions['absolute_error_percent'] / predictions['mase_scale_percent'].clip(lower=1e-06)
    return (obs, curves, predictions)

def fit_partially_pooled_dfo(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    direct_params = pd.read_csv(DIRECT_DIR / 'direct_fit_parameters.csv')
    init = direct_params[direct_params['model'].eq('shifted_first_order')].set_index('curve_id')
    curves = sorted(obs['curve_id'].unique())
    descriptor_cols = ['heat_temperature_C', 'heat_duration_days', 'storage_temperature_C_std', 'water_cement_ratio_std', 'moisture_index', 'restraint_index']
    first = obs.groupby('curve_id', as_index=False).first().set_index('curve_id')
    x0 = first.loc[curves, descriptor_cols].astype(float)
    missing = x0.isna().astype(float)
    x0 = x0.fillna(x0.median())
    x0 = (x0 - x0.mean()) / x0.std().replace(0.0, 1.0)
    X = np.c_[np.ones(len(curves)), x0.to_numpy(), missing.to_numpy()]
    source = first.loc[curves, 'source_short'].astype(str)
    source_dummies = pd.get_dummies(source, drop_first=True, dtype=float).to_numpy()
    X = np.c_[X, source_dummies]
    theta = np.c_[np.log(init.loc[curves, 'amplitude_percent'].clip(1e-05, 5.0)), np.log(init.loc[curves, 'rate_normalized'].clip(1e-05, 100.0)), np.log(init.loc[curves, 'onset_normalized'].clip(0.0001, 3.0))]
    ridge = 0.25
    beta = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ theta)
    random_sd = np.maximum(np.std(theta - X @ beta, axis=0, ddof=1), np.array([0.2, 0.35, 0.35]))
    curve_index = {curve: i for i, curve in enumerate(curves)}
    for _ in range(3):
        updated = []
        for curve in curves:
            i = curve_index[curve]
            g = obs[obs['curve_id'].eq(curve) & obs['split'].eq('train')].sort_values('time_days')
            horizon = max(float(g['time_days'].max()), 1.0)
            tau = g['time_days'].to_numpy(float) / horizon
            y = g['expansion_percent'].to_numpy(float)
            scale = max(float(np.std(y)), 0.015)
            prior = X[i] @ beta

            def residual(v: np.ndarray) -> np.ndarray:
                amp, rate, onset = np.exp(v)
                pred = amp * (1.0 - np.exp(-rate * np.maximum(tau - onset, 0.0)))
                return np.r_[(pred - y) / scale, (v - prior) / random_sd]
            result = least_squares(residual, theta[i], bounds=(np.log([1e-05, 1e-05, 0.0001]), np.log([5.0, 100.0, 3.0])), loss='linear', max_nfev=5000)
            updated.append(result.x)
        theta = np.asarray(updated)
        beta = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ theta)
        random_sd = np.maximum(np.std(theta - X @ beta, axis=0, ddof=1), np.array([0.15, 0.25, 0.25]))
    parameter_rows = []
    prediction_rows = []
    for curve in curves:
        i = curve_index[curve]
        amp, rate, onset = np.exp(theta[i])
        g = obs[obs['curve_id'].eq(curve)].sort_values('time_days')
        horizon = float(g[g['split'].eq('train')]['time_days'].max())
        tau = g['time_days'].to_numpy(float) / max(horizon, 1.0)
        pred = amp * (1.0 - np.exp(-rate * np.maximum(tau - onset, 0.0)))
        parameter_rows.append({'curve_id': curve, 'record_id': g['record_id'].iloc[0], 'source_short': g['source_short'].iloc[0], 'model': 'nlme_dfo', 'amplitude_percent': amp, 'rate_normalized': rate, 'rate_per_day': rate / max(horizon, 1.0), 'onset_normalized': onset, 'onset_days': onset * horizon, 'training_horizon_days': horizon})
        for (_, row), value in zip(g.iterrows(), pred):
            prediction_rows.append({'observation_id': row['observation_id'], 'curve_id': curve, 'record_id': row['record_id'], 'source_short': row['source_short'], 'model': 'nlme_dfo', 'seed': -1, 'split': row['split'], 'time_days': row['time_days'], 'observed_expansion_percent': row['expansion_percent'], 'predicted_expansion_percent': value})
    params = pd.DataFrame(parameter_rows)
    predictions = pd.DataFrame(prediction_rows)
    params.to_csv(OUT / 'nlme_parameters.csv', index=False)
    predictions.to_csv(OUT / 'nlme_predictions.csv', index=False)
    return (params, predictions)

def add_nlme(obs: pd.DataFrame, curves: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    params, nlme = fit_partially_pooled_dfo(obs)
    nlme = nlme[nlme['split'].eq('terminal_test')].copy()
    nlme['prediction_sd_percent'] = np.nan
    anchors = predictions[['curve_id', 'training_horizon_days', 'boundary_expansion_percent', 'mase_scale_percent', 'response_class', 'terminal_expansion_percent', 'descriptor_completeness']].drop_duplicates('curve_id')
    nlme = nlme.merge(anchors, on='curve_id', how='left')
    nlme['lead_days'] = nlme['time_days'] - nlme['training_horizon_days']
    nlme['lead_tau'] = nlme['lead_days'] / nlme['training_horizon_days']
    nlme['observed_increment_percent'] = nlme['observed_expansion_percent'] - nlme['boundary_expansion_percent']
    nlme['predicted_increment_percent'] = nlme['predicted_expansion_percent'] - nlme['boundary_expansion_percent']
    nlme['absolute_error_percent'] = np.abs(nlme['observed_expansion_percent'] - nlme['predicted_expansion_percent'])
    nlme['increment_absolute_error_percent'] = np.abs(nlme['observed_increment_percent'] - nlme['predicted_increment_percent'])
    nlme['scaled_absolute_error'] = nlme['absolute_error_percent'] / nlme['mase_scale_percent'].clip(lower=1e-06)
    return (params, pd.concat([predictions, nlme[predictions.columns]], ignore_index=True))

def aggregate_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    curve_rows = []
    for model, group in predictions.groupby('model'):
        metrics = metric_values(group['observed_expansion_percent'].to_numpy(), group['predicted_expansion_percent'].to_numpy())
        per_curve = []
        for curve, cg in group.groupby('curve_id'):
            m = metric_values(cg['observed_expansion_percent'].to_numpy(), cg['predicted_expansion_percent'].to_numpy())
            persistence = predictions[predictions['model'].eq('persistence') & predictions['curve_id'].eq(curve)]
            pmae = float(persistence['absolute_error_percent'].mean()) if len(persistence) else math.nan
            mae = float(cg['absolute_error_percent'].mean())
            row = {'model': model, 'curve_id': curve, 'source_short': cg['source_short'].iloc[0], 'response_class': cg['response_class'].iloc[0], 'mae_percent': mae, 'rmse_percent': m['rmse_percent'], 'mase': float(cg['scaled_absolute_error'].mean()), 'increment_mae_percent': float(cg['increment_absolute_error_percent'].mean()), 'skill_vs_persistence': 1.0 - mae / pmae if pmae > 1e-10 else math.nan, 'terminal_expansion_percent': float(cg['terminal_expansion_percent'].iloc[0])}
            per_curve.append(row)
            curve_rows.append(row)
        per = pd.DataFrame(per_curve)
        rows.append({'model': model, 'label': MODEL_LABELS.get(model, model), **metrics, 'increment_mae_percent': float(group['increment_absolute_error_percent'].mean()), 'increment_rmse_percent': float(np.sqrt(np.mean((group['observed_increment_percent'] - group['predicted_increment_percent']) ** 2))), 'mean_curve_mae_percent': float(per['mae_percent'].mean()), 'mean_curve_rmse_percent': float(per['rmse_percent'].mean()), 'median_curve_mase': float(per['mase'].median()), 'median_skill_vs_persistence': float(per['skill_vs_persistence'].median()), 'curves_skill_positive': int((per['skill_vs_persistence'] > 0).sum())})
    aggregate = pd.DataFrame(rows).sort_values('mean_curve_mae_percent')
    curve_metrics = pd.DataFrame(curve_rows)
    aggregate.to_csv(OUT / 'central_benchmark_metrics.csv', index=False)
    curve_metrics.to_csv(OUT / 'curve_level_metrics.csv', index=False)
    return (aggregate, curve_metrics)

def stratified_metrics(predictions: pd.DataFrame) -> None:
    rows = []
    for strata in ['source_short', 'response_class']:
        for (level, model), g in predictions.groupby([strata, 'model']):
            values = metric_values(g['observed_expansion_percent'].to_numpy(), g['predicted_expansion_percent'].to_numpy())
            rows.append({'stratum': strata, 'level': level, 'model': model, 'observations': len(g), **values})
    pd.DataFrame(rows).to_csv(OUT / 'metrics_by_source_and_response_class.csv', index=False)

def lead_time_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions[predictions['model'].isin(['pinn_shifted_first_order', 'data_only', 'shifted_first_order', 'larive', 'nlme_dfo', 'persistence'])]
    rows = []
    physical_bins = [-1e-09, 30, 90, 180, 365, 730, np.inf]
    normalized_bins = [-1e-09, 0.05, 0.15, 0.3, 0.6, 1.0, np.inf]
    for name, col, bins in [('physical_days', 'lead_days', physical_bins), ('normalized', 'lead_tau', normalized_bins)]:
        work = selected.copy()
        work['bin'] = pd.cut(work[col], bins=bins, include_lowest=True)
        for (model, interval), g in work.groupby(['model', 'bin'], observed=True):
            rows.append({'scale': name, 'model': model, 'bin': str(interval), 'bin_mid': float(g[col].median()), 'observations': len(g), 'curves': g['curve_id'].nunique(), 'mae_percent': float(g['absolute_error_percent'].mean()), 'rmse_percent': float(np.sqrt(np.mean((g['observed_expansion_percent'] - g['predicted_expansion_percent']) ** 2)))})
    fixed_rows = []
    for lead in [90.0, 180.0, 365.0]:
        for model, mg in selected.groupby('model'):
            errors = []
            for _, cg in mg.groupby('curve_id'):
                cg = cg.sort_values('lead_days')
                if cg['lead_days'].min() <= lead <= cg['lead_days'].max():
                    observed = np.interp(lead, cg['lead_days'], cg['observed_expansion_percent'])
                    predicted = np.interp(lead, cg['lead_days'], cg['predicted_expansion_percent'])
                    errors.append(abs(observed - predicted))
            fixed_rows.append({'lead_days': lead, 'model': model, 'curves': len(errors), 'mae_percent': float(np.mean(errors)) if errors else math.nan})
    lead_frame = pd.DataFrame(rows)
    lead_frame.to_csv(OUT / 'error_by_forecast_lead.csv', index=False)
    pd.DataFrame(fixed_rows).to_csv(OUT / 'error_at_fixed_physical_leads.csv', index=False)
    return lead_frame

def identifiability_analyses(obs: pd.DataFrame, nlme_params: pd.DataFrame) -> dict[str, object]:
    neural = pd.read_csv(PARAM_PATH)
    pin = neural[neural['model'].eq('pinn_shifted_first_order')].copy()
    pin['threshold_fraction'] = 1.0 - np.exp(-pin['rate_normalized'] * pin['onset_normalized'])
    pin_curve = pin.groupby(['curve_id', 'source_short'], as_index=False).median(numeric_only=True)
    direct = pd.read_csv(DIRECT_DIR / 'direct_fit_parameters.csv')
    direct = direct[direct['model'].eq('shifted_first_order')].copy()
    direct['threshold_fraction'] = 1.0 - np.exp(-direct['rate_normalized'] * direct['onset_normalized'])
    output = pd.concat([pin_curve[['curve_id', 'source_short', 'amplitude_percent', 'rate_normalized', 'rate_per_day', 'onset_normalized', 'onset_days', 'training_horizon_days', 'threshold_fraction']].assign(model='pinn_shifted_first_order'), direct[['curve_id', 'source_short', 'amplitude_percent', 'rate_normalized', 'rate_per_day', 'onset_normalized', 'onset_days', 'training_horizon_days', 'threshold_fraction']].assign(model='shifted_first_order'), nlme_params.assign(threshold_fraction=lambda x: 1.0 - np.exp(-x['rate_normalized'] * x['onset_normalized']))[['curve_id', 'source_short', 'amplitude_percent', 'rate_normalized', 'rate_per_day', 'onset_normalized', 'onset_days', 'training_horizon_days', 'threshold_fraction', 'model']]], ignore_index=True)
    output.to_csv(OUT / 'identifiable_parameter_summary_by_curve.csv', index=False)
    by_source = output.groupby(['model', 'source_short'])[['threshold_fraction', 'rate_normalized', 'amplitude_percent']].agg(['median', 'min', 'max'])
    by_source.to_csv(OUT / 'identifiable_parameter_summary_by_source.csv')
    descriptor = obs.groupby('curve_id', as_index=False).first()
    merged = pin_curve.merge(descriptor[['curve_id', 'storage_temperature_C_std', 'SO3_Al2O3_ratio_std', 'alkali_percent_std']], on='curve_id', how='left')
    temp = merged.dropna(subset=['storage_temperature_C_std', 'rate_per_day'])
    sulfate = merged.dropna(subset=['SO3_Al2O3_ratio_std', 'amplitude_percent'])
    alkali = merged.dropna(subset=['alkali_percent_std', 'amplitude_percent'])

    def corr(frame: pd.DataFrame, x: str, y: str) -> dict[str, float]:
        if len(frame) < 3 or frame[x].nunique() < 2:
            return {'n': len(frame), 'rho': math.nan, 'p': math.nan}
        result = spearmanr(frame[x], frame[y])
        return {'n': len(frame), 'rho': float(result.statistic), 'p': float(result.pvalue)}
    return {'target_scale_E_star_percent': json.loads(RUN_PATH.read_text())['target_scale_percent'], 'training_horizon_days': {'min': float(pin_curve['training_horizon_days'].min()), 'median': float(pin_curve['training_horizon_days'].median()), 'max': float(pin_curve['training_horizon_days'].max())}, 'kappa': {'q1': float(pin_curve['rate_normalized'].quantile(0.25)), 'median': float(pin_curve['rate_normalized'].median()), 'q3': float(pin_curve['rate_normalized'].quantile(0.75)), 'below_0p3': int((pin_curve['rate_normalized'] < 0.3).sum()), 'between_0p3_1': int(pin_curve['rate_normalized'].between(0.3, 1, inclusive='left').sum()), 'above_1': int((pin_curve['rate_normalized'] >= 1).sum())}, 'threshold_fraction_pinn': {'q1': float(pin_curve['threshold_fraction'].quantile(0.25)), 'median': float(pin_curve['threshold_fraction'].median()), 'q3': float(pin_curve['threshold_fraction'].quantile(0.75))}, 'storage_temperature_vs_rate': corr(temp, 'storage_temperature_C_std', 'rate_per_day'), 'sulfate_aluminate_vs_capacity': corr(sulfate, 'SO3_Al2O3_ratio_std', 'amplitude_percent'), 'alkali_vs_capacity': corr(alkali, 'alkali_percent_std', 'amplitude_percent')}

def profile_localization_and_offset(obs: pd.DataFrame) -> pd.DataFrame:
    first = obs.groupby('curve_id', as_index=False).first()[['curve_id', 'source_short']]
    terminal = obs.groupby('curve_id', as_index=False)['expansion_percent'].last()
    candidates = first.merge(terminal, on='curve_id').sort_values(['source_short', 'expansion_percent'])
    selected = candidates.groupby('source_short', group_keys=False).apply(lambda g: g.iloc[np.unique(np.linspace(0, len(g) - 1, min(2, len(g))).round().astype(int))], include_groups=False)
    selected_curves = selected['curve_id'].tolist()[:10]
    rows = []
    for curve in selected_curves:
        g = obs[obs['curve_id'].eq(curve) & obs['split'].eq('train')].sort_values('time_days')
        T = max(float(g['time_days'].max()), 1.0)
        t = g['time_days'].to_numpy(float) / T
        y = g['expansion_percent'].to_numpy(float)
        scale = max(float(np.std(y)), 0.015)
        for exponent in np.linspace(0.5, 2.5, 21):

            def residual(v: np.ndarray) -> np.ndarray:
                amp, rate, onset = v
                base = 1.0 - np.exp(-rate * np.maximum(t - onset, 0.0))
                return (amp * np.power(np.maximum(base, 0.0), exponent) - y) / scale
            best = least_squares(residual, [0.8 * max(y.max(), 0.02), 1.0, 0.2], bounds=([0, 0.0001, 0], [5, 100, 3]), loss='soft_l1', max_nfev=10000)
            raw = residual(best.x) * scale
            rss = float(np.sum(raw ** 2))
            bic = len(y) * math.log(max(rss / len(y), 1e-16)) + 3 * math.log(len(y))
            rows.append({'curve_id': curve, 'profile': 'localization_exponent', 'fixed_value': exponent, 'rss': rss, 'bic': bic, 'amplitude': best.x[0], 'rate': best.x[1], 'onset': best.x[2]})
        raw_y = g['expansion_percent_raw'].fillna(g['expansion_percent']).to_numpy(float)
        for offset in np.linspace(-0.01, 0.01, 21):

            def residual_offset(v: np.ndarray) -> np.ndarray:
                amp, rate, onset = v
                pred = offset + amp * (1 - np.exp(-rate * np.maximum(t - onset, 0)))
                return (pred - raw_y) / scale
            best = least_squares(residual_offset, [max(raw_y.max(), 0.02), 1, 0.2], bounds=([0, 0.0001, 0], [5, 100, 3]), loss='soft_l1', max_nfev=10000)
            raw = residual_offset(best.x) * scale
            rss = float(np.sum(raw ** 2))
            bic = len(raw_y) * math.log(max(rss / len(raw_y), 1e-16)) + 3 * math.log(len(raw_y))
            rows.append({'curve_id': curve, 'profile': 'offset_percent', 'fixed_value': offset, 'rss': rss, 'bic': bic, 'amplitude': best.x[0], 'rate': best.x[1], 'onset': best.x[2]})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'profile_likelihood_subset.csv', index=False)
    return frame

def conformal_summary(predictions: pd.DataFrame) -> dict[str, object]:
    calibration = seed_mean_predictions(pd.read_csv(CALIBRATION_PATH))
    calibration = calibration[calibration['split'].eq('internal_validation')]
    residual = np.abs(calibration['observed_expansion_percent'] - calibration['predicted_expansion_percent']).to_numpy()
    test = predictions[predictions['model'].eq('pinn_shifted_first_order')]
    test_residual = test['absolute_error_percent'].to_numpy()
    result = {}
    for nominal in [0.8, 0.9]:
        n = len(residual)
        rank = math.ceil((n + 1) * nominal)
        half = float(np.sort(residual)[rank - 1]) if rank <= n else math.inf
        result[str(nominal)] = {'calibration_n': n, 'half_width_percent': half, 'coverage': float(np.mean(test_residual <= half))}
    return result

def digitization_uncertainty(obs: pd.DataFrame, predictions: pd.DataFrame, curve_metrics: pd.DataFrame) -> dict[str, object]:
    marker_half_points = {'Aubert et al. 2013': 0.78, 'Brunetaud et al. 2007': 1.425, 'Kawabata et al. 2023': 2.145, 'Martin et al. 2023': 1.85, 'Tiecher et al. 2021': 2.0}
    sigmas = {}
    for source, g in obs.groupby('source_short'):
        valid = g.dropna(subset=['source_pdf_top', 'expansion_percent_raw'])
        slope = abs(np.polyfit(valid['source_pdf_top'], valid['expansion_percent_raw'], 1)[0])
        sigmas[source] = slope * marker_half_points[source]
    selected = predictions[predictions['model'].isin(['pinn_shifted_first_order', 'data_only', 'shifted_first_order', 'persistence'])].copy()
    pivot = selected.pivot(index='observation_id', columns='model', values='predicted_expansion_percent')
    truth = selected.drop_duplicates('observation_id').set_index('observation_id')[['observed_expansion_percent', 'curve_id', 'source_short']].loc[pivot.index]
    rng = np.random.default_rng(20260823)
    comparisons = [('data_only', 'pinn_shifted_first_order'), ('shifted_first_order', 'pinn_shifted_first_order'), ('persistence', 'pinn_shifted_first_order')]
    draws = {f'{a}_minus_{b}': [] for a, b in comparisons}
    sigma_vec = truth['source_short'].map(sigmas).to_numpy(float)
    curves = truth['curve_id'].to_numpy()
    for _ in range(1000):
        perturbed = truth['observed_expansion_percent'].to_numpy(float) + rng.normal(0, sigma_vec)
        for a, b in comparisons:
            ea = np.abs(perturbed - pivot[a].to_numpy())
            eb = np.abs(perturbed - pivot[b].to_numpy())
            frame = pd.DataFrame({'curve': curves, 'd': ea - eb})
            draws[f'{a}_minus_{b}'].append(float(frame.groupby('curve')['d'].mean().mean()))
    return {'source_half_marker_uncertainty_percent': sigmas, 'monte_carlo_curve_mean_difference': {key: {'median': float(np.median(v)), 'ci_low': float(np.quantile(v, 0.025)), 'ci_high': float(np.quantile(v, 0.975))} for key, v in draws.items()}}

def inference_summary(curve_metrics: pd.DataFrame) -> pd.DataFrame:
    archived = pd.read_csv(REV / 'compiled' / 'curve_level_seed_averaged_errors.csv').set_index('curve_id')
    pivot = archived[[column for column in archived if column.startswith('mae_percent_')]].copy()
    pivot.columns = [column.removeprefix('mae_percent_') for column in pivot.columns]
    additions = curve_metrics[curve_metrics['model'].isin(['larive', 'nlme_dfo'])].pivot(index='curve_id', columns='model', values='mae_percent')
    pivot = pivot.join(additions, how='left')
    source = archived['source_short']
    rows = []
    for comparator in ['data_only', 'shifted_first_order', 'larive', 'nlme_dfo', 'persistence', 'pinn_avrami', 'linear_last5']:
        if comparator not in pivot:
            continue
        d = (pivot[comparator] - pivot['pinn_shifted_first_order']).dropna()
        src = d.groupby(source.loc[d.index]).mean()
        signs = np.array(np.meshgrid(*[[-1, 1]] * len(src))).T.reshape(-1, len(src))
        exact = float(np.mean(np.abs(signs @ src.to_numpy()) >= abs(src.sum()) - 1e-15))
        jack = np.array([src.drop(index=s).mean() for s in src.index])
        se = math.sqrt((len(jack) - 1) / len(jack) * np.sum((jack - jack.mean()) ** 2))
        tstat = float(src.mean() / se) if se > 0 else math.inf
        pjack = float(2 * student_t.sf(abs(tstat), df=max(len(src) - 1, 1)))
        margin = 0.01
        se_curve = float(d.std(ddof=1) / math.sqrt(len(d)))
        t_low = (d.mean() + margin) / se_curve
        t_high = (d.mean() - margin) / se_curve
        p_tost = float(max(student_t.sf(t_low, len(d) - 1), student_t.cdf(t_high, len(d) - 1)))
        rows.append({'comparator': comparator, 'mean_difference_percent': float(d.mean()), 'median_difference_percent': float(d.median()), 'pinn_wins': int((d > 0).sum()), 'ties': int((d == 0).sum()), 'pinn_losses': int((d < 0).sum()), 'source_sign_test_p': exact, 'source_jackknife_p': pjack, 'tost_margin_percent': margin, 'tost_p': p_tost})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'curve_and_source_level_inference.csv', index=False)
    return frame

def mixed_error_model(predictions: pd.DataFrame) -> dict[str, object]:
    selected = predictions[predictions['model'].isin(['pinn_shifted_first_order', 'data_only', 'shifted_first_order', 'nlme_dfo', 'persistence'])].copy()
    selected['log_absolute_error'] = np.log(selected['absolute_error_percent'].clip(lower=1e-05))
    selected['log_lead_days'] = np.log1p(selected['lead_days'].clip(lower=0))
    selected['log_terminal_expansion'] = np.log(selected['terminal_expansion_percent'].clip(lower=0.0001))
    selected['source_group'] = selected['source_short'].astype('category')
    selected['curve_group'] = selected['curve_id'].astype('category')
    formula = "log_absolute_error ~ C(model, Treatment(reference='pinn_shifted_first_order')) + log_lead_days + log_terminal_expansion + descriptor_completeness"
    fit = smf.mixedlm(formula, selected, groups=selected['source_group'], vc_formula={'curve': '0 + C(curve_group)'}, re_formula='1').fit(reml=True, method='lbfgs', maxiter=500, disp=False)
    coefficients = pd.DataFrame({'term': fit.params.index, 'estimate': fit.params.values, 'standard_error': fit.bse.reindex(fit.params.index).values, 'p_value': fit.pvalues.reindex(fit.params.index).values})
    coefficients.to_csv(OUT / 'mixed_model_log_absolute_error.csv', index=False)
    return {'converged': bool(fit.converged), 'observations': int(len(selected)), 'aic': float(fit.aic) if np.isfinite(fit.aic) else None}

def main() -> None:
    obs, curves, predictions = assemble_predictions()
    nlme_params, predictions = add_nlme(obs, curves, predictions)
    predictions.to_csv(OUT / 'all_terminal_predictions.csv', index=False)
    aggregate, curve_metrics = aggregate_metrics(predictions)
    stratified_metrics(predictions)
    lead_time_metrics(predictions)
    profiles = profile_localization_and_offset(obs)
    ident = identifiability_analyses(obs, nlme_params)
    flat = obs.groupby('curve_id').apply(lambda g: float(g[g['split'].eq('terminal_test')]['expansion_percent'].iloc[-1] - g[g['split'].eq('train')]['expansion_percent'].iloc[-1]), include_groups=False)
    summary = {'flat_holdout_increment_below_0p01_percent': int((np.abs(flat) < 0.01).sum()), 'net_declining_holdouts': int((flat < 0).sum()), 'holdout_curves': int(len(flat)), 'identifiability': ident, 'conformal': conformal_summary(predictions), 'digitization_uncertainty': digitization_uncertainty(obs, predictions, curve_metrics), 'profile_subset': {'curves': int(profiles['curve_id'].nunique()), 'localization_best_m_median': float(profiles[profiles['profile'].eq('localization_exponent')].loc[lambda x: x.groupby('curve_id')['bic'].idxmin(), 'fixed_value'].median()), 'offset_best_abs_median_percent': float(profiles[profiles['profile'].eq('offset_percent')].loc[lambda x: x.groupby('curve_id')['bic'].idxmin(), 'fixed_value'].abs().median())}, 'mixed_error_model': mixed_error_model(predictions)}
    inference_summary(curve_metrics)
    (OUT / 'analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(aggregate.to_string(index=False))
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
