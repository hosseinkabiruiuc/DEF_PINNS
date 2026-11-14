import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / '.'
REV = P / 'results'
OUT = REV / 'journal_check'
spec = importlib.util.spec_from_file_location('direct', P / 'src/fit_def_kinetic_candidates_v3.py')
direct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(direct)

def summarize(frame, label, fraction):
    frame = frame.copy()
    frame['absolute_error'] = (frame.predicted_expansion_percent - frame.observed_expansion_percent).abs()
    curves = frame.groupby(['model', 'curve_id'], as_index=False).absolute_error.mean()
    return [{'fraction_elapsed_preterminal_time': fraction, 'model': name, 'estimate': label, 'pooled_mae_percent': float(group.absolute_error.mean()), 'mean_curve_mae_percent': float(curves[curves.model.eq(name)].absolute_error.mean()), 'curves': int(group.curve_id.nunique()), 'observations': int(len(group))} for name, group in frame.groupby('model')]
data = pd.read_csv(P / 'data/modeling/def_kinetics_observations_v2.csv')
rows = []
all_preds = []
for fraction in [0.5, 0.75]:
    pct = int(100 * fraction)
    cached = pd.read_csv(REV / f'direct_kinetic_fits/truncation_{pct}_training_only/direct_fit_predictions.csv')
    cached = cached[cached.split.eq('terminal_test')].copy()
    rows.extend(summarize(cached, 'deterministic', fraction))
    rows_to_add = []
    for curve, group in data.groupby('curve_id'):
        train = group[direct.training_mask(group, fraction)].sort_values('time_days')
        test = group[group.split.eq('terminal_test')]
        scale = float(train.time_days.max())
        t = train.time_days.to_numpy() / scale
        params, *_ = direct.fit_one('larive', t, train.expansion_percent.to_numpy())
        predictions = {'larive': direct.larive(test.time_days.to_numpy() / scale, params), 'persistence': np.full(len(test), float(train.iloc[-1].expansion_percent))}
        tail = train.tail(5)
        if tail.time_days.nunique() > 1:
            slope, intercept = np.polyfit(tail.time_days, tail.expansion_percent, 1)
            predictions['linear_last5'] = np.maximum(0, intercept + slope * test.time_days.to_numpy())
        else:
            predictions['linear_last5'] = predictions['persistence'].copy()
        for model, values in predictions.items():
            for row, value in zip(test.itertuples(), values):
                rows_to_add.append({'model': model, 'curve_id': curve, 'observation_id': row.observation_id, 'source_short': row.source_short, 'time_days': row.time_days, 'observed_expansion_percent': row.expansion_percent, 'predicted_expansion_percent': float(value), 'training_endpoint_days': scale})
    extra = pd.DataFrame(rows_to_add)
    rows.extend(summarize(extra, 'deterministic', fraction))
    all_preds.append(extra.assign(fraction_elapsed_preterminal_time=fraction))
    for arm in ['pinn', 'data']:
        path = REV / f'neural_models/.selected_truncation_{pct}_{arm}/neural_seed_metrics.csv'
        metrics = pd.read_csv(path)
        metrics = metrics[metrics.split.eq('terminal_test')]
        rows.append({'fraction_elapsed_preterminal_time': fraction, 'model': metrics.model.iloc[0], 'estimate': '10-seed mean', 'pooled_mae_percent': float(metrics.mae_percent.mean()), 'mean_curve_mae_percent': float(metrics.mean_curve_mae_percent.mean()), 'curves': 61, 'observations': 366})
    pd.DataFrame(rows).to_csv(OUT / 'review_all_comparator_truncation.csv', index=False)
    pd.concat(all_preds).to_csv(OUT / 'review_truncation_additional_predictions.csv', index=False)
    print('Saved all comparators at time fraction', fraction, flush=True)
new = []
for name in ['review_domain_1p5', 'review_gate_aware_1p5', 'review_curve_scale_dfo', 'review_curve_scale_data']:
    path = REV / 'neural_models' / name
    metrics = pd.read_csv(path / 'neural_seed_metrics.csv')
    metrics = metrics[metrics.split.eq('terminal_test')]
    new.append({'run': name, 'seeds': int(metrics.seed.nunique()), 'pooled_mae_percent': float(metrics.mae_percent.mean()), 'mean_curve_mae_percent': float(metrics.mean_curve_mae_percent.mean()), 'pooled_rmse_percent': float(metrics.rmse_percent.mean())})
pd.DataFrame(new).to_csv(OUT / 'review_new_sensitivity_metrics.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False), flush=True)
