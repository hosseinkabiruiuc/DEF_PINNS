import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / '.'
REV = P / 'results'
spec = importlib.util.spec_from_file_location('direct', P / 'src/fit_def_kinetic_candidates_v3.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
obs = pd.read_csv(m.DATA)
scale = float(obs.loc[obs.split.eq('train'), 'expansion_percent'].max())
model = m.MODELS['shifted_first_order'].copy()
model['lower'] = np.array([0.0, 0.02, 0.0])
model['upper'] = np.array([1.5 * scale, 30.02, 3.0])
m.MODELS = {'shifted_first_order': model}
sys.argv = [str(spec.origin), '--run-name', 'neural_support_matched', '--response-derived-onset-bound']
m.main()
out = REV / 'direct_kinetic_fits/neural_support_matched'
pred = pd.read_csv(out / 'direct_fit_predictions.csv')
pred = pred[pred.split.eq('terminal_test')].copy()
pred['error'] = pred.predicted_expansion_percent - pred.observed_expansion_percent
pred['ae'] = abs(pred.error)
curves = pred.groupby('curve_id').agg(mae=('ae', 'mean'), mse=('error', lambda x: float(np.mean(x * x))))
train = obs[obs.split.eq('train')].sort_values(['curve_id', 'time_days'])
scales = train.groupby('curve_id').expansion_percent.apply(lambda x: float(np.mean(abs(np.diff(x)))))
curves['mase'] = curves.mae / scales.clip(lower=1e-06)
result = {'model': 'support_matched_dfo', 'mae_percent': float(pred.ae.mean()), 'rmse_percent': float(np.sqrt(np.mean(pred.error ** 2))), 'r2': float(1 - np.sum(pred.error ** 2) / np.sum((pred.observed_expansion_percent - pred.observed_expansion_percent.mean()) ** 2)), 'mean_curve_mae_percent': float(curves.mae.mean()), 'mean_curve_rmse_percent': float(np.sqrt(curves.mse).mean()), 'median_curve_mase': float(curves.mase.median()), 'amplitude_upper_percent': 1.5 * scale, 'onset_bound': 'same training crossing and margin as neural model'}
(REV / 'journal_check/support_matched_summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
